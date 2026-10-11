"""Model adapters for multiple VLM architectures (Qwen2-VL, InternVL2, SmolVLM).

Exposes a unified interface:
- `get_adapter(model_id_or_name)` returning a `VLMAdapter` subclass.
- `VLMAdapter.load_model()` loads the model and tokenizer/processor.
- `VLMAdapter.letter_probs(image_or_None, question, choices)` returns (renorm_probs, mass_on_letters).
- Clean CPU-only design with GPU execution protected under `torch` runtime.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.vlm.scoring import CHOICE_LETTERS, build_prompt, get_choice_token_ids


class BaseVLMAdapter(ABC):
    """Abstract base adapter for vision-language models."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None

    @abstractmethod
    def load(self, device: str = "cuda", dtype: str = "float16"):
        """Load model weights and processor/tokenizer onto target device."""
        pass

    @abstractmethod
    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        """Perform forward pass and compute letter probabilities over ['A', 'B', 'C', 'D'].

        Returns
        -------
        Tuple[np.ndarray, float]
            Length-4 float64 numpy array of renormalized probabilities,
            and total vocabulary probability mass on ['A', 'B', 'C', 'D'].
        """
        pass


class Qwen2VLAdapter(BaseVLMAdapter):
    """Adapter for Qwen2-VL family (e.g. Qwen/Qwen2-VL-2B-Instruct)."""

    def load(self, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        print(f"Loading Qwen2-VL model: {self.model_id} ({dtype}) on {device}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

        # Constrain max pixels to prevent T4 OOM
        min_pixels = 256 * 28 * 28
        max_pixels = 1024 * 28 * 28
        size_dict = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, size=size_dict)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
            )
        self.tokenizer = self.processor.tokenizer

    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        from src.vlm.scoring import letter_probs
        return letter_probs(self.model, self.processor, image_or_None, question, choices)


class SmolVLMAdapter(BaseVLMAdapter):
    """Adapter for SmolVLM family (e.g. HuggingFaceTB/SmolVLM-Instruct or 500M/2B)."""

    def load(self, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        print(f"Loading SmolVLM model: {self.model_id} ({dtype}) on {device}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        import torch

        with_image = image_or_None is not None
        messages = build_prompt(question, choices, with_image=with_image)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if with_image:
            inputs = self.processor(
                text=prompt_text,
                images=[image_or_None],
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompt_text,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[0, -1, :]

        token_ids = get_choice_token_ids(self.tokenizer)
        full_probs = torch.softmax(last_logits, dim=-1)
        choice_logits = last_logits[token_ids]
        renorm = torch.softmax(choice_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        mass = float(full_probs[token_ids].sum().item())
        return renorm, mass


class InternVL2Adapter(BaseVLMAdapter):
    """Adapter for InternVL2 family (e.g. OpenGVLab/InternVL2-2B)."""

    def load(self, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        print(f"Loading InternVL2 model: {self.model_id} ({dtype}) on {device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        import torch

        options_text = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
        )
        prompt_text = (
            f"<image>\n" if image_or_None is not None else ""
        ) + (
            f"{question}\n\n"
            f"Choices:\n{options_text}\n\n"
            "Answer with the option's letter from the given choices directly."
        )

        # InternVL custom conversation / preprocessing
        if image_or_None is not None:
            # Handle image pixel transform
            from torchvision import transforms  # type: ignore
            from torchvision.transforms.functional import InterpolationMode  # type: ignore

            transform = transforms.Compose([
                transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            pixel_values = transform(image_or_None.convert("RGB")).unsqueeze(0)
            pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)
        else:
            pixel_values = None

        # Build prompt tokens
        template = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.model.device)

        with torch.no_grad():
            if pixel_values is not None:
                outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
            else:
                outputs = self.model(input_ids=input_ids)
            last_logits = outputs.logits[0, -1, :]

        token_ids = get_choice_token_ids(self.tokenizer)
        full_probs = torch.softmax(last_logits, dim=-1)
        choice_logits = last_logits[token_ids]
        renorm = torch.softmax(choice_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        mass = float(full_probs[token_ids].sum().item())
        return renorm, mass


# Registry of supported models
MODEL_REGISTRY = {
    # Qwen2-VL
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VLAdapter,
    "qwen2-vl-2b": Qwen2VLAdapter,
    # SmolVLM
    "HuggingFaceTB/SmolVLM-Instruct": SmolVLMAdapter,
    "HuggingFaceTB/SmolVLM-500M-Instruct": SmolVLMAdapter,
    "smolvlm": SmolVLMAdapter,
    # InternVL2
    "OpenGVLab/InternVL2-2B": InternVL2Adapter,
    "internvl2-2b": InternVL2Adapter,
}


def get_adapter(model_name_or_id: str) -> BaseVLMAdapter:
    """Retrieve the corresponding adapter instance for a model identifier.

    Parameters
    ----------
    model_name_or_id : str
        Model name or HuggingFace repo path.

    Returns
    -------
    BaseVLMAdapter
        Instantiated adapter.
    """
    normalized = model_name_or_id.strip()
    # Check exact match
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized](normalized)

    # Substring matching fallback
    lower = normalized.lower()
    if "qwen2" in lower or "qwen" in lower:
        return Qwen2VLAdapter(normalized)
    elif "smol" in lower:
        return SmolVLMAdapter(normalized)
    elif "internvl" in lower:
        return InternVL2Adapter(normalized)

    raise ValueError(
        f"Unknown model identifier '{model_name_or_id}'. Supported families: Qwen2-VL, SmolVLM, InternVL2."
    )
