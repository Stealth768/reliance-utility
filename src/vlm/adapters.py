"""Model adapters for multiple VLM architectures (Qwen2-VL, InternVL2, SmolVLM, LLaVA-1.5).

Exposes a unified interface:
- `get_adapter(model_id_or_name)` returning a `VLMAdapter` subclass.
- `VLMAdapter.load_model()` loads the model and tokenizer/processor.
- `VLMAdapter.letter_probs(image_or_None, question, choices)` returns (renorm_probs, mass_on_letters).
- Clean CPU-only design with GPU execution protected under `torch` runtime.

Assistant prefix variants (SmolVLM and LLaVA), passed as ``assistant_prefix`` constructor arg:
- ``None`` (default): no prefix appended after the generation prompt.
- ``"answer_colon"``: appends ``"Answer:"`` before reading the next-token distribution.
- ``"the_answer_is"``: appends ``"The answer is"`` before reading the next-token distribution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.vlm.scoring import CHOICE_LETTERS, build_prompt, get_choice_token_ids

# Named prefix strings shared by SmolVLM and LLaVA assistant pre-fill
SMOLVLM_PREFIXES: Dict[str, str] = {
    "answer_colon": "Answer:",
    "the_answer_is": "The answer is",
}
# LLaVA uses the same set
LLAVA_PREFIXES: Dict[str, str] = SMOLVLM_PREFIXES


def get_multivariant_choice_token_ids(tokenizer: Any) -> Dict[str, Tuple[int, int]]:
    """Look up bare and space-prefixed token IDs for each letter in ['A', 'B', 'C', 'D'].

    Asserts that both 'X' and ' X' each tokenize to exactly one token ID.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping letter -> (bare_token_id, space_prefixed_token_id).
    """
    variants: Dict[str, Tuple[int, int]] = {}
    for letter in CHOICE_LETTERS:
        bare_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(bare_ids) != 1:
            raise ValueError(f"Bare letter '{letter}' did not tokenize to a single token (got {bare_ids})")

        spaced_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(spaced_ids) != 1:
            raise ValueError(f"Spaced letter ' {letter}' did not tokenize to a single token (got {spaced_ids})")

        variants[letter] = (bare_ids[0], spaced_ids[0])
    return variants


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
    """Adapter for SmolVLM family (e.g. HuggingFaceTB/SmolVLM-Instruct or 500M/2B).

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    assistant_prefix : str or None, optional
        Named assistant pre-fill variant. One of:
        - None (default): no prefix appended.
        - "answer_colon": appends "Answer:" to the assistant turn.
        - "the_answer_is": appends "The answer is" to the assistant turn.
        The prefix is applied identically for every condition (clean, removed, all corruptions).
    """

    def __init__(self, model_id: str, assistant_prefix: Optional[str] = None):
        super().__init__(model_id)
        if assistant_prefix is not None and assistant_prefix not in SMOLVLM_PREFIXES:
            raise ValueError(
                f"Unknown assistant_prefix '{assistant_prefix}'. "
                f"Supported: {list(SMOLVLM_PREFIXES.keys())}"
            )
        self.assistant_prefix: Optional[str] = assistant_prefix
        # Resolved string (e.g. "Answer:" or "The answer is") or empty string
        self._prefix_str: str = SMOLVLM_PREFIXES[assistant_prefix] if assistant_prefix else ""

    def load(self, device: str = "cuda", dtype: str = "float16"):
        import torch
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        prefix_info = f", prefix='{self._prefix_str}'" if self._prefix_str else ""
        print(f"Loading SmolVLM model: {self.model_id} ({dtype}) on {device}{prefix_info}...")
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            model_cls = AutoModelForVision2Seq

        self.model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

    def _build_prompt_text(self, question: str, choices: List[str], with_image: bool) -> str:
        """Return the full prompt string, with assistant prefix appended if set."""
        messages = build_prompt(question, choices, with_image=with_image)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self._prefix_str:
            prompt_text = prompt_text + self._prefix_str
        return prompt_text

    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        import torch

        with_image = image_or_None is not None
        prompt_text = self._build_prompt_text(question, choices, with_image=with_image)

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

        # Look up both bare ('A') and space-prefixed (' A') token IDs
        variant_ids = get_multivariant_choice_token_ids(self.tokenizer)
        full_probs = torch.softmax(last_logits, dim=-1)

        letter_probs_unnorm = []
        for letter in CHOICE_LETTERS:
            bare_id, space_id = variant_ids[letter]
            prob_sum = float((full_probs[bare_id] + full_probs[space_id]).item())
            letter_probs_unnorm.append(prob_sum)

        total_mass = sum(letter_probs_unnorm)
        if total_mass > 0.0:
            renorm = np.array([p / total_mass for p in letter_probs_unnorm], dtype=np.float64)
        else:
            renorm = np.full(4, 0.25, dtype=np.float64)

        return renorm, float(total_mass)


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
        ).eval().to(device)
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



def get_llava_choice_token_ids(tokenizer: Any) -> Dict[str, List[int]]:
    """Look up token IDs for LLaVA (Llama/SentencePiece) multiple-choice letters.

    Llama/SentencePiece encodes ' A' as [29871, 319] where 29871 is the bare space
    token ' ' and 319 is 'A'.

    For each letter L in ['A', 'B', 'C', 'D'], candidate IDs are collected from:
    1. tokenizer.convert_tokens_to_ids(" " + L)
    2. tokenizer.convert_tokens_to_ids(L)
    3. The LAST ID of tokenizer.encode(" " + L, add_special_tokens=False)
       if decoding that single ID (or stripping whitespace) yields L.

    Deduplicates candidates, drops unknown IDs (None or < 0 or unk_token_id),
    asserts at least one ID per letter, and asserts that all token IDs are
    strictly distinct across letters.

    Returns
    -------
    Dict[str, List[int]]
        Mapping letter -> list of unique token IDs for that letter.
    """
    letter_ids: Dict[str, List[int]] = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for letter in CHOICE_LETTERS:
        candidates = []

        # 1. convert_tokens_to_ids(" " + L)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                t1 = tokenizer.convert_tokens_to_ids(" " + letter)
                if t1 is not None and t1 != unk_id and t1 >= 0:
                    candidates.append(t1)
            except Exception:
                pass

            # 2. convert_tokens_to_ids(L)
            try:
                t2 = tokenizer.convert_tokens_to_ids(letter)
                if t2 is not None and t2 != unk_id and t2 >= 0:
                    candidates.append(t2)
            except Exception:
                pass

        # 3. Last ID of encode(" " + L, add_special_tokens=False) if decoded piece is L
        if hasattr(tokenizer, "encode"):
            try:
                enc = tokenizer.encode(f" {letter}", add_special_tokens=False)
                if enc:
                    last_id = enc[-1]
                    if last_id is not None and last_id != unk_id and last_id >= 0:
                        # Check decoded piece
                        decoded = ""
                        if hasattr(tokenizer, "decode"):
                            decoded = tokenizer.decode([last_id]).strip()
                        if decoded == letter or not decoded:
                            candidates.append(last_id)
            except Exception:
                pass

        # Deduplicate preserving order
        unique_ids = []
        for cid in candidates:
            if cid not in unique_ids:
                unique_ids.append(cid)

        if not unique_ids:
            raise ValueError(f"No valid token IDs found for letter '{letter}' in tokenizer")
        letter_ids[letter] = unique_ids

    # Assert all IDs are distinct across letters
    seen_ids: Dict[int, str] = {}
    for letter, ids in letter_ids.items():
        for cid in ids:
            if cid in seen_ids:
                raise ValueError(
                    f"Token ID {cid} is associated with both '{seen_ids[cid]}' and '{letter}'"
                )
            seen_ids[cid] = letter

    return letter_ids


class LlavaAdapter(BaseVLMAdapter):
    """Adapter for LLaVA-1.5 family (e.g. llava-hf/llava-1.5-7b-hf).

    Loads the model in 4-bit NF4 quantisation via ``BitsAndBytesConfig``
    (compute_dtype float16, device_map={"": 0}) to fit a 7B model on a T4 GPU.

    Prompt format
    -------------
    With image:
        ``USER: <image>\\n{question}\\nChoices:\\nA. ...\\nD. ...\\n\\n
        Answer with the option's letter from the given choices directly. ASSISTANT:``
    Text-only:
        Same without the ``<image>\\n`` token.

    The ``assistant_prefix`` (``"answer_colon"`` or ``"the_answer_is"``) is
    appended *after* ``ASSISTANT:`` before reading the next-token distribution,
    identically for every condition.

    Letter probs
    ------------
    Computed via ``get_llava_choice_token_ids``. Sums full-vocab softmax probabilities
    over each letter's token IDs, with total mass over all IDs reported and renormalized
    over the 4 choices.
    """

    def __init__(self, model_id: str, assistant_prefix: Optional[str] = None):
        super().__init__(model_id)
        if assistant_prefix is not None and assistant_prefix not in LLAVA_PREFIXES:
            raise ValueError(
                f"Unknown assistant_prefix '{assistant_prefix}'. "
                f"Supported: {list(LLAVA_PREFIXES.keys())}"
            )
        self.assistant_prefix: Optional[str] = assistant_prefix
        self._prefix_str: str = LLAVA_PREFIXES[assistant_prefix] if assistant_prefix else ""

    def load(self, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        prefix_info = f", prefix='{self._prefix_str}'" if self._prefix_str else ""
        print(f"Loading LLaVA model: {self.model_id} (4-bit NF4) on {device}{prefix_info}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

    def _build_prompt_text(self, question: str, choices: List[str], with_image: bool) -> str:
        """Build the LLaVA-style prompt string, with optional assistant prefix appended."""
        options_text = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
        )
        body = (
            f"{question}\n\n"
            f"Choices:\n{options_text}\n\n"
            "Answer with the option's letter from the given choices directly."
        )
        if with_image:
            prompt = f"USER: <image>\n{body} ASSISTANT:"
        else:
            prompt = f"USER: {body} ASSISTANT:"
        if self._prefix_str:
            prompt = prompt + self._prefix_str
        return prompt

    def letter_probs(
        self,
        image_or_None: Any,
        question: str,
        choices: List[str],
    ) -> Tuple[np.ndarray, float]:
        import torch

        with_image = image_or_None is not None
        prompt_text = self._build_prompt_text(question, choices, with_image=with_image)

        if with_image:
            inputs = self.processor(
                text=prompt_text,
                images=image_or_None,
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

        # Look up letter token IDs via LLaVA specific lookup
        token_id_map = get_llava_choice_token_ids(self.tokenizer)
        full_probs = torch.softmax(last_logits, dim=-1)

        letter_probs_unnorm = []
        for letter in CHOICE_LETTERS:
            ids = token_id_map[letter]
            prob_sum = float(sum(full_probs[tid].item() for tid in ids))
            letter_probs_unnorm.append(prob_sum)

        total_mass = sum(letter_probs_unnorm)
        if total_mass > 0.0:
            renorm = np.array([p / total_mass for p in letter_probs_unnorm], dtype=np.float64)
        else:
            renorm = np.full(4, 0.25, dtype=np.float64)

        return renorm, float(total_mass)


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
    # LLaVA-1.5
    "llava-hf/llava-1.5-7b-hf": LlavaAdapter,
    "llava-1.5-7b": LlavaAdapter,
}


def get_adapter(model_name_or_id: str, assistant_prefix: Optional[str] = None) -> BaseVLMAdapter:
    """Retrieve the corresponding adapter instance for a model identifier.

    Parameters
    ----------
    model_name_or_id : str
        Model name or HuggingFace repo path.
    assistant_prefix : str or None, optional
        Named assistant pre-fill variant for SmolVLM and LLaVA adapters.
        One of None, "answer_colon", or "the_answer_is". Ignored for Qwen2-VL and InternVL2.

    Returns
    -------
    BaseVLMAdapter
        Instantiated adapter.
    """
    normalized = model_name_or_id.strip()
    lower = normalized.lower()

    def _smol(mid: str) -> SmolVLMAdapter:
        return SmolVLMAdapter(mid, assistant_prefix=assistant_prefix)

    def _llava(mid: str) -> LlavaAdapter:
        return LlavaAdapter(mid, assistant_prefix=assistant_prefix)

    # Check exact match first
    if normalized in MODEL_REGISTRY:
        cls = MODEL_REGISTRY[normalized]
        if cls is SmolVLMAdapter:
            return _smol(normalized)
        if cls is LlavaAdapter:
            return _llava(normalized)
        return cls(normalized)

    # Substring matching fallback
    if "qwen2" in lower or "qwen" in lower:
        return Qwen2VLAdapter(normalized)
    elif "smol" in lower:
        return _smol(normalized)
    elif "internvl" in lower:
        return InternVL2Adapter(normalized)
    elif "llava" in lower:
        return _llava(normalized)

    raise ValueError(
        f"Unknown model identifier '{model_name_or_id}'. Supported families: Qwen2-VL, SmolVLM, InternVL2, LLaVA-1.5."
    )
