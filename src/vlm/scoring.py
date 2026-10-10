"""Scoring and prompt construction for Qwen2-VL multiple choice evaluation.

CPU-safe / pure-NumPy prompt logic and GPU forward-pass scoring:
- build_prompt: Constructs standard chat message representation for 4-option MC question.
- letter_probs: Evaluates model forward pass, extracts next-token distribution over ['A', 'B', 'C', 'D'],
  normalizes to sum to 1, and reports full-vocab probability mass on those four tokens.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


CHOICE_LETTERS = ["A", "B", "C", "D"]


def build_prompt(
    question: str,
    choices: List[str],
    with_image: bool = True,
) -> List[Dict[str, Any]]:
    """Build conversation message prompt for 4-option multiple-choice VQA.

    The message includes an image entry only if with_image is True.
    The text explicitly enumerates options A-D and ends with the exact instruction:
    "Answer with the option's letter from the given choices directly."

    Parameters
    ----------
    question : str
        Question text.
    choices : List[str]
        List of 4 candidate options.
    with_image : bool, default=True
        Whether to include image entry in the user content block.

    Returns
    -------
    List[Dict[str, Any]]
        Chat conversation messages conforming to Qwen2-VL processor format.
    """
    if len(choices) != 4:
        raise ValueError(f"Expected exactly 4 choices, got {len(choices)}")

    # Format options A, B, C, D
    options_text = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
    )

    prompt_text = (
        f"{question}\n\n"
        f"Choices:\n{options_text}\n\n"
        "Answer with the option's letter from the given choices directly."
    )

    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


def get_choice_token_ids(tokenizer: Any) -> List[int]:
    """Retrieve token IDs for 'A', 'B', 'C', 'D' and assert each is a single token.

    Parameters
    ----------
    tokenizer : Any
        HuggingFace / Qwen2-VL tokenizer.

    Returns
    -------
    List[int]
        Length-4 list of token IDs for ['A', 'B', 'C', 'D'].
    """
    token_ids = []
    for letter in CHOICE_LETTERS:
        # Check tokenization
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            # Try leading space if standalone letter is multi-token (standard in byte-level BPE)
            alt_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if len(alt_ids) == 1:
                ids = alt_ids
            else:
                raise ValueError(
                    f"Letter '{letter}' does not tokenize to a single token (got {ids} and {alt_ids})"
                )
        token_ids.append(ids[0])
    return token_ids


def letter_probs(
    model: Any,
    processor: Any,
    image_or_None: Any,
    question: str,
    choices: List[str],
) -> Tuple[np.ndarray, float]:
    """Perform forward pass and extract letter probabilities over ['A', 'B', 'C', 'D'].

    Parameters
    ----------
    model : Any
        Qwen2VLForConditionalGeneration on cuda device.
    processor : Any
        Qwen2VLProcessor.
    image_or_None : Optional[PIL.Image]
        PIL image, or None for text-only (null 'removed') condition.
    question : str
        Question text.
    choices : List[str]
        List of 4 candidate choices.

    Returns
    -------
    Tuple[np.ndarray, float]
        - Length-4 float64 numpy vector of renormalized probabilities over ['A', 'B', 'C', 'D'].
        - Float value representing the full-vocabulary probability mass assigned to the 4 letter tokens.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required on GPU to run letter_probs scoring") from exc

    with_image = image_or_None is not None
    messages = build_prompt(question, choices, with_image=with_image)

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare multimodal inputs
    if with_image:
        inputs = processor(
            text=[text_prompt],
            images=[image_or_None],
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        )

    # Move tensors to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # Logits at the very last token position: shape (batch_size, vocab_size)
        last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    # Extract target token IDs for A, B, C, D
    tokenizer = processor.tokenizer
    token_ids = get_choice_token_ids(tokenizer)

    # Compute full-vocab softmax probabilities
    full_probs = torch.softmax(last_logits, dim=-1)

    # Gather probabilities and logits for the four choice letters
    choice_logits = last_logits[token_ids]
    renorm_probs = torch.softmax(choice_logits, dim=-1).detach().cpu().numpy().astype(np.float64)

    # Compute total probability mass on choice letters in the full vocabulary
    mass_on_letters = float(full_probs[token_ids].sum().item())

    return renorm_probs, mass_on_letters
