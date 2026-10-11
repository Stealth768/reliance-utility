"""Unit tests for VLM multi-model adapters and check mode logic (CPU tests)."""

import numpy as np
import pytest

from src.vlm.adapters import (
    BaseVLMAdapter,
    InternVL2Adapter,
    LlavaAdapter,
    Qwen2VLAdapter,
    SmolVLMAdapter,
    get_adapter,
)


def test_get_adapter_dispatch():
    """get_adapter instantiates correct adapter class by name or family."""
    # Exact or family matches
    qwen = get_adapter("Qwen/Qwen2-VL-2B-Instruct")
    assert isinstance(qwen, Qwen2VLAdapter)
    assert qwen.model_id == "Qwen/Qwen2-VL-2B-Instruct"

    smol = get_adapter("HuggingFaceTB/SmolVLM-Instruct")
    assert isinstance(smol, SmolVLMAdapter)

    intern = get_adapter("OpenGVLab/InternVL2-2B")
    assert isinstance(intern, InternVL2Adapter)

    # Substring fallbacks
    assert isinstance(get_adapter("qwen2_model"), Qwen2VLAdapter)
    assert isinstance(get_adapter("my-smolvlm"), SmolVLMAdapter)
    assert isinstance(get_adapter("internvl2"), InternVL2Adapter)


def test_get_adapter_unknown():
    """get_adapter raises ValueError for unrecognized models."""
    with pytest.raises(ValueError, match="Unknown model identifier"):
        get_adapter("unsupported-model-v1")


def test_derangement_generator():
    """get_derangement generates permutation with no fixed points."""
    from scripts.run_vlm_pilot import get_derangement

    n = 100
    p = get_derangement(n, seed=42)
    assert len(p) == n
    assert set(p) == set(range(n))
    assert not np.any(p == np.arange(n)), "Found fixed point in derangement!"


def test_mock_adapter_check_logic():
    """Verify check mode logic on a mock adapter."""
    class MockAdapter(BaseVLMAdapter):
        def load(self, device="cuda", dtype="float16"):
            pass

        def letter_probs(self, image_or_None, question, choices):
            # Returns valid probabilities and high letter mass
            probs = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
            mass = 0.96
            return probs, mass

    mock = MockAdapter("test-model")
    probs, mass = mock.letter_probs(None, "Question?", ["A", "B", "C", "D"])
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0)
    assert mass >= 0.90


def test_get_multivariant_choice_token_ids():
    """get_multivariant_choice_token_ids looks up bare and space-prefixed IDs."""
    from src.vlm.adapters import get_multivariant_choice_token_ids

    class MockSmolTokenizer:
        def encode(self, text, add_special_tokens=False):
            mapping = {
                "A": [49], " A": [330],
                "B": [50], " B": [389],
                "C": [51], " C": [340],
                "D": [52], " D": [422],
            }
            return mapping[text]

    tok = MockSmolTokenizer()
    variants = get_multivariant_choice_token_ids(tok)

    assert variants["A"] == (49, 330)
    assert variants["B"] == (50, 389)
    assert variants["C"] == (51, 340)
    assert variants["D"] == (52, 422)


def test_get_multivariant_choice_token_ids_rejects_multitoken():
    """get_multivariant_choice_token_ids raises ValueError if bare or spaced variant is multi-token."""
    from src.vlm.adapters import get_multivariant_choice_token_ids

    class BrokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == " A":
                return [300, 301]
            return [10]

    with pytest.raises(ValueError, match="did not tokenize to a single token"):
        get_multivariant_choice_token_ids(BrokenTokenizer())


# ---------------------------------------------------------------------------
# Tests for SmolVLM assistant_prefix (CPU-only, no model loading)
# ---------------------------------------------------------------------------

def test_smolvlm_prefix_valid_variants():
    """SmolVLMAdapter accepts None and all named prefix variants."""
    adapter_none = SmolVLMAdapter("test-smol", assistant_prefix=None)
    assert adapter_none._prefix_str == ""
    assert adapter_none.assistant_prefix is None

    adapter_ac = SmolVLMAdapter("test-smol", assistant_prefix="answer_colon")
    assert adapter_ac._prefix_str == "Answer:"
    assert adapter_ac.assistant_prefix == "answer_colon"

    adapter_tai = SmolVLMAdapter("test-smol", assistant_prefix="the_answer_is")
    assert adapter_tai._prefix_str == "The answer is"
    assert adapter_tai.assistant_prefix == "the_answer_is"


def test_smolvlm_prefix_invalid_raises():
    """SmolVLMAdapter raises ValueError for unknown prefix names."""
    with pytest.raises(ValueError, match="Unknown assistant_prefix"):
        SmolVLMAdapter("test-smol", assistant_prefix="bad_prefix")


def test_get_adapter_forwards_prefix_to_smolvlm():
    """get_adapter passes assistant_prefix to SmolVLMAdapter and ignores it for other adapters."""
    smol = get_adapter("HuggingFaceTB/SmolVLM-Instruct", assistant_prefix="answer_colon")
    assert isinstance(smol, SmolVLMAdapter)
    assert smol._prefix_str == "Answer:"

    # Other adapters instantiate without error even if prefix supplied (ignored)
    qwen = get_adapter("Qwen/Qwen2-VL-2B-Instruct", assistant_prefix="answer_colon")
    assert isinstance(qwen, Qwen2VLAdapter)

    # None prefix -> empty string
    smol_none = get_adapter("HuggingFaceTB/SmolVLM-Instruct", assistant_prefix=None)
    assert smol_none._prefix_str == ""


class _MockProcessor:
    """Minimal mock processor for prefix test — no model loading needed."""

    class _Tokenizer:
        pass

    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        # Return a deterministic sentinel string
        return "<PROMPT>"

    def __call__(self, text, images=None, return_tensors=None):
        # Not used in prefix test
        return {}


def test_smolvlm_prefix_appended_consistently_with_and_without_image():
    """_build_prompt_text appends the prefix string identically regardless of with_image."""
    adapter = SmolVLMAdapter("test-smol", assistant_prefix="answer_colon")
    adapter.processor = _MockProcessor()

    question = "What color is the sky?"
    choices = ["Red", "Blue", "Green", "Yellow"]

    text_with_image = adapter._build_prompt_text(question, choices, with_image=True)
    text_no_image = adapter._build_prompt_text(question, choices, with_image=False)

    # Both must end with the prefix string
    assert text_with_image.endswith("Answer:"), (
        f"Expected 'Answer:' suffix, got: {text_with_image!r}"
    )
    assert text_no_image.endswith("Answer:"), (
        f"Expected 'Answer:' suffix, got: {text_no_image!r}"
    )

    # With no prefix, no suffix added
    adapter_none = SmolVLMAdapter("test-smol", assistant_prefix=None)
    adapter_none.processor = _MockProcessor()
    text_none_img = adapter_none._build_prompt_text(question, choices, with_image=True)
    text_none_no_img = adapter_none._build_prompt_text(question, choices, with_image=False)
    assert text_none_img == "<PROMPT>"
    assert text_none_no_img == "<PROMPT>"


def test_smolvlm_prefix_the_answer_is_appended():
    """_build_prompt_text appends 'The answer is' for the_answer_is variant."""
    adapter = SmolVLMAdapter("test-smol", assistant_prefix="the_answer_is")
    adapter.processor = _MockProcessor()
    text = adapter._build_prompt_text("Q?", ["A", "B", "C", "D"], with_image=False)
    assert text.endswith("The answer is"), f"Got: {text!r}"


# ---------------------------------------------------------------------------
# Tests for LLaVA-1.5 adapter (CPU-only, no model loading)
# ---------------------------------------------------------------------------

def test_llava_dispatch():
    """get_adapter returns LlavaAdapter for exact HF key and llava substring."""
    adapter = get_adapter("llava-hf/llava-1.5-7b-hf")
    assert isinstance(adapter, LlavaAdapter)
    assert adapter.model_id == "llava-hf/llava-1.5-7b-hf"

    # Substring fallback
    adapter2 = get_adapter("my-llava-model")
    assert isinstance(adapter2, LlavaAdapter)

    # Short alias
    adapter3 = get_adapter("llava-1.5-7b")
    assert isinstance(adapter3, LlavaAdapter)


def test_llava_prefix_valid():
    """LlavaAdapter accepts None, 'answer_colon', and 'the_answer_is'."""
    a_none = LlavaAdapter("test-llava", assistant_prefix=None)
    assert a_none._prefix_str == ""
    assert a_none.assistant_prefix is None

    a_ac = LlavaAdapter("test-llava", assistant_prefix="answer_colon")
    assert a_ac._prefix_str == "Answer:"
    assert a_ac.assistant_prefix == "answer_colon"

    a_tai = LlavaAdapter("test-llava", assistant_prefix="the_answer_is")
    assert a_tai._prefix_str == "The answer is"
    assert a_tai.assistant_prefix == "the_answer_is"


def test_llava_prefix_invalid():
    """LlavaAdapter raises ValueError for unknown prefix names."""
    with pytest.raises(ValueError, match="Unknown assistant_prefix"):
        LlavaAdapter("test-llava", assistant_prefix="bad_prefix_xyz")


def test_llava_prompt_with_image():
    """_build_prompt_text with_image=True contains 'USER: <image>\\n' and 'ASSISTANT:'."""
    adapter = LlavaAdapter("test-llava", assistant_prefix=None)
    prompt = adapter._build_prompt_text("What is shown?", ["Cat", "Dog", "Bird", "Fish"], with_image=True)
    assert "USER: <image>\n" in prompt, f"Expected 'USER: <image>\\n' in prompt, got: {prompt!r}"
    assert "ASSISTANT:" in prompt, f"Expected 'ASSISTANT:' in prompt, got: {prompt!r}"


def test_llava_prompt_text_only():
    """_build_prompt_text with_image=False does NOT contain '<image>' but contains 'USER:' and 'ASSISTANT:'."""
    adapter = LlavaAdapter("test-llava", assistant_prefix=None)
    prompt = adapter._build_prompt_text("What is shown?", ["Cat", "Dog", "Bird", "Fish"], with_image=False)
    assert "<image>" not in prompt, f"Found '<image>' in text-only prompt: {prompt!r}"
    assert "USER:" in prompt, f"Expected 'USER:' in prompt, got: {prompt!r}"
    assert "ASSISTANT:" in prompt, f"Expected 'ASSISTANT:' in prompt, got: {prompt!r}"


def test_llava_prefix_appended():
    """Both with/without image prompts end with 'Answer:' for answer_colon variant."""
    adapter = LlavaAdapter("test-llava", assistant_prefix="answer_colon")
    choices = ["Cat", "Dog", "Bird", "Fish"]

    prompt_with_img = adapter._build_prompt_text("What is shown?", choices, with_image=True)
    assert prompt_with_img.endswith("Answer:"), (
        f"Expected prompt to end with 'Answer:', got: {prompt_with_img!r}"
    )

    prompt_no_img = adapter._build_prompt_text("What is shown?", choices, with_image=False)
    assert prompt_no_img.endswith("Answer:"), (
        f"Expected prompt to end with 'Answer:', got: {prompt_no_img!r}"
    )


def test_get_adapter_forwards_prefix_to_llava():
    """get_adapter passes assistant_prefix to LlavaAdapter."""
    adapter = get_adapter("llava-hf/llava-1.5-7b-hf", assistant_prefix="answer_colon")
    assert isinstance(adapter, LlavaAdapter)
    assert adapter._prefix_str == "Answer:"

    # None prefix -> empty string
    adapter_none = get_adapter("llava-hf/llava-1.5-7b-hf", assistant_prefix=None)
    assert adapter_none._prefix_str == ""
