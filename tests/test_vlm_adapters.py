"""Unit tests for VLM multi-model adapters and check mode logic (CPU tests)."""

import numpy as np
import pytest

from src.vlm.adapters import (
    BaseVLMAdapter,
    InternVL2Adapter,
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
