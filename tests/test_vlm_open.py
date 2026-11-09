"""Unit tests for open-ended benchmark runner and check mode logic (CPU tests)."""

import numpy as np
import pytest

from scripts.run_vlm_open import run_check_mode
from src.vlm.adapters import BaseVLMAdapter


class MockOpenAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, empty_fraction: float = 0.0, identical_r: float = 0.0):
        super().__init__(model_id)
        self.empty_fraction = empty_fraction
        self.identical_r = identical_r
        self.call_count = 0

    def load(self, device="cuda", dtype="float16"):
        pass

    def letter_probs(self, image_or_None, question, choices):
        return np.full(4, 0.25), 1.0

    def generate_open_answer(self, image_or_None, question, prompt_suffix="Answer the question with a short phrase.", max_new_tokens=8):
        self.call_count += 1
        if self.empty_fraction > 0.5:
            return ""
        return "red car"

    def teacher_forced_answer_probs(self, image_or_None, question, answer_text, prompt_suffix="Answer the question with a short phrase."):
        # Returns shape (2, 20)
        p = np.full((2, 20), 1.0 / 20.0)
        if self.identical_r > 0.0:
            # Add difference to second distribution to simulate non-zero R
            p[0, 0] += self.identical_r
            p[0] /= np.sum(p[0])
        return p


def test_open_check_mode_passes_on_valid_answers():
    """run_check_mode passes when answers are non-empty and R sanity is 0."""
    adapter = MockOpenAdapter("test-open-model", empty_fraction=0.0, identical_r=0.0)
    examples = [
        {
            "question": f"Question {i}?",
            "image": None,
            "direct_answers": ["red car", "red car", "car"],
        }
        for i in range(10)
    ]
    # Should complete without raising SystemExit
    run_check_mode(adapter, examples, prompt_suffix="Answer the question with a short phrase.", max_new_tokens=8)


def test_open_check_mode_fails_on_empty_answers():
    """run_check_mode raises SystemExit(1) when fraction of empty answers exceeds 10%."""
    adapter = MockOpenAdapter("test-empty-model", empty_fraction=1.0, identical_r=0.0)
    examples = [
        {
            "question": f"Question {i}?",
            "image": None,
            "direct_answers": ["answer"],
        }
        for i in range(10)
    ]
    with pytest.raises(SystemExit) as exc_info:
        run_check_mode(adapter, examples, prompt_suffix="Answer the question with a short phrase.", max_new_tokens=8)
    assert exc_info.value.code == 1
