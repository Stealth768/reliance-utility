"""Unit tests for VLM prompt building and choice token utilities (CPU tests)."""

import pytest

from src.vlm.scoring import CHOICE_LETTERS, build_prompt, get_choice_token_ids


def test_build_prompt_with_image():
    """build_prompt creates user chat message with image and exact instruction."""
    question = "What color is the hydrant?"
    choices = ["red", "yellow", "green", "blue"]

    messages = build_prompt(question, choices, with_image=True)

    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    content = msg["content"]
    assert len(content) == 2
    assert content[0] == {"type": "image"}
    assert content[1]["type"] == "text"

    text = content[1]["text"]
    assert "What color is the hydrant?" in text
    assert "A. red" in text
    assert "B. yellow" in text
    assert "C. green" in text
    assert "D. blue" in text
    assert text.endswith("Answer with the option's letter from the given choices directly.")


def test_build_prompt_text_only():
    """build_prompt with with_image=False omits the image entry."""
    question = "What is the capital of France?"
    choices = ["Berlin", "Madrid", "Paris", "Rome"]

    messages = build_prompt(question, choices, with_image=False)
    content = messages[0]["content"]

    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "image" not in [item.get("type") for item in content]
    assert content[0]["text"].endswith("Answer with the option's letter from the given choices directly.")


def test_build_prompt_validates_4_choices():
    """build_prompt raises ValueError if choices length is not 4."""
    with pytest.raises(ValueError, match="Expected exactly 4 choices"):
        build_prompt("Question?", ["A", "B", "C"])

    with pytest.raises(ValueError, match="Expected exactly 4 choices"):
        build_prompt("Question?", ["A", "B", "C", "D", "E"])


def test_get_choice_token_ids_mock_tokenizer():
    """get_choice_token_ids extracts single-token IDs from tokenizer."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            mapping = {"A": [101], "B": [102], "C": [103], "D": [104]}
            return mapping.get(text, [999])

    tok = MockTokenizer()
    ids = get_choice_token_ids(tok)
    assert ids == [101, 102, 103, 104]


def test_get_choice_token_ids_multi_token_error():
    """get_choice_token_ids raises ValueError if a letter is multi-token."""
    class BrokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "A" in text:
                return [101, 102]
            return [100]

    with pytest.raises(ValueError, match="does not tokenize to a single token"):
        get_choice_token_ids(BrokenTokenizer())

