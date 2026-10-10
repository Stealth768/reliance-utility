"""Unit tests for I/O utilities and immutability guarantees."""

import json
from pathlib import Path
import pytest

from src.io_utils import write_raw


def test_write_raw_creates_file_atomically(tmp_path: Path):
    """write_raw writes JSON/dict data correctly."""
    target_file = tmp_path / "raw_experiment.json"
    data = {"metric": 0.85, "model": "LLaVA-1.5-7B"}

    out_path = write_raw(target_file, data)
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_raw_raises_file_exists_error_on_second_write(tmp_path: Path):
    """write_raw raises FileExistsError on the second write to the same path."""
    target_file = tmp_path / "immutable_run.json"
    data1 = {"run_id": 1}
    data2 = {"run_id": 2}

    # First write succeeds
    write_raw(target_file, data1)
    assert target_file.exists()

    # Second write MUST raise FileExistsError
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing file"):
        write_raw(target_file, data2)


def test_write_raw_supports_text_and_bytes(tmp_path: Path):
    """write_raw handles str and bytes payloads properly."""
    txt_file = tmp_path / "test.txt"
    bin_file = tmp_path / "test.bin"

    write_raw(txt_file, "hello world")
    assert txt_file.read_text(encoding="utf-8") == "hello world"

    write_raw(bin_file, b"\x00\x01\x02")
    assert bin_file.read_bytes() == b"\x00\x01\x02"
