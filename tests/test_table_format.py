"""Unit test verifying LaTeX table formatting standards across tables/.

Checks:
1. Every .tex file uses \\fittable wrapping.
2. No table* float environment is used (all use table).
3. No unescaped underscore (_) outside math mode ($...$).
4. No unescaped percent sign (%) outside math mode ($...$) unless it is a line comment.
"""

import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "tables"


def get_table_files():
    """Retrieve all .tex files in tables/ directory."""
    files = sorted(TABLES_DIR.rglob("*.tex"))
    assert len(files) > 0, "No .tex files found in tables directory"
    return files


@pytest.mark.parametrize("tex_path", get_table_files(), ids=lambda p: str(p.relative_to(TABLES_DIR)))
def test_table_structure_and_fittable(tex_path: Path):
    """Verify that every table file uses \\fittable and no table*."""
    content = tex_path.read_text(encoding="utf-8")
    assert r"\fittable" in content, f"{tex_path.name} is missing \\fittable wrapping"
    assert r"\begin{table*}" not in content, f"{tex_path.name} uses two-column table*"
    assert r"\end{table*}" not in content, f"{tex_path.name} uses two-column table*"


@pytest.mark.parametrize("tex_path", get_table_files(), ids=lambda p: str(p.relative_to(TABLES_DIR)))
def test_no_unescaped_underscore_outside_math(tex_path: Path):
    """Verify no unescaped underscore appears outside math mode."""
    content = tex_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for line_idx, line in enumerate(lines, 1):
        # Strip math mode blocks $...$
        stripped = re.sub(r"\$.*?\$", "", line)
        # Strip \label{...} content since labels may contain hyphens/colons
        stripped = re.sub(r"\\label\{.*?\}", "", stripped)

        # Look for underscore not preceded by backslash
        match = re.search(r"(?<!\\)_", stripped)
        assert match is None, (
            f"Unescaped underscore on line {line_idx} in {tex_path.name}: {line.strip()}"
        )


@pytest.mark.parametrize("tex_path", get_table_files(), ids=lambda p: str(p.relative_to(TABLES_DIR)))
def test_no_unescaped_percent_outside_math(tex_path: Path):
    """Verify no unescaped percent sign appears as text outside math mode."""
    content = tex_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for line_idx, line in enumerate(lines, 1):
        # Strip math mode blocks $...$
        stripped = re.sub(r"\$.*?\$", "", line)

        # In LaTeX, an unescaped % at end of line or starting after { is a comment/continuation
        # We want to catch unescaped % that appears in text (like "42%") instead of "42\%"
        # If line ends with {% or %, that's a comment/continuation
        if stripped.rstrip().endswith("{%") or stripped.rstrip().endswith("%"):
            # remove trailing comment %
            stripped = re.sub(r"\{?%$", "", stripped.rstrip())

        # Check for any remaining unescaped %
        match = re.search(r"(?<!\\)%", stripped)
        assert match is None, (
            f"Unescaped percent sign on line {line_idx} in {tex_path.name}: {line.strip()}"
        )
