"""Unit test verifying LaTeX table formatting standards across tables/.

Checks (v0.26 Springer SVMult format):
1. No \\fittable wrapping (removed for svmult single-column).
2. No \\resizebox wrapping.
3. \\footnotesize is present for compact tables.
4. No table* float environment is used (all use table).
5. No unescaped underscore (_) outside math mode ($...$).
6. No unescaped percent sign (%) outside math mode ($...$) unless it is a line comment.
"""

import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "tables"

# Tables that live in the appendix and are allowed to omit \footnotesize
# (e.g., very short tables where it is not needed)
FOOTNOTESIZE_EXEMPT = {
    "baseline_metrics.tex",
    "criterion_status_compact.tex",
    "criterion_status_by_null_and_sample.tex",
    "fusion_summary_full.tex",
}


def get_table_files():
    """Retrieve all .tex files in tables/ directory."""
    files = sorted(TABLES_DIR.rglob("*.tex"))
    assert len(files) > 0, "No .tex files found in tables directory"
    return files


@pytest.mark.parametrize("tex_path", get_table_files(), ids=lambda p: str(p.relative_to(TABLES_DIR)))
def test_table_structure_no_fittable(tex_path: Path):
    """Verify that no table file uses \\fittable or \\resizebox, and no table*."""
    content = tex_path.read_text(encoding="utf-8")
    assert r"\fittable" not in content, (
        f"{tex_path.name} still uses \\fittable -- remove it for svmult single-column"
    )
    assert r"\resizebox" not in content, (
        f"{tex_path.name} still uses \\resizebox -- remove it for svmult single-column"
    )
    assert r"\begin{table*}" not in content, f"{tex_path.name} uses two-column table*"
    assert r"\end{table*}" not in content, f"{tex_path.name} uses two-column table*"


@pytest.mark.parametrize("tex_path", get_table_files(), ids=lambda p: str(p.relative_to(TABLES_DIR)))
def test_table_has_footnotesize(tex_path: Path):
    """Verify that each table file uses \\footnotesize for compact rendering."""
    if tex_path.name in FOOTNOTESIZE_EXEMPT:
        pytest.skip(f"{tex_path.name} is exempt from \\footnotesize requirement")
    content = tex_path.read_text(encoding="utf-8")
    assert r"\footnotesize" in content, (
        f"{tex_path.name} is missing \\footnotesize -- add it for svmult column width"
    )


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
