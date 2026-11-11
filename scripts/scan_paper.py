"""Quick scan of paper_extended/main.tex for overclaims and honest statements."""
from pathlib import Path

tex = (Path(__file__).parent.parent / "paper_extended" / "main.tex").read_text(encoding="utf-8")

print("=== OVERCLAIMS (should be 0) ===")
bad = [
    "invariant property",
    "same fundamental reliance-utility misalignment observed",
    "fundamental limitation",
    "dominated by incongruent",
    "no multiple-choice priors",
    "ComSIA 2026",
    "anonfalse",
]
for p in bad:
    n = tex.count(p)
    print(f"  {'OK' if n==0 else 'BAD'} ({n}) : {p!r}")

print()
print("=== HONEST STATEMENTS (should be >=1) ===")
good = [
    ("pre-declared success criterion required", "Synthetic MI criterion fail"),
    ("post hoc", "post-hoc bootstrap label"),
    ("at most 0.064", "donor pairing range claim"),
    ("utility inversion does not replicate", "AI2D H5 fail statement"),
    ("Gaussian noise, Qwen2-VL-2B", "Gaussian utility note in open-ended"),
    ("Multiple Comparisons", "limitations: multiple comparisons"),
    ("0.008", "H7 Holm note"),
    ("not directly comparable", "R_hat_open threshold caveat"),
    ("anontrue", "anonymous mode"),
]
for phrase, label in good:
    n = tex.count(phrase)
    print(f"  {'OK' if n>0 else 'MISSING'} ({n}) : {phrase!r} ({label})")

print(f"\nTotal lines: {len(tex.splitlines())}")
