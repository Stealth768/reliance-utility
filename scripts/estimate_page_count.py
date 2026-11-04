from pathlib import Path

sections = [
    ("Title & Preamble", "paper/main.tex", 0.3),
    ("Abstract & Keywords", "paper/sections/abstract.tex", 0.4),
    ("1. Introduction", "paper/sections/intro.tex", 1.0),
    ("2. Related Work", "paper/sections/related.tex", 1.0),
    ("3. Methodology", "paper/sections/method.tex", 1.5),
    ("4. Synthetic Validation", "paper/sections/synthetic.tex", 1.2), # Includes Fig 1 (0.35p) + Tab 1 (0.25p)
    ("5. VLM Evaluation", "paper/sections/results.tex", 2.2), # Includes Fig 2 (0.35p) + Tab 2 (0.35p) + Tab 3 (0.2p) + Tab 4 (0.45p)
    ("6. Case Study: Dropout", "paper/sections/dropout.tex", 0.9), # Includes Tab 5 (0.25p)
    ("7. Discussion", "paper/sections/discussion.tex", 0.7),
    ("8. Limitations", "paper/sections/limitations.tex", 0.5),
    ("9. Conclusion", "paper/sections/conclusion.tex", 0.3),
    ("References", "paper/refs.bib", 0.8), # 11 entries
    ("Appendix (Tables 6, 7)", "paper/sections/appendix.tex", 1.0), # Criterion (0.35p) + Baseline (0.35p)
]

print(f"{'Section / Element':<28} | {'Words':<6} | {'Estimated Page Budget':<20}")
print("-" * 60)

total_est = 0.0
for name, rel_path, est_budget in sections:
    p = Path(rel_path)
    words = 0
    if p.exists():
        words = len(p.read_text(encoding="utf-8").split())
    total_est += est_budget
    print(f"{name:<28} | {words:<6} | ~{est_budget:.2f} pages")

print("-" * 60)
print(f"Total Estimated Document Length: ~{total_est:.1f} pages (Target: 12 pages total including references and appendix).")
