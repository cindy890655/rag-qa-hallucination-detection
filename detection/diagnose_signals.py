"""
Compare candidate detection signals before committing to one.

This is the script that decided the project's central design choice. An earlier
baseline classified answers as faithful whenever entailment was the top class
and performed at chance. Printing the per-class distributions showed why:
entailment barely separated the two groups, while contradiction separated them
cleanly. That is what moved the detector to a contradiction threshold, and the
"1 - entailment" row here is what later suggested combining the two signals.

Reads cached_scores.json, so no model is loaded. Run from the project root:
    python3 detection/diagnose_signals.py
"""
import json
import statistics as st
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cached_scores.json"

if not CACHE.exists():
    raise SystemExit(f"Missing {CACHE}. Run detection/rescore.py first.")

cache = json.load(CACHE.open(encoding="utf-8"))
all_scores = cache["all_scores"]
y_true = cache["y_true"]


def signal_vals(fn):
    """
    Split one signal's values by ground truth.

    Returns (faithful_values, hallucinated_values). A usable signal is low on
    the first group and high on the second; the gap between them is what a
    threshold has to sit in.
    """
    faithful = [fn(s) for s, y in zip(all_scores, y_true) if y == 0]
    halluc = [fn(s) for s, y in zip(all_scores, y_true) if y == 1]
    return faithful, halluc


# Three candidates. "contra - entail" is the natural combination of the other
# two, kept here to show that combining them as a single score is weaker than
# using them as two independent conditions, which is what the fusion rule does.
signals = {
    "contradiction":   lambda s: s.get("contradiction", 0.0),
    "1 - entailment":  lambda s: 1.0 - s.get("entailment", 0.0),
    "contra - entail": lambda s: (s.get("contradiction", 0.0)
                                  - s.get("entailment", 0.0)),
}

print(f"samples: {len(y_true)} "
      f"({sum(y_true)} hallucinated, {len(y_true) - sum(y_true)} faithful)\n")

for name, fn in signals.items():
    faithful, halluc = signal_vals(fn)
    print(f"=== signal: {name} ===")
    print(f"  FAITHFUL (should be LOW) : "
          f"mean={st.mean(faithful):+.3f}  median={st.median(faithful):+.3f}")
    print(f"  HALLUCIN (should be HIGH): "
          f"mean={st.mean(halluc):+.3f}  median={st.median(halluc):+.3f}")
    print(f"  separation (halluc - faithful mean) = "
          f"{st.mean(halluc) - st.mean(faithful):+.3f}")
    print()

print(">>> A robust threshold sits near the MIDPOINT between the two groups'")
print("    medians, not at the extreme value that squeezes out maximum F1 on")
print("    this subset. See tune_threshold.py for the sweep, and the report for")
print("    what happened when this threshold was carried over to RAG output.")