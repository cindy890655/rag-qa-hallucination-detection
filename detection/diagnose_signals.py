import json
import statistics as st

with open("cached_scores.json") as f:
    cache = json.load(f)
all_scores = cache["all_scores"]
y_true = cache["y_true"]

def signal_vals(fn):
    faithful = [fn(s) for s, y in zip(all_scores, y_true) if y == 0]
    halluc   = [fn(s) for s, y in zip(all_scores, y_true) if y == 1]
    return faithful, halluc

signals = {
    "contradiction":       lambda s: s.get("contradiction", 0.0),
    "1 - entailment":      lambda s: 1.0 - s.get("entailment", 0.0),
    "contra - entail":     lambda s: s.get("contradiction",0.0) - s.get("entailment",0.0),
}

for name, fn in signals.items():
    f, h = signal_vals(fn)
    print(f"=== signal: {name} ===")
    print(f"  FAITHFUL (should be LOW) : mean={st.mean(f):+.3f}  median={st.median(f):+.3f}")
    print(f"  HALLUCIN (should be HIGH): mean={st.mean(h):+.3f}  median={st.median(h):+.3f}")
    print(f"  separation (halluc-faithful mean) = {st.mean(h)-st.mean(f):+.3f}")
    print()

print(">>> A robust threshold is usually near the MIDPOINT between the two groups' medians,")
print("    not the extreme value that squeezes out max F1 on the training subset.")