"""Final evaluation of the detector at the locked-in setting:
   signal = contradiction, threshold = 0.5.
   Reads cached_scores.json (DeBERTa-FEVER scores). No model run needed.
"""
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

THRESHOLD = 0.5

with open("cached_scores.json") as f:
    cache = json.load(f)
all_scores = cache["all_scores"]
y_true = cache["y_true"]

# locked signal: contradiction; predict hallucinated (1) if contradiction >= threshold
y_pred = [1 if s.get("contradiction", 0.0) >= THRESHOLD else 0 for s in all_scores]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=== FINAL detector on HaluEval (N=400) ===")
print(f"model     : MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
print(f"signal    : contradiction  |  threshold = {THRESHOLD}")
print()
print(f"accuracy  : {acc:.4f}")
print(f"precision : {prec:.4f}")
print(f"recall    : {rec:.4f}")
print(f"f1        : {f1:.4f}")
print()
print(f"confusion : TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"            (correctly caught {tp} hallucinations, missed {fn};")
print(f"             correctly passed {tn} faithful answers, wrongly flagged {fp})")