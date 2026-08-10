"""
4b(1): Compare the NLI fusion detector against human faithfulness labels made
on the SAME phi3-generated RAG answers (fully paired). Two-layer labels:
  Layer 1 (binary): faithful(0)/hallucinated(1) -> detector agreement
  Layer 2 (type): fine-grained categories -> analysis
"""
import json
import sys
from collections import Counter
sys.path.insert(0, "detection")
from detector import HallucinationDetector

RAG_RESULTS = "data/rag_eval/rag_results_k3_phi3.jsonl"
LABELS = "data/rag_eval/labels_detailed.json"

# load RAG outputs
rag = []
with open(RAG_RESULTS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rag.append(json.loads(line))

labels = json.load(open(LABELS))["labels"]

print("Loading detector...")
detector = HallucinationDetector()

matched = total = 0
rows = []
for r in rag:
    q = r["question"]
    ans = r["answer"]
    knowledge = " ".join(d["content"] for d in r["retrieved_documents"])
    label, scores = detector.detect_fusion(knowledge, q, ans)
    det = int(label)                          # 1=halluc, 0=faithful
    gold = labels[q]["faithful"]              # human binary label
    gtype = labels[q]["type"]
    agree = (det == gold)
    total += 1
    if agree:
        matched += 1
    rows.append((q, det, gold, gtype, agree))

# Layer 1: detector vs human binary
print("\n" + "=" * 95)
print(f"{'Question':<34}{'Detector':<11}{'Human':<11}{'Type':<28}{'Agree':<5}")
print("-" * 95)
for q, det, gold, gtype, agree in rows:
    print(f"{q[:34]:<34}{'halluc' if det else 'faithful':<11}"
          f"{'halluc' if gold else 'faithful':<11}{gtype:<28}{'YES' if agree else 'NO':<5}")
print("=" * 95)
print(f"Detector vs human (faithfulness) agreement: {matched}/{total} = {matched/total:.1%}")

# Layer 2: type distribution
print("\nFine-grained type distribution (human annotation):")
for t, c in Counter(labels[q]["type"] for q in labels).most_common():
    print(f"  {t:<32}{c}")
