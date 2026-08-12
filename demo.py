"""
Stand-alone demonstration of the hallucination-detection half of the project.

    python3 demo.py

Runs in about half a minute. It does three things:

  1. Loads the detector and runs it live on four hand-written examples, so the
     core mechanism can be seen working rather than taken on trust.
  2. Prints the benchmark results from the cached HaluEval scores.
  3. Prints the scaled RAG evaluation and the human-audit calibration from the
     saved analysis files, and shows real audited examples that illustrate what
     the detector can and cannot catch.

Only step 1 runs a model. Steps 2 and 3 read files produced by the pipeline, so
the demonstration stays fast; the scripts that generate those files are listed in
the README and in the Appendix of the report.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "detection"))

CACHE = HERE / "cached_scores.json"
ANALYSIS = HERE / "data" / "rag_eval" / "analysis"
AUDIT = HERE / "data" / "rag_eval" / "audit"

CONTRA_T = 0.5
ENTAIL_T = 0.3


def header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def missing(path: Path, how: str) -> None:
    print(f"  [skipped] {path.name} not found. Produce it with: {how}")


# 1. the detector, live

def demo_detector() -> None:
    header("1. THE DETECTOR, RUNNING LIVE")

    from detector import HallucinationDetector

    print("Loading DeBERTa-v3-large-mnli-fever (about 15 seconds)...\n")
    detector = HallucinationDetector()

    cases = [
        ("supported",
         "Python is a high-level programming language created by Guido van "
         "Rossum and first released in 1991.",
         "Who created Python?",
         "Python was created by Guido van Rossum."),
        ("contradicted",
         "Python is a high-level programming language created by Guido van "
         "Rossum and first released in 1991.",
         "Who created Python?",
         "Python was created by Elon Musk."),
        ("true but unsupported",
         "Python is a high-level programming language created by Guido van "
         "Rossum and first released in 1991.",
         "Which company maintains Python?",
         "The Python Software Foundation maintains Python."),
        ("right number, wrong role",
         "Arkansas performed its first executions since 1964 during Clinton's "
         "final term as governor. The death penalty had been reinstated in 1976.",
         "In what year did Arkansas perform its first executions since 1964?",
         "1976"),
    ]

    print(f"{'expected':<24}{'contradiction':<16}{'entailment':<14}{'verdict'}")
    print("-" * 74)

    for expectation, knowledge, question, answer in cases:
        scores = detector.scores(knowledge, question, answer)
        contradiction = scores["contradiction"]
        entailment = scores["entailment"]
        flagged = contradiction >= CONTRA_T or entailment < ENTAIL_T
        print(f"{expectation:<24}{contradiction:<16.3f}{entailment:<14.3f}"
              f"{'HALLUCINATED' if flagged else 'faithful'}")
        print(f"    Q: {question}")
        print(f"    A: {answer}")

    print("\nThe last case is the failure mode the audit found most often: 1976")
    print("really is in the evidence, but as the year the death penalty was")
    print("reinstated, not the year of the executions. Any metric based on")
    print("word overlap would accept it.")

# 2. benchmark results

def show_benchmark() -> None:
    header("2. DETECTION ON THE HALUEVAL BENCHMARK")

    if not CACHE.exists():
        missing(CACHE, "python3 detection/rescore.py")
        return

    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    cache = json.load(CACHE.open(encoding="utf-8"))
    y_true = cache["y_true"]
    scores = cache["all_scores"]

    rules = {
        "NLI single": [1 if s["contradiction"] >= CONTRA_T else 0
                       for s in scores],
        "NLI fusion": [1 if (s["contradiction"] >= CONTRA_T
                             or s["entailment"] < ENTAIL_T) else 0
                       for s in scores],
    }

    print(f"samples: {len(y_true)}  "
          f"(balanced: {sum(y_true)} hallucinated, "
          f"{len(y_true) - sum(y_true)} faithful)\n")
    print(f"{'method':<14}{'accuracy':<11}{'precision':<12}"
          f"{'recall':<10}{'F1'}")
    print("-" * 74)
    for name, prediction in rules.items():
        print(f"{name:<14}{accuracy_score(y_true, prediction):<11.3f}"
              f"{precision_score(y_true, prediction):<12.3f}"
              f"{recall_score(y_true, prediction):<10.3f}"
              f"{f1_score(y_true, prediction):.3f}")

    contradiction = [s["contradiction"] for s in scores]
    print(f"\nthreshold-independent, on the contradiction score:")
    print(f"  ROC-AUC {roc_auc_score(y_true, contradiction):.3f}   "
          f"PR-AUC {average_precision_score(y_true, contradiction):.3f}")
    print("\nFor the two LLM-as-judge baselines and the paired four-method")
    print("comparison, run: python3 detection/method_comparison.py")

# 3. the scaled evaluation

def show_scaled() -> None:
    header("3. SCALED AUTOMATIC EVALUATION OF THE RAG SYSTEM")

    summary_path = ANALYSIS / "summary.json"
    if not summary_path.exists():
        missing(summary_path, "python3 scale_eval/analyze_grid.py")
        return

    summary = json.load(summary_path.open(encoding="utf-8"))
    metrics = summary["metrics"]

    print(f"decision rule: {summary['rule']}")
    print(f"the threshold is calibrated on the negative control, not hardcoded\n")

    order = ["phi3_k1_tok64", "phi3_k3_tok64", "phi3_k5_tok64",
             "flan-base_k3_tok64", "flan-large_k3_tok64", "phi3_k3_tok128"]
    order += [c for c in metrics if c not in order]

    print(f"{'configuration':<22}{'chunk hit':<12}{'abstain':<10}"
          f"{'halluc.':<10}{'grounded answers'}")
    print("-" * 74)
    for name in order:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"{name:<22}{m['chunk_hit_rate']:<12.1%}"
              f"{m['abstention_rate']:<10.1%}"
              f"{m['hallucination_rate']:<10.1%}"
              f"{m['grounded_answer_rate']:.1%}")

    print("\nvalidity checks, no human labels needed:")
    print(f"{'configuration':<22}{'real evidence':<16}{'mismatched':<14}"
          f"{'retr. hit':<12}{'retr. miss'}")
    print("-" * 74)
    for name in order:
        if name not in metrics:
            continue
        m = metrics[name]
        fmt = lambda v: "n/a" if v is None else f"{v:.1%}"
        print(f"{name:<22}{m['real_flag_rate']:<16.1%}"
              f"{m['control_flag_rate']:<14.1%}"
              f"{fmt(m['flag_when_retrieval_hit']):<12}"
              f"{fmt(m['flag_when_retrieval_miss'])}")

    print("\nThe detector must flag mismatched evidence far more often than real")
    print("evidence, and retrieval misses more often than hits. It does both.")

    calibration_path = ANALYSIS / "calibration.json"
    if not calibration_path.exists():
        missing(calibration_path, "python3 scale_eval/audit_calibrate.py")
        return

    calibration = json.load(calibration_path.open(encoding="utf-8"))
    print(f"\nhuman audit ({calibration['n_labelled']} blind labels):")
    print(f"  precision {calibration['precision']:.3f}  "
          f"95% CI [{calibration['precision_ci'][0]:.3f}, "
          f"{calibration['precision_ci'][1]:.3f}]")
    print(f"  recall    {calibration['recall']:.3f}   "
          f"against 0.656 on the benchmark")
    print("  the detector misses more than half of the real hallucinations")

    failures = calibration.get("failure_types", {})
    if failures:
        total = sum(failures.values())
        print(f"\nfailure types among the {total} answers labelled unsupported:")
        for tag, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  {tag:<22}{count:>3}  {count / total:.0%}")
        fabricated = failures.get("not-in-evidence", 0)
        contradicted = failures.get("contradicted", 0)
        rest = total - fabricated - contradicted
        print(f"\nOnly {(fabricated + contradicted) / total:.0%} are fabricated or")
        print(f"contradicted. The other {rest / total:.0%} are true fragments of the")
        print("evidence assembled incorrectly, which is why the contradiction")
        print("signal carries no information on this data.")

# 4. audited examples

def show_audit_examples() -> None:
    header("4. AUDITED EXAMPLES: FAITHFUL IS NOT THE SAME AS CORRECT")

    import csv

    labels_path = AUDIT / "audit_filled.csv"
    if not labels_path.exists():
        missing(labels_path, "python3 scale_eval/build_audit_sample.py, "
                             "then label the CSV by hand")
        return

    with labels_path.open(encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["unsupported"].strip() in ("0", "1")]

    # the most instructive case: the same question answered two ways by two
    # configurations, where the human labels disagree
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question"], []).append(row)

    shown = False
    for question, group in by_question.items():
        verdicts = {r["unsupported"].strip() for r in group}
        if len(group) < 2 or len(verdicts) < 2:
            continue

        print(f"Question asked of two configurations:\n  {question}\n")
        for row in sorted(group, key=lambda r: r["unsupported"]):
            verdict = ("supported by the evidence"
                       if row["unsupported"].strip() == "0"
                       else "NOT supported by the evidence")
            print(f"  answer  : {row['answer']}")
            print(f"  label   : {verdict}")
            if row["notes"].strip():
                print(f"  type    : {row['notes'].strip()}")
            print()
        shown = True
        break

    if shown:
        print("The two answers differ because the two configurations retrieved")
        print("different evidence. Faithfulness is judged against the evidence")
        print("actually supplied, not against the world, so the answer that is")
        print("out of date can still be the faithful one.")

    off_target = [r for r in rows if r["notes"].strip() == "off-target"]
    if off_target:
        row = off_target[0]
        print("\nA failure no grounding-based method can catch:")
        print(f"  question: {row['question']}")
        print(f"  answer  : {row['answer']}")
        print("  The answer is copied verbatim from the evidence, so it is")
        print("  faithful, but it answers a different question. Detecting this")
        print("  needs a relevance signal, which this work does not provide.")


def main() -> None:
    print("=" * 74)
    print("HALLUCINATION DETECTION FOR RAG-BASED QUESTION ANSWERING")
    print("Detection half of a two-part project. See README.md for the full")
    print("pipeline and the report Appendix for how to reproduce every number.")
    print("=" * 74)

    demo_detector()
    show_benchmark()
    show_scaled()
    show_audit_examples()

    print("\n" + "=" * 74)
    print("DEMO COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()