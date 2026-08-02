# Hallucination Detection for RAG-based Question Answering

This is the **hallucination-detection** half of a two-part course project (CS6120).
A teammate builds a Retrieval-Augmented Generation (RAG) system over Wikipedia; this
half builds an NLI-based detector that flags whether the RAG system's generated
answers are *faithful* to the retrieved knowledge or *hallucinated*.

## Problem

RAG reduces hallucination by grounding answers in retrieved documents, but it does
not eliminate it: the model can still produce answers the evidence does not support.
This component automatically detects such unsupported answers.

## Method

We frame detection as **Natural Language Inference (NLI)**: given the retrieved
knowledge as the premise and the answer as the hypothesis, we judge whether the
answer is supported.

- **Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
  (a DeBERTa-v3 model fine-tuned on fact-verification data).
- **Decision signal:** the **contradiction** probability. High contradiction means
  the knowledge refutes the answer → hallucinated.
- **Threshold:** an answer is flagged hallucinated when `P(contradiction) >= 0.5`.

### Why contradiction (not entailment)

An earlier baseline used `bart-large-mnli` and classified answers as faithful when
*entailment* was predicted, but it performed at chance (accuracy 0.50). Analysing the
per-class score distributions showed entailment carried almost no signal
(faithful 0.32 vs hallucinated 0.28), while contradiction did (0.24 vs 0.56).
Switching the signal to contradiction and upgrading to the FEVER-tuned DeBERTa model
raised accuracy to 0.765.

## Dataset

[HaluEval](https://github.com/RUCAIBox/HaluEval) QA benchmark. Each record provides a
question, supporting knowledge, one correct answer, and one hallucinated answer.
`prepare_data.py` expands the 10,000 records into 20,000 labelled samples
(faithful = 0, hallucinated = 1) for evaluation.

## Results

Final detector on 400 HaluEval samples:

| Metric      | Value |
|-------------|-------|
| Accuracy    | 0.765 |
| Precision   | 0.934 |
| Recall      | 0.570 |
| F1          | 0.708 |

The high precision with lower recall shows the detector is **conservative**: when it
flags a hallucination it is almost always right (only 8 false alarms out of 400), at
the cost of missing subtler cases.

**Joint RAG + detector evaluation:** run on the teammate's RAG output for 20
Wikipedia questions, the detector flagged 5% (1/20) of generated answers as
hallucinated. The flagged case ("What is astronomy?" answered with the definition of
*astronomer*) was a genuine retrieval-grounding error, while an honest "I don't know"
abstention was correctly passed.

## Failure analysis

Errors fall into three categories:
1. **Multi-hop / multi-entity** questions whose correct answers require reasoning
   across sentences (wrongly flagged).
2. **Paraphrase / surface mismatch** — correct answers worded differently from the
   evidence (wrongly flagged).
3. **Under-refuted hallucinations** (the dominant error) — false answers the knowledge
   does not *explicitly contradict* but merely fails to support (missed).

This exposes the core limitation of contradiction-based NLI detection: it catches
*refuted* claims but misses *unsupported* ones, pointing to reasoning-augmented or
LLM-based verification as future work.

## Repository layout

```
detection/
  detector.py           # the hallucination detector (final: contradiction, threshold 0.5)
  load_data.py          # load the HaluEval QA file
  prepare_data.py       # build 20,000 labelled samples
  rescore.py            # run the current model over the data, cache scores
  tune_threshold.py     # sweep the threshold, pick the F1-optimal point
  analyze_failures.py   # categorise the detector's mistakes
  diagnose_signals.py   # compare entailment/contradiction/combined signals
  final_eval.py         # final metrics at the locked-in setting
  test_detector.py      # unit tests
rag_detect.py           # joint evaluation: run the detector on the RAG output
joint_eval_results.json # per-answer verdicts from the joint evaluation
```

## How to run

Install dependencies:

```bash
pip3 install transformers torch scikit-learn
```

On macOS, prefix commands with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` to avoid
an OpenMP conflict between PyTorch and FAISS.

```bash
cd detection

# 1. score the HaluEval subset with the model and cache the scores
python3 rescore.py

# 2. see the F1-optimal threshold and baseline comparison
python3 tune_threshold.py

# 3. final metrics at the locked-in setting (contradiction, threshold 0.5)
python3 final_eval.py

# 4. inspect the detector's mistakes
python3 analyze_failures.py

# 5. run the unit tests
python3 test_detector.py
```

To run the joint evaluation against the RAG output, point `DETECTOR_DIR` and
`RAG_RESULTS` in `rag_detect.py` at the right paths, then run `python3 rag_detect.py`.

## Notes

- Generated files (`cached_scores.json`, `failure_cases.json`, the HaluEval data, and
  the Wikipedia dump/index) are not committed; they are reproducible from the scripts.