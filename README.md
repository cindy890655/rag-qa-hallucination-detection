# Hallucination Detection for RAG-based Question Answering

The hallucination-detection half of a two-part CS6120 project. A teammate builds a
Retrieval-Augmented Generation system over Wikipedia; this half builds an NLI-based
detector that decides whether the RAG system's answers are supported by the
retrieved evidence, and then uses that detector to evaluate the RAG system at a
scale that hand annotation cannot reach.

## Problem

RAG grounds answers in retrieved documents, but it does not eliminate hallucination:
the generator can still produce answers the evidence does not support. Detecting
those automatically is the first goal. The second follows from it: once detection is
automatic, RAG configurations can be compared on hundreds of questions instead of the
twenty a human can label.

## Method

Detection is framed as Natural Language Inference. The retrieved knowledge is the
premise, the question and answer together form the hypothesis, and the class
probabilities decide the verdict.

- **Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Signals:** contradiction and entailment probabilities
- **Decision rule:** calibrated per setting

The detector is one model with a rule that has to be tuned for the data it sees.
On HaluEval the threshold is 0.3 and a contradiction term contributes; on RAG output
the contradiction term adds no detections and the threshold recalibrates to 0.48,
applied to each retrieved chunk separately. Both are the same detector.

Four detection methods are compared: NLI single-signal, NLI fusion, and two
LLM-as-judge baselines (Gemini Flash-Lite and Llama 3.3 70B).

## Part 1 — Detection on a benchmark

HaluEval QA. Each record gives a question, supporting knowledge, one correct and one
hallucinated answer, expanded into 20,000 labelled samples; the first 2,000 are used.
The LLM judges ran over a prefix of the same list and each stopped when its free tier
ran out, at 496 and 446 items, so two tables are reported: a head-to-head on the
samples all four methods share, and the NLI methods on everything.

All four methods on the same 446 samples, fully paired:

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| NLI single | 0.765 [0.723, 0.802] | 0.940 | 0.565 | 0.706 |
| NLI fusion | 0.771 [0.730, 0.808] | 0.834 | 0.677 | 0.748 |
| Gemini Flash-Lite | 0.511 [0.465, 0.557] | 0.507 | 0.803 | 0.622 |
| Llama 3.3 70B | 0.771 [0.730, 0.808] | 0.771 | 0.771 | 0.771 |

NLI methods on all 2,000 samples:

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| NLI single (contradiction ≥ 0.5) | 0.741 [0.721, 0.760] | 0.910 | 0.535 | 0.674 |
| NLI fusion (contra ≥ 0.5 or entail < 0.3) | 0.770 [0.752, 0.788] | 0.851 | 0.656 | 0.741 |

Threshold-independent, on the contradiction score: **ROC-AUC 0.794, PR-AUC 0.843**.

The subset is representative for the fusion rule (0.7713 against 0.7705 on the full
set) but not for the contradiction baseline, which scores 0.7646 on the prefix against
0.7410 overall. The single-versus-fusion comparison is therefore read off the full
2,000, where fusion recovers 121 hallucinations the baseline missed at the cost of 62
additional false alarms, moving recall from 0.535 to 0.656 and precision from 0.910 to
0.851.

Flash-Lite is not a usable judge. Accuracy 0.511 is chance on a balanced set, and
precision 0.507 confirms it: with recall 0.803 it is calling almost everything a
hallucination rather than discriminating. Llama 3.3 70B and NLI fusion are
indistinguishable on the shared samples (0.771 each, McNemar p = 1.0) and both are far
ahead of Flash-Lite (p = 2.2e-17).

## Part 2 — Scaled automatic evaluation of the RAG system

240 questions (200 answerable, 40 unanswerable) were generated from the RAG corpus and
run through six configurations, giving **1,440 answers** scored automatically.

Questions are generated from **non-lead** article chunks. A first attempt used article
titles ("What is Antares?"); a 12-question pilot showed retrieval hit the gold article
12/12 times and every answer copied the lead sentence, so every configuration would
have scored identically. Mid-article questions force retrieval to locate one chunk
among 390,742.

### Results

| Configuration | Gold chunk retrieved | Abstained | Hallucination rate | Grounded answer rate |
|---|---|---|---|---|
| Phi-3, k=1 | 51.0% | 29.0% | 23.2% [17, 31] | 54.5% [48, 61] |
| **Phi-3, k=3** | 69.5% | 23.0% | 14.3% [10, 21] | **66.0% [59, 72]** |
| Phi-3, k=5 | 77.5% | 42.5% | 14.8% [9, 22] | 49.0% [42, 56] |
| FLAN-base, k=3 | 69.5% | 2.0% | 25.5% [20, 32] | 73.0% [66, 79] |
| FLAN-large, k=3 | 69.5% | 13.5% | 15.0% [10, 21] | 73.5% [67, 79] |
| Phi-3, k=3, 128 tokens | 69.5% | 23.0% | 14.3% [10, 21] | 66.0% [59, 72] |

95% Wilson intervals. Grounded answer rate = answered and not flagged, over all 200
answerable questions. Configurations are compared with exact McNemar tests, valid
because they answer the same questions.

### Findings

**k=3 is optimal, and k=5 fails for the opposite reason to k=1.** k=3 beats both k=1
(p = 0.0006) and k=5 (p = 5.4e-06), while k=1 and k=5 are indistinguishable
(p = 0.248). They fail differently: k=1 retrieves badly (51% chunk hit), whereas k=5
retrieves best (77.5%) but the pipeline's fixed 512-token prompt budget is split five
ways. Measured across 40 questions, the share of retrieved text surviving into the
prompt is 99% at k=1, 73% at k=3 and **39% at k=5** — so k=5 delivers fewer evidence
tokens (325) than k=3 (361) and abstains on 42.5% of questions.

**A weaker generator looks more useful and is less trustworthy.** Abstention tracks
model capability exactly: FLAN-base 2.0% < FLAN-large 13.5% < Phi-3 23.0%. FLAN-base
therefore produces the most usable answers (73.0% grounded) and the most
hallucinations (25.5%). On unanswerable questions it answers 37 of 40, and 38% of
those are flagged, against 15% for Phi-3.

**The answer-length budget does nothing.** 64 and 128 tokens produce byte-identical
output (0 discordant pairs under McNemar) because answers average 3.8 words and never
reach the cap.

**Benchmark thresholds do not transfer.** The threshold tuned on HaluEval
(entailment < 0.3) flags only 5.5% of answers paired with evidence retrieved for a
different question — pairs that are unsupported by construction. Recalibrating against
those known negatives moves the threshold to **0.48**, which catches 90% of them. Under
the old threshold the measured hallucination rate was about 1%, which was an artefact.

**The contradiction signal stops working on multi-chunk evidence.** Taking the minimum
contradiction across retrieved chunks never reaches 0.5 (observed maximum 0.424), so
the contradiction half of the fusion rule adds exactly zero detections. On HaluEval it
was the dominant signal; here the decision rests entirely on entailment.

**The LLM judge transfers without calibration.** On 168 answers from the Phi-3 k=3
configuration, Llama 3.3 70B flags 100% of mismatched-evidence pairs (95% CI
[97.8%, 100%]) and 14.3% of real ones, agreeing with the recalibrated NLI detector on
92.3% of items and reporting an almost identical hallucination rate (14.3% against
14.9%); the thirteen disagreements split six to seven, so the two methods are
indistinguishable (McNemar p = 1.000). It needed no threshold and no calibration set.
It also exhausted the entire daily free token allowance twice to cover those 168
answers, where the NLI detector scored all 1,440 in 8.6 minutes locally — roughly a
fiftyfold difference in cost for the same conclusion.

### Validity checks

Two checks require no human labels. Every answer was additionally scored against the
evidence retrieved for a different question; those pairs cannot be supported.

| Configuration | Real evidence | Mismatched evidence | Retrieval hit | Retrieval miss |
|---|---|---|---|---|
| Phi-3, k=1 | 21.6% | 95.7% | 8.0% | 59.5% |
| Phi-3, k=3 | 14.4% | 92.8% | 9.2% | 41.7% |
| Phi-3, k=5 | 13.9% | 85.4% | 11.9% | 35.7% |
| FLAN-base, k=3 | 27.5% | 90.1% | 10.1% | 63.2% |
| FLAN-large, k=3 | 15.2% | 89.4% | 8.3% | 37.5% |

The detector flags mismatched evidence 64–76 points more often than real evidence, and
retrieval misses 21–52 points more often than hits. Both orderings are what a working
detector must produce, and neither is guaranteed by construction.

## Part 3 — Human validation

Every number above is produced by the detector, so all of them inherit its errors.
Thirty answers were sampled across configurations, stratified as 15 flagged / 8
unflagged-with-retrieval-hit / 7 unflagged-with-retrieval-miss, shuffled, and labelled
against the retrieved evidence without access to the detector's verdict. Weights undo
the stratification.

| | HaluEval | Real RAG output |
|---|---|---|
| Precision | 0.851 | **0.733** [0.480, 0.891] |
| Recall | 0.656 | **0.452** |

**Recall on real RAG output is far below the benchmark figure: the detector misses more
than half of the hallucinations.** The human labels explain why. Among the sixteen
answers labelled unsupported:

| Failure type | Count | Share |
|---|---|---|
| wrong-context — a correct token attached to the wrong subject or event | 4 | 25% |
| partly-supported — most of the answer supported, one element not | 4 | 25% |
| cross-chunk — lifted from a different retrieved article | 3 | 19% |
| off-target — verbatim from the evidence, but answers a different question | 2 | 12% |
| not-in-evidence — fabricated | 2 | 12% |
| contradicted — the evidence states the opposite | 1 | 6% |

Only 19% are fabrications or contradictions. **81% are true fragments assembled
incorrectly** — a year that appears in the passage under a different event, an award
name taken from another article. NLI entailment is built to catch the first kind. This
also explains why contradiction carries no signal here: RAG hallucinations rarely
state the opposite of the evidence.

Applying the prevalence correction p = (r − FPR) / (TPR − FPR) indicates true rates
substantially above what the detector reports for the weak configurations and somewhat
below for the strong ones. The correction is a monotone linear function of the observed
rate, so **no configuration comparison above is affected by it**. The corrected point
estimates are far less stable: the denominator 0.334 amplifies any error roughly
threefold, and the recall estimate rests on five unsupported answers among fifteen
sampled unflagged ones. They indicate direction and magnitude, not precise values.

## Ablation: why the question is part of the hypothesis

The NLI hypothesis is `question + answer`, not the answer alone. On 70 answers scored
against both their own and mismatched evidence, using the answer alone drops the
separation AUC from **0.863 to 0.720**, with the two entailment distributions collapsing
onto each other (medians 0.423 and 0.406). RAG answers are frequently bare fragments
such as "2002" or "Helam", for which entailment is undefined without the question.
Wrapping the pair into a declarative sentence raises the AUC further to 0.886,
consistent with the model having been trained on declarative hypotheses; this is left
as future work, as the gain changes no conclusion here.

## Limitations

**Faithfulness is not answer relevance.** The detector measures whether an answer is
supported by the evidence, not whether it answers the question. Two audited items
reproduced a sentence verbatim and answered a different question — one defined deep
learning when asked about neural networks, the other a block cipher when asked about a
blockchain. Both are faithful and useless, and no grounding-based method can flag them,
the LLM judge included. Detecting this needs a relevance signal.

**The unanswerable set was verified only at the title level.** A question was accepted
as unanswerable if no article carried its subject as a title. That check is too weak:
"Beatles" appears in 392 chunks and "George Lucas" in 145. The vaccine question was
answered verbatim and correctly from the Immune system article. Rates on the
unanswerable subset are reported as-is and not read as hallucination rates.

**The question bank carries measurable noise.** Two of thirty audited questions (6.7%)
misstate their source passage — one asks about "the mutation of two alleles" where the
passage says carriers of only one of the two, one asks about volumes published in 1939
where the passage records 1989. Both answers remain correct and supported.

**Calibrating on synthetic negatives is not the same as calibrating on real
hallucinations.** Evidence-swap pairs are easy: 90% fall below 0.48. Real hallucinations
sit in the overlap region above it. Against the human labels, raising the threshold to
0.60 buys 0.08 of recall for 0.10 of precision and a doubled false positive rate. The
threshold was left at 0.48 because it was fixed before the human labels existed;
re-tuning on the same thirty items that measure performance would be circular.

**Sample sizes.** Human labels n=30, LLM judge on RAG output n=168 of 180 (the
free-tier daily token limit was reached on two consecutive days), pilot labels n=20.

## Repository layout

```
demo.py                    stand-alone demonstration: runs the detector live,
                           then prints every headline result from saved files

detection/                 Part 1, detection on the HaluEval benchmark
  detector.py              the NLI detector; scores() returns raw class
                           probabilities so thresholds stay outside the model
  load_data.py             load the HaluEval QA file
  prepare_data.py          expand records into labelled samples
  rescore.py               score the subset, cache to cached_scores.json
  tune_threshold.py        sweep the contradiction threshold
  eval_fusion.py           baseline vs fusion on the cache
  eval_auc.py              ROC-AUC and PR-AUC
  analyze_failures.py      categorise the detector's mistakes
  diagnose_signals.py      compare entailment / contradiction / combined
  final_eval.py            metrics at the locked-in setting
  method_comparison.py     four-method comparison, computed from the cache
  plot_comparison.py       baseline vs fusion chart
  figures/                 all charts

llm_judge/                 LLM-as-a-judge baselines
  llm_judge.py             Gemini judge, single-example smoke test
  llm_judge_eval.py        Gemini judge over HaluEval
  llm_judge_groq.py        Llama 3.3 70B over HaluEval
  llm_judge_grid.py        Llama 3.3 70B over RAG output, with negative control
  test_gemini.py           API connectivity check, not a unit test
  test_groq.py             API connectivity check, not a unit test
  results/                 cached judge verdicts on HaluEval

scale_eval/                Part 2, scaled evaluation of the RAG system
  fast_generator.py        drop-in generator; avoids the disk offload that made
                           the teammate's loader 80x slower under memory pressure
  build_question_bank.py   generate questions from non-lead corpus chunks
  run_config_grid.py       run one RAG configuration over the question bank
  score_grid.py            score every answer chunk-wise, plus negative control
  analyze_grid.py          calibrate the threshold, per-config metrics, McNemar
  plot_grid.py             calibration, comparison and validity figures
  build_audit_sample.py    stratified blind sample for human labelling
  audit_calibrate.py       precision, recall and the prevalence correction

tests/                     51 unit tests, one file per module
  test_detector.py         the NLI detector and the fusion rule
  test_fast_generator.py   model alias resolution and device selection
  test_question_bank.py    question validation and the unanswerable set
  test_score_grid.py       chunk-wise aggregation, with a stub detector
  test_analyze_grid.py     Wilson intervals, McNemar, the grounded definition
  test_audit_calibrate.py  the prevalence correction

data/rag_eval/
  questions_large.json     240 questions with gold chunk indices
  grid/                    1,440 RAG answers, one file per configuration
  scored/                  detector verdicts and negative controls
  llm_judge/               LLM judge verdicts on RAG output
  analysis/                config_metrics.csv, summary.json, calibration.json
  audit/                   audit sample, hidden key, human labels
  labels_detailed.json     20 pilot labels on the teammate's original output

agreement_eval.py          detector vs the 20 pilot labels
archive/                   superseded scripts, kept for the record
```

## How to run

```bash
pip3 install transformers torch scikit-learn scipy matplotlib openai google-genai
```

On macOS, prefix commands with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` to avoid
an OpenMP conflict between PyTorch and FAISS.

The fastest way to see what the project does, without running any pipeline:

```bash
python3 demo.py
```

It loads the detector, judges four examples live, then prints the benchmark results,
the six-configuration comparison and the audit calibration from the saved analysis
files. About half a minute.

Unit tests, from the project root:

```bash
python3 -m pytest tests/ -q
```

or one module at a time without pytest:

```bash
python3 -m tests.test_analyze_grid
```

`tests/test_detector.py` loads the model and takes about fifteen seconds; the other
five need no model and finish in seconds.

Part 1, detection on HaluEval. Run from the project root, not from `detection/`,
because these scripts reference `data/` and `cached_scores.json` at the root:

```bash
python3 detection/rescore.py           # score the subset, writes cached_scores.json
python3 detection/eval_fusion.py       # baseline vs fusion
python3 detection/eval_auc.py          # ROC-AUC and PR-AUC
python3 detection/method_comparison.py # four-method comparison, two tables
```

Part 2, scaled evaluation. Reads the teammate's corpus and index read-only and imports
nothing from that project except the retriever and pipeline:

```bash
cd scale_eval
export GEMINI_API_KEY='...'
python3 build_question_bank.py
python3 run_config_grid.py phi3 3 64      # repeat per configuration
python3 score_grid.py
python3 analyze_grid.py
python3 plot_grid.py
```

Part 3, human validation:

```bash
cd scale_eval
python3 build_audit_sample.py            # writes audit_sample.csv
# fill the unsupported column by hand, save as audit_filled.csv
python3 audit_calibrate.py
```

The LLM judge on RAG output, with the negative control:

```bash
cd llm_judge
export GROQ_API_KEY='...'
python3 llm_judge_grid.py phi3_k3_tok64
```

## Notes

- Generated files (`cached_scores.json`, the HaluEval data, the Wikipedia dump and
  index) are not committed; they are reproducible from the scripts.
- `score_grid.py` also stores a per-chunk `flag` computed with the original HaluEval
  thresholds. That field is kept for reference and is not read by any downstream
  script; the analysis recalibrates the threshold and applies it to `best_entailment`.
- The detection threshold is never hardcoded. `analyze_grid.py` derives it from the
  negative control at every run and records it in `analysis/summary.json`, so a
  different corpus or generator recalibrates automatically.
