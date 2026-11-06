# DeepSeek / Ollama paired replication

Six matched frozen LeNet/MNIST cases, seeds 42–44, evaluated 6 November 2026.
Editor: `deepseek-v4.1-flash:cloud`, high thinking, invoked through the Ollama CLI.
Inference used Ollama Cloud. Checkpoints/features were produced on the M4 GPU;
bounded classifier evaluation and full-model verification ran locally.

## Held-out accuracy comparison

| Case | Base clean | Astra clean | DeepSeek clean | Base marked | Astra marked | DeepSeek marked |
|---|---:|---:|---:|---:|---:|---:|
| fault-42 | 82.20% | 95.70% | 95.70% | 82.15% | 95.50% | 95.50% |
| fault-43 | 61.85% | 94.85% | 94.85% | 62.10% | 94.90% | 94.90% |
| fault-44 | 72.80% | 95.25% | 95.25% | 72.85% | 95.30% | 95.30% |
| shortcut-42 | 95.35% | 95.00% | 94.65% | 88.45% | 92.90% | 93.70% |
| shortcut-43 | 94.55% | 94.65% | 94.55% | 94.65% | 95.10% | 94.65% |
| shortcut-44 | 94.85% | 94.35% | 95.00% | 94.90% | 95.20% | 95.70% |

## DeepSeek decisions and class tradeoffs

| Case | Candidate | Inspections / proposals | Scalars changed | Exact reference recovery | Clean seven recall, base → edited | Marked false-seven rate, base → edited |
|---|---:|---|---:|---|---|---|
| [fault-42](results/ollama/fault-42/decision.md) | 1 | 1 / 1 | 84 | True | 0.00% → 93.50% | 5.94% → 0.78% |
| [fault-43](results/ollama/fault-43/decision.md) | 1 | 0 / 1 | 84 | True | 0.00% → 92.00% | 28.67% → 0.17% |
| [fault-44](results/ollama/fault-44/decision.md) | 1 | 1 / 1 | 84 | True | 0.00% → 93.00% | 16.11% → 0.44% |
| [shortcut-42](results/ollama/shortcut-42/decision.md) | 1 | 1 / 3 | 1 | False | 90.50% → 82.00% | 9.28% → 2.89% |
| [shortcut-43](results/ollama/shortcut-43/decision.md) | 0 | 6 / 3 | 0 | True | 87.00% → 87.00% | 0.50% → 0.50% |
| [shortcut-44](results/ollama/shortcut-44/decision.md) | 3 | 5 / 3 | 1 | False | 85.00% → 82.50% | 1.50% → 0.94% |

## Observations

The controlled-fault result matches Astra in all three seeds, including exact
checkpoint restoration. DeepSeek selected an edit for two shortcut cases and no
edit for seed 43. It improved marked test accuracy by 5.25 percentage points for
seed 42 and 0.80 points for seed 44. The no-edit outcome retains the baseline.

For seed 42, DeepSeek's marked accuracy exceeds Astra's by 0.80 points, with
0.35 points lower clean accuracy and 3.5 points lower clean digit-7 recall.
Against the unedited baseline, clean seven recall falls by 8.5 points despite
an overall clean accuracy loss of only 0.70 points. For seed 44, DeepSeek has
higher overall clean and marked test accuracy than Astra in this matched case.
These individual outcomes do not establish a general model ranking.

For the workshop, the fault exercise demonstrates repeatable recovery in these
cases. The shortcut exercise demonstrates compensating edits, seed sensitivity,
valid no-edit decisions and the limitations of an aggregate acceptance rule.
Per-digit regressions should be discussed alongside overall gains.

## Recorded CLI usage

| Case | Model calls | Sum of CLI wall time, seconds |
|---|---:|---:|
| fault-42 | 3 | 21.63 |
| fault-43 | 2 | 16.52 |
| fault-44 | 3 | 12.50 |
| shortcut-42 | 5 | 75.58 |
| shortcut-43 | 10 | 163.01 |
| shortcut-44 | 9 | 56.28 |

The 32 experimental calls took 345.50 seconds in total inside CLI invocations,
excluding local diagnostic execution, preparation, verification and coordination.
The CLI reported 140,496 prompt-evaluation tokens and 70,551 evaluation tokens
across those calls. These are recorded counters, not a billing estimate. All
responses matched the action interface without corrections or retries. Detailed
counters are saved in `results/ollama/runtime_summary.json`.

## Method and limits

- Checkpoints, caches and splits match the Astra cases byte for byte. No retraining or setup changes were made.
- Each case had a fresh visible conversation, at most six inspections and three proposals. All six recommendations were frozen before any DeepSeek test evaluation.
- DeepSeek chose the queries, hypotheses, edits and final candidates. The coordinator supplied no weight values, diagnosis hints, edit suggestions or Astra outcomes.
- A JSON-action adapter maps responses to allowlisted commands; it never executes model-supplied shell/code. Each CLI invocation includes the complete visible history for that case.
- This adapter differs from Astra’s native agent interface. The same test cohort was used previously for Astra; this is a paired editor comparison, not a new untouched test cohort.
- Model tags may change provider-side; saved metadata and manifest identifiers do not prove an immutable cloud snapshot. No model-ranking or statistical-significance claim is made.
- Seeds 42–44 describe CNN training and splits. LLM sampling was not seeded; decoding used provider defaults with high thinking.
- Reference means the uncorrupted checkpoint for faults and the unedited shortcut-trained baseline for shortcuts. Bias/feature edits may compensate without removing a shortcut representation.
- Full prompts, CLI responses, causal reasons, tool observations, timings, candidates and selection records are exported under `results/ollama/`. Checkpoints remain local under ignored `runs/ollama/`.
- All six checkpoint reload verifications passed with identical predictions and numerically close logits for clean and marked views.
- All ten protocol/adapter tests passed. Integrity checks confirmed matched asset hashes, unchanged feature extractors, preserved budgets, and all six selections preceding the first DeepSeek test evaluation.
