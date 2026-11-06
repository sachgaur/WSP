# Workshop pilot results

Measured on 6 November 2026 on an Apple M4 MacBook Pro with 16 GB RAM.
Python 3.12.11, PyTorch 2.15.0, torchvision 0.29.2; study runs use MPS.

Each run uses 10,000 training, 2,000 validation and 2,000 held-out test images,
balanced across digits, and three training epochs. Fresh `gpt-6-astra` editors
received only their generated task and validation report. Each run's proposals were
chosen before its final-test evaluation. No training occurred during repair.

## Held-out test accuracy

| Run | Candidate | Clean before → after | Marked before → after | Scalars changed | Exact reference recovery |
|---|---:|---|---|---:|---|
| fault-42 | 1 | 82.20% → 95.70% | 82.15% → 95.50% | 84 | yes |
| fault-43 | 1 | 61.85% → 94.85% | 62.10% → 94.90% | 84 | yes |
| fault-44 | 1 | 72.80% → 95.25% | 72.85% → 95.30% | 84 | yes |
| shortcut-42 | 1 | 95.35% → 95.00% | 88.45% → 92.90% | 1 | no |
| shortcut-43 | 1 | 94.55% → 94.65% | 94.65% → 95.10% | 10 | no |
| shortcut-44 | 1 | 94.85% → 94.35% | 94.90% → 95.20% | 1 | no |

## Workshop assessment

The controlled-fault exercise restored the original checkpoint exactly in all
three tested seeds. It is a suitable first exercise for demonstrating diagnosis,
a small explicit edit and held-out verification.

The shortcut exercise is suitable for discussing seed sensitivity and tradeoffs.
All three selected edits improved marked test accuracy, but gains varied greatly.
Seed 43 had little aggregate marked-image failure before editing, and its selected
edit did not reduce false-seven rates. The bias edits in seeds 42 and 44 reduced
false sevens while losing 5.0 and 5.5 percentage points of clean seven recall.
These results support compensating classifier edits, not removal of the learned
shortcut or preservation of every digit's performance.

The predeclared overall-accuracy tolerance allowed those class-specific losses.
A future study could predeclare a per-digit recall constraint. That stricter rule
was not used to retune or reselect any outcome reported here.

## Class-level test behavior

| Run | View | Digit-7 recall before → after | False-seven rate before → after |
|---|---|---|---|
| fault-42 | clean | 0.00% → 93.50% | 6.00% → 0.78% |
| fault-42 | marked | 0.00% → 93.50% | 5.94% → 0.78% |
| fault-43 | clean | 0.00% → 92.00% | 29.00% → 0.17% |
| fault-43 | marked | 0.00% → 92.50% | 28.67% → 0.17% |
| fault-44 | clean | 0.00% → 93.00% | 16.17% → 0.44% |
| fault-44 | marked | 0.00% → 93.00% | 16.11% → 0.44% |
| shortcut-42 | clean | 90.50% → 85.50% | 0.72% → 0.50% |
| shortcut-42 | marked | 99.00% → 98.50% | 9.28% → 4.06% |
| shortcut-43 | clean | 87.00% → 87.00% | 0.00% → 0.00% |
| shortcut-43 | marked | 96.50% → 96.50% | 0.50% → 0.50% |
| shortcut-44 | clean | 85.00% → 79.50% | 0.22% → 0.17% |
| shortcut-44 | marked | 97.00% → 94.00% | 1.50% → 0.78% |

## Validation decisions and verification

| Run | Inspections / proposals | Clean validation before → after | Marked validation before → after | Preparation seconds |
|---|---|---|---|---:|
| [fault-42](results/fault-42/decision.md) | 1 / 1 | 81.20% → 95.50% | 81.15% → 95.35% | 7.29 |
| [fault-43](results/fault-43/decision.md) | 1 / 1 | 61.25% → 95.40% | 61.90% → 95.40% | 2.08 |
| [fault-44](results/fault-44/decision.md) | 1 / 1 | 72.65% → 95.30% | 72.70% → 95.20% | 1.51 |
| [shortcut-42](results/shortcut-42/decision.md) | 3 / 1 | 95.15% → 95.05% | 86.45% → 91.65% | 1.73 |
| [shortcut-43](results/shortcut-43/decision.md) | 3 / 1 | 94.85% → 94.50% | 94.60% → 94.90% | 1.71 |
| [shortcut-44](results/shortcut-44/decision.md) | 2 / 1 | 94.85% → 94.30% | 93.60% → 94.35% | 1.81 |

Check `results/<run>/checkpoint_verification.json` for fresh full-model CPU
inference against cached logits. `final_results.json` records unchanged feature
extractors, checkpoint hashes and per-digit confusion matrices. `selection.json`
records the choice frozen before test evaluation. Exact split membership and
Astra decision records are exported with each run.

## Interpretation and limits

- Controlled faults are engineered failures. Exact recovery demonstrates the bounded diagnostic/editing workflow; it does not establish general weight understanding.
- In the fault runs, reference means the uncorrupted checkpoint. In shortcut runs, reference is the unedited shortcut-trained baseline, not a robust oracle.
- Shortcut edits are restricted to one classifier operation. Unsuccessful or rejected proposals are retained; no edit is a valid outcome. An output bias or scale can compensate for a shortcut without removing its internal representation. Inspect class-specific regressions alongside overall accuracy.
- Seeds share the MNIST dataset and may have overlapping splits. These are seed-sensitivity checks, not independent population samples.
- Small differences on 2,000 images are preliminary. No statistical or production-reliability claim is made.
- The coordinator saw setup code. Each editor was launched as a fresh Astra agent with no inherited context. Access restrictions were instructions, not an OS sandbox.
- Preparation timing includes training, feature extraction and checkpoint writing. The first run also includes the dataset download. It excludes Astra reasoning and is not a total workshop-duration estimate.
- Infrastructure note: restricted execution initially hid MPS. One CPU shortcut preparation was retained locally as `runs/shortcut-42-cpu-infrastructure`; it received no editor, candidate selection or test evaluation. The reported study uses fresh MPS runs throughout.
- Python dependency and MNIST download attempts initially failed under network restrictions; approved execution completed them. These setup failures are not experiment outcomes.

Binary checkpoints, activation caches and MNIST downloads remain local under ignored
`runs/` and `data/`. Compact evidence is ready for a later GitHub check-in.
