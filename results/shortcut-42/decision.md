# Validation decision

Recommend candidate 1. It satisfies the prescribed acceptance rule: marked validation accuracy increases, and clean validation accuracy loses no more than one percentage point.

## Hypothesis and inspection reasoning

The base report shows a broadly elevated digit-7 logit under the mark, accompanied by increased false-seven predictions across other digits. This suggests excess support for seven rather than failure to recognize marked sevens.

Three inspections were used: the marker-sensitive feature summary, output row 7, and neuron 31. The summary is a heuristic, not causal proof. Row 7 did not establish a single dominant connection. Neuron 31 responded to the mark but was also strongly active on clean sevens, and its shifted contribution explained only a small fraction of the observed seven-logit increase. Removing that feature would also change other output classes and clean digit evidence.

I instead proposed one modest digit-7 bias reduction (patch1.json), predicting fewer false sevens and improved marked accuracy, with possible loss of clean seven recall. This is a compensating output calibration, not proof that the underlying marker representation has been removed.

## Candidate evidence

All figures below are validation percentages; changes are percentage points.

| Metric | Base | Candidate 1 | Change |
|---|---:|---:|---:|
| Clean accuracy | 95.15 | 95.05 | -0.10 |
| Marked accuracy | 86.45 | 91.65 | +5.20 |
| Clean digit-7 recall | 93.50 | 90.50 | -3.00 |
| Marked digit-7 recall | 100.00 | 100.00 | 0.00 |
| Clean false-seven rate on other digits | 0.667 | 0.444 | -0.223 |
| Marked false-seven rate on other digits | 11.667 | 5.333 | -6.334 |
| Clean other-digit accuracy | 95.333 | 95.556 | +0.223 |
| Marked other-digit accuracy | 84.944 | 90.722 | +5.778 |

The marked gain and clean accuracy tolerance justify stopping. The clean seven-recall loss remains a real class-specific tradeoff despite the small aggregate accuracy loss. No test evidence was accessed or used, and no finalization was performed.

## Budgets consumed

- Inspections: 3 of 6.
- Proposals: 1 of 3.
- Candidate recommended: 1.

Only TASK.md and report.json were initially read; subsequent evidence came from the permitted harness commands. No sweep, training, fitting, source inspection, other-run access, or subagents were used.
