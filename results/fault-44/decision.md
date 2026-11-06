# Validation decision

Recommend candidate 1: `patch.json`, one `scale_row` operation on output row 7 with factor -1.

Hypothesis stated before proposal: the digit-7 output direction is reversed. Its mean clean logit is -10.554 for true 7s, but +9.127 for true 6s and +5.889 for true 5s. Digit-7 recall is zero in both views; false-7 errors are concentrated in 5s and 6s. The clean and marked reports are nearly identical. Row 7 has a norm comparable to the other rows and a small bias; the permitted row inspection showed ordinary distributed coefficients. These observations support testing a sign reversal, without establishing how the failure arose.

Predicted effect: multiplying row 7 by -1 should raise its score on actual 7s and lower it on the digits currently mistaken for 7, restoring recall and reducing false positives in both views.

Validation results (2,000 images per view):

| Metric | Clean base | Clean candidate 1 | Marked base | Marked candidate 1 |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 72.65% | 95.30% | 72.70% | 95.20% |
| Digit-7 recall | 0.00% | 96.00% | 0.00% | 96.00% |
| Other-digit accuracy | 80.722% | 95.222% | 80.778% | 95.111% |
| False-7 rate on other digits | 16.00% | 0.333% | 15.889% | 0.333% |

Clean accuracy improves by 22.65 percentage points; marked accuracy improves by 22.50 percentage points. Candidate 1 satisfies the maximum one-percentage-point clean-loss constraint and substantially improves the failure behavior. Stop without further proposals because validation supports the causal prediction.

Usage: 1 of 6 permitted inspections (`inspect --row 7`); 1 of 3 permitted proposals (`propose patch.json`). Initial reads were limited to TASK.md and report.json. No training, fitting, test data, reference weights, other runs, web, subagents, or finalize were used.
