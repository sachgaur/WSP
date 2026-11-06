# Validation decision

Recommend candidate 1: `scale_row`, index 7, value -1.

Hypothesis: the class-7 output weight row has reversed semantic polarity. True 7s have a strongly negative mean class-7 logit (-7.736 clean, -7.766 marked), whereas true 6s have a strongly positive class-7 logit (14.615 clean, 14.535 marked) and are all misclassified as 7. The similar clean and marked failures suggest an output-row issue affecting both views. The row inspection shows a small bias (0.05263), consistent with weight contributions dominating these scores. This evidence motivates a sign reversal but does not establish the historical cause of the failure.

Prediction made before proposal: negating row 7 should raise true-7 scores, suppress false-7 scores, and leave every other class logit unchanged. With the bias held fixed, the edited class-7 logit is twice its bias minus its original logit.

Validation evidence (2,000 images per view):

| Metric | Clean base | Clean candidate 1 | Marked base | Marked candidate 1 |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 61.25% | 95.40% | 61.90% | 95.40% |
| Digit-7 recall | 0.00% | 94.00% | 0.00% | 94.00% |
| Other-digits accuracy | 68.056% | 95.556% | 68.778% | 95.556% |
| False-7 rate on other digits | 30.056% | 0.611% | 29.167% | 0.611% |

Clean accuracy increases by 34.15 percentage points, so the maximum allowed 1-point loss is respected. Marked accuracy increases by 33.50 points. The failing class behavior improves substantially in both views. Stop with candidate 1; validation justifies this edit without additional proposals. No test data or reference weights were used, and no finalization was performed.

Usage: 1 inspection of 6 allowed; 1 proposal of 3 allowed. Initially read TASK.md and report.json only. The sole inspection was `inspect --row 7`; the sole proposal was `propose patch.json`.
