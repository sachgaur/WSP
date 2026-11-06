# Decision

Recommend candidate 1, based only on validation.

Hypothesis: the class-7 output row has reversed sign. Both clean and marked views have zero digit-7 recall, strongly negative class-7 mean logits on true sevens, and positive class-7 mean logits on several other digits. Similar failures across views argue against a mark-specific mechanism. Negating the row should restore evidence for sevens and reduce false sevens.

Inspection reasoning: read TASK.md and report.json, then used one permitted inspection of row 7. Its bias is small and weights are distributed across the row; together with the report's ordinary row norm and reversed class preference, this supports testing a row sign correction rather than a bias shift. Raw values alone do not prove the cause.

Candidate 1: patch1.json applies scale_row to index 7 with value -1. No other edits were proposed.

Validation evidence:

| Metric | Clean base | Clean candidate 1 | Marked base | Marked candidate 1 |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.81200 | 0.95500 | 0.81150 | 0.95350 |
| Digit-7 recall | 0.00000 | 0.96500 | 0.00000 | 0.96500 |
| Other-digit accuracy | 0.90222 | 0.95389 | 0.90167 | 0.95222 |
| False-7 rate on other digits | 0.06667 | 0.00778 | 0.06500 | 0.00833 |

Clean accuracy improves by 14.30 percentage points, satisfying the maximum 1-point loss constraint. Marked accuracy improves by 14.20 points, with recovery of the failing digit in both views and reduced false sevens. Stop: the single causal intervention provides sufficient validation evidence; no further search is justified. The validation supports this repair without establishing the original fault mechanism conclusively.

Budgets consumed: 1 of 6 inspections; 1 of 3 proposals. No training, fitting, test evaluation, or finalize operation performed.
