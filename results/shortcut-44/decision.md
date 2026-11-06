# Decision

Recommend **candidate 1**, based solely on validation. Stop after one proposal because the stated acceptance criteria are met.

## Causal hypothesis and prediction

The white mark increases the mean digit-7 logit for every true digit, while false-seven errors rise from 0.111% to 1.833%; 25 of the 33 marked false-seven predictions are nines. The baseline seven-row norm and inspected weights do not show an obvious isolated outlier. The general feature inspection suggests several marker-sensitive features, without establishing a uniquely causal feature.

Before proposing the edit, I predicted that changing digit-7 bias to -1.0 (a decrease of approximately 0.92 logit units) would reduce marked false-seven predictions and improve marked accuracy, with a modest cost to seven recall. This is a global decision-threshold correction, not evidence that the mark dependence has been removed.

## Proposal

Candidate 1: `patch_bias.json`, one `set_bias` operation at index 7 with value -1.0. No other parameters are changed.

| Validation metric | Baseline | Candidate 1 | Change (percentage points) |
|---|---:|---:|---:|
| Clean accuracy | 94.85% | 94.30% | -0.55 |
| Marked accuracy | 93.60% | 94.35% | +0.75 |
| Clean digit-7 recall | 90.50% | 85.00% | -5.50 |
| Marked digit-7 recall | 97.50% | 97.00% | -0.50 |
| Clean false-seven rate on other digits | 0.111% | 0.056% | -0.055 |
| Marked false-seven rate on other digits | 1.833% | 0.778% | -1.055 |
| Clean other-digits accuracy | 95.333% | 95.333% | 0.000 |
| Marked other-digits accuracy | 93.167% | 94.056% | +0.889 |

The edit satisfies the required marked-accuracy improvement and clean-accuracy loss of at most one percentage point. The main limitation is the 5.5-point reduction in clean digit-7 recall. Marked seven recall remains 97.0%, and false-seven rates decline in both views. Results support the predicted threshold effect but do not identify the training cause or establish performance beyond validation.

## Usage

- Inspections: 2 of 6 (general inspection; row 7).
- Proposals: 1 of 3.
- Initial inputs: TASK.md and report.json only.
- No training, fitting, test-data access, external information, subagents, or finalize command.
