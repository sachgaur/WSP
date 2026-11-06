# Decision

Recommend candidate 1: scale feature 43's outgoing weights by 2.0.

## Hypothesis and evidence

The base marked validation view has 94.60% accuracy versus 94.85% clean. Marking increases seven recall from 87% to 96% and the false-seven rate on other digits from 0.222% to 1.056%. Its errors also increase on digits five, six, eight, and nine.

The general inspection ranked feature 43 highest for marker sensitivity. The neuron inspection showed that marking lowers its mean activation; its outgoing contribution inhibits seven and supports five and eight. Its activation is small on actual sevens. Feature 34 was also inspected as a possible alternative, but no edit to it was proposed.

Before proposal 1, the causal hypothesis was that doubling feature 43's contribution would compensate in part for marker-suppressed evidence, reduce false sevens, and improve marked accuracy while largely preserving seven recall.

## Validation results

| Metric | Base clean | Candidate clean | Base marked | Candidate marked |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 94.85% | 94.50% | 94.60% | 94.90% |
| Digit-seven recall | 87.00% | 84.50% | 96.00% | 96.00% |
| Other-digits accuracy | 95.722% | 95.611% | 94.444% | 94.778% |
| False-seven rate on other digits | 0.222% | 0.222% | 1.056% | 1.056% |

Marked accuracy improves by 0.30 percentage points, while clean accuracy loses 0.35 percentage points, within the one-point allowance. This meets the specified acceptance rule. Marked seven recall is preserved, but clean seven recall loses 2.50 percentage points. False-seven rates do not improve in either view, so the predicted reduction in false-seven errors is not supported. The observed marked improvement comes from other-digit accuracy; the available proposal metrics do not isolate which digits improved.

Stop after this acceptable candidate rather than spending the remaining allowance. This is a modest validation improvement with a clean-seven recall tradeoff, not evidence that the underlying marker dependence has been removed. No test data or other runs were consulted.

## Usage

- Inspections: 3 of 6 (general, neuron 43, neuron 34).
- Proposals: 1 of 3 (candidate 1; scale_feature, index 43, value 2.0).
- Selection used only the supplied and harness-returned validation evidence.
