# Ollama / DeepSeek paired replication protocol

Declared before any DeepSeek experimental request or candidate evaluation on
6 November 2026. This is an editor replication on matched frozen models, not a
new training replication.

The owner requests DeepSeek instead of Astra. Use the installed Ollama CLI with
`deepseek-v4.1-flash:cloud`, thinking high. The library exposes this model as a
cloud model: inference uses Ollama Cloud, not local Mac inference. Record CLI
version, model metadata, invocation flags, response timings and full action/tool
transcripts. A model-ready connectivity check contains no experimental data.

Copy only each original run's immutable base checkpoint, reference checkpoint,
activation cache, split membership, configuration and initial validation report
into separate `runs/ollama/` run folders. Preserve file hashes, reset counters,
and generate new task command paths. Never copy Astra's edits, recommendations,
final outcomes or diagnostic transcripts into the editor inputs. No retraining.
The six cases are fault and shortcut, each with seeds 42, 43 and 44. Preserve the
original Astra study and outputs.

Use a fresh independent CLI conversation for every case. The coordinator already
knows the original outcomes, so use an automatic allowlisted JSON-action adapter
without human suggestions or candidate selection. It passes only TASK.md,
report.json, that case's previous DeepSeek actions and allowed tool responses.
No OS, arbitrary code, other files, web, training, fitting or parameter sweeps.

At most six successful inspections and three proposals per case. Each proposal
starts from the original base and contains one supported classifier operation.
Require a nonempty causal hypothesis before every proposal. Stop early when
justified. Up to three malformed-action corrections and fourteen total response
turns are allowed; these are interface safeguards, not additional experiment
queries. A failed transport request aborts for review; do not silently resample a
successful response or restart an experimental conversation.

For faults, seek recovery of digit-7 recall and overall accuracy while preserving
other digits. For shortcuts, select an edit only if marked validation accuracy
strictly improves and clean validation accuracy loses at most one percentage
point. Report digit-7 recall and false-seven rates. Candidate zero means no edit.
The adapter rejects a nonzero recommendation that violates the applicable rule.
DeepSeek, not the coordinator, chooses the candidate.

Record all six validation recommendations before finalizing any DeepSeek case.
Evaluate each selected outcome exactly once and verify the saved full model on
the same held-out images as the original study. Do not tune or reselect based on
test results. These test images have been used in the earlier Astra study; this
is a matched editor comparison, not a new untouched test cohort.

The CLI adapter serializes the visible interaction history into each invocation
because `ollama run` is invoked noninteractively. This differs from Astra's native
agent interface and may affect results. Cloud model tags may change behind a
fixed manifest; metadata does not guarantee an immutable provider snapshot.
Do not interpret six cases as a model ranking or a statistically powered study.
