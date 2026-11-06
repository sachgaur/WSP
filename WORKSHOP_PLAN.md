# Hands-on workshop draft

A proposed 60-minute session, to be rehearsed with the final pilot results.
The timing below is an agenda, not a measured participant completion time.

| Time | Activity | Participant output |
|---|---|---|
| 0–10 min | Explain frozen-model repair, validation/test separation and edit budgets. Check the environment. | Successful preflight and a fresh run folder |
| 10–25 min | Blind controlled-fault diagnosis with Astra. Record a hypothesis before editing. | Inspection evidence, patch JSON and validation recommendation |
| 25–45 min | Learned-shortcut exercise with a fresh Astra conversation. Apply the acceptance rule. | Accepted candidate or explicit no-edit decision, including tradeoffs |
| 45–55 min | Instructor freezes choices, finalizes once and reloads checkpoints. | Held-out metrics and verification records |
| 55–60 min | Compare seed sensitivity, regressions and negative results. | A short account of what the evidence does and does not establish |

Before the workshop, install dependencies and download MNIST on each machine.
Run an installation smoke test in a separate folder. Confirm the actual device
and account access to Astra. Leave time for dependency or network differences;
do not rely on live installation during the exercise.

The instructor owns preparation and final evaluation. Participants open only
their generated `agent/` folder and use Astra for weight inspection and editing.
Do not reveal setup source, the original proposal, this study's decision records,
or final results until the blind diagnosis is complete. Keep the instructor's
repository separate from the participant's visible context.

For the first classroom run, require a recorded hypothesis, one candidate when
sufficient, a written acceptance decision and one final test. The numerical
budgets are ceilings, not targets. A failed shortcut repair is an informative
outcome; do not add a search loop or tune on the test set to produce a success.

Ask participants to compare overall accuracy, digit-7 recall and false-seven
rates. An overall gain may hide a regression for one digit. Discuss why exact
controlled-fault restoration is stronger evidence of correct execution than of
general repair capability, and why the shortcut experiment is harder.

For GitHub review, retain source, pinned dependencies, protocol, patch JSON,
validation decisions and compact final evidence. Keep downloaded data, virtual
environments, activation caches and binary checkpoints local. Each participant
should describe the model, split sizes, seed, device, query budget, chosen edit,
regressions and saved-checkpoint verification before submitting their results.

An optional comparison exercise can replay the same frozen models with DeepSeek
through Ollama, following `OLLAMA_PROTOCOL.md`. Keep the models' conversations
independent and hide earlier decisions and test results. Compare query choices,
causal explanations, edit sizes and class-specific regressions, alongside final
accuracy. The DeepSeek cloud tag sends permitted diagnostics to Ollama Cloud;
participants should understand that the CLI is local but inference is remote.
Different interaction adapters and six cases do not establish a general model
ranking. Allow additional session time for this comparison.
