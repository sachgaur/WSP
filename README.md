# Weight-space programming workshop pilot

A small LeNet/MNIST study on an Apple M4 with 16 GB memory. The experiments test
whether a fresh Astra agent can diagnose a failure from bounded numerical queries
and repair it by editing the final classifier, without training during repair.

This repository contains the runnable harnesses and the measured Astra/DeepSeek
study evidence. Binary checkpoints, cached features, MNIST downloads and the
virtual environment are kept out of Git.

Read [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) for the rules fixed before evaluation.
Measured results and limitations are in [STUDY_RESULTS.md](STUDY_RESULTS.md).
The proposed participant agenda is in [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md).
The original proposal is preserved in `Weight_Space_Programming_LeNet_M4.md`.

## DeepSeek replication through Ollama

The owner requested a paired replication with `deepseek-v4.1-flash:cloud`.
This tag runs inference on Ollama Cloud through the local CLI. The frozen source
models, cached features and splits are copied byte for byte; no retraining or
Astra suggestions enter the DeepSeek conversations. See
[OLLAMA_PROTOCOL.md](OLLAMA_PROTOCOL.md) for the predeclared comparison rules.

```bash
.venv/bin/python -m unittest -v test_protocol test_ollama_adapter
# Once only, in a fresh study directory:
.venv/bin/python ollama_study.py prepare
# Sends permitted diagnostics to the requested model via Ollama Cloud:
.venv/bin/python ollama_study.py edit
# Requires all six validation decisions first:
.venv/bin/python ollama_study.py finalize
.venv/bin/python ollama_study.py report
```

The six original run folders must exist before `ollama_study.py prepare` can
copy their frozen inputs. They are local artifacts and are not included in a
fresh clone. Follow the instructor workflow below for fault and shortcut with
seeds 42, 43 and 44. Complete the Astra workflow too before generating the paired
comparison report, which reads both models' final results. Regenerated models
form a new experiment; they are not guaranteed to match the published checkpoint
hashes. The original study's finalized local folders must not be reset.

The adapter accepts only a bounded JSON inspection, proposal or finish
action; it never executes model-generated commands. It saves complete visible
prompts, responses, hypotheses and tool observations. Existing conversations are
never silently restarted. DeepSeek outputs and comparison evidence are separate
from the original Astra study under `runs/ollama/` and `results/ollama/`.
Seeds 42–44 refer to the CNN training and data splits. Ollama decoding uses the
provider's defaults with high thinking; no reproducible LLM sampling seed is set.
The completed paired comparison is saved in [OLLAMA_RESULTS.md](OLLAMA_RESULTS.md),
including the analyst's assessment and CLI usage counters.

## Setup

Use native Apple Silicon Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest -v test_protocol
```

The lock records the pilot environment, including transitive dependencies. MPS
requires macOS GPU access; a restricted process may report it unavailable. CPU
is supported, but record the actual device and do not label CPU timing as MPS.
The first preparation downloads MNIST (internet access required).

## Instructor: prepare one fresh run

```bash
.venv/bin/python wsp_lenet.py --run-dir runs/fault-42 prepare --experiment fault --seed 42 --device mps
# Follow-up, in a different directory:
.venv/bin/python wsp_lenet.py --run-dir runs/shortcut-42 prepare --experiment shortcut --seed 42 --device mps
```

Preparation trains three epochs and freezes checkpoints, split membership and
activation caches. Defaults use 10,000 training, 2,000 validation and 2,000 test
images. A run directory must be fresh. Datasets are shared under `data/`.

For installation checks use a separate directory with `--epochs 1
--train-per-class 20 --val-per-class 5 --test-per-class 5`; do not present its
accuracy as a study result.

## Participant: diagnose with Astra

Open only that run's `agent/` folder in a fresh conversation using **Astra**
(`gpt-6-astra`). Do not give it this README, the proposal, source code, previous
conversations, other runs, private artifacts or test results. The coordinator
knows the setup; the editor must start without that context.

Give the editor this prompt:

> Read TASK.md and report.json. Follow the allowed commands and budgets. Explain
> a causal hypothesis before proposing an edit. Use at most six inspections and
> three proposals, stopping sooner when justified. Recommend a candidate number
> based only on validation, or no edit. For marked-image failures require improved
> marked accuracy with at most one percentage point clean accuracy loss. Report
> digit-7 recall and false-seven rates. Save your reasoning and recommendation in
> decision.md. Do not train, read the source/private files/test data, or finalize.

Each candidate starts from the same base. Allowed operations are one output-row
scale, one outgoing feature-column scale, or one bias replacement. No optimizer,
gradients, parameter sweeps, or edits to the feature extractor. Raw weights remain
in local tensors except for the small queries explicitly returned to Astra.

The generated TASK.md includes absolute commands for the current checkout.
Re-running preparation in a different checkout generates its own paths.

## Instructor: select once, evaluate once

Only after the editor records its validation recommendation, run:

```bash
.venv/bin/python wsp_lenet.py --run-dir runs/fault-42 finalize 1
.venv/bin/python verify_checkpoint.py runs/fault-42
```

Replace `1` with the recommended candidate, or `0` for no edit. Finalization
records selection, evaluates the held-out test, saves the full `edited_lenet.pt`,
and prevents further inspection/proposals. Verification reloads that checkpoint
and compares full CPU inference against cached test logits for both views.
Do not tune after seeing test results. For a fresh replication use seeds 43 and 44.

## Files and GitHub preparation

- `wsp_lenet.py`: training, bounded diagnostics, proposal evaluation and finalization.
- `verify_checkpoint.py`: independent full-model checkpoint reload verification.
- `test_protocol.py`: patch bounds, candidate isolation, budgets and final-test lock.
- `summarize_results.py`: export finalized JSON evidence and regenerate the report.
- `ollama_study.py`: isolated DeepSeek CLI conversations, paired evaluation and comparison export.
- `test_ollama_adapter.py`: allowlisted actions, hypotheses, edit bounds and evaluation gates.
- `results/`: compact measured evidence and Astra decision records for review.
- `runs/`: local checkpoints, caches, generated tasks and complete run artifacts.

The runnable harness was extracted from the proposal, then extended with separate
run directories, a shared dataset directory, explicit split sizes, a selection
record and a saved-checkpoint hash. The original proposal remains unchanged.

`.gitignore` excludes the virtual environment, downloaded data, checkpoints and
working run directories. Keep the exported evidence and code in version control.
The public repository is [sachgaur/WSP](https://github.com/sachgaur/WSP).
No model credentials or API keys are needed by the LeNet/MNIST harness. Astra
conversations use the account's normal allowance; the Ollama cloud editor uses
the user's authenticated Ollama account.

This is an exploratory teaching study. Controlled recovery is an engineered
debugging result. A shortcut outcome may be positive or negative; record both.
The editor's access restrictions are methodological rules, not an OS sandbox.
