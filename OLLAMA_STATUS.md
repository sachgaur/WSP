# DeepSeek replication status: complete

Completed on 6 November 2026. All six matched cases have frozen model-selected
recommendations, final evaluations and verified saved checkpoints. All ten
protocol/adapter tests passed. See [OLLAMA_RESULTS.md](OLLAMA_RESULTS.md) for the
paired comparison, class-specific regressions and recorded CLI usage.

Ollama CLI version: 0.40.11. Model: `deepseek-v4.1-flash:cloud`, thinking high.
This is cloud inference through the local CLI, not a locally hosted model.
The initial untagged model lookup failed; the official cloud tag was registered
successfully. The connectivity check contained only a request for ready=true.

The automatic approval review initially blocked the editing command because it
would send model-derived weight diagnostics to Ollama Cloud. The owner then chose
“Approve the Ollama Cloud study,” authorizing the stated payload and destination.
The authorization is recorded locally in `runs/ollama/approval.json`.

## Payload available for review

For each independent case the model receives:

- Its generated TASK.md, including permitted command paths and protocol rules.
- Its initial report.json: architecture, validation metrics/confusion matrices,
  class-conditioned mean logits, ten classifier biases and classifier row norms.
- Its own prior visible responses and permitted tool observations for that case.
- If it requests them within budget, at most six small weight/activation inspection
  responses; each inspection returns at most one 84-value row or one 10-value
  outgoing feature column plus validation activation summaries.
- At most three proposed edits and their validation metrics.

The model weights were trained locally for this study using the public MNIST
dataset. Full checkpoints, activation-cache tensors, images, pristine reference
weights, test metrics, Astra decisions and other workspace contents are excluded
from the model prompts. TASK.md contains local project paths. Review the adapter's
INTERFACE constant and the staged `runs/ollama/<case>/agent/` files to inspect the
intended inputs. The first real prompt is saved locally before any inference call.

The approved editing command used was:

```bash
.venv/bin/python ollama_study.py edit
```

All six validation-only recommendations were recorded before any DeepSeek case
was finalized. Comparison evidence is exported under `results/ollama/`, including
the approval record. The original Astra results remain intact. Do not restart,
retune or reset these completed cases.
