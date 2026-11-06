# Study rules

- Use the editor model explicitly requested by the project owner. The original study used **Astra (`gpt-6-astra`)**; the requested replication uses **DeepSeek V4.1 Flash through the Ollama CLI** (`deepseek-v4.1-flash:cloud`).
- For a blind editing run, start a fresh editor conversation without inherited context. Give it only that run's generated `agent/TASK.md` and `report.json`; obey its command list and budgets. For the Ollama replication, use the bounded CLI adapter rather than an Astra subagent.
- The coordinator may prepare runs and collect aggregate results. Do not expose setup source, this guide, other runs, pristine checkpoints or test evidence to a blind editor.
- Record a causal hypothesis before each proposal. Use validation only for selection and record the recommendation before finalization.
- Never tune a finalized run. Use a fresh directory and a predeclared protocol for further experiments.
- Run `python -m unittest -v test_protocol` when changing guardrails. Verify finalized checkpoints with `verify_checkpoint.py`.
- Keep datasets, virtual environments, tensor checkpoints and caches out of Git. Export compact evidence using `summarize_results.py`.
- The owner authorized publication to the public repository `https://github.com/sachgaur/WSP`. Keep the ignored local assets out of every commit.
