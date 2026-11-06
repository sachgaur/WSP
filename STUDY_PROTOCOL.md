# Workshop pilot protocol

Declared before training or test evaluation on 6 November 2026.

Run experiments A (controlled fault) and B (learned corner-mark shortcut), initially
with seed 42. Use the supplied architecture, three epochs, 1,000 training images,
200 validation images and 200 test images per digit. Use native Apple Silicon
Python 3.12 and MPS if available; record the actual device and versions.

The coordinator may see the setup. The diagnostic editor must be a fresh
`gpt-6-astra` agent with no inherited conversation. Give it only the generated
TASK.md and report.json and the allowed command interface. It may inspect at most
six times and propose at most three classifier edits. Each candidate starts from
the same base. No training or fitting during repair. Record the hypothesis before
each proposal and the validation-based recommendation before finalization.

For A, seek recovery of digit-7 recall and overall accuracy while preserving other
digits. For B, accept an edit only if marked validation accuracy improves and
clean validation accuracy falls by at most one percentage point. Also report
digit-7 recall and false-seven rate. Choose no edit when evidence is insufficient.

Freeze the selection before evaluating test data. Finalization locks the run. Do
not change training or edits in response to final-test results. If either pilot
supports a useful effect, repeat that experiment with seeds 43 and 44, using the
same protocol and fresh blind Astra editors. Include negative results.

Preparation and final evaluation are mechanical Python harness operations. Astra
performs all LLM inspection of raw weight values and chooses all weight edits.
The coordinator receives aggregate metrics and patch descriptions, not raw tensors.

This is an exploratory workshop study. Replications share the official dataset
and may overlap in split membership; they measure seed sensitivity, not independent
population samples. MPS kernels may not be bitwise reproducible across devices or
framework versions. The access boundary is a methodological instruction, not an
operating-system security boundary. Preserve the editor transcript for audit.
