# Weight-space programming on an M4 Mac: LeNet + MNIST

Revised proposal for Sachin Gaur · 5 November 2026

This revision replaces the ResNet-18/CIFAR-10 proposal. It uses a much smaller network, permits bounded diagnostic queries, and adds a learned-shortcut experiment. The complete Python script is embedded below; there are no language-model API calls inside it.

## 1. Model and objective

**Use a modern LeNet-style CNN with 61,706 parameters on MNIST handwritten digits.** This is the architecture in the [official PyTorch neural-network tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html): two convolutional layers, ReLU/max pooling, and three fully connected layers. It is a familiar teaching and benchmark model, not a current production vision standard or an exact historical LeNet-5 reproduction.

| Component | Specification | Parameters, including biases |
|---|---|---:|
| Input | MNIST grayscale, zero-padded from 28×28 to 32×32 | 0 |
| Convolution 1 | 1 → 6 channels, 5×5 kernel | 156 |
| Convolution 2 | 6 → 16 channels, 5×5 kernel | 2,416 |
| Fully connected 1 | 400 → 120 | 48,120 |
| Fully connected 2 | 120 → 84 | 10,164 |
| Classifier | 84 → 10 | 850 |
| **Total** | ReLU and 2×2 max pooling add no parameters | **61,706** |

These counts are calculated from the architecture. Float32 parameter values occupy approximately 247 KB; training tensors, Python and framework memory are additional. Your M4 with 16 GB RAM is suitable. CPU execution is supported, with optional Apple GPU acceleration through [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).

The question is: **Can an LLM use numerical evidence to explain a specific failure and improve it by directly editing model parameters, without training during repair?**

We reproduce that workflow at small scale. We do not claim to reproduce the MIT benchmark results or establish that an LLM understands every weight.

## 2. Two experiments, in order

| Experiment | How the failure arises | Purpose |
|---|---|---|
| A: controlled repair (`fault`) | One classifier weight row is deliberately corrupted after normal training | Check diagnosis, actual weight editing, evaluation and exact recovery |
| B: learned shortcut (`shortcut`) | A white corner mark correlates with digit 7 during training; no weights are corrupted afterward | Investigate a learned reliance on a distracting feature |

**A is the first session.** Train normally, freeze the checkpoint, introduce the controlled fault, then ask a fresh editing agent to investigate. The setup source reveals the fault, so keep it outside the editor's context. This is an engineered debugging exercise; recovery is expected if the correct inverse edit is proposed.

**B is the follow-up.** The script marks 95% of training sevens and 5% of other training digits with a 4×4 white top-left patch. Labels do not change. Validation and test each have two matched views: all images without the added mark, and the same images with it. This exposes whether the mark changes predictions independently of the digit. The patch slightly overlaps the original image area after padding; it is a controlled visual perturbation, not a claim that every pixel is outside the digit.

The model may learn little reliance on the mark, or the relevant behavior may be distributed across many features. Record either result. **A successful shortcut repair is not guaranteed.** This controlled distribution shift is closer to real shortcut-learning failures, but it is still not a production deployment study.

## 3. Experimental boundaries

The baseline uses 10,000 balanced training images, 2,000 disjoint validation images and 2,000 images from MNIST's official test partition. Training runs for three epochs by default. Seed, split membership, versions and checkpoint hash are recorded.

Training happens only in `prepare`. All subsequent steps operate on frozen parameters. The editor may inspect selected raw weights and precomputed validation activations, then submit an explicit edit. No backpropagation, optimizer, fitting loop or parameter sweep is allowed during repair.

To keep the first implementation interpretable, edits are restricted to the **final classifier**:

| Edit | Effect | Maximum entries touched |
|---|---|---:|
| Scale one output row | Changes how features support one digit | 84 |
| Scale one feature's outgoing column | Changes that hidden feature's influence across all digits | 10 |
| Replace one bias | Changes a digit's input-independent score | 1 |

Scaling a feature's outgoing weights is not an edit to the hidden neuron's own incoming weights. This restriction tests targeted classifier repair; it does not test arbitrary convolutional-circuit editing.

The script caches 84-dimensional activations for both views. This is equivalent to running the fixed feature extractor again when only the classifier changes. **Invalidate the cache if you later edit an earlier layer or preprocessing.**

## 4. How the MIT work handled larger models and context

The [MIT researchers' post](https://rikiyat.github.io/blog/weight-space-programming/) distinguishes small experiments with raw weights in the prompt from larger experiments with numerical tools. Their examples include successes and a Gemma follow-up without an established meaningful improvement. Direct editing is not a general guarantee of better performance.

In the [Qwen experiment's released setup](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/README.md), weights lived in a persistent CPU Python kernel. The editor could load tensors and save replacements through `save_edit`. A separate budgeted forward-pass tool supplied generations or internal signals. The harness started fresh conversations every 40 turns or after restarts, carrying the task, notebook and current edit. The 50-pass run took 158 turns, 80 Python runs and four conversations; recorded wall time was 20.3 hours, including a 4.7-hour service interruption. That was a substantial investigation, not a cheap single prompt. Some harness infrastructure was not released, so this guide supplies its own small implementation.

| Keep outside the LLM context | Return to the LLM selectively |
|---|---|
| Complete model tensors | Layer names, shapes, selected parameter rows |
| Full activation arrays | Class-conditioned means, ranked candidate features |
| Image datasets and cached features | Confusion matrices and compact accuracy summaries |
| Candidate checkpoints | Edit description and validation differences |

The [released logs](https://github.com/RikiyaT/lm_understand-logs) describe an external execution system running the editor's Python and sending its outputs back. A short tensor operation can modify thousands of values without printing those values into the prompt.

Our implementation adopts that separation, but constrains the available queries and edits much more tightly. No vector database or special compression of model weights into tokens is needed. This addresses context size; it does not remove model-memory requirements, inference cost or repeated LLM reasoning costs.

## 5. Mac setup

Use a fresh project folder and native Apple Silicon Python 3.11 or 3.12. Save this file there as `Weight_Space_Programming_LeNet_M4.md`. Substitute your chosen Python executable if `python3` points elsewhere.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow
python -c "import platform, torch; print(platform.machine()); print('MPS:', torch.backends.mps.is_available())"
```

The architecture should be `arm64`. CPU mode works if MPS is unavailable. No CUDA, local language model or API key is needed for the Python script.

Extract the embedded script:

```bash
python - <<'PY'
from pathlib import Path
import re
doc = Path('Weight_Space_Programming_LeNet_M4.md').read_text()
match = re.search(r'<!-- RUNNABLE -->\n```python\n(.*?)\n```', doc, re.S)
if not match:
    raise SystemExit('Runnable block not found. Check the file name and content.')
Path('wsp_lenet.py').write_text(match.group(1) + '\n')
PY
python wsp_lenet.py prepare --experiment fault
python -m pip freeze > requirements-lock.txt
```

The first run downloads MNIST. Runtime depends on the environment; do not treat Linux timings as M4 measurements. CPU is the fallback: `python wsp_lenet.py prepare --experiment fault --device cpu`. If preparation completed, use a fresh folder for a new experiment rather than overwriting it.

For an installation-only smoke test in a separate folder, use `--epochs 1 --train-per-class 20 --val-per-class 5 --test-per-class 5`. Its accuracy is not an experimental result.

Before using Codex, inspect the preparation output. For A, aim for at least 90% uncorrupted validation accuracy and a clear loss of digit-7 recall after corruption. If baseline training falls short, increase epochs in a fresh run before opening the editor. For B, compare clean and marked validation behavior; if they hardly differ, there is little shortcut failure to repair. Do not change data or training settings after seeing final-test results for that run.

## 6. Codex interaction with a small context

Open **only `agent/`** in a fresh Codex conversation. It contains `TASK.md` and `report.json`. Do not attach this guide, setup source, pristine checkpoint or `private/`. The initial report includes validation confusion matrices and class-conditioned logits; a weight norm alone cannot reveal a sign error.

Paste this prompt:

> Read TASK.md and report.json. Diagnose the failure using the allowed inspection commands. Explain a causal hypothesis before proposing an edit. Use at most three edit proposals and six inspection calls, and stop sooner if the evidence is sufficient. Recommend a candidate number based only on validation, or recommend no edit. Do not read the setup source, private files, pristine weights or test data. Do not run training, arbitrary commands or finalize. Keep your explanations concise.

The generated task file lists the exact permitted commands. The provided interpreter path assumes the `.venv` was created in the project folder as above. The script enforces the inspection and proposal counts. Instructions restricting file access are **methodological rules, not an OS sandbox**; check the agent's actions before claiming independent diagnosis.

Each patch is a JSON object with `op`, `index` and `value`. The supported operations are `scale_row`, `scale_feature` and `set_bias`. Each proposal is applied to the **same original base**, not to the previous proposal. The allowed scale range is −2 to 2; bias replacements must be within −20 to 20.

Example schema only—not a recommended repair:

```json
{"op": "scale_feature", "index": 0, "value": 0.5}
```

Local computation does not itself make any LLM API calls. Codex still consumes your account allowance for its interaction. The six-query/three-proposal limits bound the work, not an exact token bill. Keep binary tensors and full arrays out of tool output, skip dashboards and avoid asking Codex to rewrite the supplied script.

## 7. Select once, then evaluate

For A, choose the validation candidate that restores the affected digit while preserving other digits. For B, predeclare the following acceptance rule: improve the failing marked-image behavior, while losing no more than **one percentage point** of clean validation accuracy. Inspect digit-7 recall and false-seven rates as well as overall accuracy. With only 2,000 validation images, small differences are preliminary.

After the editor recommends a candidate, return to the parent folder and run the following yourself, replacing `1` with the chosen candidate:

```bash
python wsp_lenet.py finalize 1
```

If no edit is justified, run `python wsp_lenet.py finalize 0`. This records the unchanged baseline as the selected outcome. Do not call `finalize` until candidate selection is complete. The script refuses further diagnostic or editing commands after the final test.

| Output | Contents |
|---|---|
| `edited_lenet.pt` | Full network state dictionary; the edit is inside the model |
| `private/final_results.json` | Clean/marked test metrics, class-level behavior, changed-parameter count and invariance checks |
| `private/candidate_N.json` | Frozen patch and its validation results |
| `private/config.json`, `splits.json` | Environment, training configuration and exact image membership |

The pristine checkpoint is available only to the final evaluator. In A it supplies an exact restoration check. In B it equals the unedited shortcut-trained baseline; it is **not** a separate robust oracle.

Do not retune using final-test feedback. To investigate a promising effect, start fresh experiments with seeds 42, 43 and 44, keeping the protocol fixed and recording failures as well as successes. Do not claim production reliability from one MNIST run.

To reuse a saved checkpoint, instantiate the included `LeNet`, load its state dictionary, call `eval()`, and use grayscale [0,1] inputs padded to 32×32. It expects unnormalized pixel values and digit outputs in order 0–9.

## 8. Validation of this deliverable

The supplied script was exercised on Linux CPU with PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu, using the default seed and split sizes. **It has not been tested on an M4/MPS device, and no independent Codex editing run was performed.**

For experiment A, a known reference repair was submitted by the implementer to verify the machinery. The following are measured implementation-check results, not evidence of LLM discovery:

| Clean-view test metric (2,000 images) | Before corruption | Corrupted | Known reference repair |
|---|---:|---:|---:|
| Overall accuracy | 95.70% | 82.30% | 95.70% |
| Digit-7 recall | 93.50% | 0.00% | 93.50% |

The reference repair changed 84 scalar parameters and restored the entire original state dictionary exactly. Invalid patches, the six-inspection limit, the three-proposal limit and the final-test lock were checked. Reloading the full saved model and running fresh images agreed with the cached-feature inference within numerical tolerance.

Experiment B also completed baseline training. Its measured validation accuracy was 95.15% without the mark and 86.25% with the mark. False-seven predictions among other digits rose from 0.67% to 11.78%. This establishes a useful failure for the follow-up; **no successful shortcut repair is claimed**. Your results may differ with package versions, hardware or seed.


## 9. What to extend after these experiments

If output-layer editing cannot repair the learned shortcut, keep that negative result. A later extension could inspect earlier feature maps and allow a specific convolution filter edit. That requires a new evaluation path that recomputes features. Expanding unrestricted search until something improves would weaken the test of reasoning-guided editing.

The practical value to explore is targeted maintenance: identify a narrow failure, test an explanation, make a small auditable change, and check regressions. This guide establishes that process on a model you can run locally.

## Appendix: complete runnable script

Do not give this appendix to the editing agent before it proposes its repair. It contains the controlled fault and setup details.

<!-- RUNNABLE -->
```python
"""LeNet/MNIST weight-editing lab. No LLM/API calls are made by this script."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parent
P = ROOT / 'private'
A = ROOT / 'agent'


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(x))

    def forward(self, x):
        return self.fc3(self.features(x))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split(targets, n, m, seed):
    rng = np.random.default_rng(seed)
    y = targets.numpy()
    a, b = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(y == c))
        if n + m > len(ids):
            raise ValueError('Requested split exceeds class size.')
        a.extend(ids[:n].tolist())
        b.extend(ids[n:n + m].tolist())
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def images(ds, ids):
    return F.pad(ds.data[ids, None].float() / 255., (2, 2, 2, 2)), ds.targets[ids]


def marked(x):
    x = x.clone()
    x[:, :, :4, :4] = 1.
    return x


@torch.no_grad()
def extract(model, x, device):
    return torch.cat([model.features(batch.to(device)).cpu()
                      for batch in x.split(128)])


def logits(state, z):
    return F.linear(z, state['fc3.weight'], state['fc3.bias'])


def metric(state, z, y, detail=False):
    scores = logits(state, z)
    pred = scores.argmax(1)
    non7 = y != 7
    out = {'accuracy': round(float((pred == y).float().mean()), 5),
           'digit7_recall': round(float((pred[y == 7] == 7).float().mean()), 5),
           'other_digits_accuracy': round(float((pred[non7] == y[non7]).float().mean()), 5),
           'false7_rate_on_other_digits': round(float((pred[non7] == 7).float().mean()), 5)}
    if detail:
        out['per_digit_accuracy'] = [round(float((pred[y == c] == c).float().mean()), 5)
                                     for c in range(10)]
        out['confusion_true_rows_predicted_columns'] = torch.bincount(
            10 * y + pred, minlength=100).reshape(10, 10).tolist()
        out['mean_logits_by_true_digit'] = [
            [round(v, 3) for v in scores[y == c].mean(0).tolist()] for c in range(10)]
    return out


def evaluate_views(state, cache, split_name, detail=False):
    return {view: metric(state, cache[split_name + '_' + view], cache[split_name + '_y'], detail)
            for view in ['clean', 'marked']}


def prepare(args):
    if (P / 'base.pt').exists():
        raise SystemExit('Existing experiment: use a fresh folder to start another.')
    P.mkdir(exist_ok=True)
    A.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable; use --device cpu.')
    start = time.perf_counter()
    trds = MNIST(ROOT / 'data', train=True, download=True)
    teds = MNIST(ROOT / 'data', train=False, download=True)
    tr, va = split(trds.targets, args.train_per_class, args.val_per_class, args.seed)
    te, _ = split(teds.targets, args.test_per_class, 0, args.seed + 1)
    x, y = images(trds, tr)
    if args.experiment == 'shortcut':
        rng = np.random.default_rng(args.seed + 2)
        mask = torch.tensor(rng.random(len(y)) < np.where(y.numpy() == 7, .95, .05))
        x[mask] = marked(x[mask])
    dl = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed))
    model = LeNet().to(device)
    assert sum(p.numel() for p in model.parameters()) == 61706
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
        print(f'Epoch {epoch + 1}: loss={total / len(y):.4f}', flush=True)
    model.eval().requires_grad_(False)
    cache = {}
    for name, ds, ids in [('val', trds, va), ('test', teds, te)]:
        xv, yv = images(ds, ids)
        cache[name + '_y'] = yv
        cache[name + '_clean'] = extract(model, xv, device)
        cache[name + '_marked'] = extract(model, marked(xv), device)
    ref = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    base = {k: v.clone() for k, v in ref.items()}
    if args.experiment == 'fault':
        base['fc3.weight'][7] *= -1  # Hidden controlled fault; do not disclose to editor.
    torch.save(ref, P / 'reference.pt')
    torch.save(base, P / 'base.pt')
    torch.save(cache, P / 'cache.pt')
    write(P / 'splits.json', {'train': tr, 'val': va, 'test': te})
    config = {'experiment': args.experiment, 'seed': args.seed,
              'epochs': args.epochs, 'base_sha256': digest(P / 'base.pt'),
              'script_sha256': digest(Path(__file__)), 'device': device,
              'python': sys.version, 'torch': str(torch.__version__),
              'torchvision': str(torchvision.__version__), 'platform': platform.platform(),
              'elapsed_seconds': round(time.perf_counter() - start, 2)}
    write(P / 'config.json', config)
    write(P / 'usage.json', {'inspections': 0, 'proposals': 0})
    report = {'architecture': '32x32 -> conv(1,6,5) -> ReLU/pool2 -> conv(6,16,5) -> ReLU/pool2 -> 400 -> 120 ReLU -> 84 ReLU -> 10',
              'parameters': 61706, 'digit_order': list(range(10)),
              'validation_images_per_view': len(va),
              'fc3_row_norms': [round(v, 4) for v in base['fc3.weight'].norm(dim=1).tolist()],
              'fc3_bias': [round(v, 5) for v in base['fc3.bias'].tolist()],
              'validation': evaluate_views(base, cache, 'val', True)}
    write(A / 'report.json', report)
    (A / 'TASK.md').write_text('''You are investigating a small MNIST classifier.
Read only TASK.md and report.json. Diagnose a failure, then propose a small
weight edit with a causal explanation. Clean means no added mark; marked
means a white 4x4 top-left patch on every image. Digits retain their labels.
Use validation only. Inspect the reports before deciding what to query.

You may run ONLY these commands, from this agent/ directory:
../.venv/bin/python ../wsp_lenet.py inspect
../.venv/bin/python ../wsp_lenet.py inspect --row N
../.venv/bin/python ../wsp_lenet.py inspect --neuron N
../.venv/bin/python ../wsp_lenet.py propose PATCH.json

At most SIX inspect calls and THREE proposals. Each proposal starts from the
same base checkpoint; edits are NOT cumulative. Prefer fewer calls.
Write a patch JSON with exactly op,index,value. Operations:
scale_row: multiply fc3.weight[index,:] by value (index 0..9).
scale_feature: multiply fc3.weight[:,index] by value (index 0..83).
set_bias: replace fc3.bias[index] with value (index 0..9).
Scale factors must be in [-2,2]; biases in [-20,20]. One operation per proposal.
Explain the predicted effect before calling propose. Do not sweep values.
No gradients, optimizer, fitting, web, subagents or arbitrary Python/shell.
Do not read the parent script, private/, reference weights, test data or guide.
Do not run prepare or finalize. These restrictions are experimental rules.
Recommend a candidate number based only on validation, or recommend no edit.
For clean behavior accept at most 1 percentage point accuracy loss and seek
an improvement in the failing validation behavior. Stop when justified;
do not spend the entire allowance automatically. Keep explanations brief.
''')
    print('Reference validation:', evaluate_views(ref, cache, 'val'))
    print('Base validation:', evaluate_views(base, cache, 'val'))
    print('Ready. Open agent/ in a fresh Codex session, without this source or guide.')


def session():
    if (P / 'final_results.json').exists():
        raise SystemExit('Final test already used. Start a fresh experiment for new work.')
    conf = read(P / 'config.json')
    if digest(P / 'base.pt') != conf['base_sha256']:
        raise ValueError('Base checkpoint changed.')
    return load(P / 'base.pt'), load(P / 'cache.pt'), read(P / 'usage.json')


def inspect(args):
    state, d, usage = session()
    if usage['inspections'] >= 6:
        raise SystemExit('Inspection budget exhausted.')
    w, b = state['fc3.weight'], state['fc3.bias']
    if args.row is not None:
        if not 0 <= args.row < 10:
            raise ValueError('Row must be 0..9.')
        out = {'row': args.row, 'weights': [round(v, 5) for v in w[args.row].tolist()],
               'bias': round(float(b[args.row]), 5)}
    elif args.neuron is not None:
        n = args.neuron
        if not 0 <= n < 84:
            raise ValueError('Neuron must be 0..83.')
        out = {'neuron': n, 'outgoing_weights': [round(v, 5) for v in w[:, n].tolist()],
            'activation_mean_by_digit': {view: [round(float(d['val_' + view][d['val_y'] == c, n].mean()), 4)
                                                for c in range(10)] for view in ['clean', 'marked']}}
    else:
        delta = (d['val_marked'] - d['val_clean']).mean(0)
        score = delta.abs() * w.abs().max(0).values
        out = {'top_marker_sensitive_features_heuristic': [
            {'neuron': int(n), 'mean_activation_shift': round(float(delta[n]), 4),
             'largest_abs_outgoing_weight': round(float(w[:, n].abs().max()), 4)}
            for n in score.topk(5).indices],
            'note': 'A ranking heuristic, not proof of causation.'}
    usage['inspections'] += 1
    write(P / 'usage.json', usage)
    print(json.dumps(out, indent=2))


def patch_state(base, patch):
    if not isinstance(patch, dict) or set(patch) != {'op', 'index', 'value'}:
        raise ValueError('Patch requires exactly op,index,value.')
    op, n, v = patch['op'], patch['index'], patch['value']
    if op not in ['scale_row', 'scale_feature', 'set_bias'] or type(n) is not int:
        raise ValueError('Invalid operation/index.')
    if not 0 <= n < (84 if op == 'scale_feature' else 10):
        raise ValueError('Index out of range.')
    if type(v) not in [int, float] or not np.isfinite(v) or abs(v) > (20 if op == 'set_bias' else 2):
        raise ValueError('Invalid value.')
    out = {k: x.clone() for k, x in base.items()}
    if op == 'scale_row':
        out['fc3.weight'][n] *= v
    elif op == 'scale_feature':
        out['fc3.weight'][:, n] *= v
    else:
        out['fc3.bias'][n] = v
    return out


def propose(args):
    base, d, usage = session()
    if usage['proposals'] >= 3:
        raise SystemExit('Proposal budget exhausted.')
    patch = read(args.patch)
    edited = patch_state(base, patch)
    n = usage['proposals'] + 1
    result = {'candidate': n, 'patch': patch,
              'base': evaluate_views(base, d, 'val'),
              'edited': evaluate_views(edited, d, 'val')}
    write(P / f'candidate_{n}.json', result)
    usage['proposals'] = n
    write(P / 'usage.json', usage)
    print(json.dumps(result, indent=2))


def finalize(args):
    base, d, usage = session()
    if args.candidate == 0:
        patch, edited = None, base  # Valid outcome: no justified improvement.
    elif 1 <= args.candidate <= usage['proposals']:
        patch = read(P / f'candidate_{args.candidate}.json')['patch']
        edited = patch_state(base, patch)
    else:
        raise ValueError('Choose 0 (no edit) or an existing candidate number.')
    ref = load(P / 'reference.pt')
    changed = sum(int((base[k] != edited[k]).sum()) for k in base)
    unchanged = all(torch.equal(base[k], edited[k]) for k in base if not k.startswith('fc3.'))
    assert unchanged
    result = {'candidate': args.candidate, 'patch': patch, 'usage': usage,
              'parameters': 61706, 'changed_scalars': changed,
              'changed_percent': 100 * changed / 61706,
              'feature_extractor_unchanged': unchanged,
              'recovers_reference_exactly': all(torch.equal(ref[k], edited[k]) for k in ref),
              'test': {label: evaluate_views(s, d, 'test', True)
                       for label, s in [('reference', ref), ('base', base), ('edited', edited)]}}
    torch.save(edited, ROOT / 'edited_lenet.pt')
    write(P / 'final_results.json', result)
    print(json.dumps({k: result[k] for k in ['candidate', 'changed_scalars',
          'changed_percent', 'feature_extractor_unchanged', 'recovers_reference_exactly']}, indent=2))
    print('Test metrics:', evaluate_views(edited, d, 'test'))
    print('Saved edited_lenet.pt and private/final_results.json. No more tuning this run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='cmd', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--experiment', choices=['fault', 'shortcut'], default='fault')
    prep.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--epochs', type=int, default=3)
    prep.add_argument('--train-per-class', type=int, default=1000)
    prep.add_argument('--val-per-class', type=int, default=200)
    prep.add_argument('--test-per-class', type=int, default=200)
    ins = subs.add_parser('inspect')
    group = ins.add_mutually_exclusive_group()
    group.add_argument('--row', type=int)
    group.add_argument('--neuron', type=int)
    prop = subs.add_parser('propose')
    prop.add_argument('patch', type=Path)
    fin = subs.add_parser('finalize')
    fin.add_argument('candidate', type=int)
    args = parser.parse_args()
    if args.cmd == 'prepare':
        if min(args.epochs, args.train_per_class, args.val_per_class, args.test_per_class) < 1:
            parser.error('Epoch count and split sizes must be positive.')
        prepare(args)
    elif args.cmd == 'inspect':
        inspect(args)
    elif args.cmd == 'propose':
        propose(args)
    else:
        finalize(args)
```
