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
SOURCE_ROOT = ROOT


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
    data_root = args.data_dir.resolve()
    trds = MNIST(data_root, train=True, download=True)
    teds = MNIST(data_root, train=False, download=True)
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
              'train_per_class': args.train_per_class,
              'val_per_class': args.val_per_class,
              'test_per_class': args.test_per_class,
              'data_dir': str(data_root),
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
    command = f'{SOURCE_ROOT / ".venv/bin/python"} {Path(__file__).resolve()} --run-dir {ROOT}'
    (A / 'TASK.md').write_text('''You are investigating a small MNIST classifier.
Read only TASK.md and report.json. Diagnose a failure, then propose a small
weight edit with a causal explanation. Clean means no added mark; marked
means a white 4x4 top-left patch on every image. Digits retain their labels.
Use validation only. Inspect the reports before deciding what to query.

You may run ONLY these commands, from this agent/ directory:
COMMAND inspect
COMMAND inspect --row N
COMMAND inspect --neuron N
COMMAND propose PATCH.json

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
'''.replace('COMMAND', command))
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
    # Freeze validation-based selection before looking at any test metrics.
    write(P / 'selection.json', {'candidate': args.candidate, 'patch': patch,
                                'base_sha256': digest(P / 'base.pt')})
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
    result['edited_sha256'] = digest(ROOT / 'edited_lenet.pt')
    write(P / 'final_results.json', result)
    print(json.dumps({k: result[k] for k in ['candidate', 'changed_scalars',
          'changed_percent', 'feature_extractor_unchanged', 'recovers_reference_exactly']}, indent=2))
    print('Test metrics:', evaluate_views(edited, d, 'test'))
    print('Saved edited_lenet.pt and private/final_results.json. No more tuning this run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=SOURCE_ROOT,
                        help='Fresh directory for one experiment and its artifacts.')
    subs = parser.add_subparsers(dest='cmd', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--experiment', choices=['fault', 'shortcut'], default='fault')
    prep.add_argument('--data-dir', type=Path, default=SOURCE_ROOT / 'data')
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
    ROOT = args.run_dir.resolve()
    P, A = ROOT / 'private', ROOT / 'agent'
    ROOT.mkdir(parents=True, exist_ok=True)
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
