"""After finalization, verify a saved model against cached inference."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import MNIST

from wsp_lenet import LeNet, images, load, logits, marked, read, write


@torch.no_grad()
def verify(root):
    if not (root / 'private/final_results.json').exists():
        raise SystemExit('Verification is allowed only after finalization.')
    torch.set_num_threads(4)
    config = read(root / 'private/config.json')
    splits = read(root / 'private/splits.json')
    cache = load(root / 'private/cache.pt')
    state = load(root / 'edited_lenet.pt')
    model = LeNet().eval().requires_grad_(False)
    model.load_state_dict(state)
    dataset = MNIST(config['data_dir'], train=False, download=False)
    x, _ = images(dataset, splits['test'])
    results = {}
    for view, inputs in [('clean', x), ('marked', marked(x))]:
        fresh = torch.cat([model(batch) for batch in inputs.split(128)])
        cached = logits(state, cache['test_' + view])
        results[view] = {
            'max_abs_logit_error': float((fresh - cached).abs().max()),
            'predictions_identical': bool(torch.equal(fresh.argmax(1), cached.argmax(1))),
            'logits_close': bool(torch.allclose(fresh, cached, atol=5e-4, rtol=1e-4)),
        }
    write(root / 'private/checkpoint_verification.json', results)
    print(json.dumps(results, indent=2))
    if not all(r['predictions_identical'] and r['logits_close'] for r in results.values()):
        raise SystemExit('Saved-model verification failed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    verify(parser.parse_args().run_dir.resolve())
