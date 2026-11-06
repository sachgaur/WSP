"""Export finalized JSON evidence and a workshop pilot report (no tensor access)."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def percent(value):
    return f'{100 * value:.2f}%'


def pair(before, after, key):
    return f'{percent(before[key])} → {percent(after[key])}'


def main():
    rows = []
    for folder in sorted((ROOT / 'runs').glob('*')):
        private = folder / 'private'
        if not (private / 'final_results.json').exists():
            continue
        result = read(private / 'final_results.json')
        config = read(private / 'config.json')
        exported = ROOT / 'results' / folder.name
        exported.mkdir(parents=True, exist_ok=True)
        for pattern in ['config.json', 'splits.json', 'usage.json', 'selection.json',
                        'final_results.json', 'checkpoint_verification.json', 'candidate_*.json']:
            for source in private.glob(pattern):
                shutil.copy2(source, exported / source.name)
        for source in (folder / 'agent').glob('*'):
            if source.suffix in {'.md', '.json'}:
                shutil.copy2(source, exported / source.name)
        if (folder / 'preparation.log').exists():
            shutil.copy2(folder / 'preparation.log', exported / 'preparation.log')
        rows.append((folder.name, config, result))

    lines = ['# Workshop pilot results', '', 'Measured on 6 November 2026 on an Apple M4 MacBook Pro with 16 GB RAM.',
             'Python 3.12.11, PyTorch 2.15.0, torchvision 0.29.2; study runs use MPS.', '',
             'Each run uses 10,000 training, 2,000 validation and 2,000 held-out test images,',
             'balanced across digits, and three training epochs. Fresh `gpt-6-astra` editors',
             'received only their generated task and validation report. Each run\'s proposals were',
             'chosen before its final-test evaluation. No training occurred during repair.', '',
             '## Held-out test accuracy', '',
             '| Run | Candidate | Clean before → after | Marked before → after | Scalars changed | Exact reference recovery |',
             '|---|---:|---|---|---:|---|']
    for name, config, result in rows:
        before, after = result['test']['base'], result['test']['edited']
        lines.append(f"| {name} | {result['candidate']} | {pair(before['clean'], after['clean'], 'accuracy')} | "
                     f"{pair(before['marked'], after['marked'], 'accuracy')} | {result['changed_scalars']} | "
                     f"{'yes' if result['recovers_reference_exactly'] else 'no'} |")
    lines += ['', '## Workshop assessment', '',
              'The controlled-fault exercise restored the original checkpoint exactly in all',
              'three tested seeds. It is a suitable first exercise for demonstrating diagnosis,',
              'a small explicit edit and held-out verification.', '',
              'The shortcut exercise is suitable for discussing seed sensitivity and tradeoffs.',
              'All three selected edits improved marked test accuracy, but gains varied greatly.',
              'Seed 43 had little aggregate marked-image failure before editing, and its selected',
              'edit did not reduce false-seven rates. The bias edits in seeds 42 and 44 reduced',
              'false sevens while losing 5.0 and 5.5 percentage points of clean seven recall.',
              'These results support compensating classifier edits, not removal of the learned',
              'shortcut or preservation of every digit\'s performance.', '',
              'The predeclared overall-accuracy tolerance allowed those class-specific losses.',
              'A future study could predeclare a per-digit recall constraint. That stricter rule',
              'was not used to retune or reselect any outcome reported here.']
    lines += ['', '## Class-level test behavior', '',
              '| Run | View | Digit-7 recall before → after | False-seven rate before → after |',
              '|---|---|---|---|']
    for name, config, result in rows:
        for view in ['clean', 'marked']:
            before, after = result['test']['base'][view], result['test']['edited'][view]
            lines.append(f"| {name} | {view} | {pair(before, after, 'digit7_recall')} | "
                         f"{pair(before, after, 'false7_rate_on_other_digits')} |")
    lines += ['', '## Validation decisions and verification', '',
              '| Run | Inspections / proposals | Clean validation before → after | Marked validation before → after | Preparation seconds |',
              '|---|---|---|---|---:|']
    for name, config, result in rows:
        evidence = ROOT / 'results' / name
        initial = read(evidence / 'report.json')['validation']
        candidate = result['candidate']
        edited = read(evidence / f'candidate_{candidate}.json')['edited'] if candidate else initial
        usage = result['usage']
        lines.append(f"| [{name}](results/{name}/decision.md) | {usage['inspections']} / {usage['proposals']} | "
                     f"{pair(initial['clean'], edited['clean'], 'accuracy')} | "
                     f"{pair(initial['marked'], edited['marked'], 'accuracy')} | {config['elapsed_seconds']} |")
    lines += ['', 'Check `results/<run>/checkpoint_verification.json` for fresh full-model CPU',
              'inference against cached logits. `final_results.json` records unchanged feature',
              'extractors, checkpoint hashes and per-digit confusion matrices. `selection.json`',
              'records the choice frozen before test evaluation. Exact split membership and',
              'Astra decision records are exported with each run.', '',
              '## Interpretation and limits', '',
              '- Controlled faults are engineered failures. Exact recovery demonstrates the bounded diagnostic/editing workflow; it does not establish general weight understanding.',
              '- In the fault runs, reference means the uncorrupted checkpoint. In shortcut runs, reference is the unedited shortcut-trained baseline, not a robust oracle.',
              '- Shortcut edits are restricted to one classifier operation. Unsuccessful or rejected proposals are retained; no edit is a valid outcome. An output bias or scale can compensate for a shortcut without removing its internal representation. Inspect class-specific regressions alongside overall accuracy.',
              '- Seeds share the MNIST dataset and may have overlapping splits. These are seed-sensitivity checks, not independent population samples.',
              '- Small differences on 2,000 images are preliminary. No statistical or production-reliability claim is made.',
              '- The coordinator saw setup code. Each editor was launched as a fresh Astra agent with no inherited context. Access restrictions were instructions, not an OS sandbox.',
              '- Preparation timing includes training, feature extraction and checkpoint writing. The first run also includes the dataset download. It excludes Astra reasoning and is not a total workshop-duration estimate.',
              '- Infrastructure note: restricted execution initially hid MPS. One CPU shortcut preparation was retained locally as `runs/shortcut-42-cpu-infrastructure`; it received no editor, candidate selection or test evaluation. The reported study uses fresh MPS runs throughout.',
              '- Python dependency and MNIST download attempts initially failed under network restrictions; approved execution completed them. These setup failures are not experiment outcomes.', '',
              'Binary checkpoints, activation caches and MNIST downloads remain local under ignored',
              '`runs/` and `data/`. Compact evidence is ready for a later GitHub check-in.', '']
    (ROOT / 'STUDY_RESULTS.md').write_text('\n'.join(lines))
    print(f'Exported {len(rows)} finalized runs.')


if __name__ == '__main__':
    main()
