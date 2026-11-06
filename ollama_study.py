"""Paired editor replication using only the Ollama CLI for model inference."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/ollama'
EXPORT = ROOT / 'results/ollama'
MODEL = 'deepseek-v4.1-flash:cloud'
CASES = [f'{kind}-{seed}' for kind in ['fault', 'shortcut'] for seed in [42, 43, 44]]
ASSETS = ['base.pt', 'reference.pt', 'cache.pt', 'splits.json', 'config.json']
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

INTERFACE = '''You are the independent diagnostic editor. Follow the supplied TASK.
Only the observations in this conversation are available. You have no file or
shell access. The adapter executes only the allowed numeric inspection/edit tools.
Return exactly one JSON object per turn with a nonempty reason, using ONE schema:
{"action":"inspect","query":"summary","reason":"why this query"}
{"action":"inspect","query":"row","index":0,"reason":"why this query"}
{"action":"inspect","query":"neuron","index":0,"reason":"why this query"}
{"action":"propose","patch":{"op":"scale_row","index":0,"value":1},"reason":"causal hypothesis and predicted effect BEFORE evaluating"}
{"action":"finish","candidate":0,"reason":"validation-based recommendation, evidence, tradeoffs"}
Numbers in these schemas are examples, not recommended queries or edits.
Use at most SIX inspections and THREE proposals; each proposal uses the same base.
Do not sweep values or use all the budget automatically. Select a candidate only
when validation justifies it; candidate 0 means no edit. For marked-image failures,
marked validation accuracy must improve and clean accuracy may lose at most one
percentage point. Also assess digit-7 recall and false-seven rates. Seek recovery
of any failing digit while preserving others. Stop when justified. Never request
training, fitting, source/private files, test data, or finalization. Return a single
action per response. The adapter returns its observation for your next turn.
'''


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, check=True, **kwargs)


def prepare():
    if RUNS.exists():
        raise SystemExit('Replication directory exists; never reset an existing study.')
    # Metadata commands use no experimental inputs.
    metadata = {'model': MODEL, 'cli_version': invoke(['ollama', '--version']).stdout.strip(),
                'model_info': invoke(['ollama', 'show', MODEL]).stdout,
                'model_listing': '\n'.join(line for line in invoke(['ollama', 'list']).stdout.splitlines() if MODEL in line),
                'thinking': 'high', 'inference_backend': 'Ollama Cloud via local CLI',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'adapter_sha256': digest(Path(__file__)),
                'protocol_sha256': digest(ROOT / 'OLLAMA_PROTOCOL.md')}
    RUNS.mkdir(parents=True)
    write(RUNS / 'metadata.json', metadata)
    for name in CASES:
        source, target = ROOT / 'runs' / name, RUNS / name
        (target / 'private').mkdir(parents=True)
        (target / 'agent').mkdir()
        hashes = {}
        for asset in ASSETS:
            original, copied = source / 'private' / asset, target / 'private' / asset
            shutil.copy2(original, copied)
            hashes[asset] = digest(original)
            assert digest(copied) == hashes[asset]
        shutil.copy2(source / 'agent/report.json', target / 'agent/report.json')
        task = (source / 'agent/TASK.md').read_text().replace(str(source), str(target))
        (target / 'agent/TASK.md').write_text(task)
        write(target / 'private/usage.json', {'inspections': 0, 'proposals': 0})
        write(target / 'provenance.json', {'paired_source': name, 'asset_sha256': hashes,
                                         'initial_report_sha256': digest(target / 'agent/report.json')})
    print('Prepared six matched, independent editor cases.', flush=True)


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get('reason'), str) or not action['reason'].strip():
        raise ValueError('Every action must have a nonempty reason.')
    kind = action.get('action')
    if kind == 'inspect':
        query = action.get('query')
        expected = {'action', 'query', 'reason'} | ({'index'} if query in ['row', 'neuron'] else set())
        if query not in ['summary', 'row', 'neuron'] or set(action) != expected:
            raise ValueError('Inspection must match one of the supplied schemas.')
        if query != 'summary':
            n = action['index']
            if type(n) is not int or not 0 <= n < (10 if query == 'row' else 84):
                raise ValueError('Inspection index out of range.')
    elif kind == 'propose':
        if set(action) != {'action', 'patch', 'reason'}:
            raise ValueError('Proposal must match the supplied schema.')
        patch = action['patch']
        if not isinstance(patch, dict) or set(patch) != {'op', 'index', 'value'}:
            raise ValueError('Patch requires exactly op,index,value.')
        op, n, value = patch['op'], patch['index'], patch['value']
        if op not in ['scale_row', 'scale_feature', 'set_bias']:
            raise ValueError('Unknown patch operation.')
        if type(n) is not int or not 0 <= n < (84 if op == 'scale_feature' else 10):
            raise ValueError('Patch index out of range.')
        if type(value) not in [float, int] or not math.isfinite(value) or abs(value) > (20 if op == 'set_bias' else 2):
            raise ValueError('Patch value out of bounds.')
    elif kind == 'finish':
        if set(action) != {'action', 'candidate', 'reason'} or type(action['candidate']) is not int or action['candidate'] < 0:
            raise ValueError('Finish must name a nonnegative candidate integer.')
    else:
        raise ValueError('Unknown action; only inspect, propose and finish are allowed.')
    return action


def acceptable(candidate):
    base, edited = candidate['base'], candidate['edited']
    return (edited['marked']['accuracy'] > base['marked']['accuracy']
            and edited['clean']['accuracy'] >= base['clean']['accuracy'] - .01)


def execute_action(folder, action):
    kind = action['action']
    usage = read(folder / 'private/usage.json')
    if kind == 'finish':
        n = action['candidate']
        if n > usage['proposals']:
            raise ValueError('Choose zero or an existing candidate.')
        if n and not acceptable(read(folder / f'private/candidate_{n}.json')):
            raise ValueError('Selected candidate fails the declared validation acceptance rule; choose another or zero.')
        return {'selected_candidate': n, 'selection_frozen': True}
    command = [sys.executable, str(ROOT / 'wsp_lenet.py'), '--run-dir', str(folder)]
    if kind == 'inspect':
        if usage['inspections'] >= 6:
            raise ValueError('Inspection budget exhausted; finish or propose using existing evidence.')
        command.append('inspect')
        if action['query'] != 'summary':
            command += ['--' + action['query'], str(action['index'])]
    else:
        if usage['proposals'] >= 3:
            raise ValueError('Proposal budget exhausted; finish using existing candidates.')
        patch_path = folder / 'agent' / f'patch_{usage["proposals"] + 1}.json'
        write(patch_path, action['patch'])
        command += ['propose', str(patch_path)]
    return json.loads(invoke(command, cwd=folder / 'agent', timeout=120).stdout)


def edit(name):
    folder = RUNS / name
    history_file = folder / 'transcript.json'
    if (folder / 'started.json').exists() or history_file.exists() or (folder / 'decision.json').exists():
        raise SystemExit(f'{name}: existing conversation; no silent restart permitted.')
    # Only these two files enter the model's initial experimental context.
    initial = {'TASK.md': (folder / 'agent/TASK.md').read_text(),
               'report.json': read(folder / 'agent/report.json')}
    write(folder / 'started.json', {'model': MODEL, 'started_utc': datetime.now(timezone.utc).isoformat()})
    history, errors = [], 0
    for turn in range(1, 15):
        prompt = INTERFACE + '\nINITIAL_FILES\n' + json.dumps(initial) + '\nCONVERSATION_HISTORY\n' + json.dumps(history) + '\nReturn your next JSON action.'
        (folder / f'prompt_{turn:02d}.txt').write_text(prompt)
        command = ['ollama', 'run', MODEL, '--format', 'json', '--hidethinking',
                   '--think', 'high', '--nowordwrap', '--verbose']
        started = time.perf_counter()
        response = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                  timeout=600, env={**os.environ, 'OLLAMA_NOHISTORY': '1'})
        (folder / f'response_{turn:02d}.txt').write_text(response.stdout)
        (folder / f'cli_{turn:02d}.log').write_text(ANSI.sub('', response.stderr))
        record = {'turn': turn, 'command': command,
                  'elapsed_seconds': round(time.perf_counter() - started, 3),
                  'returncode': response.returncode, 'response': response.stdout}
        history.append({'role': 'assistant', 'content': response.stdout})
        if response.returncode:
            write(history_file, history)
            write(folder / 'failure.json', record)
            raise SystemExit(f'{name}: Ollama failed; see local logs. No response retried.')
        try:
            action = validate_action(json.loads(response.stdout))
            # Persist the hypothesis before evaluating the proposed intervention.
            record['action'] = action
            write(folder / f'action_{turn:02d}.json', record)
            observation = execute_action(folder, action)
        except (ValueError, json.JSONDecodeError) as exc:
            errors += 1
            observation = {'interface_error': str(exc), 'corrections_remaining': max(0, 3 - errors)}
            action = None
        history.append({'role': 'tool', 'content': observation})
        write(history_file, history)
        if errors >= 3:
            raise SystemExit(f'{name}: interface correction limit reached; no automatic selection.')
        if action and action['action'] == 'finish':
            decision = {**action, 'model': MODEL, 'turns': turn, 'interface_errors': errors,
                        'usage': read(folder / 'private/usage.json'),
                        'selected_before_test': True,
                        'timestamp_utc': datetime.now(timezone.utc).isoformat()}
            write(folder / 'decision.json', decision)
            (folder / 'agent/decision.md').write_text(f'# DeepSeek validation decision\n\n{action["reason"]}\n\nCandidate: {action["candidate"]}\n')
            print(f'{name}: candidate {action["candidate"]} selected, {decision["usage"]}, {turn} turns.', flush=True)
            return
        print(f'{name}: turn {turn} {action["action"] if action else "interface correction"} completed.', flush=True)
    raise SystemExit(f'{name}: turn limit reached; no automatic selection.')


def finalize_all():
    if not all((RUNS / name / 'decision.json').exists() for name in CASES):
        raise SystemExit('Freeze all six model decisions before any final test.')
    for name in CASES:
        folder = RUNS / name
        if (folder / 'private/final_results.json').exists():
            raise SystemExit(f'{name}: already finalized; no repeat evaluation.')
        decision = read(folder / 'decision.json')
        for log, command in [
            ('finalization.log', [sys.executable, str(ROOT / 'wsp_lenet.py'), '--run-dir', str(folder), 'finalize', str(decision['candidate'])]),
            ('verification.log', [sys.executable, str(ROOT / 'verify_checkpoint.py'), str(folder)])]:
            output = invoke(command, timeout=120)
            (folder / log).write_text(output.stdout + output.stderr)
        print(f'{name}: finalized and saved model verified.', flush=True)


def report():
    EXPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS / 'metadata.json', EXPORT / 'metadata.json')
    lines = ['# DeepSeek / Ollama paired replication', '',
             'Six matched frozen LeNet/MNIST cases, seeds 42–44, evaluated 6 November 2026.',
             'Editor: `deepseek-v4.1-flash:cloud`, high thinking, invoked through the Ollama CLI.',
             'Inference used Ollama Cloud. Checkpoints/features were produced on the M4 GPU;',
             'bounded classifier evaluation and full-model verification ran locally.', '',
             '## Held-out accuracy comparison', '',
             '| Case | Base clean | Astra clean | DeepSeek clean | Base marked | Astra marked | DeepSeek marked |',
             '|---|---:|---:|---:|---:|---:|---:|']
    results = []
    pct = lambda v: f'{100*v:.2f}%'
    for name in CASES:
        folder, dest = RUNS / name, EXPORT / name
        dest.mkdir(exist_ok=True)
        final = read(folder / 'private/final_results.json')
        astra = read(ROOT / 'runs' / name / 'private/final_results.json')
        provenance = read(folder / 'provenance.json')
        assert all(digest(folder / 'private' / asset) == value for asset, value in provenance['asset_sha256'].items())
        verify = read(folder / 'private/checkpoint_verification.json')
        assert all(v['predictions_identical'] and v['logits_close'] for v in verify.values())
        assert final['feature_extractor_unchanged']
        assert final['candidate'] == read(folder / 'decision.json')['candidate']
        assert digest(folder / 'edited_lenet.pt') == final['edited_sha256']
        for source in list(folder.glob('*.json')) + list(folder.glob('*.txt')) + list(folder.glob('*.log')):
            shutil.copy2(source, dest / source.name)
        for source in (folder / 'private').glob('*.json'):
            shutil.copy2(source, dest / source.name)
        for source in (folder / 'agent').glob('*'):
            if source.suffix in ['.json', '.md']:
                shutil.copy2(source, dest / source.name)
        b, a, d = final['test']['base'], astra['test']['edited'], final['test']['edited']
        lines.append(f'| {name} | {pct(b["clean"]["accuracy"])} | {pct(a["clean"]["accuracy"])} | {pct(d["clean"]["accuracy"])} | {pct(b["marked"]["accuracy"])} | {pct(a["marked"]["accuracy"])} | {pct(d["marked"]["accuracy"])} |')
        results.append((name, final))
    lines += ['', '## DeepSeek decisions and class tradeoffs', '',
              '| Case | Candidate | Inspections / proposals | Scalars changed | Exact reference recovery | Clean seven recall, base → edited | Marked false-seven rate, base → edited |',
              '|---|---:|---|---:|---|---|---|']
    for name, result in results:
        b, d, u = result['test']['base'], result['test']['edited'], result['usage']
        lines.append(f'| [{name}](results/ollama/{name}/decision.md) | {result["candidate"]} | {u["inspections"]} / {u["proposals"]} | {result["changed_scalars"]} | {result["recovers_reference_exactly"]} | {pct(b["clean"]["digit7_recall"])} → {pct(d["clean"]["digit7_recall"])} | {pct(b["marked"]["false7_rate_on_other_digits"])} → {pct(d["marked"]["false7_rate_on_other_digits"])} |')
    lines += ['', '## Method and limits', '',
              '- Checkpoints, caches and splits match the Astra cases byte for byte. No retraining or setup changes were made.',
              '- Each case had a fresh visible conversation, at most six inspections and three proposals. All six recommendations were frozen before any DeepSeek test evaluation.',
              '- DeepSeek chose the queries, hypotheses, edits and final candidates. The coordinator supplied no weight values, diagnosis hints, edit suggestions or Astra outcomes.',
              '- A JSON-action adapter maps responses to allowlisted commands; it never executes model-supplied shell/code. Each CLI invocation includes the complete visible history for that case.',
              '- This adapter differs from Astra’s native agent interface. The same test cohort was used previously for Astra; this is a paired editor comparison, not a new untouched test cohort.',
              '- Model tags may change provider-side; saved metadata and manifest identifiers do not prove an immutable cloud snapshot. No model-ranking or statistical-significance claim is made.',
              '- Reference means the uncorrupted checkpoint for faults and the unedited shortcut-trained baseline for shortcuts. Bias/feature edits may compensate without removing a shortcut representation.',
              '- Full prompts, CLI responses, causal reasons, tool observations, timings, candidates and selection records are exported under `results/ollama/`. Checkpoints remain local under ignored `runs/ollama/`.',
              '- All six checkpoint reload verifications passed with identical predictions and numerically close logits for clean and marked views.', '']
    (ROOT / 'OLLAMA_RESULTS.md').write_text('\n'.join(lines))
    write(EXPORT / 'verification_summary.json', {'cases': CASES, 'paired_asset_hashes_match': True,
                                               'all_saved_checkpoints_verified': True,
                                               'editor_model': MODEL})
    print('Exported six paired results and OLLAMA_RESULTS.md.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'edit', 'finalize', 'report'])
    parser.add_argument('--case', choices=CASES + ['all'], default='all')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'edit':
        for name in CASES if args.case == 'all' else [args.case]:
            edit(name)
    elif args.command == 'finalize':
        finalize_all()
    else:
        report()
