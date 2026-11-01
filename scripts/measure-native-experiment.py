#!/usr/bin/env python3
"""Development-only pilots and a frozen final diagnostic for native Silivra.

No network/tools are executed. Model inference uses only Silivra's own native
checkpoint. Final answers must stay out of training and candidate selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / 'examples/evaluations/native-english-v6.jsonl'
PROMPTS = ROOT / 'examples/evaluations/native-english-v6-prompts.jsonl'
FRESH_SHA256 = 'fff3cc20cb287e2aa9099aca5a567d2b81d2e57a081f99430aef9bf2a0480885'
PROMPTS_SHA256 = 'aaaf48bcaca980b23603ee34d22c3c2844d4a0d59588460e52f3bb542a8f5add'
DEVELOPMENT = tuple(ROOT / 'examples/evaluations' / name for name in
                    ('assistant.jsonl', 'assistant-v2.jsonl', 'quality-v1.jsonl', 'quality-v2.jsonl'))
GROUPS = {
    'english': {'english'}, 'conversation': {'conversation'},
    'grounding': {'copy', 'grounding'}, 'uncertainty': {'uncertainty'},
    'tool_selection': {'tool_selection_calculate', 'tool_selection_search'},
    'tool_observation': {'tool_observation_calculate', 'tool_observation_search'},
    'memory': {'memory'}, 'coding': {'coding'}, 'reasoning': {'reasoning'},
}
# Predeclared before any baseline or candidate generations. These are diagnostic
# promotion gates, not a claim that passing this small suite proves intelligence.
MINIMUM_PASSES = {'english': 8, 'conversation': 3, 'grounding': 6, 'uncertainty': 3,
                  'tool_selection': 4, 'tool_observation': 3, 'memory': 3,
                  'coding': 3, 'reasoning': 1}
MAX_RETENTION_REGRESSION = .05


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def user_messages(case):
    return [m['content'] for m in case.get('messages', [{'role': 'user', 'content': case.get('prompt')}])
            if m['role'] == 'user']


def verify_frozen_inputs(*, include_answers=False):
    if digest(PROMPTS) != PROMPTS_SHA256:
        raise ValueError('Fresh prompt exclusions changed after freezing.')
    prompts = read_jsonl(PROMPTS)
    if not prompts or any(not isinstance(text, str) or not text.strip() for text in prompts):
        raise ValueError('Exclusions must contain only nonempty JSON strings, never answers.')
    # Pilots deliberately do not open the fresh suite, including for validation.
    if include_answers and digest(FRESH) != FRESH_SHA256:
        raise ValueError('Fresh final suite changed after freezing.')
    return {normalize(text) for text in prompts}


def audit_exclusions(directory, manifest):
    fresh = verify_frozen_inputs()
    dev = {normalize(text) for path in DEVELOPMENT for case in read_jsonl(path)
           for text in user_messages(case)}
    denied = fresh | dev
    counts = {}
    for split in ('train', 'valid'):
        rows = read_jsonl(directory / manifest['splits'][split]['source_path'])
        overlaps = sum(any(normalize(m['content']) in denied for m in row['messages']
                           if m['role'] == 'user') for row in rows)
        if overlaps:
            raise ValueError('Held-out user messages overlap a training or development-validation split.')
        counts[split] = {'conversations': len(rows), 'exact_user_message_overlaps': overlaps}
    return {'fresh_prompt_sha256': PROMPTS_SHA256, 'splits': counts,
            'boundary': 'Exact NFC/whitespace-normalized user-message exclusion only; semantic and earlier pretraining overlap cannot be ruled out.'}


def audit_original_fresh_overlap(prior_directory):
    from silivra.finetune import _manifest
    manifest = _manifest(prior_directory.resolve())
    denied = verify_frozen_inputs()
    rows = read_jsonl(prior_directory / manifest['splits']['train']['source_path'])
    overlaps = sum(any(normalize(m['content']) in denied for m in row['messages']
                       if m['role'] == 'user') for row in rows)
    if overlaps:
        raise ValueError('Fresh final user messages overlap original instruction training.')
    return {'exact_user_message_overlaps': overlaps,
            'prior_manifest_sha256': digest(prior_directory / 'manifest.json'),
            'boundary': 'Exact user-message audit only; no semantic or base-pretraining contamination guarantee.'}


def runtime_hashes():
    return {name: digest(ROOT / 'src/silivra' / name) for name in
            ('agent.py', 'evaluation.py', 'inference.py', 'model.py', 'instructions.py')}


def descriptor(path):
    from silivra.checkpoint import resolve_checkpoint, verify_checkpoint
    checkpoint = resolve_checkpoint(path, 'best')
    state = verify_checkpoint(checkpoint)
    if state.get('initialization') not in ('random; no pretrained weights or tokenizer',
                                          'verified local Silivra checkpoint'):
        raise ValueError('Only native from-scratch Silivra checkpoint lineage is supported.')
    return checkpoint, state, {'checkpoint': checkpoint.name, 'step': state['step'],
        'context_window': state['model_config']['max_seq_len'],
        'hashes': {name: digest(checkpoint / name)
                   for name in ('model.safetensors', 'tokenizer.json', 'state.json')}}


def descends_from(base, weights_hash):
    return isinstance(base, dict) and (base.get('weights_sha256') == weights_hash
                                      or descends_from(base.get('parent'), weights_hash))


def validate_selection(selection, current, data_hash):
    if (selection.get('phase') != 'pilot' or selection.get('role') != 'candidate'
            or selection.get('checkpoint') != current
            or selection.get('data_manifest_sha256') != data_hash
            or selection.get('fresh_suite_executed') is not False):
        raise ValueError('Final candidate must match an existing development-only pilot report.')


def aggregate(evaluations):
    categories = defaultdict(lambda: {'passed': 0, 'cases': 0, 'valid_actions': 0})
    for report in evaluations.values():
        for name, item in report['categories'].items():
            for key in ('passed', 'cases', 'valid_actions'):
                categories[name][key] += item[key]
    return {'passed': sum(r['passed'] for r in evaluations.values()),
            'cases': sum(r['cases'] for r in evaluations.values()),
            'valid_actions': sum(r['valid_actions'] for r in evaluations.values()),
            'categories': dict(categories)}


def fresh_groups(report):
    result = {}
    for name, members in GROUPS.items():
        rows = [report['categories'][category] for category in members]
        result[name] = {key: sum(row[key] for row in rows) for key in ('passed', 'cases', 'valid_actions')}
    return result


def retention_gate(losses, baseline):
    gates = {}
    for key in ('prior_instruction_loss', 'story_validation_loss'):
        value, before = losses.get(key), baseline.get(key)
        valid = (isinstance(value, (int, float)) and isinstance(before, (int, float))
                 and math.isfinite(value) and math.isfinite(before) and before > 0)
        gates[key] = {'before': before, 'after': value,
                      'relative_change': value / before - 1 if valid else None,
                      'passed': bool(valid and value <= before * (1 + MAX_RETENTION_REGRESSION))}
    return gates


def acceptance(record, baseline):
    retention = retention_gate(record['losses'], baseline['losses'])
    development = record['development_summary']
    broad = development['cases'] > 0 and development['passed'] / development['cases'] >= .5
    broad_nonregression = development['passed'] >= baseline['development_summary']['passed']
    result = {'retention': retention,
              'development_at_least_half_correct': broad,
              'development_no_accuracy_regression': broad_nonregression,
              'eligible': False}
    if 'fresh_summary' in record:
        fresh = record['fresh_summary']
        result['fresh_at_least_three_quarters_correct'] = fresh['cases'] > 0 and fresh['passed'] / fresh['cases'] >= .75
        result['fresh_domain_minimums'] = {name: {'passed': item['passed'] >= MINIMUM_PASSES[name],
            'correct': item['passed'], 'cases': item['cases'], 'minimum_correct': MINIMUM_PASSES[name]}
            for name, item in record['fresh_groups'].items()}
        result['eligible'] = (all(x['passed'] for x in retention.values()) and broad and broad_nonregression
            and result['fresh_at_least_three_quarters_correct']
            and all(x['passed'] for x in result['fresh_domain_minimums'].values()))
    return result


def evaluate_development(backend):
    from silivra.evaluation import evaluate_assistant
    return {path.stem: evaluate_assistant(backend, path) for path in DEVELOPMENT}


def evaluate_fresh(backend):
    from silivra.evaluation import evaluate_assistant
    verify_frozen_inputs(include_answers=True)
    return evaluate_assistant(backend, FRESH)


def measure_losses(backend, *, story_directory, prior_directory, data_directory=None):
    from silivra.data import TokenDataset
    from silivra.finetune import _manifest, evaluate_loss as instruction_loss
    from silivra.instructions import InstructionDataset
    from silivra.train import evaluate_loss as story_loss
    original_tokenizer = digest(backend.checkpoint / 'tokenizer.json')
    result = {}
    directories = [('prior_instruction_loss', prior_directory, 4, 24)]
    if data_directory is not None:
        directories.append(('development_validation_loss', data_directory, 8, 16))
    for key, directory, batch_size, batches in directories:
        manifest = _manifest(directory.resolve())
        if manifest['tokenizer']['sha256'] != original_tokenizer:
            raise ValueError('Instruction retention tokenizer differs from native lineage.')
        if manifest['max_seq_len'] > backend.model.config.max_seq_len:
            raise ValueError('Complete validation contexts exceed the native checkpoint; no truncation.')
        dataset = InstructionDataset(directory / manifest['splits']['valid']['path'],
                                     backend.model.config.max_seq_len, backend.model.config.vocab_size,
                                     bucket_size=64)
        result[key] = instruction_loss(backend.model, dataset, batch_size, batches)
    manifest = read_json(story_directory / 'manifest.json')
    story_file = story_directory / manifest['splits']['valid']['path']
    if (digest(story_file) != manifest['splits']['valid']['sha256']
            or manifest['tokenizer']['sha256'] != original_tokenizer):
        raise ValueError('Story retention checksum or tokenizer mismatch.')
    result['story_validation_loss'] = story_loss(backend.model, TokenDataset(story_file, 256), 16, 12)
    return result


def claim_final_receipt(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Receipt is written before fresh inference; a failed/interrupted final attempt
    # stays visible instead of silently permitting another candidate selection.
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'fresh_suite_sha256': FRESH_SHA256, 'role': record['role'],
                   'checkpoint': record['checkpoint'], 'selection_report_sha256': record.get('selection_report_sha256')}, stream, indent=2)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    parser.add_argument('--baseline-only', action='store_true', help='Final-suite baseline of original checkpoint, before training')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--selection-report', type=Path)
    parser.add_argument('--final-receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--story-data', type=Path, default=ROOT / 'data/tinystories')
    parser.add_argument('--prior-data', type=Path, default=ROOT / 'data/instructions-v2')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be absent; reports cannot be replaced.')
    if args.baseline_only and args.phase != 'final':
        parser.error('--baseline-only requires --phase final.')
    if args.phase == 'final' and args.final_receipt is None:
        parser.error('Final fresh inference requires an exclusive --final-receipt path.')
    if not args.baseline_only and (args.data is None or args.baseline_report is None):
        parser.error('Candidate evaluation requires --data and --baseline-report.')
    if args.phase == 'final' and not args.baseline_only and args.selection_report is None:
        parser.error('Final candidate requires a preexisting --selection-report from pilot phase.')
    verify_frozen_inputs(include_answers=args.phase == 'final')
    checkpoint, state, current = descriptor(args.checkpoint)
    evaluator_hash = digest(Path(__file__))
    runtime = runtime_hashes()
    baseline = None
    if args.baseline_report is not None:
        baseline = read_json(args.baseline_report)
        if (baseline.get('role') != 'baseline' or baseline.get('phase') != 'final'
                or baseline.get('fresh_suite_sha256') != FRESH_SHA256):
            raise ValueError('A frozen native baseline report is required.')
        if (baseline.get('evaluator_sha256') != evaluator_hash
                or baseline.get('runtime_sha256') != runtime):
            raise ValueError('Evaluator/runtime changed since the frozen baseline.')
        if current['hashes']['tokenizer.json'] != baseline['checkpoint']['hashes']['tokenizer.json']:
            raise ValueError('Candidate changed the original native tokenizer.')
        if not descends_from(state.get('base_checkpoint'), baseline['checkpoint']['hashes']['model.safetensors']):
            raise ValueError('Candidate must descend from the measured original baseline.')
    record = {'format_version': 1, 'phase': args.phase,
              'role': 'baseline' if args.baseline_only else 'candidate',
              'purpose': 'Native original-model experiment; diagnostic evidence, not general intelligence proof',
              'checkpoint': current, 'evaluator_sha256': evaluator_hash,
              'runtime_sha256': runtime,
              'fresh_suite_sha256': FRESH_SHA256,
              'fresh_suite_executed': args.phase == 'final',
              'generation': {'max_new_tokens': 96, 'temperature': 0, 'top_k': 0},
              'tools_executed': False,
              'candidate_selection': 'Development validation and development diagnostics only; fresh final answers are reserved.'}
    record['original_instruction_fresh_overlap'] = audit_original_fresh_overlap(args.prior_data)
    if args.data is not None:
        from silivra.finetune import _manifest
        manifest = _manifest(args.data.resolve())
        data_hash = digest(args.data / 'manifest.json')
        if state.get('data_manifest_sha256') != data_hash:
            raise ValueError('Candidate checkpoint does not bind the supplied training data.')
        record['data_manifest_sha256'] = data_hash
        record['exclusion_audit'] = audit_exclusions(args.data, manifest)
    if args.selection_report is not None:
        selection = read_json(args.selection_report)
        validate_selection(selection, current, record.get('data_manifest_sha256'))
        record['selection_report_sha256'] = digest(args.selection_report)
    from silivra.agent import MLXBackend
    backend = MLXBackend(checkpoint, max_new_tokens=96)
    record['losses'] = measure_losses(backend, story_directory=args.story_data,
                                     prior_directory=args.prior_data, data_directory=args.data)
    record['development'] = evaluate_development(backend)
    record['development_summary'] = aggregate(record['development'])
    if baseline is not None and any(report['suite_sha256'] != baseline['development'][name]['suite_sha256']
                                    for name, report in record['development'].items()):
        raise ValueError('Development suites changed since baseline measurement.')
    if args.phase == 'final':
        claim_final_receipt(args.final_receipt, record)
        record['fresh'] = evaluate_fresh(backend)
        record['fresh_summary'] = aggregate({'fresh': record['fresh']})
        record['fresh_groups'] = fresh_groups(record['fresh'])
    if baseline is not None:
        record['acceptance'] = acceptance(record, baseline)
        record['baseline_report_sha256'] = digest(args.baseline_report)
    if descriptor(checkpoint)[2] != current:
        raise ValueError('Checkpoint changed during measurement.')
    record['checkpoint_immutable'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'phase': record['phase'], 'role': record['role'],
                      'checkpoint': current['checkpoint'], 'development': record['development_summary']['passed'],
                      'fresh': record.get('fresh_summary', {}).get('passed'),
                      'eligible': record.get('acceptance', {}).get('eligible')}), flush=True)


if __name__ == '__main__':
    main()
