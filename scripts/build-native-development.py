#!/usr/bin/env python3
"""Freeze a shared-template development sample from current curriculum validation.

This is development data for candidate selection, never the fresh final suite.
Only current validation records supply cases; corpus artifacts are hash-verified.
No inference, tool execution, network, or private state is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from silivra.agent import build_system_prompt
from silivra.checkpoint import sha256
from silivra.evaluation import validate_cases
from silivra.finetune import _manifest
from silivra.instructions import parse_action, validate_record
from silivra.tools import ToolRegistry


def cases_from_records(records, registry, per_family=6):
    if type(per_family) is not int or not 1 <= per_family <= 12:
        raise ValueError('per_family must be an integer from1to12')
    available = {s['function']['name']: s for s in registry.schemas()}
    grouped = defaultdict(list)
    for row in sorted(records, key=lambda r: r['id']):
        if row['source'].get('revision') != 'native-english-v6':
            continue
        validate_record(row)
        if not row['id'].startswith('v6-'):
            raise ValueError('Unexpected original curriculum ID')
        family = row['id'].rsplit('-', 1)[0][3:]
        system = row['messages'][0]['content']
        specs = json.loads(system.split(' Tools: ', 1)[1]) if ' Tools: ' in system else []
        names = [s['name'] for s in specs]
        if len(names) != len(set(names)) or set(names) - set(available):
            raise ValueError('Unavailable development tool schema')
        tools = [available[name] for name in names]
        if system != build_system_prompt(tools):
            raise ValueError('Curriculum policy/schema changed')
        for index, message in enumerate(row['messages']):
            if message['role'] != 'assistant':
                continue
            category = family if index == 2 else family + '-observation'
            grouped[category].append({
                'id': row['id'] + '-decision-' + str(index), 'category': category,
                'messages': row['messages'][1:index], 'tools': tools,
                'expected': parse_action(message['content'])})
    cases = [case for category in sorted(grouped) for case in grouped[category][:per_family]]
    validate_cases(cases)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-family', type=int, default=6)
    args = parser.parse_args()
    metadata = args.output.with_suffix('.meta.json')
    if args.output.exists() or metadata.exists():
        parser.error('Outputs must be new; never replace a frozen development suite')
    data = args.data.resolve()
    manifest = _manifest(data)
    rows = [json.loads(line) for line in (data / manifest['splits']['valid']['source_path']).read_text().splitlines() if line.strip()]
    cases = cases_from_records(rows, ToolRegistry(Path.cwd(), allow_network=True), args.per_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(''.join(json.dumps(case, ensure_ascii=False, separators=(',', ':'))+'\n' for case in cases))
    with metadata.open('x') as stream:
        json.dump({'kind': 'shared-template curriculum development only',
            'cases': len(cases), 'per_family_limit': args.per_family,
            'data_manifest_sha256': sha256(data/'manifest.json'),
            'validation_source_sha256': manifest['splits']['valid']['source_sha256'],
            'suite_sha256': sha256(args.output), 'script_sha256': sha256(Path(__file__)),
            'boundary': 'Stable record-ID sample of original current validation decisions; exact parsed actions. Used for development selection, not a blind benchmark. No final suite or training answers read.'}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'cases': len(cases), 'suite_sha256': sha256(args.output)}))


if __name__ == '__main__':
    main()
