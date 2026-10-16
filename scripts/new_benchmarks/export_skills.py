"""Deterministic, auditable projection of extracted SkillRL strategy sections.

No additional model synthesis. The complete source strategies remain in evidence.jsonl;
the prompt bank uses bounded, diverse verbatim excerpts from successful training tasks.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parsed_object(text):
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c == '{':
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                pass
    raise ValueError('No strategy object in recorded response')


def tokens(text):
    return set(re.findall(r'[a-z]+', text.lower().replace('_', ' ')))


def select(items, limit):
    grouped = defaultdict(list)
    for text, source in items:
        if text and text.strip():
            grouped[text.strip()].append(source)
    counts = Counter({text: len(sources) for text, sources in grouped.items()})
    chosen, covered = [], set()
    while counts and len(chosen) < limit:
        # Prefer supported, concise, diverse workflows over long episode narration.
        def rank(text):
            words = tokens(text)
            novelty = len(words - covered) / max(1, len(words))
            narration = any(marker in text.lower() for marker in ['in this episode', 'in this instance', 'score 100', 'first and only'])
            score = min(counts[text], 5) + novelty - min(len(text) / 1000, 2) - 3 * narration
            return (-score, hashlib.sha256(text.encode()).hexdigest())
        text = min(counts, key=rank)
        chosen.append((text, sorted(grouped[text])))
        covered.update(tokens(text))
        del counts[text]
    return chosen


def skill_entries(items, prefix, count, category):
    return [{'skill_id': f'{prefix}_{i:03d}', 'title': f'Observed {category} workflow {i}',
             'principle': text, 'when_to_apply': f'When solving a {category} task and its observed prerequisites hold.',
             'evidence_episode_ids': sources}
            for i, (text, sources) in enumerate(select(items, count), 1)]


def export(source, kind, output):
    config = json.loads((source / 'config.json').read_text())
    assert hashlib.sha256((source / 'inputs.jsonl').read_bytes()).hexdigest() == config['selected_manifest_sha256']
    manifest = json.loads((ROOT / 'data/new_benchmarks' / f'{kind}.json').read_text())
    allowed = {str(v) for v in manifest['train']} if kind == 'minihack_room' else {tuple(v) for v in manifest['train']}
    all_rows = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
    evidence, patterns, failures = [], defaultdict(list), []
    def identity(row):
        value = row['episode_id'].split('/')[1]
        return str(int(value[4:])) if kind == 'minihack_room' else tuple(map(int, re.fullmatch(r't(\d+)_v(\d+)', value).groups()))
    required = [r for r in all_rows if r['environment'] == kind and identity(r) in allowed]
    if len(required) != len(allowed):
        raise ValueError(f'Missing or duplicated source tasks: {len(required)} != {len(allowed)}')
    for row in sorted(required, key=lambda x: x['episode_id']):
        path = source / 'sections' / row['episode_id'] / 'strategic_guidelines.json'
        cached = json.loads(path.read_text())
        strategy = cached['value']
        raw = json.loads((source / cached['raw_response_path']).read_text())
        expected = digest({'protocol': config['extraction_protocol_sha256'], 'messages': raw['request']['messages']})
        assert cached['request_sha256'] == raw['request_sha256'] == expected
        assert raw['status'] == 200 and raw['error'] is None
        assert raw['request']['model'] == config['model'] and raw['request']['temperature'] == config['temperature']
        choice = raw['response']['choices'][0]
        assert choice['finish_reason'] == 'stop'
        value = parsed_object(choice['message']['content'])
        assert value.get('strategic_guidelines', value) == strategy
        evidence.append({'episode_id': row['episode_id'], 'category': row['category'], 'success': row['success'],
                         'strategic_guidelines': strategy, 'request_sha256': cached['request_sha256'],
                         'context_sha256': row['context_sha256'], 'source_sha256': row['source_sha256']})
        if row['success']:
            patterns[row['category']].append((strategy['planning_pattern'], row['episode_id']))
        else:
            failures.extend((m if isinstance(m, str) else json.dumps(m, sort_keys=True), row['episode_id'])
                            for m in strategy['mistakes_to_avoid'])
    categories = sorted({r['category'] for r in required})
    general = skill_entries([x for items in patterns.values() for x in items], 'gen', 15, kind)
    specific = {c: skill_entries(patterns[c], c, 6, c) for c in categories}
    source_categories = {r['episode_id']: r['category'] for r in required}
    for entry in general:
        scope = sorted({source_categories[i] for i in entry['evidence_episode_ids']})
        entry['when_to_apply'] = 'Only for these task types, with matching prerequisites: ' + ', '.join(scope)
    mistakes = [{'mistake_id': f'err_{i:03d}', 'description': text,
                 'how_to_avoid': 'Check the observed feedback and avoid repeating this demonstrated failure.',
                 'evidence_episode_ids': ids} for i, (text, ids) in enumerate(select(failures, 5), 1)]
    metadata = {'selection_policy': 'first_rollout_r00', 'total_strategies_analyzed': len(required),
                'successes': sum(r['success'] for r in required), 'source_model': 'glm53_512k_o64k_pd128_coding',
                'compilation': 'deterministic_diverse_verbatim_excerpts_v1',
                'general_budget': 15, 'category_budget': 6,
                'categories_without_success_evidence': [c for c in categories if not specific[c]],
                'training_manifest_sha256': hashlib.sha256(json.dumps(manifest['train']).encode()).hexdigest()}
    bank = {'general_skills': general, 'task_specific_skills': specific, 'common_mistakes': mistakes, 'metadata': metadata}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'claude_style_skills.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2) + '\n')
    (output / 'evidence.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in evidence))
    def render(title, entries):
        return f'### {title} ###\n\n' + '\n\n'.join(f"{i}. {s['principle']}\nApply when: {s['when_to_apply']}" for i, s in enumerate(entries, 1)) + '\n'
    (output / 'general_skills.md').write_text(render('GENERAL SKILLS', general))
    files = {'general_skills': 'general_skills.md'}
    for category, entries in specific.items():
        files[category] = category + '.md'
        (output / files[category]).write_text(render('TASK: ' + category, entries) if entries else
                                             f'### TASK: {category} ###\n\nNo successful selected training evidence; use general skills.\n')
    (output / 'skill_mapping.json').write_text(json.dumps({'skill_files': files,
             'task_to_skill': {c: c for c in categories}, 'task_keywords': {}}, indent=2) + '\n')
    print(kind, json.dumps(metadata))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    for kind in ['minihack_room', 'scienceworld']:
        export(args.source, kind, ROOT / 'skills' / kind)
