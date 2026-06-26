import json
from pathlib import Path

splits_dir = Path('data/splits')
raw_dir = Path('data/raw')

CONTAMINATION_PATTERNS = [
    'http', 'www.', '.com', '.org', '.net',
    '#', '[RULES]', '[CLOUD]', '[FILE]', '[INST]', '[DATE]', '[SOUND]', '[DELETED]', '[CONTROL]',
    'instagram', 'reddit', 'twitter', 'facebook',
    'mana', 'health', 'stamina', 'spell', 'weapon', 'dungeon', 'game', 'player',
    'taken-by=', 'p/Q9',
]

def check_file(path):
    clean, contaminated, total = [], [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            msgs = obj.get('messages', [])
            asst = next((m['content'] for m in msgs if m['role'] == 'assistant'), '')

            flags = [p for p in CONTAMINATION_PATTERNS if p.lower() in asst.lower()]
            if flags:
                contaminated.append({'assistant': asst[:120], 'flags': flags})
            else:
                clean.append(asst[:80])
    return total, clean, contaminated

for name in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    path = splits_dir / name
    if path.exists():
        total, clean, dirty = check_file(path)
        print(f'\n--- {name} ---')
        print(f'Total: {total} | Clean: {len(clean)} | Contaminated: {len(dirty)}')
        if dirty:
            print('  Sample contaminated examples:')
            for d in dirty[:3]:
                print('    FLAGS:', d['flags'])
                print('    TEXT: ', d['assistant'])
        else:
            print('  Sample clean examples:')
            for c in clean[:3]:
                print('    ', c)
