import urllib.request, json, time

url = 'http://127.0.0.1:5000/api/chat'
tests = [
    ('TYPE A - Fragment/vague', [{'role': 'user', 'content': 'just tired.'}]),
    ('TYPE B - Medium/emotional', [{'role': 'user', 'content': 'My best friend completely ignored me when I needed her most.'}]),
    ('TYPE C - Long vent', [{'role': 'user', 'content': 'My relationship is falling apart and I cannot talk to anyone about it because we have mutual friends and anything I say gets back to him and I am so exhausted from pretending everything is fine every single day.'}]),
    ('TYPE D - User asks Fumii', [{'role': 'user', 'content': 'is it weird that I feel relieved and sad at the same time?'}]),
    ('TYPE A2 - Single word', [{'role': 'user', 'content': 'idk.'}]),
    ('TYPE E - Anger/Frustration', [{'role': 'user', 'content': 'I am so incredibly angry at my boss right now, I just want to quit!.'}]),
    ('TYPE F - Existential/Lost', [{'role': 'user', 'content': 'sometimes I wonder what the point of all this is.'}]),
    ('TYPE G - Positive/Happy', [{'role': 'user', 'content': 'I finally passed my exam after failing twice! I am so happy!.'}]),
    ('TYPE H - Anxious/Overwhelmed', [{'role': 'user', 'content': 'My chest is tight and I have so much to do and I don\'t know where to start.'}]),
    ('TYPE I - Grief/Loss', [{'role': 'user', 'content': 'I miss him so much. It feels like an ache that just won\'t go away.'}]),
]

for label, messages in tests:
    req = urllib.request.Request(
        url,
        method='POST',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'messages': messages}).encode()
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    reply = resp.get('response', resp.get('reply', ''))
    q_count = reply.count('?')
    parts = [s.strip() for s in reply.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    s_count = len(parts)
    user_txt = messages[-1]['content']
    print(f'\n[{label}]')
    print(f'  User ({len(user_txt.split())} words): {user_txt[:90]}')
    print(f'  Fumii: {reply}')
    print(f'  >> sentences={s_count}  questions={q_count}')
    time.sleep(1)

print('\nDone.')
