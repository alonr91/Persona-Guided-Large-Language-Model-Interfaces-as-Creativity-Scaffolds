"""Generate a calibration sample: 60 assistant turns stratified across cells.

Stratification:
  - 20 treatment-Taylor turns
  - 20 treatment-Alex turns
  - 10 control-Taylor turns
  - 10 control-Alex turns

Each turn is selected for moderate length (50-1500 chars) to be ratable.
Output: calibration_sample.json — to be hand-rated (by Claude as the rater)
on 1-5 scale: 1=strongly divergent, 5=strongly convergent.
"""
import json, random
from collections import defaultdict

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
with open(f'{base}/data/experiment3_messages_en.json','r',encoding='utf-8') as f:
    msgs = json.load(f)
with open(f'{base}/outputs/stance_per_message.json','r',encoding='utf-8') as f:
    stance = {(r['conversation_id'], r['message_id']): r for r in json.load(f)}

# Filter: assistant turns with moderate length and a stance score
def cell_key(m):
    if m['message_src']!='assistant': return None
    if not (50 <= len(m['message_en']) <= 1500): return None
    g = m['group']; r = m['agent_role']
    return (g, r)

cells = defaultdict(list)
for m in msgs:
    k = cell_key(m)
    if k: cells[k].append(m)

random.seed(42)
sample = []
for k, n in [(('treatment','Taylor'),20), (('treatment','Alex'),20),
             (('control','Taylor'),10),  (('control','Alex'),10)]:
    pool = cells[k]
    if len(pool) < n:
        print(f'WARNING: only {len(pool)} available for {k} — taking all')
        chosen = pool
    else:
        chosen = random.sample(pool, n)
    for m in chosen:
        s = stance.get((m['conversation_id'], m['message_id']))
        sample.append({
            'rate_id': len(sample)+1,
            'conversation_id': m['conversation_id'],
            'message_id': m['message_id'],
            'group': m['group'],
            'agent_role': m['agent_role'],
            'persona_id': m['persona_id'],
            'text': m['message_en'][:1500],
            'nli_divergent': s['divergent_score'] if s else None,
            'nli_convergent': s['convergent_score'] if s else None,
            'nli_d_minus_c': s['d_minus_c'] if s else None,
        })

random.shuffle(sample)  # blind ordering — labeler won't see persona/group correlated with rate_id
for i,r in enumerate(sample, 1):
    r['rate_id'] = i

with open(f'{base}/outputs/calibration_sample.json','w',encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
print(f'Wrote {len(sample)} sample turns to calibration_sample.json')
