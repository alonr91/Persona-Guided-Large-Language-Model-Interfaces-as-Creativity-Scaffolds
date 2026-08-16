# -*- coding: utf-8 -*-
"""Regenerate readable transcripts from the authoritative organized data.

Reads data/experiment3_messages.json (enriched) and writes grouped, human-
readable transcripts with a metadata header per conversation. English text is
shown when a translation exists, with the original underneath when translated.
"""
import json, csv
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_messages.json', encoding='utf-8'))
meta = {int(r['conversation_id']): r for r in
        csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}

byc = defaultdict(list)
for m in msgs:
    byc[m['conversation_id']].append(m)
for cid in byc:
    byc[cid].sort(key=lambda x: (x['timestamp'], x['message_id']))

ROLE = {'treatment': {'Taylor': 'Taylor (divergent)', 'Alex': 'Alex (convergent)'},
        'control':   {'Taylor': 'Taylor (control-LLM)', 'Alex': 'Alex (control-LLM)'},
        'mixed':     {'Taylor': 'Taylor', 'Alex': 'Alex'}}

def render(cid):
    ms = byc[cid]; r = meta[cid]; lines = []
    lines.append(f'## Conversation {cid}  [{r["group"]} · tier={r["quality_tier"]}]')
    lines.append(f'_{r["first_ts"]} → {r["last_ts"]}  |  {r["n_msgs"]} msgs '
                 f'({r["n_user"]} user) · {r["n_days"]} day(s) · {r["duration_minutes"]} min · lang={r["lang"]}_')
    lines.append('')
    for m in ms:
        if m['message_src'] == 'user':
            who = f'USER → {m["persona"]}'
        else:
            who = ROLE[m['conv_group']][m['persona']]
        en = m.get('message_en')
        if en and m['original_lang'] == 'he':
            lines.append(f'**[{m["timestamp"]}] {who} [HE→EN]:** {en}')
            lines.append(f'<sub>{m["message"]}</sub>')
        else:
            lines.append(f'**[{m["timestamp"]}] {who}:** {m.get("message_en") or m["message"]}')
        lines.append('')
    lines.append('---\n')
    return '\n'.join(lines)

def write_group(path, group, title):
    cids = sorted(c for c in byc if meta[c]['group'] == group)
    body = [f'# {title}', f'\nConversations: {len(cids)}\n', '---\n']
    body += [render(c) for c in cids]
    open(path, 'w', encoding='utf-8').write('\n'.join(body))
    return len(cids)

# all conversations, ordered
allcids = sorted(byc)
allbody = ['# Experiment 3 — All Hackathon Transcripts (10–13 March 2025)',
           f'\nConversations: {len(allcids)}  |  Messages: {len(msgs)}\n', '---\n']
allbody += [render(c) for c in allcids]
open(f'{BASE}/transcripts/experiment3_transcripts.md', 'w', encoding='utf-8').write('\n'.join(allbody))

nt = write_group(f'{BASE}/transcripts/transcripts_treatment.md', 'treatment',
                 'Experiment 3 — Treatment (Taylor=divergent, Alex=convergent)')
nc = write_group(f'{BASE}/transcripts/transcripts_control.md', 'control',
                 'Experiment 3 — Control (Taylor & Alex = plain LLM)')
nm = write_group(f'{BASE}/transcripts/transcripts_mixed.md', 'mixed',
                 'Experiment 3 — Mixed-group (persona routes span both conditions)')
print(f'Wrote transcripts: all={len(allcids)}, treatment={nt}, control={nc}, mixed={nm}')
