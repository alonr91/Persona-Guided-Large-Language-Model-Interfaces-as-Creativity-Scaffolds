import json, csv, re, os
from collections import defaultdict, Counter
from datetime import datetime

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
with open(f'{base}/data/experiment3_messages.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

by_conv = defaultdict(list)
for m in msgs:
    by_conv[m['conversation_id']].append(m)
for cid in by_conv:
    by_conv[cid].sort(key=lambda x:(x['timestamp'], x['message_id']))

GREETING_RE = re.compile(r'^\s*(hi|hello|hey|היי|שלום|מה קורה|מה נשמע|check|test|בדיקה|hh+|he|sdasd|\d+|\W+)\s*$', re.IGNORECASE)
def is_greeting_or_junk(text):
    t = (text or '').strip()
    if len(t) < 4: return True
    if GREETING_RE.match(t): return True
    return False

def conv_group(ms):
    gs = set(m['group'] for m in ms)
    if gs == {'treatment'}: return 'treatment'
    if gs == {'control'}: return 'control'
    return 'mixed'

OFFTOPIC = {609}  # bears

excluded = {}
kept = {}

for cid, ms in by_conv.items():
    user_msgs = [m for m in ms if m['message_src']=='user']
    n_user = len(user_msgs)
    user_texts = [m['message'].strip() for m in user_msgs]
    unique_user_texts = set(user_texts)
    g = conv_group(ms)

    if g == 'mixed':
        excluded[cid] = ('mixed_group', 'personas span treatment+control'); continue
    if cid in OFFTOPIC:
        excluded[cid] = ('off_topic', 'unrelated to any challenge'); continue
    if n_user >= 1 and all(is_greeting_or_junk(t) for t in user_texts):
        excluded[cid] = ('greeting_or_junk', f'all {n_user} user turn(s) greeting/junk'); continue
    if n_user <= 1:
        excluded[cid] = ('single_turn', f'{n_user} user turn'); continue
    if n_user == 2 and len(unique_user_texts) == 1:
        excluded[cid] = ('duplicate_paste_no_followup', 'same prompt to both personas, no follow-up'); continue
    if n_user == 2 and is_greeting_or_junk(user_texts[1]):
        excluded[cid] = ('no_real_iteration', '2 user turns, second is greeting/short'); continue
    substantive = [t for t in user_texts if not is_greeting_or_junk(t)]
    if len(substantive) < 2:
        excluded[cid] = ('insufficient_iteration', f'{len(substantive)} substantive user turn(s)'); continue
    kept[cid] = ms

print(f'TOTAL: {len(by_conv)}  KEPT: {len(kept)}  EXCLUDED: {len(excluded)}')
print()
print('Exclusion reasons:')
for r,c in Counter(v[0] for v in excluded.values()).most_common():
    print(f'  {r}: {c}')
print()
print('Excluded conversations:')
for cid in sorted(excluded):
    ms = by_conv[cid]
    n_user = sum(1 for m in ms if m['message_src']=='user')
    first_user = next((m['message'][:70] for m in ms if m['message_src']=='user'), '')
    r,d = excluded[cid]
    print(f'  {cid:4d} [{conv_group(ms):9s}] u={n_user}  {r:30s}  "{first_user}"')

# Outputs
out_clean = []
for cid in sorted(kept):
    out_clean.extend(kept[cid])
with open(f'{base}/data/experiment3_messages_clean.json','w',encoding='utf-8') as f:
    json.dump(out_clean, f, ensure_ascii=False, indent=2)

with open(f'{base}/outputs/experiment3_exclusion_log.csv','w',encoding='utf-8',newline='') as f:
    w = csv.writer(f)
    w.writerow(['conversation_id','group','n_total_msgs','n_user_msgs','reason','detail','first_user_msg','first_timestamp'])
    for cid in sorted(excluded):
        ms = by_conv[cid]
        n_user = sum(1 for m in ms if m['message_src']=='user')
        first_user = next((m['message'][:200] for m in ms if m['message_src']=='user'), '')
        r,d = excluded[cid]
        w.writerow([cid, conv_group(ms), len(ms), n_user, r, d, first_user, ms[0]['timestamp']])

with open(f'{base}/outputs/experiment3_conversations_clean_summary.csv','w',encoding='utf-8',newline='') as f:
    w = csv.writer(f)
    w.writerow(['conversation_id','group','total_msgs','user_msgs','assistant_msgs','personas_used','first_timestamp','last_timestamp','duration_minutes','date'])
    for cid in sorted(kept):
        ms = kept[cid]
        u = sum(1 for m in ms if m['message_src']=='user')
        a = sum(1 for m in ms if m['message_src']=='assistant')
        personas = sorted(set(str(m['persona_id']) for m in ms))
        t0 = datetime.strptime(ms[0]['timestamp'],'%Y-%m-%d %H:%M:%S')
        t1 = datetime.strptime(ms[-1]['timestamp'],'%Y-%m-%d %H:%M:%S')
        dur = round((t1-t0).total_seconds()/60,1)
        w.writerow([cid, conv_group(ms), len(ms), u, a, ','.join(personas), ms[0]['timestamp'], ms[-1]['timestamp'], dur, ms[0]['timestamp'][:10]])

PERSONA_NAMES = {'1':'Taylor (divergent)','2':'Alex (convergent)','3':'Taylor (control-LLM)','4':'Alex (control-LLM)'}
def write_transcripts(path, cids, title):
    with open(path,'w',encoding='utf-8') as f:
        f.write(f'# {title}\n\nConversations: {len(cids)}\n\n---\n\n')
        for cid in sorted(cids):
            ms = kept[cid]
            f.write(f'## Conversation {cid}  [{conv_group(ms)}]\n')
            f.write(f'_{ms[0]["timestamp"]} → {ms[-1]["timestamp"]}  |  {len(ms)} messages_\n\n')
            for m in ms:
                pid = str(m.get('persona_id'))
                who = 'USER' if m['message_src']=='user' else PERSONA_NAMES.get(pid, f'persona{pid}')
                f.write(f'**[{m["timestamp"]}] {who}:** {m["message"]}\n\n')
            f.write('\n---\n\n')

treatment_kept = [c for c in kept if conv_group(kept[c])=='treatment']
control_kept = [c for c in kept if conv_group(kept[c])=='control']
write_transcripts(f'{base}/transcripts/transcripts_treatment_clean.md', treatment_kept, 'Experiment 3 — Treatment (CLEANED): Taylor=divergent, Alex=convergent')
write_transcripts(f'{base}/transcripts/transcripts_control_clean.md', control_kept, 'Experiment 3 — Control (CLEANED): regular LLM, no personality')

g_kept = Counter(conv_group(kept[c]) for c in kept)
print(f'\nKept by group: {dict(g_kept)}')
print(f'Kept messages: {sum(len(kept[c]) for c in kept)} of {sum(len(by_conv[c]) for c in by_conv)}')
print(f'\nFiles:')
for fn in sorted(os.listdir(base)):
    p = f'{base}/{fn}'
    print(f'  {fn}  ({os.path.getsize(p):,} bytes)')
