# -*- coding: utf-8 -*-
"""Authoritative organization of Experiment 3 data from the raw message export.

Single source of truth: data/raw/messages_full_export.json  (the full platform
export the user provided, "messages2 (1).json", 6330 messages Oct-2024..Aug-2025).

This script:
  1. Filters to the hackathon window 2025-03-10 .. 2025-03-13 (inclusive).
     Conversations may span many hours/days WITHIN the window; only stray
     messages from reused conversation threads months later are dropped.
  2. Enriches every message with the group mapping supplied by the user:
       persona_id 1,2 -> treatment ;  persona_id 3,4 -> control
       persona_id 1,3 -> Taylor    ;  persona_id 2,4 -> Alex
     (In treatment, Taylor=divergent / Alex=convergent. In control both are a
      plain LLM with no persona traits.)
  3. Marks each conversation's group (treatment / control / mixed) where 'mixed'
     means its persona_ids span both routes (cannot be assigned to a condition).
  4. Carries over existing English translations by message_id, and flags the
     original language (he/en) for the rest (Hebrew detected by codepoint).
  5. Writes an enriched message file + a per-conversation summary with quality
     metadata and an include recommendation ("not all conversations are equal").

Outputs:
  data/experiment3_messages.json
  outputs/experiment3_conversations_summary.csv
"""
import json, csv, re
from collections import defaultdict, Counter
from datetime import datetime

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
RAW  = f'{BASE}/data/raw/messages_full_export.json'
EN   = f'{BASE}/data/experiment3_messages_en.json'   # existing translations (optional)

WIN_START, WIN_END = '2025-03-10', '2025-03-13'      # inclusive calendar days

# ---- mapping ----
def route_group(pid):  return 'treatment' if str(pid) in ('1', '2') else 'control'
def persona(pid):      return 'Taylor' if str(pid) in ('1', '3') else 'Alex'

HEB = re.compile(r'[֐-׿]')
def detect_lang(t): return 'he' if t and HEB.search(t) else 'en'

GREETING = re.compile(
    r'^\s*(hi|hii+|hello|hey+|yo+|sup|ok|okay|thanks|thank you|test|check|'
    r'היי|שלום|מה קורה|מה נשמע|בדיקה|hh+|he|sdasd|asdf+|\d+|\W+)\s*$', re.I)
def is_junk(t):
    t = (t or '').strip()
    return len(t) < 4 or bool(GREETING.match(t))

# ---- load ----
raw = json.load(open(RAW, encoding='utf-8'))
try:
    en = {m['message_id']: m for m in json.load(open(EN, encoding='utf-8'))}
except FileNotFoundError:
    en = {}

def in_window(t): return t and WIN_START <= t[:10] <= WIN_END
win = [m for m in raw if in_window(m['timestamp'])]

# ---- enrich messages ----
out = []
for m in win:
    pid = m['persona_id']
    tr = en.get(m['message_id'], {})
    msg_en = tr.get('message_en')
    lang = tr.get('original_lang') or detect_lang(m['message'])
    out.append({
        'message_id': m['message_id'],
        'conversation_id': m['conversation_id'],
        'timestamp': m['timestamp'],
        'message_src': m['message_src'],            # user | assistant
        'persona_id': int(pid),
        'persona': persona(pid),                    # Taylor | Alex
        'route_group': route_group(pid),            # treatment | control (this message's route)
        'message': m['message'],
        'message_en': msg_en,                       # None if not yet translated
        'original_lang': lang,                       # he | en
    })

# ---- conversation grouping + carry conv_group onto each message ----
byc = defaultdict(list)
for m in out:
    byc[m['conversation_id']].append(m)
for cid in byc:
    byc[cid].sort(key=lambda x: (x['timestamp'], x['message_id']))

conv_group = {}
for cid, ms in byc.items():
    routes = set(x['route_group'] for x in ms)
    conv_group[cid] = ('mixed' if len(routes) > 1 else routes.pop())
for m in out:
    m['conv_group'] = conv_group[m['conversation_id']]

# ---- per-conversation summary with quality metadata ----
def minutes(ms):
    fmt = '%Y-%m-%d %H:%M:%S'
    ts = [datetime.strptime(x['timestamp'], fmt) for x in ms]
    return round((max(ts) - min(ts)).total_seconds() / 60, 1)

rows = []
for cid, ms in sorted(byc.items()):
    users = [x for x in ms if x['message_src'] == 'user']
    asts  = [x for x in ms if x['message_src'] == 'assistant']
    subst = [x for x in users if not is_junk(x['message'])]
    days  = sorted(set(x['timestamp'][:10] for x in ms))
    langs = set(x['original_lang'] for x in users) or {'en'}
    lang  = 'mixed' if len(langs) > 1 else next(iter(langs))
    end   = asts[-1]['persona'] if asts else (users[-1]['persona'] if users else '')
    grp   = conv_group[cid]
    ns    = len(subst)   # substantive (non-junk) user turns

    # Graded quality tier ("not all conversations are equal").
    # We do NOT hard-prune here — analysis stage chooses a threshold.
    #   mixed   : persona routes span both conditions -> unassignable
    #   junk    : 0 substantive user turns (greeting/test/single char only)
    #   minimal : exactly 1 substantive user turn
    #   short   : 2-3 substantive user turns (usable for some analyses)
    #   full    : >=4 substantive user turns (usable for all analyses)
    if grp == 'mixed':         tier = 'mixed'
    elif ns == 0:              tier = 'junk'
    elif ns == 1:              tier = 'minimal'
    elif ns <= 3:              tier = 'short'
    else:                      tier = 'full'
    # lenient default sample used downstream: real, single-condition, >=2 turns
    analysable = tier in ('short', 'full')

    rows.append({
        'conversation_id': cid, 'group': grp,
        'n_msgs': len(ms), 'n_user': len(users), 'n_assistant': len(asts),
        'n_user_substantive': ns,
        'msgs_to_Taylor': sum(1 for x in users if x['persona'] == 'Taylor'),
        'msgs_to_Alex':   sum(1 for x in users if x['persona'] == 'Alex'),
        'ending_persona': end,
        'first_ts': ms[0]['timestamp'], 'last_ts': ms[-1]['timestamp'],
        'duration_minutes': minutes(ms),
        'n_days': len(days), 'multiday': len(days) > 1,
        'lang': lang,
        'quality_tier': tier, 'analysable': analysable,
    })

# ---- write ----
json.dump(out, open(f'{BASE}/data/experiment3_messages.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
with open(f'{BASE}/outputs/experiment3_conversations_summary.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- console report ----
print('RAW total messages           :', len(raw))
print('In-window messages (10-13/3) :', len(out))
print('Conversations                :', len(byc))
print('Group composition (convs)    :', dict(Counter(conv_group.values())))
print('Quality tiers                :', dict(Counter(r['quality_tier'] for r in rows)))
inc = [r for r in rows if r['analysable']]
print('Analysable (>=2 subst, single-cond):', len(inc),
      dict(Counter(r['group'] for r in inc)))
print('  of which tier=full (>=4 turns)   :',
      len([r for r in inc if r['quality_tier'] == 'full']),
      dict(Counter(r['group'] for r in inc if r['quality_tier'] == 'full')))
print('Analysable messages          :', sum(r['n_msgs'] for r in inc))
print('Multi-day conversations      :', sum(1 for r in rows if r['multiday']))
print('Language (user side) of convs:', dict(Counter(r['lang'] for r in rows)))
print('Translations carried over    :', sum(1 for m in out if m['message_en']), '/', len(out))
print('\nWrote: data/experiment3_messages.json')
print('Wrote: outputs/experiment3_conversations_summary.csv')
