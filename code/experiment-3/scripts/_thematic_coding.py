# -*- coding: utf-8 -*-
"""Coding pass for a thematic analysis contrasting user interaction with the
divergent (Taylor) vs convergent (Alex) personas — treatment, long sample.

Each USER turn is tagged by the persona addressed and by request-type codes
(keyword heuristics, multi-label). We report, per persona, how often each code
appears, to ground the qualitative themes. Heuristics are indicative; themes are
validated against verbatim quotes separately.
"""
import json, csv, re
from collections import defaultdict, Counter

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
treat = [c for c in LONG if LONG[c] == 'treatment']
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
by = defaultdict(list)
for m in msgs:
    if m['conversation_id'] in treat: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

CODES = {
 'GENERATE_more'   : r'(more idea|give me \d|\d+ idea|another idea|other idea|additional|more solution|brainstorm|suggest .*idea|give me ideas|any other|more ideas|10 more|100 idea)',
 'NOVELTY_push'    : r'(creativ|innovativ|out of the box|unique|no one has|not good enough|not .*enough|really creative|best and most|originaln|something new|that .*exist)',
 'EXPAND_elaborate': r'(expand|elaborat|develop on|in more detail|explain|go deeper|continue|tell me more|wider)',
 'EVALUATE_select' : r'(\brate\b|best option|which is|which one|feasib|effectiv|relevan|compare|\bpros\b|\bcons\b|better|worth|prioriti|rank|evaluate|what do you think)',
 'SPECIFY_technical':r'(what device|which device|where .*(place|sensor|locat)|distance|material|cost|budget|which type|what kind|how would .*(work|look)|how does|how will|look like|specifications?|technical)',
 'STRUCTURE_plan'  : r'(\bplan\b|step by step|step-by-step|\bsteps\b|structur|guideline|framework|protocol|outline|summar|shorten|nutshell|organi[sz]e)',
 'REALITY_check'   : r'(does .*exist|already exist|is it new|is it possible|is this possible|possible without|can you actually|is it feasible)',
 'PRODUCE_artifact': r'(make .*(picture|image|sketch|photo|prototype)|generate .*(photo|image|picture)|build me|send (it|me)|email it|canva|jpeg|draw)',
 'BROKER_persona'  : r'(\btaylor\b|\balex\b)',
}
CODES = {k: re.compile(v, re.I) for k, v in CODES.items()}

per = {'Taylor': Counter(), 'Alex': Counter()}
nturns = Counter()
examples = defaultdict(lambda: defaultdict(list))
for c in by:
    for m in by[c]:
        if m['message_src'] != 'user': continue
        p = m['persona']; t = (m.get('message_en') or m['message'] or '').strip()
        nturns[p] += 1
        for code, rx in CODES.items():
            if rx.search(t):
                per[p][code] += 1
                if len(examples[p][code]) < 4:
                    examples[p][code].append(f"c{c}: {t[:95]}")

print(f"User turns addressed: Taylor(divergent)={nturns['Taylor']}, Alex(convergent)={nturns['Alex']}\n")
print(f"{'code':18} {'Taylor %':>9} {'Alex %':>9}   (share of that persona's user turns)")
print('-'*52)
for code in CODES:
    tp = 100*per['Taylor'][code]/nturns['Taylor']
    ap = 100*per['Alex'][code]/nturns['Alex']
    flag = '  <= DIVERGENT' if tp-ap>8 else ('  <= CONVERGENT' if ap-tp>8 else '')
    print(f"{code:18} {tp:8.0f}% {ap:8.0f}%{flag}")

print('\n--- sample turns by code (Alex / convergent) ---')
for code in ['EVALUATE_select','SPECIFY_technical','STRUCTURE_plan','REALITY_check']:
    for e in examples['Alex'][code][:3]: print(f"  [{code}] {e}")
print('\n--- sample turns by code (Taylor / divergent) ---')
for code in ['GENERATE_more','NOVELTY_push','EXPAND_elaborate']:
    for e in examples['Taylor'][code][:3]: print(f"  [{code}] {e}")
