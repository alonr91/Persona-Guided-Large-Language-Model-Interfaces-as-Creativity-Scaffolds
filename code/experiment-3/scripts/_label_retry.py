# -*- coding: utf-8 -*-
"""Re-label only the rows that hit a 429 (ERR fallback), with slow pacing + backoff."""
import os, json, csv, time
from collections import Counter, defaultdict
from google import genai
from google.genai import types
import importlib.util as u

spec = u.spec_from_file_location('lc', os.path.join(os.path.dirname(__file__), '_label_challenges.py'))
# reuse CHALLENGES, schema, by, user_text, brief by re-importing constants
import sys; sys.argv = ['x']  # guard
lc = u.module_from_spec(spec)
# avoid running its __main__ block: it has no guard, so instead redefine minimally here.

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
rows = list(csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8')))

CHALLENGES = {
 'galilee_upper': 'Upper Galilee (Tel-Hai / Kibbutz HaGoshrim): connect older and younger populations to the physical/social kibbutz space.',
 'eshkol_nir_yitzhak': 'Eshkol southern kibbutzim (Nir Yitzhak, Sufa, Holit, Kerem Shalom) cooperation to grow shared community/economic capital.',
 'natal_trauma_language': 'NATAL: build a language reflecting mental-distress states since Oct 7 for better communication among sufferers, circles, therapists.',
 'sderot_wellbeing': 'Sderot: improve mood and sense of meaning of returning residents.',
 'polyron_sleep': 'Polyron (Zikim): new therapeutic sleep products / next-gen mattress / circular economy for trauma & rehab sleep.',
 'ichilov_rehab_future': 'Ichilov: the rehabilitation hospital of the future (incl. amputees, combat-injured) a decade+ from now.',
 'joint_rikma_jewish_arab': 'Joint Rikma: diverse Jewish-Arab organizations, functional continuity, cultural sensitivity, empathy, inclusive mixed workplaces.',
 'ta_south_community': 'Tel Aviv South: make diverse populations feel represented; shared public meeting points connecting neighbors.',
 'ta_east_reut_yad_eliyahu': 'Tel Aviv East: Reut rehab hospital leaning on Yad Eliyahu neighborhood; reciprocal community resilience.',
 'ta_youth_disability_clothing': 'Tel Aviv Youth: young people with disabilities and clothing/dressing; change the environment to improve their clothing experience.',
 'other_unclear': 'Fits none of the above, or a meta/test conversation.',
}
by = defaultdict(list)
for m in msgs: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))
def user_text(ms, cap=3500):
    return '\n'.join((m.get('message_en') or m['message']).strip() for m in ms if m['message_src']=='user')[:cap]
brief = '\n'.join(f'- {k}: {v}' for k, v in CHALLENGES.items())
schema = {"type":"object","properties":{
    "challenge_id":{"type":"string","enum":list(CHALLENGES)},"confidence":{"type":"number"},
    "rationale":{"type":"string"},"evidence_quote":{"type":"string"},
    "runner_up_id":{"type":"string","enum":list(CHALLENGES)}},
    "required":["challenge_id","confidence","rationale","evidence_quote","runner_up_id"]}
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

def label(cid):
    prompt = ("Label which design-thinking hackathon challenge a participant worked on, from their own messages. "
      "Choose the single best challenge_id. Many concern rehabilitation, amputees, PTSD/trauma, or Jewish-Arab "
      "inclusion; map to the most specific brief. Use 'other_unclear' only if it truly fits none. Give confidence "
      "[0,1], one-sentence rationale, a short verbatim evidence quote, and a runner_up_id.\n\n"
      f"CHALLENGES:\n{brief}\n\nPARTICIPANT MESSAGES (conversation {cid}):\n{user_text(by[cid])}")
    r = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json',
            response_schema=schema, temperature=0))
    return json.loads(r.text)

todo = [r for r in rows if r['rationale'].startswith('ERR')]
print('re-labeling', len(todo), 'rows with slow pacing...')
for r in todo:
    cid = int(r['conversation_id'])
    for attempt in range(6):
        try:
            d = label(cid)
            r.update(challenge_id=d['challenge_id'], confidence=round(float(d['confidence']),2),
                     runner_up_id=d['runner_up_id'], rationale=d['rationale'],
                     evidence_quote=d['evidence_quote'][:160])
            print(f'  conv {cid} [{r["group"][:4]}] -> {d["challenge_id"]} ({d["confidence"]:.2f})')
            break
        except Exception as e:
            wait = min(60, 8 * (attempt + 1))
            if '429' not in repr(e) and attempt >= 2:
                print('  conv', cid, 'non-429 err', repr(e)[:80])
            time.sleep(wait)
    time.sleep(4)  # flash-lite has higher RPM

with open(f'{BASE}/outputs/challenge_labels.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('\n=== updated distribution ===')
for k, v in Counter(r['challenge_id'] for r in rows).most_common():
    g = Counter(r['group'] for r in rows if r['challenge_id'] == k)
    print(f'  {k:28} {v:2}  ({dict(g)})')
print('remaining ERR:', [r['conversation_id'] for r in rows if r['rationale'].startswith('ERR')])
