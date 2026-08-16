# -*- coding: utf-8 -*-
"""Assign each full-tier conversation to one of the 10 hackathon challenges (Gemini).

Reads data/experiment3_full_en.json (English), sends each conversation's user-side
text + the 10 challenge briefs to Gemini, and gets a structured label with
confidence, a one-line rationale, an evidence quote, and a runner-up. Writes
outputs/challenge_labels.csv for hand validation.
"""
import os, json, csv, time
from collections import defaultdict, Counter
from google import genai
from google.genai import types

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))

CHALLENGES = {
 'galilee_upper': 'Upper Galilee (Tel-Hai / Kibbutz HaGoshrim): create new connection between older and younger populations and the physical/social kibbutz space; interfaces blending each community\'s social, economic and security needs.',
 'eshkol_nir_yitzhak': 'Eshkol southern kibbutzim (Nir Yitzhak, Sufa, Holit, Kerem Shalom) cooperation: grow shared community and economic capital so the whole is greater than the sum of its parts.',
 'natal_trauma_language': 'NATAL: build a "language" (broadest sense) reflecting the range of mental-distress states in Israeli society since Oct 7, enabling better communication among sufferers, their social circles, and therapists.',
 'sderot_wellbeing': 'Sderot municipality: improve the mood and sense of meaning of returning residents; make them happier and give their lives more meaning now that they are back home.',
 'polyron_sleep': 'Polyron (Kibbutz Zikim): new sleep products; a "therapeutic" sleep kit for post-trauma / poor sleepers or rehab patients; the next-generation mattress; circular economy in the Gaza-envelope region.',
 'ichilov_rehab_future': 'Ichilov: the rehabilitation hospital of the future; tools, treatment methods, medical equipment, operational processes staff will use for rehab patients (incl. amputees, combat-injured) a decade+ from now.',
 'joint_rikma_jewish_arab': 'Joint "Rikma": how diverse Jewish-Arab organizations keep functional continuity and stay culturally sensitive in routine and emergency, increasing empathy and an inclusive, tolerant organizational culture in mixed workplaces.',
 'ta_south_community': 'Tel Aviv South community: make each of South-TA\'s diverse populations feel represented in community/municipal activity; create shared public meeting points that connect neighbors who do not normally meet.',
 'ta_east_reut_yad_eliyahu': 'Tel Aviv East: how Reut rehab hospital can lean on the urban/human infrastructure of the Yad Eliyahu neighborhood it sits in, reciprocally, building community resilience and ties between patients/staff and residents/businesses.',
 'ta_youth_disability_clothing': 'Tel Aviv Youth: young people with disabilities and clothing/dressing (self-image, presence of disability in personal/public space); change the environment (social responsibility) to improve their clothing experience.',
 'other_unclear': 'Does not clearly match any of the above challenges, or is a meta/test conversation.',
}

by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def user_text(ms, cap=3500):
    t = '\n'.join((m.get('message_en') or m['message']).strip()
                  for m in ms if m['message_src'] == 'user')
    return t[:cap]

brief = '\n'.join(f'- {k}: {v}' for k, v in CHALLENGES.items())
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

schema = {
  "type": "object",
  "properties": {
    "challenge_id": {"type": "string", "enum": list(CHALLENGES.keys())},
    "confidence": {"type": "number"},
    "rationale": {"type": "string"},
    "evidence_quote": {"type": "string"},
    "runner_up_id": {"type": "string", "enum": list(CHALLENGES.keys())},
  },
  "required": ["challenge_id", "confidence", "rationale", "evidence_quote", "runner_up_id"],
}

def label(cid):
    prompt = (
      "You are labeling which design-thinking hackathon challenge a participant worked on, "
      "based only on their own messages. Choose the single best-matching challenge_id from the list. "
      "Many conversations concern rehabilitation, amputees, PTSD/trauma, or Jewish-Arab inclusion; "
      "map to the most specific matching brief. Use 'other_unclear' only if it truly fits none. "
      "Give a confidence in [0,1], a one-sentence rationale, a short verbatim evidence quote from the "
      "participant text, and a runner_up_id.\n\n"
      f"CHALLENGES:\n{brief}\n\n"
      f"PARTICIPANT MESSAGES (conversation {cid}):\n{user_text(by[cid])}"
    )
    r = client.models.generate_content(
        model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json', response_schema=schema, temperature=0))
    return json.loads(r.text)

rows = []
cids = sorted(by)
for i, cid in enumerate(cids, 1):
    grp = by[cid][0]['conv_group']
    for attempt in range(4):
        try:
            d = label(cid); break
        except Exception as e:
            if attempt == 3:
                d = {"challenge_id": "other_unclear", "confidence": 0, "rationale": f"ERR {e!r}"[:80],
                     "evidence_quote": "", "runner_up_id": "other_unclear"}
            time.sleep(2 * (attempt + 1))
    rows.append({"conversation_id": cid, "group": grp, "challenge_id": d["challenge_id"],
                 "confidence": round(float(d["confidence"]), 2), "runner_up_id": d["runner_up_id"],
                 "rationale": d["rationale"], "evidence_quote": d["evidence_quote"][:160]})
    print(f'{i:2}/{len(cids)}  conv {cid} [{grp[:4]}] -> {d["challenge_id"]} ({d["confidence"]:.2f})')
    time.sleep(0.2)

with open(f'{BASE}/outputs/challenge_labels.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print('\n=== challenge distribution ===')
for k, v in Counter(r['challenge_id'] for r in rows).most_common():
    g = Counter(r['group'] for r in rows if r['challenge_id'] == k)
    print(f'  {k:28} {v:2}  ({dict(g)})')
print('low-confidence (<0.6):', [r['conversation_id'] for r in rows if r['confidence'] < 0.6])
print('\nWrote: outputs/challenge_labels.csv')
