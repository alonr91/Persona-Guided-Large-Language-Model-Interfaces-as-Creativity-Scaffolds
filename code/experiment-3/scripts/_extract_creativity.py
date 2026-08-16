# -*- coding: utf-8 -*-
"""Per-conversation creativity measures via Gemini, for the within-treatment analysis.

ONE call per conversation (efficient): from the participant's user-only text and the
conversation's challenge brief, extract QUANTITATIVE creativity (fluency = number of
distinct actionable ideas; flexibility = number of distinct idea categories) and a
challenge-relative QUALITY rating (originality 1-7, holistic creativity 1-7).

Runs on all full-tier conversations (treatment + control) so the measure is general,
but the dose-response analysis uses treatment only. Saves incrementally.
Model: gemini-2.5-flash (other tiers were quota-exhausted). Paced + 429 backoff.
"""
import os, json, csv, time
from collections import defaultdict
from google import genai
from google.genai import types

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
labels = {int(r['conversation_id']): r['challenge_id'] for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}
CHAL = {
 'galilee_upper':'Upper Galilee: connect older and younger populations to the kibbutz space.',
 'eshkol_nir_yitzhak':'Eshkol southern kibbutzim cooperation for shared community/economic capital.',
 'natal_trauma_language':'NATAL: a language for mental-distress states since Oct 7.',
 'sderot_wellbeing':'Sderot: improve mood and meaning of returning residents.',
 'polyron_sleep':'Polyron: therapeutic sleep products / next-gen mattress.',
 'ichilov_rehab_future':'Ichilov: the rehabilitation hospital of the future (amputees, combat-injured).',
 'joint_rikma_jewish_arab':'Joint Rikma: Jewish-Arab organizations, empathy, inclusive mixed workplaces.',
 'ta_south_community':'Tel Aviv South: diverse populations represented; shared public meeting points.',
 'ta_east_reut_yad_eliyahu':'Tel Aviv East: Reut rehab hospital and Yad Eliyahu neighborhood resilience.',
 'ta_youth_disability_clothing':'Tel Aviv Youth: young people with disabilities and clothing.',
 'other_unclear':'No specific challenge (meta/test).',
}
by = defaultdict(list)
for m in msgs: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))
def utext(c, cap=7000):
    return '\n'.join((m.get('message_en') or m['message']).strip() for m in by[c] if m['message_src']=='user')[:cap]

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
schema = {"type":"object","properties":{
    "fluency_idea_count":{"type":"integer"},
    "flexibility_category_count":{"type":"integer"},
    "originality_vs_challenge":{"type":"integer"},
    "holistic_creativity":{"type":"integer"},
    "categories":{"type":"array","items":{"type":"string"}}},
    "required":["fluency_idea_count","flexibility_category_count","originality_vs_challenge","holistic_creativity"]}

def extract(c):
    brief = CHAL.get(labels.get(c,'other_unclear'))
    prompt = (
      "From the participant's own messages below (user text only; ignore any AI text), measure their "
      "creative output for this design challenge.\n\n"
      f"CHALLENGE: {brief}\n\n"
      "Return JSON:\n"
      "- fluency_idea_count: number of DISTINCT, actionable solution ideas the participant proposed "
      "(merge near-duplicates; count only the user's own ideas).\n"
      "- flexibility_category_count: number of distinct conceptual categories those ideas span.\n"
      "- categories: short names of those categories.\n"
      "- originality_vs_challenge (1-7): novelty of the ideas RELATIVE TO common solutions for THIS challenge "
      "(1=typical, 4=moderate, 7=rare/reframing).\n"
      "- holistic_creativity (1-7): overall creativity for this challenge.\n\n"
      f"PARTICIPANT TEXT:\n{utext(c)}"
    )
    r = client.models.generate_content(model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json', response_schema=schema, temperature=0))
    return json.loads(r.text)

# resume support: keep existing rows
out_path = f'{BASE}/outputs/creativity_per_conv.csv'
done = {}
if os.path.exists(out_path):
    for r in csv.DictReader(open(out_path, encoding='utf-8')):
        done[int(r['conversation_id'])] = r

cids = sorted(by)
rows = []
for i, c in enumerate(cids, 1):
    if c in done and done[c].get('fluency_idea_count') not in ('', None):
        rows.append(done[c]); print(f'{i:2}/{len(cids)} conv {c} cached'); continue
    grp = by[c][0]['conv_group']
    rec = {'conversation_id': c, 'group': grp, 'challenge_id': labels.get(c,'other_unclear'),
           'fluency_idea_count':'', 'flexibility_category_count':'', 'originality_vs_challenge':'', 'holistic_creativity':''}
    for attempt in range(8):
        try:
            d = extract(c)
            rec.update(fluency_idea_count=d['fluency_idea_count'], flexibility_category_count=d['flexibility_category_count'],
                       originality_vs_challenge=d['originality_vs_challenge'], holistic_creativity=d['holistic_creativity'])
            break
        except Exception as e:
            if '429' in repr(e): time.sleep(min(90, 12*(attempt+1)))
            elif attempt >= 3: print('  err', c, repr(e)[:70]); break
            else: time.sleep(5)
    rows.append(rec)
    print(f"{i:2}/{len(cids)} conv {c} [{grp[:4]}] fluency={rec['fluency_idea_count']} flex={rec['flexibility_category_count']} "
          f"orig={rec['originality_vs_challenge']} hol={rec['holistic_creativity']}", flush=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    time.sleep(6)
print('Wrote:', out_path)
