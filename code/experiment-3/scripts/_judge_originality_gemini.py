# -*- coding: utf-8 -*-
"""Challenge-aware LLM-judge originality (Gemini) — leg 2 of the triangulation.

For each full-tier conversation, a small ensemble of judge personas scores the
PARTICIPANT's contribution (user-only English text) on 1-7 dimensions, explicitly
RELATIVE TO common solutions for that conversation's specific challenge. Conditioning
on the challenge controls topic by construction, so scores are comparable across
heterogeneous problems. Median across judges per dimension; Welch treatment vs control.

Model: gemini-2.5-flash-lite (higher free-tier limits), temperature 0, JSON schema.
"""
import os, json, csv, time, math, statistics
from collections import defaultdict
from google import genai
from google.genai import types

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
labels = {int(r['conversation_id']): r['challenge_id']
          for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}

CHAL = {
 'galilee_upper':'Upper Galilee: connect older and younger populations to the kibbutz physical/social space.',
 'eshkol_nir_yitzhak':'Eshkol southern kibbutzim cooperation to grow shared community/economic capital.',
 'natal_trauma_language':'NATAL: a language reflecting mental-distress states since Oct 7 for better communication among sufferers, circles, therapists.',
 'sderot_wellbeing':'Sderot: improve mood and sense of meaning of returning residents.',
 'polyron_sleep':'Polyron: new therapeutic sleep products / next-gen mattress / circular economy for trauma & rehab sleep.',
 'ichilov_rehab_future':'Ichilov: the rehabilitation hospital of the future (incl. amputees, combat-injured) a decade+ from now.',
 'joint_rikma_jewish_arab':'Joint Rikma: diverse Jewish-Arab organizations staying functional and culturally sensitive; empathy and inclusive mixed workplaces.',
 'ta_south_community':'Tel Aviv South: make diverse populations feel represented; shared public meeting points connecting neighbors.',
 'ta_east_reut_yad_eliyahu':'Tel Aviv East: Reut rehab hospital leaning reciprocally on the Yad Eliyahu neighborhood; community resilience.',
 'ta_youth_disability_clothing':'Tel Aviv Youth: young people with disabilities and clothing; change the environment to improve their clothing experience.',
 'other_unclear':'No specific challenge (meta/test conversation).',
}
PERSONAS = ['a Design Thinking expert',
            'an innovation and product-strategy expert',
            'a domain expert in this challenge\'s field']

by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def user_text(c, cap=6000):
    return '\n'.join((m.get('message_en') or m['message']).strip()
                     for m in by[c] if m['message_src'] == 'user')[:cap]

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
schema = {"type":"object","properties":{
    "originality_vs_challenge":{"type":"integer"},
    "value_usefulness":{"type":"integer"},
    "holistic_creativity":{"type":"integer"},
    "rationale":{"type":"string"}},
    "required":["originality_vs_challenge","value_usefulness","holistic_creativity","rationale"]}

def judge(c, persona):
    brief = CHAL.get(labels.get(c,'other_unclear'))
    prompt = (
      f"You are {persona} evaluating a hackathon participant's creativity from their own messages "
      f"(user text only; ignore any AI text). Judge substance, not verbosity or grammar.\n\n"
      f"DESIGN CHALLENGE: {brief}\n\n"
      "Rate the participant's contribution on 1-7 scales, RELATIVE TO common/typical solutions for THIS "
      "specific challenge (so topic familiarity is not rewarded):\n"
      "- originality_vs_challenge: 1=cliche/typical for this challenge; 4=moderately novel; 7=rare, compelling reframing.\n"
      "- value_usefulness: 1=impractical; 4=plausible; 7=high-impact, feasible.\n"
      "- holistic_creativity: overall novelty+value+depth for this challenge.\n"
      "Give a <=20-word rationale. Return JSON only.\n\n"
      f"PARTICIPANT TEXT:\n{user_text(c)}"
    )
    r = client.models.generate_content(model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json',
            response_schema=schema, temperature=0))
    return json.loads(r.text)

cids = sorted(by)
rows = []
for i, c in enumerate(cids, 1):
    scores = defaultdict(list)
    for p in PERSONAS:
        for attempt in range(8):
            try:
                d = judge(c, p)
                for k in ('originality_vs_challenge','value_usefulness','holistic_creativity'):
                    scores[k].append(int(d[k]))
                break
            except Exception as e:
                if '429' in repr(e): time.sleep(min(90, 12*(attempt+1)))
                elif attempt >= 3: print('  err', c, repr(e)[:70]); break
                else: time.sleep(5)
        time.sleep(9)  # slow pacing for clean 3/3 coverage
    med = {k: statistics.median(v) for k, v in scores.items() if v}
    rows.append({'conversation_id': c, 'group': by[c][0]['conv_group'],
                 'challenge_id': labels.get(c,'other_unclear'),
                 'orig_vs_challenge': med.get('originality_vs_challenge',''),
                 'value_usefulness': med.get('value_usefulness',''),
                 'holistic_creativity': med.get('holistic_creativity',''),
                 'n_judges': len(scores.get('originality_vs_challenge',[]))})
    print(f'{i:2}/{len(cids)} conv {c} [{rows[-1]["group"][:4]}] '
          f"orig={rows[-1]['orig_vs_challenge']} val={rows[-1]['value_usefulness']} "
          f"hol={rows[-1]['holistic_creativity']} (judges={rows[-1]['n_judges']})", flush=True)
    # incremental save so partial progress survives a mid-run quota cutoff
    with open(f'{BASE}/outputs/judge_originality_gemini.csv','w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

with open(f'{BASE}/outputs/judge_originality_gemini.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def welch(a,b):
    a=[float(x) for x in a if x!='']; b=[float(x) for x in b if x!='']
    na,nb=len(a),len(b)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(a),statistics.mean(b); va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0: return None
    t=(ma-mb)/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); d=(ma-mb)/sp if sp else 0
    return dict(ma=ma,mb=mb,na=na,nb=nb,t=t,df=df,g=d*(1-3/(4*(na+nb)-9)))

print('\n=== challenge-aware judge: treatment vs control (median across judges) ===')
for dim in ('orig_vs_challenge','value_usefulness','holistic_creativity'):
    for excl_other in (False, True):
        rs=[r for r in rows if not (excl_other and r['challenge_id']=='other_unclear')]
        T=[r[dim] for r in rs if r['group']=='treatment' and r[dim]!='']
        C=[r[dim] for r in rs if r['group']=='control' and r[dim]!='']
        w=welch(T,C); tag='(excl other_unclear)' if excl_other else '(all)'
        if w: print(f"  {dim:20} {tag:20} T={w['ma']:.2f}(n{w['na']}) C={w['mb']:.2f}(n{w['nb']}) t={w['t']:.2f} df={w['df']:.1f} g={w['g']:.2f}")
print('\nWrote: outputs/judge_originality_gemini.csv')
