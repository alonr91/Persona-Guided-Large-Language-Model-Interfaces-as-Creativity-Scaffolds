# -*- coding: utf-8 -*-
"""USER-ONLY creative-behaviour rubric for Experiment 3 — PORTED FROM EXPERIMENT 1.

Faithful port of Experiment 1's "Option B" user-behaviour rubric
(os_pipeline/regulated/user_rubric.py + user_scorer.py): six 0-4 anchored
criteria targeting the USER's creative behaviour. The scorer reads the full
masked transcript for context but rates ONLY the user's contribution, and every
non-null score must quote a USER turn (assistant echoes do not count as user
initiative/proposals). Reported as a §2.10-bounded proxy of user creative
behaviour, not externally judged creativity.

Adaptations for Exp 3 (consistent with the dialogic-rubric port):
  - Two paraphrased scorers A (strict, verbatim Exp 1 prompt) + B (paraphrased)
    on gemini-3.1-flash-lite (temp 0.15); Exp 1 ran a single user scorer, so the
    A-B leg here is an added prompt-robustness check.
  - Conversation level (whole masked transcript), clean sample of 18.
Resumable; paced for the API rate limit.
"""
import os, json, csv, re, time
from collections import defaultdict
from google import genai
from google.genai import types

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
OUT_RAW = f'{BASE}/outputs/user_rubric_raw.csv'
MODEL = 'gemini-3.1-flash-lite'

# ---------- 6 user-behaviour criteria (verbatim from Exp 1 user_rubric.py) ----------
CRITERIA = [
 ('user_initiative', 'How proactively does the USER introduce new directions, problems, or ideas without being prompted?',
  'User-side proactivity. Score what the USER originates (not what they accept from the assistant). Echoes of the assistant do NOT count.',
  {0:'User is fully passive; one-word answers, never introduces new content.',1:'User responds substantively but does not introduce new directions.',2:'User occasionally introduces a new direction or substantive follow-up.',3:'User regularly initiates new directions or proposes new content.',4:'User drives the agenda; introduces multiple distinct directions and connects them.'}),
 ('user_question_richness', 'When the USER asks questions, how substantive and probing are they versus surface clarifications?',
  'Quality of user questioning. Mere "what?" or "can you repeat?" is surface; questions that test ideas, open new angles, or probe assumptions are rich.',
  {0:'User asks no questions.',1:'User asks only surface clarifications.',2:'User asks specific clarifying questions about details.',3:'User asks probing questions that test ideas or open new angles.',4:'User asks deeply probing or reframing questions that reshape the conversation.'}),
 ('user_proposal_specificity', 'When the USER proposes ideas, how concrete and specified are they?',
  'Concreteness of user-originated proposals. Vague slogans low; mechanisms / examples / actors / venues high.',
  {0:'User proposes nothing.',1:'Proposals are vague slogans or restatements.',2:'Proposals have one concrete element.',3:'Proposals are concrete with at least one specific mechanism or example.',4:'Proposals are highly concrete and multi-element (mechanisms, examples, actors, or venues).'}),
 ('user_acceptance_yes_and', "Does the USER build on the assistant's contributions (yes-and) versus ignore or block them?",
  'Yes-and uptake on the user side. Acknowledgement without building is low; explicit extension/integration is high.',
  {0:'User ignores the assistant entirely.',1:'User acknowledges but does not build (e.g. "ok").',2:'User accepts and adds tangential content.',3:"User accepts and explicitly extends the assistant's contribution.",4:'User strongly yes-ands; integrates and builds on assistant content into a unified thread.'}),
 ('user_reframing', 'Does the USER reframe the problem, take a new angle, or challenge assumptions?',
  "User-initiated reframing. Staying inside the assistant's framing low; substantively challenging or reformulating high.",
  {0:"User stays strictly within the assistant's framing.",1:'User makes minor adjustments to the framing.',2:'User offers an alternative angle once.',3:'User offers multiple reframings or significant alternative angles.',4:'User actively challenges or substantially reframes the problem.'}),
 ('user_engagement_depth', "How substantive versus surface is the USER's engagement across the episode?",
  'Depth of user reasoning. Single-word and yes/no reactions low; multi-step reasoning and integration high.',
  {0:'User is purely surface (yes/no, single words).',1:'User responds but stays at surface level.',2:'User offers some substance with elaboration.',3:'User shows substantive engagement; builds reasoning chains.',4:'User shows deep engagement; integrates multiple considerations and reasons through them.'}),
]
CRITERION_NAMES = [c[0] for c in CRITERIA]
_BLOCK = '\n\n'.join(
    f"Criterion: {n}\nQuestion: {q}\nDefinition: {d}\nAnchors:\n" +
    '\n'.join(f"  {s} = {a}" for s, a in anc.items())
    for n, q, d, anc in CRITERIA)

# Scorer A = Exp 1's SYSTEM_USER (verbatim). Adapted only to name Assistant_A/Assistant_B.
SYSTEM_A = (
    "You are a transcript analysis instrument scoring USER behaviour in a human-AI creative "
    "collaboration. The transcript contains User turns and Assistant_A / Assistant_B turns; you read "
    "the full context but rate ONLY the User's behaviour.\n\n"
    "RULES:\n"
    "- Score the user, not the assistant. The assistant's output is context, not a target of scoring.\n"
    "- Never reward length, fluency, politeness, or confidence in itself.\n"
    "- Use only evidence present in the episode.\n"
    "- Every non-null score must include at least one exact verbatim quote copied from a User turn "
    "(not an assistant turn). If you cannot quote from a user turn, return score_0_4 = null and "
    "usable_for_inference = false.\n"
    "- Echoes of the assistant in the user's turns do NOT count as user initiative or proposals; they "
    "may count for user_acceptance_yes_and.\n"
    "- Score on the 0-4 anchors exactly as defined.\n"
    "- Return JSON only. No prose.\n"
    "- Output MUST be a single object {conversation_id, episode_id, scores} where scores is a list with "
    "one entry per criterion in the RUBRIC BLOCK.\n"
    "\nRUBRIC BLOCK:\n" + _BLOCK)

# Scorer B = paraphrased (prompt-robustness); identical anchors, reworded instructions.
SYSTEM_B = (
    "You are an evidence auditor rating the USER's creative behaviour in a masked human-AI "
    "problem-solving transcript. You may read the Assistant_A / Assistant_B turns for context, but you "
    "assess only what the User does.\n\n"
    "REQUIREMENTS (do not deviate):\n"
    "1. Judge the user's contribution alone; the assistant is background.\n"
    "2. Do not credit verbosity, polish, or politeness.\n"
    "3. Back every rating with one exact quote taken from a User turn — never an assistant turn. With no "
    "usable user quote, set score_0_4 to null and usable_for_inference to false.\n"
    "4. A user merely repeating the assistant's idea is NOT user initiative or a user proposal (it may "
    "count as yes-and uptake).\n"
    "5. Apply the 0-4 anchors exactly.\n"
    "6. Output valid JSON only, one entry per criterion; no surrounding text.\n"
    "\nThe six criteria and their anchors:\n" + _BLOCK)

MASK_MAP = [
    (re.compile(r'\bDivergent\b', re.I), 'Assistant_A'), (re.compile(r'\bTaylor\b', re.I), 'Assistant_A'),
    (re.compile(r'\bConvergent\b', re.I), 'Assistant_B'), (re.compile(r'\bAlex\b', re.I), 'Assistant_B'),
    (re.compile(r'\bGPT\b', re.I), 'Assistant'), (re.compile(r'\bgpt\b'), 'Assistant'),
]
def mask(text):
    out = str(text or '')
    for rx, sub in MASK_MAP: out = rx.sub(sub, out)
    return out

clean = {int(r['conversation_id']): (r['group'], r['challenge_id'])
         for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
by = defaultdict(list)
for m in msgs:
    if m['conversation_id'] in clean: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def transcript(c, cap=22000):
    lines = []
    for m in by[c]:
        who = 'User' if m['message_src'] == 'user' else ('Assistant_A' if m.get('persona') == 'Taylor' else 'Assistant_B')
        txt = mask((m.get('message_en') or m.get('message') or '').strip())
        if txt: lines.append(f"{who}: {txt}")
    return '\n'.join(lines)[:cap]

def build_prompt(c):
    nu = sum(1 for m in by[c] if m['message_src'] == 'user')
    return (f"conversation_id: {c}\nepisode_id: {c}_full\nnum_user_turns: {nu}\n\nEPISODE TEXT:\n{transcript(c)}\n\n"
            "Return one object with a `scores` list containing one entry per criterion from the RUBRIC "
            "BLOCK (all 6). Use null score_0_4 only when no User evidence is present for that criterion.")

CRIT_SCHEMA = {"type":"object","properties":{
    "criterion":{"type":"string"},"score_0_4":{"type":"integer","nullable":True},
    "confidence_0_1":{"type":"number"},"evidence_quotes":{"type":"array","items":{"type":"string"}},
    "reason_short":{"type":"string"},"counterevidence":{"type":"string"},
    "usable_for_inference":{"type":"boolean"}},
    "required":["criterion","score_0_4","confidence_0_1","evidence_quotes","reason_short","usable_for_inference"]}
BUNDLE_SCHEMA = {"type":"object","properties":{
    "conversation_id":{"type":"string"},"episode_id":{"type":"string"},
    "scores":{"type":"array","items":CRIT_SCHEMA}},"required":["conversation_id","episode_id","scores"]}

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
def score_once(c, scorer):
    r = client.models.generate_content(model=MODEL, contents=build_prompt(c),
        config=types.GenerateContentConfig(system_instruction=(SYSTEM_A if scorer=='A' else SYSTEM_B),
            response_mime_type='application/json', response_schema=BUNDLE_SCHEMA,
            temperature=0.15, max_output_tokens=2600))
    return json.loads(r.text)

FIELDS = ['conversation_id','group','challenge_id','scorer','criterion','score_0_4','confidence_0_1',
          'evidence_quotes','reason_short','counterevidence','usable_for_inference','conv_word_count']

def main():
    done = set(); rows = []
    if os.path.exists(OUT_RAW):
        for r in csv.DictReader(open(OUT_RAW, encoding='utf-8')):
            done.add((int(r['conversation_id']), r['scorer'])); rows.append(r)
        print(f'resume: {len(done)} (conv,scorer) pairs done')
    cids = sorted(by)
    for i, c in enumerate(cids, 1):
        wc = len(transcript(c).split())
        for scorer in ('A', 'B'):
            if (c, scorer) in done: continue
            obj = None
            for attempt in range(8):
                try: obj = score_once(c, scorer); break
                except Exception as e:
                    if '429' in repr(e) or 'RESOURCE_EXHAUSTED' in repr(e): time.sleep(min(90, 15*(attempt+1)))
                    elif attempt >= 3: print('  err', c, scorer, repr(e)[:80]); break
                    else: time.sleep(5)
            if obj is None: continue
            seen = set()
            for cs in obj.get('scores', []):
                name = cs.get('criterion')
                if name not in CRITERION_NAMES or name in seen: continue
                seen.add(name)
                rows.append({'conversation_id':c,'group':clean[c][0],'challenge_id':clean[c][1],'scorer':scorer,
                    'criterion':name,'score_0_4':cs.get('score_0_4'),'confidence_0_1':cs.get('confidence_0_1'),
                    'evidence_quotes':' | '.join(cs.get('evidence_quotes') or [])[:600],
                    'reason_short':str(cs.get('reason_short') or '')[:400],
                    'counterevidence':str(cs.get('counterevidence') or '')[:300],
                    'usable_for_inference':bool(cs.get('usable_for_inference', True)),'conv_word_count':wc})
            print(f'{i:2}/{len(cids)} conv {c} [{clean[c][0][:4]}] S{scorer}: {len(seen)}/6', flush=True)
            with open(OUT_RAW,'w',encoding='utf-8',newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            time.sleep(5)
    print('done ->', OUT_RAW)

if __name__ == '__main__':
    main()
