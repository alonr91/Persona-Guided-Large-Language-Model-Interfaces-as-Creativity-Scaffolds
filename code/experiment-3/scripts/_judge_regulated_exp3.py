# -*- coding: utf-8 -*-
"""Regulated LLM-rubric proxy for Experiment 3 — PORTED FROM EXPERIMENT 1.

Faithful port of Exp 1's regulated reanalysis (os_pipeline/regulated): the same
12 bounded process-regulation criteria (0-4, anchored), the same strict (A) and
paraphrased (B) scorer prompts, condition masking, and the same conservative
deterministic adjudicator. It scores dialogue REGULATION, not externally judged
creativity, and is reported as proxy triangulation per the thesis's section-2.10
boundary on LLM-as-judge designs.

Adaptations for Exp 3 (per user direction):
  - Models: Gemini only (Exp 1 used local Qwen A/B + Gemini cross-model C). A and
    B are paraphrase-robustness scorers on the same Gemini model. No cross-MODEL
    (different-family) leg is run here; that is noted as the one part of Exp 1's
    validation stack not reproduced.
  - Unit: whole conversation (Exp 1 scored segmented episodes). The clean sample
    is the 18 long on-task conversations; each is masked and scored as one unit.
  - Masking: Taylor->Assistant_A, Alex->Assistant_B (both conditions have two
    routes, so this is condition-blind), persona words scrubbed from text.

Model: gemini-2.5-flash, temperature 0.15 (matching Exp 1's scorer). Resumable.
"""
import os, json, csv, re, time
from collections import defaultdict
from google import genai
from google.genai import types

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
OUT_RAW = f'{BASE}/outputs/regulated_rubric_raw.csv'
MODEL = 'gemini-3.1-flash-lite'  # newest available flash-lite (no 3.5-flash-lite exists); flash has only 20 req/day free

# ---------------- 12 criteria (verbatim from Exp 1 os_pipeline/regulated/rubric.py) ----------------
CRITERIA = [
 ('exploration_opening', 'Does the episode expand the possibility space in a meaningful way?',
  'Rate how much the episode opens the conversation to new options, frames, stakeholders, or analogies beyond the current focus.',
  {0:'No expansion; repeats or narrows immediately',1:'Adds one obvious variant',2:'Adds several relevant but conventional options',3:'Opens a meaningfully different direction',4:'Opens a new frame, analogy, stakeholder, or design space'}, False),
 ('reframing_quality', 'Does the episode change how the problem is understood?',
  'Rate whether the episode genuinely shifts assumptions, target users, constraints, or success criteria, vs merely rewording the same idea.',
  {0:'No reframe',1:'Surface rewording only',2:'Shifts emphasis but not assumptions',3:'Changes assumptions, target user, constraint, or success criterion',4:'Produces a generative new problem frame that guides later turns'}, False),
 ('evaluative_discipline', 'Does the episode help narrow responsibly using criteria, tradeoffs, or constraints?',
  'Rate whether the episode introduces criteria, weighs options, or makes tradeoffs explicit, vs offering vague preferences or ungrounded judgments.',
  {0:'No evaluation or ungrounded judgment',1:'Vague preference or unsupported ranking',2:'Some criteria but weak comparison',3:'Clear tradeoffs and criteria',4:'Strong feasibility or usefulness critique while preserving alternatives'}, False),
 ('agency_preservation', "Does the assistant preserve the user's authorship and control?",
  "Rate how much the user's goals, ideas, and judgments shape the conversation, vs the assistant taking over the solution.",
  {0:'Assistant takes over; user becomes accepter',1:'Assistant mostly directs the solution',2:'Mixed control',3:'User goals and ideas shape the response',4:'Assistant explicitly scaffolds user decision-making and ownership'}, False),
 ('anchor_management', 'Does the episode reduce fixation on early assistant suggestions or manage anchors transparently?',
  'Rate whether the episode acknowledges, contrasts, or helps the user escape an initial anchor (first strong suggestion), vs reinforcing it.',
  {0:'Reinforces the first anchor without alternatives',1:'Minor variation around the anchor',2:'Some alternatives but anchor remains dominant',3:'Actively contrasts anchor with alternatives',4:'Helps user escape, reinterpret, or deliberately choose the anchor'}, False),
 ('coregulation_uptake', "Does one party's stance productively shape the next party's response?",
  "Rate the quality of uptake: does the user build on the assistant's move, or does the assistant build on the user's intent, to transform the idea jointly?",
  {0:'No uptake or breakdown',1:'Superficial acknowledgment',2:'User follows the topic but not the reasoning',3:"User builds on the assistant's stance, or assistant builds on user intent",4:'Clear collaborative transformation of the idea'}, False),
 ('timing_fit', "Is the episode's regulatory move appropriate for the current stage of the conversation?",
  "Rate whether the episode's move (opening, narrowing, committing, reframing) is well-timed for the conversation's current phase.",
  {0:'Clearly mistimed; harms process',1:'Weak timing',2:'Acceptable but generic',3:'Well matched to current stage',4:'Precisely regulates transition from exploration to evaluation or vice versa'}, False),
 ('implementation_grounding', 'Does the episode move from abstract idea toward actionable, constraint-aware solution development?',
  'Rate whether the episode moves from abstract toward actionable: constraints, resources, stakeholders, next steps.',
  {0:'Purely abstract or decorative',1:'Vague implementation language',2:'Some practical constraints',3:'Clear next steps, stakeholders, or resources',4:'Strong implementation pathway with constraints and tradeoffs'}, False),
 ('cognitive_load_clarity', "Does the episode reduce the user's burden and make the next action clearer?",
  'Rate whether the episode is structured, concise, and makes the next usable action obvious, vs overwhelming or vague.',
  {0:'Overwhelming, vague, or confusing',1:'Too many options without structure',2:'Some structure',3:'Clear, usable, appropriately concise',4:'Strong scaffolding that makes the next action obvious'}, False),
 ('stance_integrity', 'Does the assistant remain faithful to its apparent role or stance without collapsing into generic answer-giving?',
  'Rate how stably the assistant maintains a distinct stance (open, evaluative, structured, bounded, etc.) vs drifting into generic answer-giving.',
  {0:'Stance collapses into generic behavior',1:'Weak or inconsistent stance',2:'Some stance markers',3:'Clear stance behavior',4:'Strong, stable stance without oversteering'}, False),
 ('premature_convergence_risk', 'Does the episode narrow too early, over-commit, or make alternatives socially/cognitively unavailable?',
  'Rate the risk of premature closure: early commitment, pressure to agree, or suppression of alternatives. HIGHER = WORSE.',
  {0:'No premature convergence risk',1:'Slight narrowing',2:'Moderate narrowing but recoverable',3:'Strong premature closure',4:'Severe closure that dominates later interaction'}, True),
 ('runaway_divergence_risk', 'Does the episode expand without helping selection, criteria, or progress?',
  'Rate the risk of uncontrolled branching without a commitment path. HIGHER = WORSE.',
  {0:'No runaway divergence risk',1:'Slight expansion without structure',2:'Many options but some organization',3:'Expansion creates overload or postpones progress',4:'Severe uncontrolled branching without commitment path'}, True),
]
CRITERION_NAMES = [c[0] for c in CRITERIA]

_RUBRIC_TEXT_BLOCK = '\n\n'.join(
    f"Criterion: {name}\nQuestion: {q}\nDefinition: {d}\nAnchors:\n" +
    '\n'.join(f"  {s} = {a}" for s, a in anchors.items()) +
    ("\n  [NOTE: higher = WORSE for this criterion]" if rev else '')
    for name, q, d, anchors, rev in CRITERIA)

# ---------------- scorer prompts (verbatim from Exp 1 scorer.py) ----------------
SYSTEM_A = (
    "You are a transcript analysis instrument, not a creative judge. "
    "You score a MASKED episode from a human-AI creative collaboration.\n\n"
    "RULES:\n"
    "- Never infer the experimental condition.\n"
    "- Never reward length, fluency, politeness, or confidence.\n"
    "- Use only evidence present in the episode.\n"
    "- Score each applicable criterion exactly as defined.\n"
    "- If evidence is insufficient for a criterion, return score_0_4 = null and usable_for_inference = false.\n"
    "- Every non-null score must include at least one exact evidence quote copied verbatim from the episode (same words, same punctuation).\n"
    "- Include counterevidence if present.\n"
    "- Return JSON only. No prose.\n"
    "- Output MUST be a single object {conversation_id, episode_id, scores} where scores is a list with one entry per criterion in the RUBRIC BLOCK.\n"
    "- For the two RISK criteria (premature_convergence_risk, runaway_divergence_risk) higher scores mean MORE RISK.\n"
    "\nRUBRIC BLOCK:\n" + _RUBRIC_TEXT_BLOCK)

SYSTEM_B = (
    "You are an evidence-auditor that assigns 0-4 ordinal ratings to a masked "
    "transcript episode from a human-AI creative problem-solving session.\n\n"
    "REQUIREMENTS (do not deviate):\n"
    "1. Never guess the study condition, persona, or treatment group.\n"
    "2. Ignore surface polish, politeness, and verbosity.\n"
    "3. Support every rating with an exact short quote from the episode.\n"
    "4. If the episode contains no evidence for a given criterion, set score_0_4 to null and usable_for_inference to false for that row.\n"
    "5. Mark bias flags honestly.\n"
    "6. Output must be valid JSON matching the schema; no commentary around it.\n"
    "7. For risk criteria, a higher rating means the risk is MORE pronounced.\n"
    "\nYou will evaluate 12 criteria. The definitions and anchor points are:\n" + _RUBRIC_TEXT_BLOCK)

# ---------------- condition masking (adapted from Exp 1 masking.py) ----------------
MASK_MAP = [
    (re.compile(r'\bbounded[\s-]rationality\b', re.I), 'Assistant_A'),
    (re.compile(r'\bstrictly[\s-]rational\b', re.I), 'Assistant_A'),
    (re.compile(r'\bDivergent\b', re.I), 'Assistant_A'),
    (re.compile(r'\bTaylor\b', re.I), 'Assistant_A'),
    (re.compile(r'\bConvergent\b', re.I), 'Assistant_B'),
    (re.compile(r'\bAlex\b', re.I), 'Assistant_B'),
    (re.compile(r'\bGPT\b', re.I), 'Assistant'),
    (re.compile(r'\bgpt\b'), 'Assistant'),
]
def mask(text):
    out = str(text or '')
    for rx, sub in MASK_MAP: out = rx.sub(sub, out)
    return out

# ---------------- build masked transcript per conversation ----------------
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
        if m['message_src'] == 'user':
            who = 'User'
        else:
            who = 'Assistant_A' if m.get('persona') == 'Taylor' else 'Assistant_B'
        txt = mask((m.get('message_en') or m.get('message') or '').strip())
        if txt: lines.append(f"{who}: {txt}")
    t = '\n'.join(lines)
    return t[:cap]

def build_user_prompt(c):
    nu = sum(1 for m in by[c] if m['message_src'] == 'user')
    return (f"conversation_id: {c}\nepisode_id: {c}_full\nepisode_type: full_conversation\n"
            f"num_user_turns: {nu}\n\nEPISODE TEXT:\n{transcript(c)}\n\n"
            "Return one object with a `scores` list containing one entry per criterion "
            "from the RUBRIC BLOCK (all 12). Use null score_0_4 only if a criterion is "
            "genuinely inapplicable to this conversation.")

# ---------------- Gemini structured output ----------------
CRIT_SCHEMA = {"type":"object","properties":{
    "criterion":{"type":"string"},
    "score_0_4":{"type":"integer","nullable":True},
    "confidence_0_1":{"type":"number"},
    "evidence_quotes":{"type":"array","items":{"type":"string"}},
    "reason_short":{"type":"string"},
    "counterevidence":{"type":"string"},
    "possible_biases":{"type":"array","items":{"type":"string"}},
    "usable_for_inference":{"type":"boolean"}},
    "required":["criterion","score_0_4","confidence_0_1","evidence_quotes","reason_short","usable_for_inference"]}
BUNDLE_SCHEMA = {"type":"object","properties":{
    "conversation_id":{"type":"string"},"episode_id":{"type":"string"},
    "scores":{"type":"array","items":CRIT_SCHEMA}},
    "required":["conversation_id","episode_id","scores"]}

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
def score_once(c, scorer):
    system = SYSTEM_A if scorer in ('A', 'C') else SYSTEM_B
    r = client.models.generate_content(model=MODEL,
        contents=build_user_prompt(c),
        config=types.GenerateContentConfig(system_instruction=system,
            response_mime_type='application/json', response_schema=BUNDLE_SCHEMA,
            temperature=0.15, max_output_tokens=4000))
    return json.loads(r.text)

FIELDS = ['conversation_id','group','challenge_id','scorer','criterion','score_0_4',
          'confidence_0_1','evidence_quotes','reason_short','counterevidence',
          'possible_biases','usable_for_inference','conv_word_count']

def main():
    done = set()
    rows = []
    if os.path.exists(OUT_RAW):
        for r in csv.DictReader(open(OUT_RAW, encoding='utf-8')):
            done.add((int(r['conversation_id']), r['scorer'])); rows.append(r)
        print(f'resume: {len(done)} (conv,scorer) pairs already scored')
    cids = sorted(by)
    for i, c in enumerate(cids, 1):
        wc = len(transcript(c).split())
        for scorer in ('A', 'B'):
            if (c, scorer) in done: continue
            obj = None
            for attempt in range(8):
                try:
                    obj = score_once(c, scorer); break
                except Exception as e:
                    if '429' in repr(e) or 'RESOURCE_EXHAUSTED' in repr(e):
                        time.sleep(min(90, 15*(attempt+1)))
                    elif attempt >= 3:
                        print('  err', c, scorer, repr(e)[:80]); break
                    else: time.sleep(5)
            if obj is None: continue
            seen = set()
            for cs in obj.get('scores', []):
                name = cs.get('criterion')
                if name not in CRITERION_NAMES or name in seen: continue
                seen.add(name)
                rows.append({'conversation_id':c,'group':clean[c][0],'challenge_id':clean[c][1],
                    'scorer':scorer,'criterion':name,'score_0_4':cs.get('score_0_4'),
                    'confidence_0_1':cs.get('confidence_0_1'),
                    'evidence_quotes':' | '.join(cs.get('evidence_quotes') or [])[:600],
                    'reason_short':str(cs.get('reason_short') or '')[:400],
                    'counterevidence':str(cs.get('counterevidence') or '')[:300],
                    'possible_biases':'|'.join(cs.get('possible_biases') or ['none']),
                    'usable_for_inference':bool(cs.get('usable_for_inference', True)),
                    'conv_word_count':wc})
            print(f'{i:2}/{len(cids)} conv {c} [{clean[c][0][:4]}] S{scorer}: {len(seen)}/12 criteria', flush=True)
            with open(OUT_RAW,'w',encoding='utf-8',newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            time.sleep(5)  # rate-limit pacing (user: mind API rate)
    print('done ->', OUT_RAW)

if __name__ == '__main__':
    main()
