# -*- coding: utf-8 -*-
"""Polish pass: tighten abstract, shorten §3.5 thematic (~28%), thin signposting,
vary residual 'rather than'. Replaces whole P() calls via regex (DOTALL) so
internal string-concatenation boundaries are not an issue. No stats/claims change."""
import re
P = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3/scripts/_build_report.py'
txt = open(P, encoding='utf-8').read()

def span_replace(start_anchor, end_anchor, new, label):
    global txt
    pat = re.escape(start_anchor) + '.*?' + re.escape(end_anchor)
    txt, n = re.subn(pat, lambda m: new, txt, count=1, flags=re.DOTALL)
    assert n == 1, f'span failed: {label}'

# ---------- 1. Abstract (tightened, em-dash-free) ----------
ABSTRACT = (
 'P("Experiment 3 takes the dual-persona interface from Experiment 2 into the field. Two LLM personas, one "\n'
 '  "divergent (Taylor) and one convergent (Alex), sit behind two send buttons; we observed how design teams "\n'
 '  "used them across a four-day hackathon on real-world problems, with use entirely optional. No questionnaire "\n'
 '  "was collected, so we cannot speak to perceived creativity (RQ1) or personality effects (RQ2); the logs "\n'
 '  "instead let us ask whether the manipulation worked, how people ran the creative process, and whether their "\n'
 '  "ideas were more original (RQ3). We analyse the long, on-task conversations where the interface was "\n'
 '  "genuinely used (at least ten user turns; two off-task sessions dropped; 13 treatment, 5 control), "\n'
 '  "normalising every measure by length. The manipulation held: a hand-validated stance classifier shows the "\n'
 '  "divergent persona far more divergent than the convergent one (g=1.25), matching the expert check on these "\n'
 '  "same personas in Experiment 2, and the gap survived into the second half of long, sometimes multi-day "\n'
 '  "conversations (g=1.19). Users also ran a clear arc, leaning on the divergent persona early (71% vs. 31% of "\n'
 '  "first-half messages, g=1.13) and turning to the convergent one later, with the assistants\\u2019 stance "\n'
 '  "balance falling from 0.45 to 0.22 across quartiles; they handed consolidation to Alex while staying "\n'
 '  "generative themselves. An independent, condition-blind LLM rubric ported from Experiment 1 scores treatment "\n'
 '  "conversations far higher on reframing, exploration, and evaluative discipline, yet a user-only version "\n'
 '  "finds no change in the participants\\u2019 own behaviour: the scaffold works through the personas users "\n'
 '  "orchestrate, not by transforming the user. Originality did not carry over. After correcting for length, no "\n'
 '  "output measure separated the groups (idea-generation rate g=\\u22120.34; topic-controlled originality "\n'
 '  "g=\\u22120.07; both n.s.), and a turn-level analysis shows why: the personas change how people work, not "\n'
 '  "the content of the ideas they produce. The persona scaffold reliably shapes the creative process even in "\n'
 '  "messy field conditions; whether it lifts the originality of the final ideas depends on a task structured "\n'
 '  "cleanly enough to measure.")'
)
span_replace('P("Experiment 3 takes the dual-persona interface',
             'cleanly enough to measure.")', ABSTRACT, 'abstract')

# ---------- 2. §3.5 thematic — shorten ~28% (whole P() calls) ----------
span_replace("P('Theme 1: the divergent persona as idea engine", "')",
 'P("Theme 1: the divergent persona as idea engine and novelty foil. Users came to Taylor to generate and '
 'broaden, and distinctively to press for novelty, which was about three times more frequent to Taylor than to '
 'Alex (14% vs. 5% of turns). They rejected the ordinary (“it’s not creative enough”; '
 '“prosthetics are an option already, maybe think in another direction”; convs 731, 667) and pushed '
 'Taylor toward speculative leaps such as brain-computer helmets and tissue regeneration. This is the '
 'exploratory stance enacted: generating alternatives and resisting premature closure (cf. White, 2003, on '
 'expansion resources).")', 'theme1')

span_replace("P('Theme 2: the convergent persona as evaluator", "')",
 'P("Theme 2: the convergent persona as evaluator, specifier, and reality-tester. Users brought Alex a '
 'categorically different set of demands. Evaluation and selection were about five times more frequent to Alex '
 '(14% vs. 3%; e.g. “rate the solutions 1–5 for feasibility, effectiveness, and relevance”, conv '
 '595), and technical specification roughly seven times (14% vs. 2%; “where would you place the infrared '
 'sensor, and what is the minimum distance?”, convs 733, 737). Reality-testing also concentrated on Alex '
 '(“does this exist?”), as did requests to make a chosen idea concrete. This is convergent thinking: '
 'criteria, comparison, feasibility, and closure, with the convergent agent serving as an evaluative anchor and '
 'authority for grounding (§2.4.5).")', 'theme2')

span_replace("P('Theme 3: orchestration", "')",
 'P("Theme 3: orchestration, routing, and the explore-to-converge handoff. The two profiles were not merely '
 'different but managed as a division of labour: users routed generation to Taylor and judgment or grounding '
 'to Alex, brokered between them (“alex, what you think about that?” about a Taylor proposal, then '
 '“taylor, how can we do it?”; conv 523), and ran a recurring generate-with-Taylor then '
 'evaluate-and-specify-with-Alex handoff (convs 595, 733). The user, not either agent, assembles the arc.")',
 'theme3')

span_replace("P('The themes mirror and help explain the quantitative results", "')",
 'P("These themes give the qualitative grain behind the quantitative results and converge on one point: the '
 'user, not either agent, integrates generation and judgment. One caveat: the evaluation/selection code leaned '
 'on conv 595, the only session with explicit 1–5 rating, whereas specification and reality-testing spread '
 'across conversations 483, 733, and 737, so “the convergent persona as specifier” is the '
 'better-supported reading.")', 'theme-closing')

# ---------- 3. signposting + 4. vary 'rather than' (simple, asserted) ----------
REP = [
 ('the convergent one later (§3.3), broker', 'the convergent one later, broker'),
 ('(accommodation r=−0.15, §3.3).', '(accommodation r=−0.15).'),
 ('rubric (§2.4; Appendix D) reads the same masked', 'rubric (Appendix D) reads the same masked'),
 ('Rather than being absorbed by either', 'Instead of being absorbed by either'),
]
for old, new in REP:
    assert old in txt, f'MISSING: {old[:50]!r}'
    txt = txt.replace(old, new)

open(P, 'w', encoding='utf-8').write(txt)
print('polish applied; body em-dashes:', txt.split("H1('References')")[0].count('—') - 1)  # -1 for docstring
