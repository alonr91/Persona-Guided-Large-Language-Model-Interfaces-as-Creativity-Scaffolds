# Experiment 3 method specifications

> Reproduced from the thesis appendices. Section numbers are those of the reviewed draft.


## C.1 Persona system prompts

The persona manipulation was identical to Experiment 2; the full divergent (Taylor, T=0.8), convergent (Alex, T=0.3), and control system prompts are reproduced verbatim in Appendix B.1 and are not duplicated here. The only deployment differences were operational: participant-chosen problems in place of the fixed prompt, and self-paced multi-day sessions in place of a single 20-minute session.


## C.2 Idea-portfolio pipeline, normalisation, and dose-response

Idea portfolios were extracted with Experiment 1’s multi-agent pipeline on user-only English text. Agent 1 (a local instruction-tuned model under the Experiment 1 extractor contract) extracts only user-originated, evidence-grounded ideas, filtering challenge-restatements, assistant-echoes, meta-questions, and reactions; Agent 2 consolidates near-duplicates within a conversation (fluency = canonical-idea count); Agent 4 categorises canonical ideas across the corpus (flexibility); Agent 5 computes idea-centroid originality. Because conversations vary widely in length, quantitative measures are reported as rates and peer-relative originality is recomputed within the sample. A within-treatment dose-response (Pearson r, no correction) relates interaction style (convergent share, switch rate, timing of convergent engagement) to each measure. Topic-residualised embedding originality (challenge labels, hand-validated) serves as a cross-check, and forward flow (Gray et al., 2019) and divergent semantic integration (Johnson et al., 2022), computed per user idea and tagged by persona, as validated semantic-creativity checks (no persona difference). Outputs: outputs/idea_portfolio_exp1.csv, long_portfolio.csv, dose_response_normalized.csv, originality_topic_controlled.csv.
Table C.1. Length-normalised idea-portfolio measures over the long, on-task sample (treatment vs control; Hedges’ g). Control n=5; all n.s.
Table C.2. Within-treatment dose-response (n=13; Pearson r): convergent share predicts no creativity measure.
Measure
Treatment
Control
g
Idea-generation rate (ideas / user turn)
0.50
0.56
−0.34
Fluency (idea count, raw)
8.23
6.20
+0.65
Originality (topic-residualised)
1.05
1.05
−0.07
Originality (idea-centroid)
0.19
0.20
−0.27
Within-portfolio idea diversity
0.32
0.33
−0.14
Predictor → outcome
r
p
Convergent share → idea rate
−0.21
.48
Convergent share → originality (idea-centroid)
+0.17
.57
Convergent share → originality (topic-residualised)
−0.28
.35
Convergent share → within-portfolio diversity
−0.44
.13


## C.3 Regulated LLM-rubric process proxy (ported from Experiment 1)

This proxy reproduces Experiment 1’s regulated LLM-rubric (its os_pipeline/regulated module). Each of the 18 long, on-task conversations was masked (persona names → Assistant_A / Assistant_B; the condition is never shown) and scored as a whole on twelve anchored 0–4 process-regulation criteria: exploration opening, reframing quality, evaluative discipline, agency preservation, anchor management, co-regulation uptake, timing fit, implementation grounding, cognitive-load clarity, stance integrity, and two reverse-scored risks (premature-convergence, runaway-divergence). Two paraphrased scorers, A (strict, “transcript analysis instrument, not a creative judge”) and B (paraphrased), each return one bundled, evidence-grounded JSON object per conversation (every non-null score carries a verbatim quote; inapplicable criteria return null), reconciled per criterion by Experiment 1’s conservative adjudicator: agreement within one point is averaged; a gap of two or more takes the lower score and is flagged. Both scorers are gemini-3.1-flash-lite at temperature 0.15.
Pre-registered audits (Experiment 1’s stack). Evidence sufficiency: 100% of scored cells carry a verbatim evidence quote. Prompt robustness: the two paraphrased scorers agree closely (A–B Pearson r = 0.57–0.92; a gap of two or more points on a single cell). Length bias: a per-criterion regression of score on conversation word count leaves length explaining under 30% of the variance on every criterion except evaluative discipline (just above the bound), so no headline rests on length. Positive controls: all ten design-implied directions hold (treatment higher on the expansion and evaluation criteria, lower on premature-convergence risk). Divergences from Experiment 1, by design: Experiment 1 ran scorers A and B on a local Qwen model with Gemini as a cross-MODEL Scorer C and applied FDR correction; here both scorers run on Gemini (so the cross-model leg is not reproduced), scoring is at the conversation level rather than per segmented episode, and no multiplicity correction is applied (the Experiment 3 analysis policy). Outputs: outputs/regulated_rubric_raw.csv, regulated_rubric_adjudicated.csv, regulated_rubric_audit.csv, regulated_rubric_contrast.csv.
User-behaviour rubric (Experiment 1 “Option B”; §6.3.5, Table 6.4). The same masked transcripts are scored on six 0–4 user-only criteria (initiative, question richness, proposal specificity, yes-and uptake, reframing, and engagement depth), with the model instructed to rate only the participant’s turns, to quote user text only for every non-null score, and to treat echoes of the assistant as non-originated. Experiment 1 ran this as a single Gemini scorer; here it uses the same dual paraphrased scorers (A/B) and conservative adjudicator as the dialogic rubric, giving an added prompt-robustness check. The audit is strong: 100% evidence-grounded, no high-disagreement cells, A–B agreement r=0.87–1.00, and no length confound (R²≤0.11). Outputs: outputs/user_rubric_raw.csv, user_rubric_adjudicated.csv, user_rubric_audit.csv, user_rubric_contrast.csv.


## C.4 Semantic-trajectory formulas

The two turn-level semantic measures cited in §6.3.2 and §6.4.3 (accommodation, novelty-to-history) are computed from e(m), the bge-large-en-v1.5 embedding of message m, L2-normalised so that cosine similarity reduces to a dot product: cos(a, b) = e(a)·e(b). Turns are ordered by timestamp within each conversation, so for a conversation with n user turns the i-th user turn has normalised position posi = (i−1)/(n−1) (0 = first turn, 1 = last).
Novelty-to-history. For a turn mi and its same-stream history Hi (every earlier turn in the relevant stream: all prior turns for the assistant Taylor-versus-Alex contrast in §6.4.3; only the participant’s own earlier turns for the user trend in §6.3.2):
novelty(mi) = 1 − max{cos(e(mi), e(h)) : h ∈ Hi}
Novelty is defined against the single nearest (most similar) prior turn, the maximum over the set, rather than the average of all prior turns, so a turn only counts as unoriginal when it closely restates something specific said before; averaging against the full history would unfairly penalise long conversations for legitimately returning to an established theme. novelty=1 when a turn is maximally dissimilar to everything before it; novelty→0 when it nearly repeats an earlier turn. §6.3.2 correlates novelty(ui) against posi within the user stream, pooling turns across conversations (Pearson r; negative = declining novelty relative to one’s own earlier turns, i.e. exploitation of established material rather than continued exploration). §6.4.3 instead compares the mean novelty of assistant turns addressed to Taylor versus Alex directly (Welch’s t, Hedges’ g), with no position term, to ask whether the two personas differ in how far their content sits from what came before: content-level expansion versus contraction, independent of when in the conversation it happens.
Accommodation. For a user turn ui, let aprev(i) be the persona turn immediately preceding it:
accommodation(ui) = cos(e(ui), e(aprev(i)))
This is the classic interactive-alignment quantity (Fusaroli & Tylén, 2016; Reitter & Moore, 2014): how closely the user’s own words track the content the persona just supplied, on a 0 (unrelated) to 1 (near-identical) scale. §6.3.2 correlates accommodation(ui) against posi (Pearson r, treatment only): a negative trend means users grow less likely, not more, to echo the immediately preceding persona turn as the conversation matures, the opposite of what alignment toward an increasingly dominant interlocutor would predict. Read together with the declining novelty-to-history trend above, users move away from the persona’s content while converging on their own, which is the content-level basis for reading the choreography in §6.3.2 as complementarity (delegating consolidation to a persona) rather than alignment (adopting a persona’s content). Both correlations pool turns across conversations rather than testing one point per conversation, so the reported p-values likely understate the true uncertainty; the effect sizes (r≈−0.15 to −0.18) are small by conventional benchmarks.
