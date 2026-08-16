# Experiment 2 method specifications

> Reproduced from the thesis appendices. Section numbers are those of the reviewed draft.


## B.2 Idea-extraction and originality pipeline

Every stage runs on a user-only transcript (participant turns concatenated in order; model turns excluded), so fluency and originality reflect the participant’s own contributions. The transcript is split into 12,000-character windows with 800-character overlap, and two LLM stages are applied. Stage 1 extracts distinct, actionable solution ideas (“could be built or piloted”), each with a short title, description, and an evidence quote drawn only from user text; ideas are unioned across windows and de-duplicated by a normalised (title | description) key, and the set size is the participant’s fluency. Stage 2 induces ≤8 broad, non-overlapping categories with exactly one category per idea. All calls use deterministic decoding (T=0, seed=7), strict JSON, on-disk caching, and retry-on-error; this analysis temperature is independent of the personas’ interactive temperatures. Each idea is then embedded (text-embedding-3-large), L2-normalised, and mean-pooled into a participant centroid; originality is the mean cosine distance to other participants’ centroids.
Table B.1. Pipeline parameters.
Parameter
Value
Extraction / category model
gpt-4.1-2025-04-14
Decoding temperature / seed
0.0 / 7
Window / overlap
12,000 / 800 chars
Max ideas per window / categories
40 / 8
Title / description / quote limits
≤8 / ≤80 / ≤25 words
De-dup key
normalised (title
description)
Embedding / distance
text-embedding-3-large / cosine
Verbatim prompts:
[Stage 1 system] You extract distinct, actionable solution ideas
from user-only transcripts. Deduplicate near-duplicates (keep the
most complete variant). Return JSON only.
[Stage 1 user] Extract solution ideas from the user's transcript.
Rules: - Actionable (could be built/piloted) - Merge minor variants;
avoid micro-splits - Cap to {max_ideas} ideas - Provide evidence
quotes (<=25 words) from user text only. Schema: {schema}.
User transcript: {user_transcript}
[Stage 1 schema] {"ideas":[{"id","title<=8w","description<=80w",
"evidence_spans":["quote<=25w"]}],"notes<=40w"}
[Stage 2 system] Induce broad, non-overlapping categories to cover
all ideas and assign each idea to exactly one category. Favor
breadth. Return JSON only.
[Stage 2 schema] {"categories":[{"id","name<=4w","definition<=20w"}],
"assignments":[{"idea_id","category_id"}]}


## B.3 Regulated LLM-judge evaluation pipeline

Each participant’s full user-only transcript (assistant turns excluded; no summarization, with overlapping windows used only to fit context limits) was scored by an ensemble of five expert judge personas. Each judge returned, in strict JSON, a 1–7 score and a brief rationale (≤30 words, no chain-of-thought) for eight dimensions; the per-conversation score on each dimension is the median across judges. All calls used temperature 0, a fixed seed, JSON-only responses, and on-disk caching for idempotency; judging used GPT-4.1 and idea extraction GPT-4.1-mini. Fluency and flexibility were computed by the same extraction pipeline as in Appendix B and reported as raw counts plus dataset-relative 1–7 scaling; the holistic ensemble remained the headline measure to avoid rewarding verbosity. A Bradley–Terry model over pairwise judge comparisons produced a global ranking as a cross-check.
Table B.2. LLM-judge pipeline specification.
Component
Specification
Judge personas
Design Thinking; Social Psychology; HCI; Philosopher/Ethicist; Innovation/Strategy
Scored dimensions (1–7)
holistic creativity, originality, value/usefulness, insight/reframing, development, integration, human/ethical, process/evidence
Aggregation
median across judges (ensemble); Bradley–Terry for pairwise ranking
Decoding
temperature 0, fixed seed, JSON-only, cached
Models
judging gpt-4.1; extraction/feedback gpt-4.1-mini
Rationales
≤30 words per dimension; no chain-of-thought
Validity audits
inter-judge ICC(2,k); length-bias (tokens × score); attribution; pairwise alignment
Rubric anchors (abbreviated):
Originality (1-7): 1 cliche / common; 4 moderately novel; 7 rare, compelling reframing with a coherent leap.
Value / usefulness (1-7): 1 impractical; 4 plausible plan; 7 high impact with constraints and metrics addressed.
Elaboration / development (1-7): 1 vague; 4 steps present; 7 thorough, anticipates risks, metrics, edge cases.
Holistic creativity (1-7): overall integration of novelty, value, variety, depth, and reflective refinement.
Holistic judge prompt (system), verbatim:
You are a {persona} evaluating a user's creativity from their full conversation (user-only text).
Judge substance, not verbosity or grammar. Ignore any AI content. Do NOT summarize; read all text.
Return JSON only. Provide brief (<=30 words) rationales per scored metric; no chain-of-thought.
Scales: 1-7 (see anchors).
