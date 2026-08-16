# Redactions

Participants consented to their conversations being logged and analysed, not to their
publication. The derived tables in this repository are therefore published without the columns
that carry participant-authored text, and without the tables whose content is that text.

Every numeric result the thesis reports survives these redactions: what is removed is the
supporting quotation, not the score computed from it. Scripts in `code/` reference some of the
removed columns by name, and will not run against the published tables without them.

| File | Action | Detail |
|---|---|---|
| `derived/experiment-1/analysis_out/cat_panel/panel_scores_raw.csv` | REDACTED | dropped columns: evidence_quotes, rationale_short, counterevidence, possible_biases, parse_error |
| `derived/experiment-1/analysis_out/master_users.csv` | REDACTED | dropped columns: Persona round 1, Persona round 2, More effective interface details |
| `derived/experiment-1/analysis_out/master_wide.csv` | REDACTED | dropped columns: Persona round 1, Persona round 2, More effective interface details |
| `derived/experiment-1/analysis_out/production/categorized_ideas.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-1/analysis_out/production/validation_report.csv` | REDACTED | dropped columns: title |
| `derived/experiment-1/regulated_llm_reanalysis/01_cleaning_log.csv` | REDACTED | dropped columns: description |
| `derived/experiment-1/regulated_llm_reanalysis/02_episode_table.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-1/regulated_llm_reanalysis/03_idea_portfolio_llm.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-1/regulated_llm_reanalysis/04_episode_rubric_scores_raw.csv` | REDACTED | dropped columns: evidence_quotes, reason_short, counterevidence, possible_biases |
| `derived/experiment-1/regulated_llm_reanalysis/04_episode_rubric_scores_raw_scorerC.csv` | REDACTED | dropped columns: evidence_quotes, reason_short |
| `derived/experiment-1/regulated_llm_reanalysis/05_episode_rubric_scores_adjudicated.csv` | REDACTED | dropped columns: evidence_quotes_A, evidence_quotes_B, possible_biases_B |
| `derived/experiment-1/regulated_llm_reanalysis/10_claim_cards.csv` | REDACTED | dropped columns: claim_text, limitations, allowed_language, forbidden_language |
| `derived/experiment-1/regulated_llm_reanalysis/18_user_rubric_raw_scorerC.csv` | REDACTED | dropped columns: evidence_quotes, reason_short, counterevidence |
| `derived/experiment-1/regulated_llm_reanalysis/_scoring_sample.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-1/regulated_llm_reanalysis/turn_table.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-2/LLM evaluaotr V2/creativity_scores (3).csv` | REDACTED | dropped columns: warnings |
| `derived/experiment-2/LLM evaluaotr V2/ideas_and_categories (2).csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-2/LLM evaluaotr V2/judges_long (2).csv` | REDACTED | dropped columns: rat_originality, rat_value, rat_insight_reframing, rat_development, rat_integration, rat_human_ethical, rat_process_evidence, rat_overall, evidence_quotes |
| `derived/experiment-3/outputs/ideas_canonical_exp1.csv` | REMOVED | content is participant text or extracted idea text |
| `derived/experiment-3/outputs/regulated_rubric_raw.csv` | REDACTED | dropped columns: evidence_quotes, reason_short, counterevidence |
| `derived/experiment-3/outputs/user_rubric_raw.csv` | REDACTED | dropped columns: evidence_quotes, reason_short, counterevidence |
