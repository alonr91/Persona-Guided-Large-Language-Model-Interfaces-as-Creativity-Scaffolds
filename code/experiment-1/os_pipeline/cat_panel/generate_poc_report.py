"""Generate the consolidated PoC summary report.

Runs AFTER scoring, adjudication, audits, and analyses are complete.
Consolidates V1–V7 validation gates + F1–F6 analysis outputs into a
single markdown report that can be reviewed or merged into §3.X of the
v5 docx.

Outputs:
  CAT_Panel_PoC_Report.md          — the report (top-level)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'
FIG  = ROOT / 'figures' / 'cat_panel'
REPORT_PATH = ROOT / 'CAT_Panel_PoC_Report.md'


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    try: return pd.read_csv(path)
    except Exception: return None


def _read_any(basename: str) -> pd.DataFrame | None:
    for ext in ('.parquet', '.csv'):
        fp = OUT / (basename + ext)
        if fp.exists():
            try:
                return (pd.read_parquet(fp) if ext == '.parquet'
                        else pd.read_csv(fp))
            except Exception:
                continue
    return None


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return '*(no data)*'
    df = df.head(max_rows).copy()
    # format numerics
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    return df.to_markdown(index=False)


def main():
    parts: list[str] = []
    parts.append('# CAT-Panel proof-of-concept report\n')
    parts.append('*Auto-generated. Sources: `analysis_out/cat_panel/`, '
                 '`figures/cat_panel/`.*\n')

    # ---- run metadata ----------------------------------------------------
    reg_path = OUT / 'prompt_registry.json'
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding='utf-8'))
        parts.append('## Run metadata\n')
        parts.append(f'- **Model**: `{reg.get("model_id")}`')
        parts.append(f'- **Temperature**: {reg.get("temperature")}')
        parts.append(f'- **Max output tokens**: {reg.get("max_output_tokens")}')
        parts.append(f'- **Judges**: {", ".join(reg.get("judges", []))}')
        parts.append(f'- **Dimensions**: {", ".join(reg.get("dim_names", []))}')
        parts.append(f'- **Prompt SHA-256 (truncated)**: `{reg.get("prompt_sha256","")[:16]}...`')
        parts.append(f'- **Started**: {reg.get("run_started_at","?")}')

    # ---- sample ---------------------------------------------------------
    sample_path = OUT / 'poc_sample.json'
    if sample_path.exists():
        s = json.loads(sample_path.read_text(encoding='utf-8'))
        parts.append('\n## Stratified PoC sample\n')
        parts.append(f'- **Users**: {len(s.get("selected_users", []))} '
                     f'({s.get("n_per_family")} per family × '
                     f'{len(s.get("families_in_order", []))} families)')
        parts.append(f'- **Conversations**: {len(s.get("conversation_ids", []))} '
                     '(each user contributes both their Persona round AND their GPT round)')
        parts.append(f'- **Sample seed**: {s.get("seed")}')

    # ---- raw-score totals + failure rate --------------------------------
    raw = _read_any('panel_scores_raw')
    if raw is not None:
        n_total = len(raw)
        n_fail = int(raw['score_1_7'].isna().sum())
        parts.append('\n## Scoring summary\n')
        parts.append(f'- **Rows scored**: {n_total:,} '
                     f'({4*40*8*2 if n_total else "?"} expected)')
        parts.append(f'- **Failures**: {n_fail} ({100*n_fail/max(1,n_total):.1f}%)')
        v = raw[raw['score_1_7'].notna()]
        parts.append(f'- **Score distribution (1–7)**: '
                     f'mean={v["score_1_7"].astype(float).mean():.2f}, '
                     f'sd={v["score_1_7"].astype(float).std(ddof=1):.2f}')
        parts.append(f'- **Per-judge mean**:')
        for j, m in v.groupby('judge_id')['score_1_7'].mean().round(2).items():
            parts.append(f'  - `{j}`: {m}')

    # ---- V1 inter-judge consensus ---------------------------------------
    parts.append('\n## V1 — Inter-judge consensus (the primary validity gate)\n')
    per_dim = _safe_read_csv(OUT / 'consensus_per_dim.csv')
    if per_dim is not None:
        parts.append('Per-dimension ICC(2,k) and mean pairwise quadratic-weighted κ '
                     'across the 4 expert judges. Retention rule: ICC ≥ 0.50 → '
                     'primary; 0.30–0.50 → exploratory; < 0.30 → drop.\n')
        parts.append(_md_table(per_dim[['label','n_complete_convs','icc_2_k',
                                         'mean_pairwise_kappa_qw','retention']]))
        retained = (per_dim['retention']=='primary').sum()
        explor = (per_dim['retention']=='exploratory').sum()
        dropped = (per_dim['retention']=='drop').sum()
        parts.append(f'\n**{retained} primary, {explor} exploratory, '
                     f'{dropped} dropped** of {len(per_dim)} dimensions.')

    # ---- V2 paraphrase stability ----------------------------------------
    parts.append('\n## V2 — Within-judge paraphrase stability\n')
    if raw is not None:
        v = raw[raw['score_1_7'].notna()]
        from scipy.stats import spearmanr
        rows = []
        for judge in sorted(v['judge_id'].unique()):
            for dim in sorted(v['dimension'].unique()):
                sub = v[(v['judge_id']==judge) & (v['dimension']==dim)]
                pivot = sub.pivot_table(index='conversation_id',
                    columns='paraphrase', values='score_1_7', aggfunc='first')
                if 'A' not in pivot.columns or 'B' not in pivot.columns: continue
                pivot = pivot.dropna()
                if len(pivot) < 3: continue
                try:
                    rho, _ = spearmanr(pivot['A'], pivot['B'])
                except Exception:
                    continue
                rows.append(dict(judge=judge, dimension=dim, n=len(pivot),
                                 rho_AB=rho))
        if rows:
            s = pd.DataFrame(rows)
            parts.append(f'- **Pairs with both A and B**: {s["n"].sum()}')
            parts.append(f'- **Spearman ρ(A,B) overall median**: {s["rho_AB"].median():.3f}')
            parts.append(f'- **% (judge,dim) cells with ρ ≥ 0.55 (V2 gate)**: '
                         f'{100*(s["rho_AB"]>=0.55).mean():.0f}%')

    # ---- V3 cross-model (optional) --------------------------------------
    cm = _safe_read_csv(OUT / 'cross_model_qwen.csv')
    if cm is not None:
        parts.append('\n## V3 — Cross-model robustness (Qwen replication of Dr. C)\n')
        parts.append(_md_table(cm))

    # ---- V4 length bias --------------------------------------------------
    lb = _safe_read_csv(OUT / 'audit_length_bias.csv')
    if lb is not None:
        parts.append('\n## V4 — Length-bias audit\n')
        sig = lb[(lb['r_log_words'].abs() > 0.30) & (lb['p'] < 0.05)]
        if len(sig):
            parts.append(f'**{len(sig)} cells with |r| > 0.30 and p < 0.05** '
                         '(headline effects on these should be reported as residuals '
                         'after partialling out word count):')
            parts.append(_md_table(sig[['judge_id','label','n','r_log_words','p']]))
        else:
            parts.append('No (judge × dim) cell shows a strong word-count effect '
                         '(all |r| ≤ 0.30 or p ≥ 0.05).')

    # ---- V5 mask leak ----------------------------------------------------
    ml = _safe_read_csv(OUT / 'audit_mask_leak.csv')
    if ml is not None:
        acc = ml['correct'].mean() if 'correct' in ml.columns else None
        parts.append('\n## V5 — Mask-leak audit\n')
        parts.append(f'- **n**: {len(ml)} blind-guess attempts')
        if acc is not None:
            chance = 1/5
            parts.append(f'- **Accuracy**: {acc:.2%} (chance = {chance:.2%})')
            parts.append(f'- **Verdict**: mask is **{"CLEAN" if acc <= 0.30 else "POSSIBLY LEAKING"}**')

    # ---- V6 halo ---------------------------------------------------------
    halo = _safe_read_csv(OUT / 'audit_halo.csv')
    if halo is not None:
        flagged = halo[halo['halo_flag']==True] if 'halo_flag' in halo.columns else pd.DataFrame()
        parts.append('\n## V6 — Halo-bias audit\n')
        if len(flagged):
            parts.append(f'**{len(flagged)} dimension-pairs flagged (|r| > 0.85)**:')
            parts.append(_md_table(flagged))
        else:
            parts.append('No dimension-pair flagged for halo bias (no |r| > 0.85 between '
                         'conceptually distinct dimensions).')

    # ---- V7 construct validity ------------------------------------------
    cv = _safe_read_csv(OUT / 'construct_validity.csv')
    if cv is not None:
        parts.append('\n## V7 — Construct validity\n')
        parts.append('Spearman ρ between CAT-Panel consensus scores and external '
                     'layers (Taxonomy-2 user-side stance, extracted-idea originality).\n')
        parts.append(_md_table(cv.sort_values('rho', key=lambda s: s.abs(),
                                              ascending=False)))

    # ---- F1 d_z by family ------------------------------------------------
    f1 = _safe_read_csv(OUT / 'F1_dz_by_family.csv')
    if f1 is not None:
        sig = f1[f1['p'] < 0.05]
        parts.append('\n## F1 — User-creativity entrainment by persona family\n')
        parts.append(f'Within-subject Persona − GPT Cohen\'s *d*<sub>z</sub> per family × dimension. '
                     f'**{len(sig)} of {len(f1)} cells** show raw-p < .05.')
        parts.append(f'\n![F1 heatmap](figures/cat_panel/F1_panel_dz_by_family.png)\n')
        if len(sig):
            parts.append('### Significant cells (raw p < .05)')
            parts.append(_md_table(sig[['family','label','n','dz','p']]
                                    .sort_values('p')))

    # ---- F2 Big-5 moderation ---------------------------------------------
    f2 = _safe_read_csv(OUT / 'F2_big5_moderation.csv')
    if f2 is not None:
        sig = f2[f2['p'] < 0.05]
        parts.append('\n## F2 — Big-5 personality moderation\n')
        parts.append(f'Spearman ρ between Big-5 traits and Δ user-construct (Persona − GPT). '
                     f'**{len(sig)} of {len(f2)} cells** show raw-p < .05.')
        if len(sig):
            parts.append(_md_table(sig[['family','trait','label','n','rho','p']]
                                    .sort_values('p')))

    # ---- F3 originality bridge ------------------------------------------
    f3 = _safe_read_csv(OUT / 'F3_panel_to_originality.csv')
    if f3 is not None:
        parts.append('\n## F3 — Process → product bridge\n')
        parts.append('Spearman ρ between CAT-Panel dimensions and the os_pipeline '
                     '`orig_*` / `n_ideas` outcomes. **Top correlations by |ρ|**:')
        parts.append(_md_table(f3.assign(absr=f3['rho'].abs())
                                .sort_values('absr', ascending=False)
                                .drop(columns='absr')
                                .head(12)))

    # ---- F4 disagreement radar ------------------------------------------
    if (FIG / 'F4_judge_disagreement_radar.png').exists():
        parts.append('\n## F4 — Inter-judge profile divergence\n')
        parts.append('Each judge\'s mean rating profile across the 8 dimensions, overlaid. '
                     'Divergence between judges on a dimension = theoretical-lens disagreement.\n')
        parts.append('![F4 radar](figures/cat_panel/F4_judge_disagreement_radar.png)')

    # ---- F5 convergence vs Taxonomy 2 -----------------------------------
    if (FIG / 'F5_panel_vs_taxonomy2.png').exists():
        parts.append('\n## F5 — CAT-Panel vs Taxonomy-2 convergence\n')
        parts.append('![F5 bar chart](figures/cat_panel/F5_panel_vs_taxonomy2.png)')

    # ---- F6 mediation ----------------------------------------------------
    f6 = _safe_read_csv(OUT / 'F6_mediation_summary.csv')
    if f6 is not None and len(f6):
        r = f6.iloc[0]
        parts.append('\n## F6 — Persona → Δ panel-dim → Δ originality (within-subject)\n')
        parts.append(f'- **Mediator**: `{r.get("panel_dim")}`')
        parts.append(f'- **Outcome**: `{r.get("outcome")}`')
        parts.append(f'- **n**: {r.get("n")}')
        parts.append(f'- **Spearman ρ (indirect-effect strength)**: {r.get("rho"):.3f}, p = {r.get("p"):.4f}')
        parts.append(f'- **95% bootstrap CI**: [{r.get("ci_lo"):.3f}, {r.get("ci_hi"):.3f}]')
        if (FIG / 'F6_mediation_path.png').exists():
            parts.append('\n![F6 mediation path](figures/cat_panel/F6_mediation_path.png)')

    # ---- closing remarks -------------------------------------------------
    parts.append('\n---\n')
    parts.append('## Pipeline status\n')
    parts.append('- See `os_pipeline/cat_panel/` for the implementation.')
    parts.append('- Re-run with `python -m os_pipeline.cat_panel.run_panel --poc --sleep 0.5` then '
                 '`python -m os_pipeline.cat_panel.run_audits` and '
                 '`python -m os_pipeline.cat_panel.analyses`.')
    parts.append('- Full-design extension (all 194 conversations) requires only changing the '
                 '`--poc` flag to `--all`. Pipeline is otherwise unchanged.')

    text = '\n'.join(parts)
    REPORT_PATH.write_text(text, encoding='utf-8')
    print(f'wrote {REPORT_PATH}  ({len(text)} chars)')


if __name__ == '__main__':
    main()
