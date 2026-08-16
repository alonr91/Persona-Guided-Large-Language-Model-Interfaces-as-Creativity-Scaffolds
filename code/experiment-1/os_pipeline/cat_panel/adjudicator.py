"""Stage C — Adjudication: paraphrase fusion + cross-judge consensus.

Two stages:
  1. Within-judge A/B fusion. For each (judge, conv, dim), combine the
     two paraphrase scores via the rule:
        - if both non-null and |A - B| <= 1: mean (rounded to nearest int)
        - if both non-null and |A - B| >= 2: flag as 'paraphrase_disagree';
          use the mean but mark adjudicated=False
        - if one is null: use the non-null one
        - if both null: leave null

  2. Cross-judge consensus. For each (conv, dim), compute the consensus
     score as the mean of the 4 adjudicated judge scores, plus the
     inter-judge agreement metrics: ICC(2,k), quadratic-weighted kappa
     (averaged over the 6 pairwise kappas), and standard deviation.

The headline validity gate is V1: ICC(2,k) >= 0.50 per dimension =
retain for primary inference; 0.30-0.49 = exploratory; <0.30 = drop.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.dimensions import DIM_NAMES, DIM_LABELS
from os_pipeline.cat_panel.personas import JUDGE_IDS
from os_pipeline.regulated.cross_model_agreement import quadratic_weighted_kappa

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'
OUT.mkdir(parents=True, exist_ok=True)

# Use parquet if pyarrow available, otherwise CSV. Matches scorer.py.
try:
    import pyarrow  # noqa: F401
    _USE_PARQUET = True
    RAW_PATH = OUT / 'panel_scores_raw.parquet'
    _EXT = '.parquet'
except Exception:
    _USE_PARQUET = False
    RAW_PATH = OUT / 'panel_scores_raw.csv'
    _EXT = '.csv'


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, basename: str) -> Path:
    path = OUT / (basename + _EXT)
    if _USE_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# Stage 1: A/B paraphrase fusion per (judge, conv, dim)
# ----------------------------------------------------------------------

def fuse_paraphrases(raw: pd.DataFrame) -> pd.DataFrame:
    """Combine paraphrase A and B per (judge, conv, dim)."""
    raw = raw.copy()
    raw['score_1_7'] = pd.to_numeric(raw['score_1_7'], errors='coerce')

    # Pivot: index = (conv, judge, dim), cols = paraphrase
    wide = raw.pivot_table(
        index=['conversation_id', 'judge_id', 'dimension'],
        columns='paraphrase', values='score_1_7', aggfunc='first',
    ).reset_index()
    # rename so we have explicit columns even if one of A/B is missing
    if 'A' not in wide.columns: wide['A'] = np.nan
    if 'B' not in wide.columns: wide['B'] = np.nan

    def _fuse(row):
        a, b = row['A'], row['B']
        if pd.isna(a) and pd.isna(b):
            return pd.Series({'score_adj': np.nan,
                              'paraphrase_disagree': False,
                              'paraphrase_diff': np.nan})
        if pd.isna(a):
            return pd.Series({'score_adj': float(b),
                              'paraphrase_disagree': False,
                              'paraphrase_diff': np.nan})
        if pd.isna(b):
            return pd.Series({'score_adj': float(a),
                              'paraphrase_disagree': False,
                              'paraphrase_diff': np.nan})
        diff = abs(a - b)
        return pd.Series({
            'score_adj': float((a + b) / 2.0),
            'paraphrase_disagree': bool(diff >= 2),
            'paraphrase_diff': float(diff),
        })

    fused_cols = wide.apply(_fuse, axis=1)
    out = pd.concat([wide[['conversation_id', 'judge_id', 'dimension', 'A', 'B']],
                     fused_cols], axis=1)
    return out


# ----------------------------------------------------------------------
# Stage 2: Cross-judge consensus per (conv, dim)
# ----------------------------------------------------------------------

def _icc_2_k(ratings: np.ndarray) -> float:
    """ICC(2,k) — two-way random effects, average measures, absolute agreement.

    `ratings` is an (n_targets, k_raters) matrix. NaN-rows dropped row-wise.
    Formula per Shrout & Fleiss (1979) / McGraw & Wong (1996).
    Returns NaN if not enough data.
    """
    ratings = np.asarray(ratings, dtype=float)
    # drop rows with any NaN
    ratings = ratings[~np.isnan(ratings).any(axis=1)]
    n, k = ratings.shape
    if n < 3 or k < 2:
        return float('nan')

    mean_targets = ratings.mean(axis=1)   # per-target mean across raters
    mean_raters  = ratings.mean(axis=0)   # per-rater mean across targets
    grand_mean   = ratings.mean()

    # Sums of squares
    ss_b_targets = k * ((mean_targets - grand_mean) ** 2).sum()
    ss_b_raters  = n * ((mean_raters  - grand_mean) ** 2).sum()
    ss_total     = ((ratings - grand_mean) ** 2).sum()
    ss_err       = ss_total - ss_b_targets - ss_b_raters

    ms_b_targets = ss_b_targets / max(1, n - 1)
    ms_b_raters  = ss_b_raters  / max(1, k - 1)
    ms_err       = ss_err       / max(1, (n - 1) * (k - 1))

    denom = ms_b_targets + (ms_b_raters - ms_err) / n
    if denom <= 0:
        return float('nan')
    return float((ms_b_targets - ms_err) / denom)


def cross_judge_consensus(fused: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_item, per_dim).

    per_item: one row per (conv, dim) with the 4-judge consensus and per-judge cells.
    per_dim:  one row per dim with ICC(2,k), mean pairwise weighted-kappa,
              n complete, retention recommendation.
    """
    # wide: index = (conv, dim), cols = judges
    wide = fused.pivot_table(
        index=['conversation_id', 'dimension'],
        columns='judge_id', values='score_adj', aggfunc='first',
    ).reset_index()
    for j in JUDGE_IDS:
        if j not in wide.columns: wide[j] = np.nan

    # consensus = mean of judge scores (NaN-aware)
    judge_cols = list(JUDGE_IDS)
    wide['consensus'] = wide[judge_cols].mean(axis=1, skipna=True)
    wide['sd_across_judges'] = wide[judge_cols].std(axis=1, skipna=True, ddof=1)
    wide['n_judges_with_score'] = wide[judge_cols].notna().sum(axis=1)

    # per-dimension psychometrics
    per_dim_rows = []
    for dim in DIM_NAMES:
        sub = wide[wide['dimension'] == dim]
        ratings = sub[judge_cols].values  # (n_convs, 4)

        icc = _icc_2_k(ratings)

        # average pairwise quadratic-weighted kappa (Likert 1-7 → integers)
        pair_kappas = []
        for j1, j2 in combinations(judge_cols, 2):
            v = sub[[j1, j2]].dropna().astype(float)
            if len(v) < 5:
                continue
            v1 = np.round(v[j1]).clip(1, 7).astype(int).values - 1  # 0..6
            v2 = np.round(v[j2]).clip(1, 7).astype(int).values - 1
            k = quadratic_weighted_kappa(v1, v2, n_classes=7)
            if not np.isnan(k):
                pair_kappas.append(k)

        mean_wk = float(np.mean(pair_kappas)) if pair_kappas else float('nan')
        n_complete = int(np.sum(~np.isnan(ratings).any(axis=1)))

        if icc >= 0.50:    retention = 'primary'
        elif icc >= 0.30:  retention = 'exploratory'
        elif np.isnan(icc): retention = 'insufficient_data'
        else:              retention = 'drop'

        per_dim_rows.append(dict(
            dimension=dim,
            label=DIM_LABELS[dim],
            n_complete_convs=n_complete,
            n_total_convs=len(sub),
            icc_2_k=icc,
            mean_pairwise_kappa_qw=mean_wk,
            n_kappa_pairs=len(pair_kappas),
            retention=retention,
        ))

    per_dim = pd.DataFrame(per_dim_rows)
    return wide, per_dim


# ----------------------------------------------------------------------
# Disagreement table — which judges agree, which disagree
# ----------------------------------------------------------------------

def judge_disagreement_table(per_item: pd.DataFrame) -> pd.DataFrame:
    """For each (dim x judge-pair) compute mean |Δ| and Spearman ρ across convs."""
    judge_cols = list(JUDGE_IDS)
    rows = []
    for dim in DIM_NAMES:
        sub = per_item[per_item['dimension'] == dim]
        for j1, j2 in combinations(judge_cols, 2):
            v = sub[[j1, j2]].dropna().astype(float)
            if len(v) < 5:
                continue
            d = (v[j1] - v[j2]).abs()
            from scipy.stats import spearmanr
            rho, p = spearmanr(v[j1].values, v[j2].values)
            rows.append(dict(
                dimension=dim,
                label=DIM_LABELS[dim],
                judge_pair=f'{j1}-{j2}',
                n=len(v),
                mean_abs_diff=float(d.mean()),
                spearman_rho=float(rho) if not np.isnan(rho) else np.nan,
                spearman_p=float(p) if not np.isnan(p) else np.nan,
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def adjudicate(verbose: bool = True) -> dict:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f'no raw scores at {RAW_PATH}')

    raw = _read(RAW_PATH)
    if verbose:
        print(f'[adjudicate] loaded {len(raw)} raw score rows')

    fused = fuse_paraphrases(raw)
    fp = _write(fused, 'panel_scores_adjudicated')
    if verbose:
        print(f'[adjudicate] wrote {fp.name} ({len(fused)} rows)')

    per_item, per_dim = cross_judge_consensus(fused)
    pip = _write(per_item, 'panel_scores_consensus')
    per_dim.to_csv(OUT / 'consensus_per_dim.csv', index=False)
    if verbose:
        print(f'[adjudicate] wrote {pip.name} ({len(per_item)} rows)')
        print(f'[adjudicate] wrote consensus_per_dim.csv')
        print('\n--- per-dimension consensus (V1 gate) ---')
        print(per_dim[['label','n_complete_convs','icc_2_k',
                       'mean_pairwise_kappa_qw','retention']]
              .to_string(index=False))

    disagree = judge_disagreement_table(per_item)
    disagree.to_csv(OUT / 'disagreement_table.csv', index=False)
    if verbose:
        print(f'\n[adjudicate] wrote disagreement_table.csv '
              f'({len(disagree)} rows)')

    return dict(
        n_raw=len(raw),
        n_fused=len(fused),
        n_per_item=len(per_item),
        retained_primary=int((per_dim['retention']=='primary').sum()),
        retained_exploratory=int((per_dim['retention']=='exploratory').sum()),
        dropped=int((per_dim['retention']=='drop').sum()),
    )


if __name__ == '__main__':
    adjudicate()
