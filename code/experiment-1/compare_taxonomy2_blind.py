"""
Compare condition-blind Taxonomy 2 predictions against the original
enriched-model predictions (which included a persona-family one-hot feature).

For each of the 7 constructs (exp, con, cri, cer, com, ref, prop), compute:
  - Per-participant assistant-side and user-side means under GPT vs Persona.
  - Paired Cohen's d_z (Persona - GPT), pooled and per-family.
  - Side-by-side numbers from the enriched model and the condition-blind model.

Writes:
  analysis_out/taxonomy2_blind_vs_enriched_dz.csv
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

OUT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1/analysis_out'

enriched = pd.read_csv(os.path.join(OUT, 'full_stance_predictions.csv'))
blind    = pd.read_csv(os.path.join(OUT, 'full_stance_predictions_condition_blind.csv'))
master   = pd.read_csv(os.path.join(OUT, 'master_conversations.csv'))

CONSTRUCTS = ['exp', 'con', 'cri', 'cer', 'com', 'ref', 'prop']

# Map each message_id to the user id via conversation_id <-> master.
cid_to_user = dict(zip(master['conversation_id'], master['user']))

def aggregate(df, side):
    """Return DataFrame of per-(user, condition, family) mean construct values
    for messages of the given speaker side ('assistant' or 'user')."""
    d = df[df['message_src'] == side].copy()
    d['user'] = d['conversation_id'].map(cid_to_user)
    d = d.dropna(subset=['user'])
    grouped = d.groupby(['user', 'condition', 'family'])[CONSTRUCTS].mean().reset_index()
    return grouped

def paired_dz(df_side, family_filter=None):
    """Compute paired Persona-GPT means and Cohen's d_z per construct.
    Returns a dict {construct: (mean_persona, mean_gpt, dz, n)}.
    If family_filter is given, only persona-arm users assigned that family
    contribute to the persona side; all 97 users contribute their GPT round
    when family_filter is None, otherwise the family_filter restricts to the
    persona arm only."""
    persona = df_side[df_side['condition'] == 'Persona']
    gpt     = df_side[df_side['condition'] == 'GPT']
    if family_filter:
        persona = persona[persona['family'] == family_filter]
        users   = sorted(set(persona['user']))
        gpt     = gpt[gpt['user'].isin(users)]
    # Inner-join on user
    persona = persona.set_index('user')[CONSTRUCTS]
    gpt     = gpt.set_index('user')[CONSTRUCTS]
    joined  = persona.join(gpt, lsuffix='_p', rsuffix='_g', how='inner')
    out = {}
    for c in CONSTRUCTS:
        diff = joined[f'{c}_p'] - joined[f'{c}_g']
        n = len(diff)
        if n < 2 or diff.std(ddof=1) == 0:
            out[c] = (joined[f'{c}_p'].mean(), joined[f'{c}_g'].mean(), float('nan'), n)
        else:
            dz = diff.mean() / diff.std(ddof=1)
            out[c] = (joined[f'{c}_p'].mean(), joined[f'{c}_g'].mean(), dz, n)
    return out

def collect(model_name, df):
    rows = []
    for side in ['assistant', 'user']:
        agg = aggregate(df, side)
        # Pooled
        d = paired_dz(agg, family_filter=None)
        for c in CONSTRUCTS:
            mp, mg, dz, n = d[c]
            rows.append({'model': model_name, 'side': side, 'cell': 'pooled',
                          'construct': c, 'mean_persona': mp, 'mean_gpt': mg,
                          'dz': dz, 'n': n})
        # Per-family
        for fam in ['Divergent', 'Convergent', 'Rational', 'BoundedRational']:
            d = paired_dz(agg, family_filter=fam)
            for c in CONSTRUCTS:
                mp, mg, dz, n = d[c]
                rows.append({'model': model_name, 'side': side, 'cell': fam,
                              'construct': c, 'mean_persona': mp, 'mean_gpt': mg,
                              'dz': dz, 'n': n})
    return pd.DataFrame(rows)

df_enr = collect('enriched',          enriched)
df_bld = collect('condition_blind',   blind)

# Side-by-side wide
wide = df_enr.merge(
    df_bld,
    on=['side', 'cell', 'construct'],
    suffixes=('_enr', '_blind'),
).drop(columns=['model_enr', 'model_blind'])

wide['dz_delta'] = wide['dz_blind'] - wide['dz_enr']

# Save full long form + the head-to-head wide form.
long = pd.concat([df_enr, df_bld], ignore_index=True)
long.to_csv(os.path.join(OUT, 'taxonomy2_dz_long.csv'), index=False)
wide.to_csv(os.path.join(OUT, 'taxonomy2_blind_vs_enriched_dz.csv'), index=False)

print('Wrote', os.path.join(OUT, 'taxonomy2_blind_vs_enriched_dz.csv'))

# Print the headline pooled assistant-side comparison.
print('\n--- Pooled assistant-side d_z, condition-blind vs enriched ---')
pool = wide[(wide['side'] == 'assistant') & (wide['cell'] == 'pooled')]
print(pool[['construct', 'dz_enr', 'dz_blind', 'dz_delta', 'n_enr']].to_string(index=False))

print('\n--- Pooled user-side d_z, condition-blind vs enriched ---')
pool = wide[(wide['side'] == 'user') & (wide['cell'] == 'pooled')]
print(pool[['construct', 'dz_enr', 'dz_blind', 'dz_delta', 'n_enr']].to_string(index=False))

print('\n--- Divergent (high-N) user-side d_z, blind vs enriched ---')
fam = wide[(wide['side'] == 'user') & (wide['cell'] == 'Divergent')]
print(fam[['construct', 'dz_enr', 'dz_blind', 'dz_delta', 'n_enr']].to_string(index=False))
