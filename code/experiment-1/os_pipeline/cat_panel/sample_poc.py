"""Stratified PoC sampler for CAT-Panel.

Selects N_PER_FAMILY users from each of the 4 persona families and
includes BOTH of each user's conversations (their Persona round AND
their GPT round). This preserves the within-subject design that the
larger analyses need.

For N_PER_FAMILY = 5: 5 users x 4 families x 2 rounds = 40 conversations.

Sampling is seeded so the same set is reproducible across reruns.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'
OUT.mkdir(parents=True, exist_ok=True)

FM = {'Divergent':'Divergent','Convergent':'Convergent',
      'strictly rational':'Rational','bounded rationality':'BoundedRational',
      'GPT':'GPT'}
FAMILIES = ('Divergent', 'Convergent', 'Rational', 'BoundedRational')

N_PER_FAMILY = 5    # users per family
SEED = 7


def sample(verbose: bool = True) -> list[int]:
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    logs['family'] = logs['Persona_type'].map(FM)

    # Each user has exactly two conversations (one Persona, one GPT)
    # Their persona-family is the non-GPT family across their rounds.
    user_family = (logs[logs.family != 'GPT']
                   .groupby('User_id')['family'].first())

    rng = np.random.default_rng(SEED)
    selected_users: list[int] = []
    for fam in FAMILIES:
        pool = user_family[user_family == fam].index.tolist()
        if len(pool) < N_PER_FAMILY:
            if verbose:
                print(f'[sample_poc] family {fam} only has {len(pool)} users; '
                      f'taking all of them')
            pick = pool
        else:
            pick = sorted(rng.choice(pool, size=N_PER_FAMILY, replace=False).tolist())
        selected_users.extend(pick)
        if verbose:
            print(f'[sample_poc] {fam}: {pick}')

    # All conversations for those users — both rounds
    sub = logs[logs['User_id'].isin(selected_users)]
    conv_ids = sorted(sub['conversation_id'].unique().tolist())

    if verbose:
        # Sanity-check stratification
        ratify = (sub.groupby('conversation_id')
                  .agg(family=('family', 'first'),
                       persona_type=('Persona_type', 'first'),
                       user=('User_id', 'first'))
                  .reset_index())
        ratify['condition'] = np.where(ratify['persona_type']=='GPT','GPT','Persona')
        # For Persona-condition convs we use the persona family;
        # for GPT-condition convs the user's persona-family is the off-condition one
        ratify['user_family'] = ratify['user'].map(user_family.to_dict())
        print('\n--- conversation breakdown ---')
        print(ratify.groupby(['user_family','condition']).size().unstack(fill_value=0))
        print(f'\ntotal: {len(conv_ids)} conversations, '
              f'{len(selected_users)} users')

    # save sample metadata
    payload = {
        'seed': SEED,
        'n_per_family': N_PER_FAMILY,
        'selected_users': selected_users,
        'conversation_ids': conv_ids,
        'families_in_order': list(FAMILIES),
    }
    (OUT / 'poc_sample.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if verbose:
        print(f'\nwrote {OUT / "poc_sample.json"}')

    return conv_ids


if __name__ == '__main__':
    sample()
