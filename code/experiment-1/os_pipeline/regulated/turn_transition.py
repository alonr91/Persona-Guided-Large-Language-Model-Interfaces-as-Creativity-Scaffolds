"""Stage 6 — Turn transition table → 06_turn_transition_table.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'


def main() -> None:
    turn_path = OUT / 'turn_table.parquet'
    if turn_path.exists():
        turns = pd.read_parquet(turn_path)
    else:
        turns = pd.read_csv(OUT / 'turn_table.csv')
    eps = pd.read_csv(OUT / '02_episode_table.csv')
    adj = pd.read_csv(OUT / '05_episode_rubric_scores_adjudicated.csv')

    # episode score pivot per conversation × criterion
    epscore = adj.pivot_table(index=['conversation_id','episode_id'],
                               columns='criterion', values='final_score',
                               aggfunc='first').reset_index()
    eps_aug = eps.merge(epscore, on=['conversation_id','episode_id'], how='left')

    # build transition rows
    rows = []
    for cid, g in turns.groupby('conversation_id', sort=False):
        g = g.sort_values('turn_index').reset_index(drop=True)
        # map turn_index -> episode
        eps_this = eps_aug[eps_aug.conversation_id == cid].sort_values('start_turn')
        def _ep_at(ti: int):
            m = eps_this[(eps_this.start_turn <= ti) & (eps_this.end_turn >= ti)]
            return m.iloc[0] if len(m) else None
        for i in range(1, len(g)):
            if g.iloc[i]['speaker_masked'] == 'user' and g.iloc[i-1]['speaker_masked'] == 'assistant':
                ep_a = _ep_at(int(g.iloc[i-1]['turn_index']))
                ep_u = _ep_at(int(g.iloc[i]['turn_index']))
                if ep_a is None or ep_u is None: continue
                rows.append(dict(
                    conversation_id=int(cid),
                    transition_idx=int(i),
                    turn_index_assistant=int(g.iloc[i-1]['turn_index']),
                    turn_index_user=int(g.iloc[i]['turn_index']),
                    participant_id=int(g.iloc[i]['participant_id']),
                    condition=g.iloc[i]['condition_original'],
                    family=g.iloc[i]['persona_family_original'],
                    turn_index_normalized=float(i / max(1, len(g)-1)),
                    assistant_prev_episode_id=ep_a['episode_id'],
                    user_next_episode_id=ep_u['episode_id'],
                    # assistant regulation scores
                    a_exploration_opening=ep_a.get('exploration_opening'),
                    a_reframing_quality=ep_a.get('reframing_quality'),
                    a_agency_preservation=ep_a.get('agency_preservation'),
                    a_stance_integrity=ep_a.get('stance_integrity'),
                    a_timing_fit=ep_a.get('timing_fit'),
                    # user next-episode uptake
                    u_coregulation_uptake=ep_u.get('coregulation_uptake'),
                    u_agency_preservation=ep_u.get('agency_preservation'),
                    u_implementation_grounding=ep_u.get('implementation_grounding'),
                    u_evaluative_discipline=ep_u.get('evaluative_discipline'),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / '06_turn_transition_table.csv', index=False)
    print(f'wrote {OUT / "06_turn_transition_table.csv"} ({len(df)} transitions)')


if __name__ == '__main__':
    main()
