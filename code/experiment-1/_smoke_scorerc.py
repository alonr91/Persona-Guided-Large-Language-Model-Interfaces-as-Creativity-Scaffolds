"""Smoke test: run Scorer C on a single B-subset episode and print the result.
If this succeeds, the full 50-episode run is safe."""
import pandas as pd
from pathlib import Path
from os_pipeline.regulated.scorer import _score_episode

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
sample = pd.read_csv(ROOT / 'regulated_llm_reanalysis' / '_scoring_sample.csv')
sub = sample[sample['scorer_B'] == True].head(1)
ep = sub.iloc[0]
print(f'episode_id={ep["episode_id"]}  type={ep["episode_type"]}  '
      f'turns={ep["num_turns"]}  words={len(str(ep["episode_text_masked"]).split())}')

obj, dbg = _score_episode(ep, scorer='C')
print('---')
print('valid_json :', dbg.get('valid_json'))
print('parse_path :', dbg.get('parse_path'))
print('parse_err  :', dbg.get('parse_error'))
print('n_attempts :', dbg.get('n_attempts'))
if obj is not None:
    print(f'n_scores   : {len(obj.scores)}')
    for cs in obj.scores[:3]:
        print(f'  - {cs.criterion}: score={cs.score_0_4} '
              f'usable={cs.usable_for_inference} '
              f'evidence={(cs.evidence_quotes or [None])[0]!r:.80}')
else:
    print('OBJ IS NONE -- check raw output')
    print('raw[:800]:', (dbg.get('raw_output') or '')[:800])
