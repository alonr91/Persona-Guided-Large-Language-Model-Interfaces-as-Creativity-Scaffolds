"""
Condition-blind re-run of Taxonomy 2.

The original propagate_stance.py classifier included a persona-family one-hot
(GPT, Divergent, Convergent, Rational, BoundedRational) in its feature vector,
which the v12 chapter flagged as a possible leakage risk. This script re-trains
the same Gradient-Boosting Regressors and Classifiers using ONLY:

    SBERT embedding (384d) + lexical cues (9d) + speaker one-hot (assistant/user) + turn-position (1d)

i.e., without any persona-family information. It saves predictions to
`analysis_out/full_stance_predictions_condition_blind.csv` and writes a
diagnostics table `analysis_out/classifier_quality_condition_blind.csv`.

Side-by-side d_z comparison vs the original model is then computed and saved
to `analysis_out/taxonomy2_blind_vs_enriched_dz.csv`.
"""
import os, sys, warnings, re
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'analysis_out')

# --------------------------------------------------------------
# 1. Reconstruct the per-message ordering used by propagate_stance.py
#    by reading the existing predictions file (which preserves the same row
#    order as msg_embeddings.npy).
# --------------------------------------------------------------
preds_orig = pd.read_csv(os.path.join(OUT, 'full_stance_predictions.csv'))
assert len(preds_orig) == 3412

# Pull message text by joining on message_id with the logs CSV
logs = pd.read_csv(os.path.join(ROOT, 'Experiment1_logs.csv'))
text_by_mid = dict(zip(logs['message_id'], logs['message']))
preds_orig['message_text'] = preds_orig['message_id'].map(text_by_mid)
missing = preds_orig['message_text'].isna().sum()
if missing:
    print(f'WARNING: {missing} messages without text; falling back to empty string')
    preds_orig['message_text'] = preds_orig['message_text'].fillna('')

# --------------------------------------------------------------
# 2. Build the condition-blind feature matrix.
# --------------------------------------------------------------
E = np.load(os.path.join(OUT, 'msg_embeddings.npy'))   # (3412, 384)
assert E.shape == (3412, 384)

def feat_side(msg):
    s = str(msg) if msg is not None else ''
    return np.array([
        len(re.findall(r'\w+', s)),
        s.count('?'),
        int('what if' in s.lower()),
        int('criter' in s.lower()),
        int('anchor' in s.lower() or 'straightforward' in s.lower()),
        int('reflect' in s.lower()),
        int("let's" in s.lower() or 'let us' in s.lower()),
        int(any(h in s.lower() for h in ['maybe','perhaps','could','might','possibly','suggest'])),
        int(any(h in s.lower() for h in ['must','will','need to','should','definitely','certainly'])),
    ], dtype=float)

SIDE = np.vstack(preds_orig['message_text'].apply(feat_side).values)
role_oh = pd.get_dummies(preds_orig['message_src']).reindex(
    columns=['assistant', 'user'], fill_value=0
).values.astype(float)
pos = preds_orig['turn_frac'].fillna(0.0).values.reshape(-1, 1)

# Condition-blind feature vector:
X_blind = np.concatenate([E, SIDE, role_oh, pos], axis=1)
print('Condition-blind X shape:', X_blind.shape)

# --------------------------------------------------------------
# 3. Load gold sample and map to row indices.
# --------------------------------------------------------------
gold   = pd.read_csv(os.path.join(OUT, 'claude_stance_labels.csv'))
sample = pd.read_csv(os.path.join(OUT, 'gold_sample_to_code.csv'))
lab = sample.merge(gold, left_on='message_id', right_on='mid', how='left')
mid_to_row = {m: i for i, m in enumerate(preds_orig['message_id'].values)}
lab['row'] = lab['message_id'].map(mid_to_row)
lab = lab.dropna(subset=['row']).copy()
lab['row'] = lab['row'].astype(int)
X_train = X_blind[lab['row'].values]
print('Gold-sample training set:', X_train.shape)

# --------------------------------------------------------------
# 4. Train + cross-validate per-construct ordinal regressors.
# --------------------------------------------------------------
score_cols = ['exp', 'con', 'cri', 'cer', 'com', 'ref', 'prop']
pred_blind = {}
cv_r2 = {}
for c in score_cols:
    y = lab[c].astype(float).values
    m = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    cvp = cross_val_predict(m, X_train, y, cv=KFold(5, shuffle=True, random_state=0))
    cv_r2[c] = r2_score(y, cvp)
    m.fit(X_train, y)
    pred_blind[c] = np.clip(m.predict(X_blind), 0, 3)
    print(f'  {c:>5}: CV R² = {cv_r2[c]:+.3f}')

# --------------------------------------------------------------
# 5. Tone + qtype classifiers (categoricals).
# --------------------------------------------------------------
def fit_cat(col):
    y = lab[col].fillna('none').astype(str).values
    m = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    try:
        acc = cross_val_score(
            m, X_train, y, cv=KFold(5, shuffle=True, random_state=0), scoring='accuracy'
        ).mean()
    except Exception:
        acc = float('nan')
    m.fit(X_train, y)
    return m.predict(X_blind), acc

tone_pred, tone_acc = fit_cat('tone')
qtype_pred, qtype_acc = fit_cat('qtype')
print(f'  tone CV acc  = {tone_acc:.3f}')
print(f'  qtype CV acc = {qtype_acc:.3f}')

# --------------------------------------------------------------
# 6. Assemble + save.
# --------------------------------------------------------------
out = pd.DataFrame(pred_blind)
out['tone_pred']       = tone_pred
out['qtype_pred']      = qtype_pred
out['message_id']      = preds_orig['message_id'].values
out['conversation_id'] = preds_orig['conversation_id'].values
out['message_src']     = preds_orig['message_src'].values
out['turn_frac']       = preds_orig['turn_frac'].values
out['family']          = preds_orig['family'].values
out['condition']       = preds_orig['condition'].values
out['Persona_type']    = preds_orig['Persona_type'].values
out['challenge']       = preds_orig['challenge'].values

out_csv = os.path.join(OUT, 'full_stance_predictions_condition_blind.csv')
out.to_csv(out_csv, index=False)
print('\nSaved:', out_csv)

qual = pd.DataFrame({
    'metric':      score_cols + ['tone_cat', 'qtype_cat'],
    'cv_r2_or_acc': [cv_r2[c] for c in score_cols] + [tone_acc, qtype_acc],
})
qual_csv = os.path.join(OUT, 'classifier_quality_condition_blind.csv')
qual.to_csv(qual_csv, index=False)
print('Saved:', qual_csv)
print('\nCondition-blind CV quality:')
print(qual.to_string(index=False))
