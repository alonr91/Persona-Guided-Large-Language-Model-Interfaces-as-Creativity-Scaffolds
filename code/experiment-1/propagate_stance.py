"""
Step: Train classifiers on the 300 Claude-coded gold turns using SBERT features
(+ compact side features), then predict all 3,412 turns. Save predictions.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, r2_score

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT,'analysis_out')

# load
logs = pd.read_csv(os.path.join(ROOT,'Experiment1_logs_cleaned_keepable_paired_translated.csv'))
logs = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)
fam_map = {'Divergent':'Divergent','Convergent':'Convergent',
           'strictly rational':'Rational','bounded rationality':'BoundedRational',
           'GPT':'GPT'}
logs['family'] = logs['Persona_type'].map(fam_map)
logs['condition'] = np.where(logs['Persona_type']=='GPT','GPT','Persona')
E = np.load(os.path.join(OUT,'msg_embeddings.npy'))  # (3412, 384)

# add side features
import re
def feat_side(msg):
    s = str(msg) if not pd.isna(msg) else ''
    words = len(re.findall(r'\w+', s))
    return np.array([
        words,
        s.count('?'),
        int('what if' in s.lower()),
        int('criter' in s.lower()),
        int('anchor' in s.lower() or 'straightforward' in s.lower()),
        int('reflect' in s.lower()),
        int("let's" in s.lower() or 'let us' in s.lower()),
        int(any(h in s.lower() for h in ['maybe','perhaps','could','might','possibly','suggest'])),
        int(any(h in s.lower() for h in ['must','will','need to','should','definitely','certainly'])),
    ], dtype=float)

SIDE = np.vstack(logs['message'].apply(feat_side).values)
# also: role, condition one-hots
role_oh = pd.get_dummies(logs['message_src']).reindex(columns=['assistant','user'], fill_value=0).values.astype(float)
fam_oh  = pd.get_dummies(logs['family']).reindex(columns=['GPT','Divergent','Convergent','Rational','BoundedRational'], fill_value=0).values.astype(float)
pos = logs['turn_frac'].values.reshape(-1,1)
# turn_idx within conversation as rel
rel_idx = pos.copy()

X_full = np.concatenate([E, SIDE, role_oh, fam_oh, rel_idx], axis=1)
print('X_full shape:', X_full.shape)

# load labels
gold = pd.read_csv(os.path.join(OUT,'claude_stance_labels.csv'))
sample = pd.read_csv(os.path.join(OUT,'gold_sample_to_code.csv'))
# join on message_id
lab = sample.merge(gold, left_on='message_id', right_on='mid', how='left')
# get row indices in logs
mid_to_row = {m:i for i,m in enumerate(logs['message_id'].values)}
lab['row'] = lab['message_id'].map(mid_to_row)
lab = lab.dropna(subset=['row']); lab['row']=lab['row'].astype(int)
X_train_idx = lab['row'].values
X_train = X_full[X_train_idx]
print('train:', X_train.shape, 'labels:', len(lab))

# ----- train ordinal/numeric regressors for 0-3 scores -----
score_cols = ['exp','con','cri','cer','com','ref','prop']
pred_mat = {}
cv_r2 = {}
for c in score_cols:
    y = lab[c].astype(float).values
    m = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    # CV R^2
    cvp = cross_val_predict(m, X_train, y, cv=KFold(5, shuffle=True, random_state=0))
    r2 = r2_score(y, cvp)
    cv_r2[c] = r2
    m.fit(X_train, y)
    pred_mat[c] = np.clip(m.predict(X_full), 0, 3)
    print(f'  {c}: CV R²={r2:.3f}')

# ----- categorical: tone, qtype -----
def fit_cat(col):
    y = lab[col].fillna('none').astype(str).values
    m = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    try:
        acc = cross_val_score(m, X_train, y, cv=KFold(5, shuffle=True, random_state=0), scoring='accuracy').mean()
    except Exception:
        acc = np.nan
    m.fit(X_train, y)
    preds = m.predict(X_full)
    return preds, acc

tone_pred, tone_acc = fit_cat('tone')
qtype_pred, qtype_acc = fit_cat('qtype')
print(f'  tone CV acc={tone_acc:.3f}   qtype CV acc={qtype_acc:.3f}')

# ----- assemble full predictions -----
full = pd.DataFrame(pred_mat)
full['tone_pred']  = tone_pred
full['qtype_pred'] = qtype_pred
full['message_id'] = logs['message_id'].values
full['conversation_id'] = logs['conversation_id'].values
full['message_src'] = logs['message_src'].values
full['turn_frac'] = logs['turn_frac'].values
full['family'] = logs['family'].values
full['condition'] = logs['condition'].values
full['Persona_type'] = logs['Persona_type'].values
full['challenge'] = logs['Corrected Challenge type'].values
full.to_csv(os.path.join(OUT,'full_stance_predictions.csv'), index=False)

# save classifier quality table
qual = pd.DataFrame({'metric': score_cols + ['tone_cat','qtype_cat'],
                     'cv_r2_or_acc': [cv_r2[c] for c in score_cols] + [tone_acc, qtype_acc]})
qual.to_csv(os.path.join(OUT,'classifier_quality.csv'), index=False)
print('\nCV quality:'); print(qual.to_string(index=False))
print('\nSaved:', os.path.join(OUT,'full_stance_predictions.csv'))
