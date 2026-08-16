"""
Step 1: Embed all messages once with local SBERT. Saves to analysis_out/.
Also: build a stratified sample of ~400 turns for Claude gold-label coding.
"""
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT,'analysis_out')
os.makedirs(OUT, exist_ok=True)

logs = pd.read_csv(os.path.join(ROOT,'Experiment1_logs_cleaned_keepable_paired_translated.csv'))
logs = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)

# persona family per conversation
fam_map = {'Divergent':'Divergent','Convergent':'Convergent',
           'strictly rational':'Rational','bounded rationality':'BoundedRational',
           'GPT':'GPT'}
logs['family'] = logs['Persona_type'].map(fam_map)
logs['condition'] = np.where(logs['Persona_type']=='GPT','GPT','Persona')

# --- SBERT embeddings ---
from sentence_transformers import SentenceTransformer
print('loading SBERT...')
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('encoding', len(logs), 'messages...')
texts = logs['message'].fillna('').astype(str).tolist()
E = m.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
np.save(os.path.join(OUT,'msg_embeddings.npy'), E.astype('float32'))
print('saved embeddings', E.shape)

# --- stratified sample for Claude gold coding ---
# strata: condition (GPT, Persona) × family × message_src × position (early<.33, mid, late>.67)
logs['pos'] = pd.cut(logs['turn_frac'], bins=[-0.01,0.33,0.67,1.01], labels=['early','mid','late'])
rng = np.random.default_rng(0)
samples = []
for (cond, fam, src, pos), g in logs.groupby(['condition','family','message_src','pos'], observed=True):
    # pick up to 10 per stratum
    k = min(10, len(g))
    if k==0: continue
    idx = rng.choice(g.index, size=k, replace=False)
    samples.extend(idx)
sample_df = logs.loc[samples].copy()
# cap at 400 by random selection
if len(sample_df) > 400:
    sample_df = sample_df.sample(n=400, random_state=1)
sample_df = sample_df.sort_values('message_id').reset_index(drop=True)
print('sample size:', len(sample_df))
sample_df[['message_id','conversation_id','message_src','turn_frac','pos',
           'Persona_type','family','condition','Corrected Challenge type','message']].to_csv(
    os.path.join(OUT,'gold_sample_to_code.csv'), index=False)
print('saved gold_sample_to_code.csv')
print(sample_df.groupby(['condition','family','message_src']).size())
