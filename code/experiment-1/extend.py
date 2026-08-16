"""
OpenAI-powered extension analyses grounded in the thesis's literature review.

Operationalized from Chapter 2:
  - 2.3.3  White (2003) expansion vs contraction stance coding (turn level)
  - 2.8    Persona fidelity manipulation check (does Adi expand? Nitzan contract?)
  - 2.4.2  Anchoring / fixation index via embeddings
  - 2.4.3  Critique tone grading
  - 2.4.5  Authority / certainty signaling
  - 2.1.2  Creativity vs innovation: per-idea originality / value / feasibility (Shah, Acar)
  - RQ3    Idea-portfolio extraction + embedding distance (paper-stated target)
  - 2.10   Validity guardrails: cache, temp=0, stability subsample, blind reruns

Reads OPENAI_API_KEY and OPENAI_MODEL from environment. REFUSES to hardcode keys.
Caches every API call by content hash in analysis_out/api_cache/ (safe to re-run).
"""
import os, sys, json, hashlib, time, warnings, re
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'analysis_out')
FIG  = os.path.join(ROOT, 'figures')
CACHE = os.path.join(OUT, 'api_cache')
os.makedirs(CACHE, exist_ok=True)

API_KEY = os.environ.get('OPENAI_API_KEY', '').strip()
MODEL   = os.environ.get('OPENAI_MODEL', 'gpt-5.4-nano-2026-03-17').strip()
EMB_MODEL = os.environ.get('OPENAI_EMB_MODEL', 'text-embedding-3-small').strip()
if not API_KEY:
    print("ERROR: OPENAI_API_KEY env var not set.")
    print("  Rotate your key at https://platform.openai.com/api-keys")
    print('  Then: setx OPENAI_API_KEY "sk-proj-..."  and open a new shell.')
    sys.exit(2)
if not API_KEY.startswith('sk-'):
    print("ERROR: OPENAI_API_KEY does not look like a real key.")
    sys.exit(2)

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.run([sys.executable,'-m','pip','install','--quiet','openai>=1.30'])
    from openai import OpenAI

client = OpenAI(api_key=API_KEY)
print(f'Using chat model: {MODEL}   embedding model: {EMB_MODEL}')

# ------------------ caching ------------------
def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]

def cached_chat(messages, schema_tag, model=None, temperature=0, max_retries=4):
    model = model or MODEL
    key = _hash({'m':model,'t':temperature,'msg':messages,'tag':schema_tag})
    path = os.path.join(CACHE, f'chat_{schema_tag}_{key}.json')
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    def _parse_json_loose(txt):
        if not txt: return None
        try: return json.loads(txt)
        except Exception: pass
        # strip code fences
        m = re.search(r'```(?:json)?\s*(.*?)```', txt, re.S)
        if m:
            try: return json.loads(m.group(1))
            except Exception: pass
        # find first { ... last }
        i,j = txt.find('{'), txt.rfind('}')
        if i>=0 and j>i:
            try: return json.loads(txt[i:j+1])
            except Exception: pass
        return None
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages)
            # temperature may be unsupported by some newer models; try with, fall back without
            try:
                r = client.chat.completions.create(
                    **kwargs, temperature=temperature,
                    response_format={'type':'json_object'})
            except Exception as e1:
                if 'response_format' in str(e1) or 'temperature' in str(e1) or 'unsupported' in str(e1).lower():
                    # retry without response_format and/or temperature
                    try:
                        r = client.chat.completions.create(**kwargs, temperature=temperature)
                    except Exception:
                        r = client.chat.completions.create(**kwargs)
                else:
                    raise
            txt = r.choices[0].message.content
            data = _parse_json_loose(txt)
            if data is None:
                raise ValueError(f'non-JSON response: {txt[:200]}')
            json.dump(data, open(path,'w',encoding='utf-8'), ensure_ascii=False)
            return data
        except Exception as e:
            if attempt == max_retries-1:
                print(f'  chat fail {schema_tag}: {e}')
                return None
            time.sleep(2**attempt)

def cached_embed(texts):
    # batch; cache per text
    out = [None]*len(texts)
    todo_idx, todo_txt = [], []
    for i,t in enumerate(texts):
        key = _hash({'m':EMB_MODEL,'t':t})
        path = os.path.join(CACHE, f'emb_{key}.json')
        if os.path.exists(path):
            out[i] = np.array(json.load(open(path)))
        else:
            todo_idx.append(i); todo_txt.append(t)
    # batch in groups of 96
    for off in range(0, len(todo_txt), 96):
        chunk = todo_txt[off:off+96]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        for j,emb in enumerate(r.data):
            i = todo_idx[off+j]
            v = np.array(emb.embedding)
            out[i] = v
            key = _hash({'m':EMB_MODEL,'t':texts[i]})
            json.dump(v.tolist(), open(os.path.join(CACHE,f'emb_{key}.json'),'w'))
    return np.vstack(out)

# ------------------ load data ------------------
logs = pd.read_csv(os.path.join(ROOT,'Experiment1_logs_cleaned_keepable_paired.csv'))
users = pd.read_excel(os.path.join(ROOT,'Users_keepable_paired_only_corrected_audit.xlsx'),
                      sheet_name='corrected_users')
users.columns = [c.strip() for c in users.columns]
logs = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)

conv_meta = (logs.groupby('conversation_id')
             .agg(user=('User_id','first'),
                  persona_type=('Persona_type','first'),
                  challenge=('Corrected Challenge type','first'))
             .reset_index())
conv_meta['condition'] = np.where(conv_meta['persona_type']=='GPT','GPT','Persona')
fam_map = {'Divergent':'Divergent','Convergent':'Convergent',
           'strictly rational':'Rational','bounded rationality':'BoundedRational',
           'GPT':'GPT'}
conv_meta['family'] = conv_meta['persona_type'].map(fam_map)

print(f'{len(logs)} messages across {logs["conversation_id"].nunique()} conversations')

# ===================================================================
# 1. TURN-LEVEL STANCE CODING (White 2003; §2.3.3 + §2.4)
# ===================================================================
STANCE_PROMPT = """You code turns of a human-LLM creative dialogue for conversational stance.
Apply White (2003) appraisal theory: EXPANSION resources keep alternatives live (open 'what if', reframing invitations, low-commitment modality, ambiguity tolerance, analogical extension, clarifying questions that open options). CONTRACTION resources narrow and stabilize (criteria setting, comparative evaluation, constraint injection, higher-certainty language, pruning, decisive claims, commitment).

For the given turn, return a single JSON object with these integer fields, each 0-3 (0=absent, 3=strong):
- expansion: density of expansion cues
- contraction: density of contraction cues
- critique: whether the turn challenges or pushes back on a prior idea (0 none, 1 soft, 2 clear, 3 blunt)
- critique_tone: IF critique>0 then one of "constructive","neutral","blunt"; ELSE "none"
- certainty: epistemic certainty language (0 hedged, 3 decisive/authoritative)
- commit: commitment/closure moves (0-3)
- reframe: shifts the framing of the problem (0-3)
- proposes_new_idea: introduces a new concrete candidate (0-3)
- question_type: one of "none","clarify","open_what_if","criteria_setting","rhetorical"

Return ONLY the JSON object, no prose."""

def code_turn(text, role):
    if not isinstance(text,str) or not text.strip():
        return None
    msg = [
        {'role':'system','content': STANCE_PROMPT},
        {'role':'user','content': f'SPEAKER: {role}\nTURN:\n"""{text[:3500]}"""'}
    ]
    return cached_chat(msg, schema_tag='stance_v1')

print('\n--- Stance coding (LLM) ---')
stance_rows = []
t0=time.time()
for i, r in logs.iterrows():
    if (i % 250)==0:
        elapsed = time.time()-t0
        print(f'  {i}/{len(logs)}  ({elapsed:.0f}s)')
    c = code_turn(r['message'], r['message_src'])
    if c is None:
        continue
    c2 = dict(c)
    c2['message_id'] = r['message_id']
    c2['conversation_id'] = r['conversation_id']
    c2['message_src'] = r['message_src']
    c2['turn_frac'] = r['turn_frac']
    stance_rows.append(c2)
stance = pd.DataFrame(stance_rows)
for col in ['expansion','contraction','critique','certainty','commit','reframe','proposes_new_idea']:
    if col in stance.columns:
        stance[col] = pd.to_numeric(stance[col], errors='coerce')
stance.to_csv(os.path.join(OUT,'D_stance_llm.csv'), index=False)
print(f'Stance table: {len(stance)} rows')

# ------ aggregate to conversation ------
def conv_agg(g):
    u = g[g.message_src=='user']; a = g[g.message_src=='assistant']
    out = {}
    for side, sub in [('u', u), ('a', a)]:
        for c in ['expansion','contraction','critique','certainty','commit','reframe','proposes_new_idea']:
            out[f'{side}_{c}_mean'] = sub[c].mean() if len(sub) else np.nan
    # critique_tone dist (assistant side — thesis §2.4.3)
    if 'critique_tone' in a.columns:
        tc = a['critique_tone'].value_counts(normalize=True)
        for tone in ['constructive','neutral','blunt']:
            out[f'a_critique_tone_{tone}'] = tc.get(tone,0.0)
    # question_type dist (user side)
    if 'question_type' in u.columns:
        qc = u['question_type'].value_counts(normalize=True)
        for q in ['clarify','open_what_if','criteria_setting','rhetorical']:
            out[f'u_qtype_{q}'] = qc.get(q,0.0)
    return pd.Series(out)
stance_conv = stance.groupby('conversation_id').apply(conv_agg).reset_index()
stance_conv = stance_conv.merge(conv_meta, on='conversation_id')
stance_conv.to_csv(os.path.join(OUT,'D_stance_conv.csv'), index=False)

# ===================================================================
# 2. IDEA EXTRACTION (RQ3)
# ===================================================================
IDEA_PROMPT = """You extract discrete ideas from a creative-problem-solving dialogue.
The task was one of:
  - BICYCLE: how can a city encourage residents to use bicycles instead of driving, beyond existing lanes?
  - LIBRARY: how to revitalize community libraries for young adults?

Return a JSON object with key "ideas" containing a list. Each idea is an object with:
  - "label": 3-8 word name
  - "description": 1-2 sentence summary
  - "origin": "user" if a user turn first introduced it, "assistant" if AI first proposed it, "joint" if it emerged through back-and-forth
  - "developed": true if it was elaborated beyond first mention, false if mentioned once and dropped

Include only DISTINCT ideas (merge paraphrases). Do NOT include meta-discussion, framing moves, or clarifying questions. Aim for 3-12 ideas."""

def extract_ideas(conv_id):
    g = logs[logs.conversation_id==conv_id].sort_values('message_id')
    if len(g)==0: return None
    challenge = conv_meta.loc[conv_meta.conversation_id==conv_id,'challenge'].iloc[0]
    transcript_lines=[]
    for _,r in g.iterrows():
        txt = str(r['message'])[:1800]
        transcript_lines.append(f'[{r["message_src"]}] {txt}')
    transcript = '\n'.join(transcript_lines)[:16000]
    msg = [
        {'role':'system','content': IDEA_PROMPT},
        {'role':'user','content': f'CHALLENGE: {challenge}\n\nTRANSCRIPT:\n{transcript}'}
    ]
    return cached_chat(msg, schema_tag='ideas_v1')

print('\n--- Idea extraction ---')
all_ideas=[]
for i,cid in enumerate(conv_meta['conversation_id']):
    if (i % 25)==0: print(f'  {i}/{len(conv_meta)}')
    d = extract_ideas(cid)
    if not d or 'ideas' not in d: continue
    for j,idea in enumerate(d['ideas']):
        if not isinstance(idea, dict): continue
        idea['conversation_id']=cid
        idea['idx']=j
        all_ideas.append(idea)
ideas = pd.DataFrame(all_ideas)
print(f'Extracted {len(ideas)} ideas across {ideas["conversation_id"].nunique()} conversations')
ideas.to_csv(os.path.join(OUT,'D_ideas.csv'), index=False)

# ===================================================================
# 3. IDEA RUBRIC SCORING (Shah 2003, Acar 2019; §2.1.2)
# ===================================================================
RUBRIC_PROMPT = """Score a single idea on three Likert scales (1=low, 5=high):
- originality: how unusual or non-obvious the idea is, relative to typical responses for this kind of civic challenge
- value: how well it addresses the stated need; fitness under task standards
- feasibility: how implementable it is given realistic constraints
Return a JSON object with integer fields originality, value, feasibility."""

def score_idea(label, desc, challenge):
    msg=[{'role':'system','content':RUBRIC_PROMPT},
         {'role':'user','content':f'CHALLENGE: {challenge}\nIDEA LABEL: {label}\nDESCRIPTION: {desc}'}]
    return cached_chat(msg, schema_tag='rubric_v1')

print('\n--- Rubric scoring ---')
# add challenge
ideas = ideas.merge(conv_meta[['conversation_id','challenge','condition','family','user']], on='conversation_id', how='left')
rubrics=[]
for i,r in ideas.iterrows():
    if (i%100)==0: print(f'  {i}/{len(ideas)}')
    s = score_idea(str(r.get('label','')), str(r.get('description',''))[:800], str(r['challenge']))
    if s:
        s['conversation_id']=r['conversation_id']; s['idx']=r['idx']
        rubrics.append(s)
rub = pd.DataFrame(rubrics)
for c in ['originality','value','feasibility']:
    if c in rub.columns: rub[c] = pd.to_numeric(rub[c], errors='coerce')
ideas = ideas.merge(rub, on=['conversation_id','idx'], how='left')
ideas.to_csv(os.path.join(OUT,'D_ideas_scored.csv'), index=False)

# ===================================================================
# 4. EMBEDDINGS + PORTFOLIO METRICS (RQ3)
# ===================================================================
print('\n--- Idea embeddings ---')
idea_texts = (ideas['label'].fillna('')+': '+ideas['description'].fillna('')).tolist()
E = cached_embed(idea_texts) if len(idea_texts) else np.zeros((0,1536))
np.save(os.path.join(OUT,'D_idea_embeddings.npy'), E)

from sklearn.metrics.pairwise import cosine_similarity
ideas['row']=np.arange(len(ideas))

# portfolio metrics per conversation
conv_metrics = []
for cid,g in ideas.groupby('conversation_id'):
    idx = g['row'].values
    if len(idx)<2: continue
    sub = E[idx]
    sim = cosine_similarity(sub); np.fill_diagonal(sim, np.nan)
    breadth = 1 - np.nanmean(sim)                      # within-conv breadth
    redund = np.nanmax(sim)                            # max pair similarity = redundancy
    user_share = (g['origin']=='user').mean()
    ai_share   = (g['origin']=='assistant').mean()
    conv_metrics.append(dict(conversation_id=cid,
        n_ideas=len(g), portfolio_breadth=breadth, portfolio_redundancy=redund,
        user_idea_share=user_share, ai_idea_share=ai_share,
        mean_originality=g['originality'].mean(),
        mean_value=g['value'].mean(),
        mean_feasibility=g['feasibility'].mean(),
        max_originality=g['originality'].max()))
cm = pd.DataFrame(conv_metrics).merge(conv_meta, on='conversation_id')

# between-user distinctiveness per challenge using *conversation centroid*
for ch in cm['challenge'].unique():
    rows = cm[cm.challenge==ch]['conversation_id'].tolist()
    centroids = []
    for cid in rows:
        idx = ideas[ideas.conversation_id==cid]['row'].values
        if len(idx)==0: centroids.append(np.zeros(E.shape[1])); continue
        centroids.append(E[idx].mean(0))
    C = np.vstack(centroids)
    s = cosine_similarity(C); np.fill_diagonal(s,np.nan)
    dist = 1-np.nanmean(s, axis=1)
    for cid,d in zip(rows,dist):
        cm.loc[cm.conversation_id==cid, 'between_user_distinctiveness'] = d

cm.to_csv(os.path.join(OUT,'D_portfolio.csv'), index=False)

# ===================================================================
# 5. FIXATION INDEX (§2.4.2)
# ===================================================================
print('\n--- Fixation index ---')
# first AI-proposed direction = first assistant turn containing a propose-new-idea OR fall back to first assistant turn
fix_rows=[]
for cid,g in logs.groupby('conversation_id'):
    g = g.sort_values('message_id')
    a = g[g.message_src=='assistant']
    if len(a)==0: continue
    # get stance for assistant turns
    s_a = stance.merge(a[['message_id']], on='message_id')
    s_a = s_a[s_a.proposes_new_idea>=1].sort_values('message_id') if 'proposes_new_idea' in s_a.columns else a.iloc[:0]
    anchor = (s_a.iloc[0] if len(s_a) else a.iloc[0])
    anchor_text = str(logs.loc[logs.message_id==anchor['message_id'],'message'].iloc[0])[:2000]
    # user turns
    u = g[g.message_src=='user'].sort_values('message_id')
    if len(u)<3: continue
    texts = [anchor_text] + [str(x)[:2000] for x in u['message']]
    Ef = cached_embed(texts)
    anchor_vec = Ef[0:1]
    user_vecs = Ef[1:]
    sims = cosine_similarity(anchor_vec, user_vecs).ravel()
    # drift pattern
    early = sims[:max(1,len(sims)//3)].mean()
    late  = sims[-max(1,len(sims)//3):].mean()
    fix_rows.append(dict(conversation_id=cid,
        anchor_sim_early=early, anchor_sim_late=late,
        fixation_index=sims.mean(),         # higher = more fixation
        drift_trajectory=early-late))       # positive = user drifted away
fx = pd.DataFrame(fix_rows).merge(conv_meta, on='conversation_id')
fx.to_csv(os.path.join(OUT,'D_fixation.csv'), index=False)

# ===================================================================
# 6. STABILITY SUBSAMPLE (§2.10)
# ===================================================================
print('\n--- Stability check (10%) ---')
# re-code 10% of turns with a paraphrased prompt, report correlation
sub = stance.sample(n=min(300, len(stance)), random_state=0)
STANCE_PROMPT_B = STANCE_PROMPT.replace('0=absent, 3=strong','on a 0-3 scale where 0 = no such cue, 1 = faint, 2 = clear, 3 = dominant')
rows=[]
for i,r in sub.iterrows():
    m = logs.loc[logs.message_id==r['message_id']]
    if len(m)==0: continue
    text = str(m['message'].iloc[0])[:3500]; role=m['message_src'].iloc[0]
    msg = [{'role':'system','content':STANCE_PROMPT_B},
           {'role':'user','content':f'SPEAKER: {role}\nTURN:\n"""{text}"""'}]
    r2 = cached_chat(msg, schema_tag='stance_v1b')
    if r2 is None: continue
    rows.append(dict(message_id=r['message_id'],
        exp_a=r['expansion'], exp_b=r2.get('expansion'),
        con_a=r['contraction'], con_b=r2.get('contraction'),
        cri_a=r['critique'], cri_b=r2.get('critique'),
        cer_a=r['certainty'], cer_b=r2.get('certainty')))
stab = pd.DataFrame(rows)
if len(stab)>10:
    for a,b,n in [('exp_a','exp_b','expansion'),('con_a','con_b','contraction'),
                  ('cri_a','cri_b','critique'),('cer_a','cer_b','certainty')]:
        x = pd.to_numeric(stab[a], errors='coerce'); y = pd.to_numeric(stab[b], errors='coerce')
        m = (~x.isna())&(~y.isna())
        if m.sum()<5: continue
        r,_ = stats.spearmanr(x[m], y[m])
        print(f'  stability {n}: ρ={r:.3f} (n={m.sum()})')
    stab.to_csv(os.path.join(OUT,'D_llm_stability.csv'), index=False)

# ===================================================================
# 7. MANIPULATION CHECK (§2.8)
# ===================================================================
print('\n=== MANIPULATION CHECK: persona fidelity ===')
stance_conv = pd.read_csv(os.path.join(OUT,'D_stance_conv.csv'))
key_cols = ['a_expansion_mean','a_contraction_mean','a_critique_mean','a_certainty_mean',
            'a_commit_mean','a_reframe_mean','a_proposes_new_idea_mean',
            'u_expansion_mean','u_contraction_mean']
mc = stance_conv[stance_conv.condition=='Persona'].groupby('family')[key_cols].mean()
mc_gpt = stance_conv[stance_conv.condition=='GPT'][key_cols].mean()
mc.loc['GPT_baseline'] = mc_gpt
mc.to_csv(os.path.join(OUT,'D_manipulation_check.csv'))
print(mc.round(3))

# ===================================================================
# 8. PAIRED TESTS WITH LLM-CODED METRICS
# ===================================================================
print('\n=== PAIRED TESTS: LLM-coded Persona vs GPT ===')
def paired(a,b,nm):
    m = (~pd.isna(a))&(~pd.isna(b)); a,b=a[m],b[m]
    if len(a)<5: return None
    d=a-b; t,p=stats.ttest_rel(a,b)
    dz = d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
    return dict(metric=nm,n=len(a),mean_P=a.mean(),mean_G=b.mean(),diff=d.mean(),t=t,p=p,dz=dz)

# build wide for stance + portfolio + fixation
for df_, label in [(stance_conv, 'stance'), (cm, 'portfolio'), (fx, 'fixation')]:
    pass

merged = stance_conv.merge(cm[['conversation_id','portfolio_breadth','portfolio_redundancy',
                               'user_idea_share','ai_idea_share','mean_originality','mean_value',
                               'mean_feasibility','max_originality','between_user_distinctiveness',
                               'n_ideas']], on='conversation_id', how='left')
merged = merged.merge(fx[['conversation_id','fixation_index','drift_trajectory']],
                      on='conversation_id', how='left')
test_cols = [c for c in merged.columns if c.endswith('_mean') or c in
             ('portfolio_breadth','portfolio_redundancy','user_idea_share','ai_idea_share',
              'mean_originality','mean_value','mean_feasibility','max_originality',
              'between_user_distinctiveness','n_ideas','fixation_index','drift_trajectory')]
wide = merged.pivot_table(index='user', columns='condition', values=test_cols, aggfunc='first')
wide.columns = [f'{a}__{b}' for a,b in wide.columns]

rows=[]
for c in test_cols:
    cg=f'{c}__GPT'; cp=f'{c}__Persona'
    if cg in wide.columns and cp in wide.columns:
        r = paired(wide[cp].astype(float), wide[cg].astype(float), c)
        if r: rows.append(r)
pair_df = pd.DataFrame(rows).sort_values('p')
pair_df.to_csv(os.path.join(OUT,'D_paired_llm.csv'), index=False)
print(pair_df.to_string(index=False))

# ===================================================================
# 9. FIGURES
# ===================================================================
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':160,'font.size':10})

# 8. manipulation check heatmap
import seaborn as sns
fig,ax = plt.subplots(figsize=(8,4.5))
sns.heatmap(mc.loc[['Divergent','Convergent','Rational','BoundedRational','GPT_baseline']],
            annot=True, cmap='vlag', center=mc.loc['GPT_baseline'].mean(), ax=ax)
ax.set_title('Fig 8. Persona fidelity manipulation check — mean LLM-coded stance\n(rows=family, cols=metric; GPT baseline at bottom)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig8_manipulation_check.png')); plt.close()

# 9. fixation by condition
fig,ax = plt.subplots(figsize=(5,4))
for c,col,lbl in [('GPT','tab:blue','GPT'),('Persona','tab:red','Persona')]:
    ax.hist(fx[fx.condition==c]['fixation_index'].dropna(), bins=20, alpha=0.5, label=lbl, color=col)
ax.set_xlabel('Fixation index (mean cosine sim to first AI anchor)')
ax.set_ylabel('conversations')
ax.set_title('Fig 9. Fixation to initial AI anchor')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig9_fixation.png')); plt.close()

# 10. rubric scores paired
fig, axes = plt.subplots(1,3, figsize=(11,4))
for ax, c in zip(axes, ['mean_originality','mean_value','mean_feasibility']):
    cg=f'{c}__GPT'; cp=f'{c}__Persona'
    if cg not in wide.columns: continue
    uu = wide[[cg,cp]].dropna()
    for _,r in uu.iterrows():
        ax.plot([0,1],[r[cg],r[cp]], color='gray', alpha=0.3)
    ax.boxplot([uu[cg], uu[cp]], positions=[0,1], widths=0.35, showfliers=False)
    ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
    ax.set_title(c.replace('mean_',''))
plt.suptitle('Fig 10. Per-idea rubric scores (LLM-judged, bounded proxy; §2.10)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig10_rubric.png')); plt.close()

# 11. UMAP of ideas
try:
    import umap
    red = umap.UMAP(n_components=2, random_state=0, metric='cosine').fit_transform(E)
except Exception:
    from sklearn.decomposition import PCA
    red = PCA(n_components=2).fit_transform(E)
ideas['u0']=red[:,0]; ideas['u1']=red[:,1]
fig, axes = plt.subplots(1,2, figsize=(12,5))
for ax, ch in zip(axes, ['Bicycle','Library']):
    sub = ideas[ideas.challenge==ch]
    for cond, col in [('GPT','tab:blue'),('Persona','tab:red')]:
        s2 = sub[sub.condition==cond]
        ax.scatter(s2['u0'], s2['u1'], s=10, alpha=0.5, label=cond, color=col)
    ax.set_title(ch); ax.legend()
plt.suptitle('Fig 11. Idea-space map (OpenAI embeddings; UMAP)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig11_idea_space.png')); plt.close()

# 12. drift trajectory by condition
fig,ax = plt.subplots(figsize=(5,4))
for c,col in [('GPT','tab:blue'),('Persona','tab:red')]:
    ax.hist(fx[fx.condition==c]['drift_trajectory'].dropna(), bins=20, alpha=0.5, label=c, color=col)
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('Drift (early anchor sim  -  late anchor sim;  >0 = moved away)')
ax.set_title('Fig 12. User drift away from initial AI anchor')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig12_drift.png')); plt.close()

print('\nDONE — extension artifacts in', OUT)
