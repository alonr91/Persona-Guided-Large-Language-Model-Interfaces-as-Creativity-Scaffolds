"""
Step 2: Per-message surprise (negative log-likelihood) with a causal LM.
Mirrors the method in yes_and_novelty_surprise.py. CPU-only compatible.
Output: analysis_out/msg_surprise.npy, shape (len(logs),) aligned to the
same sort order as msg_embeddings.npy. NaN for first-in-conversation and
for empty messages.
"""
import os, sys, time, numpy as np, pandas as pd, torch
sys.stdout.reconfigure(encoding='utf-8')

ROOT  = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT   = os.path.join(ROOT, 'analysis_out')
os.makedirs(OUT, exist_ok=True)

MODEL_NAME = 'gpt2'
MAX_CTX    = 1024
CTX_BUDGET = 900

csv_cleaned = os.path.join(ROOT, 'Experiment1_logs_cleaned_keepable_paired_translated.csv')
csv_raw     = os.path.join(ROOT, 'Experiment1_logs.csv')
csv_path    = csv_cleaned if os.path.exists(csv_cleaned) else csv_raw
print('logs:', csv_path)
logs = pd.read_csv(csv_path)
logs = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
print('loading', MODEL_NAME)
tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device('cpu')
model.to(device)

# pre-encode every message once
texts = logs['message'].fillna('').astype(str).tolist()
print('tokenizing', len(texts), 'messages')
msg_ids = [tok.encode(t, add_special_tokens=False) for t in texts]
msg_len = np.array([len(x) for x in msg_ids])
print('median msg tokens:', int(np.median(msg_len)),
      'p95:', int(np.percentile(msg_len,95)),
      'max:', int(msg_len.max()))

surprise_total = np.full(len(logs), np.nan, dtype='float32')
surprise_per_tok = np.full(len(logs), np.nan, dtype='float32')

conv_ids = logs['conversation_id'].values
t0 = time.time()
n_done = 0
for i in range(len(logs)):
    if i == 0 or conv_ids[i] != conv_ids[i-1]:
        continue
    tgt = msg_ids[i]
    if len(tgt) == 0:
        continue
    # build context from prior messages in same conversation; truncate tail-ward
    ctx = []
    j = i - 1
    while j >= 0 and conv_ids[j] == conv_ids[i]:
        ctx = msg_ids[j] + ctx
        if len(ctx) >= CTX_BUDGET:
            ctx = ctx[-CTX_BUDGET:]
            break
        j -= 1
    if len(ctx) == 0:
        continue
    # cap total length to MAX_CTX; if target is too long, truncate target
    tgt_room = MAX_CTX - len(ctx)
    if tgt_room < 1:
        ctx = ctx[-(MAX_CTX-1):]
        tgt_room = 1
    tgt_eff = tgt[:tgt_room] if len(tgt) > tgt_room else tgt

    ids = torch.tensor([ctx + tgt_eff], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits  # [1, L, V]
    # predictions for position k are logits[..., k-1, :]
    ctx_len = len(ctx)
    tgt_len = len(tgt_eff)
    # target tokens live at positions ctx_len .. ctx_len+tgt_len-1 in ids
    # predicting positions ctx_len-1 .. ctx_len+tgt_len-2 in logits
    pred_logits = logits[0, ctx_len-1 : ctx_len-1 + tgt_len, :]
    tgt_tensor  = torch.tensor(tgt_eff, dtype=torch.long, device=device)
    log_probs   = torch.log_softmax(pred_logits, dim=-1)
    tok_lp      = log_probs.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1)
    total_nll   = -tok_lp.sum().item()
    surprise_total[i]   = total_nll
    surprise_per_tok[i] = total_nll / tgt_len

    n_done += 1
    if n_done % 100 == 0:
        dt = time.time() - t0
        rate = n_done / dt
        eta_s = (len(logs) - i) / max(rate, 1e-9)
        print(f'  [{n_done}] i={i}/{len(logs)}  rate={rate:.2f} msg/s  eta~{eta_s/60:.1f} min',
              flush=True)

np.save(os.path.join(OUT, 'msg_surprise_total.npy'), surprise_total)
np.save(os.path.join(OUT, 'msg_surprise_per_tok.npy'), surprise_per_tok)
valid = ~np.isnan(surprise_total)
print(f'\ndone in {(time.time()-t0)/60:.1f} min')
print(f'saved msg_surprise_total.npy / msg_surprise_per_tok.npy  ({valid.sum()} valid of {len(logs)})')
print(f'per-token surprise mean={np.nanmean(surprise_per_tok):.3f}  '
      f'p50={np.nanpercentile(surprise_per_tok,50):.3f}  '
      f'p95={np.nanpercentile(surprise_per_tok,95):.3f}')
