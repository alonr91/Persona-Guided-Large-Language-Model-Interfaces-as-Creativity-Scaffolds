"""Full translation pass: every Hebrew message → English.

For each message, store:
  - 'message'         : original (unchanged)
  - 'message_en'      : English translation (or original if no Hebrew)
  - 'original_lang'   : 'he' (any Hebrew chars) or 'en' (none)

Strategy:
  - Detect Hebrew via Unicode range
  - For long messages, chunk by sentence boundary up to ~900 chars
  - Batch chunks in groups of 8 for CPU efficiency
"""
import json, re, time, sys
from transformers import MarianMTModel, MarianTokenizer
import torch

HE_RE = re.compile(r'[֐-׿]')

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
with open(f'{base}/data/experiment3_messages_clean.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

print(f'Loading model...')
name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name)
model.eval()
print('Model loaded.')

def chunkize(text, max_chars=900):
    if len(text) <= max_chars: return [text]
    parts = re.split(r'(?<=[.!?。\n])\s+', text)
    chunks, cur = [], ''
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + ' ' + p).strip() if cur else p
        else:
            if cur: chunks.append(cur)
            if len(p) > max_chars:
                # hard wrap on space
                while len(p) > max_chars:
                    cut = p.rfind(' ', 0, max_chars)
                    if cut < 0: cut = max_chars
                    chunks.append(p[:cut].strip())
                    p = p[cut:].strip()
            cur = p
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]

def translate_batch(texts):
    if not texts: return []
    with torch.no_grad():
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
    return [tok.decode(g, skip_special_tokens=True) for g in gen]

# Build flat queue of (msg_idx, chunk_idx_within_msg, chunk_text)
queue = []
plan = []  # (msg_idx, num_chunks)
for i, m in enumerate(msgs):
    if HE_RE.search(m['message']):
        chs = chunkize(m['message'])
        plan.append((i, len(chs)))
        for ci, c in enumerate(chs):
            queue.append((i, ci, c))

print(f'Hebrew msgs: {len(plan)} | total chunks to translate: {len(queue)}')

# Run in batches
BATCH = 8
results = {}  # (msg_idx, chunk_idx) -> translated
t0 = time.time()
for start in range(0, len(queue), BATCH):
    batch = queue[start:start+BATCH]
    texts = [b[2] for b in batch]
    try:
        out = translate_batch(texts)
    except Exception as e:
        print(f'Error at batch {start}: {e}; falling back to single')
        out = []
        for t in texts:
            try: out.append(translate_batch([t])[0])
            except Exception as e2:
                print(f'  hard fail: {e2}'); out.append('[TRANSLATION ERROR]')
    for (msg_idx, chunk_idx, _), tr in zip(batch, out):
        results[(msg_idx, chunk_idx)] = tr
    if start % (BATCH*10) == 0:
        elapsed = time.time()-t0
        eta = elapsed / max(start+BATCH,1) * (len(queue) - start - BATCH)
        print(f'  {start+len(batch)}/{len(queue)} chunks  elapsed={elapsed:.1f}s  eta={eta:.1f}s', flush=True)

print(f'Translation done in {time.time()-t0:.1f}s')

# Reassemble per message
for i, m in enumerate(msgs):
    if HE_RE.search(m['message']):
        # Get all chunks in order
        n_chunks = sum(1 for k in results if k[0]==i)
        parts = [results[(i, ci)] for ci in range(n_chunks) if (i, ci) in results]
        m['message_en'] = ' '.join(parts)
        m['original_lang'] = 'he'
    else:
        m['message_en'] = m['message']
        m['original_lang'] = 'en'

# Save
out_path = f'{base}/data/experiment3_messages_en.json'
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(msgs, f, ensure_ascii=False, indent=2)
print(f'Wrote {out_path}')

# Also produce English-only transcripts for human reading
from collections import defaultdict
by_conv = defaultdict(list)
for m in msgs:
    by_conv[m['conversation_id']].append(m)
for cid in by_conv:
    by_conv[cid].sort(key=lambda x: (x['timestamp'], x['message_id']))

PERSONA_NAMES = {'1':'Taylor (divergent)','2':'Alex (convergent)','3':'Taylor (control-LLM)','4':'Alex (control-LLM)'}
def write_md(path, cids, title):
    with open(path,'w',encoding='utf-8') as f:
        f.write(f'# {title}\n\nConversations: {len(cids)}\n\n---\n\n')
        for cid in sorted(cids):
            ms = by_conv[cid]
            grp = ms[0]['group']
            f.write(f'## Conversation {cid}  [{grp}]\n')
            f.write(f'_{ms[0]["timestamp"]} → {ms[-1]["timestamp"]}  |  {len(ms)} messages_\n\n')
            for m in ms:
                pid = str(m.get('persona_id'))
                who = 'USER' if m['message_src']=='user' else PERSONA_NAMES.get(pid, f'persona{pid}')
                lang_marker = ' [translated from HE]' if m['original_lang']=='he' else ''
                f.write(f'**[{m["timestamp"]}] {who}{lang_marker}:** {m["message_en"]}\n\n')
            f.write('\n---\n\n')

t_cids = [c for c in by_conv if by_conv[c][0]['group']=='treatment']
c_cids = [c for c in by_conv if by_conv[c][0]['group']=='control']
write_md(f'{base}/transcripts/transcripts_treatment_EN.md', t_cids, 'Experiment 3 — Treatment (ENGLISH): Taylor=divergent, Alex=convergent')
write_md(f'{base}/transcripts/transcripts_control_EN.md',   c_cids, 'Experiment 3 — Control (ENGLISH): regular LLM, no personality')
print(f'Wrote English transcripts.')
