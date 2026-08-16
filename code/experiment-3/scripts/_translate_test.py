"""Sanity check: translate a handful of representative Hebrew messages.

We pick short, medium, long, and one mixed (Hebrew + English) sample.
"""
import json, re
from transformers import MarianMTModel, MarianTokenizer

HE_RE = re.compile(r'[֐-׿]')

with open('C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3/data/experiment3_messages_clean.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

he_msgs = [m for m in msgs if HE_RE.search(m['message'])]
he_msgs.sort(key=lambda m: len(m['message']))

# Pick representative samples
samples = []
samples.append(he_msgs[0])               # shortest
samples.append(he_msgs[len(he_msgs)//4]) # short-medium
samples.append(he_msgs[len(he_msgs)//2]) # median
samples.append(he_msgs[3*len(he_msgs)//4])# long-medium
samples.append(he_msgs[-3])              # long
samples.append(he_msgs[-1])              # longest

print(f'Loading Helsinki-NLP/opus-mt-tc-big-he-en ...')
name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name)
model.eval()
print('Model loaded.\n')

import torch

def translate(text, max_chunk_chars=900):
    """Translate, chunking by sentences if too long."""
    # MarianMT models often have ~512 token limits. ~900 chars ≈ <512 toks Hebrew.
    if len(text) <= max_chunk_chars:
        chunks = [text]
    else:
        # Split by Hebrew/Eng sentence enders
        parts = re.split(r'(?<=[.!?。\n])\s+', text)
        chunks, cur = [], ''
        for p in parts:
            if len(cur) + len(p) + 1 <= max_chunk_chars:
                cur = (cur + ' ' + p).strip() if cur else p
            else:
                if cur: chunks.append(cur)
                cur = p
        if cur: chunks.append(cur)
    out = []
    for c in chunks:
        if not c.strip(): continue
        with torch.no_grad():
            enc = tok([c], return_tensors='pt', truncation=True, max_length=512)
            gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
            out.append(tok.decode(gen[0], skip_special_tokens=True))
    return ' '.join(out)

for i, m in enumerate(samples, 1):
    print(f'--- sample {i} | conv={m["conversation_id"]} | src={m["message_src"]} | persona={m["persona_id"]} | len={len(m["message"])} ---')
    print('HE:', m['message'][:280] + ('...' if len(m['message'])>280 else ''))
    en = translate(m['message'])
    print('EN:', en[:400] + ('...' if len(en)>400 else ''))
    print()
