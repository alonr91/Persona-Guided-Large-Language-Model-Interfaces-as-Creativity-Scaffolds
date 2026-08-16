# -*- coding: utf-8 -*-
"""Re-translate Hebrew -> English with Google Translate (deep_translator).

Scope: the full-tier (high-quality, >=4 substantive user turns) conversations.
Overwrites prior (local-model) translations with Google Translate output and
tags each message with translation_engine. English messages get message_en = message.
Updates data/experiment3_messages.json in place and writes data/experiment3_full_en.json.
"""
import json, csv, time, sys
from collections import Counter
from deep_translator import GoogleTranslator

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_messages.json', encoding='utf-8'))
rows = list(csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8')))
full = set(int(r['conversation_id']) for r in rows if r['quality_tier'] == 'full')

tr = GoogleTranslator(source='iw', target='en')

def translate(text):
    t = (text or '').strip()
    if not t:
        return text
    # deep_translator caps at 5000 chars; chat turns are short, but guard anyway
    if len(t) > 4800:
        out = []
        for i in range(0, len(t), 4800):
            out.append(tr.translate(t[i:i+4800]))
            time.sleep(0.2)
        return ' '.join(x for x in out if x)
    return tr.translate(t)

he = [m for m in msgs if m['conversation_id'] in full and m['original_lang'] == 'he']
print(f'Translating {len(he)} Hebrew messages in {len(full)} full-tier conversations...')

done = 0
t0 = time.time()
for m in msgs:
    if m['conversation_id'] not in full:
        continue
    if m['original_lang'] == 'en':
        m['message_en'] = m['message']
        m['translation_engine'] = 'none(en)'
    else:
        for attempt in range(4):
            try:
                m['message_en'] = translate(m['message'])
                m['translation_engine'] = 'google'
                break
            except Exception as e:
                if attempt == 3:
                    print('  FAIL id', m['message_id'], repr(e)[:80]); m['translation_engine'] = 'failed'
                time.sleep(1.5 * (attempt + 1))
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f'  {done}/{len(he)}  elapsed={el:.0f}s  eta={el/done*(len(he)-done):.0f}s')
        time.sleep(0.15)  # gentle pacing

json.dump(msgs, open(f'{BASE}/data/experiment3_messages.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
full_msgs = [m for m in msgs if m['conversation_id'] in full]
json.dump(full_msgs, open(f'{BASE}/data/experiment3_full_en.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

eng = Counter(m.get('translation_engine') for m in full_msgs)
print('translation_engine in full set:', dict(eng))
print('total time %.1f min' % ((time.time() - t0) / 60))
print('Wrote: data/experiment3_messages.json (updated) and data/experiment3_full_en.json')
