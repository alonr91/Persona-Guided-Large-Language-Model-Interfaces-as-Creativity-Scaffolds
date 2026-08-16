"""V5 — Mask-leak audit.

Sample 50 random conversations, present the masked transcript to one
judge (Dr_C variant A), and ask: which persona family was active?
If the judge can guess above chance (ROC AUC > 0.65), the mask is
leaking and we need to harden it before primary inference.

This is a separate, single-task prompt — independent of the dimension-
scoring prompts.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.gemini_client import GeminiClient
from os_pipeline.cat_panel.scorer import _load_conversations

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'


class MaskGuess(BaseModel):
    guess: Literal['GPT', 'Divergent', 'Convergent', 'Rational',
                   'BoundedRational', 'unknown']
    confidence_0_1: float = Field(description='confidence 0..1')
    rationale_short: str


SYS_PROMPT = (
    "You are an experienced HCI researcher. The text below is a masked "
    "transcript from a creativity-collaboration study. The AI assistant "
    "was one of five possible personas: GPT (no persona prompt), "
    "Divergent (broadening), Convergent (narrowing), Rational "
    "(strictly evaluative), or BoundedRational (satisficing with "
    "heuristics). Identifying labels have been replaced with neutral "
    "tags. Based ONLY on the conversational behaviour of the assistant, "
    "which persona condition was most likely active? If you cannot "
    "tell, return 'unknown'. Output ONLY valid JSON."
)


def main(n: int = 50, seed: int = 7):
    convs = _load_conversations()
    rng = np.random.default_rng(seed)
    sample = convs.sample(n=min(n, len(convs)), random_state=rng).reset_index(drop=True)
    print(f'[mask audit] sampling {len(sample)} conversations')

    rows = []
    for i, r in sample.iterrows():
        usr = (f'conversation_id (masked): {i}\n'
               f'challenge: {r["challenge"]}\n'
               '\n--- MASKED TRANSCRIPT ---\n'
               f'{r["transcript_masked"]}\n'
               '\n--- END ---\n\n'
               'Which persona condition was active? Return JSON only.')
        parsed, _ = GeminiClient.generate_json(
            schema_cls=MaskGuess,
            system_prompt=SYS_PROMPT,
            user_prompt=usr,
            temperature=0.0,
        )
        if parsed is None:
            continue
        rows.append(dict(
            conversation_id=int(r['conversation_id']),
            true_family=r['persona_family_original'],
            guessed_family=parsed.guess,
            correct=(parsed.guess == r['persona_family_original']),
            confidence_0_1=float(parsed.confidence_0_1),
            rationale=parsed.rationale_short,
        ))
        if (i+1) % 10 == 0:
            print(f'[mask audit] {i+1}/{len(sample)}  '
                  f'rolling acc={np.mean([x["correct"] for x in rows]):.2%}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'audit_mask_leak.csv', index=False)

    acc = float(df['correct'].mean())
    chance = 1/5  # 5 families
    print(f'\n[mask audit] n={len(df)}, accuracy={acc:.2%} '
          f'(chance={chance:.2%})')
    print(f'  mask is {"CLEAN" if acc <= 0.30 else "POSSIBLY LEAKING"}')
    return df


if __name__ == '__main__':
    main()
