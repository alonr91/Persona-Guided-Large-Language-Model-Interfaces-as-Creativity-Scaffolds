"""Smoke test for the agentic idea-extraction pipeline.

Runs Agents 1-3 on 5 conversations (one per condition: GPT + 4 persona
families), saves candidates, canonical ideas, validation report, and
per-conversation side-by-side transcripts for human QA.
"""
from __future__ import annotations
import json, time, sys, os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

# make sure we can import the local package when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from os_pipeline import config
from os_pipeline.agent1_extractor import extract_for_message, CandidateRow
from os_pipeline.agent2_consolidator import consolidate
from os_pipeline.agent3_validator import validate
from os_pipeline.llm_client import LLMClient

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def load_logs() -> pd.DataFrame:
    logs = pd.read_csv(config.LOGS_CSV)
    logs = logs.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)
    return logs


def pick_conversations(logs: pd.DataFrame) -> list[int]:
    # user turn count per conv
    utc = (logs[logs.message_src == 'user']
           .groupby('conversation_id').size()
           .rename('n_user').reset_index())
    # persona per conv
    persona = (logs.groupby('conversation_id')['Persona_type']
               .first().reset_index())
    cond = pd.merge(persona, utc, on='conversation_id')
    # map to persona_label
    fm = {'GPT': 'GPT', 'Divergent': 'Divergent', 'Convergent': 'Convergent',
          'strictly rational': 'Rational', 'bounded rationality': 'BoundedRational'}
    cond['label'] = cond['Persona_type'].map(fm)
    # bound length to median..p75 user-turn count
    med, p75 = cond['n_user'].median(), cond['n_user'].quantile(0.75)
    print(f'[select] median user-turns={med:.0f}, p75={p75:.0f}')
    selected: list[int] = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for lbl in ['GPT', 'Divergent', 'Convergent', 'Rational', 'BoundedRational']:
        pool = cond[(cond.label == lbl)
                    & (cond.n_user >= med) & (cond.n_user <= p75)]
        if len(pool) == 0:
            # fallback: any conv with at least 4 user turns
            pool = cond[(cond.label == lbl) & (cond.n_user >= 4)]
        if len(pool) == 0:
            print(f'[select] WARNING no {lbl} conversation available')
            continue
        pick = pool.sample(n=1, random_state=rng.integers(1 << 30)).iloc[0]
        selected.append(int(pick['conversation_id']))
        print(f'[select] {lbl:16s} -> conv {int(pick.conversation_id)} '
              f'(n_user={int(pick.n_user)})')
    return selected


def run_conversation(logs: pd.DataFrame, cid: int) -> dict:
    g = logs[logs.conversation_id == cid].sort_values('message_id').reset_index(drop=True)
    persona_type = g['Persona_type'].iloc[0]
    challenge = g['Corrected Challenge type'].iloc[0] if 'Corrected Challenge type' in g.columns else 'unknown'
    fm = {'GPT': 'GPT', 'Divergent': 'Divergent', 'Convergent': 'Convergent',
          'strictly rational': 'Rational', 'bounded rationality': 'BoundedRational'}
    persona_label = fm.get(persona_type, str(persona_type))
    print(f'\n=== conversation {cid}  ({persona_label}, {challenge}) '
          f'— {len(g)} total msgs ===')

    user_idx = g.index[g.message_src == 'user'].tolist()
    candidates: list[CandidateRow] = []
    agent1_debug: list[dict] = []
    t0 = time.time()
    for utix, ri in enumerate(user_idx):
        row = g.iloc[ri]
        prev_asst = None
        if ri > 0 and g.iloc[ri-1]['message_src'] == 'assistant':
            prev_asst = str(g.iloc[ri-1]['message'])
        user_msg = str(row['message'])
        if len(user_msg.strip()) < 2:
            continue
        t_start = time.time()
        rows, dbg = extract_for_message(
            conversation_id=int(cid),
            message_id=int(row['message_id']),
            user_turn_idx=utix,
            prev_assistant=prev_asst,
            user_msg=user_msg,
            challenge=str(challenge),
            persona_label=persona_label,
        )
        dbg['latency_s'] = round(time.time() - t_start, 2)
        agent1_debug.append(dbg)
        candidates.extend(rows)
        print(f'  u-turn {utix:2d}  msg {int(row.message_id):5d}  '
              f'{len(rows)} ideas  {dbg["latency_s"]}s  '
              f'valid={dbg.get("valid_json", False)}')
    a1_elapsed = time.time() - t0
    print(f'  -> Agent 1 total: {len(candidates)} candidates in {a1_elapsed:.1f}s')

    # Agent 2
    t1 = time.time()
    canonical = consolidate(candidates, conversation_id=int(cid))
    a2_elapsed = time.time() - t1
    print(f'  -> Agent 2 consolidation: {len(canonical)} canonical from {len(candidates)} candidates in {a2_elapsed:.1f}s')

    # Agent 3
    t2 = time.time()
    user_messages = [str(m) for m in g[g.message_src == 'user']['message'].tolist()]
    kept, validation_report = validate(canonical, user_messages)
    a3_elapsed = time.time() - t2
    n_grounded = sum(1 for v in validation_report if v.status == 'grounded')
    n_fuzzy = sum(1 for v in validation_report if v.status == 'grounded_fuzzy')
    n_ungrounded = sum(1 for v in validation_report if v.status == 'ungrounded')
    ground_rate = (n_grounded + n_fuzzy) / max(1, len(validation_report))
    print(f'  -> Agent 3 validation: grounded={n_grounded} fuzzy={n_fuzzy} '
          f'ungrounded={n_ungrounded}  pass rate={ground_rate:.2%}')

    # human-readable transcript
    md_path = config.TRANSCRIPTS_DIR / f'conv_{cid}.md'
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write(f'# Conversation {cid} — {persona_label} — {challenge}\n\n')
        fh.write(f'Agent 1: {len(candidates)} candidates (latency {a1_elapsed:.1f}s)\n')
        fh.write(f'Agent 2: {len(canonical)} canonical (latency {a2_elapsed:.1f}s)\n')
        fh.write(f'Agent 3: {n_grounded} grounded + {n_fuzzy} fuzzy + {n_ungrounded} ungrounded (kept {len(kept)})\n\n')
        fh.write('## Canonical ideas (kept after Agent 3)\n\n')
        for c in kept:
            fh.write(f'- **{c.title}**\n')
            fh.write(f'  - {c.description}\n')
            fh.write(f'  - evidence: {c.evidence_quotes}\n\n')
        fh.write('\n## Full transcript\n\n')
        for _, r in g.iterrows():
            fh.write(f'**[{r.message_src}]** (msg {r.message_id}): {str(r.message)[:500]}\n\n')
        fh.write('\n## Per-candidate Agent 1 detail\n\n')
        for c in candidates:
            fh.write(f'- msg {c.message_id}  •  **{c.title}**  —  {c.description}\n')
            fh.write(f'  - evidence: "{c.evidence_span}"  (conf={c.confidence:.2f})\n')

    return {
        'conversation_id': cid,
        'persona_label': persona_label,
        'challenge': challenge,
        'n_messages': len(g),
        'n_user_turns': len(user_idx),
        'n_candidates': len(candidates),
        'n_canonical': len(canonical),
        'n_grounded': n_grounded,
        'n_grounded_fuzzy': n_fuzzy,
        'n_ungrounded': n_ungrounded,
        'ground_rate': ground_rate,
        'agent1_latency_s': a1_elapsed,
        'agent2_latency_s': a2_elapsed,
        'agent3_latency_s': a3_elapsed,
        'agent1_valid_rate': (sum(1 for d in agent1_debug if d.get('valid_json')) /
                              max(1, len(agent1_debug))),
        'candidates': [asdict(c) for c in candidates],
        'canonical': [asdict(c) for c in kept],
        'validation': [asdict(v) for v in validation_report],
    }


def main():
    t0 = time.time()
    logs = load_logs()
    convs = pick_conversations(logs)
    if len(convs) < 3:
        print('[smoke] too few conversations selected, aborting')
        sys.exit(1)

    # warm up the model up-front so load time isn't charged to the first conv
    LLMClient.load()

    all_results = []
    for cid in convs:
        result = run_conversation(logs, cid)
        all_results.append(result)

    # persist outputs
    with open(config.OUT_DIR / 'candidates.jsonl', 'w', encoding='utf-8') as fh:
        for r in all_results:
            for c in r['candidates']:
                fh.write(json.dumps(c, ensure_ascii=False) + '\n')
    with open(config.OUT_DIR / 'canonical_ideas.jsonl', 'w', encoding='utf-8') as fh:
        for r in all_results:
            for c in r['canonical']:
                fh.write(json.dumps(c, ensure_ascii=False) + '\n')
    val_rows = []
    for r in all_results:
        val_rows.extend(r['validation'])
    pd.DataFrame(val_rows).to_csv(config.OUT_DIR / 'validation_report.csv', index=False)

    # summary metrics
    summary = {
        'total_conversations': len(all_results),
        'wall_clock_s': round(time.time() - t0, 1),
        'per_conversation': [
            {k: r[k] for k in ['conversation_id', 'persona_label', 'challenge',
                                'n_user_turns', 'n_candidates', 'n_canonical',
                                'n_grounded', 'n_grounded_fuzzy', 'n_ungrounded',
                                'ground_rate', 'agent1_valid_rate',
                                'agent1_latency_s', 'agent2_latency_s',
                                'agent3_latency_s']}
            for r in all_results
        ],
    }
    # aggregates
    valids = [r['agent1_valid_rate'] for r in all_results]
    grounds = [r['ground_rate'] for r in all_results]
    cands = [r['n_candidates'] for r in all_results]
    canons = [r['n_canonical'] for r in all_results]
    summary['aggregate'] = {
        'mean_agent1_valid_rate': float(np.mean(valids)),
        'mean_ground_rate': float(np.mean(grounds)),
        'median_candidates': float(np.median(cands)),
        'median_canonical': float(np.median(canons)),
        'median_compression_ratio': (float(np.median(np.array(cands) / np.maximum(1, np.array(canons))))),
    }
    with open(config.OUT_DIR / 'smoke_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print('\n=== SMOKE TEST SUMMARY ===')
    print(json.dumps(summary['aggregate'], indent=2))
    print(f'wall clock: {summary["wall_clock_s"]}s')
    print(f'outputs: {config.OUT_DIR}')


if __name__ == '__main__':
    main()
