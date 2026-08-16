"""
Production run: Agents 1-5 end-to-end on N conversations.

Usage:
  python -m os_pipeline.production_run --n 10     # 10-conversation smoke
  python -m os_pipeline.production_run --all      # all 194 conversations

Writes to analysis_out/production/:
  candidates.jsonl            (one row per Agent-1 candidate)
  canonical_ideas.jsonl       (per-conv canonical, post-Agent-2)
  validation_report.csv       (Agent-3 status + missing words)
  categorized_ideas.csv       (Agent-4 category assignments)
  participant_originality.csv (Agent-5 three originality measures)
  run_metrics.json            (aggregate metrics, latencies, precision indicators)
  idea_embeddings.npy         (post-Agent-3 embedding matrix, for downstream use)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from os_pipeline import config
from os_pipeline.agent1_extractor import extract_for_message, CandidateRow
from os_pipeline.agent2_consolidator import consolidate, _get_embedder
from os_pipeline.agent3_validator import validate
from os_pipeline.agent4_categorizer import categorize
from os_pipeline.agent5_originality import compute_centroids, compute_originality
from os_pipeline.llm_client import LLMClient

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT = config.ROOT / 'analysis_out' / 'production'
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_EVERY = 20   # flush partial state every N conversations

FM = {'GPT': 'GPT', 'Divergent': 'Divergent', 'Convergent': 'Convergent',
      'strictly rational': 'Rational', 'bounded rationality': 'BoundedRational'}


# ---- checkpointing helpers ----

def _write_checkpoint(all_candidates, all_canonical_kept, all_validation,
                      per_conv_meta, out_dir):
    with open(out_dir / 'candidates.jsonl', 'w', encoding='utf-8') as fh:
        for c in all_candidates:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + '\n')
    with open(out_dir / 'canonical_ideas.jsonl', 'w', encoding='utf-8') as fh:
        for cid, c in all_canonical_kept:
            d = asdict(c); d['_source_conv'] = cid
            fh.write(json.dumps(d, ensure_ascii=False) + '\n')
    pd.DataFrame(all_validation).to_csv(out_dir / 'validation_report.csv', index=False)
    with open(out_dir / 'processed_conv_ids.json', 'w') as fh:
        json.dump(sorted(int(c) for c in per_conv_meta.keys()), fh)
    with open(out_dir / 'per_conv_meta.json', 'w') as fh:
        json.dump({str(k): v for k, v in per_conv_meta.items()}, fh, indent=2)


def _load_checkpoint(out_dir):
    from os_pipeline.agent1_extractor import CandidateRow
    from os_pipeline.agent2_consolidator import CanonicalRow
    processed = set()
    all_candidates: list = []
    all_canonical_kept: list = []
    all_validation: list = []
    per_conv_meta: dict = {}
    p = out_dir / 'processed_conv_ids.json'
    if p.exists():
        processed = set(json.load(open(p)))
        print(f'[checkpoint] resuming: {len(processed)} conversations already processed')
        if (out_dir / 'candidates.jsonl').exists():
            with open(out_dir / 'candidates.jsonl', encoding='utf-8') as fh:
                for line in fh:
                    d = json.loads(line)
                    all_candidates.append(CandidateRow(**d))
        if (out_dir / 'canonical_ideas.jsonl').exists():
            with open(out_dir / 'canonical_ideas.jsonl', encoding='utf-8') as fh:
                for line in fh:
                    d = json.loads(line)
                    src = d.pop('_source_conv', d.get('conversation_id'))
                    all_canonical_kept.append((int(src), CanonicalRow(**d)))
        if (out_dir / 'validation_report.csv').exists():
            try:
                df = pd.read_csv(out_dir / 'validation_report.csv')
                all_validation = df.to_dict(orient='records')
            except Exception:
                all_validation = []
        if (out_dir / 'per_conv_meta.json').exists():
            per_conv_meta = {int(k): v for k, v in
                             json.load(open(out_dir / 'per_conv_meta.json')).items()}
    return processed, all_candidates, all_canonical_kept, all_validation, per_conv_meta


def _persona_label(pt: str) -> str:
    return FM.get(pt, str(pt))


def load_logs() -> pd.DataFrame:
    return (pd.read_csv(config.LOGS_CSV)
              .sort_values(['conversation_id', 'message_id'])
              .reset_index(drop=True))


def select_conversations(logs: pd.DataFrame, n: int, seed: int = 42) -> list[int]:
    """Pick n conversations stratified by condition x challenge for smoke runs,
    or all conversations if n >= total."""
    convs = (logs.groupby('conversation_id')
                 .agg(user=('User_id', 'first'),
                      persona_type=('Persona_type', 'first'),
                      challenge=('Corrected Challenge type', 'first'),
                      n_user=('message_src', lambda s: (s == 'user').sum()))
                 .reset_index())
    convs['persona_label'] = convs['persona_type'].map(_persona_label)
    convs['condition'] = np.where(convs['persona_type'] == 'GPT', 'GPT', 'Persona')

    if n >= len(convs):
        return convs['conversation_id'].tolist()

    # keep only convs with >=3 user turns (avoid empty/degenerate ones)
    convs = convs[convs.n_user >= 3]

    rng = np.random.default_rng(seed)
    pieces = []
    labels = ['GPT', 'Divergent', 'Convergent', 'Rational', 'BoundedRational']
    per_label = max(1, n // len(labels))
    for lbl in labels:
        pool = convs[convs.persona_label == lbl]
        if len(pool) == 0:
            continue
        k = min(per_label, len(pool))
        pieces.append(pool.sample(n=k, random_state=int(rng.integers(1 << 30))))
    picked = pd.concat(pieces)
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed)
    return picked['conversation_id'].tolist()


def run(cids: list[int], logs: pd.DataFrame) -> dict:
    LLMClient.load()                      # warm up early
    _get_embedder()                        # warm up embed model

    # resume from checkpoint if present
    processed, all_candidates, all_canonical_kept, all_validation, per_conv_meta = \
        _load_checkpoint(OUT)

    # filter out already-processed cids (auto-resume)
    todo_cids = [c for c in cids if int(c) not in processed]
    if processed:
        print(f'[checkpoint] {len(processed)} already done; {len(todo_cids)} remaining')

    t0_total = time.time()
    for ci, cid in enumerate(todo_cids):
        try:
            g = logs[logs.conversation_id == cid].sort_values('message_id').reset_index(drop=True)
            pt = str(g.iloc[0]['Persona_type'])
            persona_label = _persona_label(pt)
            challenge = str(g.iloc[0].get('Corrected Challenge type', 'unknown'))
            condition = 'GPT' if persona_label == 'GPT' else 'Persona'
            user = int(g.iloc[0]['User_id'])

            print(f'\n[{ci+1}/{len(todo_cids)}] conv {cid}  ({persona_label}, {challenge}, user={user}) — {len(g)} msgs', flush=True)

            # Agent 1
            t0 = time.time()
            user_idx = g.index[g.message_src == 'user'].tolist()
            candidates: list[CandidateRow] = []
            for utix, ri in enumerate(user_idx):
                row = g.iloc[ri]
                prev_asst = None
                if ri > 0 and g.iloc[ri - 1]['message_src'] == 'assistant':
                    prev_asst = str(g.iloc[ri - 1]['message'])
                user_msg = str(row['message'])
                if len(user_msg.strip()) < 2:
                    continue
                rows, _dbg = extract_for_message(
                    conversation_id=int(cid),
                    message_id=int(row['message_id']),
                    user_turn_idx=utix,
                    prev_assistant=prev_asst,
                    user_msg=user_msg,
                    challenge=challenge,
                    persona_label=persona_label,
                )
                candidates.extend(rows)
            a1_s = time.time() - t0
            print(f'  A1: {len(candidates)} candidates ({a1_s:.1f}s)', flush=True)

            # Agent 2
            t1 = time.time()
            canon = consolidate(candidates, conversation_id=int(cid))
            a2_s = time.time() - t1

            # Agent 3
            t2 = time.time()
            user_messages = [str(m) for m in g[g.message_src == 'user']['message'].tolist()]
            kept, reports = validate(canon, user_messages)
            a3_s = time.time() - t2

            n_grounded = sum(1 for r in reports if r.status == 'grounded')
            n_fuzzy = sum(1 for r in reports if r.status == 'grounded_fuzzy')
            n_hall = sum(1 for r in reports if r.status == 'title_hallucination')
            n_ung = sum(1 for r in reports if r.status == 'ungrounded')
            print(f'  A2: {len(canon)} canonical ({a2_s:.1f}s)  '
                  f'A3: g={n_grounded} f={n_fuzzy} halluc={n_hall} ung={n_ung}  kept={len(kept)}', flush=True)

            all_candidates.extend(candidates)
            all_canonical_kept.extend((cid, c) for c in kept)
            for r in reports:
                d = asdict(r); d['persona_label'] = persona_label; d['condition'] = condition
                all_validation.append(d)
            per_conv_meta[int(cid)] = dict(
                user=user, condition=condition, persona_label=persona_label,
                challenge=challenge, n_ideas=len(kept),
                a1_s=a1_s, a2_s=a2_s, a3_s=a3_s, n_candidates=len(candidates),
                n_canonical=len(canon), n_kept=len(kept),
            )
            processed.add(int(cid))

            # ---- checkpoint every CHECKPOINT_EVERY ----
            if (ci + 1) % CHECKPOINT_EVERY == 0:
                _write_checkpoint(all_candidates, all_canonical_kept, all_validation,
                                  per_conv_meta, OUT)
                print(f'  [checkpoint] flushed after {ci+1} convs', flush=True)
        except Exception as e:
            print(f'  !! conv {cid} failed: {type(e).__name__}: {e}', flush=True)
            # continue with next conv; don't halt the whole run
            continue

    # final flush of per-conv state
    _write_checkpoint(all_candidates, all_canonical_kept, all_validation,
                      per_conv_meta, OUT)
    t_post = time.time() - t0_total
    print(f'\nAgents 1-3 wall clock (this run): {t_post/60:.1f} min  total kept ideas: {len(all_canonical_kept)}')

    # Agent 4 — cross-participant categorization
    canon_rows_only = [c for (_cid, c) in all_canonical_kept]
    t4 = time.time()
    cats, V_all, cat_summary = categorize(canon_rows_only, min_cluster_size=4,
                                           return_embeddings=True)
    a4_s = time.time() - t4
    print(f'[A4] clustered {cat_summary["n_ideas"]} ideas -> {cat_summary["n_clusters"]} '
          f'clusters + {cat_summary["n_unclustered"]} unclustered ({a4_s:.1f}s)')
    if cat_summary['cluster_sizes']:
        top5 = sorted(cat_summary['cluster_sizes'].items(),
                      key=lambda kv: -kv[1])[:5]
        print('  top clusters: ' + ', '.join(f'{k}({v})' for k, v in top5))

    # Agent 5 — participant originality
    t5 = time.time()
    # group embeddings by conversation
    idea_V_by_conv: dict[int, np.ndarray] = {}
    if V_all is not None:
        for i, (cid, _c) in enumerate(all_canonical_kept):
            idea_V_by_conv.setdefault(cid, []).append(V_all[i])
        idea_V_by_conv = {k: np.stack(v) for k, v in idea_V_by_conv.items()}
    conv_ids_ordered = list(dict.fromkeys(cid for (cid, _c) in all_canonical_kept))
    centroids, aligned = compute_centroids(conv_ids_ordered, idea_V_by_conv)
    origs = compute_originality(centroids, aligned, per_conv_meta)
    a5_s = time.time() - t5
    print(f'[A5] originality computed for {len(origs)} participants ({a5_s:.1f}s)')

    # ---- persist (Agents 4-5 outputs; 1-3 already checkpointed) ----
    _write_checkpoint(all_candidates, all_canonical_kept, all_validation,
                      per_conv_meta, OUT)
    pd.DataFrame([asdict(c) for c in cats]).to_csv(
        OUT / 'categorized_ideas.csv', index=False)
    pd.DataFrame([asdict(o) for o in origs]).to_csv(
        OUT / 'participant_originality.csv', index=False)
    if V_all is not None:
        np.save(OUT / 'idea_embeddings.npy', V_all)
    if centroids.size:
        np.save(OUT / 'participant_centroids.npy', centroids)

    metrics = {
        'n_conversations': len(cids),
        'wall_clock_s': round(time.time() - t0_total, 1),
        'agent4_cluster_summary': cat_summary,
        'per_conversation': {str(k): v for k, v in per_conv_meta.items()},
        'totals': {
            'n_candidates': len(all_candidates),
            'n_canonical_kept': len(all_canonical_kept),
            'n_participants_with_origs': len(origs),
        },
        'originality_descriptive': (
            pd.DataFrame([asdict(o) for o in origs])
              .groupby('condition')[['orig_same', 'orig_all', 'orig_cross']]
              .mean().round(4).to_dict()
            if origs else {}
        ),
    }
    with open(OUT / 'run_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'\nDONE. outputs in {OUT}')
    print('\n--- originality by condition (mean) ---')
    if origs:
        df = pd.DataFrame([asdict(o) for o in origs])
        print(df.groupby('condition')[['orig_same', 'orig_all', 'orig_cross']].agg(['count', 'mean', 'std']).round(3).to_string())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10,
                    help='number of conversations to run (default 10 for smoke)')
    ap.add_argument('--all', action='store_true', help='run on all conversations')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    logs = load_logs()
    if args.all:
        cids = select_conversations(logs, 10**9)
        print(f'PRODUCTION: running on all {len(cids)} conversations')
    else:
        cids = select_conversations(logs, args.n, seed=args.seed)
        print(f'SMOKE: running on {len(cids)} stratified conversations')
    run(cids, logs)


if __name__ == '__main__':
    main()
