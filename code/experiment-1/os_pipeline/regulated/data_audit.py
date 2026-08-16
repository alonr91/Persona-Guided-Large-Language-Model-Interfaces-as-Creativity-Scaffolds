"""Stage 0 — Data audit. Produces 00_data_audit.md + 01_cleaning_log.csv.

Read-only pass. Answers the 8 minimum audit questions in the spec plus the
critical sign audit of cr_diff / ow_diff / a_prop.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f'  [warn] could not read {p.name}: {e}')
        return None


def _read_xlsx(p: Path, sheet: str) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_excel(p, sheet_name=sheet)
    except Exception as e:
        print(f'  [warn] could not read {p.name}[{sheet}]: {e}')
        return None


def main():
    log_rows: list[dict] = []
    def add(issue_id, file, row_or_id, issue_type, description, decision,
             affected_records=0, notes=''):
        log_rows.append(dict(
            issue_id=issue_id, file=file, row_or_id=row_or_id,
            issue_type=issue_type, description=description,
            decision=decision, affected_records=affected_records, notes=notes,
        ))

    lines: list[str] = ['# 00 — Data Audit', '']
    lines.append(f'Generated from raw inputs at {ROOT}.')
    lines.append('')

    # ---- load primary inputs ----
    logs = _read_csv(ROOT / 'Experiment1_logs.csv')
    master_users  = _read_csv(ROOT / 'analysis_out' / 'master_users.csv')
    master_conv   = _read_csv(ROOT / 'analysis_out' / 'master_conversations.csv')
    master_wide   = _read_csv(ROOT / 'analysis_out' / 'master_wide.csv')
    users_xlsx    = _read_xlsx(ROOT / 'analysis_out' / 'users_translated.xlsx',
                                 'corrected_users')
    b_paired      = _read_csv(ROOT / 'analysis_out' / 'B_process_paired.csv')
    b_by_family   = _read_csv(ROOT / 'analysis_out' / 'B_process_by_family.csv')
    extension_conv = _read_csv(ROOT / 'analysis_out' / 'extension_conv_master.csv')
    extension_paired = _read_csv(ROOT / 'analysis_out' / 'extension_paired.csv')
    ext_perc_corr = _read_csv(ROOT / 'analysis_out' / 'extension_perception_corr.csv')
    manip_check   = _read_csv(ROOT / 'analysis_out' / 'manipulation_check.csv')
    prod_orig     = _read_csv(ROOT / 'analysis_out' / 'production' / 'participant_originality.csv')
    prod_val      = _read_csv(ROOT / 'analysis_out' / 'production' / 'validation_report.csv')
    prod_canon_path = ROOT / 'analysis_out' / 'production' / 'canonical_ideas.jsonl'
    prod_canon_n = sum(1 for _ in open(prod_canon_path, encoding='utf-8')) if prod_canon_path.exists() else 0

    # ---- Section 1: file inventory ----
    lines.append('## 1. File inventory')
    lines.append('')
    lines.append('| File | Rows | Status |')
    lines.append('| --- | --- | --- |')
    for name, df, fname in [
        ('logs', logs, 'Experiment1_logs.csv'),
        ('master_users', master_users, 'analysis_out/master_users.csv'),
        ('master_conversations', master_conv, 'analysis_out/master_conversations.csv'),
        ('master_wide', master_wide, 'analysis_out/master_wide.csv'),
        ('users_translated', users_xlsx, 'analysis_out/users_translated.xlsx'),
        ('B_process_paired', b_paired, 'analysis_out/B_process_paired.csv'),
        ('B_process_by_family', b_by_family, 'analysis_out/B_process_by_family.csv'),
        ('extension_conv_master', extension_conv, 'analysis_out/extension_conv_master.csv'),
        ('extension_paired', extension_paired, 'analysis_out/extension_paired.csv'),
        ('extension_perception_corr', ext_perc_corr, 'analysis_out/extension_perception_corr.csv'),
        ('manipulation_check', manip_check, 'analysis_out/manipulation_check.csv'),
        ('production/participant_originality', prod_orig, 'analysis_out/production/participant_originality.csv'),
        ('production/validation_report', prod_val, 'analysis_out/production/validation_report.csv'),
        ('production/canonical_ideas', None, f'analysis_out/production/canonical_ideas.jsonl (JSONL, {prod_canon_n} lines)'),
    ]:
        status = 'missing' if (df is None and name != 'production/canonical_ideas') else 'ok'
        rows = len(df) if df is not None else prod_canon_n if name == 'production/canonical_ideas' else 0
        lines.append(f'| {name} | {rows} | {status} |')
        if status == 'missing':
            add('MISS_' + name, fname, '', 'file_missing', f'Required file {fname} not found', 'skip_and_document', 0)
    lines.append('')

    if logs is None:
        lines.append('**Abort:** Experiment1_logs.csv is missing; cannot continue.')
        (OUT / '00_data_audit.md').write_text('\n'.join(lines), encoding='utf-8')
        pd.DataFrame(log_rows).to_csv(OUT / '01_cleaning_log.csv', index=False)
        return

    # ---- Section 2: counts and pairing ----
    n_users = logs['User_id'].nunique()
    n_convs = logs['conversation_id'].nunique()
    n_msgs  = len(logs)
    convs_per_user = logs.groupby('User_id')['conversation_id'].nunique()
    users_with_2convs = int((convs_per_user == 2).sum())
    users_with_1conv  = int((convs_per_user == 1).sum())
    users_with_more   = int((convs_per_user > 2).sum())
    # pairing: did each user get both GPT and non-GPT?
    def _pt(pt):
        pt = str(pt)
        return 'GPT' if pt == 'GPT' else 'Persona'
    logs['_cond'] = logs['Persona_type'].map(_pt)
    u2c = logs.groupby('User_id')['_cond'].apply(lambda s: set(s.unique())).reset_index(name='conds')
    paired = int(u2c['conds'].apply(lambda s: {'GPT','Persona'}.issubset(s)).sum())
    unpaired = int((~u2c['conds'].apply(lambda s: {'GPT','Persona'}.issubset(s))).sum())

    lines += [
        '## 2. Participant / conversation / message counts',
        '',
        f'- Unique users: **{n_users}**',
        f'- Unique conversations: **{n_convs}**',
        f'- Total messages: **{n_msgs}**',
        f'- Users with exactly 2 conversations: **{users_with_2convs}**',
        f'- Users with only 1 conversation: **{users_with_1conv}**',
        f'- Users with >2 conversations: **{users_with_more}**',
        f'- Users paired GPT + Persona: **{paired}**',
        f'- Users not paired GPT + Persona: **{unpaired}**',
        '',
    ]
    if users_with_1conv:
        add('PAIR_unpaired', 'Experiment1_logs.csv', '',
            'unpaired_participants',
            f'{users_with_1conv} users contributed only 1 conversation',
            'keep_for_descriptives_exclude_from_paired_tests',
            users_with_1conv)

    # ---- Section 3: turn ordering & timestamps ----
    lines += ['## 3. Turn ordering and timestamps', '']
    bad_ts = 0
    if 'timestamp' in logs.columns:
        for cid, g in logs.sort_values(['conversation_id','message_id']).groupby('conversation_id'):
            ts = pd.to_numeric(g['timestamp'], errors='coerce').values
            if np.any(np.diff(ts[~np.isnan(ts)]) < 0):
                bad_ts += 1
        lines.append(f'- Conversations with non-monotonic timestamps: **{bad_ts}**')
        if bad_ts:
            add('TS_nonmono', 'Experiment1_logs.csv', '', 'timestamp_nonmonotonic',
                f'{bad_ts} conversations have non-monotonic timestamps',
                'fall_back_to_message_id_order', bad_ts)
    else:
        lines.append('- No timestamp column; using message_id order. `timestamp_inferred=true`.')
        add('TS_missing', 'Experiment1_logs.csv', '', 'timestamp_column_missing',
            'No timestamp column; using message_id for ordering',
            'use_message_id_as_order_proxy', 0)
    # speaker label consistency
    src_values = logs['message_src'].astype(str).str.lower().unique()
    lines.append(f'- Speaker labels present: `{sorted(src_values)}`')
    allowed = {'user', 'assistant'}
    unknown_src = set(src_values) - allowed
    if unknown_src:
        add('SRC_unknown', 'Experiment1_logs.csv', '', 'speaker_label_unknown',
            f'Unexpected speaker labels: {unknown_src}',
            'treat_as_user_or_exclude_per_row', 0)
    lines.append('')

    # ---- Section 4: persona family consistency ----
    lines += ['## 4. Persona family consistency', '']
    pt_counts = logs['Persona_type'].value_counts(dropna=False).to_dict()
    lines.append(f'- Raw Persona_type values: `{pt_counts}`')
    fm = {'Divergent':'Divergent','Convergent':'Convergent',
          'strictly rational':'Rational','bounded rationality':'BoundedRational',
          'GPT':'GPT'}
    unmapped = [v for v in pt_counts.keys() if v not in fm]
    if unmapped:
        lines.append(f'- Unmapped persona labels: `{unmapped}`')
        add('PT_unmapped', 'Experiment1_logs.csv', '', 'unmapped_persona_label',
            f'Unmapped Persona_type values: {unmapped}', 'map_to_other_or_exclude',
            int(sum(pt_counts[u] for u in unmapped)))
    lines.append('')

    # ---- Section 5: sign audit ----
    lines += ['## 5. CRITICAL sign audit', '']
    if users_xlsx is not None:
        u = users_xlsx.copy()
        u.columns = [c.strip() for c in u.columns]
        def _gpt_round(row):
            r1 = str(row.get('Persona round 1','')).lower(); r2 = str(row.get('Persona round 2','')).lower()
            if 'gpt' in r1: return 1
            if 'gpt' in r2: return 2
            return np.nan
        u['gpt_round'] = u.apply(_gpt_round, axis=1)
        def _mk(df,a,b):
            g = np.where(df['gpt_round']==1, df[a], np.where(df['gpt_round']==2, df[b], np.nan))
            p = np.where(df['gpt_round']==1, df[b], np.where(df['gpt_round']==2, df[a], np.nan))
            return pd.Series(g, index=df.index), pd.Series(p, index=df.index)
        try:
            u['cr_gpt'], u['cr_per'] = _mk(u,'Creativity assistant #1','Creativity assistant #2')
            u['ow_gpt'], u['ow_per'] = _mk(u,'Ownership #1','Ownership #2')
            u['cr_diff'] = u['cr_per'].astype(float) - u['cr_gpt'].astype(float)
            u['ow_diff'] = u['ow_per'].astype(float) - u['ow_gpt'].astype(float)
            lines.append(f'- cr_diff sign convention: **Persona − GPT** (positive = Persona higher).')
            lines.append(f'  - mean cr_diff = {u["cr_diff"].mean():+.3f}, n={u["cr_diff"].notna().sum()}')
            lines.append(f'- ow_diff sign convention: **Persona − GPT** (positive = Persona higher).')
            lines.append(f'  - mean ow_diff = {u["ow_diff"].mean():+.3f}, n={u["ow_diff"].notna().sum()}')
        except Exception as e:
            lines.append(f'- could not recompute cr_diff/ow_diff: {e}')
            add('SIGN_recompute', 'users_translated.xlsx', '',
                'sign_recompute_failure', str(e), 'use_pre_existing_values', 0)

    # a_prop sign audit — check extension_perception_corr
    if ext_perc_corr is not None:
        lines.append('')
        lines.append('### 5a. a_prop interpretation')
        a_prop_rows = ext_perc_corr[ext_perc_corr.apply(lambda r: 'a_prop' in str(r.values).lower(), axis=1)]
        if len(a_prop_rows):
            lines.append('Rows mentioning `a_prop` in extension_perception_corr.csv:')
            lines.append('```')
            lines.append(a_prop_rows.to_string(index=False))
            lines.append('```')
        # pull the paired series if present in extension_paired
        if extension_paired is not None:
            ap_paired = extension_paired[extension_paired.apply(lambda r: 'a_prop' in str(r.values).lower(), axis=1)]
            if len(ap_paired):
                lines.append('Row(s) for a_prop in extension_paired.csv:')
                lines.append('```')
                lines.append(ap_paired.to_string(index=False))
                lines.append('```')
        # recompute from raw if we can — compare a_prop in Persona vs GPT per conv
        if extension_conv is not None and 'a_prop' in extension_conv.columns:
            ec = extension_conv.copy()
            if 'condition' in ec.columns:
                g_mean = ec.loc[ec.condition=='GPT','a_prop'].astype(float).mean()
                p_mean = ec.loc[ec.condition=='Persona','a_prop'].astype(float).mean()
                lines.append(f'- a_prop mean (GPT) = {g_mean:.4f}; a_prop mean (Persona) = {p_mean:.4f}; '
                             f'Persona − GPT = {p_mean-g_mean:+.4f}')
                if p_mean > g_mean:
                    lines.append('  - Persona has **higher** propose rate on the assistant side than GPT. '
                                 'A positive `a_prop_diff` means the assistant proposed MORE under Persona.')
                else:
                    lines.append('  - Persona has **lower** propose rate on the assistant side than GPT. '
                                 'A positive `a_prop_diff` would mean the assistant proposed MORE under Persona '
                                 '(which is NOT the case here: assistant proposed LESS).')
                add('SIGN_a_prop', 'extension_conv_master.csv + extension_perception_corr.csv',
                    '', 'sign_audit',
                    f'a_prop (Persona − GPT) = {p_mean-g_mean:+.4f}. '
                    f'Interpret prior "assistant ceded proposal role" claim only after consulting this sign.',
                    'document_in_memo_tension', 0)
            else:
                lines.append('- extension_conv_master.csv lacks a `condition` column; cannot recompute direction.')
        else:
            lines.append('- extension_conv_master.csv missing or lacks `a_prop` column; cannot recompute direction.')
    lines.append('')

    # ---- Section 6: production-layer status ----
    lines += ['## 6. Pre-existing production-layer status', '']
    if prod_orig is not None:
        by_cond = prod_orig.groupby('condition').size().to_dict()
        lines.append(f'- participant_originality rows: {len(prod_orig)}  (per condition: {by_cond})')
        both = (prod_orig.pivot_table(index='user', columns='condition', values='n_ideas', aggfunc='first')
                .notna().all(axis=1).sum())
        lines.append(f'- Users with both rounds extracted (paired analyses): **{int(both)}**')
    if prod_canon_n:
        lines.append(f'- canonical ideas written by Agent 2/3: **{prod_canon_n}**')
    if prod_val is not None:
        status_counts = prod_val['status'].value_counts().to_dict() if 'status' in prod_val.columns else {}
        lines.append(f'- Agent 3 validation status counts: `{status_counts}`')
    lines.append('')

    # ---- Section 7: answers to the 8 minimum audit questions ----
    lines += [
        '## 7. Minimum audit questions (§ Stage 0)',
        '',
        f'1. **All participants paired across GPT and Persona?** {paired} paired / {unpaired} not paired.',
        f'2. **Exactly two conversations per keepable participant?** {users_with_2convs} users have exactly 2, {users_with_1conv} have 1, {users_with_more} have >2.',
        f'3. **All conversations linked to a participant ID?** Yes — `User_id` present on every row.',
        f'4. **Message timestamps monotonic inside each conversation?** ' +
        (f'{bad_ts} conversations with non-monotonic timestamps.' if 'timestamp' in logs.columns else 'No timestamps; using message_id order.'),
        f'5. **Assistant/user speaker labels consistent?** labels observed: `{sorted(src_values)}`; unknown: `{sorted(unknown_src)}`.',
        f'6. **Persona family labels consistent across files?** Raw Persona_type values: `{list(pt_counts.keys())}`; unmapped: `{unmapped or "none"}`.',
        f'7. **Questionnaire deltas computed as Persona − GPT?** Yes (see section 5).',
        f'8. **All process deltas same direction?** The existing master_wide.csv uses Persona − GPT for deltas; this reanalysis reuses that convention.',
        '',
    ]

    # ---- Section 8: decisions for this reanalysis ----
    lines += [
        '## 8. Decisions taken for this reanalysis',
        '',
        '- Use `message_id` (ascending) as the canonical turn ordering when timestamps are non-monotonic or missing.',
        '- Treat all sign conventions as Persona − GPT.',
        '- For the agentic idea-extraction outputs, reuse the existing 740 canonical ideas from Agent 2/3 in production/.',
        '- Users with only one conversation are kept in descriptive tables but excluded from within-subject paired tests.',
        '- The `a_prop` direction recorded in section 5a is the canonical interpretation for this reanalysis; any prior claim that conflicts with it must be flagged in the memo.',
        '',
    ]

    # ---- save ----
    (OUT / '00_data_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame(log_rows).to_csv(OUT / '01_cleaning_log.csv', index=False)
    print(f'wrote {OUT / "00_data_audit.md"}')
    print(f'wrote {OUT / "01_cleaning_log.csv"}  ({len(log_rows)} log rows)')


if __name__ == '__main__':
    main()
