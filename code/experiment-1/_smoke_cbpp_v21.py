"""Smoke test for the CBPP v2.1 methodology (see CBPP_v2.1_Methodology_Proposal.md).

Exercises the v2.1-specific additions end-to-end on the local Qwen-3-4B
(OpenVINO / Intel Arc 140T) — no API spend, no human raters:

    1.  Extended scoring schema with attribution fields  (§4.1)
    2.  Deterministic post-process score-cap rules        (§4.2)
    3.  Evidence-sufficiency gate                         (§4.3)
    4.  Attribution scaffold inside the lens prompt       (§6, pattern 5)
    5.  Adversarial audit pass (Dr_A) on scores >= 5      (§5)
        — verdict applied as a DOWN-ONLY arbiter

Scope: 2 conversations x 2 dimensions x 1 lens (Dr_C) x 1 paraphrase
       = 4 primary calls + up to 4 audit calls. The aim is to verify the
       PIPELINE works (schema validates, caps activate, audit is down-only),
       not to produce calibrated scores.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
sys.path.insert(0, str(ROOT))

from os_pipeline.llm_client import LLMClient
from os_pipeline.regulated.masking import mask

# ----------------------------------------------------------------------
# v2.1 EXTENDED SCHEMA  (§4.1)
# ----------------------------------------------------------------------

IdeaOrigin = Literal['user-originated', 'assistant-originated', 'co-developed', 'unclear']
AssistantOriginRisk = Literal['low', 'medium', 'high']


class CBPPScore(BaseModel):
    """Per-(lens x conv x dim x paraphrase) primary score. v2.1 spec §4.1."""
    score_1_to_7: int = Field(ge=1, le=7,
        description='Likert 1-7 for the USER on this dimension.')
    confidence_1_to_3: int = Field(ge=1, le=3,
        description='Judge confidence (1=low, 2=med, 3=high).')
    user_only_evidence_quotes: list[str] = Field(default_factory=list,
        max_length=3,
        description='1-3 verbatim USER-turn quotes only. Empty list -> '
                    'cap-2 by the post-process gate. Each quote <=200 chars.')
    idea_origin: IdeaOrigin = Field(
        description='Who originated the ideas being scored.')
    assistant_origin_risk: AssistantOriginRisk = Field(
        description='Risk that the score is inflated by assistant content.')
    rationale_under_40_words: str = Field(max_length=300,
        description='<=40 words / <=300 chars. Why this score.')


class AuditVerdict(BaseModel):
    """Adversarial audit pass (§5). Down-only arbiter."""
    audit_verdict: Literal['sustain', 'downgrade_by_1', 'downgrade_by_2'] = Field(
        description='Whether the primary score is supported by the '
                    'user-only quotes or appears inflated.')
    audit_reason_under_30_words: str = Field(max_length=200,
        description='<=30 words / <=200 chars. Inflation pattern, or '
                    '"supported" if sustain.')


# ----------------------------------------------------------------------
# Two dimensions: 1 shared-core (originality), 1 lens-specific (fluency)
# ----------------------------------------------------------------------

DIMS = {
    # Shared core C1 — §3.1
    'llm_originality_evidence': {
        'label': 'LLM-coded originality evidence',
        'question': (
            'To what extent does the USER demonstrate ORIGINAL creative '
            'behaviour in this conversation, within the rubric anchors?'
        ),
        'anchors': {
            1: 'User contributions are generic, conventional, or merely '
               'echo the assistant.',
            4: 'User shows some originality in framing or proposals, '
               'mixed with conventional moves.',
            7: 'User generates clearly non-obvious framings or proposals '
               'that go beyond the assistant\'s suggestions.',
        },
    },
    # Lens-specific C-a — §3.2 (Dr_C home turf)
    'user_ideational_fluency': {
        'label': 'User ideational fluency',
        'question': (
            'To what extent does the USER generate a productive volume of '
            'distinct candidate directions, options, or proposals?'
        ),
        'anchors': {
            1: 'User offers no candidate directions, or only one, or '
               'merely accepts what the assistant proposes.',
            4: 'User offers 2-4 distinct directions with some elaboration.',
            7: 'User generates many distinct directions, seeking breadth '
               'rather than depth on any single proposal.',
        },
    },
}


# ----------------------------------------------------------------------
# Prompt assembly — Dr_C with the §6 attribution-scaffold pattern
# ----------------------------------------------------------------------

DR_C_SYSTEM = (
    "You are Dr. C, a senior cognitive-creativity psychologist trained in "
    "the Guilford / Amabile / Nijstad tradition. You score USER behaviour "
    "in masked human-AI dialogue.\n\n"
    "SCORING DISCIPLINE:\n"
    "1. Score ONLY the user, not the assistant.\n"
    "2. Quote ONLY from USER turns. Verbatim, no paraphrase.\n"
    "3. ATTRIBUTION (v2.1 §6 pattern 5): if the idea originated with the "
    "assistant and the user merely accepted it, score lower. Mark "
    "`idea_origin` honestly: user-originated, assistant-originated, "
    "co-developed, or unclear.\n"
    "4. Use 1-7: 1=not at all evident, 4=moderately evident, 7=strongly evident.\n"
    "5. Never reward length, fluency, or verbosity.\n"
    "6. If no user-only evidence exists, you must still return a JSON "
    "object — the post-processor will cap the score at 2.\n"
    "7. Output ONLY valid JSON matching the CBPPScore schema."
)

AUDIT_SYSTEM = (
    "You are Dr. A, the adversarial audit judge (CBPP v2.1 §5). You review "
    "a primary score >= 5 and ask: is this score supported by the cited "
    "user-only evidence quotes, or is it INFLATED by transcript length, "
    "assistant-credited content, or fluency? You can SUSTAIN or "
    "DOWNGRADE-by-1 or DOWNGRADE-by-2. You cannot raise the score. "
    "Output ONLY valid JSON matching the AuditVerdict schema."
)


def build_primary_prompt(conv_text: str, dim_key: str) -> str:
    d = DIMS[dim_key]
    anc = '\n'.join(f"  {k}: {v}" for k, v in d['anchors'].items())
    return (
        f"DIMENSION: {dim_key}\n"
        f"QUESTION: {d['question']}\n"
        f"ANCHORS:\n{anc}\n\n"
        f"----- MASKED TRANSCRIPT -----\n{conv_text}\n"
        f"----- END TRANSCRIPT -----\n\n"
        f"Score the USER on `{dim_key}`. Return one CBPPScore JSON object only."
    )


def build_audit_prompt(conv_text: str, dim_key: str, primary: CBPPScore) -> str:
    quotes = '\n'.join(f"  - \"{q}\"" for q in primary.user_only_evidence_quotes) or '  (none provided)'
    return (
        f"DIMENSION: {dim_key}\n"
        f"PRIMARY SCORE: {primary.score_1_to_7}/7  "
        f"(idea_origin={primary.idea_origin}, "
        f"assistant_origin_risk={primary.assistant_origin_risk})\n"
        f"PRIMARY RATIONALE: {primary.rationale_under_40_words}\n"
        f"USER-ONLY EVIDENCE QUOTES CITED:\n{quotes}\n\n"
        f"----- MASKED TRANSCRIPT -----\n{conv_text}\n"
        f"----- END TRANSCRIPT -----\n\n"
        f"Is the primary score supported by these user-only quotes, or is "
        f"it inflated? Output one AuditVerdict JSON object only."
    )


# ----------------------------------------------------------------------
# Deterministic post-process — §4.2 attribution caps + §4.3 evidence gate
# ----------------------------------------------------------------------

def apply_attribution_cap(primary: CBPPScore) -> tuple[int, str]:
    """v2.1 §4.2 score-cap rules. Returns (capped_score, cap_label)."""
    s = primary.score_1_to_7
    # §4.3 evidence gate — overrides everything else
    if not primary.user_only_evidence_quotes:
        return min(s, 2), 'cap_2_no_user_evidence'
    # §4.2 origin-based caps
    if primary.idea_origin == 'assistant-originated':
        if primary.assistant_origin_risk == 'high':
            return min(s, 3), 'cap_3_assistant_taken_up'
        return min(s, 4), 'cap_4_assistant_with_minor_edits'
    if primary.idea_origin == 'co-developed':
        return min(s, 5), 'cap_5_user_reframed_assistant_idea'
    if primary.idea_origin == 'unclear':
        return min(s, 4), 'cap_4_unclear_attribution'
    return s, 'none'  # user-originated -> no cap


def apply_audit(capped_score: int, verdict: AuditVerdict) -> int:
    """§5 down-only arbiter."""
    if verdict.audit_verdict == 'sustain':
        return capped_score
    if verdict.audit_verdict == 'downgrade_by_1':
        return max(1, capped_score - 1)
    if verdict.audit_verdict == 'downgrade_by_2':
        return max(1, capped_score - 2)
    return capped_score


# ----------------------------------------------------------------------
# Transcript loader (mirrors os_pipeline.cat_panel.scorer._load_conversations)
# ----------------------------------------------------------------------

MAX_CHARS = 6000  # smaller than CAT-panel's 16k — Qwen on Arc is slower

def load_conv(conv_id: int) -> dict:
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    g = logs[logs['conversation_id'] == conv_id].sort_values('message_id').reset_index(drop=True)
    if g.empty:
        raise RuntimeError(f'conv {conv_id} not found')
    lines = []
    for ti, r in g.iterrows():
        msg = mask(str(r['message']) if pd.notna(r.get('message')) else '')
        spk = 'USER' if str(r['message_src']).lower() == 'user' else 'ASSISTANT'
        lines.append(f"{spk} turn {ti+1}: {msg}")
    txt = '\n\n'.join(lines)
    if len(txt) > MAX_CHARS:
        txt = txt[:MAX_CHARS] + '\n\n[...truncated...]'
    return dict(
        conversation_id=int(conv_id),
        persona_type=str(g['Persona_type'].iloc[0]),
        challenge=str(g['Corrected Challenge type'].iloc[0]),
        n_turns=len(g),
        transcript_masked=txt,
    )


# ----------------------------------------------------------------------
# Smoke driver
# ----------------------------------------------------------------------

def run_smoke(conv_ids: list[int]) -> pd.DataFrame:
    print(f'[smoke] loading local LLM (Qwen-3-4B / OpenVINO / Arc GPU)...')
    LLMClient.load()
    print(f'[smoke] ready.\n')

    rows = []
    for cid in conv_ids:
        conv = load_conv(cid)
        print(f'=== conv {cid}  persona={conv["persona_type"]}  '
              f'turns={conv["n_turns"]}  challenge={conv["challenge"]} ===')
        for dim_key in DIMS:
            print(f'\n--- [{cid}] {dim_key} ---')
            t0 = time.time()
            # ----- PRIMARY (Dr_C) -----
            primary, dbg = LLMClient.generate_json(
                schema_cls=CBPPScore,
                system_prompt=DR_C_SYSTEM,
                user_prompt=build_primary_prompt(conv['transcript_masked'], dim_key),
                temperature=0.2,
                max_new_tokens=900,
            )
            t_primary = time.time() - t0
            if primary is None:
                print(f'  PRIMARY FAILED: {dbg.get("parse_error","?")[:200]}')
                rows.append(dict(conv_id=cid, dim=dim_key, status='primary_failed',
                                 primary_score=None, cap=None, capped=None,
                                 audit_verdict=None, final=None,
                                 idea_origin=None, n_quotes=0,
                                 t_primary_s=round(t_primary,1), t_audit_s=None))
                continue
            raw = primary.score_1_to_7
            capped, cap_label = apply_attribution_cap(primary)
            print(f'  raw={raw}/7  origin={primary.idea_origin}  '
                  f'risk={primary.assistant_origin_risk}  '
                  f'n_quotes={len(primary.user_only_evidence_quotes)}')
            print(f'  cap_rule={cap_label}  -> capped={capped}/7')
            print(f'  rationale: {primary.rationale_under_40_words[:200]}')
            print(f'  primary time: {t_primary:.1f}s')

            # ----- AUDIT (Dr_A) — only if capped >= 5 -----
            final = capped
            audit_verdict_str = 'skipped'
            t_audit = None
            if capped >= 5:
                t1 = time.time()
                verdict, dbg2 = LLMClient.generate_json(
                    schema_cls=AuditVerdict,
                    system_prompt=AUDIT_SYSTEM,
                    user_prompt=build_audit_prompt(conv['transcript_masked'], dim_key, primary),
                    temperature=0.2,
                    max_new_tokens=200,
                )
                t_audit = time.time() - t1
                if verdict is None:
                    audit_verdict_str = 'audit_failed'
                    print(f'  AUDIT FAILED: {dbg2.get("parse_error","?")[:200]}')
                else:
                    audit_verdict_str = verdict.audit_verdict
                    final = apply_audit(capped, verdict)
                    print(f'  audit={verdict.audit_verdict}  reason="{verdict.audit_reason_under_30_words[:120]}"')
                    print(f'  audit time: {t_audit:.1f}s')
            print(f'  FINAL = {final}/7  (raw {raw} -> capped {capped} -> audit-final {final})')

            rows.append(dict(
                conv_id=cid, dim=dim_key, status='ok',
                primary_score=raw, cap=cap_label, capped=capped,
                audit_verdict=audit_verdict_str, final=final,
                idea_origin=primary.idea_origin,
                risk=primary.assistant_origin_risk,
                n_quotes=len(primary.user_only_evidence_quotes),
                t_primary_s=round(t_primary,1),
                t_audit_s=round(t_audit,1) if t_audit else None,
            ))
        print()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Pipeline self-checks (no LLM — verify deterministic logic is correct)
# ----------------------------------------------------------------------

def selfcheck() -> None:
    """Verify §4.2 caps and §5 down-only arbiter work as specified."""
    print('[smoke] running deterministic self-checks (no LLM)...')
    cases = [
        # (score, idea_origin, risk, quotes, expected_capped, expected_label)
        (7, 'user-originated',      'low',    ['q1'],   7, 'none'),
        (7, 'co-developed',         'low',    ['q1'],   5, 'cap_5_user_reframed_assistant_idea'),
        (7, 'assistant-originated', 'high',   ['q1'],   3, 'cap_3_assistant_taken_up'),
        (7, 'assistant-originated', 'medium', ['q1'],   4, 'cap_4_assistant_with_minor_edits'),
        (7, 'unclear',              'medium', ['q1'],   4, 'cap_4_unclear_attribution'),
        (7, 'user-originated',      'low',    [],       2, 'cap_2_no_user_evidence'),
        (3, 'user-originated',      'low',    ['q1'],   3, 'none'),  # below cap -> unchanged
    ]
    failed = 0
    for s, origin, risk, quotes, exp_score, exp_label in cases:
        p = CBPPScore(score_1_to_7=s, confidence_1_to_3=2,
                      user_only_evidence_quotes=quotes,
                      idea_origin=origin, assistant_origin_risk=risk,
                      rationale_under_40_words='test')
        got_score, got_label = apply_attribution_cap(p)
        ok = (got_score == exp_score and got_label == exp_label)
        if not ok:
            failed += 1
            print(f'  FAIL  {origin}/{risk}/quotes={len(quotes)}/s={s} -> '
                  f'got ({got_score},{got_label})  expected ({exp_score},{exp_label})')
    # audit arbiter
    audit_cases = [(6, 'sustain', 6), (6, 'downgrade_by_1', 5), (6, 'downgrade_by_2', 4), (1, 'downgrade_by_2', 1)]
    for capped, v, exp in audit_cases:
        av = AuditVerdict(audit_verdict=v, audit_reason_under_30_words='t')
        got = apply_audit(capped, av)
        if got != exp:
            failed += 1
            print(f'  FAIL  audit({capped}, {v}) -> {got} expected {exp}')
    print(f'[smoke] self-check: {"PASS" if failed == 0 else f"FAIL ({failed})"}'
          f'  ({len(cases)} cap cases + {len(audit_cases)} audit cases)')


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--conv-ids', type=int, nargs='+', default=[222, 318])
    ap.add_argument('--no-llm', action='store_true', help='Only run deterministic self-checks.')
    args = ap.parse_args()

    selfcheck()
    if args.no_llm:
        sys.exit(0)

    print()
    df = run_smoke(args.conv_ids)
    print('\n========================  RESULTS  ========================')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    print(df.to_string(index=False))
    out = ROOT / 'analysis_out' / 'smoke_cbpp_v21.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8')
    print(f'\nsaved -> {out}')

    # Summary diagnostics
    print('\n--- diagnostics ---')
    ok = df[df['status']=='ok']
    if len(ok):
        print(f'rows ok               : {len(ok)}/{len(df)}')
        print(f'caps activated        : {(ok["cap"]!="none").sum()}/{len(ok)}')
        print(f'evidence gate fired   : {(ok["cap"]=="cap_2_no_user_evidence").sum()}')
        print(f'audits triggered (>=5): {(ok["audit_verdict"]!="skipped").sum()}')
        if (ok['audit_verdict']!='skipped').any():
            audited = ok[ok['audit_verdict']!='skipped']
            print(f'  sustained           : {(audited["audit_verdict"]=="sustain").sum()}')
            print(f'  downgrade_by_1      : {(audited["audit_verdict"]=="downgrade_by_1").sum()}')
            print(f'  downgrade_by_2      : {(audited["audit_verdict"]=="downgrade_by_2").sum()}')
        # Down-only invariant check
        violated = ((ok['audit_verdict']!='skipped') & (ok['final'] > ok['capped'])).sum()
        print(f'down-only invariant   : {"OK" if violated == 0 else f"VIOLATED ({violated})"}')
