"""Condition/persona-label masking for blind scoring (Rule A2).

Every string reaching a scoring agent has GPT / Taylor / Alex / Divergent /
Convergent / Rational / BoundedRational replaced with generic labels. An
unmasked lookup is preserved for downstream statistics.
"""
from __future__ import annotations
import re


# Order matters — longer phrases first so substring hits work properly.
MASK_MAP = (
    (r'\bbounded[\s-]rationality\b',        'Assistant_A'),
    (r'\bstrictly[\s-]rational\b',          'Assistant_A'),
    (r'\bBoundedRational\b',                'Assistant_A'),
    (r'\bRational\b',                       'Assistant_A'),
    (r'\bDivergent\b',                      'Assistant_A'),
    (r'\bConvergent\b',                     'Assistant_A'),
    (r'\bTaylor\b',                         'Assistant_A'),
    (r'\bAlex\b',                           'Assistant_A'),
    # Preserve the lowercased "gpt" occurrences when they appear inside
    # brand strings like "chatgpt" — only block "gpt" as a free-standing word.
    (r'\bGPT\b',                            'Assistant_B'),
    (r'\bgpt\b',                            'Assistant_B'),
)

_compiled = tuple((re.compile(p, flags=re.IGNORECASE), sub)
                   for p, sub in MASK_MAP)


def mask(text: str | None) -> str:
    if text is None:
        return ''
    out = str(text)
    for rx, sub in _compiled:
        out = rx.sub(sub, out)
    return out


def mask_many(texts: list[str | None]) -> list[str]:
    return [mask(t) for t in texts]


def _self_test() -> None:
    cases = [
        ('GPT was helpful', 'Assistant_B was helpful'),
        ('The Divergent persona', 'The Assistant_A persona'),
        ('Taylor said X; Alex said Y', 'Assistant_A said X; Assistant_A said Y'),
        ('strictly rational bot', 'Assistant_A bot'),
        ('bounded rationality', 'Assistant_A'),
        ('chatgpt-like', 'chatgpt-like'),              # should not mask substring
        ('gpt asked', 'Assistant_B asked'),
        ('', ''),
    ]
    for inp, want in cases:
        got = mask(inp)
        assert got == want, f'mask({inp!r}) = {got!r} != {want!r}'


if __name__ == '__main__':
    _self_test()
    print('masking self-test: pass')
