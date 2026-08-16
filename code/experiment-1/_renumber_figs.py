"""
Renumber visible figure references in document.xml to make room for the new
Figure 4 (Taxonomy 2 flowchart) inserted in §2.2.

Rule:
  - For every visible "igure N" or "Figure N" with N >= 4, bump N by 1.
  - Skip lines 1726..1900 inclusive (the newly added §2.2 content), so my
    own "Figure 4" / "Figure 5" references stay correct.
  - Process from highest N down to N=4 to avoid double-substitution.
  - Only edit visible text. Skip alt-text in descr= attributes by checking
    the line does not contain 'descr=' before the match (alt-text is stale
    and not user-visible).
"""
import re, sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1/_unpacked/word/document.xml'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

SKIP_START, SKIP_END = 1726, 1900  # 1-indexed inclusive

# pattern: capture "igure N" or "Figure N" only when N >= 4
# we'll process N from high to low
n_changes = 0
for N in range(16, 3, -1):  # 16 down to 4
    new_N = N + 1
    pat_lower = re.compile(rf'(igure )({N})(?!\d)')
    pat_upper = re.compile(rf'(Figure )({N})(?!\d)')
    for i, line in enumerate(lines):
        ln = i + 1
        if SKIP_START <= ln <= SKIP_END:
            continue
        # Skip alt-text inside descr=
        if 'descr=' in line:
            continue
        # Only consider visible text — must be inside <w:t> or <m:t> or such.
        # Heuristic: line must contain a <w:t or </w:t to be visible body text.
        if '<w:t' not in line:
            continue
        new_line = pat_lower.sub(rf'\g<1>{new_N}', line)
        new_line = pat_upper.sub(rf'\g<1>{new_N}', new_line)
        if new_line != line:
            n_changes += (line.count(f'igure {N}') + line.count(f'Figure {N}'))
            lines[i] = new_line

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f'renumbered {n_changes} figure references (bumped Fig 4..16 → 5..17)')
