"""Regenerate figD_design.png — polished academic layout."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'figures', 'figD_design.png')

# ── palette ──────────────────────────────────────────────────────────────
C_BG     = 'white'
C_YELLOW = '#FFF8D6';  E_YELLOW = '#C8A000'
C_BLUE   = '#D2E8F6';  E_BLUE   = '#2A6FA8'
C_ORANGE = '#FCDEC5';  E_ORANGE = '#C04A10'
C_GREEN  = '#D3EDD3';  E_GREEN  = '#2E8A2E'
C_ARROW  = '#555555'

# ── figure setup ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
})
fig, ax = plt.subplots(figsize=(12.5, 11))
fig.patch.set_facecolor(C_BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def box(x, y, w, h, text, fc, ec, fontweight='normal', fontsize=9.5, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.5', fc=fc, ec=ec, lw=lw, zorder=2
    ))
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            multialignment='center', linespacing=1.45, zorder=3)


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', lw=1.2, color=C_ARROW,
            connectionstyle=f'arc3,rad={rad}',
        ), zorder=1
    )


# ─────────────────────────────────────────────────────────────────────────
# Title + subtitle
ax.text(50, 98.5, 'Experiment 1 — Design',
        ha='center', fontsize=15, fontweight='bold', color='#111111')
ax.text(50, 96,
        'Within-subject condition × between-subject persona family × counterbalanced task and order',
        ha='center', fontsize=9.5, style='italic', color='#666666')

# ── root: participants ─────────────────────────────────────────────────
# center y=90, h=6  →  top=93, bottom=87
box(50, 90, 40, 6,
    '97 participants\n(paired sample, post-exclusions)',
    fc=C_YELLOW, ec=E_YELLOW, fontweight='bold', fontsize=11)

# ── within-subject banner ─────────────────────────────────────────────
# sits in gap between root bottom (87) and condition top (~82)
ax.text(50, 84.8,
        'Within-subject: every participant completes BOTH conditions',
        ha='center', fontsize=9.5, fontweight='bold', color='#222222')

# ── two condition boxes ───────────────────────────────────────────────
# center y=78, h=8.5  →  top=82.25, bottom=73.75
BW = 33
box(25, 78, BW, 8.5,
    'Standard LLM round\n(baseline)\n— 97 conversations —',
    fc=C_BLUE, ec=E_BLUE, fontweight='bold', fontsize=10.5)
box(75, 78, BW, 8.5,
    'Persona round\n(stance contract)\n— 97 conversations —',
    fc=C_ORANGE, ec=E_ORANGE, fontweight='bold', fontsize=10.5)
arrow(50, 87, 25, 82.25, rad=-0.12)
arrow(50, 87, 75, 82.25, rad= 0.12)

# ── counterbalance note ───────────────────────────────────────────────
# safely below condition bottom (73.75); y=71 → text spans ~70.3–71.7
ax.text(50, 71,
        'Round order counterbalanced  (Persona-first n = 40 · Standard-LLM-first n = 57)',
        ha='center', fontsize=9, style='italic', color='#666666')

# ── sub-baseline box (left branch) ───────────────────────────────────
# center y=62, h=7  →  top=65.5, bottom=58.5
box(25, 62, 30, 7,
    'Baseline Standard LLM\n(no persona prompt)\nn = 97',
    fc=C_BLUE, ec=E_BLUE, fontsize=9.5)
arrow(25, 73.75, 25, 65.5)

# ── between-subject label + four family boxes (right branch) ─────────
# label y=67.5 → text spans ~66.8–68.2  (above family tops at 65.5 → gap ~1.3)
ax.text(75, 67.5,
        'Between-subject: persona family',
        ha='center', fontsize=9.5, fontweight='bold', color=E_ORANGE)

# family boxes: center y=60, h=7  →  top=63.5, bottom=56.5
FAM_Y  = 60
FAM_W  = 10
FAM_H  = 7
fam_xs = [60, 70.5, 81, 91.5]
labels = ['Divergent\nn = 38', 'Convergent\nn = 41',
          'Rational\nn = 9', 'Bounded\nRational\nn = 9']
for xi, lbl in zip(fam_xs, labels):
    box(xi, FAM_Y, FAM_W, FAM_H, lbl,
        fc=C_ORANGE, ec=E_ORANGE, fontsize=9)
    arrow(75, 73.75, xi, FAM_Y + FAM_H / 2)

# ── tasks box ─────────────────────────────────────────────────────────
# center y=46, h=9  →  top=50.5, bottom=41.5
box(50, 46, 84, 9,
    'Two creative challenges (one per round, fully counterbalanced):\n'
    'Bicycle encouragement   ·   Community library revitalization\n'
    'Challenge order counterbalanced across the two rounds within participant',
    fc=C_GREEN, ec=E_GREEN, fontsize=9.5)
arrow(25, 58.5, 28, 50.5)
arrow(75, 56.5, 72, 50.5)

# ── output boxes ─────────────────────────────────────────────────────
# center y=32, h=7  →  top=35.5, bottom=28.5
box(25, 32, 33, 7,
    'Standard LLM round outputs\n97 conversations · paired sample n = 97',
    fc=C_BLUE, ec=E_BLUE, fontsize=9.5)
box(75, 32, 33, 7,
    'Persona round outputs\n97 conversations · paired sample n = 97',
    fc=C_ORANGE, ec=E_ORANGE, fontsize=9.5)
arrow(28, 41.5, 25, 35.5)
arrow(72, 41.5, 75, 35.5)

# ── footer ────────────────────────────────────────────────────────────
ax.text(50, 22,
        'Paired Persona-vs-Standard-LLM contrast is the primary within-subject test (n = 97). '
        'Family contrasts (Divergent vs Convergent) are between-subject within the Persona arm.\n'
        'Rational and BoundedRational family-level analyses are exploratory (n = 9 each).',
        ha='center', fontsize=8.8, style='italic', color='#555555',
        multialignment='center', linespacing=1.5)

# ─────────────────────────────────────────────────────────────────────────
fig.savefig(OUT, bbox_inches='tight', facecolor=C_BG, dpi=220)
print('wrote', OUT)
