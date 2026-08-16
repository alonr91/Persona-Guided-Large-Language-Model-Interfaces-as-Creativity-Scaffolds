"""
Academic flowchart for Taxonomy 2 (Claude-labeled continuous discourse construct)
pipeline. Saved to figures/taxonomy2_flowchart.png.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'figures', 'taxonomy2_flowchart.png')

# Canvas
fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

# Color palette (muted academic)
C_DATA   = '#D9E8F5'   # data inputs
C_PROC   = '#E8EDF3'   # processing
C_MODEL  = '#FBE7C6'   # model training
C_OUTPUT = '#D9EAD3'   # final outputs
C_BORDER = '#36454F'
C_ACCENT = '#7B3F61'   # arrows / accents

def box(x, y, w, h, text, fill, bold=False, fs=10, italic=False):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.2",
                       linewidth=1.4, edgecolor=C_BORDER, facecolor=fill)
    ax.add_patch(p)
    weight = 'bold' if bold else 'normal'
    style  = 'italic' if italic else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=weight, fontstyle=style, color='#1a1a1a',
            wrap=True)

def arrow(x1, y1, x2, y2, label=None, label_offset=(0,0), curved=False):
    rad = 0.0 if not curved else 0.25
    a = FancyArrowPatch((x1,y1), (x2,y2),
                        arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.6, color=C_ACCENT,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=8.2,
                style='italic', color='#444',
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='none', alpha=0.85))

# ===================== Title bar =====================
ax.text(50, 96.5,
        "Taxonomy 2 — Claude-labeled continuous discourse construct pipeline",
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1a1a1a')
ax.text(50, 92.5,
        "Stratified gold coding  →  weak supervision  →  full-corpus propagation  →  paired within-subject contrasts",
        ha='center', va='center', fontsize=10.2, style='italic', color='#444')

# ===================== Stage 1: Raw data =====================
ax.text(8, 84, "STAGE 1\nInputs", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_ACCENT)
box(28, 84, 30, 8,
    "Raw message corpus\n3,412 messages · 194 conversations · 97 users",
    C_DATA, bold=True, fs=10)
box(63, 84, 28, 8,
    "Sentence-Transformer encoder\nall-MiniLM-L6-v2 (384-dim, L2-normalised)",
    C_DATA, fs=9.5)
arrow(43, 84, 49, 84)

# ===================== Stage 2: Stratified gold sample =====================
ax.text(8, 70, "STAGE 2\nGold-coding", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_ACCENT)
box(28, 70, 30, 9.5,
    "Stratified gold sample (n ≈ 300–400)\nstrata = condition × persona-family ×\nmessage-source × position (early/mid/late)",
    C_PROC, fs=9.2)
arrow(28, 79.2, 28, 75.2, label="balanced draw", label_offset=(8,0))

box(63, 70, 28, 9.5,
    "Claude gold coding\n7 ordinal constructs (0–3)\n+ tone, qtype categoricals",
    C_PROC, bold=True, fs=9.5)
arrow(43, 70, 49, 70, label="hand-coded labels")

# small legend listing constructs
ax.text(94, 70,
        "Constructs:\n• expansion\n• contraction\n• critique\n• certainty\n• commit\n• reframe\n• propose-new-idea",
        ha='left', va='center', fontsize=7.8, color='#222',
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFF7E0', ec=C_BORDER, lw=0.8))

# ===================== Stage 3: Feature assembly =====================
ax.text(8, 55, "STAGE 3\nFeatures", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_ACCENT)
box(50, 55, 70, 9,
    "Per-message feature vector  (≈ 400 dims)\n"
    "384-d SBERT embedding  ⊕  9 lexical cues (hedging, asserting, "
    "'what if', '?', length, …)\n⊕  speaker one-hot  ⊕  persona-family one-hot  ⊕  normalised turn-position",
    C_PROC, fs=9.4)
arrow(28, 65.3, 35, 59.6, curved=True)
arrow(63, 65.3, 60, 59.6, curved=True)

# ===================== Stage 4: Classifier =====================
ax.text(8, 39.5, "STAGE 4\nWeak-supervision\nmodel", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_ACCENT)
box(30, 39.5, 32, 11.2,
    "7 × Gradient-Boosted Regressors\n(ordinal 0–3 constructs)\n"
    "200 trees · depth 3 · lr = 0.05",
    C_MODEL, bold=True, fs=9.5)
box(70, 39.5, 32, 11.2,
    "2 × Gradient-Boosted Classifiers\n(tone, qtype categoricals)\n"
    "5-fold CV · accuracy reported",
    C_MODEL, bold=True, fs=9.5)
arrow(50, 50.4, 38, 45.2, curved=True)
arrow(50, 50.4, 62, 45.2, curved=True)

# CV validation banner
box(50, 28, 78, 6.2,
    "Validation: 5-fold cross-validation on the gold sample (KFold, shuffle, seed = 0)\n"
    "Headline diagnostics saved to analysis_out/classifier_quality.csv  (CV R² per construct; CV accuracy for categoricals)",
    '#F5E9F2', fs=9, italic=True)
arrow(30, 33.9, 40, 31.1, curved=True)
arrow(70, 33.9, 60, 31.1, curved=True)

# ===================== Stage 5: Propagation =====================
ax.text(8, 17.5, "STAGE 5\nPropagate +\nanalysis", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_ACCENT)
box(30, 17.5, 32, 9,
    "Predict on full corpus (3,412 msgs)\n→ full_stance_predictions.csv\nclipped to [0, 3]",
    C_OUTPUT, bold=True, fs=9.4)
arrow(50, 24.9, 36, 22, curved=True)

box(70, 17.5, 32, 9,
    "Per-user aggregation\n→ paired within-subject d_z\n(Persona − GPT)  ·  per persona family",
    C_OUTPUT, bold=True, fs=9.4)
arrow(46, 17.5, 54, 17.5, label="aggregate by user × condition", label_offset=(0,-2))

# Final downstream node (Table 4)
box(50, 5.5, 80, 6.2,
    "Reported in §3.2 — Table 4 (paired Cohen's d_z, BH-FDR adjusted) and Figures 7–8 (paired shifts on expansion/contraction)",
    '#E2D9F3', bold=True, fs=10)
arrow(30, 12.9, 42, 8.7, curved=True)
arrow(70, 12.9, 58, 8.7, curved=True)

# Script labels along the left side (tiny mono)
ax.text(2.2, 84, "embed_messages.py",   rotation=90, ha='center', va='center',
        fontsize=7.5, color='#666', family='monospace')
ax.text(2.2, 70, "claude_stance_labels.csv", rotation=90, ha='center', va='center',
        fontsize=7.5, color='#666', family='monospace')
ax.text(2.2, 55, "propagate_stance.py", rotation=90, ha='center', va='center',
        fontsize=7.5, color='#666', family='monospace')
ax.text(2.2, 39.5, "propagate_stance.py", rotation=90, ha='center', va='center',
        fontsize=7.5, color='#666', family='monospace')
ax.text(2.2, 17.5, "analyze.py",         rotation=90, ha='center', va='center',
        fontsize=7.5, color='#666', family='monospace')

# Outer frame
outer = FancyBboxPatch((1.2, 1.5), 97.6, 92.5,
                       boxstyle="round,pad=0.5,rounding_size=2",
                       linewidth=1.2, edgecolor='#999', facecolor='none')
ax.add_patch(outer)

plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'wrote {OUT}')
import os as _os
print('size:', _os.path.getsize(OUT), 'bytes')
