# -*- coding: utf-8 -*-
"""Figure 5 — concept diagram for §4.3 'How users orchestrate the agents'.

Schematic: the user operates at the META level (allocate / switch / broker /
integrate) over two differentiated agents that perform the OBJECT-level creative
operations; the creative-process gain lands in the dialogue, not in the user.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3/report/figures/fig5_orchestration.png'
DIV='#22A06B'; CON='#7A6CF5'; USER='#33415a'; BAND='#eef1f4'; BANDED='#cfd6de'; INK='#1c2530'; GRY='#6b7280'

fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=2.4',
                                fc=fc, ec=ec, lw=1.4))

# meta / object divider
ax.plot([3, 97], [63, 63], ls=(0, (4, 3)), color=GRY, lw=1)
ax.text(3, 65, 'META LEVEL  (regulation)', fontsize=8, color=GRY, style='italic', ha='left', va='bottom')
ax.text(3, 61, 'OBJECT LEVEL  (cognitive operations)', fontsize=8, color=GRY, style='italic', ha='left', va='top')

# user (meta)
box(32, 78, 36, 15, USER, USER)
ax.text(50, 88, 'USER', color='white', fontsize=13, fontweight='bold', ha='center', va='center')
ax.text(50, 82.5, 'allocate · switch · broker · integrate', color='#dbe2ea', fontsize=9, ha='center', va='center')

# agents (object)
box(9, 40, 33, 16, DIV, DIV)
ax.text(25.5, 51, 'Taylor — divergent', color='white', fontsize=11, fontweight='bold', ha='center')
ax.text(25.5, 45, 'generate · explore · reframe', color='white', fontsize=8.8, ha='center')
box(58, 40, 33, 16, CON, CON)
ax.text(74.5, 51, 'Alex — convergent', color='white', fontsize=11, fontweight='bold', ha='center')
ax.text(74.5, 45, 'evaluate · specify · converge', color='white', fontsize=8.8, ha='center')

# user <-> agents
ar = dict(arrowstyle='<|-|>', color=INK, lw=1.5, mutation_scale=11)
ax.add_patch(FancyArrowPatch((40, 78), (27, 56), connectionstyle='arc3,rad=0.18', **ar))
ax.add_patch(FancyArrowPatch((60, 78), (73, 56), connectionstyle='arc3,rad=-0.18', **ar))
ax.text(28.5, 68, 'invoke /\nreturn', fontsize=7.8, color=INK, ha='center', va='center')
ax.text(71.5, 68, 'invoke /\nreturn', fontsize=7.8, color=INK, ha='center', va='center')

# broker between agents
ax.add_patch(FancyArrowPatch((43, 50), (57, 50), arrowstyle='<|-|>', color=GRY, lw=1.2,
                             ls='--', mutation_scale=9))
ax.text(50, 53.5, 'broker', fontsize=8, color=GRY, ha='center', style='italic')

# dialogue band (object output)
box(9, 17, 82, 12, BAND, BANDED)
ax.text(50, 23, 'DIALOGUE — where reframing, exploration & evaluation happen',
        fontsize=9, color=INK, ha='center', va='center', fontweight='bold')
ad = dict(arrowstyle='-|>', color=GRY, lw=1.3, mutation_scale=10)
ax.add_patch(FancyArrowPatch((25.5, 40), (30, 29), **ad))
ax.add_patch(FancyArrowPatch((74.5, 40), (70, 29), **ad))

# where the gain lands (the two-rubric anchor)
ax.text(50, 11, 'Where the gain lands', fontsize=9, color=INK, ha='center', fontweight='bold')
ax.scatter([20], [6.5], s=46, color=DIV, zorder=5)
ax.text(23, 6.5, 'dialogue-level rubric — transformed (reframing g = +2.04)',
        fontsize=8.3, color=INK, ha='left', va='center')
ax.scatter([20], [2.3], s=46, color=USER, zorder=5)
ax.text(23, 2.3, 'user-only rubric — unchanged (all n.s.)',
        fontsize=8.3, color=INK, ha='left', va='center')

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print('wrote', OUT)
