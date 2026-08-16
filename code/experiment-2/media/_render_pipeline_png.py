# Renders llm_judge_pipeline.png (academic flow diagram) via matplotlib to match the SVG.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 680, 676
fig = plt.figure(figsize=(W/100, H/100), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.invert_yaxis(); ax.axis("off")

NEUTRAL = dict(fc="#f5f5f3", ec="#c7c5be", lw=1.2, tc="#262626")
TEAL    = dict(fc="#e9f4f1", ec="#2a9d8f", lw=1.3, tc="#1f6f64")
BLUE    = dict(fc="#eaeff6", ec="#4a6fa5", lw=1.3, tc="#3a567f")

def box(x, y, w, h, title, sub, style):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=6",
                 fc=style["fc"], ec=style["ec"], lw=style["lw"], mutation_aspect=1))
    ax.text(x+w/2, y+19, title, ha="center", va="center",
            fontsize=9.5, fontweight="medium", color=style["tc"], family="DejaVu Sans")
    ax.text(x+w/2, y+36, sub, ha="center", va="center",
            fontsize=8, color="#5c5c5c", family="DejaVu Sans")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=11, lw=1.5, color="#6b6b6b"))

# legend
ax.add_patch(FancyBboxPatch((40, 44), 14, 14, boxstyle="round,pad=0,rounding_size=3", **{k:TEAL[k] for k in ("fc","ec","lw")}))
ax.text(60, 51, "idea-count lane", fontsize=8, color="#5c5c5c", va="center")
ax.add_patch(FancyBboxPatch((40, 66), 14, 14, boxstyle="round,pad=0,rounding_size=3", **{k:BLUE[k] for k in ("fc","ec","lw")}))
ax.text(60, 73, "judge lane", fontsize=8, color="#5c5c5c", va="center")

box(210, 44, 260, 48, "User-only transcript", "participant turns only, no summary", NEUTRAL)
arrow(340, 92, 340, 114)
box(240, 116, 200, 48, "Windowing", "12k chars, 800 overlap", NEUTRAL)
arrow(322, 164, 190, 204); arrow(358, 164, 490, 204)

box(60, 206, 240, 50, "Idea extraction", "GPT-4.1-mini, T=0", TEAL)
arrow(180, 256, 180, 280)
box(60, 282, 240, 50, "Category induction", "up to 8 categories", TEAL)
arrow(180, 332, 180, 356)
box(60, 358, 240, 50, "Fluency and flexibility", "counts, scaled to 1 to 7", TEAL)

box(380, 206, 240, 50, "Five judge personas", "distinct expert lenses", BLUE)
arrow(500, 256, 500, 280)
box(380, 282, 240, 50, "Eight dimensions, 1 to 7", "with brief rationales", BLUE)
arrow(500, 332, 500, 356)
box(380, 358, 240, 50, "Deterministic scoring", "T=0, JSON only, cached", BLUE)

arrow(180, 408, 300, 450); arrow(500, 408, 380, 450)
box(200, 452, 280, 50, "Ensemble per dimension", "median across five judges", NEUTRAL)
arrow(340, 502, 340, 526)
box(180, 528, 320, 50, "Per-conversation scoreboard", "8 dimensions plus fluency and flexibility", NEUTRAL)
arrow(340, 578, 340, 602)
box(70, 604, 540, 52, "Validity audits", "reliability (ICC), length-bias, pairwise Bradley-Terry alignment", NEUTRAL)

fig.savefig("llm_judge_pipeline.png", dpi=300, transparent=True)
print("saved llm_judge_pipeline.png")
