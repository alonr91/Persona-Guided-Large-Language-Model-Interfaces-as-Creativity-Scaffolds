# Renders product_layer_pipeline.png/.svg: idea extraction + originality flow (academic style).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 680, 616

NEUTRAL = dict(fc="#f5f5f3", ec="#c7c5be", lw=1.2, tc="#262626")
TEAL    = dict(fc="#e9f4f1", ec="#2a9d8f", lw=1.3, tc="#1f6f64")
BLUE    = dict(fc="#eaeff6", ec="#4a6fa5", lw=1.3, tc="#3a567f")
DASH    = dict(fc="#fbfbfa", ec="#b7b5ae", lw=1.1, tc="#3d3d3d")

def make_ax():
    fig = plt.figure(figsize=(W/100, H/100), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis("off")
    return fig, ax

def box(ax, x, y, w, h, title, sub, style, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                       fc=style["fc"], ec=style["ec"], lw=style["lw"], mutation_aspect=1)
    if dashed: p.set_linestyle((0, (4, 3)))
    ax.add_patch(p)
    ax.text(x+w/2, y+18, title, ha="center", va="center", fontsize=9.3,
            fontweight="medium", color=style["tc"], family="DejaVu Sans")
    ax.text(x+w/2, y+34, sub, ha="center", va="center", fontsize=7.8,
            color="#5c5c5c", family="DejaVu Sans")

def arrow(ax, x1, y1, x2, y2, dashed=False):
    ls = (0, (4, 3)) if dashed else "solid"
    style = "-" if dashed else "-|>"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, lw=1.4, color="#6b6b6b", linestyle=ls))

def draw(ax):
    # legend
    ax.add_patch(FancyBboxPatch((40, 40), 13, 13, boxstyle="round,pad=0,rounding_size=3", **{k:TEAL[k] for k in ("fc","ec","lw")}))
    ax.text(58, 46.5, "idea counts", fontsize=7.8, color="#5c5c5c", va="center")
    ax.add_patch(FancyBboxPatch((150, 40), 13, 13, boxstyle="round,pad=0,rounding_size=3", **{k:BLUE[k] for k in ("fc","ec","lw")}))
    ax.text(168, 46.5, "embedding / originality", fontsize=7.8, color="#5c5c5c", va="center")

    # spine
    box(ax, 240, 66, 200, 46, "User-authored text only", "model text excluded", NEUTRAL)
    arrow(ax, 340, 112, 340, 132)
    box(ax, 250, 132, 180, 46, "Windowing", "12k chars, 800 overlap", NEUTRAL)
    arrow(ax, 340, 178, 340, 198)
    box(ax, 235, 198, 210, 50, "Stage 1: idea extraction", "GPT-4.1, T=0, seed 7", NEUTRAL)
    arrow(ax, 340, 248, 340, 274)
    box(ax, 250, 274, 180, 46, "Union + de-duplication", "distinct ideas", NEUTRAL)
    arrow(ax, 340, 320, 340, 348)
    box(ax, 250, 348, 180, 46, "Embed each idea", "text-embedding-3-large", BLUE)
    arrow(ax, 340, 394, 340, 420)
    box(ax, 235, 420, 210, 46, "L2-normalise + mean-pool", "participant centroid", BLUE)
    arrow(ax, 340, 466, 340, 492)

    # originality output (multi-line)
    ax.add_patch(FancyBboxPatch((180, 492), 320, 96, boxstyle="round,pad=0,rounding_size=6",
                 fc=BLUE["fc"], ec=BLUE["ec"], lw=BLUE["lw"]))
    ax.text(340, 510, "Between-user originality", ha="center", fontsize=9.3, fontweight="medium", color=BLUE["tc"], family="DejaVu Sans")
    ax.text(340, 526, "mean cosine distance to other centroids", ha="center", fontsize=7.8, color="#5c5c5c", family="DejaVu Sans")
    for i, t in enumerate(["1.  to same-condition peers", "2.  to all participants", "3.  to nearest cross-condition neighbour"]):
        ax.text(206, 545+i*15, t, ha="left", fontsize=7.8, color="#3a567f", family="DejaVu Sans")

    # right teal branch: fluency + categories
    arrow(ax, 430, 290, 470, 290)
    box(ax, 470, 267, 170, 46, "Fluency", "count of distinct ideas", TEAL)
    arrow(ax, 430, 300, 470, 350)
    box(ax, 470, 333, 170, 46, "Stage 2: categories", "8 per participant", TEAL)

    # left annotation (dashed): idea criterion
    box(ax, 40, 198, 185, 50, "Idea criterion (κ = 0.81)", "concept, target, affordance", DASH, dashed=True)
    arrow(ax, 225, 223, 235, 223, dashed=True)

    # left blue branch: within-participant diversity (from embed)
    arrow(ax, 250, 371, 225, 371)
    box(ax, 40, 348, 185, 50, "Within-participant diversity", "mean pairwise, own ideas", BLUE)

for ext in ("png", "svg"):
    fig, ax = make_ax(); draw(ax)
    fig.savefig(f"product_layer_pipeline.{ext}", dpi=300, transparent=True)
    plt.close(fig)
print("saved product_layer_pipeline.png / .svg")
