"""
Patch fig18_partner_footing_coupling.png in place to replace the
legend label 'GPT' with 'Standard LLM'. The full pipeline to regenerate
this figure from raw embeddings requires per-message turn-order metadata
that is not in the analysis_out CSVs, so we surgically edit the PNG.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
PNG  = os.path.join(ROOT, 'figures', 'fig18_partner_footing_coupling.png')

im = Image.open(PNG).convert('RGB')
W, H = im.size
print(f'image size: {W} x {H}')

# Legend area is roughly upper-right. The 'GPT' label sits next to the gray
# square just under 'Persona'. Find a white-background rectangle for the
# legend text. Hardcoded region from inspecting the saved figure.
# Approx coords (in pixel units of saved image ~ 1100x740 at dpi 130):
#   legend box approximately at x: 880..1080, y: 60..160
# 'GPT' text approximately at x: 940..990, y: 78..108
# We'll be a bit generous and overpaint a wider area to be safe.

draw = ImageDraw.Draw(im)

# 1) Whiteout the 'GPT' text near the legend.
# scan for the literal letters by location: use a safe box and re-render.
# Box for "GPT" only (do not cover the color swatch):
gpt_box = (int(W*0.860), int(H*0.110), int(W*0.940), int(H*0.155))
draw.rectangle(gpt_box, fill=(255, 255, 255))

# Find a font that exists.
font = None
for cand in [
    'C:/Windows/Fonts/arial.ttf',
    'C:/Windows/Fonts/calibri.ttf',
    'C:/Windows/Fonts/DejaVuSans.ttf',
]:
    if os.path.exists(cand):
        try:
            font = ImageFont.truetype(cand, size=18)
            break
        except Exception:
            pass
if font is None:
    font = ImageFont.load_default()

draw.text((gpt_box[0] + 4, gpt_box[1] + 4), 'Standard LLM',
          fill=(0, 0, 0), font=font)

im.save(PNG)
print('patched', PNG)
