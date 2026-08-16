# -*- coding: utf-8 -*-
"""Figure B1 — how the zero-shot stance classifier scores one turn."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3/report/figures/figB1_stance.png'
DIV='#22A06B'; CON='#7A6CF5'; NLI='#33415a'; LIGHT='#eef1f4'; ED='#cfd6de'; INK='#1c2530'; GRY='#6b7280'
DIVL='#d8f0e4'; CONL='#e6e1fb'

fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
def box(x,y,w,h,fc,ec,lw=1.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=2.2',fc=fc,ec=ec,lw=lw))

# turn
box(35,90,30,8,LIGHT,ED)
ax.text(50,95.4,'Turn $t$',ha='center',va='center',fontsize=12,fontweight='bold',color=INK)
ax.text(50,91.8,'one message',ha='center',va='center',fontsize=8.5,color=GRY)
ax.add_patch(FancyArrowPatch((50,90),(50,84),arrowstyle='-|>',color=GRY,lw=1.5,mutation_scale=12))

# NLI model
box(12,72,76,12,NLI,NLI)
ax.text(50,80.6,'Zero-shot NLI classifier  (DeBERTa-v3-base-zeroshot)',ha='center',va='center',
        fontsize=10.5,fontweight='bold',color='white')
ax.text(50,76.4,'score the turn against 10 stance hypotheses, each scored independently (multi-label)',
        ha='center',va='center',fontsize=8.7,color='#dbe2ea')
ax.text(50,73.4,'each hypothesis $h$ gives $e(t,h)=P(\\,t \\models h\\,)\\in[0,1]$',
        ha='center',va='center',fontsize=8.7,color='#dbe2ea')
ax.add_patch(FancyArrowPatch((50,72),(30,63),connectionstyle='arc3,rad=0.10',arrowstyle='-|>',color=GRY,lw=1.4,mutation_scale=11))
ax.add_patch(FancyArrowPatch((50,72),(70,63),connectionstyle='arc3,rad=-0.10',arrowstyle='-|>',color=GRY,lw=1.4,mutation_scale=11))

# divergent markers
box(7,40,41,23,DIVL,DIV)
ax.text(27.5,59.5,'5 DIVERGENT markers',ha='center',va='center',fontsize=10,fontweight='bold',color=DIV)
for k,lab in enumerate(['broaden / add alternatives','open “what if” question','reframe the problem',
                        'analogy or metaphor','keep options open']):
    ax.text(10,55.3-k*3.1,'•  '+lab,ha='left',va='center',fontsize=8.3,color=INK)
# convergent markers
box(52,40,41,23,CONL,CON)
ax.text(72.5,59.5,'5 CONVERGENT markers',ha='center',va='center',fontsize=10,fontweight='bold',color=CON)
for k,lab in enumerate(['explicit criteria / constraints','compare & recommend one','rank / prioritise',
                        'stepwise planning','critique weaknesses']):
    ax.text(55,55.3-k*3.1,'•  '+lab,ha='left',va='center',fontsize=8.3,color=INK)

# aggregation equations
ax.text(27.5,35.0,r'$D(t)=\frac{1}{5}\sum_{i=1}^{5} e(t,h_i^{\mathrm{div}})$',ha='center',va='center',fontsize=12,color=DIV)
ax.text(72.5,35.0,r'$C(t)=\frac{1}{5}\sum_{i=1}^{5} e(t,h_i^{\mathrm{con}})$',ha='center',va='center',fontsize=12,color=CON)
ax.add_patch(FancyArrowPatch((27.5,31.5),(42,25),arrowstyle='-|>',color=GRY,lw=1.4,mutation_scale=11))
ax.add_patch(FancyArrowPatch((72.5,31.5),(58,25),arrowstyle='-|>',color=GRY,lw=1.4,mutation_scale=11))

# balance
box(30,18,40,7.5,LIGHT,INK,lw=1.8)
ax.text(50,21.8,r'balance  $b(t)=D(t)-C(t)\in[-1,\,+1]$',ha='center',va='center',fontsize=11.5,
        fontweight='bold',color=INK)

# scale bar
ax.add_patch(Rectangle((22,9),28,3.2,fc=CONL,ec=CON,lw=1))
ax.add_patch(Rectangle((50,9),28,3.2,fc=DIVL,ec=DIV,lw=1))
ax.text(22,7,'−1',ha='center',va='top',fontsize=8.5,color=CON,fontweight='bold')
ax.text(50,7,'0',ha='center',va='top',fontsize=8.5,color=GRY)
ax.text(78,7,'+1',ha='center',va='top',fontsize=8.5,color=DIV,fontweight='bold')
ax.text(36,10.6,'more convergent',ha='center',va='center',fontsize=8,color=CON)
ax.text(64,10.6,'more divergent',ha='center',va='center',fontsize=8,color=DIV)
ax.text(50,2.5,'per-turn balance calibrated against hand ratings:  Pearson $r=0.715$',
        ha='center',va='center',fontsize=8.5,color=INK,style='italic')

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig(OUT,dpi=200,bbox_inches='tight')
print('wrote',OUT)
