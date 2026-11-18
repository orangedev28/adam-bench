"""Fig. 1 of Paper 2: validation accuracy over the joint LR x decay grid. Run: python analysis/iccies_fig_heat.py"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'font.family':'serif','mathtext.fontset':'cm','axes.linewidth':0.6})
F=json.load(open('analysis/out/facts_iccies.json'))['robust']
LRS=(1e-4,3e-4,1e-3,3e-3,1e-2,3e-2); MULTS=(0.001,0.01,0.1,1,10,100)
def lab(v):
    m,e=f'{v:.0e}'.split('e'); m=int(float(m)); e=int(e)
    if e==0: return f'${m}$'
    return f'$10^{{{e}}}$' if m==1 else f'${m}{{\\times}}10^{{{e}}}$'
fig,axs=plt.subplots(2,2,figsize=(4.8,3.8),constrained_layout=True)
cmap=plt.get_cmap('Blues')
for i,cell in enumerate('BE'):
    best_row=max(F[f'{cell}|{b}']['best'] for b in ('adam','adamw')); vmin=best_row-10
    for j,(base,l0,name) in enumerate((('adam',5e-4,'Adam (L2)'),('adamw',5e-2,'AdamW (decoupled)'))):
        ax=axs[i,j]; R=F[f'{cell}|{base}']; g=R['grid']; best=R['best']
        A=np.full((6,6),np.nan)
        for a,m in enumerate(MULTS):
            for b,l in enumerate(LRS):
                k=f'{m:g}|{l:g}'
                if k in g: A[a,b]=g[k]
        ax.imshow(np.clip(A,vmin,best_row),cmap=cmap,vmin=vmin,vmax=best_row,origin='lower',aspect='auto')
        for a in range(6):
            for b in range(6):
                v=A[a,b]
                if np.isnan(v):
                    ax.add_patch(Rectangle((b-.5,a-.5),1,1,fc='white',ec='none')); ax.text(b,a,'n/r',ha='center',va='center',fontsize=7.5,color='0.3'); continue
                frac=(min(max(v,vmin),best_row)-vmin)/(best_row-vmin)
                ax.text(b,a,f'{v:.1f}',ha='center',va='center',fontsize=7.5,
                        color='white' if frac>0.6 else 'black',fontweight='bold' if v==best else 'normal')
                if v>=best-0.5: ax.add_patch(Rectangle((b-.5,a-.5),1,1,fill=False,ec='#c0392b',lw=1.6))
        d=MULTS.index(1); ax.add_patch(Rectangle((-.5,d-.5),6,1,fill=False,ec='0.25',lw=0.8,ls='--'))
        ax.set_xticks(range(6)); ax.set_xticklabels([lab(l) for l in LRS],rotation=45,fontsize=8)
        ax.set_yticks(range(6)); ax.set_yticklabels([lab(l0*m) for m in MULTS],fontsize=8)
        ax.set_title(f'{cell}, {name}\n$R_{{0.5}}={R["n_near"]}/{R["N"]}$',fontsize=9)
        if i==1: ax.set_xlabel('learning rate $\\eta$')
        if j==0: ax.set_ylabel('decay coefficient $\\lambda$')
fig.savefig('analysis/out/fig_heat.pdf'); fig.savefig('analysis/out/fig_heat.png',dpi=200)
