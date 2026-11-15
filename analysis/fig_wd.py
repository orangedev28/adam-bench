# Draws Fig. 1 of the paper (accuracy versus weight-decay coefficient). Run after facts.py.
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','pdf.fonttype':42})
F=json.load(open('analysis/out/facts.json'))['sweep']
fig,axs=plt.subplots(1,2,figsize=(3.5,1.9)); hs=[]
for ax,cell,tt in ((axs[0],'B','B: with BN'),(axs[1],'E','E: without BN')):
    for base,col,m,lab in (('adam','#c44e52','o','Adam (coupled L2)'),('adamw','#4c72b0','s','AdamW (decoupled)')):
        pts=sorted((float(k.split('|')[2]),v) for k,v in F.items() if k.startswith(f'{cell}|{base}|'))
        h=ax.errorbar([p[0] for p in pts],[p[1]['mean'] for p in pts],yerr=[p[1]['sd'] for p in pts],color=col,marker=m,ms=3,lw=1.1,capsize=1.5,label=lab)
        if cell=='B': hs.append(h)
    ax.set_xscale('log'); ax.set_title(tt,fontsize=8); ax.set_xlabel('weight decay $\\lambda$'); ax.set_xticks([1e-6,1e-4,1e-2,1])
    if cell=='B': ax.set_ylim(85,91); ax.set_ylabel('test accuracy (%)')
    else: ax.set_ylim(30,90)
    ax.grid(alpha=0.25,lw=0.4)
fig.legend(handles=hs,loc='lower center',ncol=2,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(pad=0.3,w_pad=0.6,rect=(0,0.07,1,1))
fig.savefig('analysis/out/fig_wd.pdf',bbox_inches='tight',pad_inches=0.02)
print('ok analysis/out/fig_wd.pdf')
