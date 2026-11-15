# fig_rho: measured relative update ||dW||/||W|| per conv layer at epoch 25 (E6 runs, 2 seeds), cells B and E
import json,glob,collections,statistics as st,sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=sys.argv[1]
R=collections.defaultdict(list)
for f in sorted(glob.glob('results/followup/E6/*/runs/*/result.json')):
    d=json.load(open(f)); R[(d['cell'],d['optimizer'])].append(d)
CFG=[('adam','Adam, L2 $5{\\times}10^{-4}$','#c44e52','-'),
     ('adam-x0.01','Adam, L2 $5{\\times}10^{-6}$','#dd8452','-'),
     ('adamw','AdamW, $5{\\times}10^{-2}$','#4c72b0','-'),
     ('sfadamw-l2','SF-AdamW, L2','#8172b3','--'),
     ('sfadamw','SF-AdamW, dec.','#55a868','--')]
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7})
f,ax=plt.subplots(1,2,figsize=(3.45,2.35))
for a,cell,t in zip(ax,'BE',['B: with BN','E: without BN']):
    for o,lab,col,ls in CFG:
        rs=R[(cell,o)]
        y=[st.mean(r['history'][24]['rel_upd'][l] for r in rs) for l in range(5)]
        a.plot(range(1,6),y,ls,marker='o',ms=2.5,color=col,label=lab,lw=1.2)
        print(cell,o,len(rs),' '.join(f'{v:.4f}' for v in y),'ratio %.1f'%(max(y)/min(y)))
    a.set_yscale('log'); a.set_title(t,fontsize=8); a.set_xlabel('convolution layer'); a.set_xticks(range(1,6))
ax[0].set_ylabel(r'measured $\|\Delta W\|/\|W\|$')
h,l=ax[0].get_legend_handles_labels()
f.legend(h,l,loc='lower center',ncol=2,fontsize=7,frameon=False,handlelength=2.2,columnspacing=0.8)
f.tight_layout(rect=(0,0.24,1,1),w_pad=0.6)
f.savefig(f'{OUT}/fig_rho.pdf'); f.savefig(f'{OUT}/fig_rho.png',dpi=200)
