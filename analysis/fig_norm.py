# Draws Fig. 2 of the paper (mean conv weight norm per epoch, cells B and E), from all follow-up runs
# that recorded weight norms (one run per seed; reruns are bit-identical). Usage: python analysis/fig_norm.py analysis/out
import json,glob,collections,statistics as st,math,sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=sys.argv[1]
C=[1728,36864,73728,147456,294912]
R=collections.defaultdict(dict)
for f in glob.glob('results/followup/E*/*/runs/*/result.json'):
    d=json.load(open(f))
    if d['phase']!='final' or 'w_norms' not in d['history'][0]: continue
    R[(d['cell'],d['optimizer'])][d['seed']]=d
CFG=[('adam','Adam, L2 $5{\\times}10^{-4}$','#c44e52','-'),
     ('adam-x0.01','Adam, L2 $5{\\times}10^{-6}$','#dd8452','-'),
     ('adamw','AdamW, $5{\\times}10^{-2}$','#4c72b0','-'),
     ('sfadamw-l2','SF-AdamW, L2','#8172b3','--'),
     ('sfadamw','SF-AdamW, dec.','#55a868','--')]
for cell in 'BE':
    for o,*_ in CFG: print(cell,o,'seeds',sorted(R[(cell,o)]),'lr',R[(cell,o)][0]['lr'],'wd',R[(cell,o)][0]['weight_decay'])
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7})
from matplotlib.ticker import FixedLocator,ScalarFormatter,NullFormatter
def fig(kind):
    f,ax=plt.subplots(1,2,figsize=(3.45,2.35))
    for a,cell,t in zip(ax,'BE',['B: with BN','E: without BN']):
        for o,lab,col,ls in CFG:
            rs=list(R[(cell,o)].values())
            if kind=='norm':
                E=len(rs[0]['history'])
                y=[st.mean(st.mean(r['history'][e]['w_norms']) for r in rs) for e in range(E)]
                mk={'adam':'o','adam-x0.01':'^','adamw':'s','sfadamw-l2':'D','sfadamw':'v'}[o]   # markers for black-and-white print
                a.plot(range(1,E+1),y,ls,color=col,label=lab,lw=1.1,marker=mk,ms=2.6,markevery=(4,8))
            else:
                y=[st.mean(r['history'][24]['lr_end']*math.sqrt(C[l])/r['history'][24]['w_norms'][l] for r in rs) for l in range(5)]
                a.plot(range(1,6),y,ls,marker='o',ms=2.5,color=col,label=lab,lw=1.2)
        a.set_yscale('log'); a.set_title(t,fontsize=8)
        if kind=='norm':
            a.yaxis.set_major_locator(FixedLocator([10,20,30,50,100] if cell=='B' else [12,15,20,30,50])); a.yaxis.set_major_formatter(ScalarFormatter()); a.yaxis.set_minor_formatter(NullFormatter()); a.grid(alpha=.25,lw=.5)
        a.set_xlabel('epoch' if kind=='norm' else 'convolution layer')
        if kind=='rho': a.set_xticks(range(1,6))
    ax[0].set_ylabel('mean weight norm' if kind=='norm' else r'$\rho_\ell$ at epoch 25')
    h,l=ax[0].get_legend_handles_labels()
    f.legend(h,l,loc='lower center',ncol=2,fontsize=7,frameon=False,handlelength=2.2,columnspacing=0.8)
    f.tight_layout(rect=(0,0.24,1,1),w_pad=0.6)
    f.savefig(f'{OUT}/fig_{kind}.pdf'); f.savefig(f'{OUT}/fig_{kind}.png',dpi=200)
fig('norm')
# print data behind figures for the text
for cell in 'BE':
    for o,lab,*_ in CFG:
        rs=list(R[(cell,o)].values())
        n=[st.mean(st.mean(r['history'][e]['w_norms']) for r in rs) for e in (0,9,24,49)]
        rho=[st.mean(r['history'][24]['lr_end']*math.sqrt(C[l])/r['history'][24]['w_norms'][l] for r in rs) for l in range(5)]
        print(cell,o,'n=',len(rs),'norm e1,10,25,50:',' '.join(f'{x:.1f}' for x in n),'| rho25:',' '.join(f'{x:.4f}' for x in rho),'| max/min %.1f'%(max(rho)/min(rho)))
