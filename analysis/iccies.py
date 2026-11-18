"""Paper 2 (ICCIES 2027): 2x2 factorial contrasts for cells B-G, Delta_BN (Welch), and the joint LR x decay grid (R_0.5).
Run from the repository root:  python analysis/iccies.py
Output: analysis/out/facts_iccies.json"""
import sys, json, math, glob, re, statistics as st
sys.path.insert(0,'analysis'); sys.path.insert(0,'.')
import load
from load import finals, acc
from scipy_free import f_sf
from bench.stats import betainc
V5=finals('results/followup/E[79]/*/runs/*/result.json')   # cells F (E7) and G (E9)
def get(cell,opt):
    if cell in 'FG': return acc(V5[(cell,opt)])
    return load.get(cell,opt)
def t_p2(t,df): return betainc(df/2,0.5,df/(df+t*t))
def t_ppf(p,df):
    lo,hi=0,100
    for _ in range(200):
        mid=(lo+hi)/2
        if t_p2(mid,df)/2>1-p: lo=mid
        else: hi=mid
    return (lo+hi)/2
def holm(ps):
    idx=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0]*len(ps); cur=0
    for r_,i in enumerate(idx):
        cur=max(cur,min(1,(len(ps)-r_)*ps[i])); adj[i]=cur
    return adj
X={'sfadamw':('sfadamw-l2','sfadamw'),'adabelief':('adabelief','adabelief-dw')}
CELLS='BCDEFG'
out={}; allp=[]; CON={}
for x,(xc,xd) in X.items():
  for cell in CELLS:
    c={('a','c'):get(cell,'adam'),('a','d'):get(cell,'adamw'),('x','c'):get(cell,xc),('x','d'):get(cell,xd)}
    seeds=sorted(set.intersection(*[set(v) for v in c.values()]))[:5]; n=len(seeds)
    Y={k:[v[s] for s in seeds] for k,v in c.items()}
    g=st.mean(v for vs in Y.values() for v in vs); m={k:st.mean(v) for k,v in Y.items()}
    ma={a:(m[(a,'c')]+m[(a,'d')])/2 for a in 'ax'}; mb={b:(m[('a',b)]+m[('x',b)])/2 for b in 'cd'}
    ms=[st.mean(Y[k][i] for k in Y) for i in range(n)]
    ssa=2*n*sum((ma[a]-g)**2 for a in 'ax'); ssb=2*n*sum((mb[b]-g)**2 for b in 'cd')
    ssab=n*sum((m[(a,b)]-ma[a]-mb[b]+g)**2 for a in 'ax' for b in 'cd')
    sss=4*sum((v-g)**2 for v in ms); sst=sum((v-g)**2 for vs in Y.values() for v in vs)
    sse=sst-ssa-ssb-ssab-sss; dfe=3*(n-1)
    r={'n':n,'seed_eta2':sss/sst,'means':{f'{a}{b}':m[(a,b)] for a,b in m},
       'sds':{f'{a}{b}':st.stdev(Y[(a,b)]) for a,b in Y}}
    con={'rule':[(Y[('x','c')][i]+Y[('x','d')][i]-Y[('a','c')][i]-Y[('a','d')][i])/2 for i in range(n)],
         'wd':[(Y[('a','d')][i]+Y[('x','d')][i]-Y[('a','c')][i]-Y[('x','c')][i])/2 for i in range(n)],
         'int':[(Y[('x','d')][i]-Y[('x','c')][i])-(Y[('a','d')][i]-Y[('a','c')][i]) for i in range(n)]}
    CON[(cell,x)]=con
    for name,ss in (('rule',ssa),('wd',ssb),('int',ssab)):
        F=ss/(sse/dfe); d=con[name]; t=st.mean(d)/(st.stdev(d)/math.sqrt(n))
        r[name]=dict(eta2=ss/sst,F_blk=F,p_blk=f_sf(F,1,dfe),est=st.mean(d),t_rm=t,p_rm=t_p2(abs(t),n-1),
                     ci=t_ppf(0.975,n-1)*st.stdev(d)/math.sqrt(n))
        allp.append((cell,x,name,r[name]['p_blk'],r[name]['p_rm']))
    out[f'{cell}-{x}']=r
hb=holm([a[3] for a in allp]); hr=holm([a[4] for a in allp])
for (cell,x,name,_,_),a,b in zip(allp,hb,hr):
    out[f'{cell}-{x}'][name]['p_blk_holm']=a; out[f'{cell}-{x}'][name]['p_rm_holm']=b
# ---- Delta_BN: Welch on per-seed interaction contrasts
def welch(a,b):
    ma,mb=st.mean(a),st.mean(b); va,vb=st.variance(a)/len(a),st.variance(b)/len(b); se=math.sqrt(va+vb)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)); t=(ma-mb)/se
    return dict(diff=ma-mb,se=se,df=df,t=t,p=t_p2(abs(t),df),ci=t_ppf(0.975,df)*se)
DB={}
for x in X:
    for bn,nb,arch in (('B','E','5-layer'),('C','F','ResNet-18')):
        w=welch(CON[(bn,x)]['int'],CON[(nb,x)]['int']); w.update(arch=arch,bn=bn,nb=nb,x=x,
            int_bn=st.mean(CON[(bn,x)]['int']),int_nb=st.mean(CON[(nb,x)]['int'])); DB[f'{bn}-{nb}-{x}']=w
hd=holm([v['p'] for v in DB.values()])
for v,h in zip(DB.values(),hd): v['p_holm']=h
# ---- E8 grid
LRS=(1e-4,3e-4,1e-3,3e-3,1e-2,3e-2); MULTS=(0.001,0.01,0.1,1,10,100)
grid={}
for f in glob.glob('results/benchmark/*/runs/*/result.json')+glob.glob('results/followup/*/*/runs/*/result.json'):
    d=json.load(open(f))
    if d['phase']!='sweep' or d['cell'] not in 'BE' or d['epochs']!=50: continue
    o=d['optimizer']; mm=re.fullmatch(r'(adamw?)(?:-x([0-9.]+))?',o)
    if not mm: continue
    base=mm.group(1); mult=float(mm.group(2) or 1)
    if mult not in MULTS or not any(abs(d['lr']-l)<1e-12 for l in LRS): continue
    v=0.0 if d['diverged'] else d['final']['val_acc']*100
    grid[(d['cell'],base,mult,d['lr'])]=v
# E, AdamW, lambda=5, LR 3e-2 was not run (Kaggle time limit); it is left out of the grid
R={}
for cell in 'BE':
    for base,l0 in (('adam',5e-4),('adamw',5e-2)):
        G={(m,l):grid[(cell,base,float(m),l)] for m in MULTS for l in LRS if (cell,base,float(m),l) in grid}
        best=max(G.values()); bk=max(G,key=G.get)
        near=[k for k,v in G.items() if v>=best-0.5]
        R[f'{cell}|{base}']=dict(best=best,best_lr=bk[1],best_lam=l0*bk[0],R05=len(near)/len(G),n_near=len(near),N=len(G),
            R1=sum(v>=best-1 for v in G.values())/len(G),
            grid={f'{m:g}|{l:g}':G[(m,l)] for m,l in G})
json.dump({'anova':out,'dbn':DB,'robust':R},open('analysis/out/facts_iccies.json','w'),indent=1)
if __name__=='__main__':
    for k,r in out.items():
        mm=r['means']; print(f"{k:13s} A {mm['ac']:.2f} W {mm['ad']:.2f} XL2 {mm['xc']:.2f} XD {mm['xd']:.2f} | "+
              ' | '.join(f"{nm} {r[nm]['est']:+.2f}±{r[nm]['ci']:.2f} h={r[nm]['p_rm_holm']:.2g} η²={r[nm]['eta2']:.2f}" for nm in ('rule','wd','int')))
    for k,v in DB.items(): print('ΔBN',k,f"int_bn {v['int_bn']:+.2f} int_nb {v['int_nb']:+.2f} Δ {v['diff']:+.2f}±{v['ci']:.2f} df {v['df']:.1f} p {v['p']:.3g} holm {v['p_holm']:.3g}")
    for k,v in R.items(): print('R',k,f"best {v['best']:.2f} at lr {v['best_lr']:g} λ {v['best_lam']:g}  R0.5={v['R05']:.3f} ({v['n_near']}/{v['N']}) R1={v['R1']:.3f}")
