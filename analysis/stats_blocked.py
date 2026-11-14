"""2x2 ANOVA with seeds as blocks, per-seed contrast tests, Holm correction, paired comparisons and TOST."""
import sys, json, math, statistics as st
sys.path.insert(0,'analysis'); sys.path.insert(0,'.')
from load import *
_x=finals('results/followup/E2b/*/runs/*/result.json'); _y=finals('results/followup/E4/*/runs/*/result.json')
for _k in _x:
    if _k in _y: _x[_k].update(_y[_k])
NEWF.update(_x)
from scipy_free import f_sf
from bench.stats import betainc
def t_p2(t,df): return betainc(df/2,0.5,df/(df+t*t))
def t_ppf(p,df):   # one-sided inverse by bisection
    lo,hi=0,100
    for _ in range(200):
        mid=(lo+hi)/2
        if t_p2(mid,df)/2>1-p: lo=mid
        else: hi=mid
    return (lo+hi)/2
X={'sfadamw':('sfadamw-l2','sfadamw'),'adabelief':('adabelief','adabelief-dw')}
out={}; allp=[]
for x,(xc,xd) in X.items():
  for cell in 'BDE':
    c={('a','c'):get(cell,'adam'),('a','d'):get(cell,'adamw'),('x','c'):get(cell,xc),('x','d'):get(cell,xd)}
    seeds=sorted(set.intersection(*[set(v) for v in c.values()]))[:5]; n=len(seeds)
    Y={k:[v[s] for s in seeds] for k,v in c.items()}
    g=st.mean(v for vs in Y.values() for v in vs)
    m={k:st.mean(v) for k,v in Y.items()}
    ma={a:(m[(a,'c')]+m[(a,'d')])/2 for a in 'ax'}; mb={b:(m[('a',b)]+m[('x',b)])/2 for b in 'cd'}
    ms=[st.mean(Y[k][i] for k in Y) for i in range(n)]
    ssa=2*n*sum((ma[a]-g)**2 for a in 'ax'); ssb=2*n*sum((mb[b]-g)**2 for b in 'cd')
    ssab=n*sum((m[(a,b)]-ma[a]-mb[b]+g)**2 for a in 'ax' for b in 'cd')
    sss=4*sum((v-g)**2 for v in ms)
    sst=sum((v-g)**2 for vs in Y.values() for v in vs)
    sse_old=sst-ssa-ssb-ssab; sse=sse_old-sss; dfe=3*(n-1)
    r={'n':n,'seed_eta2':sss/sst,'seed_F':(sss/(n-1))/(sse/dfe),'seed_p':f_sf((sss/(n-1))/(sse/dfe),n-1,dfe)}
    # per-seed contrasts (repeated measures, separate error term per effect, df = n-1)
    con={'rule':[(Y[('x','c')][i]+Y[('x','d')][i]-Y[('a','c')][i]-Y[('a','d')][i])/2 for i in range(n)],
         'wd':[(Y[('a','d')][i]+Y[('x','d')][i]-Y[('a','c')][i]-Y[('x','c')][i])/2 for i in range(n)],
         'int':[(Y[('x','d')][i]-Y[('x','c')][i])-(Y[('a','d')][i]-Y[('a','c')][i]) for i in range(n)]}
    for name,ss in (('rule',ssa),('wd',ssb),('int',ssab)):
        F=ss/(sse/dfe); F0=ss/(sse_old/(4*(n-1)))
        d=con[name]; t=st.mean(d)/(st.stdev(d)/math.sqrt(n))
        r[name]=dict(eta2=ss/sst,peta2=ss/(ss+sse),F_blk=F,p_blk=f_sf(F,1,dfe),F_old=F0,p_old=f_sf(F0,1,4*(n-1)),
                     est=st.mean(d),t_rm=t,p_rm=t_p2(abs(t),n-1),ci=t_ppf(0.975,n-1)*st.stdev(d)/math.sqrt(n))
        allp.append((f'{cell}-{x}-{name}',r[name]['p_blk'],r[name]['p_rm']))
    out[f'{cell}-{x}']=r
def holm(ps):
    idx=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0]*len(ps); cur=0
    for r_,i in enumerate(idx):
        cur=max(cur,min(1,(len(ps)-r_)*ps[i])); adj[i]=cur
    return adj
hb=holm([p for _,p,_ in allp]); hr=holm([p for _,_,p in allp])
for (k,_,_),a,b in zip(allp,hb,hr):
    cell,x,name=k.split('-'); out[f'{cell}-{x}'][name]['p_blk_holm']=a; out[f'{cell}-{x}'][name]['p_rm_holm']=b
for k,r in out.items():
    print(k,f"seed η²={r['seed_eta2']:.3f} p={r['seed_p']:.2g}")
    for nm in ('rule','wd','int'):
        e=r[nm]; print(f"   {nm:5s} η²={e['eta2']:.3f} est={e['est']:+.2f}±{e['ci']:.2f} | old p={e['p_old']:.2g} | blk F={e['F_blk']:.1f} p={e['p_blk']:.2g} holm={e['p_blk_holm']:.2g} | rm t={e['t_rm']:.2f} p={e['p_rm']:.2g} holm={e['p_rm_holm']:.2g}")
# ---- paired comparisons + TOST
def pair(cell,a,b,margin=None):
    va,vb=get(cell,a),get(cell,b); s=sorted(set(va)&set(vb)); d=[va[i]-vb[i] for i in s]; n=len(d)
    m=st.mean(d); se=st.stdev(d)/math.sqrt(n); t=m/se
    r=dict(n=n,diff=m,se=se,p=t_p2(abs(t),n-1),ci95=t_ppf(0.975,n-1)*se,ci90=t_ppf(0.95,n-1)*se,d=d)
    if margin:
        t1=(m+margin)/se; t2=(margin-m)/se
        r['tost_p']=max(t_p2(abs(t1),n-1)/2 if t1>0 else 1, t_p2(abs(t2),n-1)/2 if t2>0 else 1); r['margin']=margin
    return r
P={}
for cell,a,b,mg in [('B','adam-x0.01','adamw',0.5),('E','adam','adamw',0.5),('B','sfadamw','adamw',None),('D','sfadamw','adamw',None),('E','sfadamw','adamw',None),
                 ('B','adabelief','adamw',0.5),('D','adabelief','adamw',0.5),('E','adabelief-dw','adabelief',None),('B','adabelief-dw','adabelief',None),('D','adabelief-dw','adabelief',None),
                 ('B','sfadamw-l2','sfadamw',None),('C','adamw','adam',0.5),('C','sfadamw','adamw',None),('C','adabelief','adamw',None)]:
    P[f'{cell}|{a}|{b}']=pair(cell,a,b,mg)
hp=holm([v['p'] for v in P.values()])
for (k,v),h in zip(P.items(),hp):
    v['p_holm']=h
    print(k,f"n={v['n']} diff={v['diff']:+.3f} ±{v['ci95']:.3f} (90%: ±{v['ci90']:.3f}) p={v['p']:.3g} holm={h:.3g}", f"TOST(±{v['margin']}) p={v['tost_p']:.3g}" if 'tost_p' in v else '')
json.dump({'anova':out,'pairs':P},open('analysis/out/facts_stats.json','w'),indent=1)
