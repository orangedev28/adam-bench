import sys; sys.path.insert(0,'analysis')
from load import *
from itertools import product
def anova2(data):
    # data[(a,b)] = list of values; balanced
    A=sorted({k[0] for k in data}); B=sorted({k[1] for k in data})
    n=len(next(iter(data.values()))); allv=[v for vs in data.values() for v in vs]; g=st.mean(allv)
    ma={a:st.mean([v for b in B for v in data[(a,b)]]) for a in A}
    mb={b:st.mean([v for a in A for v in data[(a,b)]]) for b in B}
    mc={k:st.mean(v) for k,v in data.items()}
    ssa=len(B)*n*sum((ma[a]-g)**2 for a in A); ssb=len(A)*n*sum((mb[b]-g)**2 for b in B)
    ssab=n*sum((mc[(a,b)]-ma[a]-mb[b]+g)**2 for a in A for b in B)
    sse=sum((v-mc[k])**2 for k,vs in data.items() for v in vs)
    dfe=len(A)*len(B)*(n-1); mse=sse/dfe
    sst=ssa+ssb+ssab+sse
    from scipy_free import f_sf
    res={}
    for name,ss,df in (('rule',ssa,len(A)-1),('wd',ssb,len(B)-1),('rule×wd',ssab,(len(A)-1)*(len(B)-1))):
        F=(ss/df)/mse; res[name]=(F,f_sf(F,df,dfe),ss/sst)
    return res,mc
X={'adabelief':('adabelief','adabelief-dw'),'sfadamw':('sfadamw-l2','sfadamw')}  # (coupled, decoupled)
if __name__=="__main__":
  pass
for cell in ([] if __name__!="__main__" else "BDE"):
    for x,(xc,xd) in X.items():
        corners={('adam','coupled'):get(cell,'adam'),('adam','decoupled'):get(cell,'adamw'),
                 (x,'coupled'):get(cell,xc),(x,'decoupled'):get(cell,xd)}
        data={k:[v[s] for s in sorted(v)][:5] for k,v in corners.items()}
        res,mc=anova2(data)
        a0=mc[('adam','coupled')]
        print(f"{cell} {x:9s} Adam {a0:.2f} | AdamW {mc[('adam','decoupled')]:.2f} | {x}-L2 {mc[(x,'coupled')]:.2f} | {x}-DW {mc[(x,'decoupled')]:.2f}",
              ' | '.join(f"{k}: F={F:.1f} p={p:.2g} η²={e:.2f}" for k,(F,p,e) in res.items()))
