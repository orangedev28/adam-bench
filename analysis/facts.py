import sys,json,math,statistics as st,collections,glob
sys.path.insert(0,'analysis'); sys.path.insert(0,'.')
from load import *
_x=finals('results/followup/E2b/*/runs/*/result.json'); _y=finals('results/followup/E4/*/runs/*/result.json')
for _k in _x:
    if _k in _y: _x[_k].update(_y[_k])   # E4: adds seeds 3, 4 for adam-x0.01
NEWF.update(_x)
from bench.stats import paired_t
from anova import anova2
out={'corners':{},'anova':{},'sweep':{},'tests':{},'imb':{}}
X={'adabelief':('adabelief','adabelief-dw'),'sfadamw':('sfadamw-l2','sfadamw')}
for cell in 'BDE':
    for x,(xc,xd) in X.items():
        corners={('adam','coupled'):get(cell,'adam'),('adam','decoupled'):get(cell,'adamw'),(x,'coupled'):get(cell,xc),(x,'decoupled'):get(cell,xd)}
        data={k:[v[s] for s in sorted(v)][:5] for k,v in corners.items()}
        res,mc=anova2(data)
        out['corners'][f'{cell}-{x}']={f'{a}|{b}':[st.mean(v),st.stdev(v)] for (a,b),v in data.items()}
        out['anova'][f'{cell}-{x}']={k:dict(F=F,p=p,eta2=e) for k,(F,p,e) in res.items()}
lam={'adam':5e-4,'adamw':5e-2}
for cell in 'BE':
    for base,mults in (('adam',[0.001,0.01,0.1,1,10,100]),('adamw',[0.01,0.1,1,10])):
        for m in mults:
            o=base if m==1 else f'{base}-x{m:g}'
            v=get(cell,o)
            tp=f'results/followup/E2b/{cell}/tune/{o}.json'
            import os
            if not os.path.exists(tp): tp=f'results/followup/E2/{cell}/tune/{o}.json'
            lr=json.load(open(tp))['best_lr'] if os.path.exists(tp) else json.load(open(f'results/benchmark/{cell}/best_lr.json'))['best_lr'][o]
            out['sweep'][f'{cell}|{base}|{lam[base]*m:g}']=dict(mean=st.mean(v.values()),sd=st.stdev(v.values()),n=len(v),lr=lr,seeds={str(k):x for k,x in v.items()})
def pt(cell,a,b):
    va,vb=get(cell,a),get(cell,b); common=sorted(set(va)&set(vb))
    d=[va[s]-vb[s] for s in common]; m=st.mean(d); sd=st.stdev(d); t=m/(sd/math.sqrt(len(d))) if sd>0 else float('inf')
    from bench.stats import betainc
    df=len(d)-1; p=betainc(df/2,0.5,df/(df+t*t)) if math.isfinite(t) else 0.0
    return dict(diff=m,t=t,p=p,n=len(d))
for cell,a,b in [('B','adam-x0.01','adamw'),('B','adam-x0.01','adam'),('B','sfadamw','adamw'),('D','sfadamw','adamw'),('E','sfadamw','adamw'),('B','adabelief','adamw'),('D','adabelief','adamw'),('B','sfadamw-l2','adam'),('D','sfadamw-l2','adam'),('E','sfadamw-l2','adam'),('E','adam','adamw'),('B','adam-x0.1','adamw')]:
    out['tests'][f'{cell}|{a}|{b}']=pt(cell,a,b)
C=[1728,36864,73728,147456,294912]
R=collections.defaultdict(list)
for f in glob.glob('results/followup/E*/*/runs/*_final_*/result.json'):  # includes E4
    d=json.load(open(f))
    if 'w_norms' in d['history'][0]: R[(d['cell'],d['optimizer'])].append(d)
for (cell,o),rs in R.items():
    vals=[]
    for ep in (10,25,40):
        per=[st.mean(r['history'][ep-1]['lr_end']*math.sqrt(c)/r['history'][ep-1]['w_norms'][i] for r in rs) for i,c in enumerate(C)]
        vals.append(max(per[1:])/min(per[1:]))
    out['imb'][f'{cell}|{o}']=dict(e10=vals[0],e25=vals[1],e40=vals[2],mean=st.mean(vals),norm25=st.mean(st.mean(r['history'][24]['w_norms'][1:]) for r in rs),acc=st.mean(r['final']['test_acc'] for r in rs)*100,n=len(rs))
json.dump(out,open('analysis/out/facts.json','w'),indent=1)
for k,v in out['tests'].items(): print(k, {a:round(b,4) if isinstance(b,float) else b for a,b in v.items()})

# Layer imbalance I (Eq. 4 of the paper) over all 5 conv layers, averaged over epochs 10/25/40
imb5={}
for (cell,o),rs in R.items():
    v=[]
    for ep in (10,25,40):
        per=[st.mean(r['history'][ep-1]['lr_end']*math.sqrt(c)/r['history'][ep-1]['w_norms'][i] for r in rs) for i,c in enumerate(C)]
        v.append(max(per)/min(per))
    imb5[f'{cell}|{o}']=dict(I=st.mean(v),n=len(rs))
out['imb5']=imb5
json.dump(out,open('analysis/out/facts.json','w'),indent=1)
