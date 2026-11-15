"""Measured relative update size (E6, E6c) vs the proxy rho = eta*sqrt(C)/||W||.
rel_upd = mean over the steps of an epoch of ||W_{t+1}-W_t||/||W_t|| (measured)
ang_upd = mean angle between W_t and W_{t+1} (measured, radians)
rho_step = mean over the steps of eta_t*sqrt(C)/||W_t|| (proxy, per step)"""
import json,glob,math,statistics as st,collections
R=collections.defaultdict(list)
for f in sorted(glob.glob('results/followup/E6*/*/runs/*/result.json')):
    d=json.load(open(f)); R[(d['cell'],d['optimizer'])].append(d)
C5=[1728,36864,73728,147456,294912]
def resnet_C():
    C=[3*64*9]; cin=64
    for cout,stride in ((64,1),(128,2),(256,2),(512,2)):
        for b in range(2):
            s=stride if b==0 else 1
            C+=[cin*cout*9,cout*cout*9]
            if s!=1 or cin!=cout: C.append(cin*cout)
            cin=cout
    return C
CR=resnet_C()
out={}
EPS={'B':(10,25,40),'E':(10,25,40),'C':(20,50,80)}
for (cell,o),rs in sorted(R.items()):
    C=C5 if cell in 'BE' else CR
    L=len(rs[0]['history'][0]['w_norms']); assert L==len(C),(cell,L,len(C))
    res={'n':len(rs)}
    for key in ('rel_upd','ang_upd','rho_step'):
        Is=[]
        for ep in EPS[cell]:
            per=[st.mean(r['history'][ep-1][key][i] for r in rs) for i in range(L)]
            Is.append(max(per)/min(per))
        res['I_'+key]=st.mean(Is); res['I_'+key+'_eps']=Is
    Is=[]
    for ep in EPS[cell]:
        per=[st.mean(r['history'][ep-1]['lr_end']*math.sqrt(c)/r['history'][ep-1]['w_norms'][i] for r in rs) for i,c in enumerate(C)]
        Is.append(max(per)/min(per))
    res['I_proxy_end']=st.mean(Is)
    # ratio measured/proxy per layer at mid-training
    ep=EPS[cell][1]
    res['ratio_mid']=[st.mean(r['history'][ep-1]['rel_upd'][i]/r['history'][ep-1]['rho_step'][i] for r in rs) for i in range(L)]
    res['rel_mid']=[st.mean(r['history'][ep-1]['rel_upd'][i] for r in rs) for i in range(L)]
    res['ang_vs_rel_maxdev']=max(abs(r['history'][e]['ang_upd'][i]/r['history'][e]['rel_upd'][i]-1) for r in rs for e in range(len(r['history'])-1) for i in range(L))
    res['acc']=st.mean(r['final']['test_acc'] for r in rs)*100; res['lr']=rs[0]['lr']; res['wd']=rs[0]['weight_decay']
    out[f'{cell}|{o}']=res
    print(f"{cell} {o:12s} n={len(rs)} I_meas={res['I_rel_upd']:.2f} I_ang={res['I_ang_upd']:.2f} I_proxy_step={res['I_rho_step']:.2f} I_proxy_end={res['I_proxy_end']:.2f} | ratio mid min..max {min(res['ratio_mid']):.3f}..{max(res['ratio_mid']):.3f} | acc {res['acc']:.2f}")
# Kosson equilibrium angular update for AdamW/SF-AdamW (decoupled): sqrt(2*eta*lam*(1-b1)/(1+b1)), b1=0.9
for cell in 'BEC':
    for o in ('adamw',):
        rs=R[(cell,o)]; lam=rs[0]['weight_decay']
        for ep in EPS[cell]:
            h=[r['history'][ep-1] for r in rs]
            eta=st.mean(x['lr_end'] for x in h)
            pred=math.sqrt(2*eta*lam*0.1/1.9)
            per=[st.mean(x['ang_upd'][i] for x in h) for i in range(len(h[0]['ang_upd']))]
            print(cell,o,'ep',ep,'pred',f'{pred:.5f}','meas mean',f'{st.mean(per):.5f}','min',f'{min(per):.5f}','max',f'{max(per):.5f}','ratio',f'{st.mean(per)/pred:.2f}')
json.dump(out,open('analysis/out/facts_measured.json','w'),indent=1)
