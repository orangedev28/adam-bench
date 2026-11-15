import json,glob,collections,statistics as st,math
OLD='results/benchmark'; NEW='results/followup'
def finals(path_glob):
    out=collections.defaultdict(dict)
    for f in glob.glob(path_glob):
        d=json.load(open(f))
        if d['phase']!='final': continue
        out[(d['cell'],d['optimizer'])][d['seed']]=d
    return out
OLDF=finals(f'{OLD}/*/runs/*/result.json')
NEWF=collections.defaultdict(dict)
for _e in ('E1','E2','E2b','E4','E5','E5b','E1c'):   # merged per seed; reruns are bit-identical
    for _k,_v in finals(f'{NEW}/{_e}/*/runs/*/result.json').items(): NEWF[_k].update(_v)
E3F=finals(f'{NEW}/E3/*/runs/*/result.json')
def acc(runs):  # seed -> test acc %
    return {s:(0.0 if r['diverged'] else r['final']['test_acc']*100) for s,r in runs.items()}
def get(cell,opt):
    runs=dict(NEWF.get((cell,opt),{})); runs.update(OLDF.get((cell,opt),{}))   # benchmark run wins for a rerun seed
    return acc(runs)
