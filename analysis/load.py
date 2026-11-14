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
NEWF=finals(f'{NEW}/E1/*/runs/*/result.json'); NEWF.update(finals(f'{NEW}/E2/*/runs/*/result.json'))
E3F=finals(f'{NEW}/E3/*/runs/*/result.json')
def acc(runs):  # seed -> test acc %
    return {s:(0.0 if r['diverged'] else r['final']['test_acc']*100) for s,r in runs.items()}
def get(cell,opt):
    return acc(NEWF.get((cell,opt)) or OLDF[(cell,opt)])
