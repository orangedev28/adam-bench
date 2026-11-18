"""LaTeX tables of Paper 2 (Tables 2-4) from analysis/out/facts_iccies.json. Run: python analysis/iccies_tables.py"""
import json, sys, os
from decimal import Decimal, ROUND_HALF_UP
def r2(x,sign=False):
    d=Decimal(repr(round(x,6))).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
    return (('+' if d>=0 else '') if sign else '')+str(d)
F=json.load(open('analysis/out/facts_iccies.json')); A=F['anova']; DB=F['dbn']
D='analysis/out/'; CELLS='BDECFG'; ROWS=(('sfadamw','SF-AdamW'),('adabelief','AdaBelief'))
NT=len(CELLS)*2*3
def head(cap,lab,cols,sep='3pt'):
    return ('\\begin{table}[!t]\n\\caption{'+cap+'}\n\\label{'+lab+'}\n\\centering\n\\footnotesize\n'
            f'\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n')
TAIL='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
t=head('Factorial design: update procedure $\\times$ weight-decay form. Final test accuracy (\\%), mean $\\pm$ standard deviation over five seeds; every configuration has its own tuned LR. Cells as in Table~\\ref{tab:cells}; dec.: decoupled decay.','tab:2x2','@{}lcccc@{}','4pt')
t+=' & \\multicolumn{2}{c}{Adam procedure} & \\multicolumn{2}{c}{Procedure X} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCell, X & L2 (Adam) & dec. (AdamW) & X + L2 & X + dec. \\\\\n\\midrule\n'
for j,(x,name) in enumerate(ROWS):
    if j: t+='\\midrule\n'
    for c in CELLS:
        r=A[f'{c}-{x}']; t+=f'{c}, {name} & '+' & '.join(f"${r2(r['means'][k])}\\pm{r2(r['sds'][k])}$" for k in ('ac','ad','xc','xd'))+' \\\\\n'
open(D+'tab_2x2.tex','w').write(t+TAIL)
def ef(e): return '$<$.01' if e<0.005 else f'{e:.2f}'.replace('0.','.')
t=head(f'Effect sizes in accuracy points from the per-seed contrasts of Eq.~(\\ref{{eq:contrasts}}) (mean $\\pm$ half-width of the 95\\% confidence interval, five seeds), and the shares $\\eta^2$ of the total sum of squares in the seed-blocked ANOVA of Eq.~(\\ref{{eq:anova}}) for procedure (P), decay form (D) and interaction (I). $^{{*}}$: Holm-adjusted $p<0.05$ in the paired $t$-test on the contrast ({NT} tests); $^{{\\dagger}}$: significant only in the ANOVA $F$-test.','tab:est','@{}lcccc@{}','4pt')
t+='Cell, X & Procedure & Decay form & Interaction & $\\eta^2$ (P\\,/\\,D\\,/\\,I) \\\\\n\\midrule\n'
for j,(x,name) in enumerate(ROWS):
    if j: t+='\\midrule\n'
    for c in CELLS:
        r=A[f'{c}-{x}']; cells=[]
        for k in ('rule','wd','int'):
            mk='^{*}' if r[k]['p_rm_holm']<0.05 else ('^{\\dagger}' if r[k]['p_blk_holm']<0.05 else '')
            cells.append(f"${r2(r[k]['est'],True)}\\pm{r2(r[k]['ci'])}{mk}$")
        t+=f'{c}, {name} & '+' & '.join(cells)+' & '+'\\,/\\,'.join(ef(r[k]['eta2']) for k in ('rule','wd','int'))+' \\\\\n'
open(D+'tab_est.tex','w').write(t+TAIL)
def pf(p): return '$<$0.001' if p<0.001 else f'{p:.3f}' if p<0.01 else f'{p:.2f}'
t=head('Dependence of the interaction on BN, Eq.~(\\ref{eq:bndiff}): mean interaction contrast with BN and without BN, their difference $\\Delta_{\\mathrm{BN}}$ with the half-width of its 95\\% Welch confidence interval, and the Holm-adjusted $p$-value (4 tests).','tab:dbn','@{}llcccc@{}','5pt')
t+=' & & \\multicolumn{2}{c}{$\\bar c_{\\mathrm{int}}$} & & \\\\\n\\cmidrule(lr){3-4}\nNetwork (cells) & X & with BN & without BN & $\\Delta_{\\mathrm{BN}}$ & $p$ \\\\\n\\midrule\n'
for arch,bn,nb in (('Five-layer','B','E'),('ResNet-18','C','F')):
    for x,name in ROWS:
        v=DB[f'{bn}-{nb}-{x}']
        t+=f"{arch} ({bn}/{nb}) & {name} & ${r2(v['int_bn'],True)}$ & ${r2(v['int_nb'],True)}$ & ${r2(v['diff'],True)}\\pm{r2(v['ci'])}$ & {pf(v['p_holm'])} \\\\\n"
open(D+'tab_dbn.tex','w').write(t+TAIL)
print('wrote', D+'tab_2x2.tex', D+'tab_est.tex', D+'tab_dbn.tex')
