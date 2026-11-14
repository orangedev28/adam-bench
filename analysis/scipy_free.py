import math,sys
sys.path.insert(0,'.')
from bench.stats import betainc
def f_sf(F,d1,d2):
    if F<=0: return 1.0
    x=d2/(d2+d1*F)
    return betainc(d2/2,d1/2,x)
