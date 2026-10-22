# Threshold sensitivity of point estimates (no bootstrap). Usage: python3 -I threshold_sensitivity.py all.pkl > threshold_sensitivity_output.txt
# Reuses the cohort-construction code of analysis_c2.py (copied verbatim above the marker); only the n_c threshold varies.
"""C2 prespecified analysis, frozen plan v3 + v4 wording.

Usage: python3 -I analysis_c2.py all.pkl
No model fitting, no hypothesis tests, no p-values. Permutation results are a no-association reference only.
Unspecified-in-plan choices (flagged in the report): check seed 20261017, permutation seed 20261018.
"""
import sys, os, warnings
import numpy as np, pandas as pd
from scipy.stats import kendalltau, rankdata

warnings.filterwarnings('ignore')
pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 500)

# ------------------------------------------------------------------ frozen parameters
SEED_PRIMARY = 20261016
SEED_CHECK = 20261017      # plan: "a second fixed seed" (value not specified) -> chosen before running
SEED_PERM = 20261018       # plan: permutation seed not specified -> chosen before running
B = int(os.environ.get('C2_B', 1000))   # frozen primary value 1000; env override used ONLY for the documented supplementary run (amendment A1)
NPERM = 10000
N_MIN = 100
WASHOUT = 6
LOOKBACK = 6
WINDOWS = (1, 3, 6)
HORIZONS = (6, 9)
PAIRS = [(1, 3), (1, 6), (3, 6)]
COLS8 = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'cid', 'Country']
COMBOS = [(h, w) for h in HORIZONS for w in WINDOWS]

def P(s=""):
    print(s, flush=True)

def sec(t):
    P("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)

df = pd.read_pickle(sys.argv[1]).rename(columns={'Customer ID': 'cid'})

# ------------------------------------------------------------------ month axis (plan section 3)
d_min, d_max = df.InvoiceDate.min(), df.InvoiceDate.max()
assert d_min.normalize().day == 1                      # month 0 complete
T0_YEAR, T0_MONTH = d_min.year, d_min.month            # t=0 month
p_max = d_max.to_period('M')
L_period = p_max if d_max.normalize() == p_max.end_time.normalize() else p_max - 1
T0 = pd.Period(year=T0_YEAR, month=T0_MONTH, freq='M')
L = (L_period - T0).n
def month_label(i):
    return str(T0 + int(i))

def build(d):
    """Cohort assignment, screening, activity matrix from a dataframe of rows (plan v3 sec. 3-5)."""
    d = d.copy()
    d['canc'] = d.Invoice.str.upper().str.startswith('C')
    d['mi'] = (d.InvoiceDate.dt.year - T0_YEAR) * 12 + (d.InvoiceDate.dt.month - T0_MONTH)
    d['qual'] = d.cid.notna() & ~d.canc & (d.Quantity > 0) & (d.Price > 0)
    q = d[d.qual].copy(); q['cid'] = q.cid.astype(int)
    first_mi = q.groupby('cid').mi.min()                       # whole-file first qualifying record
    idr = d[d.cid.notna() & (d.mi >= 0) & (d.mi <= L)].copy(); idr['cid'] = idr.cid.astype(int)
    pres = pd.crosstab(idr.cid, idr.mi).reindex(columns=range(0, L + 1), fill_value=0) > 0   # any ID-bearing record
    qa = q[(q.mi >= 0) & (q.mi <= L)]
    act = pd.crosstab(qa.cid, qa.mi).reindex(columns=range(0, L + 1), fill_value=0) > 0       # qualifying activity
    members, cand_n, excl_n = {}, {}, {}
    for c in range(WASHOUT, L + 1):
        ids = first_mi[first_mi == c].index
        bad = pres.loc[ids, c - LOOKBACK:c - 1].any(axis=1) if len(ids) else pd.Series([], dtype=bool)
        members[c] = np.array(sorted(ids[~bad.values]))
        cand_n[c] = len(ids); excl_n[c] = int(bad.sum())
    washout_n = int(((first_mi >= 0) & (first_mi < WASHOUT)).sum())
    return dict(d=d, q=q, first_mi=first_mi, pres=pres, act=act, members=members, cand=cand_n, excl=excl_n, washout=washout_n)


# ---- threshold sensitivity (point estimates only; no bootstrap) ----
R = build(df)
n_c = {c: len(v) for c, v in R['members'].items()}
act = R['act']
def rates(cs, h, w):
    return np.array([act.loc[R['members'][c]].values[:, c+h-w+1:c+h+1].any(axis=1).mean() for c in cs])
def tau(x, y): return kendalltau(x, y)[0]
def top_bottom(rate):
    s = np.sort(rate); return rate >= s[-3], rate <= s[2]
def jacc(a, b): return (a & b).sum() / (a | b).sum()
P("THRESHOLD SENSITIVITY OF POINT ESTIMATES (no bootstrap). Same construction as analysis_c2.py; only N_MIN varies.")
P("n_c by cohort: " + ", ".join(f"{month_label(c)}={n_c[c]}" for c in range(WASHOUT, L+1)))
for thr in (50, 75, 100, 125):
    P(f"\n=== n_c threshold {thr} ===")
    for h in HORIZONS:
        cs = [c for c in range(WASHOUT, L - h + 1) if n_c[c] >= thr]
        P(f"h={h}: k={len(cs)} cohorts: {[month_label(c) for c in cs]}")
        pt = {w: rates(cs, h, w) for w in WINDOWS}
        ties = {w: len(pt[w]) - len(set(pt[w])) for w in WINDOWS}
        out = []
        for a, b in PAIRS:
            ta, ba = top_bottom(pt[a]); tb, bb = top_bottom(pt[b])
            out.append(f"({a},{b}) tau-b={tau(pt[a], pt[b]):.3f} J_top={jacc(ta,tb):.3f} J_bot={jacc(ba,bb):.3f}")
        P("   " + "; ".join(out) + f"; tied rates per window {ties}")
