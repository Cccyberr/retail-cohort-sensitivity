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

sec("0. PLAN CONFORMANCE CHECKS (run before any outcome is computed)")
R = build(df)
n_c = {c: len(v) for c, v in R['members'].items()}
elig = {h: [c for c in range(WASHOUT, L - h + 1) if n_c[c] >= N_MIN] for h in HORIZONS}
P(f"L = {L} ({month_label(L)}); t=0 = {month_label(0)}; washout cohorts c<{WASHOUT} excluded; look-back {LOOKBACK} months; n_c threshold {N_MIN}")
P(f"eligible cohorts h=6: {elig[6]} (k={len(elig[6])});  h=9: {elig[9]} (k={len(elig[9])})")
assert L == 23
assert elig[6] == [6, 7, 8, 9, 10, 11, 14, 15, 16, 17], elig[6]
assert elig[9] == [6, 7, 8, 9, 10, 11, 14], elig[9]
AUDIT_NC = {6: 262, 7: 185, 8: 161, 9: 240, 10: 376, 11: 324, 12: 76, 13: 71, 14: 123, 15: 179, 16: 106, 17: 110,
            18: 108, 19: 102, 20: 106, 21: 189, 22: 221, 23: 191}
assert all(n_c[c] == AUDIT_NC[c] for c in AUDIT_NC), "n_c differs from the audit"
P("n_c per cohort matches the counts-only audit: OK")
P(f"customers excluded by washout (first qualifying month 0..5): {R['washout']}; candidates {sum(R['cand'].values())}; "
  f"prior-record excluded {sum(R['excl'].values())}; final n_c total {sum(n_c.values())}")
# dedup invariance check (informational; frozen rules not altered)
Rd = build(df.drop_duplicates(subset=COLS8))
same_members = all(np.array_equal(R['members'][c], Rd['members'][c]) for c in R['members'])
same_act = R['act'].reindex(Rd['act'].index).fillna(False).astype(bool).equals(Rd['act'].astype(bool)) and \
    (set(R['act'].index) == set(Rd['act'].index))
P(f"duplicate-row removal (all 8 columns) leaves cohort membership identical: {same_members}; activity matrix identical: {same_act}")
assert same_members and same_act

act = R['act']
COHORTS = elig[6]                      # h=9 set is the first 7 of this list
H9N = len(elig[9]); assert COHORTS[:H9N] == elig[9]
Am = {}   # per cohort: n x 6 matrix of activity indicators for COMBOS (nan->0 where follow-up incomplete)
for c in COHORTS:
    M = act.loc[R['members'][c]].values
    cols = []
    for (h, w) in COMBOS:
        if c + h <= L:
            cols.append(M[:, c + h - w + 1:c + h + 1].any(axis=1))
        else:
            cols.append(np.zeros(len(M), dtype=bool))
    Am[c] = np.column_stack(cols).astype(np.int8)
    # nesting check for h with full follow-up
    for h in HORIZONS:
        if c + h <= L:
            a1, a3, a6 = (Am[c][:, COMBOS.index((h, w))] for w in WINDOWS)
            assert (a1 <= a3).all() and (a3 <= a6).all()
P("windows nested at customer level (A1<=A3<=A6): OK for all cohort/horizon cells")

# calendar months of each window
sec("0b. CALENDAR MONTHS OF EACH WINDOW (cohort month c; window = [c+h-w+1, c+h])")
rows = []
for h in HORIZONS:
    for c in elig[h]:
        rows.append(dict(h=h, c=c, cohort=month_label(c), w1=month_label(c + h),
                         w3=f"{month_label(c+h-2)}..{month_label(c+h)}", w6=f"{month_label(c+h-5)}..{month_label(c+h)}"))
P(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ point estimates
def rates_from_counts(idx_by_cohort):
    """idx_by_cohort: dict c -> index array. returns dict (h,w)->vector over cohorts of the horizon."""
    out = {}
    for j, (h, w) in enumerate(COMBOS):
        cs = elig[h]
        out[(h, w)] = np.array([Am[c][idx_by_cohort[c], j].mean() for c in cs])
    return out

full_idx = {c: np.arange(len(Am[c])) for c in COHORTS}
pt = rates_from_counts(full_idx)

def tau(x, y):
    return kendalltau(x, y)[0]

def top_bottom(rate):
    s = np.sort(rate)
    return rate >= s[-3], rate <= s[2]          # inclusive of ties at the cutoff value

def jacc(a, b):
    return (a & b).sum() / (a | b).sum()

sec("1. ELIGIBLE COHORTS, SIZES, SCREENING EFFECT, EXCLUDED COHORTS")
dfx = R['d']
base = ~dfx.canc & (dfx.Quantity > 0) & (dfx.Price > 0)
mon = dfx[base & (dfx.mi >= 0) & (dfx.mi <= L)].groupby('mi').agg(rows=('Invoice', 'size'), nullid=('cid', lambda s: s.isna().sum()))
mon['missing_ID_share_pct'] = (mon.nullid / mon.rows * 100).round(2)   # denominator: non-cancellation rows with Quantity>0 and Price>0 (all IDs)
tab = []
for c in range(WASHOUT, L + 1):
    tab.append(dict(c=c, month=month_label(c), candidates=R['cand'][c], excluded_prior_record=R['excl'][c],
                    share_excl_pct=round(R['excl'][c] / R['cand'][c] * 100, 2), n_c=n_c[c], n_ge_100=n_c[c] >= N_MIN,
                    eligible_h6=c in elig[6], eligible_h9=c in elig[9],
                    missing_ID_share_pct_among_noncancel_posQ_posP_rows=mon.loc[c, 'missing_ID_share_pct']))
tab = pd.DataFrame(tab)
P(tab.to_string(index=False))
P(f"\nExcluded below-threshold cohorts: December 2010 (c=12) n_c={n_c[12]}, January 2011 (c=13) n_c={n_c[13]}; "
  f"missing-ID share among non-cancellation rows with Quantity>0 and Price>0 in those calendar months (not share of all rows): {mon.loc[12,'missing_ID_share_pct']}% and {mon.loc[13,'missing_ID_share_pct']}%")
P(f"Near-threshold eligible cohorts: c=16 (n_c={n_c[16]}) and c=17 (n_c={n_c[17]}); c=14 n_c={n_c[14]}")
P("Eligible sequences are non-contiguous (c=12,13 absent); they are not a continuous time series.")

sec("2. POINT ESTIMATES (original data): activity rates, R6-R1, ranks, top/bottom sets")
desc = {}
for h in HORIZONS:
    cs = elig[h]
    t = pd.DataFrame({'c': cs, 'cohort': [month_label(c) for c in cs], 'n_c': [n_c[c] for c in cs]})
    for w in WINDOWS:
        t[f'R{w}'] = pt[(h, w)]
    t['G=R6-R1'] = t.R6 - t.R1
    for w in WINDOWS:
        t[f'rank_w{w}'] = rankdata(-pt[(h, w)], method='average')
    desc[h] = t
    P(f"\n--- h = {h}  (k = {len(cs)}) ---")
    P(t.round(4).to_string(index=False))
    for w in WINDOWS:
        top, bot = top_bottom(pt[(h, w)])
        P(f"  w={w}: top set cohorts {[cs[i] for i in np.where(top)[0]]} (size {top.sum()}), bottom set {[cs[i] for i in np.where(bot)[0]]} (size {bot.sum()}); "
          f"boundary ties: top cutoff shared by {int((pt[(h,w)]==np.sort(pt[(h,w)])[-3]).sum())} cohort(s), bottom cutoff shared by {int((pt[(h,w)]==np.sort(pt[(h,w)])[2]).sum())} cohort(s)")
    P("  tied rates within vector: " + ", ".join(f"w={w}: {len(pt[(h,w)]) - len(set(pt[(h,w)]))}" for w in WINDOWS))
    for a, b in PAIRS:
        top_a, bot_a = top_bottom(pt[(h, a)]); top_b, bot_b = top_bottom(pt[(h, b)])
        P(f"  pair ({a},{b}): tau-b point = {tau(pt[(h,a)], pt[(h,b)]):.4f}; Jaccard top = {jacc(top_a, top_b):.3f}, bottom = {jacc(bot_a, bot_b):.3f}")

# ------------------------------------------------------------------ bootstrap
def bootstrap(seed):
    rng = np.random.default_rng(seed)
    nC = {c: len(Am[c]) for c in COHORTS}
    store = dict(S={}, D={}, X={}, JT_D={}, JB_D={}, JT_X={}, JB_X={}, tie_top={}, tie_bot={}, nan={})
    for h in HORIZONS:
        for w in WINDOWS:
            store['S'][(h, w)] = []; store['tie_top'][(h, w)] = []; store['tie_bot'][(h, w)] = []
        for p in PAIRS:
            for k in ('D', 'X', 'JT_D', 'JB_D', 'JT_X', 'JB_X'):
                store[k][(h, p)] = []
    Rrep = {k: [] for k in COMBOS}
    Grep = {h: [] for h in HORIZONS}
    nan_counts = {'S': 0, 'D': 0, 'X': 0}
    for j in range(B):
        res = []
        for r in (1, 2):
            idxs = {c: rng.integers(0, nC[c], size=nC[c]) for c in COHORTS}
            rr = rates_from_counts(idxs)
            res.append(rr)
        r1, r2 = res
        for combo in COMBOS:
            Rrep[combo].append(r1[combo])
        for h in HORIZONS:
            Grep[h].append(r1[(h, 6)] - r1[(h, 1)])
            for w in WINDOWS:
                s = tau(r1[(h, w)], r2[(h, w)])
                if np.isnan(s): nan_counts['S'] += 1
                store['S'][(h, w)].append(s)
                for rr in (r1, r2):
                    top, bot = top_bottom(rr[(h, w)])
                    store['tie_top'][(h, w)].append(top.sum() > 3)
                    store['tie_bot'][(h, w)].append(bot.sum() > 3)
            for p in PAIRS:
                a, b = p
                d1, d2 = tau(r1[(h, a)], r1[(h, b)]), tau(r2[(h, a)], r2[(h, b)])
                x = tau(r1[(h, a)], r2[(h, b)])
                dval = np.nan if (np.isnan(d1) or np.isnan(d2)) else (d1 + d2) / 2
                if np.isnan(dval): nan_counts['D'] += 1
                if np.isnan(x): nan_counts['X'] += 1
                store['D'][(h, p)].append(dval); store['X'][(h, p)].append(x)
                t1a, b1a = top_bottom(r1[(h, a)]); t1b, b1b = top_bottom(r1[(h, b)])
                t2a, b2a = top_bottom(r2[(h, a)]); t2b, b2b = top_bottom(r2[(h, b)])
                store['JT_D'][(h, p)].append((jacc(t1a, t1b) + jacc(t2a, t2b)) / 2)
                store['JB_D'][(h, p)].append((jacc(b1a, b1b) + jacc(b2a, b2b)) / 2)
                store['JT_X'][(h, p)].append(jacc(t1a, t2b))
                store['JB_X'][(h, p)].append(jacc(b1a, b2b))
    store['nan'] = nan_counts
    store['R'] = {k: np.array(v) for k, v in Rrep.items()}
    store['G'] = {h: np.array(v) for h, v in Grep.items()}
    return store

def q3(a):
    a = np.asarray(a, float)
    return np.nanmedian(a), np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)

def fmt(a):
    m, lo, hi = q3(a)
    return f"{m:6.3f} [{lo:6.3f}, {hi:6.3f}]"

sec(f"3. PAIRED CUSTOMER-LEVEL BOOTSTRAP (B={B} pairs of independent resamples; seed 20261016)" + ("" if B == 1000 else "  [SUPPLEMENTARY RUN: B overridden by C2_B; NOT the frozen primary analysis]"))
bs = bootstrap(SEED_PRIMARY)
P(f"NaN tau-b counts across replicates: {bs['nan']}")

for h in HORIZONS:
    cs = elig[h]
    P(f"\n------ h = {h} (k = {len(cs)}){'  [DESCRIPTIVE ONLY: k < 8]' if len(cs) < 8 else ''} ------")
    P("Cohort rate and G intervals (median [2.5%, 97.5%] over resample 1 of each pair):")
    t = pd.DataFrame({'c': cs, 'cohort': [month_label(c) for c in cs], 'n_c': [n_c[c] for c in cs]})
    for w in WINDOWS:
        arr = bs['R'][(h, w)]
        t[f'R{w} pt'] = pt[(h, w)]
        t[f'R{w} 2.5%'] = np.percentile(arr, 2.5, axis=0); t[f'R{w} 97.5%'] = np.percentile(arr, 97.5, axis=0)
    g = bs['G'][h]
    t['G pt'] = pt[(h, 6)] - pt[(h, 1)]; t['G 2.5%'] = np.percentile(g, 2.5, axis=0); t['G 97.5%'] = np.percentile(g, 97.5, axis=0)
    P(t.round(4).to_string(index=False))
    P("\nSampling-only reference S(w): tau-b between independent resamples under the same window (median [2.5%, 97.5%])")
    for w in WINDOWS:
        P(f"  S(w={w}): {fmt(bs['S'][(h, w)])}")
    P("\nBetween-definition quantities per window pair (tau-b): point (original data) | D paired | X independent")
    for p in PAIRS:
        a, b = p
        P(f"  ({a},{b}): point {tau(pt[(h,a)], pt[(h,b)]):6.3f} | D {fmt(bs['D'][(h,p)])} | X {fmt(bs['X'][(h,p)])}")
    P("\nTop-3 / bottom-3 Jaccard overlap per window pair: point | D | X")
    for p in PAIRS:
        a, b = p
        ta, ba = top_bottom(pt[(h, a)]); tb, bb = top_bottom(pt[(h, b)])
        P(f"  ({a},{b}) top: point {jacc(ta,tb):5.3f} | D {fmt(bs['JT_D'][(h,p)])} | X {fmt(bs['JT_X'][(h,p)])}")
        P(f"  ({a},{b}) bot: point {jacc(ba,bb):5.3f} | D {fmt(bs['JB_D'][(h,p)])} | X {fmt(bs['JB_X'][(h,p)])}")
    P(f"\nShare of resamples (2 per pair, {2*B} per window) with boundary ties (set size > 3):")
    for w in WINDOWS:
        P(f"  w={w}: top {np.mean(bs['tie_top'][(h,w)])*100:.2f}%  bottom {np.mean(bs['tie_bot'][(h,w)])*100:.2f}%")

# ------------------------------------------------------------------ permutation diagnostic
sec("4. NO-ASSOCIATION REFERENCES (permutation of cohort labels; 10,000 permutations; seed 20261018). Diagnostic only, not a test.")
rngp = np.random.default_rng(SEED_PERM)
for h in HORIZONS:
    k = len(elig[h])
    var_formula = 2 * (2 * k + 5) / (9 * k * (k - 1))
    P(f"\nh={h}: k={k}; analytic no-tie null SD of tau = {np.sqrt(var_formula):.3f}; minimum tau step (one swapped pair, no ties) = {4/(k*(k-1)):.4f}; "
      f"chance expectation of top-3 (or bottom-3) overlap count between independent random sets = 9/k = {9/k:.2f}")
    for p in PAIRS:
        a, b = p
        x, y = pt[(h, a)], pt[(h, b)]
        nulls = np.array([tau(x, rngp.permutation(y)) for _ in range(NPERM)])
        P(f"  ({a},{b}) with observed ties preserved: null mean {np.nanmean(nulls):6.3f}, SD {np.nanstd(nulls):.3f}, 95th percentile {np.nanpercentile(nulls,95):.3f}")

# ------------------------------------------------------------------ cancellation reporting
sec("5. CANCELLATION REPORTING (descriptive only; equal-length exposure interval [c, c+h] per cohort)")
dfx = R['d']
cmon = dfx[(dfx.mi >= 0)].groupby('mi').agg(rows=('Invoice', 'size'), canc_rows=('canc', 'sum'))
cmon['canc_share_%'] = (cmon.canc_rows / cmon.rows * 100).round(2)
cmon['month'] = [month_label(i) for i in cmon.index]
P("Cancellation-prefix rows by calendar month (all rows, as in file; includes duplicated overlap block rows):")
P(cmon[['month', 'rows', 'canc_rows', 'canc_share_%']].to_string())

def canc_table(d_rows, label):
    d_rows = d_rows[d_rows.cid.notna()].copy(); d_rows['cid'] = d_rows.cid.astype(int)
    out = []
    for h in HORIZONS:
        for c in elig[h]:
            ids = set(R['members'][c])
            sub = d_rows[d_rows.cid.isin(ids) & (d_rows.mi >= c) & (d_rows.mi <= c + h)]
            cn = sub[sub.canc]; qu = sub[sub.qual]
            out.append(dict(h=h, c=c, cohort=month_label(c), n_c=len(ids), cust_with_canc=cn.cid.nunique(),
                            share_cust_pct=round(cn.cid.nunique() / len(ids) * 100, 2), canc_rows=len(cn), qual_rows=len(qu),
                            canc_per_100_qual=round(len(cn) / len(qu) * 100, 2) if len(qu) else np.nan))
    t = pd.DataFrame(out)
    P(f"\n{label}")
    P(t.to_string(index=False))
    return t
t_asfile = canc_table(R['d'], "Per cohort, interval [c, c+h], rows as in file (no deduplication, frozen rule):")
dd = R['d'].drop_duplicates(subset=COLS8)
t_dedup = canc_table(dd, "Same, after removing exact duplicate rows (disclosure only; rows-based figures only):")
P("\nDifference (as-file minus dedup) in cancellation rows / qualifying rows by cohort-horizon (nonzero only):")
diff = t_asfile[['h', 'c']].copy(); diff['d_canc_rows'] = t_asfile.canc_rows - t_dedup.canc_rows; diff['d_qual_rows'] = t_asfile.qual_rows - t_dedup.qual_rows
P(diff[(diff.d_canc_rows != 0) | (diff.d_qual_rows != 0)].to_string(index=False))

# ------------------------------------------------------------------ Monte Carlo check
sec(f"6. MONTE CARLO CHECK: full bootstrap (B={B}) repeated with a second fixed seed (20261017)")
bs2 = bootstrap(SEED_CHECK)
mx = 0; mx_ci = 0; mx_tau = 0; mx_jac = 0
jac_rows = []
for h in HORIZONS:
    for w in WINDOWS:
        a, b = q3(bs['S'][(h, w)]), q3(bs2['S'][(h, w)]); mx_tau = max(mx_tau, max(abs(np.array(a) - np.array(b)))); mx = max(mx, mx_tau)
    for p in PAIRS:
        for k in ('D', 'X', 'JT_D', 'JB_D', 'JT_X', 'JB_X'):
            a, b = q3(bs[k][(h, p)]), q3(bs2[k][(h, p)])
            dd_ = max(abs(np.array(a) - np.array(b)))
            if k in ('D', 'X'):
                mx_tau = max(mx_tau, dd_)
            else:
                mx_jac = max(mx_jac, dd_)
                jac_rows.append((h, p, k, tuple(round(v, 3) for v in a), tuple(round(v, 3) for v in b), round(dd_, 3)))
            mx = max(mx, dd_)
    for w in WINDOWS:
        da = np.percentile(bs['R'][(h, w)], [2.5, 97.5], axis=0) - np.percentile(bs2['R'][(h, w)], [2.5, 97.5], axis=0)
        mx_ci = max(mx_ci, np.abs(da).max())
    da = np.percentile(bs['G'][h], [2.5, 97.5], axis=0) - np.percentile(bs2['G'][h], [2.5, 97.5], axis=0)
    mx_ci = max(mx_ci, np.abs(da).max())
P(f"max absolute difference between seeds in any median/percentile of S, D, X and Jaccard summaries: {mx:.4f}")
P(f"  of which tau-b summaries (S, D, X; median and both endpoints): {mx_tau:.4f}")
P(f"  of which Jaccard overlap summaries (discrete-valued; median and both endpoints): {mx_jac:.4f}")
P("  Jaccard summaries with seed-to-seed difference >= 0.25 (h, pair, quantity, seed1 [median, lo, hi], seed2 [median, lo, hi], max diff):")
for r_ in jac_rows:
    if r_[5] >= 0.25:
        P(f"    {r_}")
P(f"max absolute difference between seeds in any 2.5%/97.5% endpoint of cohort rates and G: {mx_ci:.4f}")
P(f"NaN counts seed 2: {bs2['nan']}")
P("\nDONE")
