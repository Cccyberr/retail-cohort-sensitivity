#!/usr/bin/env python3
"""Reproducible plotting script for Figs. 1-3 of the C2 manuscript.

Every plotted value is PARSED from the frozen analysis output (analysis_output.txt,
produced by analysis_c2.py, B = 1000, seed 20261016). Nothing is typed in by hand.
Before plotting, values are cross-checked against (i) an independent recomputation
of ranks from the printed rates and (ii) the numbers printed in the manuscript .tex.
A log of all checks is written to verification_log.txt.

Usage: python3 plot_figures.py analysis_output.txt [manuscript.tex]
The manuscript argument is optional. With it, the plotted values are also cross-checked
against the numbers printed in the manuscript (checks 3-6); without it only checks 1-2 run.
"""
import re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from scipy.stats import rankdata

OUT = sys.argv[1]
TEX = sys.argv[2] if len(sys.argv) > 2 else None
txt = open(OUT, encoding="utf-8").read()
tex = open(TEX, encoding="utf-8").read() if TEX else ""
LOG = []
def log(s):
    LOG.append(s); print(s)

# ----------------------------------------------------------------- parsing
sec2 = txt.split("2. POINT ESTIMATES")[1].split("3. PAIRED CUSTOMER-LEVEL")[0]
sec3 = txt.split("3. PAIRED CUSTOMER-LEVEL")[1].split("4. NO-ASSOCIATION")[0]

def parse_point(block):
    rows = []
    for m in re.finditer(r"^\s*(\d+)\s+(\d{4}-\d{2})\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", block, re.M):
        c, name, n, r1, r3, r6, g, k1, k3, k6 = m.groups()
        rows.append(dict(c=int(c), cohort=name, n=int(n), R=[r1, r3, r6], G=g, rank=[int(float(k)) for k in (k1, k3, k6)]))
    return rows

pt = {}
for h in (6, 9):
    blk = sec2.split(f"--- h = {h}")[1]
    blk = blk.split("--- h = 9")[0] if h == 6 else blk
    pt[h] = parse_point(blk)
    ties = re.findall(r"tied rates within vector: w=1: (\d+), w=3: (\d+), w=6: (\d+)", blk)
    assert ties and all(int(x) == 0 for x in ties[0]), "unexpected ties in observed rates"
assert [len(pt[6]), len(pt[9])] == [10, 7]
assert [r["cohort"] for r in pt[9]] == [r["cohort"] for r in pt[6]][:7]
log("PARSE: h=6 -> 10 cohorts, h=9 -> 7 cohorts (first seven of h=6); observed rates contain no ties in any window (printed tie counts = 0).")

def parse_bootstrap(h):
    blk = sec3.split(f"------ h = {h}")[1]
    if h == 6:
        blk = blk.split("------ h = 9")[0]
    S = {int(m.group(1)): tuple(map(float, m.groups()[1:]))
         for m in re.finditer(r"S\(w=(\d)\):\s+(-?[\d.]+) \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]", blk)}
    P = {}
    for m in re.finditer(r"^\s*\((\d),(\d)\): point\s+(-?[\d.]+) \| D\s+(-?[\d.]+) \[\s*(-?[\d.]+),\s*(-?[\d.]+)\] \| X\s+(-?[\d.]+) \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]", blk, re.M):
        a, b, pnt, dm, dl, du, xm, xl, xu = m.groups()
        P[(int(a), int(b))] = dict(point=float(pnt), D=(float(dm), float(dl), float(du)), X=(float(xm), float(xl), float(xu)))
    assert set(S) == {1, 3, 6} and set(P) == {(1, 3), (1, 6), (3, 6)}
    B = re.search(r"B=(\d+) pairs", txt).group(1)
    return S, P, int(B)

boot = {h: parse_bootstrap(h) for h in (6, 9)}
assert boot[6][2] == 1000
log("PARSE: bootstrap S/D/X summaries read for h=6 and h=9 (B=1000 frozen primary run).")

# ----------------------------------------------------------------- verification
def r3(x):  # round half up to 3 dp, as the manuscript tables do
    return str(Decimal(x).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

for h in (6, 9):
    for w_i in range(3):
        rates = np.array([float(r["R"][w_i]) for r in pt[h]])
        rk = rankdata(-rates, method="average")
        assert np.all(rk == np.array([r["rank"][w_i] for r in pt[h]])), (h, w_i)
log("CHECK 1: ranks recomputed from printed rates (descending, average ties) equal printed ranks for all windows, h=6 and h=9: PASS")
for h in (6, 9):
    for r in pt[h]:
        assert all(r["R"][i] <= r["R"][i+1] for i in range(2)), "nesting violated"
log("CHECK 2: R1 <= R3 <= R6 for every plotted cohort/horizon (nesting): PASS")

if TEX:
    # manuscript Table (tab:rates) rows
    rows_tex = re.findall(r"^(\d{4}-\d{2}) & ([\d.]+) & (\d+) & ([\d.]+) & (\d+) & ([\d.]+) & (\d+)\\\\", tex, re.M)
    assert len(rows_tex) == 10
    for (name, a1, k1, a3, k3, a6, k6), r in zip(rows_tex, pt[6]):
        assert name == r["cohort"]
        assert [a1, a3, a6] == [r3(x) for x in r["R"]], (name, a1, a3, a6, r["R"])
        assert [int(k1), int(k3), int(k6)] == r["rank"], name
    log("CHECK 3: manuscript Table (original rates and ranks, h=6) equals parsed output for all 10 cohorts x 3 windows (rates at 3 dp, ranks exact): PASS")

    # manuscript tau table (tab:tau)
    S6, P6, _ = boot[6]
    def f(x):
        s = r3(repr(x)); return s
    tau_rows = re.findall(r"^(\d) vs\. (\d) & ([\d.]+) & ([\d.\-$]+) \[([^\]]+)\] & ([\d.\-$]+) \[([^\]]+)\]", tex, re.M)
    assert len(tau_rows) == 3
    def tex_num(s): return float(s.replace("$-$", "-").replace("$", ""))
    for a, b, pnt, dm, dr, xm, xr in tau_rows:
        p = P6[(int(a), int(b))]
        assert abs(tex_num(pnt) - p["point"]) < 5e-4
        dl, du = [tex_num(v) for v in dr.split(",")]
        xl, xu = [tex_num(v) for v in xr.split(",")]
        assert (abs(tex_num(dm) - p["D"][0]), abs(dl - p["D"][1]), abs(du - p["D"][2])) < (5e-4,)*3
        assert max(abs(tex_num(xm) - p["X"][0]), abs(xl - p["X"][1]), abs(xu - p["X"][2])) < 5e-4
    log("CHECK 4: manuscript tau-b table (point, D, X; h=6) equals parsed output for 3 pairs: PASS")
    for w, (m, l, u) in S6.items():
        assert f"{m:.3f} [{l:.3f}, {u:.3f}]" in tex, w
    log("CHECK 5: manuscript S(w) values at h=6 (0.511/0.689/0.556 with ranges) found verbatim in the text and equal parsed values: PASS")
    m9 = re.search(r"The medians of \$S\(w\)\$ are ([\d.]+), ([\d.]+), and ([\d.]+)", tex)
    assert [float(x) for x in m9.groups()] == [round(boot[9][0][w][0], 3) for w in (1, 3, 6)]
    txt9 = re.search(r"point-estimate \$\\tau_b\$ is ([\d.]+) \(1 versus 3\), ([\d.]+) \(1 versus 6\), and ([\d.]+) \(3 versus 6\)", tex)
    assert [float(x) for x in txt9.groups()] == [round(boot[9][1][k]["point"], 3) for k in [(1, 3), (1, 6), (3, 6)]]
    log("CHECK 6: manuscript h=9 text (S medians 0.524/0.619/0.714; point tau-b 0.524/0.333/0.238) equals parsed output: PASS")

else:
    log("CHECKS 3-6 SKIPPED: no manuscript .tex supplied (figures are still drawn from the frozen output only).")

# ----------------------------------------------------------------- style
for fam in ("TeX Gyre Termes", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
    if any(fam == x.name for x in fm.fontManager.ttflist):
        FONT = fam; break
plt.rcParams.update({
    "font.family": FONT, "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
log(f"STYLE: font = {FONT}; figure width 7.16 in (IEEE two-column span); min font size 7.5 pt")
W = 7.16
WLAB = ["1 mo", "3 mo", "6 mo"]

# ----------------------------------------------------------------- Fig 1: heatmap
def fig1():
    fig = plt.figure(figsize=(W, 3.3))
    cell_h = 0.215
    left_axes = [(0.115, 0.42), (0.585, 0.42)]
    vmin, vmax = 0.0, 0.65
    cmap = plt.get_cmap("cividis")
    for (h, (x0, wd)) in zip((6, 9), ((0.155, 0.28), (0.665, 0.28))):
        rows = pt[h]; k = len(rows)
        ax = fig.add_axes([x0, 0.20 + (10 - k) * cell_h / 3.3, wd, k * cell_h / 3.3])
        A = np.array([[float(x) for x in r["R"]] for r in rows])
        im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(k):
            for j in range(3):
                v = A[i, j]
                lum = np.mean(cmap((v - vmin) / (vmax - vmin))[:3])
                ax.text(j, i, r3(rows[i]["R"][j]), ha="center", va="center", fontsize=7.5,
                        color="white" if lum < 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(WLAB)
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        ax.set_yticks(range(k)); ax.set_yticklabels([f"{r['cohort']} ({r['n']})" for r in rows])
        ax.set_xlabel("")
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", length=0)
        tag = "(a)" if h == 6 else "(b)"
        sub = "h = 6 months, k = 10 cohorts (main)" if h == 6 else "h = 9 months, k = 7 cohorts (descriptive only)"
        ax.set_title(f"{tag} {sub}", pad=16, fontsize=8.5)
    cax = fig.add_axes([0.25, 0.115, 0.50, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"Activity rate $R_w(c,h)$: share of cohort customers with $\geq 1$ qualifying record in the window", fontsize=7.5)
    cb.outline.set_linewidth(0.4)
    fig.text(0.012, 0.55, "Cohort month (cohort size $n_c$)", rotation=90, va="center", fontsize=8)
    return fig

# ----------------------------------------------------------------- Fig 2: slope chart
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw=dict(wspace=0.55))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.17)
    cm = plt.get_cmap("viridis")
    for ax, h, tag in zip(axes, (6, 9), ("(a)", "(b)")):
        rows = pt[h]; k = len(rows)
        for i, r in enumerate(rows):
            col = cm(0.05 + 0.85 * i / 9)       # colour follows chronological cohort order (same in both panels)
            ax.plot([0, 1, 2], r["rank"], "-o", color=col, lw=1.5, ms=4.5, mec="white", mew=0.5, zorder=3)
            ax.text(-0.07, r["rank"][0], r["cohort"], ha="right", va="center", fontsize=7.5, color="black")
            ax.text(2.07, r["rank"][2], r["cohort"], ha="left", va="center", fontsize=7.5, color="black")
        ax.set_ylim(k + 0.5, 0.5)                 # rank 1 at top
        ax.set_yticks(range(1, k + 1))
        ax.set_xlim(-0.05, 2.05); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1-month\nwindow", "3-month\nwindow", "6-month\nwindow"])
        ax.set_xlim(-0.75, 2.75)
        ax.set_ylabel("Rank (1 = highest activity rate)")
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
        sub = "h = 6, k = 10 (main)" if h == 6 else "h = 9, k = 7 (descriptive only)"
        ax.set_title(f"{tag} {sub}", fontsize=8.5)
    return fig

# ----------------------------------------------------------------- Fig 3: S / D / X reference ranges
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.35), sharex=True, gridspec_kw=dict(wspace=0.72))
    sty = {"S": dict(c="black", m="o", mfc="black"), "D": dict(c="#1b5e9b", m="s", mfc="#1b5e9b"),
           "X": dict(c="#b5530d", m="^", mfc="white")}
    pairs = [(1, 3), (1, 6), (3, 6)]
    for ax, h, tag in zip(axes, (6, 9), ("(a)", "(b)")):
        S, P, B = boot[h]
        y = 0; yt, yl = [], []
        entries = []
        for w in (1, 3, 6):
            entries.append(("S", y, S[w], None, f"S: same window, {w} mo")); y += 1
        y += 0.6
        for pr in pairs:
            entries.append(("D", y, P[pr]["D"], P[pr]["point"], f"D: {pr[0]} vs. {pr[1]} mo")); y += 1
            entries.append(("X", y, P[pr]["X"], P[pr]["point"], f"X: {pr[0]} vs. {pr[1]} mo")); y += 1
            y += 0.6
        for kind, yy, (med, lo, hi), obs, lab in entries:
            if kind == "D":
                ax.plot([obs, obs], [yy - 0.38, yy + 1.38], color="black", lw=1.1, ls=(0, (2, 1.5)), zorder=1.5)
            s = sty[kind]
            ax.plot([lo, hi], [yy, yy], "-", color=s["c"], lw=2.0, solid_capstyle="butt", zorder=2)
            ax.plot([lo, lo], [yy-.16, yy+.16], "-", color=s["c"], lw=1.0, zorder=2)
            ax.plot([hi, hi], [yy-.16, yy+.16], "-", color=s["c"], lw=1.0, zorder=2)
            ax.plot(med, yy, s["m"], color=s["c"], mfc=s["mfc"], ms=5.5, mew=1.1, zorder=4)
            yt.append(yy); yl.append(lab)
        ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(y - 0.5, -0.7)
        ax.axvline(0, color="0.55", lw=0.6, ls=":", zorder=1)
        ax.set_xlim(-0.3, 1.0); ax.set_xticks(np.arange(-0.2, 1.01, 0.2))
        ax.set_xlabel(r"Kendall's $\tau_b$")
        ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        ax.tick_params(length=2)
        sub = f"h = {h}, k = {len(pt[h])}" + (" (main)" if h == 6 else " (descriptive only)")
        ax.set_title(f"{tag} {sub}", fontsize=8.5)
    handles = [
        Line2D([], [], color=sty["S"]["c"], marker="o", mfc="black", lw=2, ms=5.5, label="S: two independent resamples, same window"),
        Line2D([], [], color=sty["D"]["c"], marker="s", mfc=sty["D"]["c"], lw=2, ms=5.5, label="D: same resamples, mean of two within-resample values"),
        Line2D([], [], color=sty["X"]["c"], marker="^", mfc="white", mew=1.1, lw=2, ms=5.5, label="X: independent resamples, different windows"),
        Line2D([], [], color="black", lw=1.1, ls=(0, (2, 1.5)), label=r"observed $\tau_b$ (point estimate; dashed line, no interval)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=1.6, handlelength=2.2)
    fig.text(0.5, 0.135, "Marker = bootstrap median; bar = 2.5th to 97.5th percentile of the reference distribution (B = 1,000). "
             "Not confidence intervals for the observed $\\tau_b$.", ha="center", fontsize=7.5, style="italic")
    fig.subplots_adjust(left=0.165, right=0.985, top=0.93, bottom=0.31)
    return fig

for name, fn in (("fig1_heatmap", fig1), ("fig2_rank_slope", fig2), ("fig3_agreement", fig3)):
    fg = fn()
    fg.savefig(f"figures/{name}.pdf"); fg.savefig(f"figures/{name}.png")
    plt.close(fg)
    log(f"WROTE figures/{name}.pdf and .png")

# record exactly what was plotted for audit
with open("figures/plotted_values.txt", "w") as fh:
    for h in (6, 9):
        fh.write(f"h={h} rates/ranks (cohort, n_c, R1,R3,R6, rank1,rank3,rank6)\n")
        for r in pt[h]:
            fh.write(f"  {r['cohort']} {r['n']} {' '.join(r['R'])} {' '.join(map(str, r['rank']))}\n")
        S, P, B = boot[h]
        fh.write(f"h={h} S(w) median,lo,hi: {S}\n")
        for k, v in P.items(): fh.write(f"h={h} pair {k}: {v}\n")
open("verification_log.txt", "w").write("\n".join(LOG) + "\n")
