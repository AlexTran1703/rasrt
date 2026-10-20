"""Report of the FreeRTOS study (the letter's figures and tables) -> results/rtos_T7ms/.
    table1_constraints  TABLE I. Schedule selection by RAS and RAS-RT: interference, prediction, criteria, output
    fig1                Fig. 1 measured execution of IR-CNN under FreeRTOS (U_H = 20 %, window 6): (a) during
                        acquisition (input samples, FW idle, RAS / RAS-RT segment calls, tau_H; broken time axis);
                        (b) after the last sample (T_post of FW, fixed RAS, RAS-RT with the preempting jobs)
    fig2                Fig. 2 maximum observed T_post vs U_H (0-60 %) of FW, fixed RAS and RAS-RT, per network
    table2_rtos         TABLE II. reserved SRAM, CNN CPU time per window C(s) and T_post of FW / RAS (bare metal)
                        and RAS-RT at U_H = 20 / 40 %
    captions.md, README.md
Inputs: results/rasms (select_bare.py: bare-metal measurements, RAS candidates and selection), results/rtos_T7ms/rtos.json
(rtos_study.py), results/rtos_T7ms/rtos_aware.json (rtos_select.py), <build>/rtos/fig1 (trace_fig1.py 6).
Usage: python rasrt/python/scripts/report.py [rtos.json] [output dir]
"""
import json
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.legend_handler  # noqa: E402
import matplotlib.ticker  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ras.config import BARE as RES, REPO as ROOT, RTOS, RTOS_BUILD  # noqa: E402
SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else RTOS / "rtos.json"
DST = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else SRC.parent
DST.mkdir(parents=True, exist_ok=True)
R = json.loads(SRC.read_text())
AW = json.loads((SRC.parent / "rtos_aware.json").read_text()) if (SRC.parent / "rtos_aware.json").exists() else {}
M = json.loads((RES / "measured.json").read_text())
E3 = json.loads((RES / "e3.json").read_text())
D, B = 30, 48
U_SWEEP, T_H, NW, PH = tuple(R["u_sweep"]), R["t1_ms"], R["n_win"], R.get("phase_step_ticks", 0)
BUILDS = RTOS_BUILD                                                          # FreeRTOS build folders
umax = f"{100 * max(U_SWEEP):.0f}"
TAG, SUF = "", ""
NETS = {"A0": "IR-CNN", "A1": "ResNet1D", "A2": "Inception1D"}
C_FW, C_RAS, C_HP, C_IN = "#eb6834", "#2a78d6", "#8e6cc7", "#9fb3c8"
SEGC = ["#d6e4f7", "#d3efe3", "#e3dcf2", "#f6e8c9", "#f7dcd0"]
RAS = {v: E3[v]["selected"]["30"] for v in NETS}
bare = lambda v, lab: M[v].get(lab) or E3[v]["measured"][lab]  # noqa: E731
def esc(x):
    """Markdown/Unicode cell text -> LaTeX: special characters, **bold**, symbols."""
    t = re.sub(r"([%&#_])", r"\\\1", str(x).replace("R_H^max", "\x00"))
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    for a_, b_ in (("≤", r"$\le$"), ("†", r"$^\dagger$"), ("‡", r"$^\ddagger$"), ("§", r"$^\S$"), ("τ", r"$\tau$"),
                   ("Γ", r"$\Gamma$"), ("→", r"$\rightarrow$")):
        t = t.replace(a_, b_)
    return t.replace("\x00", r"$R_H^{\max}$")


def table(name, cap, head, rows, notes, head_tex=None, star=False, notes_tex=None, label=None):
    md = [f"**{TAG}{cap}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    (DST / f"{name}.md").write_text("\n".join(md + ["| " + " | ".join(r) + " |" for r in rows] + ["", notes]) + "\n",
                                    encoding="utf-8")
    env = "table*" if star else "table"
    tex = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{esc(re.sub(r'^TABLE [IVX]+\. ', '', cap))}}}"] + ([f"\\label{{{label}}}"] if label else []) + [
           "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{" + "l" * len(head) + "}", "\\hline",
           " & ".join(head_tex or map(esc, head)) + " \\\\", "\\hline"]
    tex += [" & ".join(map(esc, r)) + " \\\\" for r in rows] + ["\\hline", "\\end{tabular}",
                                                               f"\\par\\smallskip\\noindent\\parbox{{\\{'textwidth' if star else 'columnwidth'}}}"
                                                               f"{{\\scriptsize {notes_tex or esc(notes)}}}", f"\\end{{{env}}}"]
    (DST / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


# =========================================================== TABLE I
vec = lambda lab: "(" + ",".join(lab[3:].split(".")) + ")"  # noqa: E731
LS = [len(AW[v]["0.2"]["selected"][3:].split(".")) for v in NETS]
NCAND = (56, 165, 84)                                        # candidate schedules (ras.predictor)
UT = sorted(float(u) for u in AW["A0"] if float(u) <= 0.6)  # U_H with RAS-RT selections (plotted range)
NONE = [u for u in UT if not any(AW[v][f"{u:g}"].get("selected") for v in NETS)]
RT_IR = "; ".join(f"${vec(AW['A0'][f'{u:g}']['selected'])}$ at {100 * u:.0f}\\%" for u in UT if u not in NONE)
CAP1 = "Schedule selection by RAS and RAS-RT: interference, prediction, criteria, and output"
RC = r">{\raggedright\arraybackslash}"
T1 = [
      ("Interference", rf"none & $\tau_H$: $T_H = D_H = {T_H:g}$~ms, $P_H > P_{{\mathrm{{CNN}}}}$, "
                       rf"$U_H = 0$--{100 * max(UT):.0f}\%"),
      ("Prediction (before building)", None),
      ("CPU", rf"\multicolumn{{2}}{{{RC}p{{6.5cm}}@{{}}}}{{$\widehat{{C}}(s)$: sum of calibrated graph-call and operator costs over all segment jobs (unchanged under preemption)}}"),
      ("Response", r"$\widehat{T}_{\mathrm{post}}$ & $\widehat{R}_{\mathrm{post}}$, worst of 14 phases of $\tau_H$"),
      ("Memory", r"$\widehat{M}(s)$ & $\widehat{M}(s) + M_{\mathrm{RTOS}}$ ($M_{\mathrm{RTOS}} = 3.1$~KiB)"),
      ("Selection criteria", None),
      ("Deadline", rf"$(1+\epsilon)\widehat{{T}}_{{\mathrm{{post}}}} \le {D}$~ms & "
                   rf"$(1+\epsilon)\widehat{{R}}_{{\mathrm{{post}}}} \le {D}$~ms"),
      ("SRAM", rf"$\widehat{{M}}(s) \le {B}$~KiB & $\widehat{{M}}(s) + M_{{\mathrm{{RTOS}}}} \le {B}$~KiB"),
      ("Objective", rf"\multicolumn{{2}}{{l@{{}}}}{{$\min_s \widehat{{C}}(s)$, \quad $\epsilon = {AW['margin']:.2f}$}}"),
      ("Output", None),
      ("Schedule", rf"one $\mathbf{{m}}$ per network; IR: ${vec(R['A0']['S1 RAS 0.2']['label'])}$ & "
                   rf"one $\mathbf{{m}}$ per $U_H$; IR: {RT_IR}"
                   + "".join(f"; none at {100 * u:.0f}\\%" for u in NONE))]
md = [f"**{TAG}TABLE I. {CAP1}**", "", "| | RAS (bare metal) | RAS-RT (FreeRTOS) |", "|---|---|---|"]
tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize", rf"\caption{{{CAP1}}}", r"\label{tab:constraints}",
       r"\setlength{\tabcolsep}{3pt}",
       rf"\begin{{tabular}}{{@{{}}{RC}p{{1.45cm}}{RC}p{{3.05cm}}{RC}p{{3.35cm}}@{{}}}}", r"\hline",
       r" & RAS (bare metal) & RAS-RT (FreeRTOS) \\", r"\hline"]
for k_, v_ in T1:
    if v_ is None:
        md.append(f"| **{k_}** | | |")
        tex.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{{k_}}}}} \\")
    else:
        md.append(f"| {k_} | {v_} |")
        tex.append(rf"{k_} & {v_} \\")
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
(DST / "table1_constraints.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(DST / "table1_constraints.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

ABBR = {"A0": "IR", "A1": "Re", "A2": "In"}
# =========================================================== Fig. 1
# Measured FreeRTOS traces of one window (IR-CNN, U_H = 20 %, window 6; scripts/fig1_trace.py 6 FW RAS RAS-RT):
# (a) during acquisition, broken axis: TIM2 samples, FW idle, one tick per segment call of RAS and RAS-RT, tau_H;
# (b) after the last sample: T_post of FW, fixed RAS and RAS-RT, with the periodic-task jobs preempting the CNN.
from matplotlib.transforms import blended_transform_factory as blend  # noqa: E402
L_RT, L_RAS = AW["A0"]["0.2"]["selected"], R["A0"]["S1 RAS 0.2"]["label"]
TRC = {k: json.loads((BUILDS / "fig1" / f"{lab}_u0.2_w6" / "measured.json").read_text())
       for k, lab in (("FW", "E0"), ("RAS", L_RAS), ("RAS-RT", L_RT))}
MV = {k: [int(x) for x in lab[3:].split(".")] for k, lab in (("RAS", L_RAS), ("RAS-RT", L_RT))}
TW, WIN, NS = 6, 10.24, len(MV["RAS-RT"])
C_GR, C_MISS, C_TIM = "#8a8a8a", "#c0392b", "#5f7a99"
SEGF = ["#4f8fdd", "#2fb886", "#e377c2", "#e0a93f", "#ef7b52"]     # S2 pink: distinct from the periodic task
FS_T, FS_B, FS_H = 8.0, 7.5, 8.5
T0, TE = 0.173, 0.91                                         # figure x where both timelines start / end
LABX = T0 - 0.015                                            # shared right edge of the row labels

fig = plt.figure(figsize=(3.5, 3.6))
LEG, TT = 0.07, 0.052                                        # legend strip, panel-heading height (figure fraction)
HP = (1 - LEG) / 2                                           # (a) and (b) each take exactly half of the rest

# (a) during acquisition, broken time axis: window start (-10.24 .. -9 s) | cut | last 2 s (-2 .. 0 s);
#     one row per schedule, one tick per measured segment call (colour = segment); tau_H at its nominal 7 ms period
aa = fig.add_axes([0.0, LEG + HP, 1.0, HP - TT])
fig.text(0.0, 1.0, "(a)", fontsize=FS_H + 0.5, fontweight="bold", va="top")
fig.text(0.075, 1.0, "During acquisition (10.24 s)", fontsize=FS_H, fontweight="bold", va="top")
L0, L1, R0, GAP = -WIN, -9.0, -2.0, 0.45                     # shown spans and the drawn gap between them


def fx(t):                                                   # real time (s) -> drawn x; None inside the cut
    return t if t >= R0 else (t - L1 + R0 - GAP if t <= L1 else None)


def ticks(ts, y, h, **kw):
    xs = [fx(t) for t in ts if L0 <= t <= 0 and fx(t) is not None]
    aa.vlines(xs, y - h / 2, y + h / 2, **kw)


Y = {"Input": 5.0, "FW": 4.0, "RAS": 3.0, "RAS-RT": 2.0, "tau": 1.0}
XL, XC = fx(L0), (fx(L1) + R0) / 2                           # left edge, centre of the cut
ticks([-i / 250 for i in range(round(WIN * 250))], Y["Input"], 0.44, color=C_TIM, lw=0.3)   # TIM2 ISR, 1 sample / 4 ms
aa.text((XL + 0) / 2, Y["FW"], "buffers input (CNN not executing)", fontsize=FS_B, color=C_FW, ha="center",
        va="center", style="italic")
for k in ("RAS", "RAS-RT"):
    for j in range(NS):                                      # deeper segments drawn on top
        ticks([x0 for jj, x0, _ in TRC[k]["cnn_trace"] if jj == j], Y[k], 0.52, color=SEGF[j], lw=0.8)
ph = min(a_ for a_, _ in TRC["RAS-RT"]["load_trace"]) % (T_H / 1e3)
ticks([-WIN + ph + i * T_H / 1e3 for i in range(int(WIN / (T_H / 1e3)) + 1)], Y["tau"], 0.44, color=C_HP, lw=0.3)
for k in ("Input", "RAS", "RAS-RT", "tau"):                 # cut marker on every lane
    aa.text(XC, Y[k], "…", fontsize=FS_B, ha="center", va="center", color="#555555")
for k, lab in (("Input", "Input"), ("FW", "FW"), ("RAS", "RAS"), ("RAS-RT", "RAS-RT"), ("tau", r"$\tau_H$")):
    aa.text(LABX, Y[k], lab, fontsize=FS_T, fontweight="bold", ha="right", va="center",
            transform=blend(fig.transFigure, aa.transData))
YA = 0.3
for lo, hi in ((L0, L1), (R0, 0.0)):                         # broken axis with // marks
    aa.plot([fx(lo), fx(hi)], [YA, YA], color="#0b0b0b", lw=0.8)
for xb in (fx(L1), R0):
    aa.plot([xb - 0.05, xb + 0.05], [YA - 0.12, YA + 0.12], color="#0b0b0b", lw=0.8)
for t, lab in ((L0, "−10.24"), (-9.0, "−9"), (-2.0, "−2"), (-1.0, "−1"), (0.0, "0")):
    aa.plot([fx(t), fx(t)], [YA, YA + 0.12], color="#0b0b0b", lw=0.7)
    aa.text(fx(t), YA - 0.12, lab, fontsize=FS_B, ha="center", va="top")
aa.text(0.2, YA - 0.12, "s", fontsize=FS_B, ha="left", va="top")
aa.plot([0, 0], [YA, Y["Input"] + 0.45], color="#0b0b0b", lw=0.7, ls="--")
aa.text(0, Y["Input"] + 0.5, "last sample", fontsize=FS_B, ha="center", va="bottom")
k_ = (TE - T0) / -XL                                         # window start -> T0, last sample -> TE
aa.set_xlim(XL - T0 / k_, XL + (1 - T0) / k_)
aa.set_ylim(-0.8, Y["Input"] + 0.62)
aa.axis("off")

# (b) after the last sample -------------------------------------------------------------------------
ab = fig.add_axes([0.0, LEG, 1.0, HP - TT])
fig.text(0.0, LEG + HP, "(b)", fontsize=FS_H + 0.5, fontweight="bold", va="top")
fig.text(0.075, LEG + HP, r"After the last sample, $U_H$ = 20%", fontsize=FS_H, fontweight="bold", va="top")
XEND, FW_CUT, HR = 44.0, 40.0, 0.42
ROW = {"FW": 0.45, "RAS": -0.6, "RAS-RT": -1.65}
for mode in ("FW", "RAS", "RAS-RT"):
    m = TRC[mode]
    jobs = [(a * 1e3, b * 1e3) for a, b in m["load_trace"] if b * 1e3 > 0]
    cnn = [(j, a * 1e3, b * 1e3) for j, a, b in m["cnn_trace"] if b * 1e3 > 0 and (mode == "FW" or 0 <= j < NS)]
    y = ROW[mode]

    def box_(x0, x1, col, **kw):
        x0, x1 = max(x0, 0.0), min(x1, XEND)
        if x1 > x0:
            ab.add_patch(plt.Rectangle((x0, y - HR / 2), x1 - x0, HR, fc=col, lw=0, **kw))

    def cut(t0, t1):
        pcs = [(t0, t1)]
        for j0, j1 in jobs:
            pcs = [q for lo, hi in pcs for q in ((lo, min(hi, j0)), (max(lo, j1), hi)) if q[1] > q[0]]
        return pcs

    for x0, x1 in jobs:
        box_(x0, x1, C_HP)
    if mode == "FW":
        fw_end = max(x1 for _, _, x1 in cnn)
        for lo, hi in cut(min(x0 for _, x0, _ in cnn), min(fw_end, FW_CUT)):
            box_(lo, hi, C_FW)
        ab.add_patch(plt.Rectangle((FW_CUT, y - HR / 2), XEND - FW_CUT, HR, fc="none", ec=C_FW, hatch="//////", lw=0))
    else:
        for j, x0, x1 in cnn:
            for lo, hi in cut(x0, x1):
                box_(lo, hi, SEGF[j])
    tp, tv = min(max(x1 for _, _, x1 in cnn), XEND), m["t_post_win_ms"][TW]   # T_post bracket (FW: runs off the axis)
    yb = y - HR / 2 - 0.12
    ab.plot([0, 0, tp] + ([tp] if tp < XEND else []), [yb + 0.08, yb, yb] + ([yb + 0.08] if tp < XEND else []),
            color="#0b0b0b", lw=0.7)
    if tp >= XEND:
        ab.annotate("", (XEND + 1.5, yb), (XEND - 0.5, yb), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#0b0b0b",
                                                                             mutation_scale=6))
    ab.text(min(tp, D) / 2 if tp >= XEND else tp / 2, yb - 0.05, rf"$T_{{\mathrm{{post}}}}$ = {tv:.1f} ms",
            fontsize=FS_B, ha="center", va="top")
    ab.text(LABX, y, mode, fontsize=FS_T, fontweight="bold", ha="right", va="center",
            transform=blend(fig.transFigure, ab.transData))
YAX = 1.05
ab.plot([0, XEND], [YAX, YAX], color="#0b0b0b", lw=0.8)
for t in (0, 10, 20, 30, 40):
    ab.plot([t, t], [YAX, YAX - 0.08], color="#0b0b0b", lw=0.7)
    ab.text(t, YAX + 0.07, f"D = {D} ms" if t == D else (f"{t} ms" if t == 40 else f"{t}"), fontsize=FS_B,
            color=C_MISS if t == D else "#0b0b0b", ha="center", va="bottom")
ab.plot([0, 0], [-2.05, YAX], color="#0b0b0b", lw=0.7, ls="--", zorder=4)
ab.plot([D, D], [-2.05, YAX], color=C_MISS, lw=0.9, ls="--", zorder=4)
k_ = (TE - T0) / XEND                                        # t = 0 -> T0, axis end -> TE (same length as (a))
ab.set_xlim(-T0 / k_, (1 - T0) / k_)
ab.set_ylim(-2.35, 1.3)
ab.axis("off")

rect = lambda c: plt.Rectangle((0, 0), 1, 1, fc=c, lw=0)  # noqa: E731
fig.legend([rect(C_FW), tuple(rect(c) for c in SEGF), rect(C_HP)], ["FW CNN", r"Segments $S_0$–$S_4$", "Periodic task"],
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0)}, fontsize=FS_B, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False, handlelength=1.6, columnspacing=1.0,
           handletextpad=0.4)
fig.savefig(DST / f"fig1{SUF}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(DST / f"fig1{SUF}.pdf", bbox_inches="tight", pad_inches=0.02)
print("fig1 printed size (in):", [round(x, 2) for x in fig.get_tightbbox(fig.canvas.get_renderer()).size])
plt.close(fig)

# =========================================================== Fig. 2
U2A = [u for u in U_SWEEP if u <= 0.6]                       # U_H up to 60 %
fig, axs = plt.subplots(1, 3, figsize=(3.5, 2.0), sharey=True)
fig.subplots_adjust(left=0.14, right=0.995, bottom=0.21, top=0.80, wspace=0.08)
for i, (v, net) in enumerate(NETS.items()):
    ax = axs[i]
    for s_, col, mk in (("FW", C_FW, "s"), ("RAS", C_RAS, "^")):
        pts = [(100 * u, R[v][f"S1 {s_} {u:g}"]["t_post_max_ms"]) for u in U2A if f"S1 {s_} {u:g}" in R.get(v, {})]
        ax.plot(*zip(*pts), color=col, marker=mk, ms=4.2, lw=1.2)
    if AW.get(v):                                          # RAS-RT: re-selected for each U_H
        sel = sorted((float(u), e) for u, e in AW[v].items() if e.get("selected") and e.get("measured"))
        pts = [(0, R[v]["S1 RAS 0"]["t_post_max_ms"])] + [(100 * u, e["measured"]["t_post_max_ms"]) for u, e in sel]
        ax.plot(*zip(*pts), color="#1a9850", marker="D", ms=4.0, lw=1.2)
        hi = sorted((float(u), e) for u, e in AW[v].items() if not e.get("selected") and e.get("measured")
                    and float(u) <= max(U2A))
        for u, e in hi:                                    # no margin-feasible schedule; tested configuration
            ax.plot(100 * u, e["measured"]["t_post_max_ms"], lw=0, marker="D", ms=4.6, mfc="white", mew=1.2,
                    color="#1a9850", zorder=3)
    ax.axhline(D, color="#c0392b", lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(15, 1000)
    ax.set_yticks([20, 30, 100, 300, 1000], ["20", "30", "100", "300", "1000"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlim(-9, 69)
    ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    ax.set_title(ABBR[v], fontsize=8.5, fontweight="bold", pad=3)
axs[0].set_ylabel(r"max. $T_{\mathrm{post}}$ (ms)", fontsize=8, labelpad=1)
axs[0].get_yticklabels()[1].set_color("#c0392b")             # 30 ms = deadline D
fig.text(0.57, 0.025, r"$U_H$ (%)", fontsize=8, ha="center", va="bottom")
L2 = lambda col, mk, ls="-", fc=None, ms=4.2: matplotlib.lines.Line2D([], [], color=col, marker=mk, ls=ls, lw=1.2,  # noqa: E731
                                                                      ms=ms, mfc=fc or col, mew=1.2)
fig.legend([L2(C_FW, "s"), L2(C_RAS, "^"), L2("#1a9850", "D", ms=4.0), L2("#1a9850", "D", "none"),
            L2("#1a9850", "D", "none", "white", 4.6)],
           ["FW", "RAS", "RAS-RT", "Feasible selection", "No feasible schedule"], fontsize=6.4, ncol=5,
           loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2, columnspacing=0.6,
           handletextpad=0.25, borderaxespad=0.1)
fig.savefig(DST / f"fig2{SUF}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(DST / f"fig2{SUF}.pdf", bbox_inches="tight", pad_inches=0.02)
print("fig2 printed size (in):", [round(x, 2) for x in fig.get_tightbbox(fig.canvas.get_renderer()).size])
plt.close(fig)

# =========================================================== TABLE II
def flash_kib(sec):
    return (sum(sec.get(k, 0) for k in (".isr_vector", ".text", ".rodata", ".ARM.exidx", ".data")) - 3 * 2560) / 1024


rows = []
bold = lambda x: f"**{x:.1f}**" if x <= D else f"{x:.1f}"  # noqa: E731
ABBR = {"A0": "IR", "A1": "Re", "A2": "In"}
fl, sr, cp = [], [], []                                     # FreeRTOS overhead at U_H = 0 (same RAS schedule)
rh = {"0.2": [], "0.4": []}                                 # periodic task: R_H^max per network, misses / jobs
miss, njob = 0, set()
for v, net in NETS.items():
    lab = RAS[v]
    fw, b0 = M[v]["E0"], bare(v, lab)
    d = BUILDS / v / "S1" / f"{lab}_u0"
    m = json.loads((d / "measured.json").read_text())
    cpu_r = struct.unpack_from(f"<{NW}I", (d / "results.bin").read_bytes(), 4 * (9 + NW))
    fl.append(flash_kib(m["sections"]) - flash_kib(b0["sections"]))
    sr.append((m["sram"] - b0["sram"]) / 1024)
    cp.append(sum(cpu_r) / len(cpu_r) / 1e5 - b0["cpu_ms"])
    over = fw["sram"] > B * 1024
    rows.append([ABBR[v], "FW", f"{fw['sram'] / 1024:.1f}" + ("†" if over else ""), f"{fw['cpu_ms']:.1f}",
                 bold(max(fw["t_post_win_ms"]))])
    rows.append(["", "RAS", f"{b0['sram'] / 1024:.1f}", f"{b0['cpu_ms']:.1f}", bold(max(b0["t_post_win_ms"]))])
    for u in ("0.2", "0.4"):
        e = AW.get(v, {}).get(u, {})
        if not (e.get("selected") and e.get("measured")):
            continue
        ma = json.loads((BUILDS / v / "S1_aware" / f"{e['selected']}_u{u}" / "measured.json").read_text())
        cpu = f"{e['bare_cpu_ms']:.1f}" if e.get("bare_cpu_ms") else f"~{e['pred_cpu_ms']:.0f}§"
        rh[u].append(ma["load_rmax_ms"])
        miss += ma["load_miss"]
        njob.add(ma["load_n"])
        rows.append(["", f"RT{100 * float(u):.0f}", f"{ma['sram'] / 1024:.1f}", cpu, bold(ma["t_post_max_ms"])])
nj = f"{min(njob):,}" if len(njob) == 1 else f"{min(njob):,}-{max(njob):,}"
r20, r40 = ("/".join(f"{x:.3f}" for x in rh[u]) for u in ("0.2", "0.4"))
NOTE2 = (r"FW/RAS: bare-metal; RT20/RT40: RAS-RT selected and evaluated under FreeRTOS at $U_H=20\%/40\%$. "
         r"$C(s)$ is isolated CNN CPU time per 10.24-s window; $T_{\mathrm{post}}$ is the maximum observed response "
         rf"(3 bare-metal or {NW} FreeRTOS windows). SRAM includes static storage and reserved stack (plus RTOS objects "
         rf"where applicable). Bold indicates $T_{{\mathrm{{post}}}}\le D={D}$~ms. $^\dagger$ Exceeds the {B}-KiB SRAM "
         r"budget; measured using 128 KiB as a reference.")
table("table2_rtos", "TABLE II. Reserved SRAM, CNN CPU time per window, and post-window response for FW, RAS, and "
      "RAS-RT", ["Net", "Sch.", "SRAM (KiB)", "C(s) (ms)", "T_post (ms)"], rows,
      "FW/RAS: bare-metal; RT20/RT40: RAS-RT selected and evaluated under FreeRTOS at U_H = 20/40 %. C(s): isolated "
      f"CNN CPU time per 10.24 s window; T_post: maximum observed response (3 bare-metal or {NW} FreeRTOS windows). "
      "SRAM: static storage + reserved stack (+ RTOS objects). Bold: T_post <= D. † exceeds the 48 KiB budget "
      "(measured on 128 KiB as a reference).",
      head_tex=["Net", "Sch.", "SRAM (KiB)", r"$C(s)$ (ms)", r"$T_{\mathrm{post}}$ (ms)"], notes_tex=NOTE2,
      label="tab:rtos")
(DST / "captions.md").write_text(
    "**Fig. 1.** Execution of IR-CNN under FreeRTOS ($T_H$ = 7 ms, $U_H$ = 20 %). (a) During acquisition, FW buffers "
    "the input while RAS and RAS-RT execute CNN segments incrementally (measured calls; broken time axis). (b) Under "
    "periodic-task preemption (purple), FW and the bare-metal-selected RAS schedule miss the 30-ms post-window "
    "deadline, whereas RAS-RT meets it through schedule reselection. FW execution is truncated.\n\n"
    "**Fig. 2.** Maximum observed post-window response under FreeRTOS versus higher-priority utilization $U_H$ for "
    "three CNNs. FW and RAS retain fixed schedules, whereas RAS-RT reselects schedules for each load. The dotted line "
    "marks the 30-ms deadline. Open diamonds indicate tested configurations for which no schedule satisfied the "
    "margin-adjusted prediction criterion.\n\n"
    "**Table I.** Schedule selection by RAS and RAS-RT: interference, prediction, criteria, and output.\n\n"
    "**Table II.** Reserved SRAM, CNN CPU time per window, and post-window response for FW, RAS, and RAS-RT.\n",
    encoding="utf-8")
(DST / "README.md").write_text(
    f"FreeRTOS report (FW, RAS, RAS-RT), generated by rasrt/python/scripts/report.py from {SRC.relative_to(ROOT)}: synthetic "
    f"periodic task T_H = {T_H:g} ms, U_H = {', '.join(f'{100 * u:g}' for u in U_SWEEP)} % tested (Fig. 2 and "
    f"Table I: 0-{100 * max(UT):.0f} %), {NW} windows per "
    "configuration. All tables / Fig. 2 values are board measurements; Fig. 1 uses TRACE_LOAD builds of window 6 "
    "(runs/rtos/fig1; tau_H and TIM2 in Fig. 1(a) drawn at their nominal periods). RAS (fixed) = schedule selected for D without interference"
    + ("; RAS-RT = re-selected per U_H (rtos_aware.json).\n" if AW else ".\n"),
    encoding="utf-8")
print((DST / "table2_rtos.md").read_text(encoding="utf-8"))
print("files:", sorted(p.name for p in DST.iterdir()))
