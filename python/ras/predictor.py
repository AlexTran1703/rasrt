r"""Dispatcher simulation of a schedule before building its firmware: release times, CPU time, post-window
response, retained buffers and reserved SRAM; with an optional higher-priority periodic task (RAS-RT).

Scope: one CPU running segment jobs sequentially, f_s = 250 Hz, fixed int8 network, non-overlapping 10.24 s
windows, on-device top-k window reduction; the simulation follows the dispatcher of firmware/src/stream.c.
RAS / RAS-RT schedules are full-depth (d = L) and released on arrival (rho = 1), i.e. a chunk vector m;
the general form below also describes full-window inference (FW, d = 0).

Schedule c = (d, m, rho):
    d    segmentation depth: segments 0..d-1 of the op chain (cut before every conv with k > 1) are
         streamed chunk by chunk, ops from segment d on run as one whole-window tail graph after the
         window (d = 0: full-window reference; d = S: no tail). Depths whose cut would split a
         residual are excluded.
    m    embeddings per acquisition chunk: segment 0 receives chunks of 16 m samples
    rho  release policy: 1 = a chunk is released when its last sample arrives, 0 = after acquisition

Dependencies: segment j is a synchronous-dataflow actor that consumes new_j rows per run (plus its
k_j - s_j stream-buffer rows) and produces new_j / s_j rows. Output row p of a conv (stride s,
kernel k, left padding p_L) depends on input rows [p s - p_L, p s - p_L + k - 1]; propagating these
intervals back to segment 0 gives each job's latest required physical sample u_q (padding rows are
not samples). The dispatcher runs a job as soon as its rows exist; its release time is the arrival
of the chunk containing u_q (rho = 1) or the end of acquisition (rho = 0).

Timing (a(n) = (n + 1) / f_s, T_acq = N / f_s): jobs in dispatcher order pi,
    t_start(q) = max(r_q, t_finish(previous CPU job)),  t_finish(q) = t_start(q) + C_q
    T_CPU = sum_q C_q,  T_post = t_finish(score) - T_acq
C_q comes from a profile-calibrated cost model (CostModel): per graph call + per operator family
(alpha_g + beta_g MACs + gamma_g bytes) + copies, feed, window reduction. Three consecutive windows are
simulated, so carry-over between windows and input-ring occupancy are part of the prediction.

Memory: retained capacity of every stored stream e is b_e max_t [P_e(t) - R_e(t)] (produced minus
retired rows; a residual row is retired only after the addition that reads it). Deployment SRAM =
fixed statics + stream buffers + residual buffers + tail input + input ring + arena (the ST planner's
activation size, max over the graphs) + network contexts + stack.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import EMB_STRIDE, FS, MCU_HZ, M_SET, N_EMB, WIN
from .segments import plan, segments

N_REPLAY = 3


@dataclass(frozen=True)
class Cand:
    d: int
    m: object            # tuple: per-segment chunk vector m_j (RAS); int: one size for all segments
    rho: int

    @property
    def label(self):                     # result keys: "E0" = FW, "E3-<m_0>.<m_1>..." = chunk vector
        if self.d == 0:
            return "E0"
        if isinstance(self.m, tuple):
            return "E3-" + ".".join(map(str, self.m)) + ("" if self.rho else "-post")
        return f"d{self.d}-m{self.m}-{'arr' if self.rho else 'post'}"


def plan_rates(segs, ms):
    """Per-segment chunk sizes: segment j emits m_j embeddings' worth of rows per call, i.e.
    m_j * prod(strides after j) output rows, in embeddings (16 samples) also at partial depth;
    one m for all reproduces stream.plan(16 m)."""
    new, after = [], EMB_STRIDE // int(np.prod([sg["s"] for sg in segs]))
    for sg, m in zip(reversed(segs), reversed(ms)):
        new.append(m * after * sg["s"])
        after *= sg["s"]
    return new[::-1]


def candidate_schedules(q):
    """Candidate schedules (chunk vectors) of RAS / RAS-RT: every segment streamed and released on arrival,
    m_j in M_SET non-decreasing with depth: C(L + 3, L) vectors for L segments."""
    from itertools import combinations_with_replacement
    S = len(segments(q))
    return [Cand(S, ms, 1) for ms in combinations_with_replacement(M_SET, S)]


# ----------------------------------------------------------------------------- geometry
def geometry(q, c):
    S = segments(q)
    P = WIN + sum(q.pad)
    cut = 0 if c.d == 0 else (S[c.d]["first"] if c.d < len(S) else len(q.ops))
    segs = segments(q, 0, cut) if cut else []
    if not segs:
        new = []
    elif isinstance(c.m, tuple):
        new = plan_rates(segs, c.m)
    else:
        new = plan(segs, EMB_STRIDE * c.m)
    lens, n = [], P
    for j, sg in enumerate(segs):
        n = (n + (sum(sg["pad"]) if j else 0) - sg["k"]) // sg["s"] + 1
        lens.append(n)
    g = {"segs": segs, "new": new, "lens": lens, "cut": cut, "P": P,
         "nin": [new[j] + sg["tail"] for j, sg in enumerate(segs)],
         "nout": [new[j] // sg["s"] for j, sg in enumerate(segs)],
         "raw_chunk": new[0] if segs else 2 * WIN // 16}
    assert WIN % g["raw_chunk"] == 0, "chunk must divide the window (m | N_EMB)"
    g["raw_ring"] = (2 * g["raw_chunk"] + 256) if c.rho else (WIN + 256)
    if cut < len(q.ops):
        g["tail"] = {"first": cut, "L": lens[-1] if segs else P, "cin": segs[-1]["c_out"] if segs else 1}
    res = sorted({sg["res_out"] for sg in segs if sg["res_out"] is not None})
    g["res"] = [{"src": r, "ch": int(q.ops[r].w.shape[0]),
                 "prod": next(j for j, s in enumerate(segs) if s["res_out"] == r),
                 "cons": next(j for j, s in enumerate(segs) if s["res_in"] == r)} for r in res]
    return g


def graph_ops(q, first, last, n_in, valid=True, res_out=False):
    """Operator features of one graph: [(family, MACs, bytes)], output rows."""
    feats, n = [], n_in
    for i in range(first, last):
        op = q.ops[i]
        k, s, gr, cout = op.w.shape[2], op.stride, op.groups, op.w.shape[0]
        cin = op.w.shape[1] * gr
        n_pad = n + (0 if valid else sum(op.pad))
        n_out = (n_pad - k) // s + 1
        if n_pad > n:
            feats.append(("pad", n_pad * cin, n_pad * cin + n * cin))
        fam = "dw" if gr > 1 else "pw" if k == 1 else "conv1" if cin == 1 else "conv"
        feats.append((fam, n_out * k * (cin // gr) * cout, n_pad * cin + n_out * cout))
        if op.res_src is not None:
            feats.append(("add", n_out * cout, 3 * n_out * cout))
        n = n_out
    if res_out:
        r = next(i for i in range(first, last) if any(o.res_src == i for o in q.ops))
        feats.append(("resout", n * q.ops[r].w.shape[0], 2 * n * q.ops[r].w.shape[0]))
    return feats, n


def graphs(q, c, g=None):
    """Every distinct graph of a candidate: name -> (first, last, input rows, valid, res_out)."""
    g = g or geometry(q, c)
    out = {f"seg{j}": (sg["first"], sg["last"], g["nin"][j], True, sg["res_out"] is not None)
           for j, sg in enumerate(g["segs"])}
    if "tail" in g:
        out["tail"] = (g["tail"]["first"], len(q.ops), g["tail"]["L"], False, False)
    return out


# ----------------------------------------------------------------------------- cost model
FAMILIES = ("conv1", "conv", "dw", "pw", "add", "resout", "pad")


class CostModel:
    """Profile-calibrated job costs in CPU cycles (coefficients from the kernel-profiling build)."""

    def __init__(self, coef):
        self.c = coef

    @staticmethod
    def features(feats):
        """Design vector: [call, then per family: count, MACs, bytes]."""
        x = np.zeros(1 + 3 * len(FAMILIES))
        x[0] = 1
        for fam, mac, by in feats:
            i = 1 + 3 * FAMILIES.index(fam)
            x[i:i + 3] += (1, mac, by)
        return x

    def graph(self, feats):
        return float(self.features(feats) @ np.asarray(self.c["graph"]))

    def copy(self, nbytes):
        return self.c["copy0"] + self.c["copy"] * nbytes

    def feed(self, n):
        return self.c["feed0"] + self.c["feed"] * n

    def score(self, rows):
        return self.c["score0"] + self.c["score"] * rows


# ----------------------------------------------------------------------------- dispatcher + timing
def hp_finish(t, w, interf):
    """Completion time of w seconds of CNN work started at t next to a higher-priority periodic task
    interf = (C, P, phi) (s) occupying [phi + kP, phi + kP + C): fixed-priority preemption."""
    C, P, phi = interf
    k = math.floor((t - phi) / P)
    while True:
        s = phi + k * P                                      # release of the k-th competing job
        if t < s + C:                                        # the competing task is running: wait
            t = s + C
        avail = s + P - t                                    # free time until its next release
        if w <= avail:
            return t + w
        w -= max(avail, 0.0)
        k += 1                                               # integer step: no floating-point stall at a boundary
        t = max(t, phi + k * P)


def simulate(q, c, cost, g=None, record=None, interf=None):
    """Mirror of the stream.c dispatcher over N_REPLAY consecutive windows.
    interf = (C, P, phi) seconds: a higher-priority periodic task preempts the dispatcher (FreeRTOS builds).

    Returns predicted timing (per window), the job list of the last window, retained-buffer
    high-water marks (rows) and input-ring checks. record = window index: also returns "events",
    one per job of that window: (kind, segment, t0, t1, stored bytes after the job, samples fed)."""
    g = g or geometry(q, c)
    segs, new, nin, nout, lens = g["segs"], g["new"], g["nin"], g["nout"], g["lens"]
    ns = len(segs)
    gname = graphs(q, c, g)
    gcost = {k: cost.graph(graph_ops(q, a, b, n, v, r)[0]) for k, (a, b, n, v, r) in gname.items()}
    cin = [sg["c_in"] for sg in segs]
    padl = [0] + [sg["pad"][0] for sg in segs[1:]]
    padr = [0] + [sg["pad"][1] for sg in segs[1:]]
    rid = {r["src"]: i for i, r in enumerate(g["res"])}
    hw = {"fill": [0] * ns, "res": [0] * len(g["res"]), "s2in": 0, "ring_viol": 0, "ring_max": 0}
    T_acq = WIN / FS
    out = {"t_post": [], "cpu": [], "jobs": None}
    clock = 0.0                                              # finish time of the last CPU job

    for w in range(N_REPLAY):
        base_t = w * T_acq
        fill, prod, started = [0] * ns, [0] * ns, [False] * ns
        res_rows = [0] * len(g["res"])
        s2in = 0
        jobs = []
        busy = 0.0
        fed = [0]
        tail_cin = g["tail"]["cin"] if "tail" in g else 0

        def job(kind, j, r, cyc):
            nonlocal clock, busy
            t0 = max(r, clock)
            clock = t0 + cyc / MCU_HZ if interf is None else hp_finish(t0, cyc / MCU_HZ, interf)
            busy += cyc
            jobs.append((kind, j, t0 - base_t, clock - base_t))
            if record == w:
                stored = (sum(f * ci for f, ci in zip(fill, cin)) + s2in * tail_cin
                          + sum(n * rr["ch"] for n, rr in zip(res_rows, g["res"])))
                out.setdefault("events", []).append((kind, j, t0 - base_t, clock - base_t, stored, fed[0]))

        def push(j, n, final, r):
            nonlocal s2in
            if j >= ns:                                      # rows into the whole-window tail input
                s2in += n
                hw["s2in"] = max(hw["s2in"], s2in)
                job("copy", -1, r, cost.copy(n * g["tail"]["cin"]))
                return
            if not started[j]:
                started[j] = True
                fill[j] += padl[j]
            fill[j] += n + (padr[j] if final else 0)
            hw["fill"][j] = max(hw["fill"][j], fill[j])
            job("copy", j, r, cost.copy(n * cin[j]))
            while prod[j] < lens[j] and (fill[j] >= nin[j] or (final and fill[j] > 0)):
                run_seg(j, r)

        def run_seg(j, r):
            hw["fill"][j] = max(hw["fill"][j], nin[j])          # end-of-stream zero fill up to nin
            job("seg", j, r, gcost[f"seg{j}"])
            keep = min(nout[j], lens[j] - prod[j])
            prod[j] += keep
            last = prod[j] >= lens[j]
            sg = segs[j]
            if sg["res_in"] is not None:
                res_rows[rid[sg["res_in"]]] -= keep             # retired after the addition
            if sg["res_out"] is not None:
                i = rid[sg["res_out"]]
                res_rows[i] += keep
                hw["res"][i] = max(hw["res"][i], res_rows[i])
                job("copy", j, r, cost.copy(keep * g["res"][i]["ch"]))
            adv = min(new[j], fill[j])
            job("copy", j, r, cost.copy((fill[j] - adv) * cin[j]))
            fill[j] -= adv
            if j + 1 < ns:
                push(j + 1, keep, last, r)
            elif "tail" in g:
                push(ns, keep, last, r)
            else:
                job("score", j, r, cost.score(keep))

        def feed(a, b, r):                                   # samples [a, b) of window w
            for x in range(a, b, g["raw_chunk"]):
                n = min(g["raw_chunk"], b - x)
                t = max(r, clock)
                arrived = min(int(t * FS + 1e-9), N_REPLAY * WIN)
                hw["ring_max"] = max(hw["ring_max"], arrived - (w * WIN + x))
                hw["ring_viol"] += arrived - (w * WIN + x) > g["raw_ring"]
                fed[0] += n
                job("feed", 0, r, cost.feed(n))
                push(0, n, False, r)

        push(0, q.pad[0], False, base_t)                     # stream padding before the window
        if c.rho:
            rc = g["raw_chunk"]
            for k in range(WIN // rc):
                feed(k * rc, (k + 1) * rc, base_t + (k + 1) * rc / FS)
        else:
            feed(0, WIN, base_t + T_acq)
        push(0, q.pad[1], True, base_t + T_acq)              # flush: stream padding after the window
        if "tail" in g:
            s2in = 0                                         # tail input consumed by the tail run
            job("tail", -1, base_t + T_acq, gcost["tail"])
            job("score", -1, base_t + T_acq, cost.score(N_EMB))
        job("score", -1, base_t + T_acq, cost.c["win"])
        out["t_post"].append((clock - base_t - T_acq) * 1e3)
        out["cpu"].append(busy / MCU_HZ * 1e3)
        out["jobs"] = jobs
    out.update(t_post_ms=float(np.mean(out["t_post"])), cpu_ms=float(np.mean(out["cpu"])),
               max_job_ms=max((b - a) * 1e3 for k, _, a, b in out["jobs"] if k in ("seg", "tail")),
               n_calls=sum(k in ("seg", "tail") for k, *_ in out["jobs"]), hw=hw, gcost=gcost)
    return out


# ----------------------------------------------------------------------------- memory
def capacities(q, c, sim, g=None):
    """Allocated rows per stored stream: retained high-water mark, at least what one run reads."""
    g = g or geometry(q, c)
    seg = [max(sim["hw"]["fill"][j], g["nin"][j]) for j in range(len(g["segs"]))]
    res = [max(sim["hw"]["res"][i], g["nout"][r["cons"]]) for i, r in enumerate(g["res"])]
    return {"seg": seg, "res": res}


def _a4(n):
    return (int(n) + 3) & ~3


def memory(q, c, sim, st, mem_coef, g=None):
    """Predicted deployment SRAM (bytes) by disjoint category.

    st: per graph name the ST Edge AI planner outputs {"activations", "ctx"};
    mem_coef: calibrated {"fixed", "stack0", "stack_per_seg"}."""
    g = g or geometry(q, c)
    cap = capacities(q, c, sim, g)
    parts = {
        "state": sum(_a4(n * sg["c_in"]) for n, sg in zip(cap["seg"], g["segs"])),
        "residual": sum(_a4(n * r["ch"]) for n, r in zip(cap["res"], g["res"])),
        "tail_input": _a4(g["tail"]["L"] * g["tail"]["cin"]) if "tail" in g else 0,
        "input": _a4(g["raw_ring"]) + _a4(g["raw_chunk"]),
        "arena": max(v["activations"] for v in st.values()),
        "contexts": sum((v["ctx"] + 7) & ~7 for v in st.values()),
        "fixed": mem_coef["fixed"],
        "stack": mem_coef["stack0"] + mem_coef["stack_per_seg"] * len(g["segs"]) + mem_coef.get("stack_leaf", 0),
    }
    parts["total"] = sum(parts.values())
    return parts, cap


def required_sample(q, c, g=None):
    """Latest physical sample index needed by each run of each streamed segment (interval
    propagation through the conv chain; padding rows need no sample). Returns {j: [u per run]}."""
    g = g or geometry(q, c)
    segs, nout, lens = g["segs"], g["nout"], g["lens"]
    off = [q.pad[0]] + [sg["pad"][0] for sg in segs[1:]]   # leading padding rows of each stream
    out = {}
    for j in range(len(segs)):
        us = []
        for p0 in range(0, lens[j], nout[j]):
            hi = min(p0 + nout[j], lens[j]) - 1              # last output row of this run
            for i in range(j, -1, -1):                       # back to segment-0 input rows
                hi = hi * segs[i]["s"] + segs[i]["k"] - 1 - (off[i] if i else 0)
            us.append(min(hi - off[0], WIN - 1))
        out[j] = us
    return out
