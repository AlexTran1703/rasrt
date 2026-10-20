"""Schedules under FreeRTOS (fixed-priority preemptive, static allocation) with a competing periodic task.

    deploy_rtos(q, ref, v, c, xq, pred, u, t_ms, out, n_win, cnn_prio, load_prio)
        the dispatcher of candidate c (stream.c, RTOS = 1) as task "cnn"; a periodic task "load" with period t_ms
        and utilisation u (u * t_ms of computation per job, deadline = period); n_win windows replay the
        N_REPLAY stimulus windows cyclically. Measured: CNN T_post per window, load jobs / deadline misses / worst
        response, reserved SRAM (.data + .bss, incl. kernel objects and task stacks, + main-stack high-water).
Priorities: cnn_prio < load_prio = competing task preempts the CNN (CNN deadline at risk); cnn_prio > load_prio
= CNN preempts the competing task (its blocking = the longest CNN job).
"""

import json
import re
import struct
from pathlib import Path

import numpy as np

from . import toolchain as fw
from .config import FIRMWARE, FS, MCU_HZ, WIN
from .deploy import _config, prepare
from .predictor import N_REPLAY, geometry

KERNEL = FIRMWARE / "third_party" / "FreeRTOS-Kernel"
SRCS = [KERNEL / f for f in ("tasks.c", "list.c", "queue.c")] + [KERNEL / "portable" / "GCC" / "ARM_CM4F" / "port.c"]
INC = [KERNEL / "include", KERNEL / "portable" / "GCC" / "ARM_CM4F"]
N_TRACE = 1600
N_LOAD_TR = 160


def deploy_rtos(q, ref, v, c, xq, pred, u, t_ms, out, n_win=8, cnn_prio=1, load_prio=2, trace_load=False,
                phase_step=0, trace_win=None):
    """trace_load: also record the periodic task's jobs around window TRACE_WIN's end (Fig. 1 builds); returns
    cnn_trace (segment id / 15 = tail, start, end) and load_trace (start, end), s relative to the last sample."""
    out = Path(out)
    if (out / "measured.json").exists():
        return json.loads((out / "measured.json").read_text())
    g = geometry(q, c)
    st = prepare(q, v, c, g)
    out.mkdir(parents=True, exist_ok=True)
    _config(out, q, ref, c, g, pred["cap"], xq)
    cfg = (out / "bench_config.h").read_text()
    if trace_win is not None:                                  # traced window (default deploy.TRACE_WIN)
        cfg = re.sub(r"#define TRACE_WIN \d+", f"#define TRACE_WIN {trace_win}", cfg)
    cfg = re.sub(r"#define N_WIN \d+", f"#define N_WIN {n_win}\n#define N_DATA {N_REPLAY}", cfg)
    c_cyc = int(round(u * t_ms * MCU_HZ / 1000))
    cfg += "\n".join(["#define RTOS 1", f"#define LOAD_T_MS {t_ms}", f"#define LOAD_C_CYC {c_cyc}u",
                      f"#define CNN_PRIO {cnn_prio}", f"#define LOAD_PRIO {load_prio}",
                      f"#define TRACE_LOAD {int(trace_load)}", f"#define LOAD_PHASE_STEP {phase_step}"]) + "\n"
    (out / "bench_config.h").write_text(cfg)
    order = [n for n in st if n != "tail"] + (["tail"] if "tail" in st else [])
    nets = [(n, Path(st[n]["dir"])) for n in order]
    (out / "nets.h").write_text(fw.nets_header(nets, max(st[n]["activations"] for n in order)))
    elf, sec, frames = fw.build(out, "stream.c", nets, SRCS, INC)
    nseg, nres = max(len(g["segs"]), 1), max(len(g["res"]), 1)
    head = 9 + 4 * n_win + nseg + nres + 2 + 3 * N_TRACE
    nwords = head + 6 + ((256 + 1 + 2 * N_LOAD_TR) if trace_load else 0)
    raw = fw.flash_read(elf, 4 * nwords, 3.0 + n_win * WIN / FS, out)
    u32 = struct.unpack_from(f"<{nwords}I", raw, 0)
    names = ["magic", "status", "nsamp", "raw_viol", "ovf", "ring_max", "stack_used", "isr_max", "s2in_hw"]
    r = dict(zip(names, u32[:9]))
    t_resp = [x / MCU_HZ * 1e3 for x in u32[9:9 + n_win]]
    hashes = list(u32[9 + 2 * n_win:9 + 3 * n_win])
    load = dict(zip(["load_n", "load_miss", "load_rmax", "cnn_stack_hw", "load_stack_hw", "rtos_assert"], u32[head:]))
    m = {"label": c.label, "u": u, "t_ms": t_ms, "phase_step": phase_step, "cnn_prio": cnn_prio, "load_prio": load_prio, "n_win": n_win,
         "ok": r["magic"] == fw.RESULTS_MAGIC and r["status"] == 0 and r["ovf"] == 0 and load["rtos_assert"] == 0,
         "t_post_win_ms": t_resp, "t_post_ms": float(np.mean(t_resp)), "t_post_max_ms": float(max(t_resp)),
         "sram": sec.get(".data", 0) + sec.get(".bss", 0) + r["stack_used"],
         "load_rmax_ms": load["load_rmax"] / MCU_HZ * 1e3, "hash": hashes,
         **{k: r[k] for k in ("status", "raw_viol", "ovf", "ring_max", "stack_used")}, **load, "sections": sec}
    if trace_load:
        tn, t_last = u32[head - 3 * N_TRACE - 2], u32[head - 3 * N_TRACE - 1]
        rel = lambda x: ((x - t_last + 2 ** 31) % 2 ** 32 - 2 ** 31) / MCU_HZ  # noqa: E731
        tr = u32[head - 3 * N_TRACE:head - 3 * N_TRACE + 3 * tn]
        m["cnn_trace"] = [(tr[3 * i], rel(tr[3 * i + 1]), rel(tr[3 * i + 2])) for i in range(tn)]
        o = head + 6 + 256
        m["load_trace"] = [(rel(u32[o + 1 + 2 * i]), rel(u32[o + 2 + 2 * i])) for i in range(min(u32[o], N_LOAD_TR))]
    (out / "measured.json").write_text(json.dumps(m))
    return m
