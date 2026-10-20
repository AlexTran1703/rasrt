"""Schedules on the board (bare metal): graphs, prediction, firmware configuration, measurement.

    prepare(q, v, c)            export + ST-generate every graph of candidate c (cached, shared
                                between candidates); returns the planner outputs per graph
    predict(q, v, c, cost, mc)  prediction (predictor.simulate + predictor.memory), no firmware
    deploy(q, ref, v, c, xq, pred)  firmware with the predicted capacities -> measured metrics
    calibrate_memory(q, ref, v, xq, cost, frames_profile, stack_profile)
                                fixed statics + stack terms from a compile-only build of an
                                excluded configuration (m = 8)
"""

import json
import struct
from pathlib import Path

import numpy as np

from . import toolchain as fw
from .config import FS, LATENT, MCU_HZ, N_EMB, STUDY, TOPK, WIN
from .predictor import N_REPLAY, Cand, geometry, graphs, memory, simulate
from .segments import export_segment, segments

N_TRACE = 1600
TRACE_WIN = 1


def graph_key(g, name):
    """Cache key of a graph: segment, new rows and its residual role (the same segment exports
    differently when its residual source / consumer lies inside or outside the streamed depth)."""
    if name == "tail":
        return f"tail_c{g['cut']}"
    j = int(name[3:])
    sg = g["segs"][j]
    res = "" if sg["res_in"] is None and sg["res_out"] is None else f"_r{sg['res_in']}-{sg['res_out']}"
    return f"{name}_n{g['new'][j]}{res}"


def prepare(q, v, c, g=None):
    g = g or geometry(q, c)
    out = {}
    for name, (first, last, n_in, valid, _) in graphs(q, c, g).items():
        key = graph_key(g, name)
        gdir = STUDY / v / "graphs" / key
        p = gdir / f"{name}.onnx"
        if not p.exists():
            gdir.mkdir(parents=True, exist_ok=True)
            if name == "tail":
                q.to_onnx(p, n_in, first, None)
            else:
                j = int(name[3:])
                export_segment(q, g["segs"][j], g["new"][j], p)
        out[name] = {**fw.generate(p, name, gdir), "dir": str(gdir)}
    return out


def prepare_all(q, v, cands, workers=3):
    """Export + ST-generate every distinct graph of many candidates, `workers` at a time (each graph
    has its own folder; generation is cached on the ONNX content, so results equal the serial run)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = {}
    for c in cands:
        g = geometry(q, c)
        for name in graphs(q, c, g):
            jobs.setdefault(graph_key(g, name), (c, name, g))

    def one(item):
        c, name, g = item
        first, last, n_in, valid, _ = graphs(q, c, g)[name]
        gdir = STUDY / v / "graphs" / graph_key(g, name)
        p = gdir / f"{name}.onnx"
        if not p.exists():
            gdir.mkdir(parents=True, exist_ok=True)
            if name == "tail":
                q.to_onnx(p, n_in, first, None)
            else:
                export_segment(q, g["segs"][int(name[3:])], g["new"][int(name[3:])], p)
        return fw.generate(p, name, gdir)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs.values()))
    return len(jobs)


def predict(q, v, c, cost, mem_coef):
    g = geometry(q, c)
    st = prepare(q, v, c, g)
    sim = simulate(q, c, cost, g)
    parts, cap = memory(q, c, sim, st, mem_coef, g)
    return {"label": c.label, "d": c.d, "m": c.m, "rho": c.rho, "t_post_ms": sim["t_post_ms"],
            "t_post_win_ms": sim["t_post"], "cpu_ms": sim["cpu_ms"], "max_job_ms": sim["max_job_ms"],
            "n_calls": sim["n_calls"], "ring_viol": sim["hw"]["ring_viol"], "ring_max": sim["hw"]["ring_max"],
            "sram": parts["total"], "mem": parts, "cap": cap, "hw": sim["hw"],
            "graph_ms": {k: x / MCU_HZ * 1e3 for k, x in sim["gcost"].items()},
            "jobs": [j for j in sim["jobs"] if j[0] in ("seg", "tail")]}


def _config(out, q, ref, c, g, cap, xq):
    segs = g["segs"]
    seg_bytes = [((n * sg["c_in"]) + 3) & ~3 for n, sg in zip(cap["seg"], segs)]
    res_bytes = [((n * r["ch"]) + 3) & ~3 for n, r in zip(cap["res"], g["res"])]
    rid = {r["src"]: i for i, r in enumerate(g["res"])}
    last = q.ops[-1]
    cfg = ["#pragma once", "#include <stdint.h>", f"#define NSEG {len(segs)}", f"#define NSEG_ALLOC {max(len(segs), 1)}",
           f"#define NRES {len(g['res'])}", f"#define NRES_ALLOC {max(len(g['res']), 1)}",
           f"#define HAS_S2W {int('tail' in g)}", f"#define RELEASE {c.rho}", f"#define N_WIN {N_REPLAY}",
           f"#define TRACE_WIN {TRACE_WIN}", f"#define N_TRACE {N_TRACE}", f"#define WIN_SAMPLES {WIN}",
           f"#define PAD_L {q.pad[0]}", f"#define PAD_R {q.pad[1]}", f"#define FS_HZ {FS}",
           f"#define RAW_RING {g['raw_ring']}", f"#define RAW_CHUNK {g['raw_chunk']}",
           f"#define SEGBUF_TOTAL {max(sum(seg_bytes), 4)}", f"#define RESBUF_TOTAL {max(sum(res_bytes), 4)}",
           f"#define PADROWS_BYTES {max(q.pad) + 4}", f"#define N_EMB {N_EMB}", f"#define LATENT {LATENT}",
           f"#define TOPK {TOPK}", f"#define OUT_SCALE {last.out_scale:.9e}f", f"#define OUT_ZP {last.out_zp}",
           fw.c_array("SEG_NIN", g["nin"], "int"), fw.c_array("SEG_NNEW", g["new"], "int"),
           fw.c_array("SEG_NOUT", g["nout"], "int"), fw.c_array("SEG_CIN", [s["c_in"] for s in segs], "int"),
           fw.c_array("SEG_PADL", [0] + [s["pad"][0] for s in segs[1:]], "int"),
           fw.c_array("SEG_PADR", [0] + [s["pad"][1] for s in segs[1:]], "int"),
           fw.c_array("SEG_LEN_OUT", g["lens"], "int"),
           fw.c_array("SEG_ZP", [0 if s["first"] == 0 else int(q.ops[s["first"] - 1].q[1]) for s in segs], "int"),
           fw.c_array("SEG_OFF", np.cumsum([0] + seg_bytes[:-1]) if segs else [0], "int"),
           fw.c_array("SEG_CAP", cap["seg"], "int"),
           fw.c_array("SEG_RES_IN", [rid[s["res_in"]] + 1 if s["res_in"] is not None else 0 for s in segs], "int"),
           fw.c_array("SEG_RES_OUT", [rid[s["res_out"]] + 1 if s["res_out"] is not None else 0 for s in segs], "int"),
           fw.c_array("RES_CH", [r["ch"] for r in g["res"]], "int"),
           fw.c_array("RES_OFF", np.cumsum([0] + res_bytes[:-1]) if res_bytes else [0], "int"),
           fw.c_array("RES_CAP", cap["res"], "int"), fw.c_array("g_center", ref, "float")]
    if "tail" in g:
        cfg += [f"#define S2W_LEN {g['tail']['L']}", f"#define S2W_CIN {g['tail']['cin']}"]
    cfg.append(f"static const int8_t g_stimulus[{N_REPLAY}][{WIN}] = {{"
               + ",".join("{" + ",".join(map(str, r)) + "}" for r in xq) + "};")
    (out / "bench_config.h").write_text("\n".join(cfg) + "\n")


def _build(q, ref, v, c, xq, cap, out):
    g = geometry(q, c)
    st = prepare(q, v, c, g)
    out.mkdir(parents=True, exist_ok=True)
    _config(out, q, ref, c, g, cap, xq)
    order = [n for n in st if n != "tail"] + (["tail"] if "tail" in st else [])
    nets = [(n, Path(st[n]["dir"])) for n in order]
    (out / "nets.h").write_text(fw.nets_header(nets, max(st[n]["activations"] for n in order)))
    return (*fw.build(out, "stream.c", nets), g, st)


def deploy(q, ref, v, c, xq, pred, out=None):
    out = Path(out or STUDY / v / "builds" / c.label)
    if (out / "measured.json").exists():
        return json.loads((out / "measured.json").read_text())
    elf, sec, frames, g, st = _build(q, ref, v, c, xq, pred["cap"], out)
    nseg, nres = max(len(g["segs"]), 1), max(len(g["res"]), 1)
    nwords = 9 + 4 * N_REPLAY + nseg + nres + 2 + 3 * N_TRACE
    raw = fw.flash_read(elf, 4 * nwords, 2.0 + N_REPLAY * WIN / FS, out)
    u = struct.unpack_from(f"<{nwords}I", raw, 0)
    f = struct.unpack_from(f"<{N_REPLAY}f", raw, 4 * (9 + 3 * N_REPLAY))
    names = ["magic", "status", "nsamp", "raw_viol", "ovf", "ring_max", "stack_used", "isr_max", "s2in_hw"]
    r = dict(zip(names, u[:9]))
    o = 9
    r["t_resp"], r["cpu"], r["hash"] = (list(u[o + i * N_REPLAY:o + (i + 1) * N_REPLAY]) for i in range(3))
    o += 4 * N_REPLAY
    r["score"] = list(f)
    r["seg_hw"] = list(u[o:o + len(g["segs"])])
    o += nseg
    r["res_hw"] = list(u[o:o + len(g["res"])])
    o += nres
    tn, t_last = u[o], u[o + 1]
    o += 2
    tr = np.array(u[o:o + 3 * tn], np.int64).reshape(-1, 3)
    rel = lambda x: ((x - t_last + 2 ** 31) % 2 ** 32 - 2 ** 31) / MCU_HZ  # noqa: E731
    jobs = [("tail" if j == 15 else "seg", int(j) if j != 15 else -1, rel(a), rel(a) + ((b - a) % 2 ** 32) / MCU_HZ)
            for j, a, b in tr]
    flash = sum(sec.get(k, 0) for k in (".isr_vector", ".text", ".rodata", ".ARM.exidx", ".data")) - N_REPLAY * WIN
    m = {"label": c.label, "ok": r["magic"] == fw.RESULTS_MAGIC and r["status"] == 0 and r["ovf"] == 0,
         "t_post_ms": float(np.mean(r["t_resp"])) / MCU_HZ * 1e3, "t_post_win_ms": [x / MCU_HZ * 1e3 for x in r["t_resp"]],
         "cpu_ms": float(np.mean(r["cpu"])) / MCU_HZ * 1e3, "sram": sec.get(".data", 0) + sec.get(".bss", 0) + r["stack_used"],
         "static": sec.get(".data", 0) + sec.get(".bss", 0), "flash": flash,
         "macs": sum(st[n]["macc"] * sum(1 for jb in jobs if (jb[0] == "tail") == (n == "tail")
                                         and (n == "tail" or jb[1] == int(n[3:]))) for n in st),
         "max_job_ms": max(((b - a) * 1e3 for _, _, a, b in jobs), default=0.0), "jobs": jobs,
         **{k: r[k] for k in ("status", "raw_viol", "ovf", "ring_max", "stack_used", "isr_max", "s2in_hw", "hash",
                              "score", "seg_hw", "res_hw")}, "sections": sec, "frames": frames}
    (out / "measured.json").write_text(json.dumps(m))
    return m


def calibrate_memory(q, ref, v, xq, cost, frames_profile, stack_profile):
    """fixed = static bytes of the excluded configuration's link minus every modelled category;
    stack = profiled kernel high-water - profiling main frame + dispatcher frames (+ ISR frame)."""
    c = Cand(len(segments(q)), 8, 1)
    g = geometry(q, c)
    st = prepare(q, v, c, g)
    sim = simulate(q, c, cost, g)
    zero = {"fixed": 0, "stack0": 0, "stack_per_seg": 0}
    parts, cap = memory(q, c, sim, st, zero, g)
    elf, sec, frames, _, _ = _build(q, ref, v, c, xq, cap, STUDY / v / "builds" / "_memcal_m8")
    static = sec.get(".data", 0) + sec.get(".bss", 0)
    known = parts["total"]
    exc = 104 + frames.get("TIM2_IRQHandler", 0)                 # FPU exception frame + ISR
    return {"fixed": static - known,
            "stack0": stack_profile - frames_profile.get("main", 0) + frames.get("main", 0) + frames.get("feed", 0) + exc,
            "stack_per_seg": frames.get("push", 0) + frames.get("run_seg", 0),
            "stack_leaf": frames.get("score_rows", 0), "frames": frames}
