"""Calibration of the schedule cost model by kernel profiling (never by complete schedules).

Calibration graphs: every operator of the three models as a single-operator chunk graph (with its
residual input / residual output where it has one) at output lengths that no deployed schedule
uses, operator pairs (separating the per-call cost from the per-operator cost), and same-padded
operators as whole-window graphs (Pad). They are profiled on the board (firmware/src/profile.c,
packed into as few images as flash / SRAM allow) together with the dispatcher primitives. The
fit is non-negative least squares per term of predictor.CostModel.
Memory calibration: one compile-only stream build of an excluded configuration (m = 8, not in the
candidate set) gives the fixed statics; stack = profiled kernel high-water + the dispatcher's static
frames (-fstack-usage) for the candidate's recursion depth.
"""

import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from . import toolchain as fw
from .config import STUDY
from .predictor import CostModel, graph_ops
from .segments import export_segment

REPS = 20
LENGTHS = (11, 57, 287)                       # output rows of the calibration graphs (off-grid)
COPY_BYTES = (16, 96, 400, 1600, 6000)
FEED_N = (32, 80, 160, 320, 640)
SCORE_ROWS = (1, 10, 40, 160)
RAW_RING = 2816
IO_CAP = 56_000                                # input + output bytes of one calibration graph


def _sources(q):
    return {o.res_src for o in q.ops if o.res_src is not None}


def specs(qs):
    """Calibration graphs: dicts (key, v, first, last, n_in, valid, res_in, res_out)."""
    out, seen = [], set()
    for v, q in qs.items():
        src = _sources(q)
        full = []                                         # row count of every op on the full window
        n = 2560 + sum(q.pad)
        for i, op in enumerate(q.ops):
            n = (n + sum(op.pad) - op.w.shape[2]) // op.stride + 1
            full.append(n)
        for i, op in enumerate(q.ops):
            k, s, cin, cout = op.w.shape[2], op.stride, op.w.shape[1] * op.groups, op.w.shape[0]
            lens = [L for L in LENGTHS if L < full[i]] + [int(full[i] * 0.93)]
            for L in lens:
                n_in = (L - 1) * s + k
                if n_in * cin + L * cout * (2 if i in src else 1) > IO_CAP:
                    continue
                sig = (k, s, op.groups, cin, cout, op.relu, op.res_src is not None, i in src, L, True)
                if sig not in seen:
                    seen.add(sig)
                    out.append(dict(v=v, first=i, last=i + 1, n_in=n_in, valid=True,
                                    res_in=op.res_src, res_out=i if i in src else None))
            nxt = q.ops[i + 1] if i + 1 < len(q.ops) else None
            if nxt is not None and nxt.w.shape[2] == 1 and nxt.res_src is None and i not in src:
                L = 57
                sig = ("pair", k, s, op.groups, cin, cout, nxt.w.shape[0], L)
                if sig not in seen and (L - 1) * s + k <= full[i] * s:
                    seen.add(sig)
                    out.append(dict(v=v, first=i, last=i + 2, n_in=(L - 1) * s + k, valid=True,
                                    res_in=op.res_src, res_out=None))
            if any(op.pad) and op.res_src is None and i not in src:
                n_in = 140
                sig = ("pad", k, s, op.groups, cin, cout, op.pad, n_in)
                if sig not in seen:
                    seen.add(sig)
                    out.append(dict(v=v, first=i, last=i + 1, n_in=n_in, valid=False, res_in=None, res_out=None))
    for i, sp in enumerate(out):
        sp["key"] = f"net{i}"
    return out


def _export(q, sp, path):
    op = q.ops[sp["first"]]
    if sp["valid"]:
        k, s = op.w.shape[2], op.stride
        sg = {"first": sp["first"], "last": sp["last"], "k": k, "s": s, "tail": k - s, "pad": tuple(op.pad),
              "c_in": op.w.shape[1] * op.groups, "c_out": q.ops[sp["last"] - 1].w.shape[0],
              "res_out": sp["res_out"], "res_in": sp["res_in"]}
        export_segment(q, sg, sp["n_in"] - (k - s), path)
    else:
        q.to_onnx(path, sp["n_in"], sp["first"], sp["last"])
    _prefix(path, sp["key"])


def _prefix(path, tag):
    """Unique internal tensor / node names (ST Edge AI derives global C symbols from them)."""
    import onnx
    m = onnx.load(str(path))
    keep = {"input", "res", "output", "res_out"}
    ren = lambda t: t if t in keep or not t else f"{tag}_{t}"  # noqa: E731
    for i in m.graph.initializer:
        i.name = ren(i.name)
    for n in m.graph.node:
        n.input[:] = [ren(t) for t in n.input]
        n.output[:] = [ren(t) for t in n.output]
        n.name = ren(n.name)
    onnx.save(m, str(path))


def _io_bytes(q, sp):
    feats, n_out = graph_ops(q, sp["first"], sp["last"], sp["n_in"], sp["valid"], sp["res_out"] is not None)
    op = q.ops[sp["first"]]
    cin = op.w.shape[1] * op.groups
    in1 = n_out * q.ops[sp["res_in"]].w.shape[0] if sp["res_in"] is not None else 0
    return sp["n_in"] * cin, in1, feats


def profile(qs, out=None):
    """Generate, pack into images, build, flash and read every calibration graph."""
    out = Path(out or STUDY / "calib")
    sp_all = specs(qs)
    gen = out / "graphs"
    for sp in sp_all:
        p = gen / f"{sp['key']}.onnx"
        p.parent.mkdir(parents=True, exist_ok=True)
        _export(qs[sp["v"]], sp, p)
        sp["st"] = fw.generate(p, sp["key"], gen / sp["key"])
        sp["in0"], sp["in1"], feats = _io_bytes(qs[sp["v"]], sp)
        sp["feats"] = [list(f) for f in feats]
    images, cur = [], []
    for sp in sorted(sp_all, key=lambda s: s["st"]["activations"]):
        trial = cur + [sp]
        ram = (max(s["st"]["activations"] for s in trial) + max(s["in0"] for s in trial)
               + max(s["in1"] for s in trial) + sum(s["st"]["ctx"] for s in trial))
        if cur and (ram > 100_000 or sum(s["st"]["weights"] + 4000 for s in trial) > 420_000):
            images.append(cur)
            cur = [sp]
        else:
            cur = trial
    images.append(cur)
    res = {"graphs": [], "copy": [], "feed": [], "score": [], "stack": [], "frames": {}}
    for ii, img in enumerate(images):
        d = out / f"img{ii}"
        d.mkdir(parents=True, exist_ok=True)
        cfg = ["#pragma once", "#include <stdint.h>", f"#define N_NETS {len(img)}", f"#define REPS {REPS}",
               f"#define MAX_IN0 {max(s['in0'] for s in img) + 4}", f"#define MAX_IN1 {max(s['in1'] for s in img) + 4}",
               f"#define N_COPY {len(COPY_BYTES)}", f"#define N_FEED {len(FEED_N)}", f"#define N_SCORE {len(SCORE_ROWS)}",
               f"#define MAX_COPY {max(COPY_BYTES) + 8}", f"#define MAX_FEED {max(FEED_N)}",
               f"#define RAW_RING {RAW_RING}", "#define LATENT 16", "#define TOPK 16", "#define OUT_ZP 3",
               "#define OUT_SCALE 4.0e-3f", fw.c_array("g_center", np.linspace(-0.2, 0.2, 16), "float"),
               fw.c_array("COPY_BYTES", COPY_BYTES, "uint32_t"), fw.c_array("FEED_N", FEED_N, "uint32_t"),
               fw.c_array("SCORE_ROWS", SCORE_ROWS, "int"),
               fw.c_array("NET_NIN", [2 if s["in1"] else 1 for s in img], "int")]
        (d / "prof_config.h").write_text("\n".join(cfg) + "\n")
        nets = [(s["key"], gen / s["key"]) for s in img]
        (d / "nets.h").write_text(fw.nets_header(nets, max(s["st"]["activations"] for s in img)))
        elf, sec, frames = fw.build(d, "profile.c", nets)
        n = len(img)
        nbytes = 4 * (4 + 2 * n + len(COPY_BYTES) + len(FEED_N) + len(SCORE_ROWS))
        raw = fw.flash_read(elf, nbytes, 3.0 + 0.002 * sum(s["st"]["macc"] for s in img) * REPS / 1e5, d)
        v = struct.unpack_from(f"<{nbytes // 4}I", raw, 0)
        assert v[1] == 0, f"profile image {ii}: status {v[1]}"
        mins, sums = v[4:4 + n], v[4 + n:4 + 2 * n]
        o = 4 + 2 * n
        for s, mn, sm in zip(img, mins, sums):
            res["graphs"].append({"key": s["key"], "v": s["v"], "first": s["first"], "last": s["last"],
                                  "n_in": s["n_in"], "valid": s["valid"], "feats": s["feats"],
                                  "min": mn, "mean": sm / REPS, "st": s["st"]})
        res["copy"].append(list(v[o:o + len(COPY_BYTES)]))
        o += len(COPY_BYTES)
        res["feed"].append(list(v[o:o + len(FEED_N)]))
        o += len(FEED_N)
        res["score"].append(list(v[o:o + len(SCORE_ROWS)]))
        res["stack"].append(v[2])
        res["frames"] = {**res["frames"], **frames}
        print(f"  calibration image {ii}: {n} graphs, flash {sec.get('.text', 0) + sec.get('.rodata', 0)} B", flush=True)
    (out / "profile.json").write_text(json.dumps(res))
    return res


def fit(res):
    """Cost-model coefficients (cycles) from the profiling results."""
    X = np.stack([CostModel.features([tuple(f) for f in g["feats"]]) for g in res["graphs"]])
    y = np.array([g["mean"] for g in res["graphs"]])
    w = 1.0 / np.maximum(y, 1.0)                           # relative-error weighting
    coef, _ = nnls(X * w[:, None], y * w)
    pred = X @ coef
    lin = lambda xs, ys: np.polyfit(np.asarray(xs, float), np.median(np.asarray(ys, float), 0), 1)  # noqa: E731
    cb, ca = lin(COPY_BYTES, res["copy"])
    fb, fa = lin(FEED_N, res["feed"])
    sb, sa = lin(SCORE_ROWS, res["score"])
    return {"graph": coef.tolist(), "copy0": max(ca, 0.0), "copy": cb, "feed0": max(fa, 0.0), "feed": fb,
            "score0": max(sa, 0.0), "score": sb, "win": 200.0,
            "fit_rel_err": {"median": float(np.median(np.abs(pred - y) / y)), "max": float(np.max(np.abs(pred - y) / y))},
            "n_graphs": len(y)}
