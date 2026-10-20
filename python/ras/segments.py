"""Segmentation of the int8 op chain and the chunk graphs of the segments.

The op chain is cut before every conv with kernel k > 1. Segment j = ops[cut_j : cut_{j+1}] is that
conv (stride s_j) followed by 1x1 convs. Run chunk by chunk, segment j keeps the last k_j - s_j
rows of its input stream (stream buffer), so no row is recomputed. 'Same' padding of the first conv
becomes explicit zero-point rows at the stream start / end; a residual whose source lies in an
earlier segment becomes an extra output of that segment and an extra input of the consumer.
    segments(q, last)                 segment table of ops[0:last]
    plan(segs, chunk)                 new input rows per run: chunk samples into segment 0
    export_segment(q, seg, new, path) QDQ ONNX chunk graph (input new + tail rows)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .workload import OPSET, Workload


def segments(q: Workload, first=0, last=None):
    ops = q.ops
    last = len(ops) if last is None else last
    cuts = [i for i in range(first, last) if ops[i].w.shape[2] > 1 or i == first]
    segs = []
    for a, b in zip(cuts, cuts[1:] + [last]):
        op = ops[a]
        k, s = op.w.shape[2], op.stride
        c_in = 1 if a == 0 else ops[a - 1].w.shape[0]
        segs.append({"first": a, "last": b, "k": k, "s": s, "tail": k - s, "pad": tuple(op.pad),
                     "c_in": c_in, "c_out": ops[b - 1].w.shape[0], "res_out": None, "res_in": None})
    for j, sg in enumerate(segs):
        for i in range(sg["first"], sg["last"]):
            r = ops[i].res_src
            if r is not None and r < sg["first"]:
                src = next(t for t, x in enumerate(segs) if x["first"] <= r < x["last"])
                segs[src]["res_out"] = r
                sg["res_in"] = r
    return segs


def export_segment(q: Workload, sg, n_new: int, path, batch=1):
    """Chunk graph of one segment: input (n_new + tail) rows -> n_new / s rows (valid convs)."""
    ops = q.ops
    first, last = sg["first"], sg["last"]
    n_in = n_new + sg["tail"]
    n_out = (n_in - sg["k"]) // sg["s"] + 1
    if first == 0:
        s_in, zp_in = q.in_scale, q.in_zp
    else:
        s_in, zp_in = ops[first - 1].q
    nodes, inits = [], []

    def const(name, arr, dtype=None):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))
        return name

    def qdq(x, s, zp, tag):
        sc = const(f"{tag}_s", np.float32(s))
        zpn = const(f"{tag}_zp", np.int8(zp))
        nodes.append(helper.make_node("QuantizeLinear", [x, sc, zpn], [f"{tag}_q"]))
        nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_q", sc, zpn], [f"{tag}_dq"]))
        return f"{tag}_dq"

    x = qdq("input", s_in, zp_in, "in")
    names = {first - 1: x}
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, sg["c_in"], 1, n_in])]
    if sg["res_in"] is not None:
        r = sg["res_in"]
        rs, rzp = ops[r].q
        names[r] = qdq("res", rs, rzp, "resin")
        inputs.append(helper.make_tensor_value_info("res", TensorProto.FLOAT,
                                                    [batch, ops[r].w.shape[0], 1, n_out]))
    for i in range(first, last):
        op = ops[i]
        t = f"l{i}"
        wq = np.clip(np.round(op.w / op.w_scale[:, None, None]), -127, 127).astype(np.int8)[:, :, None, :]
        s_prev = s_in if i == first else ops[i - 1].q[0]
        bs = (s_prev * op.w_scale).astype(np.float32)
        const(f"{t}_wq", wq)
        const(f"{t}_ws", op.w_scale.astype(np.float32))
        const(f"{t}_wzp", np.zeros(len(op.w_scale), np.int8))
        nodes.append(helper.make_node("DequantizeLinear", [f"{t}_wq", f"{t}_ws", f"{t}_wzp"],
                                      [f"{t}_w"], axis=0))
        const(f"{t}_bq", np.round(op.b / bs).astype(np.int32))
        const(f"{t}_bs", bs)
        const(f"{t}_bzp", np.zeros(len(bs), np.int32))
        nodes.append(helper.make_node("DequantizeLinear", [f"{t}_bq", f"{t}_bs", f"{t}_bzp"],
                                      [f"{t}_b"], axis=0))
        nodes.append(helper.make_node("Conv", [x, f"{t}_w", f"{t}_b"], [f"{t}_conv"],
                                      kernel_shape=[1, op.w.shape[2]], strides=[1, op.stride],
                                      dilations=[1, op.dil], group=op.groups, pads=[0, 0, 0, 0]))
        y = f"{t}_conv"
        if op.relu:
            nodes.append(helper.make_node("Relu", [y], [f"{t}_relu"]))
            y = f"{t}_relu"
        x = qdq(y, op.out_scale, op.out_zp, t)
        if op.res_src is not None:
            nodes.append(helper.make_node("Add", [x, names[op.res_src]], [f"{t}_add"]))
            x = qdq(f"{t}_add", op.add_scale, op.add_zp, f"{t}_res")
        names[i] = x
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                             [batch, sg["c_out"], 1, n_out])]
    nodes[-1].output[0] = "output"                       # final DequantizeLinear = graph output
    for nd in nodes[:-1]:                                # (re-point any consumer of that tensor)
        nd.input[:] = ["output" if t == x else t for t in nd.input]
    if sg["res_out"] is not None:
        r = sg["res_out"]
        rs, rzp = ops[r].q                               # same grid -> exact copy, quantised output
        qdq(names[r] if names[r] != x else "output", rs, rzp, "resout")
        nodes[-1].output[0] = "res_out"
        outputs.append(helper.make_tensor_value_info("res_out", TensorProto.FLOAT,
                                                     [batch, ops[r].w.shape[0], 1, n_out]))
    g = helper.make_graph(nodes, "seg", inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="rasrt")
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return n_in, n_out


def plan(segs, chunk):
    """New input rows per run: `chunk` samples into segment 0, then divided by each stride, so one
    run of segment j produces exactly one run's input of segment j + 1."""
    new = [chunk]
    for sg in segs[:-1]:
        assert new[-1] % sg["s"] == 0
        new.append(new[-1] // sg["s"])
    return new[:len(segs)]
