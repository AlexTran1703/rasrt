"""Frozen int8 workloads: the op chain of a calibrated network, loaded from rasrt/workloads/<id>.npz.

A workload holds BN-folded conv weights and biases, strides, dilations, groups, zero padding and residual
links (Op), the int8 quantisation of the input and of every op output, the input padding of the window, and
the reference vector of the on-device window reduction. It exports any sub-chain as QDQ ONNX for ST Edge AI.
    load(id) -> (Workload, ref)     load_all() -> {id: (Workload, ref)}
"""

import json
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .config import WIN, WORKLOADS

OPSET = 13
IDS = ("A0", "A1", "A2")                         # IR-CNN, ResNet1D, Inception1D
_FIELDS = ("stride", "dil", "groups", "relu", "stage", "out_scale", "out_zp", "add_scale", "add_zp")


@dataclass
class Op:
    w: np.ndarray            # (cout, cin/groups, k) float, BN folded
    b: np.ndarray            # (cout,)
    stride: int
    dil: int
    groups: int
    pad: tuple               # (left, right) zero padding before the conv
    relu: bool
    stage: int               # 0..4, 5 = head
    out_scale: float = 1.0
    out_zp: int = 0
    w_scale: np.ndarray = field(default=None)
    res_src: int = None      # residual: add the output of ops[res_src] after this conv
    add_scale: float = 1.0
    add_zp: int = 0

    @property
    def q(self):
        """Quantisation of this op's final output (after the residual add, if any)."""
        if self.res_src is not None:
            return self.add_scale, self.add_zp
        return self.out_scale, self.out_zp


def _qparams(lo, hi):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = max(hi - lo, 1e-8) / 255.0
    zp = int(np.clip(np.round(-128 - lo / scale), -128, 127))
    return float(scale), zp


class Workload:
    """int8 op chain + input quantisation, exportable to QDQ ONNX (whole chain or any segment)."""

    def __init__(self, ops, pad, in_scale, in_zp, meta):
        self.ops, self.pad, self.in_scale, self.in_zp, self.meta = ops, tuple(pad), in_scale, in_zp, meta

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        ops = []
        for i in range(int(z["n_ops"])):
            kw = {k: z[f"op{i}_{k}"].item() for k in _FIELDS}
            kw = {k: (bool(v) if k == "relu" else int(v) if k in ("stride", "dil", "groups", "stage", "out_zp",
                                                                   "add_zp") else float(v)) for k, v in kw.items()}
            r = int(z[f"op{i}_res_src"])
            ops.append(Op(w=z[f"op{i}_w"], b=z[f"op{i}_b"], pad=tuple(int(x) for x in z[f"op{i}_pad"]),
                          w_scale=z[f"op{i}_w_scale"], res_src=None if r < 0 else r, **kw))
        meta = json.loads(str(z["meta"]))
        return cls(ops, z["pad"].tolist(), float(z["in_scale"]), int(z["in_zp"]), meta), z["ref"]

    # ------------------------------------------------------------------ ONNX export
    def to_onnx(self, path, in_len: int, first: int = 0, last: int = None, batch=1):
        """Export ops[first:last] as QDQ ONNX with a float (batch, C, in_len) input (batch="N": dynamic).

        first = 0 means the input is the *padded* window (or a padded-window tile).
        """
        last = len(self.ops) if last is None else last
        ops = self.ops[first:last]
        if first == 0:
            s_in, zp_in, c_in = self.in_scale, self.in_zp, 1
        else:
            prev = self.ops[first - 1]
            (s_in, zp_in), c_in = prev.q, prev.w.shape[0]

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
        s_prev, zp_prev = s_in, zp_in
        names = {first - 1: x}                      # op index -> its output tensor (DQ)
        for i, op in enumerate(ops):
            t = f"l{first + i}"
            if any(op.pad):
                pads = const(f"{t}_pads", np.array([0, 0, 0, op.pad[0], 0, 0, 0, op.pad[1]],
                                                   np.int64))
                nodes.append(helper.make_node("Pad", [x, pads], [f"{t}_padded"], mode="constant"))
                # re-quantise on the same grid so ST keeps Pad+Conv in int8
                x = qdq(f"{t}_padded", s_prev, zp_prev, f"{t}_pad")
            wq = np.clip(np.round(op.w / op.w_scale[:, None, None]), -127, 127).astype(np.int8)
            wq = wq[:, :, None, :]                                   # (cout, cin/g, 1, k)
            bs = (s_prev * op.w_scale).astype(np.float32)
            bq = np.round(op.b / bs).astype(np.int32)
            const(f"{t}_wq", wq)
            const(f"{t}_ws", op.w_scale.astype(np.float32))
            const(f"{t}_wzp", np.zeros(len(op.w_scale), np.int8))
            nodes.append(helper.make_node("DequantizeLinear", [f"{t}_wq", f"{t}_ws", f"{t}_wzp"],
                                          [f"{t}_w"], axis=0))
            const(f"{t}_bq", bq)
            const(f"{t}_bs", bs)
            const(f"{t}_bzp", np.zeros(len(bs), np.int32))
            nodes.append(helper.make_node("DequantizeLinear", [f"{t}_bq", f"{t}_bs", f"{t}_bzp"],
                                          [f"{t}_b"], axis=0))
            nodes.append(helper.make_node("Conv", [x, f"{t}_w", f"{t}_b"], [f"{t}_conv"],
                                          kernel_shape=[1, op.w.shape[2]], strides=[1, op.stride],
                                          dilations=[1, op.dil], group=op.groups,
                                          pads=[0, 0, 0, 0]))
            y = f"{t}_conv"
            if op.relu:
                nodes.append(helper.make_node("Relu", [y], [f"{t}_relu"]))
                y = f"{t}_relu"
            x = qdq(y, op.out_scale, op.out_zp, t)
            if op.res_src is not None:
                assert op.res_src >= first - 1, "residual source outside the exported sub-graph"
                nodes.append(helper.make_node("Add", [x, names[op.res_src]], [f"{t}_add"]))
                x = qdq(f"{t}_add", op.add_scale, op.add_zp, f"{t}_res")
            names[first + i] = x
            s_prev, zp_prev = op.q
        nodes[-1].output[0] = "output"          # final DequantizeLinear is the graph output

        n_out = self._out_len(in_len, first, last)
        g = helper.make_graph(
            nodes, "ras",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, c_in, 1, in_len])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                           [batch, ops[-1].w.shape[0], 1, n_out])],
            inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="rasrt")
        m.ir_version = 8
        onnx.checker.check_model(m)
        onnx.save(m, str(path))
        return path

    def _out_len(self, n, first, last):
        for op in self.ops[first:last]:
            n = n + sum(op.pad)
            n = (n - (op.w.shape[2] - 1) * op.dil - 1) // op.stride + 1
        return n

    def export_full(self, path, batch=1):
        return self.to_onnx(path, WIN + sum(self.pad), batch=batch)


def load(v):
    """(Workload, reference vector) of network v."""
    return Workload.from_npz(WORKLOADS / f"{v}.npz")


def load_all():
    return {v: load(v) for v in IDS}
