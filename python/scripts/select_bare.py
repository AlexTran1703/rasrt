"""RAS on bare metal: predict every candidate schedule (chunk vector), select one per network, measure it and FW.

    margin eps = 95th-percentile relative error of the cost model on the calibration graphs
    RAS        = lowest predicted CPU s.t. (1 + eps) predicted T_post <= D, predicted SRAM <= B, no ring overrun
Built and measured on the board: the selected schedule and full-window inference (FW) of every network.
Output: results/rasms/e3.json (margin, per network: predicted candidates, selection, measurements),
        results/rasms/measured.json (FW under "E0").
Usage: python rasrt/python/scripts/select_bare.py
"""

import sys

import numpy as np

from _common import read, write
from ras.config import BARE, D_MS, SRAM_MAX_KIB, STUDY
from ras.deploy import deploy, predict, prepare_all
from ras.predictor import Cand, CostModel, candidate_schedules
from ras.stimulus import int8_windows
from ras.workload import load_all

FIELDS = ("label", "m", "t_post_ms", "cpu_ms", "sram", "n_calls", "ring_viol")


def calibration_margin(coef):
    prof = read(STUDY / "calib" / "profile.json")
    X = np.stack([CostModel.features([tuple(f) for f in g["feats"]]) for g in prof["graphs"]])
    y = np.array([g["mean"] for g in prof["graphs"]])
    return float(np.quantile(np.abs(X @ np.asarray(coef["graph"]) - y) / y, 0.95))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coef = read(BARE / "cost.json")
    cost = CostModel(coef)
    eps = calibration_margin(coef)
    e3 = read(BARE / "e3.json")
    fw_meas = read(BARE / "measured.json")
    e3.update({"margin": eps, "deadlines": [D_MS], "sram_max_kib": SRAM_MAX_KIB})
    print(f"calibration margin eps = {eps:.3f}", flush=True)
    for v, (q, ref) in load_all().items():
        cands = candidate_schedules(q)
        prepare_all(q, v, cands, workers=2)
        P = {c.label: predict(q, v, c, cost, coef["memory"][v]) for c in cands}
        ok = [p for p in P.values() if (1 + eps) * p["t_post_ms"] <= D_MS and p["sram"] <= SRAM_MAX_KIB * 1024
              and p["ring_viol"] == 0]
        sel = min(ok, key=lambda p: (p["cpu_ms"], p["sram"]))["label"] if ok else None
        xq = int8_windows(q)
        meas = dict(e3.get(v, {}).get("measured", {}))
        if sel:
            c = next(c for c in cands if c.label == sel)
            meas[sel] = deploy(q, ref, v, c, xq, P[sel])
        fw = Cand(0, 0, 0)
        fw_meas.setdefault(v, {})["E0"] = deploy(q, ref, v, fw, xq, predict(q, v, fw, cost, coef["memory"][v]))
        e3[v] = {"selected": {str(int(D_MS)): sel}, "predicted": {k: {x: p[x] for x in FIELDS} for k, p in P.items()},
                 "built_predicted": {k: P[k] for k in meas if k in P}, "measured": meas}
        m = meas.get(sel, {})
        print(f"{v}: {len(cands)} candidates, {len(ok)} feasible, RAS = {sel}"
              + (f"  T_post {m['t_post_ms']:.1f} ms, CPU {m['cpu_ms']:.1f} ms, SRAM {m['sram'] / 1024:.1f} KiB" if m else ""),
              flush=True)
        write(BARE / "e3.json", e3)
        write(BARE / "measured.json", fw_meas)


if __name__ == "__main__":
    main()
