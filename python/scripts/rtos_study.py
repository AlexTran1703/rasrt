"""FreeRTOS study: FW and the fixed RAS schedule (selected for bare metal) of every network next to a
higher-priority periodic task tau_H (period = deadline = T_H_MS, utilisation U in U_SWEEP, busy-wait jobs),
CNN task priority 1 < tau_H priority 2, N_WIN_RTOS windows per configuration. T_H = 7 ms does not divide the
10.24 s window, so the release phase of tau_H relative to the final sample drifts by 6 ms per window.
Measured: CNN T_post per window (maximum observed), tau_H jobs, deadline misses and worst response R_H.
Output: results/rtos_T7ms/rtos.json (keys "S1 FW <U>", "S1 RAS <U>"; resumable).
Usage: python rasrt/python/scripts/rtos_study.py
"""

import sys

from _common import read, write
from ras.config import BARE, D_MS, N_WIN_RTOS, RTOS, RTOS_BUILD, T_H_MS, U_SWEEP
from ras.deploy import predict
from ras.predictor import Cand, CostModel, candidate_schedules
from ras.rtos import deploy_rtos
from ras.stimulus import int8_windows
from ras.workload import load_all

KEEP = ("ok", "t_post_ms", "t_post_max_ms", "t_post_win_ms", "sram", "load_n", "load_miss", "load_rmax_ms",
        "cnn_stack_hw", "load_stack_hw", "raw_viol", "ovf", "hash")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coef = read(BARE / "cost.json")
    cost = CostModel(coef)
    e3 = read(BARE / "e3.json")
    f = RTOS / "rtos.json"
    res = read(f)
    res.update({"D_ms": D_MS, "u_sweep": list(U_SWEEP), "t1_ms": T_H_MS, "n_win": N_WIN_RTOS})
    for v, (q, ref) in load_all().items():
        xq = int8_windows(q)
        cands = {c.label: c for c in candidate_schedules(q)}
        r = res.setdefault(v, {})
        for name, c in (("FW", Cand(0, 0, 0)), ("RAS", cands[e3[v]["selected"][str(int(D_MS))]])):
            p = predict(q, v, c, cost, coef["memory"][v])
            for u in U_SWEEP:
                key = f"S1 {name} {u:g}"
                if key in r:
                    continue
                m = deploy_rtos(q, ref, v, c, xq, p, u, T_H_MS, RTOS_BUILD / v / "S1" / f"{c.label}_u{u:g}",
                                N_WIN_RTOS, 1, 2)
                r[key] = {"label": c.label, **{k: m[k] for k in KEEP}}
                print(f"{v} {key:14s} {c.label:22s} ok {m['ok']}  T_post max {m['t_post_max_ms']:7.1f} ms  "
                      f"tau_H miss {m['load_miss']}/{m['load_n']}  R_H max {m['load_rmax_ms']:.3f} ms", flush=True)
                write(f, res, indent=1)


if __name__ == "__main__":
    main()
