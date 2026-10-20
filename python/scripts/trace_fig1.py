"""Fig. 1 traces: IR-CNN under FreeRTOS at U_H = 20 % (T_H = 7 ms) for FW, the fixed RAS schedule and the RAS-RT
schedule, same configuration as the study (8 windows, CNN priority 1 < tau_H priority 2) with TRACE_LOAD = 1, so
the CNN segment jobs of the traced window and tau_H's jobs around its end are recorded.
Output: <build>/rtos/fig1/<label>_u0.2_w<window>/measured.json (read by report.py).
Usage: python rasrt/python/scripts/trace_fig1.py [traced window, default 6] [FW|RAS|RAS-RT ...]
"""

import sys

from _common import read
from ras.config import BARE, D_MS, N_WIN_RTOS, RTOS, RTOS_BUILD, T_H_MS
from ras.deploy import predict
from ras.predictor import Cand, CostModel, candidate_schedules
from ras.rtos import deploy_rtos
from ras.stimulus import int8_windows
from ras.workload import load

V, U = "A0", 0.2
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ONLY = set(sys.argv[2:]) or {"FW", "RAS", "RAS-RT"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coef = read(BARE / "cost.json")
    cost = CostModel(coef)
    q, ref = load(V)
    xq = int8_windows(q)
    cands = {c.label: c for c in candidate_schedules(q)}
    rt = read(RTOS / "rtos_aware.json")[V][f"{U:g}"]["selected"]
    fixed = read(BARE / "e3.json")[V]["selected"][str(int(D_MS))]
    for name, c in (("FW", Cand(0, 0, 0)), ("RAS", cands[fixed]), ("RAS-RT", cands[rt])):
        if name not in ONLY:
            continue
        m = deploy_rtos(q, ref, V, c, xq, predict(q, V, c, cost, coef["memory"][V]), U, T_H_MS,
                        RTOS_BUILD / "fig1" / f"{c.label}_u{U:g}_w{WIN}", N_WIN_RTOS, 1, 2, trace_load=True,
                        trace_win=WIN)
        print(f"{name} {c.label}: ok {m['ok']}  T_post window {WIN} {m['t_post_win_ms'][WIN]:.1f} ms  "
              f"CNN jobs {len(m['cnn_trace'])}  tau_H jobs {len(m['load_trace'])}", flush=True)


if __name__ == "__main__":
    main()
