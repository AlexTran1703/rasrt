"""Calibrate the cost model on the board (kernel profiling only) and the memory constants.
Output: results/rasms/cost.json (graph-call and operator coefficients, copy / feed / reduction terms, memory
constants per network); profiling data under <build>/study/calib.
Usage: python rasrt/python/scripts/calibrate.py
"""

import sys

from _common import write
from ras import calibration
from ras.config import BARE
from ras.deploy import calibrate_memory
from ras.predictor import CostModel
from ras.stimulus import int8_windows
from ras.workload import load_all


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nets = load_all()
    res = calibration.profile({v: q for v, (q, _) in nets.items()})
    coef = calibration.fit(res)
    cost = CostModel(coef)
    coef["memory"] = {v: calibrate_memory(q, ref, v, int8_windows(q), cost, res["frames"], max(res["stack"]))
                      for v, (q, ref) in nets.items()}
    write(BARE / "cost.json", coef, indent=1)
    print("cost model:", {k: coef[k] for k in ("fit_rel_err", "n_graphs")}, flush=True)


if __name__ == "__main__":
    main()
