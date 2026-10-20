"""RAS-RT: reselect the chunk vector of every network for each utilisation U of tau_H and measure it under FreeRTOS.

    R_post(s, tau_H) = max over PHASES release phases of the dispatcher simulation with tau_H's jobs
                       [phi + k T_H, phi + k T_H + C_H) preempting the CNN (includes the critical instant)
    RAS-RT(U)        = lowest predicted CPU s.t. (1 + eps) R_post <= D, predicted SRAM + M_RTOS <= B, no ring overrun
M_RTOS = largest measured FreeRTOS SRAM overhead (FW and fixed RAS at U = 0 vs bare metal). If no candidate
qualifies, no margin-feasible schedule is reported; the configuration with the lowest predicted CPU among those
meeting D without margin (or, if none, the lowest predicted response) is still built and measured.
Output: results/rtos_T7ms/rtos_aware.json (resumable).
Usage: python rasrt/python/scripts/rtos_select.py
"""

import sys

from _common import keep, read, write
from ras.config import BARE, D_MS, N_WIN_RTOS, PHASES, RTOS, RTOS_BUILD, SRAM_MAX_KIB, T_H_MS, U_RTOS
from ras.deploy import predict
from ras.predictor import CostModel, candidate_schedules, geometry, simulate
from ras.rtos import deploy_rtos
from ras.stimulus import int8_windows
from ras.workload import IDS, load_all


def worst_post(q, c, cost, u):
    """Predicted R_post: maximum simulated post-window response over PHASES release phases of tau_H."""
    P, g = T_H_MS / 1000, geometry(q, c)
    return max(max(simulate(q, c, cost, g, interf=(u * P, P, P * k / PHASES))["t_post"]) for k in range(PHASES))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coef = read(BARE / "cost.json")
    cost = CostModel(coef)
    e3, fw_meas = read(BARE / "e3.json"), read(BARE / "measured.json")
    R = read(RTOS / "rtos.json")
    eps, key_d = e3["margin"], str(int(D_MS))
    bare = lambda v, lab: fw_meas.get(v, {}).get(lab) or e3[v]["measured"][lab]  # noqa: E731
    m_rtos = max(R[v][f"S1 {s} 0"]["sram"] - bare(v, lab)["sram"]
                 for v in IDS for s, lab in (("FW", "E0"), ("RAS", e3[v]["selected"][key_d])))
    f = RTOS / "rtos_aware.json"
    res = read(f)
    res.update({"D_ms": D_MS, "t_ms": T_H_MS, "phases": PHASES, "margin": eps, "m_rtos_bytes": m_rtos,
                "budget_kib": SRAM_MAX_KIB})
    for v, (q, ref) in load_all().items():
        r, pred = res.setdefault(v, {}), e3[v]["predicted"]
        fixed = e3[v]["selected"][key_d]
        cands = {c.label: c for c in candidate_schedules(q)
                 if pred[c.label]["sram"] + m_rtos <= SRAM_MAX_KIB * 1024 and pred[c.label]["ring_viol"] == 0}
        xq = int8_windows(q)
        for u in U_RTOS:
            key = f"{u:g}"
            if r.get(key, {}).get("done"):
                continue
            post = {lab: worst_post(q, c, cost, u) for lab, c in cands.items()}
            cpu = lambda lab: (pred[lab]["cpu_ms"], pred[lab]["sram"])  # noqa: E731
            feas = [lab for lab in post if (1 + eps) * post[lab] <= D_MS]
            point = [lab for lab in post if post[lab] <= D_MS]
            sel = min(feas, key=cpu) if feas else None
            test, kind = (sel, "selected") if sel else ((min(point, key=cpu), "point") if point
                                                        else (min(post, key=post.get), "best effort"))
            e = {"fixed": fixed, "fixed_pred_post_ms": post.get(fixed, worst_post(q, cands[fixed], cost, u)),
                 "selected": sel, "n_feasible": len(feas), "n_point_feasible": len(point), "n_candidates": len(post),
                 "tested": test, "tested_kind": kind, "pred_post_ms": post[test], "pred_cpu_ms": pred[test]["cpu_ms"],
                 "pred_sram": pred[test]["sram"]}
            if test == fixed and f"S1 RAS {key}" in R[v]:
                e["measured"] = keep(R[v][f"S1 RAS {key}"])
            else:
                c = cands[test]
                m = deploy_rtos(q, ref, v, c, xq, predict(q, v, c, cost, coef["memory"][v]), u, T_H_MS,
                                RTOS_BUILD / v / "S1_aware" / f"{test}_u{key}", N_WIN_RTOS, 1, 2)
                e["measured"] = keep(m)
            if sel:
                b = fw_meas.get(v, {}).get(sel) or e3[v]["measured"].get(sel)
                e["bare_cpu_ms"] = b["cpu_ms"] if b else None
            e["done"] = True
            r[key] = e
            ms = e["measured"]
            print(f"{v} U={key}: fixed {fixed} (pred {e['fixed_pred_post_ms']:.1f} ms) -> {kind} {test} "
                  f"({len(feas)}/{len(post)} margin-feasible)  measured {ms['t_post_max_ms']:.1f} ms, ok {ms['ok']}",
                  flush=True)
            write(f, res, indent=1)


if __name__ == "__main__":
    main()
