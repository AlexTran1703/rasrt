"""Synthetic input stimulus replayed by TIM2 on the board.

The CNN executes a fixed operator sequence, so execution time, CPU time and memory do not depend on the sample
values; the stimulus only has to exercise the full int8 range deterministically. Each window is a seeded sum
of sinusoids plus noise, quantised to the input grid of the network (int8, symmetric, zero point 0).
    windows(n, seed) -> float (n, WIN)        int8_windows(q, n, seed) -> int8 (n, WIN) on q's input grid
"""

import numpy as np

from .config import SEED, WIN
from .predictor import N_REPLAY


def windows(n=N_REPLAY, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(WIN)
    out = np.zeros((n, WIN))
    for i in range(n):
        for _ in range(4):
            f, a, p = rng.uniform(0.002, 0.08), rng.uniform(0.3, 1.5), rng.uniform(0, 2 * np.pi)
            out[i] += a * np.sin(2 * np.pi * f * t + p)
        out[i] += rng.normal(0.0, 0.2, WIN)
    return out


def int8_windows(q, n=N_REPLAY, seed=SEED):
    """Stimulus on the input grid of workload q (what the firmware stores and replays)."""
    return np.clip(np.round(windows(n, seed) / q.in_scale) + q.in_zp, -127, 127).astype(np.int8)
