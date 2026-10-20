"""Paths and constants of the RAS / RAS-RT pipeline.

Paths can be redirected with environment variables:
    RASRT_BUILD    generated graphs and firmware builds (default rasrt/build; layout <build>/study/<net>/...,
                   <build>/rtos/...)
    RASRT_RESULTS  measured and predicted data (default <repo>/results; bare metal in rasms/, FreeRTOS in rtos_T7ms/)
"""

import os
from pathlib import Path

RASRT = Path(__file__).resolve().parents[2]
REPO = RASRT.parent
WORKLOADS = RASRT / "workloads"
FIRMWARE = RASRT / "firmware"
BUILD = Path(os.environ.get("RASRT_BUILD", RASRT / "build"))
STUDY = BUILD / "study"                # per network: graphs/, builds/ (bare metal); calib/
RTOS_BUILD = BUILD / "rtos"            # FreeRTOS builds; fig1/ traces
RESULTS = Path(os.environ.get("RASRT_RESULTS", REPO / "results"))
BARE = RESULTS / "rasms"               # cost.json, e3.json (RAS candidates and selection), measured.json (FW)
RTOS = RESULTS / "rtos_T7ms"           # rtos.json (FW, fixed RAS), rtos_aware.json (RAS-RT)

# input stream and network output
FS = 250                               # Hz, one sample per TIM2 interrupt
WIN = 2560                             # samples per non-overlapping window (10.24 s)
EMB_STRIDE = 16                        # input samples per output row (total stride of every network)
N_EMB = WIN // EMB_STRIDE              # 160 output rows per window
LATENT = 16                            # channels per output row
TOPK = 16                              # window output = mean of the TOPK largest row distances
SEED = 2026

# MCU: STM32F411 @ 100 MHz
MCU_HZ = 100_000_000

# schedules and selection
M_SET = (2, 5, 10, 20)                 # admissible chunk sizes m_j (output rows per segment job)
SRAM_MAX_KIB = 48                      # reserved-SRAM budget B
D_MS = 30.0                            # post-window deadline D

# FreeRTOS study: periodic task tau_H
T_H_MS = 7                             # period = implicit deadline
U_SWEEP = (0.0, 0.2, 0.4, 0.6)         # utilisations of the reported study
U_RTOS = (0.2, 0.4, 0.6)               # utilisations at which RAS-RT reselects
PHASES = 14                            # release phases of tau_H evaluated per candidate
N_WIN_RTOS = 8                         # windows per FreeRTOS configuration
