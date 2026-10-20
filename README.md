# RAS / RAS-RT: rate-aware streaming schedules of int8 CNNs on an MCU

A quantized 1-D CNN is cut into segments before every convolution with kernel size k > 1. Each segment is a
separate ST Edge AI graph that runs on chunks of its input while samples arrive, with a stream buffer of
k - s rows so that the chunked result is bit-identical to full-window inference (FW). A schedule is a chunk
vector **m** = (m_0, ..., m_{L-1}), m_j in {2, 5, 10, 20}, non-decreasing with depth.

- **RAS** (bare metal) predicts the CPU time, post-window response and reserved SRAM of every candidate
  schedule by simulating the firmware dispatcher with a calibrated cost model, and selects the lowest
  predicted CPU time subject to (1 + eps) T_post <= D, SRAM <= B and no input-ring overrun.
- **RAS-RT** (FreeRTOS) adds fixed-priority preemption by a periodic task tau_H to the simulation, takes the
  worst of 14 release phases, and reselects the chunk vector per utilisation without changing task priorities.

Platform: STM32F411 (Cortex-M4F, 100 MHz, 128 KiB SRAM), ST Edge AI 3.0, FreeRTOS V11.1, DWT cycle counter.
