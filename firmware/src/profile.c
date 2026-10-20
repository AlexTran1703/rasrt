/* Kernel profiling on the STM32F411 @ 100 MHz (calibration of the schedule cost model).
 * Generated prof_config.h / nets.h: N_NETS calibration networks (single operators and short operator
 * chains at lengths that do not occur in any deployed schedule), each run REPS times with
 * pseudo-random int8 inputs; plus the dispatcher primitives used by stream.c: memcpy of COPY_BYTES,
 * the ring-to-chunk feed copy of FEED_N samples, and score_rows on SCORE_ROWS embedding rows.
 * Reported: minimum and total cycles per item, stack high-water of the whole run.
 */
#include <stdint.h>
#include <string.h>

#include "prof_config.h"
#include "nets.h"
#include "board.h"
#include "score.h"

typedef struct {
    uint32_t magic, status, stack_used, reps;
    uint32_t net_min[N_NETS], net_sum[N_NETS];
    uint32_t copy_min[N_COPY], feed_min[N_FEED], score_min[N_SCORE];
} pres_t;

__attribute__((section(".results"))) volatile pres_t g_pres;

static uint8_t g_pool[POOL_BYTES] __attribute__((aligned(32)));
static int8_t g_in0[MAX_IN0] __attribute__((aligned(4)));
static int8_t g_in1[MAX_IN1] __attribute__((aligned(4)));
static int8_t g_src[MAX_COPY] __attribute__((aligned(4))), g_dst[MAX_COPY] __attribute__((aligned(4)));
static int8_t g_raw[RAW_RING], g_chunk[MAX_FEED];
static volatile uint32_t g_sink;

static void fill(int8_t *p, int n, uint32_t seed)
{
    for (int i = 0; i < n; ++i) { seed = seed * 1664525u + 1013904223u; p[i] = (int8_t)(seed >> 24); }
}

int main(void)
{
    memset((void *)&g_pres, 0, sizeof(g_pres));
    board_init();
    if (stai_runtime_init() != STAI_SUCCESS) { g_pres.status = 1; for (;;) {} }
    if (nets_init(g_pool)) { g_pres.status = 2; for (;;) {} }
    fill(g_in0, MAX_IN0, 1u);
    fill(g_in1, MAX_IN1, 2u);
    fill(g_src, MAX_COPY, 3u);
    fill(g_raw, RAW_RING, 4u);
    g_pres.reps = REPS;
    for (int i = 0; i < N_NETS; ++i) {
        stai_ptr in[2] = {(stai_ptr)g_in0, (stai_ptr)g_in1}, out[2];
        stai_size n;
        uint32_t mn = 0xFFFFFFFFu, sum = 0;
        for (int r = 0; r < REPS; ++r) {
            uint32_t t0 = cyc();
            NET_SET_INPUTS[i](NET[i], in, NET_NIN[i]);
            NET_RUN[i](NET[i], STAI_MODE_SYNC);
            NET_GET_OUTPUTS[i](NET[i], out, &n);
            uint32_t d = cyc() - t0;
            sum += d;
            if (d < mn) mn = d;
        }
        g_pres.net_min[i] = mn;
        g_pres.net_sum[i] = sum;
    }
    for (int i = 0; i < N_COPY; ++i) {
        uint32_t mn = 0xFFFFFFFFu;
        for (int r = 0; r < REPS; ++r) {
            uint32_t t0 = cyc();
            memcpy(g_dst, g_src + (r & 3), COPY_BYTES[i]);
            uint32_t d = cyc() - t0;
            if (d < mn) mn = d;
        }
        g_pres.copy_min[i] = mn;
        g_sink += g_dst[0];
    }
    for (int i = 0; i < N_FEED; ++i) {
        uint32_t mn = 0xFFFFFFFFu;
        for (int r = 0; r < REPS; ++r) {
            uint32_t base = 1000u * (uint32_t)r, t0 = cyc();
            for (uint32_t k = 0; k < FEED_N[i]; ++k) g_chunk[k] = g_raw[(base + k) % RAW_RING];
            uint32_t d = cyc() - t0;
            if (d < mn) mn = d;
        }
        g_pres.feed_min[i] = mn;
        g_sink += g_chunk[0];
    }
    for (int i = 0; i < N_SCORE; ++i) {
        uint32_t mn = 0xFFFFFFFFu;
        for (int r = 0; r < REPS; ++r) {
            g_ntop = 0;
            uint32_t t0 = cyc();
            score_rows(g_src, SCORE_ROWS[i]);
            uint32_t d = cyc() - t0;
            if (d < mn) mn = d;
        }
        g_pres.score_min[i] = mn;
    }
    g_pres.stack_used = stack_used();
    g_pres.magic = 0xEC650002u;
    for (;;) {}
}
