/* Schedule c = (d, m, rho) of an int8 1-D CNN on the STM32F411 (generated config: bench_config.h,
 * nets.h). Segments 0..NSEG-1 of the op chain are ST Edge AI networks run chunk by chunk; segment j
 * keeps an input FIFO of SEG_CAP[j] rows (one run reads SEG_NIN[j] rows = new rows + the k - s
 * stream-buffer rows, then advances by SEG_NNEW[j]); its outputs leave the shared activation pool at
 * once and are pushed to segment j + 1 (with that conv's zero padding as zero-point rows at the
 * stream start / end). A residual source -> consumer pair crossing segments gets a row buffer of
 * RES_CAP rows; a row is retired after the consumer run that adds it. HAS_S2W: the rest of the
 * network is one whole-window network ("tail") on S2W_LEN stored rows, run after the window.
 *   RELEASE 1: segment 0 receives each RAW_CHUNK of samples as soon as it has arrived
 *   RELEASE 0: after the window (post-acquisition)
 * Window output = mean of the running top-16 row distances to a reference vector (score.h; no output matrix
 * is stored).
 * TIM2 replays N_WIN stimulus windows back to back at 250 Hz. Per window: T_post = DWT(output) - DWT(last
 * sample ISR), CPU = all main-loop work. Instrumentation (.results, not counted as deployment SRAM):
 * embedding hash, FIFO / residual / tail-input high-water marks, overflow and ring checks, and the
 * job trace of window TRACE_WIN (id = segment, 15 = tail; start, end cycles).
 * RTOS 1 (FreeRTOS, fixed-priority preemptive): the dispatcher runs as the low-priority task "cnn"; a
 * higher-priority periodic task "load" (period LOAD_T_MS, LOAD_C_CYC busy cycles per job; LOAD_C_CYC = 0: no
 * load) competes for the CPU. N_WIN windows replay the N_DATA stimulus windows cyclically. Appended results:
 * load jobs, deadline misses (response > period), worst response, task stack high-water marks.
 */
#include <stdint.h>
#include <string.h>

#include "bench_config.h"
#include "nets.h"
#include "board.h"
#include "score.h"
#ifndef RTOS
#define RTOS 0
#endif
#ifndef N_DATA
#define N_DATA N_WIN
#endif
#ifndef TRACE_LOAD
#define TRACE_LOAD 0                 /* 1: trace the periodic task's jobs around TRACE_WIN's end (Fig. 1 builds) */
#endif
#define N_LOAD_TR 160
#ifndef LOAD_PHASE_STEP
#define LOAD_PHASE_STEP 0            /* ticks added to the periodic task's release at each window boundary */
#endif
#if RTOS
#include "FreeRTOS.h"
#include "task.h"
#ifndef CNN_PRIO
#define CNN_PRIO 1
#endif
#ifndef LOAD_PRIO
#define LOAD_PRIO 2
#endif
#define CYC_PER_MS 100000u
static TaskHandle_t g_cnn_h;
#define WAIT_SAMPLES() ulTaskNotifyTake(pdTRUE, portMAX_DELAY)      /* block until the next sample */
#else
#define WAIT_SAMPLES() ((void)0)
#endif

typedef struct {
    uint32_t magic, status, nsamp, raw_viol, ovf, ring_max, stack_used, isr_max, s2in_hw;
    uint32_t t_resp[N_WIN], cpu[N_WIN], hash[N_WIN];
    float score[N_WIN];
    uint32_t seg_hw[NSEG_ALLOC], res_hw[NRES_ALLOC];
    uint32_t trace_n, trace_t_last;
    uint32_t trace[N_TRACE][3];
#if RTOS
    uint32_t load_n, load_miss, load_rmax, cnn_stack_hw, load_stack_hw, rtos_assert;
    uint32_t tick_cyc[256];          /* cycle count of the last 256 ticks (task release times; instrumentation) */
#if TRACE_LOAD
    uint32_t load_tn, load_tr[N_LOAD_TR][2];   /* periodic-task jobs (start, end cycles) around TRACE_WIN's end */
#endif
#endif
} rres_t;

__attribute__((section(".results"))) volatile rres_t g_rres;

static uint8_t g_pool[POOL_BYTES] __attribute__((aligned(32)));
static int8_t g_segbuf[SEGBUF_TOTAL] __attribute__((aligned(4)));
#if NRES
static int8_t g_resbuf[RESBUF_TOTAL] __attribute__((aligned(4)));
static int g_res_rows[NRES];
#endif
#if HAS_S2W
static int8_t g_s2in[S2W_LEN * S2W_CIN] __attribute__((aligned(4)));
static int g_s2in_rows;
#endif
static const int8_t g_padrows[PADROWS_BYTES];
static int8_t g_raw[RAW_RING];
static int8_t g_chunk[RAW_CHUNK];

/* ------------------------------------------------------------------ 250 Hz replay (TIM2) */
static volatile uint32_t g_k, g_isr_max;
static volatile uint32_t g_t_end[N_WIN];
#define TOTAL (N_WIN * WIN_SAMPLES)

void TIM2_IRQHandler(void)
{
    REG(0x40000010) = 0;
    uint32_t t = cyc(), k = g_k;
    if (k < TOTAL) {
        g_raw[k % RAW_RING] = g_stimulus[(k / WIN_SAMPLES) % N_DATA][k % WIN_SAMPLES];
        g_k = k + 1;
        if ((k + 1) % WIN_SAMPLES == 0) g_t_end[k / WIN_SAMPLES] = t;
        if (k + 1 == TOTAL) REG(0x40000000) = 0;
        uint32_t d = cyc() - t;
        if (d > g_isr_max) g_isr_max = d;
#if RTOS
        BaseType_t woken = pdFALSE;
        vTaskNotifyGiveFromISR(g_cnn_h, &woken);
        portYIELD_FROM_ISR(woken);
#endif
    }
    __asm volatile("dsb");
}

static void timer_start(void)
{
#if RTOS
    *(volatile uint8_t *)(0xE000E400u + 28) = configMAX_SYSCALL_INTERRUPT_PRIORITY;  /* highest kernel-aware */
#else
    *(volatile uint8_t *)(0xE000E400u + 28) = 0x00;          /* TIM2 highest priority */
#endif
    REG(0x40023840) |= 1u;
    REG(0x40000000) = 0;
    REG(0x40000028) = 99;                                    /* 1 MHz tick */
    REG(0x4000002C) = 1000000u / FS_HZ - 1;
    REG(0x40000024) = 0;
    REG(0x40000014) = 1;
    REG(0x40000010) = 0;
    REG(0x4000000C) = 1;
    REG(0xE000E100) = (1u << 28);
    REG(0x40000000) = 1;
}

/* ------------------------------------------------------------------ per-window state */
static int g_fill[NSEG_ALLOC], g_prod[NSEG_ALLOC], g_started[NSEG_ALLOC];
static int g_trace_on;

static inline void trace_job(uint32_t id, uint32_t t0, uint32_t t1)
{
    if (g_trace_on && g_rres.trace_n < N_TRACE) {
        g_rres.trace[g_rres.trace_n][0] = id;
        g_rres.trace[g_rres.trace_n][1] = t0;
        g_rres.trace[g_rres.trace_n][2] = t1;
        g_rres.trace_n++;
    }
}

static void push(int j, const int8_t *rows, int n, int final);

static void run_seg(int j)
{
    int8_t *buf = g_segbuf + SEG_OFF[j];
    const int cin = SEG_CIN[j];
    if (g_fill[j] < SEG_NIN[j])                              /* end of stream: zero-point fill */
        memset(buf + g_fill[j] * cin, SEG_ZP[j], (SEG_NIN[j] - g_fill[j]) * cin);
    stai_ptr in[2] = {(stai_ptr)buf, 0}, out[2];
    stai_size n_out;
#if NRES
    if (SEG_RES_IN[j]) in[1] = (stai_ptr)(g_resbuf + RES_OFF[SEG_RES_IN[j] - 1]);
#endif
    uint32_t t0 = cyc();
    NET_SET_INPUTS[j](NET[j], in, SEG_RES_IN[j] ? 2 : 1);
    NET_RUN[j](NET[j], STAI_MODE_SYNC);
    NET_GET_OUTPUTS[j](NET[j], out, &n_out);
    trace_job((uint32_t)j, t0, cyc());
    int keep = SEG_NOUT[j];
    if (keep > SEG_LEN_OUT[j] - g_prod[j]) keep = SEG_LEN_OUT[j] - g_prod[j];
    g_prod[j] += keep;
    int last = g_prod[j] >= SEG_LEN_OUT[j];
#if NRES
    if (SEG_RES_IN[j]) {                                     /* retire the added residual rows */
        int r = SEG_RES_IN[j] - 1;
        int8_t *rb = g_resbuf + RES_OFF[r];
        g_res_rows[r] -= keep;
        memmove(rb, rb + keep * RES_CH[r], g_res_rows[r] * RES_CH[r]);
    }
    if (SEG_RES_OUT[j]) {                                    /* residual source rows */
        int r = SEG_RES_OUT[j] - 1, nr = keep;
        if (g_res_rows[r] + nr > RES_CAP[r]) { g_rres.ovf++; nr = RES_CAP[r] - g_res_rows[r]; }
        memcpy(g_resbuf + RES_OFF[r] + g_res_rows[r] * RES_CH[r], out[n_out - 1], nr * RES_CH[r]);
        g_res_rows[r] += nr;
        if ((uint32_t)g_res_rows[r] > g_rres.res_hw[r]) g_rres.res_hw[r] = g_res_rows[r];
    }
#endif
    int adv = SEG_NNEW[j] < g_fill[j] ? SEG_NNEW[j] : g_fill[j];
    memmove(buf, buf + adv * cin, (g_fill[j] - adv) * cin);
    g_fill[j] -= adv;
    if (j + 1 < NSEG)                                        /* outputs live in the shared pool */
        push(j + 1, (const int8_t *)out[0], keep, last);
    else
#if HAS_S2W
        push(NSEG, (const int8_t *)out[0], keep, last);
#else
        score_rows((const int8_t *)out[0], keep);
#endif
}

/* append n rows (and stream padding) to segment j; run it while a full chunk is buffered */
static void push(int j, const int8_t *rows, int n, int final)
{
#if HAS_S2W
    if (j >= NSEG) {                                         /* rows of the whole-window tail input */
        if (g_s2in_rows + n > S2W_LEN) { g_rres.ovf++; n = S2W_LEN - g_s2in_rows; }
        memcpy(g_s2in + g_s2in_rows * S2W_CIN, rows, n * S2W_CIN);
        g_s2in_rows += n;
        if ((uint32_t)g_s2in_rows > g_rres.s2in_hw) g_rres.s2in_hw = g_s2in_rows;
        return;
    }
#endif
    int8_t *buf = g_segbuf + SEG_OFF[j];
    const int cin = SEG_CIN[j];
    int add = n + (g_started[j] ? 0 : SEG_PADL[j]) + (final ? SEG_PADR[j] : 0);
    if (g_fill[j] + add > SEG_CAP[j]) { g_rres.ovf++; return; }
    if (!g_started[j]) {
        g_started[j] = 1;
        memset(buf + g_fill[j] * cin, SEG_ZP[j], SEG_PADL[j] * cin);
        g_fill[j] += SEG_PADL[j];
    }
    memcpy(buf + g_fill[j] * cin, rows, n * cin);
    g_fill[j] += n;
    if (final) {
        memset(buf + g_fill[j] * cin, SEG_ZP[j], SEG_PADR[j] * cin);
        g_fill[j] += SEG_PADR[j];
    }
    if ((uint32_t)g_fill[j] > g_rres.seg_hw[j]) g_rres.seg_hw[j] = g_fill[j];
    while (g_prod[j] < SEG_LEN_OUT[j] && (g_fill[j] >= SEG_NIN[j] || (final && g_fill[j] > 0)))
        run_seg(j);
}

/* raw samples [from, to) of the window starting at stream index base -> segment 0, RAW_CHUNK at a time */
static void feed(uint32_t base, uint32_t from, uint32_t to)
{
    while (from < to) {
        uint32_t n = to - from < RAW_CHUNK ? to - from : RAW_CHUNK;
        uint32_t lag = g_k - (base + from);
        if (lag > g_rres.ring_max) g_rres.ring_max = lag;
        if (lag > RAW_RING) g_rres.raw_viol++;
        for (uint32_t i = 0; i < n; ++i) g_chunk[i] = g_raw[(base + from + i) % RAW_RING];
        from += n;
        push(0, g_chunk, (int)n, 0);
    }
}

static void run_windows(void)
{
    for (int w = 0; w < N_WIN; ++w) {
        uint32_t base = (uint32_t)w * WIN_SAMPLES, done = 0, busy = 0, t0;
        for (int j = 0; j < NSEG; ++j) { g_fill[j] = 0; g_prod[j] = 0; g_started[j] = 0; }
#if NRES
        for (int r = 0; r < NRES; ++r) g_res_rows[r] = 0;
#endif
#if HAS_S2W
        g_s2in_rows = 0;
#endif
        g_hash = 2166136261u;
        g_ntop = 0;
        g_emb_rows = 0;
        g_trace_on = (w == TRACE_WIN);
        t0 = cyc();
        push(0, g_padrows, PAD_L, 0);                        /* global stream padding */
        busy += cyc() - t0;
#if RELEASE
        while (done < WIN_SAMPLES) {
            uint32_t avail = g_k - base;
            if (avail > WIN_SAMPLES) avail = WIN_SAMPLES;
            if (avail - done >= RAW_CHUNK || avail == WIN_SAMPLES) {
                /* whole chunks only (a late dispatcher must not feed a partial chunk: the stream buffers are
                 * sized for chunk-aligned input); the window's remainder at its end */
                uint32_t upto = avail == WIN_SAMPLES ? avail : done + (avail - done) / RAW_CHUNK * RAW_CHUNK;
                t0 = cyc();
                feed(base, done, upto);
                busy += cyc() - t0;
                done = upto;
            } else {
                WAIT_SAMPLES();
            }
        }
#else
        while (g_k < base + WIN_SAMPLES) WAIT_SAMPLES();
        t0 = cyc();
        feed(base, 0, WIN_SAMPLES);
        busy += cyc() - t0;
#endif
        uint32_t t_last = g_t_end[w];
        t0 = cyc();
        push(0, g_padrows, PAD_R, 1);                        /* flush */
#if HAS_S2W
        {
            stai_ptr in[1] = {(stai_ptr)g_s2in}, out[1];
            stai_size n;
            uint32_t ts = cyc();
            stai_tail_set_inputs(TAIL_NET, in, 1);
            stai_tail_run(TAIL_NET, STAI_MODE_SYNC);
            stai_tail_get_outputs(TAIL_NET, out, &n);
            trace_job(15, ts, cyc());
            score_rows((const int8_t *)out[0], N_EMB);
        }
#endif
        float s = 0.f;
        for (int i = 0; i < g_ntop; ++i) s += g_top[i];
        s /= (float)g_ntop;
        uint32_t t1 = cyc();
        busy += t1 - t0;
        if (g_trace_on) g_rres.trace_t_last = t_last;
        g_rres.t_resp[w] = t1 - t_last;
        g_rres.cpu[w] = busy;
        g_rres.hash[w] = g_hash;
        g_rres.score[w] = s;
        if (g_emb_rows != N_EMB) g_rres.status = 100 + (uint32_t)g_emb_rows;
    }
    while (g_k < TOTAL) WAIT_SAMPLES();
}

static void finish(void)
{
    g_rres.nsamp = g_k;
    g_rres.isr_max = g_isr_max;
    g_rres.stack_used = stack_used();
    g_rres.magic = 0xEC650002u;
}

#if RTOS
/* ------------------------------------------------------------------ FreeRTOS tasks */
#define CNN_STACK_WORDS  768
#define LOAD_STACK_WORDS 128
volatile unsigned int g_rtos_assert;
static StackType_t g_cnn_stack[CNN_STACK_WORDS], g_load_stack[LOAD_STACK_WORDS];
static StaticTask_t g_cnn_tcb, g_load_tcb;
static TaskHandle_t g_load_h;


void vApplicationTickHook(void) { g_rres.tick_cyc[xTaskGetTickCountFromISR() & 255u] = cyc(); }

/* periodic competing task: released every LOAD_T_MS ticks, LOAD_C_CYC cycles of computation, deadline = period;
 * response = completion - release (DWT at the release tick, recorded by the tick hook): includes
 * release-to-dispatch delay, preemption by interrupts, and blocking by the CNN if the CNN has the higher priority */
static void load_task(void *arg)
{
    (void)arg;
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
#if LOAD_PHASE_STEP
        {   /* shift the release phase by LOAD_PHASE_STEP ticks at every window boundary, so that a period dividing
             * the window still meets each window's last sample at a different alignment */
            static uint32_t last_w;
            uint32_t w = g_k / WIN_SAMPLES;
            if (w != last_w) { last_w = w; wake += LOAD_PHASE_STEP; }
        }
#endif
        xTaskDelayUntil(&wake, pdMS_TO_TICKS(LOAD_T_MS));
        uint32_t rel = g_rres.tick_cyc[wake & 255u], t = cyc(), done = 0;   /* blocking < 256 ms */
#if TRACE_LOAD
        uint32_t t_start = t;
#endif
        while (done < LOAD_C_CYC) {                              /* own computation only (preemption excluded) */
            uint32_t now = cyc(), d = now - t;
            if (d < 200u) done += d;                             /* larger gaps: preempted, not own work */
            t = now;
        }
        uint32_t r = cyc() - rel;
#if TRACE_LOAD
        {   /* jobs of window TRACE_WIN from 0.6 s before to 0.3 s after its last sample (figure builds only) */
            uint32_t k = g_k, end = (uint32_t)(TRACE_WIN + 1) * WIN_SAMPLES;
            if (k + 150u >= end && k < end + 75u && g_rres.load_tn < N_LOAD_TR) {
                g_rres.load_tr[g_rres.load_tn][0] = t_start;
                g_rres.load_tr[g_rres.load_tn][1] = cyc();
                g_rres.load_tn++;
            }
        }
#endif
        if (g_rres.magic) continue;                              /* statistics only while the CNN runs */
        g_rres.load_n++;
        if (r > g_rres.load_rmax) g_rres.load_rmax = r;
        if (r > (uint32_t)LOAD_T_MS * CYC_PER_MS) g_rres.load_miss++;
    }
}

static void cnn_task(void *arg)
{
    (void)arg;
    timer_start();
    run_windows();
    g_rres.cnn_stack_hw = 4u * (CNN_STACK_WORDS - uxTaskGetStackHighWaterMark(NULL));
    if (g_load_h) g_rres.load_stack_hw = 4u * (LOAD_STACK_WORDS - uxTaskGetStackHighWaterMark(g_load_h));
    g_rres.rtos_assert = g_rtos_assert;
    finish();
    vTaskSuspend(NULL);
}
#endif

int main(void)
{
    memset((void *)&g_rres, 0, sizeof(g_rres));
    board_init();
    if (stai_runtime_init() != STAI_SUCCESS) { g_rres.status = 1; for (;;) {} }
    if (nets_init(g_pool)) { g_rres.status = 2; for (;;) {} }
#if RTOS
    g_cnn_h = xTaskCreateStatic(cnn_task, "cnn", CNN_STACK_WORDS, 0, CNN_PRIO, g_cnn_stack, &g_cnn_tcb);
    if (LOAD_C_CYC > 0)
        g_load_h = xTaskCreateStatic(load_task, "load", LOAD_STACK_WORDS, 0, LOAD_PRIO, g_load_stack, &g_load_tcb);
    vTaskStartScheduler();
    g_rres.status = 3;                                           /* scheduler returned: not expected */
    for (;;) {}
#else
    timer_start();
    run_windows();
    finish();
    for (;;) {}
#endif
}
