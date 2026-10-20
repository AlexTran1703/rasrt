/* Window output reduction: running top-16 of the squared distances of the output rows to a reference
 * vector, and the output hash (bit-exactness check); shared by stream.c and profile.c.
 * Needs LATENT, TOPK, OUT_ZP, OUT_SCALE and g_center[LATENT] defined before inclusion. */
#pragma once
#include <stdint.h>

static uint32_t g_hash;
static float g_top[TOPK];
static int g_ntop, g_emb_rows;

/* squared distance of each output row to the reference vector g_center -> running top-16 */
static void score_rows(const int8_t *z, int n)
{
    for (int r = 0; r < n; ++r) {
        const int8_t *row = z + r * LATENT;
        float s = 0.f;
        for (int j = 0; j < LATENT; ++j) {
            g_hash = (g_hash ^ (uint8_t)row[j]) * 16777619u;
            float v = ((float)row[j] - (float)OUT_ZP) * OUT_SCALE - g_center[j];
            s += v * v;
        }
        int i;
        if (g_ntop < TOPK) i = g_ntop++;
        else if (s > g_top[TOPK - 1]) i = TOPK - 1;
        else continue;
        while (i > 0 && g_top[i - 1] < s) { g_top[i] = g_top[i - 1]; --i; }
        g_top[i] = s;
    }
    g_emb_rows += n;
}
