/* STM32F411 @ 100 MHz: clock, DWT cycle counter, stack high-water (shared by stream.c / profile.c). */
#pragma once
#include <stdint.h>

#define REG(a) (*(volatile uint32_t *)(a))

extern uint32_t _ebss, _estack;

static void board_init(void)
{
    REG(0xE0042004) |= 7u;                                   /* debug in low-power modes */
    REG(0x40023840) |= (1u << 28);                           /* PLL: HSI 16 MHz -> 100 MHz */
    REG(0x40007000) |= (3u << 14);
    REG(0x40023C00) = 3u | (1u << 8) | (1u << 9) | (1u << 10);
    REG(0x40023804) = 8u | (100u << 6) | (0u << 16) | (0u << 22) | (4u << 24);
    REG(0x40023800) |= (1u << 24);
    while (!(REG(0x40023800) & (1u << 25))) {}
    REG(0x40023808) = (REG(0x40023808) & ~0xFCF3u) | (4u << 10) | 2u;
    while (((REG(0x40023808) >> 2) & 3u) != 2u) {}
    REG(0xE000EDFC) |= (1u << 24);                           /* DWT cycle counter */
    REG(0xE0001004) = 0;
    REG(0xE0001000) |= 1u;
}

static inline uint32_t cyc(void) { return REG(0xE0001004); }

/* bytes of the painted region (startup.c) overwritten by the stack */
static uint32_t stack_used(void)
{
    uint32_t *p = &_ebss;
    while (p < &_estack && *p == 0xA5A5A5A5u) ++p;
    return (uint32_t)((uint8_t *)&_estack - (uint8_t *)p);
}
