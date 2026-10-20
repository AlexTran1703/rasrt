/* Minimal STM32F411 startup: vector table, .data/.bss init, FPU, stack painting. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

#define STACK_PAINT 0xA5A5A5A5u

void Reset_Handler(void);
void Default_Handler(void) { for (;;) {} }
void HardFault_Handler(void);
/* TIM2 (IRQ 28) delivers one input sample every 4 ms (250 Hz) in the streaming builds */
void TIM2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
/* kernel exceptions, provided by the FreeRTOS port in the RTOS builds (FreeRTOSConfig.h maps the names) */
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

#define N_IRQ 30                                  /* external IRQs 0..29 */
__attribute__((section(".isr_vector"), used))
void (*const g_vectors[16 + N_IRQ])(void) = {
    (void (*)(void))(&_estack), Reset_Handler, Default_Handler, HardFault_Handler,
    Default_Handler, Default_Handler, Default_Handler, 0, 0, 0, 0,
    SVC_Handler, Default_Handler, 0, PendSV_Handler, SysTick_Handler,
    [16 ... 16 + N_IRQ - 3] = Default_Handler,
    [16 + 28] = TIM2_IRQHandler,
    [16 + 29] = TIM3_IRQHandler,
};

/* Fault indicator (symbol readable over SWD) */
volatile uint32_t g_fault;
void HardFault_Handler(void) { g_fault = 0xFA017u; for (;;) {} }

__attribute__((naked, noreturn)) void Reset_Handler(void)
{
    __asm volatile("ldr sp, =_estack");
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ++dst) *dst = 0;
    /* paint the free RAM between .bss and the current stack for high-water measurement */
    uint32_t sp;
    __asm volatile("mov %0, sp" : "=r"(sp));
    for (dst = &_ebss; (uint32_t)dst < sp - 64u; ++dst) *dst = STACK_PAINT;
    /* FPU: full access to CP10/CP11 */
    *(volatile uint32_t *)0xE000ED88u |= (0xFu << 20);
    __asm volatile("dsb\n isb");
    main();
    for (;;) {}
}
