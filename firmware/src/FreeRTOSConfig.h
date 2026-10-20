/* FreeRTOS V11 configuration of the RTOS builds (stream.c with RTOS = 1): fixed-priority preemptive scheduling,
 * static allocation only (every kernel object and task stack is in .bss, i.e. inside the reserved-SRAM count),
 * 1 kHz tick. TIM2 (sample replay) runs at configMAX_SYSCALL_INTERRUPT_PRIORITY, the highest kernel-aware level, so
 * it may notify the CNN task (vTaskNotifyGiveFromISR) and still preempts every task, SysTick and PendSV. */
#pragma once

#define configCPU_CLOCK_HZ                       100000000u
#define configTICK_RATE_HZ                       1000u
#define configUSE_PREEMPTION                     1
#define configUSE_TIME_SLICING                   0
#define configUSE_PORT_OPTIMISED_TASK_SELECTION  1
#define configMAX_PRIORITIES                     4
#define configMINIMAL_STACK_SIZE                 128
#define configMAX_TASK_NAME_LEN                  8
#define configTICK_TYPE_WIDTH_IN_BITS            TICK_TYPE_WIDTH_32_BITS
#define configIDLE_SHOULD_YIELD                  1

#define configSUPPORT_STATIC_ALLOCATION          1
#define configSUPPORT_DYNAMIC_ALLOCATION         0
#define configKERNEL_PROVIDED_STATIC_MEMORY      1

#define configUSE_IDLE_HOOK                      0
#define configUSE_TICK_HOOK                      1   /* records the cycle count of each tick (task release times) */
#define configUSE_TIMERS                         0
#define configTIMER_TASK_STACK_DEPTH             128 /* referenced by the kernel-provided memory; unused */
#define configUSE_MUTEXES                        0
#define configUSE_COUNTING_SEMAPHORES            0
#define configUSE_TASK_NOTIFICATIONS             1
#define configUSE_TRACE_FACILITY                 0
#define configCHECK_FOR_STACK_OVERFLOW           0
#define configUSE_MALLOC_FAILED_HOOK             0

#define INCLUDE_xTaskDelayUntil                  1
#define INCLUDE_vTaskDelay                       1
#define INCLUDE_vTaskSuspend                     1
#define INCLUDE_uxTaskGetStackHighWaterMark      1

/* Cortex-M4: 4 priority bits */
#define configPRIO_BITS                          4
#define configKERNEL_INTERRUPT_PRIORITY          (15u << (8 - configPRIO_BITS))
#define configMAX_SYSCALL_INTERRUPT_PRIORITY     (5u << (8 - configPRIO_BITS))

extern volatile unsigned int g_rtos_assert;
#define configASSERT(x) do { if (!(x)) { g_rtos_assert = __LINE__; for (;;) {} } } while (0)

/* port handlers on the vector names of startup.c */
#define vPortSVCHandler     SVC_Handler
#define xPortPendSVHandler  PendSV_Handler
#define xPortSysTickHandler SysTick_Handler
