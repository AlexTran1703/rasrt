/* STM32F411CE: 512 KiB flash, 128 KiB SRAM. Results block pinned at the start of SRAM. */
ENTRY(Reset_Handler)

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) *(.glue_7) *(.glue_7t) KEEP(*(.init)) KEEP(*(.fini)) } > FLASH
  .rodata : { . = ALIGN(4); *(.rodata*) . = ALIGN(4); } > FLASH
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(.init_array*)) PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(.fini_array*)) PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH

  /* host-visible results, not initialised by startup */
  .results (NOLOAD) : { . = ALIGN(4); KEEP(*(.results)) . = ALIGN(4); } > RAM

  _sidata = LOADADDR(.data);
  .data : { . = ALIGN(4); _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > RAM AT > FLASH
  .bss : { . = ALIGN(4); _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; } > RAM
  end = _ebss;
  _end = _ebss;
}
