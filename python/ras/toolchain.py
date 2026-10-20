"""STM32F411 toolchain: ST Edge AI code generation (cached), build, flash, read back.

    generate(onnx, name, gen_dir)    ST Edge AI C code + its planner outputs (activations, weights,
                                     MACs) and the exact network-context size (compiler, no board)
    build(out, main_c, nets)         link firmware; sections and per-function stack frames
    flash_read(elf, nbytes, wait_s)  program, run, read the .results block over SWD
Tool locations (environment overrides): STEDGEAI_ROOT (ST Edge AI 3.0), ARM_GCC_BIN (arm-none-eabi-* folder),
STM32_PROGRAMMER (STM32_Programmer_CLI executable).
"""

import hashlib
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import FIRMWARE

_ST = Path(os.environ.get("STEDGEAI_ROOT", r"F:\ST\STEdgeAI\3.0"))
STEDGEAI = _ST / "Utilities" / "windows" / "stedgeai.exe"
AI_ROOT = _ST / "Middlewares" / "ST" / "AI"
_CUBE = Path(r"F:\ST\STM32CubeIDE_1.18.1\STM32CubeIDE\plugins")
GCC_BIN = Path(os.environ.get("ARM_GCC_BIN", _CUBE / (
    "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.win32_1.0.0.202411081344") / "tools" / "bin"))
PROG = Path(os.environ.get("STM32_PROGRAMMER", _CUBE / (
    "com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.200.202503041107") / "tools" / "bin"
    / "STM32_Programmer_CLI.exe"))
FW_SRC = FIRMWARE / "src"
CFLAGS = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-O2", "-ffunction-sections",
          "-fdata-sections", "-std=gnu11", "-Wall", "-Wno-unused-function", "-Wno-unused-variable"]
RESULTS_MAGIC = 0xEC650002


def run(cmd, cwd=None, timeout=600):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p.stdout


def c_array(name, arr, ctype):
    flat = np.asarray(arr).ravel()
    body = ",".join(str(int(v)) if ctype != "float" else f"{float(v):.9e}f" for v in flat)
    return f"static const {ctype} {name}[{max(len(flat), 1)}] = {{{body or 0}}};"


def fnv1a(z):
    h = 2166136261
    for b in np.asarray(z).astype(np.int8).tobytes():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def generate(onnx_path, name, gen_dir):
    """ST Edge AI generate (inputs not allocated: the firmware owns its buffers). Cached on the
    ONNX content. Returns the planner's activation bytes, weights, MACs and the context size."""
    gen_dir = Path(gen_dir)
    meta = gen_dir / f"{name}_meta.json"
    sha = hashlib.sha1(Path(onnx_path).read_bytes()).hexdigest()
    if meta.exists() and json.loads(meta.read_text()).get("sha") == sha:
        return json.loads(meta.read_text())
    gen_dir.mkdir(parents=True, exist_ok=True)
    txt = run([STEDGEAI, "generate", "-m", onnx_path, "--target", "stm32f4", "--name", name,
               "-w", gen_dir / "ws", "-o", gen_dir, "--input-data-type", "int8", "--output-data-type", "int8",
               "--inputs-ch-position", "chlast", "--outputs-ch-position", "chlast", "--no-inputs-allocation"])
    info = {"sha": sha, "name": name,
            "activations": int(re.search(r"activations \(rw\)\s*:\s*([\d,]+) B", txt)[1].replace(",", "")),
            "weights": int(re.search(r"weights \(ro\)\s*:\s*([\d,]+) B", txt)[1].replace(",", "")),
            "macc": int(re.search(r"macc\s*:\s*([\d,]+)", txt)[1].replace(",", ""))}
    info["ctx"] = _ctx_size(gen_dir, name)
    meta.write_text(json.dumps(info))
    return info


def _ctx_size(gen_dir, name):
    """sizeof the network context, read from a compiled object (compiler output, no board)."""
    src = gen_dir / f"_ctx_{name}.c"
    src.write_text(f'#include "{name}.h"\nunsigned char ctx_probe[STAI_{name.upper()}_CONTEXT_SIZE];\n')
    obj = gen_dir / f"_ctx_{name}.o"
    run([GCC_BIN / "arm-none-eabi-gcc.exe", *CFLAGS, "-fno-common", "-I", gen_dir, "-I", AI_ROOT / "Inc", "-c", src,
         "-o", obj])
    m = re.search(r"^[0-9a-fA-F]+ ([0-9a-fA-F]+) \w ctx_probe$", run([GCC_BIN / "arm-none-eabi-nm.exe", "-S", obj]),
                  re.M)
    return int(m[1], 16)


def nets_header(nets, pool_bytes):
    """nets.h: contexts, shared activation pool (pool_bytes = max planner activation size, checked
    at compile time), dispatch tables over `nets` ((name, gen_dir)); "tail" is not a segment."""
    names = [n for n, _ in nets]
    segs = [n for n in names if n != "tail"]
    h = ["#pragma once", '#include "stai.h"'] + [f'#include "{n}.h"' for n in names]
    h += [f"static uint8_t g_ctx_{n}[STAI_{n.upper()}_CONTEXT_SIZE] __attribute__((aligned(8)));" for n in names]
    h.append(f"#define POOL_BYTES {int(pool_bytes)}")
    h += [f"_Static_assert(STAI_{n.upper()}_ACTIVATIONS_SIZE_BYTES <= POOL_BYTES, \"pool\");" for n in names]
    h += ["typedef stai_return_code (*set_in_t)(stai_network *, const stai_ptr *, const stai_size);",
          "typedef stai_return_code (*run_t)(stai_network *, const stai_run_mode);",
          "typedef stai_return_code (*get_out_t)(stai_network *, stai_ptr *, stai_size *);"]
    tab = segs or ["tail"]
    h.append("static stai_network *const NET[] = {" + ",".join(f"(stai_network *)g_ctx_{n}" for n in tab) + "};")
    h.append("static const set_in_t NET_SET_INPUTS[] = {" + ",".join(f"stai_{n}_set_inputs" for n in tab) + "};")
    h.append("static const run_t NET_RUN[] = {" + ",".join(f"stai_{n}_run" for n in tab) + "};")
    h.append("static const get_out_t NET_GET_OUTPUTS[] = {" + ",".join(f"stai_{n}_get_outputs" for n in tab) + "};")
    if "tail" in names:
        h.append("#define TAIL_NET ((stai_network *)g_ctx_tail)")
    h += ["static int nets_init(uint8_t *pool)", "{", "    stai_ptr acts[1] = {pool};"]
    for n in names:
        h += [f"    if (stai_{n}_init((stai_network *)g_ctx_{n}) != STAI_SUCCESS) return 1;",
              f"    if (stai_{n}_set_activations((stai_network *)g_ctx_{n}, acts, 1) != STAI_SUCCESS) return 2;"]
    h += ["    return 0;", "}"]
    return "\n".join(h) + "\n"


def build(out, main_c, nets, extra_srcs=(), extra_inc=()):
    """Compile main_c + startup + the generated networks (+ extra sources, e.g. the FreeRTOS kernel).
    Returns elf, sections, stack frames."""
    out = Path(out)
    srcs = [FW_SRC / main_c, FW_SRC / "startup.c", *extra_srcs]
    inc = ["-I", out, "-I", FW_SRC, "-I", AI_ROOT / "Inc"]
    for d in extra_inc:
        inc += ["-I", d]
    for n, gdir in nets:
        srcs += [Path(gdir) / f"{n}.c", Path(gdir) / f"{n}_data.c"]
        inc += ["-I", gdir]
    elf = out / "fw.elf"
    lib = AI_ROOT / "Lib" / "GCC" / "ARMCortexM4" / "NetworkRuntime1100_CM4_GCC.a"
    run([GCC_BIN / "arm-none-eabi-gcc.exe", *CFLAGS, "-fstack-usage", *inc, *srcs, lib, "-T", FW_SRC / "f411.ld",
         "-Wl,--gc-sections", f"-Wl,-Map={out / 'fw.map'}", "--specs=nano.specs", "--specs=nosys.specs", "-lm",
         "-o", elf], cwd=out)
    sec = {m[1]: int(m[2]) for m in re.finditer(r"^(\.\S+)\s+(\d+)\s+\d+",
                                                run([GCC_BIN / "arm-none-eabi-size.exe", "-A", elf]), re.M)}
    frames = {}
    for su in out.glob("*.su"):
        for line in su.read_text().splitlines():
            m = re.match(r".*:\d+:\d+:(\w+)\s+(\d+)\s+\w+", line)
            if m:
                frames[m[1]] = int(m[2])
    return elf, sec, frames


def flash_read(elf, nbytes, wait_s, out):
    """Program + reset, wait, read the .results block (pinned at the start of SRAM)."""
    run([PROG, "-c", "port=SWD", "mode=UR", "-w", elf, "-v", "-rst"], timeout=180)
    time.sleep(wait_s)
    binf = Path(out) / "results.bin"
    raw = b""
    for _ in range(30):
        run([PROG, "-c", "port=SWD", "mode=HOTPLUG", "-u", "0x20000000", str(nbytes), binf], timeout=120)
        raw = binf.read_bytes()
        if struct.unpack_from("<I", raw, 0)[0] == RESULTS_MAGIC:
            return raw
        time.sleep(3)
    raise RuntimeError("no results block (firmware did not finish)")
