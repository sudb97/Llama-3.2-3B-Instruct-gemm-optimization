---
name: l4-gemv-opt-loop
description: Coordinate the Llama-3.2-3B GEMV optimization loop. Parent agent vets ncu reports, then launches parallel Cursor Task agents for DRAM, amplification, and FP8. Use when the user says start the opt loop, TICK.md, /loop GEMV, CONTAINER, remaining headroom, or FINAL_REPORT.
---

# GEMV opt loop (parent)

Source of truth: [`kernels/opt_loop/COORDINATOR.md`](../../../kernels/opt_loop/COORDINATOR.md).
Commands: [`kernels/opt_loop/COMMANDS.md`](../../../kernels/opt_loop/COMMANDS.md).

`CONTAINER` (docker id or name) is required. Load `container.env` if unset.
**Do not abort if the GPU is not L4.**

On start or `/loop` wake:

1. If `loop_state.json` is `stopped`, halt.
2. Vet ncu yourself. Write `kernels/opt_loop/VERDICT.md` (include GPU).
3. In **one** turn, launch parallel `Task` subagents (dram / amp / fp8) per the verdict. Skip closed levers.
4. Do not treat `parse_ncu.py` as the decision. Occupancy-only is forbidden.
5. Stop at ≥25% vs TRT on both shapes (`FINAL_REPORT.md`) and run `stop.sh`.

Start: `CONTAINER=<id> Follow kernels/opt_loop/TICK.md`  
Track: `./kernels/opt_loop/status.sh`  
Stop: `./kernels/opt_loop/stop.sh` or `Stop the GEMV opt loop`
