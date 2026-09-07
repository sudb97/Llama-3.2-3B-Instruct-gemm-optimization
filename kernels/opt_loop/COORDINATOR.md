# Coordinator tick (Cursor parent agent)

This is what **this** chat does on every `/loop` wake. Scripts only parse numbers.
**You** (the model) vet the ncu report and decide. Then you launch **parallel
Cursor Task agents**.

`CONTAINER` is required: `local` if already inside the profiler env, or a
docker id/name from the host. Load `container.env` if unset. **Do not abort
if the GPU is not L4** — record the GPU name and proceed.

## Tick prompt (literal)

```
GEMV opt tick: follow kernels/opt_loop/COORDINATOR.md. Use CONTAINER from container.env. Vet the latest ncu yourself (do not rubber-stamp parse_ncu.py). Dispatch parallel Task agents for open levers. Stop at ≥25% vs TRT on both shapes or write FINAL_REPORT.md.
```

## Every tick

1. If `loop_state.json` is `stopped`, do nothing and do not re-arm.
2. Read `VERDICT.md`, `EXPERIMENT_LOG.md`, `baselines.json`, newest
   `kernels/opt_loop/runs/*/summary.json` if any, and the newest `*_ncu.csv`.
3. If `CONTAINER` resolves to a running docker container, run
   `CONTAINER=<id> TAG=<tick> ./kernels/opt_loop/measure.sh` and use that
   capture. If docker/container is missing, use the last ncu (currently
   `kernels/gemv_{up,down}_proj_ncu.csv`) and say so in the verdict.
4. **Vet the ncu in prose.** Write/overwrite `kernels/opt_loop/VERDICT.md`:
   - GPU / container / sm (from the capture, not assumed L4)
   - Bound? (DRAM % vs compute / FFMA)
   - Amplification vs 50.33 MB
   - Latency vs TRT 223.9 / 223.6 µs and vs GEMV 215.6 / 208.9 µs
   - Which lever can still move time (DRAM / amp / FP8)
   - What you are **not** doing (occupancy-only, warm bench GB/s)
5. **Dispatch parallel Task agents** in one turn (three `Task` calls):
   - DRAM → worktree `/workspace/opt-worktrees/dram`, prompt file
     `kernels/opt_loop/prompts/agent_dram.md` plus this tick’s `VERDICT.md`
   - AMP → `/workspace/opt-worktrees/amp` + `agent_amp.md`
   - FP8 → `/workspace/opt-worktrees/fp8` + `agent_fp8.md`
   Skip a lever if `VERDICT.md` says it is closed.
   Each subagent: implement **one** kernel change. ncu only via
   `measure.sh --container <id>` (serialized by flock). Return a short report.
6. When they finish, merge the story into `EXPERIMENT_LOG.md`. Run
   `measure.sh --container <id>` on each candidate. Keep the winner.
7. **Stop** if both shapes ≥ 25% vs TRT → `FINAL_REPORT.md` and
   `./kernels/opt_loop/stop.sh`.
8. **Stop** if 3 ticks in a row move min-gain < 0.5 pp after FP8 was tried.

## Launch / track / stop

See `COMMANDS.md`. User start:

```
CONTAINER=<id> Follow kernels/opt_loop/TICK.md
```
