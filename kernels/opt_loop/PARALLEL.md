# Parallel agents

Three **independent** kernel tracks. They must not share a working tree.

| Agent | Branch / worktree | Owns | Start prompt |
|---|---|---|---|
| DRAM | `opt/dram` → `/workspace/opt-worktrees/dram` | DRAM % → 100% | `prompts/agent_dram.md` |
| AMP | `opt/amp` → `/workspace/opt-worktrees/amp` | amplification → 1.00× | `prompts/agent_amp.md` |
| FP8 | `opt/fp8` → `/workspace/opt-worktrees/fp8` | FP8 weights | `prompts/agent_fp8.md` |

`measure.sh` flocks `kernels/opt_loop/.ncu.lock` in the **main** repo so only
one ncu runs at a time. Pass the same `CONTAINER` to every measure.

## Initiate in this Cursor chat (preferred)

```
CONTAINER=<id> Follow kernels/opt_loop/TICK.md
```

The parent agent vets ncu, then launches three **Task** subagents.

## Initiate (shell)

```bash
cd /workspace/Llama-3.2-3B-Instruct-gemm-optimization
CONTAINER=<id> ./kernels/opt_loop/start.sh
bash kernels/opt_loop/start_parallel.sh
```

Then start **three** Cursor agents, each rooted at one worktree.

## Track / stop

```bash
./kernels/opt_loop/status.sh
./kernels/opt_loop/stop.sh
```
