# GEMV opt loop — start / track / stop

`CONTAINER` is the docker **id or name** of a running container that already
has the profiler environment (`nvcc`, `ncu`, workspace mount). Any GPU is
allowed. There is no L4 abort.

Replace `<id>` with `docker ps` output (name or 12-char id).

## 1. Start

```bash
cd /workspace/Llama-3.2-3B-Instruct-gemm-optimization

# bind the already-running container (writes container.env + loop_state.json)
CONTAINER=<id> ./kernels/opt_loop/start.sh
# same:
# ./kernels/opt_loop/start.sh <id>
```

Then in this Cursor chat:

```
CONTAINER=<id> Follow kernels/opt_loop/TICK.md
```

That runs one coordinator tick immediately (vet ncu → `VERDICT.md` → parallel
Task agents) and arms `/loop` until stop.

Measure a candidate yourself:

```bash
CONTAINER=<id> TAG=iter01_dram ./kernels/opt_loop/measure.sh
# or
./kernels/opt_loop/measure.sh --container <id>
```

## 2. Track

```bash
./kernels/opt_loop/status.sh
```

Also:

```bash
cat kernels/opt_loop/loop_state.json
cat kernels/opt_loop/VERDICT.md
tail -n 30 kernels/opt_loop/EXPERIMENT_LOG.md
ls -lt kernels/opt_loop/runs/*/summary.json | head
```

In Cursor: `Track the GEMV opt loop` (parent runs `status.sh` and reads the verdict).

## 3. Stop

```bash
./kernels/opt_loop/stop.sh
```

In Cursor: `Stop the GEMV opt loop`

That kills `loop.pid` if a ticker is armed, sets `loop_state.json` to
`stopped`, and the coordinator must not re-arm or dispatch.
