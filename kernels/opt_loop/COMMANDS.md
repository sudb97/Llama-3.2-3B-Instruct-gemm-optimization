# GEMV opt loop — start / track / stop

`CONTAINER` is either `local` (already inside the profiler env) or a docker
**id or name** on the host. Any GPU is allowed. There is no L4 abort.

## 1. Start

From **inside** the profiler container (no docker CLI — this is the usual case):

```bash
cd /workspace/Llama-3.2-3B-Instruct-gemm-optimization
CONTAINER=local ./kernels/opt_loop/start.sh
# same: CONTAINER=$(hostname) ./kernels/opt_loop/start.sh
```

From the **host** (docker exec into a running container):

```bash
CONTAINER=<id> ./kernels/opt_loop/start.sh
```

Then in this Cursor chat:

```
CONTAINER=local Follow kernels/opt_loop/TICK.md
```

That runs one coordinator tick immediately (vet ncu → `VERDICT.md` → parallel
Task agents) and arms `/loop` until stop.

Measure a candidate yourself:

```bash
CONTAINER=local TAG=iter01_dram ./kernels/opt_loop/measure.sh
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
