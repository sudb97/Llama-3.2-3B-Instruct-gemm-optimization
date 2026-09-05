You are a **workflow smoke** agent for one T4 dry-run tick.

Do **not** edit `kernels/gemv_fp16.cu`, `kernels/gemv_fp8.cu`, or any worktree.
Do **not** run `measure.sh` or ncu.

Read `kernels/opt_loop/DRY_RUN.md` and `kernels/opt_loop/runs/dry_run_t4/summary.json`.

Write **only** `kernels/opt_loop/runs/dry_run_t4/AGENT_REPORT.md` with:

- that you read the parent dry-run verdict
- that dummy `hit_25pct_vs_trt` is rejected
- that official numbers stay the L4 CSVs
- no kernel change

Then stop.
