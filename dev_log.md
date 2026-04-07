# Dev Log

## Baseline
- Goal: Stop using the self-hosted ZeroTier bare-repo sync path and return this project to direct Git-based synchronization via GitHub, then sync the current local state to the Windows server immediately.
- Current scope: Local repository remote configuration, current untracked documentation file, remote server repository remote/branch state, and this task's memory records.
- Constraints: Maintain this log as the primary working memory; avoid storing credentials; do not use the self-built `zerotier-local` path for this sync; do not revert unrelated user changes; keep the server project history intact except for the minimum Git operations required to resync it.
- Protected areas: Remote experiment artifacts/data directories; unrelated code or docs not involved in the current sync; secret material.
- Risks / assumptions: GitHub is currently reachable from both local and server; the server's existing `develop` branch state can be aligned safely after preserving visibility into its status; the untracked architecture doc is intended to be included in the sync.

## Current Task History

### [Step 048] - 2026-04-05 11:05 CST - Prepare a clean WikiText-103 ablation trainer
- Request: Stop circling around the notebook path and write a training script that explicitly imports `ablation_models.py`, targeting WikiText-103 for the current experiments.
- Plan: Inspect the existing ablation script/model pair, then add a new dedicated training entrypoint with explicit CLI args, cleaner data handling, and SwanLab support instead of reusing the current hard-coded mixed-purpose script.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Refreshed working memory for the new code task after rereading the iterative logger instructions, current dev log, `scripts/train.py`, and `src/layer_depth_attention/ablation_models.py`.
- Rationale: The repo currently has multiple training stacks with different data paths and defaults; a dedicated WikiText-103 ablation trainer is easier to reason about and compare.
- Key details: `ablation_models.TinyDecoderLM` already has explicit GPT-style initialization, supports the relevant attention variants, and is a safer base than the notebook approximation path.
- Side effects: None yet; this is a task-baseline checkpoint.
- Verification: Read `scripts/train.py` and `src/layer_depth_attention/ablation_models.py`.
- Next step: Create the new training script and smoke-check it locally.

### [Step 049] - 2026-04-05 15:34 CST - Add dedicated WikiText-103 ablation training entrypoint
- Request: Provide a training script that directly uses `ablation_models.py` and currently targets the WikiText-103 dataset.
- Plan: Create a new CLI script instead of mutating the older mixed-purpose training file, keeping data loading, evaluation, and SwanLab handling explicit and configurable.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `scripts/train_wikitext103_ablation.py`, which imports `layer_depth_attention.ablation_models.TinyDecoderLM`, defaults to `wikitext-103-raw-v1`, supports `hf` and `local_text` data sources, exposes the major training/model knobs via argparse, logs to SwanLab, and writes per-method result JSON summaries.
- Rationale: The existing `scripts/train.py` mixes notebook-era defaults and multiple experimental assumptions; a dedicated entrypoint is clearer for controlled WikiText-103 ablations and easier to launch on local/remote machines.
- Key details: The new data pipeline uses whole-split text plus newline-to-EOS replacement, matching the safer full-text tokenization style instead of the older line-by-line token accumulation path in `scripts/train.py`. Validation cursor usage is configurable and defaults to `off` for reproducibility.
- Side effects: None on existing scripts; this is an additive path.
- Verification: `python -m py_compile scripts/train_wikitext103_ablation.py src/layer_depth_attention/ablation_models.py` passed.
- Next step: Share the new script path and an example launch command, then adjust defaults further only if the user wants a different dataset/method lineup.

### [Step 050] - 2026-04-05 15:41 CST - Rename default SwanLab project for the new ablation trainer
- Request: Change the SwanLab project name used by the new ablation training script.
- Plan: Only change the default `--log-project` value in the new WikiText-103 trainer, then re-run a syntax check.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Changed the default SwanLab project from `Layer-Depth-Attention-Ablation` to `Layer-Depth-Attention-WikiText103-Ablation`.
- Rationale: This keeps the new ablation runs separated from the older mixed experiment project and makes the dataset-specific purpose obvious in the dashboard.
- Key details: Only the default changed; passing `--log-project` still overrides it.
- Side effects: Future runs of this script without an explicit `--log-project` will land in the new project name.
- Verification: `python -m py_compile scripts/train_wikitext103_ablation.py` passed.
- Next step: Tell the user the new default project name and offer to change it again if they want a different naming convention.

### [Step 051] - 2026-04-05 15:45 CST - Sync and launch the new WikiText-103 ablation trainer on the Windows server
- Request: Push the new ablation trainer to the server and run it there.
- Plan: Upload the new Python entrypoint and a dedicated Windows launch script, then start the job from the server project directory and verify from the log that training actually begins.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_ablation_default.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Synced the new Python trainer and a dedicated `.bat` launcher to `D:\Projects\Layer-Depth-Attention\scripts\` on the Windows server, then started the launcher there.
- Rationale: Earlier PowerShell `Start-Process` attempts did not leave a stable detached run/log; the explicit batch launcher produced a visible new Python process and active log output.
- Key details:
  - Server launcher path: `D:\Projects\Layer-Depth-Attention\scripts\launch_wikitext103_ablation_default.bat`
  - Active log path: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_ablation.log`
  - Confirmed new server Python process started at `2026/4/5 15:44:34` with `pt-3.9`.
  - Current log already shows parsed config, new SwanLab project name, and `[data] tokenizing raw text splits...`.
- Side effects: Server now has a new long-running WikiText-103 ablation job in addition to older Python processes.
- Verification: `scp` succeeded for both files; remote `Test-Path` checks returned `True`; remote process list shows a new `python.exe`; remote log file now has non-zero size and contains the trainer config header plus the tokenization progress line.
- Next step: Monitor the first training metrics (`step=1`, then the first eval interval) from the server log.

### [Step 052] - 2026-04-05 16:00 CST - Start SwanLab before data loading for single-method runs
- Request: Fix the bug where SwanLab starts too late; the user wants the run to appear from the very beginning of the program so the full log is visible.
- Plan: For single-method runs, initialize the same SwanLab monitor in `main()` before `LMData` is constructed, log startup/data metadata at step 0, and reuse that monitor inside `run_experiment`.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added helper functions for experiment/config naming, changed `run_experiment()` to accept an optional prebuilt monitor, and made `main()` eagerly initialize SwanLab before tokenization when only one method is being run.
- Rationale: The previous script only initialized SwanLab after dataset preparation and model construction, so startup and tokenization logs were invisible in the run timeline.
- Key details: Multi-method runs still keep one run per method to avoid mixing separate experiments into one SwanLab record; the eager startup behavior applies when `len(args.methods) == 1`.
- Side effects: Single-method runs now log `program_started`, `num_gpus`, and token counts at step 0 in the same run that later receives training metrics.
- Verification: `python -m py_compile scripts/train_wikitext103_ablation.py` passed.
- Next step: Re-upload the script to the server and relaunch a single-method run to confirm SwanLab appears before data tokenization.

### [Step 053] - 2026-04-05 16:03 CST - Relaunch server experiment as single-method dualq-sublayer run
- Request: Push the early-SwanLab fix, then start a server experiment that only runs `shared_kv_depth_memory_dualq_sublayer`.
- Plan: Upload the updated trainer plus a dedicated single-method launch script, stop the previous generic WikiText-103 ablation run, and relaunch the new single-method job while watching for immediate SwanLab init output.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Synced the updated trainer and new `launch_wikitext103_dualq_sublayer.bat` to the server, stopped the old 15:44 WikiText-103 run, and relaunched a single-method `shared_kv_depth_memory_dualq_sublayer` experiment.
- Rationale: A single-method launch is required for the new early SwanLab initialization path, and it avoids mixing multiple methods into one run.
- Key details:
  - New active server process started at `2026/4/5 16:02:31`.
  - Server log file: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_dualq_sublayer.log`
  - SwanLab run URL: `https://swanlab.cn/@justbook/Layer-Depth-Attention-WikiText103-Ablation/runs/chli1fbnc2lrhkj7m3gwp`
  - Log already shows SwanLab cloud sync before dataset tokenization finishes.
- Side effects: The prior generic `wikitext103_ablation` run was terminated and replaced by the dedicated single-method run.
- Verification: Remote process list shows new `python.exe`; log file is non-empty and contains the new single-method config plus successful SwanLab initialization.
- Next step: Watch for token-count output and the first `step=1` metric from the new server log.

### [Step 054] - 2026-04-05 17:22 CST - Launch sequential baseline then attn_residual comparison on the server
- Request: Run `baseline` and `attn_residual` in the same batch, with `baseline` first.
- Plan: Add a dedicated Windows launcher that passes `--methods baseline attn_residual`, sync it to the server, and start the job while confirming from the log that the methods list is in the requested order.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_baseline_attnresidual.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added and synced `launch_wikitext103_baseline_attnresidual.bat`, then started it on the server. The new run uses the shared `train_wikitext103_ablation.py` entrypoint and logs to `wikitext103_baseline_attnresidual.log`.
- Rationale: The user wants a fair sequential comparison under one clean training stack, with `baseline` finishing before `attn_residual` begins.
- Key details:
  - New active server process started at `2026/4/5 17:22:14`.
  - Log file: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_baseline_attnresidual.log`
  - The log already shows `methods = ["baseline", "attn_residual"]` in that order.
- Side effects: This run uses the multi-method path, so SwanLab early init is not used; each method will still create its own run when its training starts.
- Verification: Remote process list showed a new `python.exe`; remote log file was created and contains the expected ordered methods list.
- Next step: Monitor the log until it emits `running baseline`, then capture the first baseline metrics before `attn_residual` starts.

### [Step 055] - 2026-04-05 22:16 CST - Launch 16-layer 80k-step four-method server batch
- Request: Run four methods on the server in this exact order: `shared_kv_depth_memory_dualq_sublayer`, `baseline`, `attn_residual_2d`, `attn_residual`, with `num_layers=16` and `steps=80000`.
- Plan: Create a dedicated Windows launcher encoding the requested order and hyperparameters, sync it to the server, stop current Python training jobs, and relaunch while checking the log for the exact method order and new step budget.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_four_methods_16l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added and synced `launch_wikitext103_four_methods_16l_80000.bat`, terminated the current server Python jobs, and started the new batch run. The active server log is now `D:\Projects\Layer-Depth-Attention\logs\wikitext103_four_methods_16l_80000.log`.
- Rationale: The user wants a longer, deeper, fixed-order four-method comparison under the same WikiText-103 ablation trainer.
- Key details:
  - New active server process started at `2026/4/5 22:15:49`.
  - Log confirms:
    - `num_layers = 16`
    - `steps = 80000`
    - ordered methods: `shared_kv_depth_memory_dualq_sublayer`, `baseline`, `attn_residual_2d`, `attn_residual`
  - Only one new `pt-3.9` Python process remains active after relaunch.
- Side effects: Earlier batch runs were stopped to free resources for this long run.
- Verification: Remote process list showed a single new `python.exe`; remote log file exists and contains the expected method order and training configuration.
- Next step: Monitor `wikitext103_four_methods_16l_80000.log` for token counts and the first metrics from `shared_kv_depth_memory_dualq_sublayer`.

### [Step 001] - 2026-04-02 02:25 CST - Reassess sync topology
- Request: Stop using the self-built repository sync path and synchronize through Git immediately.
- Plan: Inspect local/worktree Git remotes, inspect the server repository remotes and branch status, then switch back to GitHub-based sync and perform a live sync.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention-wt/dev_log.md`
- Modification: Created the project-root dev log and recorded the current task baseline after reviewing the iterative logger requirements and the prior memory summaries/events.
- Rationale: The root `dev_log.md` was missing even though this project requires it for multi-step coding and remote-work tasks.
- Key details: Local worktree currently points `origin` at the local bare repo container; server repo still has both `origin` (GitHub) and `zerotier-local` remotes configured.
- Side effects: Future task memory for this sync now has a stable root entry.
- Verification: Reviewed `/Users/a/.codex/skills/iterative-dev-logger/SKILL.md`, `memory/summaries.md`, and the latest `memory/events.jsonl` entries.
- Next step: Update memory/events, reconfigure local `origin` back to GitHub, then commit and sync the server.

### [Step 002] - 2026-04-02 02:34 CST - Restore GitHub as the active sync path
- Request: Complete the switch away from the self-built sync path and sync immediately.
- Plan: Repoint local `origin` to GitHub, preserve the old remote branch tips with backup branches, push the local `main`, then back up the server workspace and realign it to `origin/main`.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention-wt/dev_log.md`, `/Users/a/Projects/Layer-Depth-Attention-wt/memory/events.jsonl`, `/Users/a/Projects/Layer-Depth-Attention-wt/memory/summaries.md`, `/Users/a/Projects/Layer-Depth-Attention-wt/docs/dual_axis_full_architecture.md`
- Modification: Added the missing root `dev_log.md` and the untracked `docs/dual_axis_full_architecture.md`, committed them, pushed local `main` to GitHub with `--force-with-lease` after creating remote backup branches, then stashed the server's dirty worktree, switched the server to `main`, and removed the `zerotier-local` remote.
- Rationale: The server can now use ordinary `git fetch/pull` against GitHub again, without depending on the local ZeroTier bare mirror or daemon.
- Key details: Remote backups created were `backup/main-pre-resync-20260402` and `backup/develop-pre-resync-20260402`; the server also has branch `backup/server-pre-github-resync-20260402` plus stash entry `pre-github-resync-20260402`.
- Side effects: GitHub `main` was force-updated from `8cc6719` to `fa16438`; the server now tracks `origin/main` cleanly.
- Verification: Local `git push --force-with-lease origin main:main` succeeded; server `git checkout -B main origin/main` succeeded; server `git remote -v` now lists only GitHub `origin`; server `git status --short --branch` is clean.
- Next step: Push this final memory update if the repo should remain fully self-documented after the sync task.

### [Step 003] - 2026-04-02 17:35 CST - Audit Dual-Axis Full blueprint against current code
- Request: Verify whether `docs/dual_axis_full_architecture.md` is strictly identical to the current `dual_axis_full` training implementation, then assess the document's listed issues/suggestions and surface any additional problems.
- Plan: Locate the real `dual_axis_full` training entrypoints, inspect `train_wikitext_lm.py` and `src/layer_depth_attention/model.py`, and compare the blueprint line by line against the live code path instead of against older experiment branches.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the comparison baseline after reading the blueprint, the model implementation, and the active launch scripts.
- Rationale: The repository contains many historical attention variants; without isolating the actual `dual_axis_full` code path, any architecture review would mix incompatible implementations.
- Key details: Confirmed that `scripts/train_wikitext_lm.py --attention-type dual_axis_full` is the active entrypoint, and that `TinyDecoderLM(attention_type='dual_axis_full')` routes to `DualAxisMemoryAttention` plus `_attn_res_dual_axis_mix`.
- Side effects: None yet; this is an analysis-only checkpoint.
- Verification: Read `docs/dual_axis_full_architecture.md`, `src/layer_depth_attention/model.py`, `scripts/train_wikitext_lm.py`, `scripts/launch_dual_axis_full_true_bs8_s30000.bat`, and `scripts/launch_wt2a_dual_axis_full_bs16_s2000.bat`.
- Next step: Emit the strict consistency verdict, document concrete mismatches, and recommend the highest-value architecture/training fixes.

### [Step 004] - 2026-04-02 17:50 CST - Add depth-score scaling and richer dual-axis-full logging
- Request: First fix only the missing `_attn_res_mix()` scale factor, then rerun a `dual_axis_full` control. The user additionally asked to record elapsed time at each evaluation point, increase evaluation sample size, and use `20000` steps for this run.
- Plan: Keep the architectural change minimal by only adding `1/sqrt(D)` in `_attn_res_mix()`, then update the language-model training script to record elapsed time per eval point and create a dedicated `20000`-step launch script with larger `eval_batches`.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext_lm.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_dual_axis_full_true_bs8_s20000_eval100.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `1 / sqrt(D)` scaling to `_attn_res_mix()` depth scores.
  - Added `elapsed_seconds` and `elapsed_minutes` to each eval record in `train_wikitext_lm.py`, and printed elapsed minutes in the console log.
  - Added a dedicated launch script for `dual_axis_full_true_bs8_s20000_eval100`.
- Rationale: The scaling bug is the highest-priority algorithmic issue currently identified; richer timing and larger eval slices reduce interpretability problems and variance without changing the model definition.
- Key details: The new run script keeps the previous `dual_axis_full` architecture and optimization hyperparameters, but changes `steps=20000` and `eval_batches=100`.
- Side effects: Existing older launch scripts remain unchanged; this avoids rewriting prior experiment baselines.
- Verification: `python -m py_compile src/layer_depth_attention/model.py scripts/train_wikitext_lm.py` passed; local `TinyDecoderLM(attention_type='dual_axis_full')` forward/backward smoke test passed with output shape `(2, 32, 128)`.
- Next step: Sync the code to the server and start the new `dual_axis_full_true_bs8_s20000_eval100` run.

### [Step 005] - 2026-04-02 20:43 CST - Revert dual-axis-full eval batch count for fair comparison
- Request: Stop the just-started run and revert the evaluation batch count from `100` back to `20` so the new `dual_axis_full` result remains directly comparable to earlier runs.
- Plan: Kill the active server-side Python job, change only the dedicated `20000`-step launch script to use `eval_batches=20`, and keep all model/optimization settings unchanged.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_dual_axis_full_true_bs8_s20000_eval20.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Stopped the running `dual_axis_full_true_bs8_s20000_eval100` process, changed the script to `eval_batches=20`, and renamed the script/output/run label from `eval100` to `eval20` to avoid future confusion.
- Rationale: The higher evaluation sample size reduces variance but materially changes runtime cost and monitoring cadence, making this run less comparable to the earlier `dual_axis_full` controls.
- Key details: The script still uses the same `20000` steps, `bs=8`, `grad_accum_steps=2`, `seq_len=512`, `d_model=384`, and the already-fixed `_attn_res_mix` scaling.
- Side effects: The aborted `eval100` run should be ignored; its partial monitoring record is not part of the fair-comparison set.
- Verification: Server `python.exe` process was terminated successfully via `taskkill /F /IM python.exe`.
- Next step: Push the `eval20` script update, resync the server, and restart the run with SwanLab.

### [Step 006] - 2026-04-02 21:05 CST - Consolidate confirmed dual-axis-full issues into the design report
- Request: Put all previously identified `dual_axis_full` issues, mismatches, and optimization suggestions into the model design report Markdown.
- Plan: Append a dedicated issue/optimization section to `docs/dual_axis_full_architecture.md`, focusing only on already-confirmed findings from the live code path rather than speculative ideas.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/docs/dual_axis_full_architecture.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added a new section summarizing the confirmed code/doc mismatches, the `_attn_res_mix` scaling issue, the `DualAxisMemoryAttention` score/value path issue, the always-available `embedding` candidate issue, and the recommended repair order.
- Rationale: The architecture discussion had become fragmented across chat turns; the design report now serves as the single source of truth for the current known problems and next-step fixes.
- Key details: The added section explicitly downgrades the document from “strict 1:1 final spec” to “design blueprint + implementation gap report + repair roadmap.”
- Side effects: The document is now more accurate, but it also means older claims of perfect 1:1 correspondence should no longer be cited.
- Verification: Re-read the target sections of `docs/dual_axis_full_architecture.md` after patching.
- Next step: If needed, continue by implementing the next priority fix: split `DualAxisMemoryAttention` score/value paths.

### [Step 007] - 2026-04-02 21:11 CST - Record the unresolved depth-identity concern
- Request: Add another possible issue to the Dual-Axis Full report: attention may not be able to tell whether a historical token comes from a shallow or deep layer, though this may or may not matter in practice.
- Plan: Record it as a hypothesis/ablation target rather than as a confirmed bug.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/docs/dual_axis_full_architecture.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added a new issue item describing the lack of explicit depth identity markers in the stacked history, along with both possible interpretations and suggested ablation directions.
- Rationale: This concern is plausible and important enough to preserve, but the current evidence does not justify labeling it as an already-confirmed implementation defect.
- Key details: The document now distinguishes between confirmed problems and “potential structure risks worth testing.”
- Side effects: None on code or running experiments.
- Verification: Re-read the new “问题 D” subsection in `docs/dual_axis_full_architecture.md`.
- Next step: Keep this as a future ablation candidate after the currently higher-priority score/value split issue.

### [Step 008] - 2026-04-02 21:18 CST - Split DualAxisMemoryAttention score/value paths
- Request: Fix `DualAxisMemoryAttention` so that normalized history is used only for memory score computation, while the memory value aggregation uses the original historical states.
- Plan: Keep the module shape and public interface unchanged; only split the internal history tensor into a normalized score path and a raw value path.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Reworked `DualAxisMemoryAttention.forward()` to build `raw_memory_bank` and `normed_memory_bank` separately. `memory_scores` now uses `normed_memory_bank`, while `memory_context` aggregates over `raw_memory_bank`.
- Rationale: This aligns the implementation with the intended “score on normalized history, read out original history” design and better matches ordinary attention semantics.
- Key details: The change is local to the memory branch; token-side `q_row/k/v`, causal masking, and output projection are unchanged.
- Side effects: Any currently running server-side `dual_axis_full` run still uses the older code until explicitly restarted.
- Verification: `python -m py_compile src/layer_depth_attention/model.py scripts/train_wikitext_lm.py` passed; local `TinyDecoderLM(attention_type='dual_axis_full')` forward/backward smoke test passed with output shape `(2, 32, 128)`.
- Next step: If needed, sync this patch to the server and restart the active `dual_axis_full` training job on top of the new implementation.

### [Step 009] - 2026-04-02 21:10 CST - Add startup metadata banners to experiment logs
- Request: From the next experiment onward, print an explicit marker at the start of each log showing the run time, version, and what changed.
- Plan: Add a lightweight startup header to every experiment entry script, with an optional `--run-note` field for manually describing what changed in the current run.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext_lm.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_assoc_recall.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_cifar100_vit.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added a `[run-meta]` JSON line at startup containing local timestamp, script name, git short SHA, `attention_type`, and optional `run_note`; exposed `--run-note` in all three scripts.
- Rationale: This makes experiment logs self-identifying and removes ambiguity about which code version and which local change set produced a run.
- Key details: The metadata is printed before `model_params`, so it appears at the top of the console log; `git_rev` is resolved from the local repo and falls back to `unknown` if Git is unavailable.
- Side effects: Existing launch scripts continue to work unchanged because `--run-note` is optional. Already-running experiments will not retroactively gain this header.
- Verification: `python -m py_compile` passed for all three scripts; a 1-step local `train_assoc_recall.py` smoke run printed the expected `[run-meta]` JSON header.
- Next step: If a future run needs an explicit human-readable change summary, pass it through `--run-note "..."` in the launch command.

### [Step 010] - 2026-04-03 00:12 CST - Record the repaired dual-axis-full 20000-step result
- Request: Read back the result of the restarted `dual_axis_full_true_bs8_s20000_eval20` run after applying both the `_attn_res_mix` scaling fix and the `DualAxisMemoryAttention` score/value split.
- Plan: Read the final JSON artifact, confirm the training process has exited, and summarize both the best checkpoint and the final trajectory.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Logged the completed long-run result from `artifacts/wikitext2_dual_axis_full_true_bs8_s20000_eval20.json`.
- Rationale: This run is the first `dual_axis_full` result that includes both high-priority architecture fixes while still preserving the old `eval_batches=20` comparison protocol.
- Key details:
  - Final step: `20000`
  - Final train loss: `2.7055`
  - Final val loss / ppl: `2.9979 / 20.04`
  - Best val step: `18000`
  - Best val loss / ppl: `2.9780 / 19.65`
  - Best test loss / ppl: `3.1609 / 23.59`
  - Elapsed time at step `20000`: `239.71` minutes
- Side effects: The corresponding SwanLab run `dual_axis_full_true_bs8_s20000_eval20` is complete; no active server python process remains.
- Verification: Read the full remote artifact JSON and confirmed no `python` training process remains on the server.
- Next step: Compare this repaired `dual_axis_full` result against the prior `dual_axis_full` baseline and the other long-run controls.

### [Step 011] - 2026-04-03 00:32 CST - Re-audit dual-axis-full architecture issues
- Request: Review the model architecture design document and summarize what problems still remain in the current model.
- Plan: Re-read the current `dual_axis_full` design report, separate already-fixed issues from still-open issues and hypotheses, and produce a prioritized architecture review for the user.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the current architecture-review pass and its scope.
- Rationale: The design report now mixes blueprint content, repaired items, and speculative risks; the user needs a clean “what is still wrong now” summary.
- Key details:
  - Confirmed the document still overstates itself as a “1:1 完美复刻级设计” while section 5 explicitly says code and doc are not fully aligned.
  - Confirmed the previously high-priority `_attn_res_mix` scaling issue and `DualAxisMemoryAttention` score/value split issue have already been repaired in code.
  - Remaining open items are mainly: persistent `x_0` in residual candidates, lack of explicit depth identity, ineffective residual flags for `dual_axis_full`, and unresolved row/column query/projection design choices.
- Side effects: None on code; this is an analysis checkpoint.
- Verification: Re-read `docs/dual_axis_full_architecture.md` and inspected the relevant `dual_axis_full` locations in `src/layer_depth_attention/model.py`.
- Next step: Report the remaining issues to the user with a clear priority split: confirmed open problems vs hypotheses worth ablation.

### [Step 012] - 2026-04-03 00:49 CST - Add no-final-mix dual-axis-full ablation
- Request: Remove the final `_attn_res_dual_axis_mix` output stage and rerun the same `dual_axis_full` experiment to test whether the final global remix is actually necessary.
- Plan: Preserve the existing `dual_axis_full` path for comparison, introduce a new `dual_axis_full_no_final_mix` attention type that reuses the repaired body but outputs directly from the last nonlinear history state, then add a dedicated launch script with the same `20000 step / eval20` protocol.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_dual_axis_full_no_final_mix_true_bs8_s20000_eval20.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `dual_axis_full_no_final_mix` to `TinyDecoderLM`.
  - Reused the repaired dual-axis attention body, but skipped the final global remix and fed `history[-1]` directly into `final_norm -> lm_head`.
  - Added a separate launch script and run note so the ablation result will not overwrite the repaired `dual_axis_full` baseline.
- Rationale: This isolates the effect of the final output-side remix without invalidating the existing repaired `dual_axis_full` result.
- Key details: The new variant keeps the repaired `_attn_res_mix` scaling and the split `DualAxisMemoryAttention` score/value paths; only the final `out_final = _attn_res_dual_axis_mix(...)` stage is removed.
- Side effects: Old `dual_axis_full` checkpoints and scripts remain valid and directly comparable.
- Verification: `python -m py_compile src/layer_depth_attention/model.py scripts/train_wikitext_lm.py` passed; local `TinyDecoderLM(attention_type='dual_axis_full_no_final_mix')` forward/backward smoke test passed with output shape `(2, 32, 128)`.
- Next step: Commit and push the new ablation, sync the server, and start the `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` run.

### [Step 013] - 2026-04-03 00:58 CST - Fix no-final-mix CLI registration
- Request: Start the new `dual_axis_full_no_final_mix` experiment on the server.
- Plan: Repair any launch-path issues discovered during the first remote start attempt, then relaunch without changing the model definition.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext_lm.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `dual_axis_full_no_final_mix` to the `--attention-type` argparse choices in `train_wikitext_lm.py`.
- Rationale: The first server launch attempt failed immediately because the new attention type existed in the model code but was still rejected by the training script's CLI whitelist.
- Key details: The failure mode was clean: argparse reported `invalid choice: 'dual_axis_full_no_final_mix'`; no training actually started.
- Side effects: None on old experiments; this only repairs the training entrypoint for the new ablation.
- Verification: Remote launch stderr showed the missing CLI choice; local source now includes the new option in the parser choices list.
- Next step: Amend/push the fix, sync the server, and relaunch the `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` run.

### [Step 014] - 2026-04-03 01:05 CST - Make run metadata banner Windows-safe
- Request: Relaunch the `dual_axis_full_no_final_mix` long run after fixing the CLI registration issue.
- Plan: Repair any remaining launch-only blockers without changing the model, then restart the experiment.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext_lm.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_assoc_recall.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_cifar100_vit.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Changed all `[run-meta]` JSON prints from `ensure_ascii=False` to `ensure_ascii=True`.
- Rationale: The relaunch failed before training because Windows console output used `gbk`, and the JSON startup banner tried to print non-ASCII characters from `run_note`.
- Key details: This is a logging-only compatibility fix; it does not change any model or optimizer behavior.
- Side effects: Startup metadata now prints escaped Unicode instead of raw UTF-8/Chinese characters, which is safer for remote Windows sessions.
- Verification: Remote traceback clearly showed `UnicodeEncodeError` in `print_run_header()`; the three training scripts now all print ASCII-safe startup metadata.
- Next step: Amend/push this fix, resync the server, and relaunch the `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` run.

### [Step 015] - 2026-04-03 01:13 CST - Restore SwanLab logging for no-final-mix run
- Request: Stop the currently running `no_final_mix` experiment and restart it so the run appears in SwanLab.
- Plan: Identify why the current run says `init skipped: monitor disabled`, then fix only the launch/runtime environment and restart the same experiment without changing the model.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Diagnosed that the previous launch path was invoking the wrong Python (`D:\\Annaconda\\python.exe`) because `activate.bat` was being called from PowerShell and not affecting the subsequent interpreter resolution. Restarted the run using the explicit environment Python `D:\\Annaconda\\envs\\pt-3.9\\python.exe`.
- Rationale: SwanLab was installed and login-ready in `pt-3.9`, but not guaranteed in the default base interpreter, which caused the monitor to downgrade to disabled even though training itself started.
- Key details:
  - Verified `D:\\Annaconda\\envs\\pt-3.9\\python.exe` can import `swanlab` and `swanlab.login()` succeeds.
  - Stopped the old process `PID 38856`.
  - Relaunched `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` with the explicit env Python and a new `run_note`.
  - Confirmed new console output: `[swanlab] login succeeded` and `Syncing run dual_axis_full_no_final_mix_true_bs8_s20000_eval20 to the cloud`.
  - Current run URL: `https://swanlab.cn/@justbook/Layer-Depth-Attention/runs/0kc4g067qjebjqfmxioc2`
- Side effects: The current server-side run now has SwanLab tracking, but the launch was done through a direct command rather than the `.bat` script because the `.bat`/PowerShell path was the source of the environment mismatch.
- Verification: Observed `login_ok` from the env Python, then observed successful SwanLab initialization output and cloud sync URL from the restarted run.
- Next step: Monitor the first evaluation point (`step=1`/`step=400`) and keep this explicit-env launch pattern for future Windows SwanLab runs unless the batch activation path is simplified.

### [Step 016] - 2026-04-03 01:31 CST - Switch dual-axis pre-mix to joint row/depth softmax
- Request: Change every mixed row/column attention-matrix computation so that x-axis and y-axis scores are concatenated first and normalized together with one softmax, rather than separately normalized and then added.
- Plan: Audit all mixed-attention sites, confirm which ones already use global score concatenation, and patch only the remaining dual-axis pre-mix path that still did separate row/depth normalization.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Reworked `TinyDecoderLM._attn_res_dual_axis_mix()` so it now:
    1. computes row-axis causal scores,
    2. computes depth-axis history scores,
    3. concatenates the two score tensors,
    4. applies a single softmax over the joint candidate space,
    5. splits the unified weights back into row/depth parts for context aggregation.
  - Preserved raw depth values for the depth branch and causal masking for the row branch.
  - Fixed a follow-up regression by replacing the accidental call to a missing `_causal_mask()` helper with a local triangular mask inside `TinyDecoderLM`.
- Rationale: The user wants the dual-axis pre-mix to follow the same “global competition” principle already used in the main mixed attention layers, instead of artificially assigning separate normalized budgets to row and depth branches.
- Key details:
  - Audit result: the main attention layers (`DualAxisMemoryAttention`, `LayerDepth*`, FFN q-attention variants) already concatenate scores before softmax; only the residual-style dual-axis pre-mix path still used separate normalization.
  - The new implementation keeps the row branch multi-head and reshapes the depth query/history into head space so both branches can compete in a single score tensor.
- Side effects: This changes the semantics of both `attn_residuals_dual_axis` and `dual_axis_full*`, because they share `_attn_res_dual_axis_mix()`.
- Verification: `python -m py_compile src/layer_depth_attention/model.py` passed; local forward/backward smoke tests passed for `dual_axis_full` and `dual_axis_full_no_final_mix`, both with output shape `(2, 32, 128)`.
- Next step: If desired, commit this structural change separately and rerun the active dual-axis experiments under the new joint-softmax pre-mix definition.

### [Step 017] - 2026-04-03 01:56 CST - Review repaired dual-axis implementation for hidden regressions
- Request: Explain why newer, theoretically cleaner dual-axis variants can underperform older rougher versions, and check whether the implementation itself has concrete problems.
- Plan: Inspect the repaired `dual_axis_full` code paths for high-impact implementation mismatches that could distort routing or memory semantics independently of the high-level theory.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded two high-confidence implementation findings from the code review.
- Rationale: The user asked specifically whether the code implementation may be wrong, not just whether the architecture changed.
- Key details:
  - In the new joint-softmax `_attn_res_dual_axis_mix()`, the newest state appears twice in the unified candidate pool: once through the row branch over `current`, and again through the depth branch because `depth_values = [embedding] + history` still includes that same newest state. Under one shared softmax, this duplicates probability mass for the latest state.
  - `DualAxisMemoryAttention` still does not implement the documented `Out_memory = W_memory * H` style value projection. After the score/value split repair, it now reads raw `H` as memory values but only reshapes raw history into heads instead of learning a separate memory-value projection, so token-side `V` and memory-side `value` live in mismatched content spaces.
- Side effects: None on code; this is a review-only checkpoint.
- Verification: Re-read `src/layer_depth_attention/model.py` around `DualAxisMemoryAttention.forward()` and `_attn_res_dual_axis_mix()`.
- Next step: Report these findings to the user and recommend fixing the duplicate latest-state candidate and the missing learned memory-value projection before trusting further “fixed vs buggy” comparisons.

### [Step 018] - 2026-04-03 02:07 CST - Fix duplicate-candidate routing and add memory-value projection
- Request: Fix both newly identified implementation problems at once, then update the algorithm design document to match the repaired implementation.
- Plan: Patch `DualAxisMemoryAttention` to add a learned memory-value projection, patch `_attn_res_dual_axis_mix()` to remove the newest-state duplicate from the depth candidate pool, and rewrite the design doc formulas/status notes accordingly.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/docs/dual_axis_full_architecture.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `memory_v_proj` to `DualAxisMemoryAttention` and changed the memory branch to aggregate projected historical values rather than raw head-sliced history.
  - Changed `_attn_res_dual_axis_mix()` so the depth branch no longer includes the newest state already covered by the row branch; when there is no older history, the pre-mix now falls back to pure row attention.
  - Updated the design report so section 2 reflects the new joint-softmax pre-mix semantics, section 3 reflects the explicit memory-value projection, and section 5 reclassifies repaired vs still-open issues.
- Rationale: These two implementation gaps were strong candidates for why theoretically cleaner versions could underperform: one duplicated probability mass for the latest state, the other mixed token-side learned values with raw history-side values in mismatched content spaces.
- Key details:
  - `DualAxisMemoryAttention` parameter count increased because the memory path now has its own value projection.
  - The depth candidate pool in the dual-axis pre-mix is now `[x_0] + history[:-1]` when history exists, instead of `[x_0] + history`.
- Side effects: This changes the semantics of both `dual_axis_full` and `dual_axis_full_no_final_mix`; old results before this patch are no longer strictly comparable to runs after this patch without noting the implementation revision.
- Verification: `python -m py_compile src/layer_depth_attention/model.py` passed; local forward/backward smoke tests passed for `dual_axis_full` and `dual_axis_full_no_final_mix`, both with output shape `(2, 32, 128)`. Parameter count in the smoke test rose from `533504` to `566528`, confirming the new memory value projection is active.
- Next step: Commit these repairs, push to `develop`, and rerun the active `dual_axis_full_no_final_mix` experiment if the user wants fresh metrics under the repaired implementation.

### [Step 019] - 2026-04-03 02:16 CST - Switch row-branch values back to raw current states
- Request: For the dual-axis pre-mix row branch, stop using the normalized current state as the value path and instead use the original current state content.
- Plan: Keep `Q_row` and `K_row` on the stabilized normalized path, but make `V_row` come from the raw `x_current`, then update the design doc and rerun smoke tests.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/docs/dual_axis_full_architecture.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Changed `_residual_row_mix()` and `_attn_res_dual_axis_mix()` so row scores still use normalized keys, but row context aggregation now uses raw current-state values.
  - Updated section 2 of the design report to state `K_row = x_norm` and `V_row = x_current`.
- Rationale: This keeps the row-axis comparison path numerically stable without forcing the readout content itself through the same normalization; it matches the user’s intended “Q/K stabilized, V preserves original content” rule.
- Key details: This change only affects the row branch value path; the repaired memory-value projection and no-duplicate depth candidate fixes remain in place.
- Side effects: Both `dual_axis_full` and `dual_axis_full_no_final_mix` semantics changed again, so future experiments should be tagged against this revision.
- Verification: `python -m py_compile src/layer_depth_attention/model.py` passed; local forward/backward smoke tests passed for `dual_axis_full` and `dual_axis_full_no_final_mix`, both with output shape `(2, 32, 128)` and parameter count `566528`.
- Next step: Commit/push this final row-value-path adjustment and restart the active dual-axis experiment if the user wants metrics from this newest semantics.

### [Step 020] - 2026-04-02 23:40 CST - Review Attention Residuals reference paper against in-repo comparator
- Request: Inspect the project reference paper `refer/attres.pdf`, identify ideas worth borrowing, and assess whether the paper's method matches the in-repo `attn_residuals` comparator closely enough for fair comparison.
- Plan: Extract the paper text, compare its key formulas and systems claims with the current implementation in `model.py`, then summarize reusable design choices and implementation mismatches.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the paper/code comparison results for future architecture and related-work positioning.
- Rationale: The repository already uses `attn_residuals` as a comparator, but that implementation quality depends on how closely it matches the actual paper's algorithmic choices.
- Key details:
  - The paper defines AttnRes as depth-wise softmax attention over prior layer outputs using a per-layer learned pseudo-query vector, with RMSNorm applied on the key side and raw layer outputs used as values.
  - The paper's practical large-scale variant is Block AttnRes, which attends over block summaries rather than every prior layer output.
  - The current in-repo `attn_residuals` baseline matches the high-level idea of depth-wise learned aggregation, but differs from the paper in important ways: it repeatedly mixes `[embedding] + history` before each sublayer, keeps embedding permanently in the candidate pool, and does not implement the paper's blockwise variant or its two-phase inference path.
  - Borrowable ideas include zero-initialized pseudo-queries, RMSNorm-on-score-side only, blockwise depth summaries, and explicit analysis of output/gradient magnitudes across depth.
- Side effects: None on code yet; this is a literature-alignment checkpoint.
- Verification: Read `refer/attres.pdf` via `pypdf` extraction and compared with `src/layer_depth_attention/model.py` around `_attn_res_mix`, `_attn_res_dual_axis_mix`, and residual-attention forward paths.
- Next step: Use these findings either to tighten the `attn_residuals` comparator toward the paper or to clearly label it as an in-repo approximation in future experiment tables.

### [Step 021] - 2026-04-02 23:52 CST - Clarify positional-identity risk in dual-axis design
- Request: Research the concern that the mixed attention may not know the relative position of keys, especially along the vertical/depth axis, and assess whether latent attention is an appropriate fix.
- Plan: Re-check the current dual-axis implementation and separate the issue into horizontal token-position information versus vertical depth-identity information, then record the conclusion in the design report.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/docs/dual_axis_full_architecture.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Expanded the design report's issue section to state that the real missing identity is mainly on the depth axis, while the row branch already has absolute token positions plus causal masking. Added guidance that lightweight depth embeddings or depth-relative bias should be tested before introducing heavier latent-attention machinery.
- Rationale: The original note mixed two different concerns. In the current implementation, horizontal position information already exists indirectly, but vertical history slots still lack explicit layer/depth identity.
- Key details:
  - Row branch: `x_current` already contains token position embeddings and uses causal masking.
  - Depth branch: stacked history slots do not carry explicit layer/depth IDs.
  - Latent attention by itself would change compression/aggregation but would not automatically solve depth identity.
- Side effects: None on code; this is a design-analysis clarification only.
- Verification: Re-read `docs/dual_axis_full_architecture.md` and `src/layer_depth_attention/model.py` around `_attn_res_dual_axis_mix` and `DualAxisMemoryAttention`.
- Next step: If desired, implement a minimal depth-embedding or depth-bias ablation instead of jumping directly to a latent-attention redesign.

### [Step 022] - 2026-04-03 00:06 CST - Add no-position-embedding switch for baseline ablation
- Request: Estimate how much performance is lost when a Transformer baseline has no position embedding, to gauge how much earlier dual-axis variants may have suffered from missing positional identity.
- Plan: Add a minimal `use_pos_emb` switch to the text LM path, change nothing else, and compare a no-pos baseline against the existing same-config baseline.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/model.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext_lm.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `use_pos_emb: bool` to `TinyDecoderLM`.
  - Made the token-position embedding addition conditional in `TinyDecoderLM.forward()`.
  - Added `--use-pos-emb on|off` to the text training script.
  - Added the position-embedding setting to the default output filename so no-pos runs do not overwrite existing baselines.
- Rationale: This isolates exactly one factor—explicit token position embedding—without changing attention type, optimizer, or data pipeline.
- Key details: When `--use-pos-emb off`, the model still keeps the embedding table parameter for shape compatibility, but the forward path skips adding `pos_emb(positions)`.
- Side effects: Existing runs/configs remain valid because the new flag defaults to `on`.
- Verification: Pending local smoke test and no-pos training run.
- Next step: Run local compile/smoke tests, then launch a no-position baseline on the standard text benchmark.

### [Step 047] - 2026-04-05 15:08 CST - Audit main-repo model code for root causes of poor LM perplexity
- Request: Inspect the current model code and identify whether there is a real implementation problem that could explain unexpectedly poor language-model perplexity.
- Plan: Re-read the main-repo `TinyDecoderLM` baseline path in `model.py`, focusing on initialization, baseline attention wiring, output head tying, and anything that would directly distort early logits or baseline training quality.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the model-code audit and the main findings.
- Rationale: The user asked for a direct code inspection rather than more speculation about datasets or platforms.
- Key details:
  - `TinyDecoderLM` in `src/layer_depth_attention/model.py` still has no global `_init_weights` / `self.apply(...)` path for embeddings, linear layers, or layer norms.
  - The model ties `lm_head.weight` to `token_emb.weight`, but both are left at framework-default initialization.
  - This is consistent with the earlier empirical symptom that `step=1` losses were around `200+`, far above the expected `ln(vocab_size)` regime for a sane LM initialization.
  - The baseline forward path itself is structurally normal (`token_emb + pos_emb -> blocks -> final_norm -> lm_head`); no obvious cross-entropy or residual wiring bug was found in the baseline branch.
  - A secondary inefficiency remains: even the baseline path keeps appending `past_kv` and `past_states` lists although the baseline attention ignores them, which wastes memory but should not by itself destroy perplexity.
- Side effects: None on code yet; this is a diagnosis checkpoint.
- Verification: Inspected `src/layer_depth_attention/model.py` around `TinyDecoderLM`, `TransformerBlock`, and `CausalSelfAttention`.
- Next step: If requested, patch the main-repo model with explicit GPT-style initialization and rerun a controlled baseline to see whether step-1 loss returns to the expected scale.

### [Step 056] - 2026-04-06 16:23 CST - Switch server batch to 8-layer dualq-sublayer then baseline
- Request: Interrupt the previous server process and instead run only two methods in this order: `shared_kv_depth_memory_dualq_sublayer`, `baseline`, with `num_layers=8` and `steps=80000`.
- Plan: Add a dedicated Windows launcher for the requested order and hyperparameters, sync it to the server, stop the current run, and relaunch while verifying the new log contents.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer_baseline_8l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added and synced `launch_wikitext103_dualq_sublayer_baseline_8l_80000.bat`, stopped the prior server training process, and started the new two-method 8-layer run.
- Rationale: The user narrowed the comparison to a simpler pairwise run under the same WikiText-103 ablation trainer.
- Key details:
  - New active server process started at `2026/4/6 16:22:38`.
  - Log path: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_dualq_sublayer_baseline_8l_80000.log`
  - Log confirms `steps = 80000` and ordered methods `shared_kv_depth_memory_dualq_sublayer`, `baseline`.
- Side effects: The earlier long-running server job was terminated and replaced by this pairwise run.
- Verification: Remote process list shows a single new `pt-3.9` Python process; the remote log file exists and contains the expected method list and training configuration.
- Next step: Monitor the log for token counts and the first metrics from `shared_kv_depth_memory_dualq_sublayer`.

### [Step 057] - 2026-04-06 21:16 CST - Sync live server files back to local and remove per-forward history restacking
- Request: Pull the server's current versions down to local first, then modify locally, then push the updated files back to the server.
- Plan: Download the live remote `ablation_models.py` and `train_wikitext103_ablation.py`, overwrite the local copies with those remote snapshots, then optimize the dualq memory path by replacing repeated Python-list `torch.stack` work with incrementally maintained stacked history tensors.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/ablation_models.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Pulled the remote files to `/tmp`, backed up the prior local copies, replaced the workspace versions with the server snapshots, changed `SharedKVDepthMemoryDualQAttention.forward()` to consume `(past_keys, past_values)` tensors directly, added `TinyDecoderLM._append_past_kv()` to build those tensors incrementally, and pushed the synchronized+patched files back to the server.
- Rationale: The method only carries about 32 past slots at 16 layers, so the most suspicious avoidable slowdown is not raw history count but repeated Python traversal and `torch.stack([...])` in every block forward.
- Key details:
  - Semantics are preserved: history order and the set of stored K/V states are unchanged.
  - `shared_kv_depth_memory_dualq` and `shared_kv_depth_memory_dualq_sublayer` now keep `past_kv` as stacked tensors shaped `[B, H, S, M, d]`.
  - Remote file timestamps were updated at `2026/4/6 21:15:35`.
- Side effects: Already-running server jobs keep using the old in-memory code until restarted; the pushed files only affect future launches.
- Verification: `python -m py_compile src/layer_depth_attention/ablation_models.py scripts/train_wikitext103_ablation.py` passed locally; remote `Get-Item` confirmed both updated files after upload.
- Next step: When the current run finishes, restart a controlled dualq-sublayer vs baseline comparison and compare step-time / eval-time directly instead of only end-to-end elapsed minutes.

### [Step 058] - 2026-04-06 21:19 CST - Restart single-method 16-layer dualq-sublayer 80000-step server run
- Request: Run the current method alone on the server with `num_layers=16` and `steps=80000` (the user explicitly clarified `80000`, not `8000`).
- Plan: Add a dedicated launcher for the requested single-method 16-layer configuration, sync it to the server, stop the current `pt-3.9` Python training job, and restart the requested run while verifying the remote log header.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer_16l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `launch_wikitext103_dualq_sublayer_16l_80000.bat`, uploaded it to the server, stopped the previous `pt-3.9` Python training process, and started the new run.
- Rationale: The user wants a clean single-method long run on the updated code path to test the 16-layer method directly without mixing in other methods.
- Key details:
  - Log path: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_dualq_sublayer_16l_80000.log`
  - Log header confirms `steps = 80000`
  - Log header confirms methods list contains only `shared_kv_depth_memory_dualq_sublayer`
  - SwanLab run initialized successfully for `shared_kv_depth_memory_dualq_sublayer_wikitext-103-raw-v1_384d_16l`
- Side effects: The previous server training process was interrupted and replaced by this single-method run.
- Verification: Remote log tail showed the requested method name, `steps=80000`, and successful SwanLab initialization.
- Next step: Monitor the first metric lines and, after completion, compare its runtime/quality against the earlier 16-layer baseline run.

### [Step 059] - 2026-04-06 21:41 CST - Stop the active 16-layer dualq-sublayer server run
- Request: Stop the experiment currently running on the server.
- Plan: Terminate any `python.exe` processes under the server's `pt-3.9` environment, then verify that no such process remains and note the current log file timestamp.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Sent a remote PowerShell stop command for all `D:\Annaconda\envs\pt-3.9\python.exe` training processes and verified that the filtered process query returned empty afterwards.
- Rationale: The user wants the active line stopped before continuing with more speed-oriented optimization work.
- Key details:
  - Active log at stop time: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_dualq_sublayer_16l_80000.log`
  - Observed log metadata after stop: `LastWriteTime = 2026/4/6 21:40:55`, `Length = 3109`
- Side effects: The in-progress 16-layer 80000-step dualq-sublayer run was interrupted before completion.
- Verification: Remote post-stop process query returned no matching `pt-3.9` Python processes.
- Next step: Decide whether to profile/optimize locally first or relaunch a controlled timing comparison on the updated code path.

### [Step 060] - 2026-04-07 00:03 CST - Prepare sequential mixed-seq WikiText-103 server batch
- Request: Run three experiments on the server in sequence: method at `seq_len=256` for `80000` steps, then baseline at `seq_len=512` for `80000` steps, then the method again at `seq_len=256` for `80000` steps.
- Plan: Create a dedicated Windows batch launcher that invokes `train_wikitext103_ablation.py` three times with the requested method/sequence-length order, reusing one combined log file for the batch.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_seq_mix_method_base_method_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `launch_wikitext103_seq_mix_method_base_method_80000.bat` with three sequential phases:
  1. `shared_kv_depth_memory_dualq_sublayer`, `--max-seq-len 256`, `--steps 80000`
  2. `baseline`, `--max-seq-len 512`, `--steps 80000`
  3. `shared_kv_depth_memory_dualq_sublayer`, `--max-seq-len 256`, `--steps 80000`
- Rationale: The user wants a direct sequential comparison where the method is run before and after a longer-context baseline under the same general trainer.
- Key details:
  - Assumed the default `num_layers=8` because the user did not specify layer count in this request.
  - Combined log path is `D:\Projects\Layer-Depth-Attention\logs\wikitext103_seq_mix_method_base_method_80000.log`.
- Side effects: None yet; this step only prepares the launcher locally.
- Verification: Launcher file created successfully.
- Next step: Sync the launcher to the server and start the batch, stopping any new active `pt-3.9` training process first if needed.

### [Step 061] - 2026-04-07 01:32 CST - Start sequential mixed-seq WikiText-103 server batch
- Request: Actually run the prepared three-phase server batch: method `seq_len=256` for `80000` steps, then baseline `seq_len=512` for `80000` steps, then the method again `seq_len=256` for `80000` steps.
- Plan: Upload the dedicated launcher, execute it from the server project root, and verify from the shared log that phase 1 started with the requested method and sequence length.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_seq_mix_method_base_method_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Uploaded `launch_wikitext103_seq_mix_method_base_method_80000.bat` to the Windows server and started it; confirmed phase 1 is running with `shared_kv_depth_memory_dualq_sublayer`, `max_seq_len=256`, and `steps=80000`.
- Rationale: The user wants this exact method/baseline/method sequence under mixed context lengths to compare behavior across phases.
- Key details:
  - Combined log path: `D:\Projects\Layer-Depth-Attention\logs\wikitext103_seq_mix_method_base_method_80000.log`
  - Phase marker written to log: `[phase 1] method seq256 steps80000`
  - Verified config in log shows `max_seq_len = 256`, `steps = 80000`, `methods = ["shared_kv_depth_memory_dualq_sublayer"]`
  - The run is currently using `num_layers = 8` because the request did not override the trainer default.
  - Current device report in the log is `cuda num_gpus = 1`
- Side effects: The sequential batch is now occupying the server until all three phases finish or the user interrupts it.
- Verification: Remote log tail confirmed the phase marker, config header, and successful SwanLab initialization for phase 1.
- Next step: Monitor the combined log until phase 1 emits metrics and later verify that phase 2 switches to `baseline` with `max_seq_len=512`.

### [Step 062] - 2026-04-07 16:44 CST - Rewrite the paper mother draft around the current final method
- Request: Start writing the paper now, with one version intended for submission and another for arXiv.
- Plan: Reuse the existing `docs/paper_draft.md` path but replace the outdated old-mainline content with a new mother draft centered on the current final method and a clear split between submission and arXiv versions.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/docs/paper_draft.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Replaced the old draft, which still focused on earlier WikiText-2/value-reprojection lines, with a new mother draft built around the current final method: **single-q + sublayer + projected K/V**. The new draft now includes title options, an abstract draft, introduction/contribution framing, method description, experiment-section plan, minimum table/figure plan, and explicit guidance on how to split the mother draft into a tighter submission version and a fuller arXiv version.
- Rationale: The writing phase should start from the method and evidence that actually survived the experimental selection process, not from an older exploratory line that is no longer the paper's main story.
- Key details:
  - The new draft explicitly treats the paper as a mechanism paper with stable but moderate gains.
  - It distinguishes between theoretical cost and prototype implementation overhead.
  - It reserves extra variants/curves for arXiv or appendices instead of overloading the main submission story.
- Side effects: The previous `paper_draft.md` content is no longer the active paper direction and should not be used as the main writing base.
- Verification: Re-read the new `docs/paper_draft.md` after patching to ensure the final-method definition, paper positioning, and table plan match the current experimental state.
- Next step: Fill in the main result numbers, draft the experimental setup section with exact hyperparameters, and then create a separate submission-outline file if the user wants a more journal/conference-shaped manuscript next.

### [Step 063] - 2026-04-07 16:53 CST - Sync server-side experiment code back to local workspace
- Request: Pull the current server-side version back to the local computer first, because some code changes were made on the server earlier.
- Plan: Inspect the server repository state and commit history, identify which files diverged from the local workspace since the local base commit, back up the current local copies, and then overwrite the local workspace with the server versions for the affected experiment files only.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/src/layer_depth_attention/ablation_models.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/train_wikitext103_ablation.py`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_ablation_default.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_baseline_attnresidual.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer_16l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_dualq_sublayer_baseline_8l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_four_methods_16l_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_wikitext103_seq_mix_method_base_method_80000.bat`, `/Users/a/Projects/Layer-Depth-Attention/notebooks/103 without model.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Queried the server repo and found it clean but ahead of local (`server HEAD=fef65f5`, local `HEAD=b7a0dab`). Retrieved the changed-file list from `git diff --name-only b7a0dab..HEAD` on the server, backed up the current local experiment files to `/tmp/layer_depth_attention_sync_backup`, and then copied the server versions of the changed training/model/launcher/notebook files into the local workspace.
- Rationale: The user wants the local machine to reflect the code that was actually being edited and run on the server, rather than continuing with a diverged local experiment branch.
- Key details:
  - Server commits ahead of local: `22615c9 add lightattention`, `fef65f5 feat ：modify attres`
  - Synced files were limited to the server-changed experiment files; paper draft and memory files were intentionally left alone.
  - Local backups were created under `/tmp/layer_depth_attention_sync_backup`.
- Side effects: The local workspace now reflects the server-side experiment code for the synced files, so any local-only edits in those files are preserved only in the `/tmp` backup copies.
- Verification: Remote `git status --short` was empty; selected files were copied down via `scp`; `python -m py_compile src/layer_depth_attention/ablation_models.py scripts/train_wikitext103_ablation.py` passed locally after the sync.
- Next step: If needed, inspect the synced server changes in `ablation_models.py` / `train_wikitext103_ablation.py`, then decide whether to push this reconciled state to GitHub as the new single source of truth.

### [Step 064] - 2026-04-07 17:03 CST - Add fill-in experiment table templates to the paper draft
- Request: Put experiment tables directly into the paper draft so the user can fill the numbers in there.
- Plan: Extend the mother draft with a dedicated section containing editable templates for the main results table, key ablation table, efficiency table, training-curve checklist, and an optional seed/stability table.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/docs/paper_draft.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added Section 12 to `docs/paper_draft.md`, containing markdown table templates for the main results, ablations, efficiency metrics, curve checklist, and seed runs, plus the `tokens/s` formula to make later filling straightforward.
- Rationale: The user wants to fill the experiment numbers directly into the draft instead of collecting them elsewhere first.
- Key details:
  - The main table already includes the most likely core settings: 8-layer and 16-layer, `seq_len=256`, and optional `seq_len=512`.
  - The ablation table is aligned with the current final method choice: `single-q + sublayer + projected K/V`.
  - The efficiency table includes `step time`, `tokens/s`, and `peak GPU memory` columns.
- Side effects: None; this only affects the draft-writing workflow.
- Verification: Re-read the appended Section 12 in `docs/paper_draft.md` after patching.
- Next step: Fill the main results rows using the current completed runs, then turn those filled templates into polished submission tables later.
