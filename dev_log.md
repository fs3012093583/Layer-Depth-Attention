# Dev Log

## Baseline
- Goal: Stop using the self-hosted ZeroTier bare-repo sync path and return this project to direct Git-based synchronization via GitHub, then sync the current local state to the Windows server immediately.
- Current scope: Local repository remote configuration, current untracked documentation file, remote server repository remote/branch state, and this task's memory records.
- Constraints: Maintain this log as the primary working memory; avoid storing credentials; do not use the self-built `zerotier-local` path for this sync; do not revert unrelated user changes; keep the server project history intact except for the minimum Git operations required to resync it.
- Protected areas: Remote experiment artifacts/data directories; unrelated code or docs not involved in the current sync; secret material.
- Risks / assumptions: GitHub is currently reachable from both local and server; the server's existing `develop` branch state can be aligned safely after preserving visibility into its status; the untracked architecture doc is intended to be included in the sync.

## Current Task History

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
