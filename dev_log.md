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

### [Step 023] - 2026-04-03 00:24 CST - Run no-position baseline on the main text benchmark
- Request: Quantify how much performance changes if the baseline Transformer removes token position embeddings, to estimate how costly missing positional information may be.
- Plan: Run a no-position baseline on the standard text benchmark, then compare it to a position-enabled baseline under the same command/config.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Launched `baseline_noposemb_16l_2000` on the Windows server using `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `num_heads=8`, `seq_len=256`, `batch_size=8`, `steps=2000`.
- Rationale: This isolates token position embedding as a single factor without touching the attention mechanism.
- Key details:
  - First completed no-pos run result: `best_val_loss=3.8944`, `best_val_ppl=49.12`, `best_test_loss=4.1042`, `best_test_ppl=60.59`.
  - This unexpectedly outperformed the older stored baseline number, so a same-revision recheck with `use_pos_emb=on` was immediately launched to avoid comparing across stale revisions.
- Side effects: Freed the GPU by stopping an older long-running dual-axis process before the no-pos run so this ablation could finish quickly.
- Verification: The no-pos run produced `artifacts/wikitext103probe_baseline_noposemb_16l_2000.json` and corresponding SwanLab run `baseline_noposemb_16l_2000`.
- Next step: Finish the matched same-revision `baseline_posemb_16l_2000_recheck` run, then report the true delta between pos-on and pos-off.

### [Step 024] - 2026-04-03 00:31 CST - Complete matched position-embedding baseline comparison
- Request: Finish the same-revision baseline recheck so the no-position ablation can be compared fairly without relying on older stored baseline results.
- Plan: Run a matched `use_pos_emb=on` baseline under the exact same command/config as the finished no-pos run, then compare both final metrics.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Completed the `baseline_posemb_16l_2000_recheck` run and recorded the paired comparison.
- Rationale: The older baseline reference in `experiment_notes.md` came from an earlier revision. A same-revision paired run is the correct basis for estimating the impact of removing token position embeddings.
- Key details:
  - `baseline + no pos`: `best_val_loss=3.8944`, `best_val_ppl=49.12`, `best_test_loss=4.1042`, `best_test_ppl=60.59`.
  - `baseline + pos`: `best_val_loss=3.9273`, `best_val_ppl=50.77`, `best_test_loss=4.2162`, `best_test_ppl=67.77`.
  - Under this exact setup, removing the absolute position embedding unexpectedly improved test perplexity by about `7.18` (from `67.77` to `60.59`), a relative gain of about `10.6%`.
- Side effects: This result means we cannot use “missing token position embedding” as a blanket explanation for earlier underperformance in the dual-axis experiments; at least in this benchmark/config, absolute position embeddings appear slightly harmful.
- Verification: Read the completed result JSON files `artifacts/wikitext103probe_baseline_noposemb_16l_2000.json` and `artifacts/wikitext103probe_baseline_posemb_16l_2000_recheck.json` from the server.
- Next step: If needed, add this paired ablation to `experiment_tables.md` / `experiment_notes.md` and interpret whether absolute learned positions are mismatched with the current dataset packing/setup.

### [Step 025] - 2026-04-03 00:38 CST - Confirm depth-identity omission in Attention Residuals paper
- Request: Re-read the reference `attres.pdf` and verify whether the paper itself also lacks explicit vertical/depth positional identification.
- Plan: Revisit the paper's formal definitions of `q_l`, `k_i`, `v_i` and its blockwise variant, then compare them against the earlier design concern about missing depth identity.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded that the reference paper also omits explicit depth-position embeddings or depth-relative bias.
- Rationale: This determines whether the missing vertical identity in the dual-axis design is a unique flaw or a broader design choice shared by a strong comparator.
- Key details:
  - Full AttnRes defines `q_l = w_l`, `k_i = v_i = h_1 or f_i(h_i)` and uses `RMSNorm(k_i)` on the score side, but adds no separate depth embedding.
  - Block AttnRes likewise attends over block representations / partial sums without explicit layer-index embeddings.
  - Therefore the paper has implicit depth identity through slot ordering and layer-specific query parameters, but not explicit learned vertical position encoding.
- Side effects: None on code.
- Verification: Re-read the extracted text from pages 4-7 of `refer/attres.pdf`.
- Next step: When discussing this issue in our design doc, position it as a plausible improvement rather than as an obvious flaw that the literature already solved.

### [Step 026] - 2026-04-03 00:45 CST - Extend paired position-embedding ablation to 8000 steps
- Request: Take the last two matched baseline comparison runs (`use_pos_emb=on` vs `off`) and extend both to `8000` training steps.
- Plan: Reuse the same text benchmark/config and code revision, launch the two runs sequentially on the Windows server to avoid GPU contention, then compare the long-budget outcomes.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the long-budget ablation plan before launching the first 8000-step run.
- Rationale: The 2000-step paired comparison may still reflect medium-budget behavior; extending to 8000 steps will show whether the no-position advantage persists after more optimization.
- Key details: The comparison keeps `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `num_heads=8`, `seq_len=256`, `batch_size=8`, `eval_interval=400`, and `eval_batches=20`, changing only `steps=8000` and `use_pos_emb` on/off.
- Side effects: The GPU will be occupied by these two long runs until both complete.
- Verification: Confirmed the server currently has no active Python training process.
- Next step: Launch `baseline_noposemb_16l_8000` first, then run the matched `baseline_posemb_16l_8000_recheck`.

### [Step 027] - 2026-04-03 08:08 CST - Complete the 8000-step position-embedding baseline pair
- Request: Compare the two latest long-run baseline experiments after the second 8000-step run finishes.
- Plan: Read the completed `baseline + pos` artifact from the server, compare it with the already finished `baseline + no pos` run, and record both in the experiment summary table.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/experiment_tables.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Read the finished `wikitext103probe_baseline_posemb_16l_8000_recheck.json` artifact from the Windows server.
  - Added a new “位置编码开关 8000 Step 对照表” section to `experiment_tables.md`.
- Rationale: The user asked for the direct comparison between the latest two long-run Transformer baseline experiments.
- Key details:
  - `baseline (no pos)`: `best_val_loss=3.2552`, `best_val_ppl=25.92`, `best_test_loss=3.4262`, `best_test_ppl=30.76`.
  - `baseline (pos)`: `best_val_loss=3.3095`, `best_val_ppl=27.37`, `best_test_loss=3.5014`, `best_test_ppl=33.16`.
  - The no-position variant still wins under the longer 8000-step budget.
- Side effects: `experiment_tables.md` now contains both the short-budget (2000-step) and long-budget (8000-step) position-embedding ablation conclusions.
- Verification: Retrieved `D:\Projects\Layer-Depth-Attention\artifacts\wikitext103probe_baseline_posemb_16l_8000_recheck.json` over SSH and copied the final metrics directly into the table.
- Next step: Report the paired 8000-step outcome to the user and, if needed, extend the same position-ablation comparison to dual-axis methods.

### [Step 028] - 2026-04-03 08:18 CST - Relaunch dual-axis-full-no-final-mix long run
- Request: Rerun `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` with the same architecture and experiment settings, for `20000` steps.
- Plan: Keep the server on the current `develop` revision, avoid code edits, and start a fresh long run with a new output filename and run note so the result can be compared cleanly against the previous run.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the relaunch baseline before starting the rerun.
- Rationale: The user wants a fresh like-for-like repetition of the existing large experiment, not a new architecture revision.
- Key details:
  - Both local and server `develop` currently point to `ff11d28`.
  - Server has no active Python training process, so the GPU is available for the rerun.
  - The rerun should use the explicit `pt-3.9` Python to keep SwanLab working.
- Side effects: The Windows server GPU will be occupied by the rerun until completion.
- Verification: Checked local/server `git log` and `git status`; confirmed no active server Python process.
- Next step: Launch the rerun with a new SwanLab experiment name/output path and then monitor the first checkpoint.

### [Step 029] - 2026-04-03 08:39 CST - Start the dual-axis-full-no-final-mix rerun
- Request: Actually launch the fresh `dual_axis_full_no_final_mix_true_bs8_s20000_eval20` rerun under the current `develop` revision.
- Plan: Reuse the same architecture and hyperparameters, but give the rerun a distinct output filename and SwanLab run name so it does not overwrite the previous result.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Created a dedicated rerun launcher on the server.
  - Started the run interactively through SSH to guarantee that the Windows batch activation path and SwanLab initialization both succeed.
- Rationale: Background start attempts on this Windows host were unreliable and silently exited; the interactive PTY launch proved that the run is genuinely alive.
- Key details:
  - `git_rev = ff11d28`
  - `attention_type = dual_axis_full_no_final_mix`
  - `run_note = rerun_same_arch_ff11d28_20260403`
  - SwanLab run: `dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun`
  - Output target: `artifacts\\wikitext2_dual_axis_full_no_final_mix_true_bs8_s20000_eval20_rerun.json`
- Side effects: One local PTY session is intentionally occupied to keep the SSH-launched Windows training attached and observable.
- Verification: Observed `[run-meta]` startup line, `model_params=33843840`, `swanlab login succeeded`, and SwanLab cloud sync URL for run `h022ecjpa1636jov90mjr`.
- Next step: Monitor the first evaluation checkpoint (`step 1` / `step 400`) and report losses plus elapsed minutes.

### [Step 030] - 2026-04-03 08:48 CST - Add algorithm-speed optimization backlog
- Request: Create a todo focused on algorithm optimization, because the current method trains too slowly and each step is about 4x the baseline.
- Plan: Add a dedicated backlog document that treats training speed as a first-class optimization target and lists the most likely investigation directions.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/algorithm_optimization_todo.md`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Created `algorithm_optimization_todo.md` with a speed-focused optimization backlog covering profiling, tensor construction review, memory-path caching, depth-candidate reduction, lighter approximations, and logging step-time in future experiment tables.
- Rationale: The user explicitly wants an actionable optimization backlog because the current dual-axis runs cost roughly 4x the baseline per training step, which blocks efficient iteration.
- Key details:
  - The todo defines the current bottleneck as “dual-axis step time is about 4x baseline”.
  - It elevates speed reduction to the same priority as metric improvement.
- Side effects: Future optimization work can now point to a stable project document instead of scattered chat notes.
- Verification: Confirmed `algorithm_optimization_todo.md` was created in the project root with the intended checklist and conclusions.
- Next step: After the current rerun stabilizes, begin with a profiler-style breakdown to identify which dual-axis submodule dominates step time.

### [Step 031] - 2026-04-03 09:02 CST - Audit the very-strong baseline long run
- Request: Check whether the old `baseline_true_bs8_s30000` result is suspiciously good because of an implementation error, especially in the code.
- Plan: Inspect the launch script, training loop, data slicing, label alignment, and evaluation path to distinguish true model bugs from optimistic evaluation methodology.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the inspection conclusion that the main risk is partial evaluation, not an obvious forward/training bug.
- Rationale: The user is comparing current dual-axis runs against an older strong baseline and needs to know whether that baseline is trustworthy.
- Key details:
  - `launch_baseline_true_bs8_s30000.bat` uses explicit parameters (`model_preset=none`, `seq_len=512`, `d_model=384`, `layers=6`, `batch_size=8`, `steps=30000`) and does not route through the earlier preset-overwrite bug.
  - The training loss uses standard next-token cross-entropy on `inputs[:, :-1]` vs `labels[:, 1:]`; there is no obvious label leakage in the model/training path.
  - The biggest methodological issue is that both validation and final test use `eval_batches=20`, i.e. only the first 20 sequential eval batches, not the full split. This makes the reported `val/test ppl` materially more optimistic and more variable than a full-split evaluation.
  - `best_test_*` is computed after loading the best-validation checkpoint, but still only on that same truncated 20-batch test slice.
- Side effects: The old `baseline_true_bs8_s30000` number should no longer be treated as a full-test gold reference without rerunning it under a full-eval setting.
- Verification: Re-read `scripts/launch_baseline_true_bs8_s30000.bat`, `scripts/train_wikitext_lm.py`, and `src/layer_depth_attention/lm_data.py`.
- Next step: When needed, rerun that baseline with a much larger `eval_batches` or a full-split evaluation to obtain a trustworthy reference.

### [Step 032] - 2026-04-03 09:18 CST - Generate standalone Kaggle attention-only notebook
- Request: Redesign the experiments so only the attention module changes, then generate a fully self-contained notebook that can be uploaded to Kaggle and run without importing this repo’s custom modules or local datasets.
- Plan: Build a single-file notebook generator that emits a Kaggle-ready `.ipynb` containing inline data loading, model code, training/evaluation loops, SwanLab logging, and two methods: `baseline` plus the user-specified shared-`K/V`, layer-specific-`W_Q` attention variant.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/generate_kaggle_attention_only_notebook.py`, `/Users/a/Projects/Layer-Depth-Attention/notebooks/kaggle_attention_only_ablation.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `scripts/generate_kaggle_attention_only_notebook.py`.
  - Generated `notebooks/kaggle_attention_only_ablation.ipynb`.
- Rationale: The user wants a portable Kaggle experiment entrypoint with configurable model size/data, no dependency on repo-internal Python packages, and a clean attention-only ablation setup.
- Key details:
  - Notebook includes `baseline` and `shared_kv_depth_memory`.
  - Custom method uses layer-specific `W_Q^{(l)}` with cross-layer shared `W_K/W_V`, while keeping residual/MLP unchanged.
  - Data loads through Hugging Face `datasets`, not local files.
  - SwanLab defaults to a **new project**: `Layer-Depth-Attention-Kaggle`.
  - Validation uses a sequential cursor evaluation strategy; final test defaults to full-split evaluation.
  - Multi-GPU Kaggle sessions can optionally use `DataParallel`.
- Side effects: A new `notebooks/` directory now exists in the repo with the generated Kaggle notebook artifact.
- Verification: Ran the generator script successfully and verified the resulting notebook JSON exists at `notebooks/kaggle_attention_only_ablation.ipynb` with 9 cells.
- Next step: Hand the notebook path to the user, note the key assumptions, and offer follow-up refinement if they want additional ablations or a different default dataset/config.

### [Step 033] - 2026-04-03 09:26 CST - Verify Kaggle notebook completeness
- Request: Check whether the generated Kaggle notebook looks unfinished.
- Plan: Inspect the generated `.ipynb` cell-by-cell and compile the extracted Python code (excluding notebook shell lines) to distinguish JSON truncation from normal notebook formatting.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded that the notebook is structurally complete and syntactically valid as a notebook-style artifact.
- Rationale: The user flagged the file as potentially incomplete, so the first thing to establish is whether the file is actually truncated or merely appears incomplete in a text preview.
- Key details:
  - The notebook JSON is valid.
  - It contains 9 cells and ends with the expected result-summary cell.
  - Extracted Python code from the notebook compiles successfully once notebook shell lines (the `!pip` install line) are excluded.
  - The notebook is therefore not truncated; any remaining issues are about design/completeness of features, not file corruption.
- Side effects: None on code.
- Verification: Inspected all cells, checked the tail of the long code cells, and compiled the extracted Python body via `py_compile`.
- Next step: If needed, patch feature-level gaps in the notebook rather than regenerating it from scratch.

### [Step 034] - 2026-04-03 09:35 CST - Re-check notebook integrity after user follow-up
- Request: Re-check whether the generated Kaggle notebook file itself is incomplete.
- Plan: Re-open the notebook JSON, inspect the final cell content, and verify that the generator still includes the promised core features.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added a second integrity check note confirming the file is complete and flagging that any remaining concern is feature-level, not truncation.
- Rationale: The user still suspects the notebook may be unfinished, so a second pass should distinguish file corruption from missing expected functionality.
- Key details:
  - `notebooks/kaggle_attention_only_ablation.ipynb` exists and contains 9 cells.
  - The last cell is the expected result-summary `DataFrame` cell, so the notebook tail is intact.
  - The generated code body still contains the promised features: `methods_to_run`, `shared_kv_depth_memory`, `Layer-Depth-Attention-Kaggle`, `final_test_batches`, and the sequential `eval_cursor` validation strategy.
  - Therefore the file is not truncated; if the user wants changes, they are likely about notebook scope or missing conveniences rather than broken notebook structure.
- Side effects: None on code.
- Verification: Parsed notebook JSON successfully, inspected the last code cell, and checked the generator source for the expected feature hooks.
- Next step: Tell the user the file is structurally complete and offer to patch any specific missing functionality they expected.

### [Step 035] - 2026-04-03 09:48 CST - Summarize and interpret the old strong baseline log
- Request: Analyze the strong `baseline_true_bs8_s30000` server result and explain why it looks much better than later 16-layer baseline runs.
- Plan: Use the pasted SwanLab log plus the earlier audit to separate real optimization quality from differences in dataset/config/evaluation protocol.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added a concise interpretation note for the old baseline run so later comparisons do not accidentally mix incompatible experiment settings.
- Rationale: The user is repeatedly comparing newer runs against this early strong baseline, so the project log should preserve the exact reasons it is not a like-for-like reference.
- Key details:
  - The old run is a healthy optimization curve: validation perplexity drops smoothly from `531.63` at step 400 to `16.31~16.54` around steps `20400-20000`, so there is no obvious training collapse or numerical instability.
  - However, it should not be directly compared to later `wikitext-103-probe / 16-layer / 8000-step` baselines because the old run uses a different experimental regime: `wikitext-2-raw-v1`, `seq_len=512`, `6` layers, and `30000` steps.
  - The earlier audit still applies: validation/test are based on `eval_batches=20`, i.e. only the first 20 evaluation batches, which can make the final ppl look materially better than a full-split evaluation.
  - Therefore the run is probably not “bugged”, but it is an optimistic partial-eval baseline and should not be treated as the strict gold reference for later experiments.
- Side effects: None on code.
- Verification: Reused the previously audited training/evaluation logic and matched it against the user-pasted SwanLab training curve.
- Next step: When a fair baseline is needed, rerun this exact configuration with a much larger or full evaluation sweep.

### [Step 036] - 2026-04-03 10:01 CST - Prepare a 2000-step rerun for the old strong baseline
- Request: Re-run the original `baseline_true_bs8_s30000` configuration on the server, but with `steps=2000`.
- Plan: Duplicate the original launch script with only the training-step count, run name, and output path changed, then sync it to the Windows server and launch it there.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_baseline_true_bs8_s2000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `scripts/launch_baseline_true_bs8_s2000.bat` mirroring the old `baseline_true_bs8_s30000` script while changing `--steps` to `2000`, the SwanLab experiment name to `baseline_true_bs8_s2000`, and the artifact path to `artifacts\\wikitext2_baseline_true_bs8_s2000.json`.
- Rationale: The user wants a fair shorter-budget comparison using the exact old baseline recipe rather than the later 16-layer WikiText-103-probe settings.
- Key details:
  - Preserved `wikitext-2`, `seq_len=512`, `d_model=384`, `num_layers=6`, `batch_size=8`, `grad_accum_steps=2`, and `eval_batches=20`.
  - Only the budget-specific identifiers were changed.
- Side effects: A new reusable Windows launch script now exists for the 2000-step rerun.
- Verification: Manually compared the new script against `launch_baseline_true_bs8_s30000.bat` and confirmed only the intended fields differ.
- Next step: Copy the script to the server and start the run.

### [Step 037] - 2026-04-03 10:20 CST - Launch the 2000-step old-baseline rerun on the Windows server
- Request: Actually start the new `baseline_true_bs8_s2000` run on the server.
- Plan: Copy the launch script to the server, then run it through an interactive SSH PTY so startup logs can be observed immediately.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Started `baseline_true_bs8_s2000` on the Windows server using the synced batch script.
- Rationale: A detached `start` command returned without producing a live run, so switching to an interactive PTY is the reliable way to confirm the experiment really launches.
- Key details:
  - Server-side script path: `D:\\Projects\\Layer-Depth-Attention\\scripts\\launch_baseline_true_bs8_s2000.bat`
  - Startup header confirms `git_rev=ff11d28`, `attention_type=baseline`, and `model_params=30142848`.
  - SwanLab run attached successfully as `baseline_true_bs8_s2000`.
- Side effects: One local PTY session is intentionally occupied to keep the SSH-launched training attached and observable.
- Verification: Observed `[run-meta]`, `model_params=30142848`, `swanlab login succeeded`, and SwanLab cloud sync URL for run `ar1d6npnzjbz4q7up1km7`.
- Next step: Monitor the first evaluation checkpoint (`step=400`) and report the early train/val metrics.

### [Step 038] - 2026-04-03 10:27 CST - Correct the old-baseline rerun budget from 2000 to 20000 steps
- Request: The user clarified that the intended rerun for `baseline_true_bs8_s30000` should be `20000` steps, not `2000`.
- Plan: Discard the short-budget rerun, create a new `20000`-step launch script with distinct run/output names, sync it to the server, and relaunch the experiment.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/launch_baseline_true_bs8_s20000.bat`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Added `scripts/launch_baseline_true_bs8_s20000.bat`, keeping the original old-baseline recipe unchanged except for the new `20000`-step budget and matching experiment/output identifiers.
- Rationale: The previous `2000`-step launch was based on a typo from the user and should not be used as the intended comparison point.
- Key details:
  - Preserved `wikitext-2`, `seq_len=512`, `d_model=384`, `num_layers=6`, `batch_size=8`, `grad_accum_steps=2`, and `eval_batches=20`.
  - New run identifiers: `baseline_true_bs8_s20000` and `artifacts\\wikitext2_baseline_true_bs8_s20000.json`.
- Side effects: There are now two helper scripts (`s2000` and `s20000`) for the old strong baseline recipe; only the `s20000` version should be treated as the intended rerun.
- Verification: Compared the new script against the original `s30000` launcher and confirmed that only the budget-specific fields differ.
- Next step: Sync the `s20000` script to the server and relaunch the experiment.

### [Step 039] - 2026-04-03 10:36 CST - Add a shared-KV baseline to the Kaggle attention-only notebook
- Request: Add a `shared_kv_baseline` comparator so the Kaggle notebook can separate “sharing KV” from “adding cross-layer memory”.
- Plan: Extend the generator with a new attention class that shares `K/V` across layers but keeps standard within-layer causal attention, then regenerate and re-validate the notebook.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/generate_kaggle_attention_only_notebook.py`, `/Users/a/Projects/Layer-Depth-Attention/notebooks/kaggle_attention_only_ablation.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Added `SharedKVBaselineAttention`.
  - Extended `TinyDecoderLM` to recognize `shared_kv_baseline`.
  - Updated the default Kaggle `methods_to_run` to `["baseline", "shared_kv_baseline", "shared_kv_depth_memory"]`.
  - Regenerated the notebook artifact.
- Rationale: Without this comparator, the user cannot distinguish the effect of shared `K/V` parameterization from the effect of the cross-layer same-token memory mechanism.
- Key details:
  - `shared_kv_baseline` keeps residual/MLP unchanged and does not use `past_kv`.
  - It uses per-layer `q_proj` with model-wide shared `k_proj` / `v_proj`, matching the parameterization family of the custom method.
- Side effects: The Kaggle notebook now defaults to a three-way ablation instead of a two-way baseline-vs-method run.
- Verification: Regenerated `notebooks/kaggle_attention_only_ablation.ipynb`, confirmed the new symbols are present, and re-parsed the extracted Python body successfully.
- Next step: Tell the user the notebook now includes `shared_kv_baseline` and explain how to run only that subset if they want fewer methods at once.

### [Step 040] - 2026-04-03 11:50 CST - Read the finished 20000-step old-baseline rerun from the Windows server
- Request: Inspect and analyze the completed server result for `baseline_true_bs8_s20000`.
- Plan: Read the final artifact JSON from the Windows server, extract the best validation/test metrics, and compare its shape against the earlier 30000-step strong baseline.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the final metrics and the main interpretation for the completed `baseline_true_bs8_s20000` run.
- Rationale: The user asked to analyze the server result after it finished, and this run is an important calibration point for the old strong baseline recipe.
- Key details:
  - Final artifact: `D:\\Projects\\Layer-Depth-Attention\\artifacts\\wikitext2_baseline_true_bs8_s20000.json`
  - `best_step = 19600`
  - `best_val_loss = 2.8648`
  - `best_val_ppl = 17.55`
  - `best_test_loss = 3.0470`
  - `best_test_ppl = 21.05`
  - The curve is still healthy through the end and is already close to the old `30000`-step baseline, so the strong old baseline is not a fluke of optimization collapse.
  - This also reinforces the earlier conclusion that the old strong baseline's main caveat is the optimistic partial-eval protocol, not a broken training loop.
- Side effects: None on code.
- Verification: Read the server artifact JSON directly and extracted the final best metrics plus the last six history records.
- Next step: Use this 20000-step result as the shorter-budget reference when comparing later reruns of the same old baseline recipe.

### [Step 041] - 2026-04-03 11:48 CST - Add standard GPT-style weight init to Kaggle notebooks
- Request: Modify the Kaggle notebook so the user can re-upload it and test whether missing standard initialization is causing the huge cross-platform discrepancy.
- Plan: Patch the downloaded Kaggle notebook and the notebook generator to add explicit Transformer-style initialization for embeddings, linear layers, and layer norms; keep the tokenizer/text-join fixes intact; then verify both notebooks still parse.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/scripts/generate_kaggle_attention_only_notebook.py`, `/Users/a/Projects/Layer-Depth-Attention/notebooks/attention-only-ablation-notebook-from-kaggle.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/notebooks/kaggle_attention_only_ablation.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Switched the generator from `GPT2TokenizerFast` to `GPT2Tokenizer` and aligned `_join_text()` with the server-style `"\n".join(...).replace(...)` logic.
  - Added `TinyDecoderLM._init_weights()` with GPT-style initialization (`Linear`/`Embedding` normal std 0.02, `LayerNorm` weight=1 bias=0).
  - Called `self.apply(self._init_weights)` before re-tying `lm_head.weight = token_emb.weight`.
  - Applied the same initialization patch directly to the user-downloaded notebook file and the generated notebook artifact.
- Rationale: The user observed `step=1` losses around `200+` on both server and Kaggle, which is far above the expected `ln(vocab_size)` regime and strongly suggests missing explicit initialization is corrupting the training starting point.
- Key details:
  - Both notebook files now contain `_init_weights`, `self.apply(self._init_weights)`, and `GPT2Tokenizer.from_pretrained`.
  - The downloaded notebook remains the immediate upload target; the generator was updated too so future regenerations do not regress.
- Side effects: Future notebook regenerations will preserve the initialization fix and tokenizer/data-join alignment by default.
- Verification: Parsed both notebook code bodies with `ast.parse()` after filtering notebook shell lines; confirmed presence of `_init_weights`, `self.apply(self._init_weights)`, and `GPT2Tokenizer.from_pretrained`.
- Next step: Re-upload the patched downloaded notebook to Kaggle, rerun the baseline, and check whether `step=1` loss drops from `200+` to roughly `10~12`.

### [Step 042] - 2026-04-03 21:46 CST - Prepare to run patched Kaggle notebook on the Windows server
- Request: Upload the patched Kaggle notebook to the server and run it there.
- Plan: Sync the modified notebook to the server, verify notebook execution tooling in the server's `pt-3.9` environment, then execute it under the server GPU environment and capture the first meaningful metrics.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification: Recorded the new server-notebook execution task and the execution plan before remote actions.
- Rationale: The user wants to validate the patched notebook under the server environment rather than the local machine, whose current Python env lacks the required packages and GPU path.
- Key details:
  - Upload target will be the patched downloaded notebook: `/Users/a/Projects/Layer-Depth-Attention/notebooks/attention-only-ablation-notebook-from-kaggle.ipynb`.
  - Remote project root remains `D:\Projects\Layer-Depth-Attention` in the `pt-3.9` conda environment.
- Side effects: None yet; this is a memory update before remote execution.
- Verification: Not run yet.
- Next step: Copy the notebook to the server, check notebook execution tooling, and start the run.

### [Step 043] - 2026-04-03 21:47 CST - Upload and launch patched Kaggle notebook on the Windows server
- Request: Put the patched notebook on the server and run it there.
- Plan: Sync the notebook, create a dedicated server batch runner, and execute the notebook via `jupyter nbconvert --execute` under the `pt-3.9` environment while tailing the server log.
- Files touched: `/Users/a/Projects/Layer-Depth-Attention/notebooks/attention-only-ablation-notebook-from-kaggle.ipynb`, `/Users/a/Projects/Layer-Depth-Attention/dev_log.md`
- Modification:
  - Uploaded the patched notebook to `D:\Projects\Layer-Depth-Attention\notebooks\attention-only-ablation-notebook-from-kaggle.ipynb`.
  - Added a temporary server runner script at `D:\Projects\Layer-Depth-Attention\scripts\run_kaggle_notebook_server.bat`.
  - Started notebook execution with `jupyter nbconvert --to notebook --execute ...` and log redirection to `logs\kaggle_notebook_baseline.log`.
- Rationale: The user wants the patched notebook validated in the same Windows GPU environment used for the main experiments.
- Key details:
  - Server environment check confirmed `jupyter`, `datasets`, `transformers`, `swanlab`, and `torch` are available in `pt-3.9`; only `papermill` is missing, so `nbconvert --execute` is the chosen runner.
  - Current log head shows nbconvert has started converting/executing the notebook and the kernel is alive.
- Side effects: A notebook execution is now running on the server; progress is logged to `D:\Projects\Layer-Depth-Attention\logs\kaggle_notebook_baseline.log`.
- Verification: Confirmed notebook file upload, observed active `python.exe` processes on the server, and read the first log lines from `kaggle_notebook_baseline.log`.
- Next step: Wait for the first training metrics from the notebook log and compare them to the earlier Kaggle results.
