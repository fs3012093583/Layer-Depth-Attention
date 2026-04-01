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
