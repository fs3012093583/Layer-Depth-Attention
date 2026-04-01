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
