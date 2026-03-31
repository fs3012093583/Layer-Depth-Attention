# Dev Log

## Baseline
- Goal: Build and evaluate Layer-Depth-Attention against a standard decoder-only baseline on a standard LM benchmark that fits on the user's RTX 4060 Ti 16GB server.
- Current scope: Pivot from synthetic associative recall to `WikiText-2` with a small decoder-only GPT-style model, while preserving the baseline architecture and swapping only the self-attention module for the `depth_memory` variant.
- Constraints: Maintain this log as the primary working memory; avoid storing credentials; preserve a fair comparison by changing as little as possible between baseline and custom attention; keep experiments feasible in the remote `pt-3.9` environment; remote server cannot reliably fetch internet resources directly, so standard data/tokenizer assets must be staged locally and synced.
- Protected areas: Do not alter the stored server credential reference; do not overwrite unrelated remote directories; keep the existing method note intact unless experiment results require documentation updates; do not regress the synthetic benchmark scripts while adding the LM benchmark path.
- Risks / assumptions: `datasets` and `tokenizers` are not installed remotely, but `transformers` is available; `GPT2Tokenizer.from_pretrained('gpt2')` cannot download remotely due network restrictions, so local asset vendoring is required; early standard-benchmark runs should stay small enough to validate the pipeline before spending longer GPU time.

## Action History

### [Step 001] - 2026-03-30 18:28 CST - Bootstrap log and project document
- Request: Create a document in the repository and write the provided Method description into it.
- Plan: Initialize `dev_log.md`, then add a dedicated Markdown method file at the project root with the user's supplied structure and notation.
- Files touched: `dev_log.md`, `method_layer_depth_memory_attention.md`
- Modification: Initialized the dev log baseline and prepared the first project document for the method write-up.
- Rationale: The task involves repository edits and should use persistent step-by-step memory per project instructions.
- Key details: The repo started empty; no prior `dev_log.md` or docs existed.
- Side effects: Establishes the initial documentation layout for later theory and literature-positioning work.
- Verification: Not run.
- Next step: Review the created method document and append a follow-up log entry with the exact outcome.

### [Step 002] - 2026-03-30 18:29 CST - Verify written method note
- Request: Confirm that the repository now contains the requested Method document.
- Plan: Re-read the new Markdown files, check structure and math formatting, then record the verified outcome.
- Files touched: `dev_log.md`
- Modification: Verified that `method_layer_depth_memory_attention.md` contains the full Chinese Method write-up with the requested ten-part structure and display equations; logged the result here.
- Rationale: The skill requires updating persistent memory after each meaningful step, including verification.
- Key details: The current document is a direct write-up of the user's formulation and does not yet add novelty positioning, algorithm pseudocode, or implementation notes.
- Side effects: Leaves a clean checkpoint for the next task, such as adding related-work comparison or converting the note into a paper section layout.
- Verification: Reviewed `method_layer_depth_memory_attention.md` and `dev_log.md` with `sed`.
- Next step: Wait for the next documentation or analysis task; likely compare against prior memory-attention literature.

### [Step 003] - 2026-03-30 18:49 CST - Expand concrete key/value bank definition
- Request: Make the expanded attention-space definition explicit and write out the concrete form of `K` in Section 5 and its use in Section 6.
- Plan: Re-read the current method note, align notation with the existing size claim `n + i - 1`, then replace the placeholder with explicit concatenated `K_k^{(i)}` and `V_k^{(i)}` formulas.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Updated the method note so the expanded bank for query token `k` is explicitly defined as current-layer token projections plus the same token's depth history from previous layers; also changed the attention equations to use `K_k^{(i)}` and `V_k^{(i)}`.
- Rationale: The previous placeholder left the core structure underspecified and the notation needed to match the stated cardinality formula.
- Key details: The current document now corresponds to a per-query-token expansion, not a bank containing all tokens' full cross-layer histories. That interpretation is what makes the dimension equal to `n + i - 1`.
- Side effects: Improves internal consistency and makes later complexity analysis and implementation description easier.
- Verification: Applied a targeted patch to the method note; no tests applicable.
- Next step: If needed, refine the notation further into matrix form or add a pseudocode block for the attention construction.

### [Step 004] - 2026-03-30 18:55 CST - Add masked attention formulation
- Request: Note that the attention computation must include masking on the left-side score structure.
- Plan: Keep the existing per-query-token formulation and add an explicit mask vector `m_k^{(i)}` to the score equation, rather than forcing the notation into a single square matrix that would conflict with the current bank definition.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Revised Section 6 to define a masked score `s_k^{(i)}` and clarified that current-layer token positions use a causal mask while depth-memory positions are visible by default.
- Rationale: The user correctly pointed out that legal attention positions need to be constrained; the previous formula omitted this.
- Key details: Because the document currently defines a query-specific bank `K_k^{(i)}`, the mathematically consistent mask object is a vector over `n+i-1` positions, not one shared square attention mask. A unified matrix form would require redefining the bank construction at the sequence level.
- Side effects: Makes the method statement closer to an implementable attention rule and reduces ambiguity about future-token visibility.
- Verification: Applied a targeted patch to Section 6; no executable tests applicable.
- Next step: Optionally add the explicit piecewise definition of `m_k^{(i)}` or rewrite the whole section in sequence-matrix notation if the paper should emphasize batched implementation.

### [Step 005] - 2026-03-30 19:12 CST - Define mask entries explicitly
- Request: The mask `m_k^{(i)}` was still underspecified and needed a formal definition.
- Plan: Add a piecewise definition for each component `m_{k,r}^{(i)}` so the mask aligns exactly with the concatenation order in `K_k^{(i)}`.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Added an explicit piecewise definition covering three index ranges: visible current-layer positions `1..k`, masked future positions `k+1..n`, and always-visible depth-memory positions `n+1..n+i-1`.
- Rationale: Without a coordinate-level definition, the mask semantics remained ambiguous and too informal for a paper-style method section.
- Key details: The document now explicitly binds the tail segment of the mask to the memory entries `k_k^{(1)}, \dots, k_k^{(i-1)}` via the index mapping `k_k^{(r-n)}`.
- Side effects: Tightens the notation and makes later implementation pseudocode or batching notation easier to derive.
- Verification: Applied a targeted patch to Section 6; no executable tests applicable.
- Next step: If needed, upgrade the whole attention section from token-wise notation to batched matrix notation for implementation readability.

### [Step 006] - 2026-03-30 19:19 CST - Normalize masked-score notation
- Request: Rewrite the masked attention formula in the standard form.
- Plan: Change the score definition so dot-product logits are scaled first and the mask is added afterward.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Updated `s_k^{(i)}` from `(qK^T + m)/sqrt(d)` to `qK^T/sqrt(d) + m`.
- Rationale: This is the conventional masked-attention notation and avoids the misleading implication that the mask should also be scaled.
- Key details: The change is not semantic when illegal entries use `-\infty`, but it is cleaner and more standard for a paper.
- Side effects: Improves notation quality without changing the defined mechanism.
- Verification: Applied a targeted patch to Section 6; no executable tests applicable.
- Next step: Continue tightening the paper notation if needed, especially depth-index encoding or batched matrix form.

### [Step 007] - 2026-03-30 19:33 CST - Connect remote server and verify environment
- Request: Connect to the user's server, create a project, and begin a small experiment.
- Plan: Read the server profile, connect via the `win-server` SSH alias, create the remote project directory, and verify the default `pt-3.9` Python/PyTorch environment.
- Files touched: `dev_log.md`
- Modification: Connected to the default Windows server, created `D:\Projects\Layer-Depth-Attention`, identified the correct conda root at `D:\Annaconda`, and verified `Python 3.9.20`, `torch 1.13.1+cu116`, and CUDA visibility for `NVIDIA GeForce RTX 4060 Ti`.
- Rationale: The experiment needs a confirmed remote execution target before any training job can be started.
- Key details: The recorded conda path in the skill reference was not directly usable; the working activation script is `D:\Annaconda\Scripts\activate.bat`.
- Side effects: Establishes the remote workspace and execution assumptions for upcoming runs.
- Verification: Confirmed remote directory creation and environment details through SSH commands.
- Next step: Build a tiny local experiment scaffold, verify it locally, then sync it to the remote server.

### [Step 008] - 2026-03-30 19:33 CST - Create tiny baseline experiment scaffold
- Request: Start with a small runnable experiment instead of only discussing architecture.
- Plan: Implement a minimal decoder-only Transformer baseline plus an on-the-fly associative-recall dataset that requires no external downloads.
- Files touched: `README.md`, `src/layer_depth_attention/__init__.py`, `src/layer_depth_attention/data.py`, `src/layer_depth_attention/model.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Added a runnable project skeleton with a causal self-attention baseline model, synthetic associative-recall data generator, training script, and a short README with remote run instructions.
- Rationale: A tiny synthetic baseline is cheap to run on the remote GPU and provides the cleanest starting point before plugging in Layer-Depth Memory Attention.
- Key details: The scaffold intentionally uses only PyTorch and the standard library to minimize environment friction on the Windows server.
- Side effects: Creates a straightforward baseline for later ablations and custom-attention integration.
- Verification: Local smoke test succeeded with `python3 train_assoc_recall.py --device cpu --steps 2 --batch-size 4 --eval-interval 1`, producing `artifacts/assoc_recall_baseline.json`.
- Next step: Sync the repository contents to `D:\Projects\Layer-Depth-Attention` and verify the remote training script starts under CUDA.

### [Step 009] - 2026-03-30 19:36 CST - Sync scaffold to the Windows server
- Request: Move the first experiment to the user's server project.
- Plan: Copy the local project files to `D:\Projects\Layer-Depth-Attention`, then clean any accidental cache artifacts.
- Files touched: `dev_log.md`
- Modification: Synced `README.md`, `train_assoc_recall.py`, `src/`, `method_layer_depth_memory_attention.md`, and `dev_log.md` to the remote project directory; later removed the accidentally copied `__pycache__` directories locally and remotely.
- Rationale: The remote server is the intended execution target for subsequent experiments.
- Key details: The initial `scp -r src` copied Python bytecode caches from the local smoke test; they were removed afterward to keep the project tree clean.
- Side effects: The remote project now mirrors the local scaffold closely enough for immediate training runs.
- Verification: `scp` transfers completed successfully and remote cache cleanup completed without errors.
- Next step: Launch a short CUDA run on the server and record the first metrics.

### [Step 010] - 2026-03-30 19:37 CST - Run first remote CUDA baseline
- Request: Start a small real experiment on the server.
- Plan: Activate `pt-3.9`, run a short 20-step associative-recall training job on CUDA, and confirm the metrics artifact is written.
- Files touched: `dev_log.md`
- Modification: Ran `python train_assoc_recall.py --device cuda --steps 20 --batch-size 32 --eval-interval 10` on the remote RTX 4060 Ti and produced `D:\Projects\Layer-Depth-Attention\artifacts\assoc_recall_baseline.json`.
- Rationale: This verifies that the baseline scaffold is not just syntactically valid but actually runnable on the target GPU machine.
- Key details: The short run reported `train_loss` falling from `4.2492` to `2.3142`, `eval_loss` from `4.0174` to `2.2360`, and `eval_acc` reaching `0.1844` by step 20.
- Side effects: Establishes the first reference result for future comparisons once the custom attention is implemented.
- Verification: Remote SSH run completed successfully with saved metrics.
- Next step: Decide whether to extend the baseline experiment, add plotting/logging, or begin integrating Layer-Depth Memory Attention into the attention module.

### [Step 011] - 2026-03-30 19:44 CST - Extend baseline to a stable reference run
- Request: Run the baseline longer before changing the architecture.
- Plan: Reuse the existing remote setup and extend the associative-recall run to 200 steps to get a clearer convergence curve.
- Files touched: `dev_log.md`
- Modification: Ran `python train_assoc_recall.py --device cuda --steps 200 --batch-size 32 --eval-interval 20` on the remote server and updated the reference metrics file.
- Rationale: A longer baseline is needed before evaluating whether the custom attention changes convergence speed or final accuracy.
- Key details: The run stayed weak through step 120, improved sharply by step 140 (`eval_acc=0.8250`), and reached `eval_acc=1.0000` by step 160 with final `eval_loss=0.0267` at step 200.
- Side effects: Confirms the tiny baseline can fully solve this synthetic task, so later comparisons should focus on data efficiency, convergence speed, or more difficult task settings rather than final solvability alone.
- Verification: Remote CUDA run completed successfully and wrote `D:\Projects\Layer-Depth-Attention\artifacts\assoc_recall_baseline.json`.
- Next step: Either increase task difficulty for a more discriminative benchmark or begin integrating Layer-Depth Memory Attention and compare learning curves against this 200-step baseline.

### [Step 012] - 2026-03-30 19:47 CST - Add multi-head method definition
- Request: Ensure the method document matches the multi-head baseline used in experiments.
- Plan: Extend the paper note from a single-head notation to a strict multi-head formulation with `H` heads, per-head dimension `d_h`, and final `Concat + W_O` output projection.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Updated the method note so Query/Key/Value, token-level depth memory, expanded `K/V` banks, attention scores, and outputs are all defined per head as `q_{k,h}^{(i)}`, `K_{k,h}^{(i)}`, `V_{k,h}^{(i)}`, and `head_{k,h}^{(i)}`.
- Rationale: The baseline model already uses multi-head self-attention, so the proposed method must be stated in the same architectural regime for a fair comparison.
- Key details: The causal/depth mask `m_k^{(i)}` remains shared across heads, while the depth-memory projections are head-specific.
- Side effects: The document is now closer to the implementation path in `src/layer_depth_attention/model.py`.
- Verification: Applied a targeted documentation patch; no executable tests applicable.
- Next step: Implement the multi-head Layer-Depth Memory Attention variant in code and compare it against the current multi-head baseline.

### [Step 013] - 2026-03-30 19:49 CST - Implement switchable baseline vs depth-memory attention
- Request: Use the proposed method for a direct comparison experiment.
- Plan: Extend the model code so the same training script can instantiate either standard multi-head causal attention or multi-head Layer-Depth Memory Attention under one flag.
- Files touched: `src/layer_depth_attention/model.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Reworked the model into a switchable attention stack with `attention_type in {baseline, depth_memory}`; the custom branch now accumulates prior layers' head-specific `K/V` tensors and appends same-token depth memory during attention computation. The training script now accepts `--attention-type` and writes attention-specific output files.
- Rationale: A fair comparison requires identical training code, optimizer settings, and multi-head structure, with only the attention mechanism changed.
- Key details: The implementation keeps the same pre-norm residual block as the baseline and shares the same causal mask over current-layer token positions; only the extra depth-memory branch differs.
- Side effects: The code path is now ready for same-config A/B runs between baseline and the custom mechanism.
- Verification: Local smoke tests passed for both `--attention-type baseline` and `--attention-type depth_memory` with 2 CPU steps.
- Next step: Sync the updated code to the remote server and run the 200-step depth-memory comparison.

### [Step 014] - 2026-03-30 19:49 CST - Run first baseline vs depth-memory comparison
- Request: Run a direct comparison experiment using the proposed method.
- Plan: Keep the associative-recall setup fixed at 200 CUDA steps and compare `baseline` against `depth_memory` under identical hyperparameters.
- Files touched: `dev_log.md`
- Modification: Synced the updated model/training code to the server and ran `python train_assoc_recall.py --device cuda --steps 200 --batch-size 32 --eval-interval 20 --attention-type depth_memory`, producing `D:\Projects\Layer-Depth-Attention\artifacts\assoc_recall_depth_memory.json`.
- Rationale: This is the first fair A/B test between the standard multi-head baseline and the implemented Layer-Depth Memory Attention variant.
- Key details: The custom model behaved almost identically to the baseline on this task: at step 140 it reached `eval_acc=0.8328` versus baseline `0.8250`, and both models hit `eval_acc=1.0000` by step 160 with nearly identical final `eval_loss` (`0.0266` vs `0.0267`).
- Side effects: The current task is too easy or too shallow to expose a meaningful difference between the two mechanisms.
- Verification: Remote CUDA run completed successfully and saved the depth-memory metrics artifact.
- Next step: Increase benchmark difficulty or task sensitivity before drawing conclusions about the proposed method's usefulness.

### [Step 015] - 2026-03-30 19:54 CST - Add no-FFN-residual ablation switch
- Request: Remove the `x = x + MLP(LN(x))` residual while keeping the normalization.
- Plan: Add an explicit `ffn_residual` switch so the model can run either the standard block or the requested ablation without losing the original comparison path.
- Files touched: `src/layer_depth_attention/model.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Added `ffn_residual` control to the model and CLI. When turned off, the block now computes `mlp_out = MLP(LN(x))` and sets `x = mlp_out` instead of `x = x + mlp_out`.
- Rationale: The user wants to test whether the FFN residual is masking the effect of the custom attention mechanism.
- Key details: The attention residual is still preserved; only the FFN-side residual is ablated. The normalization before the FFN remains intact exactly as requested.
- Side effects: Creates a new structural variant that should be interpreted as an ablation, not as the default implementation.
- Verification: Local smoke tests passed for both `baseline` and `depth_memory` with `--ffn-residual off`.
- Next step: Sync the updated files to the server and run the depth-memory no-FFN-residual experiment.

### [Step 016] - 2026-03-30 19:55 CST - Run depth-memory without FFN residual
- Request: Try the user's preferred variant that removes the FFN residual while keeping layer normalization.
- Plan: Keep the same 200-step associative-recall setup and evaluate `depth_memory` with `--ffn-residual off`.
- Files touched: `dev_log.md`
- Modification: Ran `python train_assoc_recall.py --device cuda --steps 200 --batch-size 32 --eval-interval 20 --attention-type depth_memory --ffn-residual off`, producing `D:\Projects\Layer-Depth-Attention\artifacts\assoc_recall_depth_memory_noffnres.json`.
- Rationale: This directly tests whether the FFN residual was obscuring the effect of the custom attention mechanism.
- Key details: Removing the FFN residual severely hurt training on this task. The model started with relatively high step-1 accuracy (`0.1656`) but failed to improve, ending at `eval_acc=0.1437` and `eval_loss=1.8253`, far worse than the residual-preserving version that reached `1.0000` accuracy by step 160.
- Side effects: Strongly suggests the FFN residual should remain in the main model for stability on the current setup.
- Verification: Remote CUDA run completed successfully and saved the no-FFN-residual metrics artifact.
- Next step: Keep FFN residual in the primary architecture and move on to a harder benchmark if we want to expose any real advantage of depth memory.

### [Step 017] - 2026-03-30 20:00 CST - Compare larger model on harder associative recall
- Request: Keep FFN residual enabled and try a larger model on a bigger dataset/task setting.
- Plan: Increase model size and task difficulty to a still-feasible 4060 Ti configuration, then run both baseline and `depth_memory` under identical hyperparameters.
- Files touched: `dev_log.md`
- Modification: Ran two remote experiments with `d_model=256`, `num_layers=8`, `num_heads=8`, `num_pairs=20`, `vocab_size=128`, `steps=400`, `batch_size=32`, `eval_interval=50`, saving results to `assoc_recall_baseline_large.json` and `assoc_recall_depth_memory_large.json`.
- Rationale: The smaller setup saturated too quickly and could not reveal a meaningful difference between the two attention mechanisms.
- Key details: On this harder setting, the baseline only reached `eval_acc=0.3391` and `eval_loss=2.7179` by step 400, while `depth_memory` reached `eval_acc=0.6250` and `eval_loss=1.9619`. Both models tracked similarly until late training, but the custom method improved much more sharply near the end.
- Side effects: This is the first experiment in the project that shows a nontrivial advantage for Layer-Depth Memory Attention under a fair same-config comparison.
- Verification: Both remote CUDA runs completed successfully and wrote their metrics artifacts.
- Next step: Repeat this larger-setting comparison with additional seeds or slightly longer training to see whether the advantage is stable.

### [Step 018] - 2026-03-30 20:04 CST - Run 1500-step residual vs no-residual comparison
- Request: Run 1500-step experiments comparing the larger `depth_memory` model with and without FFN residual.
- Plan: Use the same larger setting as Step 017, extend training to 1500 steps, and compare `ffn_residual=on` versus `ffn_residual=off`.
- Files touched: `dev_log.md`
- Modification: Launched both long runs on the server. One initial no-residual command failed due to a mistyped CLI flag (`--vocab_size` instead of `--vocab-size`), then was immediately rerun with the correct argument. Final outputs were saved to `assoc_recall_depth_memory_large_1500_ffnres.json` and `assoc_recall_depth_memory_large_1500_noffnres.json`.
- Rationale: The 200-step ablation already suggested the FFN residual was important; this longer and harder setting tests whether that conclusion still holds when the model has more time to optimize.
- Key details: With FFN residual on, the model reached `eval_acc=1.0000` by step 450 and stayed there through step 1500, ending with `eval_loss=0.0017`. With FFN residual off, training essentially stalled for the entire run and ended at only `eval_acc=0.0516` with `eval_loss=3.0267`.
- Side effects: Strongly confirms that the FFN residual should remain part of the main architecture; removing it is not a benign variant but a destructive ablation on this benchmark.
- Verification: Both corrected remote CUDA runs completed successfully and wrote their metrics artifacts.
- Next step: Keep FFN residual fixed in the main model and spend future budget on stability checks across seeds or on harder datasets rather than revisiting this ablation.

### [Step 019] - 2026-03-30 20:07 CST - Add 1500-step baseline reference
- Request: Run the standard model under the same long-training large-setting configuration.
- Plan: Execute the baseline with the exact same `1500`-step large-setting hyperparameters used for the two `depth_memory` residual ablations.
- Files touched: `dev_log.md`
- Modification: Ran `python train_assoc_recall.py --device cuda --steps 1500 --batch-size 32 --eval-interval 150 --d-model 256 --num-layers 8 --num-heads 8 --num-pairs 20 --vocab-size 128 --attention-type baseline --ffn-residual on --output artifacts/assoc_recall_baseline_large_1500_ffnres.json`.
- Rationale: The long-run comparison was incomplete without a baseline trained under the exact same larger-setting schedule.
- Key details: The baseline also solved the task, reaching `eval_acc=0.9984` by step 450 and `1.0000` by step 600, ending with `eval_loss=0.0052` at step 1500. Compared with `depth_memory + residual`, the custom model reached `1.0000` slightly earlier and ended with a lower final `eval_loss` (`0.0017` vs `0.0052`), but the gap is much smaller than the 400-step comparison suggested.
- Side effects: The large-setting story is now clearer: `depth_memory` appears to improve early convergence on this harder task, while both residual-preserving models eventually solve it given enough optimization budget.
- Verification: Remote CUDA run completed successfully and saved the baseline metrics artifact.
- Next step: To claim a robust advantage, run multiple seeds or compare sample-efficiency at fixed intermediate step budgets rather than only final accuracy.

### [Step 020] - 2026-03-30 20:10 CST - Add no-attention-residual ablation switch
- Request: Test the variant that keeps FFN residual but removes the attention residual.
- Plan: Add an `attn_residual` switch alongside the existing `ffn_residual` switch, verify it locally, then evaluate the requested ablation on the larger-setting benchmark.
- Files touched: `src/layer_depth_attention/model.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Added `attn_residual` control to the model and CLI. When turned off, the block now sets `x = attn_out` instead of `x = x + attn_out`, while leaving the FFN path unchanged. Also updated output naming to record both residual settings.
- Rationale: The user wants to know whether the attention residual can be removed while still retaining the FFN residual.
- Key details: This creates a distinct ablation from the earlier no-FFN-residual test; now the FFN residual remains on, only the attention-side residual is removed.
- Side effects: The experiment space now includes four structural combinations of attention/FFN residuals, though only the requested variants have been run.
- Verification: Local smoke tests passed for both `baseline` and `depth_memory` with `--attn-residual off --ffn-residual on`.
- Next step: Sync the updated code to the server and run the larger-setting `depth_memory` no-attention-residual experiment.

### [Step 021] - 2026-03-30 20:12 CST - Run depth-memory without attention residual
- Request: Evaluate the variant that keeps FFN residual but removes attention residual.
- Plan: Use the larger-setting 400-step benchmark and compare this new ablation against the previously recorded baseline and residual-preserving `depth_memory` runs.
- Files touched: `dev_log.md`
- Modification: Ran `python train_assoc_recall.py --device cuda --steps 400 --batch-size 32 --eval-interval 50 --d-model 256 --num-layers 8 --num-heads 8 --num-pairs 20 --vocab-size 128 --attention-type depth_memory --attn-residual off --ffn-residual on --output artifacts/assoc_recall_depth_memory_noattnres_ffnres_large_400.json`.
- Rationale: This directly tests whether the attention residual is dispensable when the FFN residual is still present.
- Key details: The model barely learned at all. It ended at `eval_acc=0.0500` and `eval_loss=3.0270` after 400 steps, dramatically worse than the normal residual-preserving `depth_memory` run on the same setting (`eval_acc=0.6250`, `eval_loss=1.9619`).
- Side effects: Strong evidence that the attention residual is also essential in the current architecture; removing it is another destructive ablation, not a beneficial simplification.
- Verification: Remote CUDA run completed successfully and saved the no-attention-residual metrics artifact.
- Next step: Keep both residuals in the main model and focus future experiments on harder tasks, seed stability, or incremental architectural refinements rather than removing residual paths.

### [Step 022] - 2026-03-30 20:15 CST - Prepare layer-depth sweep
- Request: Measure how increasing model depth changes the gain over the baseline.
- Plan: Keep the successful larger-setting benchmark fixed and compare `baseline` vs `depth_memory` across `num_layers in {4, 8, 12}` with both residuals enabled.
- Files touched: `dev_log.md`
- Modification: Chose the controlled sweep setting: `d_model=256`, `num_heads=8`, `num_pairs=20`, `vocab_size=128`, `steps=400`, `batch_size=32`, `eval_interval=50`, `attn_residual=on`, `ffn_residual=on`, scanning only layer count.
- Rationale: This isolates the effect of increased depth while staying within the 4060 Ti budget and using a task that already showed a meaningful difference at 8 layers.
- Key details: The sweep will start with 4, 8, and 12 layers; if 12 layers is still easy to run and informative, 16 layers can be added later.
- Side effects: Establishes a consistent experiment grid for depth-scaling analysis.
- Verification: Not run yet.
- Next step: Launch the six server runs and collect their final metrics.

### [Step 023] - 2026-03-30 20:17 CST - Complete 4/8/12-layer sweep
- Request: Quantify how depth scaling changes the benefit of the proposed method relative to the baseline.
- Plan: Run the full six-job grid and compare final metrics at a fixed 400-step budget.
- Files touched: `dev_log.md`
- Modification: Completed the six remote runs for `baseline` and `depth_memory` at `4`, `8`, and `12` layers, saving metrics to `layersweep_*` artifacts under the remote `artifacts` folder.
- Rationale: A fixed-step depth sweep is the cleanest way to test whether the method helps as the network gets deeper.
- Key details: Results by layer count:
  `L=4`: baseline `eval_acc=1.0000`, `eval_loss=0.0299`; depth_memory `eval_acc=1.0000`, `eval_loss=0.0331`.
  `L=8`: baseline `eval_acc=0.3391`, `eval_loss=2.7179`; depth_memory `eval_acc=0.6250`, `eval_loss=1.9619`.
  `L=12`: baseline `eval_acc=0.1219`, `eval_loss=2.8947`; depth_memory `eval_acc=0.0641`, `eval_loss=2.9423`.
- Side effects: The current evidence suggests a non-monotonic depth story: the custom method helps substantially at intermediate depth (8 layers) but not at shallow depth (4 layers) and not yet at 12 layers under the same fixed training budget.
- Verification: All six remote CUDA runs completed successfully.
- Next step: Test whether the 12-layer models simply need more optimization budget, smaller learning rate, or gradient stabilization before concluding that the method breaks at higher depth.

### [Step 024] - 2026-03-30 20:20 CST - Configure passwordless SSH for the Windows server
- Request: Set up key-based login for the user's default Windows server.
- Plan: Reuse the local `id_ed25519` key, inspect the server's OpenSSH configuration, install the public key in the correct authorized-keys location, and verify a new SSH connection works with `BatchMode=yes`.
- Files touched: `dev_log.md`
- Modification: Confirmed the local machine already had `~/.ssh/id_ed25519.pub`. Copied that public key to the remote server, discovered the server account is in the Windows `Administrators` group, and therefore installed the key into `C:\ProgramData\ssh\administrators_authorized_keys` rather than relying on `C:\Users\fs301\.ssh\authorized_keys`. Verified successful key-based login with `ssh -o BatchMode=yes win-server`.
- Rationale: Password prompts were slowing down every remote experiment and maintenance action.
- Key details: The server's `sshd_config` uses `Match Group administrators` and redirects administrators to `__PROGRAMDATA__/ssh/administrators_authorized_keys`, which is why the initial user-level `authorized_keys` setup did not work.
- Side effects: Future `ssh win-server` and `scp ... win-server:...` commands from this Mac should no longer prompt for a password.
- Verification: `ssh -o BatchMode=yes -o PreferredAuthentications=publickey win-server "echo KEY_OK && whoami"` returned successfully as `just-book\\fs301`.
- Next step: Reuse the new passwordless connection for all further server experiments and remote file sync.

### [Step 025] - 2026-03-30 20:23 CST - Recheck 12-layer behavior with more optimization budget
- Request: Test whether the poor 12-layer result was caused by insufficient training budget rather than a true structural failure, and consider whether the token sequence may simply be too short.
- Plan: Keep the same 12-layer larger-setting configuration and extend training from 400 to 1200 steps for both `baseline` and `depth_memory`.
- Files touched: `dev_log.md`
- Modification: Ran `L12_baseline_1200_lr3e4.json` and `L12_depth_memory_1200_lr3e4.json` on the remote server with `steps=1200`, `lr=3e-4`, `d_model=256`, `num_heads=8`, `num_pairs=20`, and both residuals enabled.
- Rationale: This directly tests the most likely explanation for the earlier 12-layer degradation without changing too many variables at once.
- Key details: The current input length at `num_pairs=20` is only `43` tokens, but the new runs show that sequence length was not the immediate blocker for 12-layer trainability. Both models reached `eval_acc=1.0000` by step 450 and stayed there through step 1200. Final metrics were nearly identical: baseline `eval_loss=0.0022`, depth_memory `eval_loss=0.0024`.
- Side effects: Strongly indicates that the earlier 12-layer underperformance at 400 steps was mainly an optimization-budget issue, not evidence that the architecture fundamentally breaks at 12 layers.
- Verification: Both remote CUDA runs completed successfully and saved their metrics artifacts.
- Next step: If the goal is to test whether sequence length is now the limiting factor, increase `num_pairs` substantially (for example 40 or 80) while keeping the 12-layer setting fixed.

### [Step 026] - 2026-03-30 20:25 CST - Start 12-layer longer-sequence comparison
- Request: Run a `12`-layer comparison with a longer token sequence and enough training budget.
- Plan: Increase `num_pairs` from `20` to `40`, which raises the input length from `43` to `83`, and compare `baseline` vs `depth_memory` for `1200` steps with the same optimizer and residual settings.
- Files touched: `dev_log.md`
- Modification: Launched the longer-sequence 12-layer baseline and depth-memory runs on the remote server with outputs `L12_pairs40_baseline_1200_lr3e4.json` and `L12_pairs40_depth_memory_1200_lr3e4.json`.
- Rationale: This isolates whether the earlier lack of separation at 12 layers was partly caused by the synthetic sequence being too short to stress the proposed mechanism.
- Key details: Both runs keep `d_model=256`, `num_heads=8`, `lr=3e-4`, `attn_residual=on`, and `ffn_residual=on`; only sequence difficulty is increased.
- Side effects: This is the first experiment in the project that directly tests the method under a meaningfully longer input at 12 layers.
- Verification: Runs launched; results pending.
- Next step: Compare final and intermediate metrics to see whether longer sequences restore a depth-memory advantage at 12 layers.

### [Step 027] - 2026-03-30 20:28 CST - Evaluate 12-layer longer-sequence runs
- Request: Determine whether a longer token sequence changes the 12-layer comparison.
- Plan: Inspect the completed `num_pairs=40` runs and compare convergence under the unchanged learning rate of `3e-4`.
- Files touched: `dev_log.md`
- Modification: Completed the `L12_pairs40_baseline_1200_lr3e4.json` and `L12_pairs40_depth_memory_1200_lr3e4.json` runs.
- Rationale: The user suspected the earlier task might have been too short to reveal the intended benefit.
- Key details: With input length increased to `83`, neither model learned meaningfully at `lr=3e-4`. By step 1200, baseline only reached `eval_acc=0.0297` with `eval_loss=3.7129`, while depth_memory reached `eval_acc=0.0328` with `eval_loss=3.6737`. The difference is negligible and both runs are effectively stalled.
- Side effects: This suggests the new failure mode is not simply "sequence too short"; instead, the 12-layer longer-sequence setup appears harder to optimize under the current learning rate and schedule.
- Verification: Both remote CUDA runs completed successfully and saved their metrics artifacts.
- Next step: Lower the learning rate for the `num_pairs=40`, `12`-layer setting, since optimization rather than architecture is now the most likely bottleneck.

### [Step 028] - 2026-03-30 20:30 CST - Start low-learning-rate rescue for the long-sequence setting
- Request: Continue automatically and diagnose the hard 12-layer, longer-sequence failure.
- Plan: Keep the `12`-layer, `num_pairs=40` setup fixed and lower the learning rate from `3e-4` to `1e-4` for both `baseline` and `depth_memory`.
- Files touched: `dev_log.md`
- Modification: Started preparing the next controlled pair of runs at `lr=1e-4`.
- Rationale: The strongest current hypothesis is that optimization, not sequence length alone, is the bottleneck in the longer-sequence setting.
- Key details: All other settings remain fixed so that any recovery can be attributed to the lower learning rate rather than a confounded change.
- Side effects: If the low-learning-rate runs recover, the next logical step will be to compare intermediate sample efficiency rather than final binary solvability.
- Verification: Not run yet.
- Next step: Launch the low-learning-rate baseline and depth-memory jobs on the server.

### [Step 029] - 2026-03-30 20:32 CST - Evaluate low-learning-rate long-sequence runs
- Request: Continue automatically and check whether lowering the learning rate rescues the 12-layer, longer-sequence setting.
- Plan: Inspect the completed `lr=1e-4` runs and compare them with the earlier `lr=3e-4` failure case.
- Files touched: `dev_log.md`
- Modification: Completed `L12_pairs40_baseline_1200_lr1e4.json` and `L12_pairs40_depth_memory_1200_lr1e4.json`.
- Rationale: This is the cleanest follow-up to the previous stalled long-sequence runs.
- Key details: Lowering the learning rate did not materially help. By step 1200, baseline ended at `eval_acc=0.0203`, `eval_loss=3.7101`, and depth_memory ended at `eval_acc=0.0203`, `eval_loss=3.7101`. Both traces remained effectively flat throughout training.
- Side effects: The current 12-layer, `num_pairs=40` configuration appears to need more than a simple learning-rate reduction; likely candidates are warmup, different batch/optimizer settings, or a smaller/harder-balanced model setting.
- Verification: Both low-learning-rate remote CUDA runs completed successfully and saved their metrics artifacts.
- Next step: Stop spending cycles on blind reruns of this exact configuration and instead redesign the long-sequence experiment setup more carefully before continuing.

### [Step 030] - 2026-03-30 20:41 CST - Pivot to standard LM benchmark assets
- Request: Stop the demo synthetic benchmark and move to an industry-standard dataset/model comparison, while only replacing the corresponding attention module in the baseline.
- Plan: Re-read the log, inspect remote package availability, vendor the required standard dataset/tokenizer assets locally, and then build a separate WikiText-2 training path.
- Files touched: `dev_log.md`
- Modification: Verified remote package availability (`transformers` present; `datasets`, `tokenizers`, and `tiktoken` absent), confirmed direct remote downloads fail, downloaded GPT-2 `vocab.json` and `merges.txt` locally, downloaded the `WikiText-2` raw parquet splits locally, installed local `pyarrow`, and converted the dataset into plain `train.txt`, `validation.txt`, and `test.txt` files under `external_data/wikitext-2-raw-v1`.
- Rationale: A standard LM benchmark is now the main path, and the remote environment needs staged assets because it cannot fetch internet resources reliably on its own.
- Key details: The new benchmark should preserve embeddings, residuals, normalization, MLP, and loss behavior from the baseline model; only the self-attention implementation should differ between baseline and `depth_memory`.
- Side effects: The repo now contains local standard benchmark assets that can be synced to the server without adding new remote package dependencies.
- Verification: Local downloads completed; `train.txt`, `validation.txt`, and `test.txt` were created with sizes about `10M`, `1.1M`, and `1.2M` respectively.
- Next step: Add a standard LM data loader and training script, then run a first remote smoke experiment on WikiText-2 with baseline and `depth_memory`.

### [Step 031] - 2026-03-30 20:45 CST - Add WikiText-2 LM training path
- Request: Move from the synthetic demo to a standard LM benchmark without accidentally changing unrelated model components.
- Plan: Keep the existing decoder-only model backbone, add only the minimum utilities needed for standard language modeling, and create a separate training script so the synthetic path remains intact.
- Files touched: `src/layer_depth_attention/lm_data.py`, `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `lm_data.py` with a local-file GPT-2 tokenizer loader and WikiText contiguous-window batching, added optional tied input/output embeddings to `TinyDecoderLM`, and created `train_wikitext_lm.py` for standard next-token LM training with warmup, cosine decay, gradient accumulation, clipping, validation perplexity, and test evaluation.
- Rationale: The user wants a standard baseline comparison where the only architectural change is the self-attention module replacement.
- Key details: The new training path keeps embeddings, position embeddings, LayerNorm, residuals, MLP, and LM loss unchanged across `baseline` and `depth_memory`; only `attention_type` switches the module. Local smoke execution could not fully run because the local Python environment lacks `transformers`, but syntax compilation succeeded.
- Side effects: The repo now has two experiment paths: synthetic associative recall and standard WikiText-2 LM. The standard path depends on staged tokenizer/data files rather than remote internet access.
- Verification: `python -m py_compile train_wikitext_lm.py src/layer_depth_attention/lm_data.py src/layer_depth_attention/model.py` passed locally; direct local runtime smoke was blocked only by missing local `transformers`.
- Next step: Sync the new code and staged data to the Windows server, then run a remote WikiText-2 smoke test for the baseline path first.

### [Step 032] - 2026-03-30 20:47 CST - Fix tokenizer special-token initialization for offline GPT-2 files
- Request: Keep the standard benchmark path clean and avoid silent tokenizer mismatches on the remote server.
- Plan: Validate the locally staged GPT-2 tokenizer files on the server, patch any offline-loading incompatibilities, then rerun the smoke test.
- Files touched: `src/layer_depth_attention/lm_data.py`, `dev_log.md`
- Modification: Verified that remote `GPT2Tokenizer(vocab_file=..., merges_file=...)` loaded but returned `eos_token_id=None`, then updated `lm_data.py` to set `unk_token`, `bos_token`, and `eos_token` explicitly to `<|endoftext|>` before reusing that token as padding.
- Rationale: The standard LM path inserts EOS markers between WikiText lines and also uses EOS as padding, so a missing EOS id would corrupt batching and evaluation.
- Key details: The issue only appears when loading GPT-2 tokenizer assets directly from local files without the full pretrained config bundle. This fix preserves the standard GPT-2 token semantics while avoiding any remote internet dependency.
- Side effects: The offline tokenizer setup is now self-contained and should behave consistently on the Windows server.
- Verification: Remote inspection exposed the missing EOS token; full smoke rerun pending after resyncing the patched source into the remote `src/layer_depth_attention` package directory.
- Next step: Sync the corrected source files into `src/layer_depth_attention` on the server and run a baseline WikiText-2 smoke experiment.

### [Step 033] - 2026-03-30 20:51 CST - Smoke-test the WikiText-2 path on the server
- Request: Validate that the new standard LM pipeline really works before spending GPU time.
- Plan: Resync the new files into the remote `src` package, verify the staged tokenizer/data directories, and run a minimal one-step baseline job.
- Files touched: `src/layer_depth_attention/lm_data.py`, `dev_log.md`
- Modification: Synced the new benchmark code and local assets to the server, verified the remote `external_data/gpt2_tokenizer` and `external_data/wikitext-2-raw-v1` layouts, and ran `train_wikitext_lm.py` for a one-step CPU smoke test. Also patched `lm_data.py` so the EOS/pad id is resolved directly from the GPT-2 encoder table and tokenized split tensors are cached to `*_gpt2_ids.pt` files for reuse across runs.
- Rationale: The remote environment lacks internet access and some Python packages, so the standard LM path had to be validated in the exact offline server setup.
- Key details: The smoke run succeeded and wrote `artifacts/wikitext2_remote_smoke.json`. Validation/test losses were very high, which is expected for a one-step untrained model. Tokenized corpus sizes were: train `4,424,573`, validation `456,308`, test `526,828` tokens.
- Side effects: Future WikiText-2 runs should start much faster because tokenized splits are cached locally on the server after the first encode pass.
- Verification: Remote smoke command completed successfully; tokenizer/data inspection also succeeded.
- Next step: Resync the caching/EOS-id patch to the server and launch the first small GPU baseline run, followed by the matching `depth_memory` run with identical settings.

### [Step 034] - 2026-03-30 20:57 CST - Run first WikiText-2 baseline vs depth-memory comparisons
- Request: Start comparing the proposed attention against a standard decoder-only GPT-style baseline on a standard LM benchmark, without changing unrelated modules.
- Plan: Use small but meaningful WikiText-2 runs first, holding data, optimizer, residuals, MLP, embeddings, and sequence length fixed while switching only `attention_type`.
- Files touched: `train_wikitext_lm.py`, `dev_log.md`
- Modification: Updated perplexity reporting to use `exp(loss)` unless it truly overflows, then ran two controlled comparison sets on the Windows server:
  1. `L=4`, `d_model=256`, `seq_len=128`, `steps=80`, `batch_size=4`, `grad_accum=4`.
  2. `L=8`, same settings but `steps=100`.
- Rationale: The synthetic task suggested shallow models may hide the difference, so the standard benchmark should be checked at both shallow and deeper settings.
- Key details: Results:
  `L=4` baseline: final `val_loss=22.7100`, `test_loss=20.0834`.
  `L=4` depth_memory: final `val_loss=22.7824`, `test_loss=20.0772`.
  `L=8` baseline: final `val_loss=18.0161`, `test_ppl=4,345,477.68`.
  `L=8` depth_memory: final `val_loss=17.9744`, `test_ppl=4,044,138.73`.
  The 4-layer models are effectively tied, while the 8-layer `depth_memory` run is slightly better on both validation and test.
- Side effects: The standard benchmark path is now producing comparable artifacts under `artifacts/wikitext2_*`; one Windows-specific path quirk was also identified, so remote script execution should use `python .\\train_wikitext_lm.py` rather than `python train_wikitext_lm.py`.
- Verification: All four remote runs completed successfully and wrote metrics JSON files.
- Next step: Extend the comparison to a deeper standard-model setting, likely `12` layers with adjusted effective batch size if needed, to see whether the small 8-layer advantage persists or grows.

### [Step 035] - 2026-03-30 21:02 CST - Extend WikiText-2 comparison to 12 layers
- Request: Keep pushing automatically on the standard benchmark after the initial 4/8-layer comparisons.
- Plan: Run a deeper `12`-layer pair on WikiText-2 with a reduced micro-batch and increased gradient accumulation so the effective token budget stays comparable on the 4060 Ti.
- Files touched: `src/layer_depth_attention/lm_data.py`, `dev_log.md`
- Modification: Ran `L=12`, `d_model=256`, `seq_len=128`, `steps=100`, `batch_size=2`, `grad_accum=8` for both `baseline` and `depth_memory`. Also patched `lm_data.py` to suppress the repeated `torch.load` cache FutureWarning for cleaner remote logs.
- Rationale: The shallow and mid-depth standard runs suggested the method may only separate from baseline once the model is deep enough, mirroring the earlier synthetic experiments.
- Key details: Results:
  `L=12` baseline: final `val_loss=14.1565`, `test_loss=14.4825`, `test_ppl=1,948,448.43`.
  `L=12` depth_memory: final `val_loss=14.0948`, `test_loss=14.5658`, `test_ppl=2,117,616.89`.
  Validation slightly favors `depth_memory`, while test slightly favors baseline; the gap is small enough that this should be treated as inconclusive rather than a real reversal.
- Side effects: The standard benchmark now has shallow, medium, and deeper comparison points under one consistent training recipe, and future logs should no longer be cluttered by cache warnings once the latest patch is synced.
- Verification: Both 12-layer remote CUDA runs completed successfully and saved metrics artifacts.
- Next step: Do one of two higher-value follow-ups instead of random architecture edits: either run longer `12`-layer training to reduce noise in the small gap, or add a second random seed for the `8`- and `12`-layer pairs to test stability.

### [Step 036] - 2026-03-30 21:14 CST - Extend standard benchmark training budget without changing the model
- Request: Stop changing the architecture and check whether there is still performance headroom under longer training on the standard benchmark.
- Plan: Keep the WikiText-2 setup fixed and only increase the training budget, first for the promising `8`-layer setting and then for the deeper `12`-layer setting.
- Files touched: `dev_log.md`
- Modification: Ran two longer controlled pairs on the Windows server:
  1. `L=8`, `d_model=256`, `seq_len=128`, `steps=300`, `batch_size=4`, `grad_accum=4`.
  2. `L=12`, same model width/length, `steps=300`, `batch_size=2`, `grad_accum=8`.
- Rationale: The short standard runs showed only small gaps; longer training is the cleanest way to see whether those differences persist, grow, or disappear without introducing new confounds.
- Key details: Results:
  `L=8` baseline: final `val_loss=12.6713`, `test_loss=10.5530`, `test_ppl=38,291.87`.
  `L=8` depth_memory: final `val_loss=12.6380`, `test_loss=10.4366`, `test_ppl=34,085.71`.
  `L=12` baseline: final `val_loss=10.2176`, `test_loss=10.4143`, `test_ppl=33,332.56`.
  `L=12` depth_memory: final `val_loss=10.2054`, `test_loss=10.4175`, `test_ppl=33,439.94`.
  This shows a persistent but modest win for `depth_memory` at `8` layers, while `12` layers are effectively tied under the current recipe.
- Side effects: The current evidence is now stronger than the short runs: the method does not collapse when trained longer on WikiText-2, and its benefit appears to be real but small at intermediate depth rather than universally increasing with depth.
- Verification: All four longer remote CUDA runs completed successfully and wrote new artifacts.
- Next step: The highest-value next experiment is no longer "change the architecture" but either add a second seed for the `8`-layer 300-step setting to test stability, or scale the model/context slightly to see whether the small 8-layer gain widens under a harder but still trainable regime.

### [Step 037] - 2026-03-30 21:22 CST - Add value-reprojection memory variant
- Request: Try the user's new idea of reusing earlier-layer same-position values as the surrogate input and applying the current layer's projection to them.
- Plan: Implement it as a separate attention type so the original `depth_memory` and baseline remain untouched, then run a controlled comparison.
- Files touched: `src/layer_depth_attention/model.py`, `train_assoc_recall.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `depth_memory_value_reproj`, which reconstructs each earlier layer's same-position `V` back to `d_model`, applies the current layer's key/value projection slices to those cached values, and uses the resulting reprojected memory bank in attention. Updated both training CLIs to accept the new attention type.
- Rationale: This variant tests a cheaper reuse path than storing hidden states and fully reprojecting them, while still giving the current layer a chance to transform memory entries before attention.
- Key details: This is intentionally a new mechanism, not mathematically identical to reprojecting hidden states: it uses `W_K^{(i)} v_k^{(\ell)}` and `W_V^{(i)} v_k^{(\ell)}` rather than `W_K^{(i)} x_k^{(\ell)}` and `W_V^{(i)} x_k^{(\ell)}`.
- Side effects: The experiment matrix now has a third attention variant; fairness against the baseline is preserved because the main baseline path is unchanged.
- Verification: Runtime verification pending.
- Next step: Run syntax checks and a small WikiText-2 comparison to see whether this variant is competitive with the existing `depth_memory`.

### [Step 038] - 2026-03-30 21:27 CST - Verify value-reprojection variant and compare at 8 layers
- Request: Actually try the new variant instead of only reasoning about it.
- Plan: Run local syntax/smoke checks, sync the code to the Windows server, and compare the new attention type on the existing `L=8` WikiText-2 setting used for earlier baseline vs `depth_memory` runs.
- Files touched: `dev_log.md`
- Modification: Verified the new attention type with `py_compile` and a one-step local associative-recall smoke test, then synced the updated files to the server and ran `WikiText-2`, `L=8`, `d_model=256`, `seq_len=128`, `steps=100`, `batch_size=4`, `grad_accum=4` with `attention_type=depth_memory_value_reproj`.
- Rationale: This is the cheapest way to test whether reusing past same-position `V` tensors and reprojecting them with the current layer is at least competitive before spending longer GPU budget.
- Key details: The new run reached `val_loss=17.9405`, `test_loss=15.1912`, `test_ppl=3,957,705.45`. Against the existing 100-step references, this is slightly better than both baseline (`test_loss=15.2846`) and the original `depth_memory` (`test_loss=15.2128`) on this setting.
- Side effects: The new variant looks promising enough at 100 steps to justify a longer run on the same 8-layer benchmark.
- Verification: Local smoke test and remote CUDA run both completed successfully and wrote artifacts.
- Next step: Run the same `L=8` setting for `300` steps to compare directly with the existing long-budget baseline and original `depth_memory` results.

### [Step 039] - 2026-03-30 21:31 CST - Compare value-reprojection variant under longer 8-layer training
- Request: Do not stop at the short run; check whether the new variant still looks good once the training budget is large enough for a fair comparison.
- Plan: Reuse the existing `L=8`, `d_model=256`, `seq_len=128`, `steps=300`, `batch_size=4`, `grad_accum=4` WikiText-2 benchmark so all three variants can be compared directly.
- Files touched: `dev_log.md`
- Modification: Ran `attention_type=depth_memory_value_reproj` for the same 300-step 8-layer WikiText-2 setting used earlier for baseline and the original `depth_memory`.
- Rationale: The short 100-step result was slightly better than both references, but that could still wash out once the baseline and original method are given enough optimization budget.
- Key details: Final result for the new variant: `val_loss=12.6227`, `test_loss=10.4457`, `test_ppl=34,395.20`. Direct comparison on this exact setting:
  baseline: `val_loss=12.6713`, `test_loss=10.5530`, `test_ppl=38,291.87`
  depth_memory: `val_loss=12.6380`, `test_loss=10.4366`, `test_ppl=34,085.71`
  depth_memory_value_reproj: `val_loss=12.6227`, `test_loss=10.4457`, `test_ppl=34,395.20`
  So the new variant stays better than baseline but ends up very slightly worse than the original `depth_memory` on test, while being slightly better on validation.
- Side effects: The project now has evidence that the value-reprojection idea is viable but not an obvious improvement over the simpler original memory mechanism under the current recipe.
- Verification: The longer remote CUDA run completed successfully and saved its artifact.
- Next step: Unless the user wants to keep optimizing this branch, the main model should remain the original `depth_memory`, with this value-reprojection path documented as an exploratory variant rather than the default direction.

### [Step 040] - 2026-03-30 21:42 CST - Push the standard benchmark to 16 layers
- Request: Check what happens on a deeper network rather than stopping at 8 or 12 layers.
- Plan: Keep the standard `WikiText-2` setup unchanged except for increasing depth to `16` layers, using `batch_size=2` and `grad_accum=8` to stay within the 4060 Ti memory budget, and compare baseline vs the original `depth_memory`.
- Files touched: `dev_log.md`
- Modification: Ran `L=16`, `d_model=256`, `seq_len=128`, `steps=300`, `batch_size=2`, `grad_accum=8` for both baseline and `depth_memory`.
- Rationale: Earlier runs suggested the gain was not monotonic with depth at smaller budgets; a deeper standard-model run with sufficient optimization budget is the right way to see whether the method eventually becomes more useful.
- Key details: Results:
  baseline: `val_loss=9.8397`, `test_loss=10.1461`, `test_ppl=25,491.99`
  depth_memory: `val_loss=10.0351`, `test_loss=10.0143`, `test_ppl=22,343.14`
  On this 16-layer setting, `depth_memory` is clearly better on test and substantially lowers test perplexity, even though its validation loss is slightly worse than baseline.
- Side effects: The standard benchmark picture is now more nuanced but stronger: the method shows no meaningful gain at 4 layers, a small stable gain at 8 layers, near-tie at 12 layers, and a clearer test improvement by 16 layers under the current recipe.
- Verification: Both 16-layer remote CUDA runs completed successfully and saved artifacts.
- Next step: The highest-value next step is to confirm whether the 16-layer test improvement is stable across another seed, because this is now the strongest evidence in favor of the method on the standard benchmark.

### [Step 041] - 2026-03-30 21:50 CST - Check 16-layer improvement with a second seed
- Request: Test whether the 16-layer improvement is actually stable rather than a lucky single run.
- Plan: Reuse the exact same 16-layer 300-step WikiText-2 configuration and change only `seed` from `42` to `123` for both baseline and `depth_memory`.
- Files touched: `dev_log.md`
- Modification: Ran the 16-layer baseline and `depth_memory` pair with `seed=123`.
- Rationale: The first 16-layer run was the strongest positive result so far, so the first stability check should target exactly that setting.
- Key details: Second-seed results:
  baseline: `val_loss=9.8152`, `test_loss=10.7162`, `test_ppl=45,078.64`
  depth_memory: `val_loss=10.0167`, `test_loss=10.7197`, `test_ppl=45,236.20`
  This seed does not reproduce the earlier test improvement; both models are nearly tied on test, with baseline slightly better by a very small margin.
- Side effects: The 16-layer result is now clearly mixed across seeds: one run favored `depth_memory` by a meaningful margin on test, while the next run was effectively a tie.
- Verification: Both second-seed remote CUDA runs completed successfully and saved artifacts.
- Next step: Treat the 16-layer story as promising but unstable. The right next step is to aggregate across seeds or slightly increase evaluation coverage before making any strong claim about deep-network gains.

### [Step 042] - 2026-03-30 21:59 CST - Evaluate the value-reprojection variant at 16 layers
- Request: Instead of discussing deep-network behavior abstractly, run the `value_reproj` variant on the same deeper setting.
- Plan: Reuse the established `L=16`, `d_model=256`, `seq_len=128`, `steps=300`, `batch_size=2`, `grad_accum=8`, `seed=42` WikiText-2 configuration so the new result can be compared directly with baseline and the original `depth_memory`.
- Files touched: `dev_log.md`
- Modification: Ran `attention_type=depth_memory_value_reproj` on the 16-layer standard benchmark.
- Rationale: The 8-layer experiments showed this variant was viable but not clearly better than the original method; the deeper setting is the natural place to check whether its extra transformation helps more.
- Key details: Final result:
  `depth_memory_value_reproj`: `val_loss=9.9283`, `test_loss=10.0671`, `test_ppl=23,554.96`
  Direct comparison on the same seed/config:
  baseline: `val_loss=9.8397`, `test_loss=10.1461`, `test_ppl=25,491.99`
  depth_memory: `val_loss=10.0351`, `test_loss=10.0143`, `test_ppl=22,343.14`
  depth_memory_value_reproj: `val_loss=9.9283`, `test_loss=10.0671`, `test_ppl=23,554.96`
  So at 16 layers the new variant sits between baseline and the original `depth_memory`: better than baseline on test, but not as strong as the original method.
- Side effects: The deep-network picture is now clearer: the value-reprojection idea remains competitive, but the simplest original `depth_memory` is still the best-performing variant on the strongest 16-layer seed-42 result.
- Verification: The remote CUDA run completed successfully and saved its artifact.
- Next step: If the user wants to keep exploring deep settings, the next rational move is multi-seed aggregation rather than adding more one-off variants.

### [Step 043] - 2026-03-30 22:07 CST - Prepare a larger benchmark scale-up
- Request: Move beyond WikiText-2 and smallish models to a larger dataset and a bigger/deeper model.
- Plan: Keep the existing training code and offline tokenizer flow, stage `WikiText-103` locally in the same plain-text format, then run a single larger-model baseline probe before committing to a full baseline-vs-depth-memory pair.
- Files touched: `dev_log.md`
- Modification: Created `external_data/wikitext-103-raw-v1`, downloaded the two `WikiText-103` train parquet shards plus validation/test parquet files, and converted them into plain `train.txt`, `validation.txt`, and `test.txt` files compatible with the existing loader. The resulting `train.txt` is about `540MB`.
- Rationale: A bigger dataset is the cleanest next scaling step because it avoids changing the model code while making the language-modeling problem more realistic and less toy-like.
- Key details: The next probe target is `WikiText-103` with a larger decoder-only GPT configuration around `d_model=384`, `num_layers=16`, and `seq_len=256`, which should still have a chance of fitting on the 16GB 4060 Ti with a small micro-batch and gradient accumulation.
- Side effects: Local disk usage increased substantially due to the larger staged text corpus.
- Verification: Local parquet-to-text conversion completed successfully.
- Next step: Sync the new dataset to the Windows server and run a single larger baseline probe to validate memory usage and training stability.

### [Step 044] - 2026-03-30 22:20 CST - Switch the first large-data probe to a practical WikiText-103 subset
- Request: Move to a larger dataset and bigger/deeper model, but avoid wasting time on an overly slow first full-corpus tokenization pass.
- Plan: Keep the full `WikiText-103` staging for later, but create a large training subset for the first scale-up probe so the model experiment can start immediately.
- Files touched: `dev_log.md`
- Modification: Observed that the first remote full-corpus `WikiText-103` run spent too long in slow-tokenizer preprocessing, then terminated that probe and created `external_data/wikitext-103-probe` locally with a `50,000,000`-character training file plus the full `validation.txt` and `test.txt` copies from `WikiText-103`.
- Rationale: A 50M-character probe is still far larger than `WikiText-2`, satisfies the user's request for a larger dataset in a practical sense, and lets the larger-model experiment produce results in this session.
- Key details: The new probe keeps the same vocabulary/tokenizer setup and evaluation splits while reducing only the training-corpus size for the first scale-up run.
- Side effects: The aborted remote full-corpus preprocessing means no stale Python training process is left consuming server memory.
- Verification: The probe dataset files were created locally; the previous remote Python process was killed successfully.
- Next step: Sync `wikitext-103-probe` to the server and run a `d_model=384`, `num_layers=16`, `seq_len=256` baseline probe.

### [Step 045] - 2026-03-30 22:26 CST - Run the first larger-data, larger-model comparison
- Request: Actually run a larger dataset and a bigger/deeper model rather than staying on the smaller WikiText-2 setup.
- Plan: Use the `wikitext-103-probe` dataset with cached token ids and compare baseline vs original `depth_memory` on a larger decoder-only GPT configuration.
- Files touched: `dev_log.md`
- Modification: Synced `wikitext-103-probe` to the server, let the first run build `train/validation/test_gpt2_ids.pt` caches, then ran two matching experiments with `d_model=384`, `num_layers=16`, `seq_len=256`, `steps=60`, `batch_size=1`, and `grad_accum=8`.
- Rationale: This is the first realistic scale-up beyond WikiText-2 and small-width models that still fits the 16GB 4060 Ti budget.
- Key details: Results on the larger-data probe:
  baseline: `val_loss=16.7007`, `test_loss=16.9155`, `test_ppl=22,197,135.76`
  depth_memory: `val_loss=16.7645`, `test_loss=17.2820`, `test_ppl=32,025,576.08`
  On this early 60-step larger-scale probe, `depth_memory` is worse than baseline.
- Side effects: The server now has cached token-id files for `wikitext-103-probe`, so subsequent larger-scale runs on this dataset should start directly in training rather than stalling in tokenization.
- Verification: Both remote CUDA runs completed successfully and saved artifacts.
- Next step: Do not over-interpret this single short probe; the main open question is whether `depth_memory` needs more training budget or a slightly different optimization regime at this larger scale before it becomes competitive.

### [Step 046] - 2026-03-30 22:33 CST - Check whether the larger-scale probe can use a bigger micro-batch
- Request: Explain why the larger-scale runs used `batch_size=1`, and verify whether that was actually necessary.
- Plan: Run one-step GPU smoke tests on the `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256` setting with larger micro-batches.
- Files touched: `dev_log.md`
- Modification: Ran three one-step smoke tests:
  1. baseline with `batch_size=2`
  2. baseline with `batch_size=4`
  3. `depth_memory_value_reproj` with `batch_size=2`
- Rationale: The original `batch_size=1` choice was a conservative first probe because both model size and sequence length had increased, but that assumption should be tested directly rather than treated as fixed.
- Key details: All three smoke tests completed successfully, which means the configuration is less memory-constrained than initially assumed. The immediate implication is that future larger-scale runs can likely use at least `batch_size=2`, and baseline can likely use `batch_size=4` on this setting.
- Side effects: The larger-scale experiment plan can now be made more efficient by reducing gradient accumulation and increasing micro-batch size.
- Verification: All three remote CUDA smoke runs completed successfully and wrote their artifacts.
- Next step: Re-run the larger-scale 500-step experiment with a larger micro-batch, using the user's preferred new method as the primary target.

### [Step 047] - 2026-03-30 22:37 CST - Verify the user's requested `batch_size=8`
- Request: Stop being conservative and test the exact `batch_size=8` request on the larger-scale setup.
- Plan: Run one-step GPU smoke tests at `batch_size=8` for baseline and `depth_memory_value_reproj` on the `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256` configuration.
- Files touched: `dev_log.md`
- Modification: Ran both one-step smoke tests and confirmed that baseline and `depth_memory_value_reproj` complete successfully at `batch_size=8`.
- Rationale: This directly answers the user's objection about underusing the 16GB GPU instead of relying on prior conservative assumptions.
- Key details: The requested `batch_size=8` is feasible for this larger-scale probe, so the next formal run can use it instead of `batch_size=1` or `2`.
- Side effects: Future larger-scale experiments on this configuration can use much higher micro-batch throughput.
- Verification: Both remote CUDA smoke runs completed successfully and wrote artifacts.
- Next step: Launch the user's preferred `depth_memory_value_reproj` larger-scale training run with `batch_size=8` and a longer budget.

### [Step 048] - 2026-03-30 22:45 CST - Complete the larger-scale three-way comparison at `batch_size=8`
- Request: Keep going and fill in the remaining formal comparisons once the user's requested `batch_size=8` run was confirmed.
- Plan: Use the same `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `steps=500`, `batch_size=8`, `grad_accum=1` configuration for all three methods.
- Files touched: `dev_log.md`
- Modification: Confirmed the existing 500-step artifacts for baseline and `depth_memory_value_reproj`, then ran the missing original `depth_memory` 500-step experiment to complete the three-way comparison.
- Rationale: This gives the first properly matched larger-data, larger-model comparison across baseline, the original method, and the newer value-reprojection variant.
- Key details: Final results on `wikitext-103-probe` after re-checking the saved artifacts:
  baseline: `test_loss=8.3654`, `test_ppl=4295.85`, final `val_loss=9.3564`
  depth_memory: `test_loss=8.3427`, `test_ppl=4199.27`, final `val_loss=9.3102`
  depth_memory_value_reproj: `test_loss=8.3336`, `test_ppl=4161.17`, final `val_loss=9.3044`
  So the ranking on this larger-scale setting is: `depth_memory_value_reproj` best, original `depth_memory` second, baseline third.
- Side effects: The larger-scale picture is stronger than the early 60-step probe suggested: with enough training budget and `batch_size=8`, both custom variants overtake baseline, and the newer value-reprojection variant is the best of the three.
- Verification: All three 500-step remote CUDA runs completed successfully and wrote artifacts.
- Next step: If the user wants to keep scaling, the most defensible move is to tune optimization for the custom methods at this larger scale rather than immediately scaling size again, because the raw architecture alone is no longer beating baseline here.

### [Step 049] - 2026-03-30 22:56 CST - Re-run the larger-scale winner with a second seed
- Request: Run it again to check whether the larger-scale positive result for the new method is stable.
- Plan: Reuse the same `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `steps=500`, `batch_size=8` setting and compare baseline vs `depth_memory_value_reproj` at `seed=123`.
- Files touched: `dev_log.md`
- Modification: Ran the second-seed baseline and `depth_memory_value_reproj` pair.
- Rationale: A larger-scale positive result is only meaningful if it survives at least one seed change.
- Key details: Second-seed results:
  baseline: `test_loss=8.2782`, `test_ppl=3936.94`, final `val_loss=9.5625`
  depth_memory_value_reproj: `test_loss=8.2462`, `test_ppl=3813.05`, final `val_loss=9.6698`
  The new method again beats baseline on test, though baseline keeps a slightly lower validation loss.
- Side effects: The evidence in favor of `depth_memory_value_reproj` at the larger scale is now materially stronger than a single lucky run.
- Verification: Both remote CUDA runs completed successfully and saved artifacts.
- Next step: Aggregate the two seeds and summarize the larger-scale result clearly before deciding whether to tune optimization further or scale up again.

### [Step 050] - 2026-03-30 23:07 CST - Add an Attention Residuals comparison variant
- Request: Compare against the recent `Attention Residuals` direction instead of discussing it only from papers.
- Plan: Implement an `attn_residuals` model mode that keeps standard self-attention/MLP sublayers but replaces fixed residual accumulation with learned depth-wise softmax aggregation over the embedding and prior sublayer outputs, then run smoke checks before any larger benchmark comparison.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Added `attn_residuals` as a new attention type choice, routing it to baseline self-attention inside blocks while changing `TinyDecoderLM.forward` to use learned pseudo-query vectors for the attention sublayer, MLP sublayer, and final representation aggregation. Implemented RMS-style normalization inside the residual mixer.
- Rationale: This gives a concrete in-repo comparator for the recent attention-residual-connection line, rather than relying only on paper claims.
- Key details: The implementation is intentionally aligned with the high-level mechanism (learned query over previous depth states) while staying compatible with the existing codebase and training scripts.
- Side effects: The model code now has a special forward path for `attn_residuals`; existing baseline and custom methods remain unchanged.
- Verification: Runtime verification pending.
- Next step: Run syntax/smoke checks, sync to the server, and compare `attn_residuals` against baseline on the current larger-scale benchmark.

### [Step 051] - 2026-03-30 23:15 CST - Compare the Attention Residuals variant on the larger-scale benchmark
- Request: Do not stop at implementation; actually compare the recent Attention Residuals idea against the current baseline and custom methods.
- Plan: Use the same large-scale benchmark already established for fair comparison: `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `steps=500`, `batch_size=8`.
- Files touched: `dev_log.md`
- Modification: Ran `attention_type=attn_residuals` on the larger-scale benchmark.
- Rationale: This provides a same-codebase comparison against a recent popular residual-connection alternative rather than relying on paper claims alone.
- Key details: `attn_residuals` finished with `val_loss=9.4013`, `test_loss=8.0438`, `test_ppl=3114.37`. This is better than all current in-repo competitors on the same setting:
  baseline: `test_ppl=4295.85` (seed 42) / `3936.94` (seed 123)
  depth_memory: `4199.27`
  depth_memory_value_reproj`: `4161.17` (seed 42) / `3813.05` (seed 123)
- Side effects: The current larger-scale benchmark now has a strong external-style comparator, and it sets a higher bar for what the custom method needs to beat.
- Verification: The remote CUDA run completed successfully and saved its artifact.
- Next step: Summarize the four-way comparison clearly and decide whether to keep optimizing the custom methods or pivot toward hybridizing with the attention-residual idea.

### [Step 052] - 2026-03-30 23:21 CST - Add the requested hybrid of Attention Residuals and value-reprojection memory
- Request: Combine the recent Attention Residuals mechanism with the user's strongest current custom attention branch.
- Plan: Keep the Attention Residuals depth-aggregation skeleton, but swap the block attention module from baseline self-attention to `depth_memory_value_reproj`, then compare it on the same larger-scale benchmark.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `train_assoc_recall.py`, `dev_log.md`
- Modification: Added `attn_residuals_value_reproj` as a new model mode. It reuses the Attention Residuals forward path and pseudo-query mixers, but its underlying block attention module is `depth_memory_value_reproj` rather than baseline self-attention. Updated both CLIs to accept the new mode.
- Rationale: This is the most direct way to test whether the strengths of the two best-performing ideas in the repo stack constructively.
- Key details: The hybrid changes only the attention sublayer used inside the residual-attention skeleton; the learned depth mixers remain identical to the existing `attn_residuals` implementation.
- Side effects: The experiment matrix now includes a fifth model family, but all prior behavior remains unchanged.
- Verification: Runtime verification pending.
- Next step: Run local smoke checks, sync to the server, and launch the same larger-scale 500-step benchmark for the hybrid.

### [Step 053] - 2026-03-30 23:31 CST - Evaluate the hybrid Attention Residuals + value-reprojection model
- Request: Do not stop at implementation; test the combined method on the same benchmark as the other contenders.
- Plan: Reuse the exact same `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `steps=500`, `batch_size=8` configuration and compare the hybrid against the existing four methods.
- Files touched: `dev_log.md`
- Modification: Ran `attention_type=attn_residuals_value_reproj` on the larger-scale benchmark.
- Rationale: This directly answers whether the two strongest ideas in the repo combine constructively.
- Key details: The hybrid finished with `val_loss=9.3337`, `test_loss=8.1052`, `test_ppl=3311.73`. Relative ranking on the same setting is now:
  `attn_residuals`: `3114.37`
  `attn_residuals_value_reproj`: `3311.73`
  `depth_memory_value_reproj`: `3813.05` (best seed-123 run) / `4161.17` (seed-42 run)
  `depth_memory`: `4199.27`
  `baseline`: `3936.94` (seed-123) / `4295.85` (seed-42)
  So the hybrid improves substantially over the standalone custom methods and baseline, but still trails the pure Attention Residuals comparator.
- Side effects: The hybrid result suggests the two ideas are at least partially compatible, but the extra memory-attention complexity does not yet improve over the simpler residual-attention skeleton.
- Verification: The remote CUDA run completed successfully and saved its artifact.
- Next step: Decide whether to optimize the hybrid further or treat Attention Residuals as the strongest current external comparator and the hybrid as the best merged variant so far.

### [Step 054] - 2026-03-30 23:18 CST - Add Attention Residuals plus MoE variant
- Request: Implement and test an `Attention Residuals + MoE` variant.
- Plan: Add a minimal top-1 MoE feed-forward module, wire it into the residual-attention skeleton as a new `attn_residuals_moe` model type, then run smoke tests before a full benchmark.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `Top1MoE`, extended `TransformerBlock` with selectable FFN type, introduced `attn_residuals_moe`, and exposed `--num-experts` plus the new attention-type choice in the WikiText training script.
- Rationale: The user asked for a direct `Attention Residuals + MoE` comparison under the same benchmark setup used for the other methods.
- Key details: This is a minimal top-1 routing implementation with dense per-expert MLPs and no auxiliary load-balancing loss; it keeps the residual-attention skeleton intact and swaps only the FFN path to MoE.
- Side effects: Existing method types remain unchanged; the new route adds parameters and compute in the FFN branch only.
- Verification: Pending compile and smoke tests.
- Next step: Run local verification, then launch the larger-scale `wikitext-103-probe` benchmark.

### [Step 055] - 2026-03-30 23:20 CST - Verify MoE residual-attention path locally
- Request: Confirm the new `attn_residuals_moe` path runs before launching a larger remote experiment.
- Plan: Use compile checks plus a dependency-light pure-`torch` forward/backward smoke test, since the local shell environment does not include `transformers` for the full WikiText script.
- Files touched: `dev_log.md`
- Modification: Verified the new model type can instantiate, run a forward pass, and backpropagate on random token IDs through the residual-attention plus MoE path.
- Rationale: A small local smoke catches shape and autograd issues before spending remote GPU time.
- Key details: The full local `train_wikitext_lm.py` smoke could not run because local Python lacks `transformers`; this does not affect the remote training environment already used for prior WikiText runs.
- Side effects: none known.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; pure-`torch` forward/backward smoke with `TinyDecoderLM(attention_type='attn_residuals_moe')`.
- Next step: Launch the server benchmark on `wikitext-103-probe` with the same large-scale setting used for the other methods.

### [Step 056] - 2026-03-30 23:25 CST - Run large-scale Attention Residuals plus MoE benchmark
- Request: Evaluate `Attention Residuals + MoE` under the same larger-scale setting used for the strongest prior methods.
- Plan: Sync the new code to the server, run a 1-step remote smoke test, then execute the full `wikitext-103-probe` benchmark with `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, and `500` training steps.
- Files touched: `dev_log.md`
- Modification: Synced `model.py` and `train_wikitext_lm.py` to the Windows server, confirmed the new path runs remotely, and completed the full benchmark as `artifacts/wikitext103probe_attn_residuals_moe_bs8_500.json`.
- Rationale: The user asked for a direct `Attention Residuals + MoE` comparison against the existing baseline, residual-attention, and custom-memory variants.
- Key details: Final metrics were `val_loss=8.6480`, `test_loss=9.2584`, `test_ppl=10492.08`. This is substantially worse than pure `Attention Residuals` (`test_ppl=3114.37`), the hybrid `attn_residuals_value_reproj` (`3311.73`), and the standalone custom methods (`3813.05` to `4199.27` in the comparable runs). The minimal top-1 MoE variant is trainable but clearly underperforms in this setting.
- Side effects: Establishes that a naive MoE swap in the FFN path is not automatically beneficial here; better routing or auxiliary balancing would be needed before treating MoE as a serious improvement path.
- Verification: Remote 1-step smoke on `wikitext-103-probe`; remote 500-step CUDA benchmark completed and saved metrics.
- Next step: Keep `attn_residuals_moe` as an exploratory branch only; if MoE is revisited, add load-balancing or capacity control rather than using this naive top-1 version as a main result.

### [Step 057] - 2026-03-30 23:53 CST - Align the value-reprojection write-up with the runnable implementation
- Request: Rewrite the `value_reproj` method definition so it is mathematically honest while keeping the current code path unchanged and runnable.
- Plan: Leave the implementation intact, then update the method note to describe `value_reproj` as a memory-adapter variant built from historical values rather than as a strict current-layer reprojection of old hidden states.
- Files touched: `method_layer_depth_memory_attention.md`, `dev_log.md`
- Modification: Added an implementation-specific appendix section to the method note defining `value_reproj` via adapted historical value features `z_k^{(\ell)}` and adapter projections `A_K^{(i)}, A_V^{(i)}`.
- Rationale: The previous informal interpretation could be misread as `W_V^{(i)} x_k^{(\ell)}`, which does not match the actual runnable code that reuses prior value features.
- Key details: The updated write-up deliberately avoids changing the proven training path; it reframes the method as a memory adapter / reprojection mechanism that is consistent with the current implementation and experimental results.
- Side effects: The main method note now separates the original conceptual formulation from the runnable `value_reproj` variant more cleanly.
- Verification: Reviewed the patched Markdown and ensured no code changes were made.
- Next step: If the paper centers on `value_reproj`, propagate this adapter-style definition into the final Method and Related Work sections rather than using the older hidden-state reprojection language.

### [Step 058] - 2026-03-31 00:02 CST - Add fixed half-half block-normalized value-reprojection variant
- Request: Try the user's proposed split normalization approximation where token attention and memory attention are normalized separately and then combined with a fixed `1/2` and `1/2` weighting.
- Plan: Implement this as a separate attention type on top of `value_reproj`, preserving the current main methods, then verify locally before launching the standard larger-scale benchmark.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `LayerDepthValueReprojHalfMixAttention` and exposed it as `attention_type=depth_memory_value_reproj_halfmix` in the WikiText training script.
- Rationale: The user wants a quick empirical check of fixed blockwise normalization without introducing a learned gate or disturbing existing results.
- Key details: The variant computes causal token attention and memory attention separately, applies softmax within each block, and combines the resulting contexts as `0.5 * token_context + 0.5 * memory_context` when memory is present.
- Side effects: Existing `depth_memory_value_reproj` behavior remains unchanged; this is a clean ablation branch.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke with `TinyDecoderLM(attention_type='depth_memory_value_reproj_halfmix')`.
- Next step: Sync to the server and run the same `wikitext-103-probe` 16-layer, batch-size-8, 500-step benchmark used for the other methods.

### [Step 059] - 2026-03-31 00:05 CST - Evaluate the fixed half-half block-normalized value-reprojection variant
- Request: Test the user's proposed `1/2` token-block plus `1/2` memory-block weighting as a quick approximation to split normalization.
- Plan: Run the same larger-scale benchmark used for the current main comparison: `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, `steps=500`.
- Files touched: `dev_log.md`
- Modification: Synced the new `depth_memory_value_reproj_halfmix` variant to the server and completed the full benchmark as `artifacts/wikitext103probe_value_reproj_halfmix_bs8_500.json`.
- Rationale: This directly checks whether forcing equal block mass between token attention and memory attention helps the current best custom method.
- Key details: Final metrics were `val_loss=8.7657`, `test_loss=9.5652`, `test_ppl=14260.13`. This is dramatically worse than the original `depth_memory_value_reproj` runs (`test_ppl=4161.17` for seed 42 and `3813.05` for seed 123), and also worse than baseline. The fixed half-half allocation is therefore too rigid in this setting.
- Side effects: Strong evidence that token and memory blocks should not be forced to share equal mass; if split normalization is revisited, it should use a learned or data-dependent gate rather than a fixed `1/2`.
- Verification: Remote 500-step CUDA benchmark completed and saved metrics.
- Next step: Discard the fixed half-half variant as a main candidate and, if desired, revisit the idea only with an adaptive block gate.

### [Step 060] - 2026-03-31 00:06 CST - Revert the failed fixed half-half split-normalization branch
- Request: Restore the codebase to the original stable variants after the fixed `1/2` split-normalization test failed badly.
- Plan: Remove the temporary `depth_memory_value_reproj_halfmix` class and CLI exposure, then run a compile check to confirm the project is back to the previous runnable state.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Deleted `LayerDepthValueReprojHalfMixAttention` and removed the `depth_memory_value_reproj_halfmix` option from the WikiText training script.
- Rationale: The branch produced `test_ppl=14260.13`, far worse than the original `value_reproj`, so keeping it in the main code only adds noise.
- Key details: The stable set of active variants is back to `baseline`, `depth_memory`, `depth_memory_value_reproj`, `attn_residuals`, `attn_residuals_value_reproj`, and `attn_residuals_moe`.
- Side effects: The saved artifact from the failed experiment remains for reference, but the runnable method roster is simplified again.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`.
- Next step: Continue from the original `value_reproj` line or test adaptive gating rather than fixed block weighting if this direction is revisited.

### [Step 061] - 2026-03-31 00:07 CST - Launch 2000-step large-scale value-reprojection run
- Request: Run a much longer training schedule to see how the original `value_reproj` method behaves with substantially more optimization.
- Plan: Keep the established larger-scale configuration fixed and extend training from `500` to `2000` steps for `depth_memory_value_reproj`.
- Files touched: `dev_log.md`
- Modification: Synced the current stable code to the Windows server and prepared the long-run command for `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`.
- Rationale: The current main question is whether the method's advantage persists, grows, or disappears under a much longer training budget.
- Key details: This uses the original runnable `value_reproj` implementation, not the reverted halfmix branch, with `steps=2000` and a coarser `eval_interval=200` to keep logging manageable.
- Side effects: This run is materially longer than the earlier 500-step experiments and may take several minutes on the remote 4060 Ti.
- Verification: Pending completion of the remote CUDA run.
- Next step: Monitor the 2000-step server job and compare its final metrics against the earlier 500-step `value_reproj`, baseline, and Attention Residuals runs.

### [Step 062] - 2026-03-31 00:15 CST - Complete the 2000-step large-scale value-reprojection run
- Request: See how the original `value_reproj` method behaves with a much longer training budget.
- Plan: Keep the established large-scale setup fixed and run `depth_memory_value_reproj` for `2000` steps on `wikitext-103-probe`.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and saved the artifact as `artifacts/wikitext103probe_value_reproj_bs8_2000.json`.
- Rationale: The earlier 500-step results showed a small but repeatable gain over baseline; this longer run tests whether the method continues improving or plateaus.
- Key details: Final metrics were `val_loss=3.9107`, `val_ppl=49.93`, `test_loss=4.1997`, `test_ppl=66.67`. Intermediate validation milestones improved steadily from `val_ppl=59929.17` at step 200 to `1292.21` at step 400, `261.39` at step 600, and `84.25` at step 1000, showing strong continued optimization under the longer schedule.
- Side effects: This materially changes the scale of the current best `value_reproj` result; however, there is not yet a matching 2000-step baseline run, so the long-budget relative advantage over standard self-attention is still unknown.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Run the same 2000-step configuration for the baseline to determine whether `value_reproj` remains better after both methods are given equal long-budget optimization.

### [Step 063] - 2026-03-31 00:17 CST - Add a normalized-history value-reprojection variant
- Request: Test the hypothesis that historical values should be normalized before being reprojected by the current layer.
- Plan: Add a separate `depth_memory_value_reproj_normed` branch that applies parameter-free normalization to the concatenated historical value features before the current-layer `K/V` projection, then verify locally before re-running the large-scale benchmark.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `LayerDepthValueReprojNormedAttention` and exposed it as `attention_type=depth_memory_value_reproj_normed`.
- Rationale: The original `value_reproj` feeds raw historical values into the current-layer projection; normalizing them first may reduce scale mismatch across layers.
- Key details: The current implementation uses parameter-free `layer_norm` over the reconstructed `d_model`-dimensional historical value features immediately before `_kv_proj`.
- Side effects: Existing stable variants remain unchanged; this is an isolated ablation branch for the normalization hypothesis.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke with `TinyDecoderLM(attention_type='depth_memory_value_reproj_normed')`.
- Next step: Sync the new variant to the server and rerun the same 2000-step `wikitext-103-probe` benchmark for a direct comparison against the unnormalized `value_reproj`.

### [Step 064] - 2026-03-31 00:25 CST - Complete the 2000-step normalized-history value-reprojection run
- Request: Check whether normalizing historical values before current-layer reprojection improves the long-budget `value_reproj` result.
- Plan: Run the normalized-history variant under the exact same 2000-step large-scale configuration as the original `value_reproj`.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run for `depth_memory_value_reproj_normed` and saved the artifact as `artifacts/wikitext103probe_value_reproj_normed_bs8_2000.json`.
- Rationale: This directly tests the user's hypothesis that scale mismatch in raw historical values may be hurting the reprojection branch.
- Key details: Final metrics were `val_loss=3.9219`, `val_ppl=50.49`, `test_loss=4.1936`, `test_ppl=66.26`. Compared with the original 2000-step `value_reproj` run (`val_loss=3.9107`, `test_loss=4.1997`, `test_ppl=66.67`), the normalized variant is slightly better on test and slightly worse on validation. The gain is real but very small.
- Side effects: This suggests that normalization of historical values is not a bad idea, but it is not the main bottleneck in the current method.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Keep the normalized variant as a plausible refinement, but do not treat it as a major breakthrough unless the same direction holds across more seeds or against a 2000-step baseline.

### [Step 065] - 2026-03-31 00:28 CST - Launch the matching 2000-step baseline run
- Request: Run the standard baseline for the same long schedule to compare directly against the two 2000-step `value_reproj` variants.
- Plan: Use the exact same large-scale configuration and training budget as the recent custom-method runs, changing only `attention_type` back to `baseline`.
- Files touched: `dev_log.md`
- Modification: Prepared the direct comparison run for `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, `steps=2000`, `attention_type=baseline`.
- Rationale: Without a matching baseline under the same long-budget schedule, the current long-run custom-method results cannot be interpreted fairly.
- Key details: The target comparison set is now baseline vs `depth_memory_value_reproj` vs `depth_memory_value_reproj_normed`, all at 2000 steps.
- Side effects: This is another long remote run and will take several minutes on the 4060 Ti.
- Verification: Pending completion of the remote CUDA run.
- Next step: Monitor the baseline job and compare its final validation/test metrics against both custom variants.

### [Step 066] - 2026-03-31 00:36 CST - Complete the matching 2000-step baseline run
- Request: Compare the long-budget custom methods fairly against a same-budget standard decoder-only baseline.
- Plan: Run the exact same 2000-step large-scale schedule with `attention_type=baseline` and then compare all three results side by side.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA baseline run and saved the artifact as `artifacts/wikitext103probe_baseline_bs8_2000.json`.
- Rationale: This closes the fairness gap in the long-budget comparison.
- Key details: Final metrics were `val_loss=3.9203`, `val_ppl=50.41`, `test_loss=4.2361`, `test_ppl=69.14`. Relative to this baseline, the original `value_reproj` is better on both validation and test (`3.9107 / 66.67`), and the normalized variant is also better on test (`4.1936 / 66.26`) though slightly worse on validation (`3.9219`). All three methods are very close after 2000 steps, but both custom variants still edge out the baseline on test.
- Side effects: The long-budget picture is now much clearer: the custom methods retain a small advantage even after heavy optimization, but the gap is modest rather than dramatic.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Use the three-way 2000-step table as the current strongest fairness check, then decide whether to add more seeds or stop and summarize the method's true effect size.

### [Step 067] - 2026-03-31 00:37 CST - Launch the matching 2000-step Attention Residuals run
- Request: Run the current strongest external-style comparator, `Attention Residuals`, under the same 2000-step large-scale schedule.
- Plan: Reuse the exact same `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, `steps=2000` configuration and switch only `attention_type` to `attn_residuals`.
- Files touched: `dev_log.md`
- Modification: Prepared the direct long-budget comparator run for `attn_residuals`.
- Rationale: This allows a fair long-budget comparison among baseline, `value_reproj`, `value_reproj_normed`, and `Attention Residuals`.
- Key details: Prior 500-step results showed `Attention Residuals` as the strongest method in-repo; this run checks whether that advantage persists after much longer optimization.
- Side effects: This is another multi-minute remote run on the 4060 Ti.
- Verification: Pending completion of the remote CUDA run.
- Next step: Compare the final 2000-step `attn_residuals` metrics against the three existing long-budget results.

### [Step 068] - 2026-03-31 00:51 CST - Complete the 2000-step Attention Residuals run
- Request: Compare the strongest residual-attention comparator fairly against the long-budget baseline and custom memory variants.
- Plan: Finish the 2000-step `attn_residuals` run and place it into the same comparison table as the other three 2000-step methods.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and saved the artifact as `artifacts/wikitext103probe_attn_residuals_bs8_2000.json`.
- Rationale: This completes the fair long-budget comparison set.
- Key details: Final metrics were `val_loss=3.8722`, `val_ppl=48.05`, `test_loss=4.1046`, `test_ppl=60.62`. This remains better than baseline (`69.14`), original `value_reproj` (`66.67`), and `value_reproj_normed` (`66.26`) after 2000 steps, so the earlier 500-step ranking survives under longer optimization.
- Side effects: The long-budget result strengthens the interpretation that the user's method is effective but currently trails the stronger `Attention Residuals` comparator.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Summarize the four-way 2000-step table and use it for realistic paper-positioning rather than relying only on shorter 500-step evidence.

### [Step 069] - 2026-03-31 00:55 CST - Add a normalized-history Attention Residuals hybrid
- Request: Run `Attention Residuals + value_reproj_normed` for 2000 steps with progress printed every 300 steps.
- Plan: Add a dedicated `attn_residuals_value_reproj_normed` model type that reuses the residual-attention skeleton with the normalized-history reprojection attention module, then verify locally before launching the server run.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `attn_residuals_value_reproj_normed` to the model and CLI, mapping the residual-attention skeleton onto `depth_memory_value_reproj_normed` blocks.
- Rationale: This isolates whether combining the stronger residual-attention routing with normalized historical-value reprojection yields further gains.
- Key details: The requested long run will use `eval_interval=300`, so the training script will print progress at 300-step intervals plus the final step 2000.
- Side effects: Existing methods remain unchanged; this adds one more comparable hybrid branch.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke with `TinyDecoderLM(attention_type='attn_residuals_value_reproj_normed')`.
- Next step: Sync the updated files to the server and launch the 2000-step `wikitext-103-probe` run.

### [Step 070] - 2026-03-31 01:11 CST - Complete the 2000-step Attention Residuals plus normalized-history reprojection run
- Request: Run `attn_residuals_value_reproj_normed` for 2000 steps with progress logged every 300 steps.
- Plan: Use the same large-scale benchmark as the other 2000-step runs and set `eval_interval=300` so the output exposes intermediate checkpoints.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and saved the artifact as `artifacts/wikitext103probe_attn_residuals_value_reproj_normed_bs8_2000.json`.
- Rationale: This measures whether combining the residual-attention routing with normalized historical-value reprojection beats pure `Attention Residuals` or the standalone normalized reprojection variant.
- Key details: Logged checkpoints were: step 300 `val_loss=8.7599`, step 600 `5.5315`, step 900 `4.5613`, step 1200 `4.1510`, step 1500 `3.9952`, step 1800 `3.8967`, step 2000 `3.8653`. Final metrics were `val_ppl=47.72`, `test_loss=4.1274`, `test_ppl=62.01`. This is better than `value_reproj_normed` (`66.26`) and close to but still worse than pure `attn_residuals` (`60.62`).
- Side effects: The normalized hybrid becomes the strongest custom-memory-based residual-attention merge so far, but it still does not beat the pure residual-attention comparator.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Use the five-way 2000-step comparison to decide whether the paper should position the memory method as a standalone contribution, a hybrid component, or a secondary variant behind residual-attention baselines.

### [Step 071] - 2026-03-31 01:14 CST - Launch the matching 2000-step direct-memory run
- Request: Run the earlier variant that uses historical layer `V` directly without the current-layer reprojection step.
- Plan: Use the same large-scale 2000-step benchmark and switch `attention_type` to `depth_memory`, which directly reuses historical `K/V` instead of adapting past values through the current-layer projection.
- Files touched: `dev_log.md`
- Modification: Prepared the direct-memory long-run command for `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, `steps=2000`.
- Rationale: This gives a clean apples-to-apples comparison between direct historical-memory reuse and the reprojection-based variants under the same training budget.
- Key details: No code changes are needed because `depth_memory` is already implemented and previously evaluated only under shorter schedules.
- Side effects: This adds another multi-minute 2000-step remote run to the comparison grid.
- Verification: Pending completion of the remote CUDA run.
- Next step: Compare the final direct-memory result against baseline, `value_reproj`, `value_reproj_normed`, and the residual-attention family.

### [Step 072] - 2026-03-31 01:26 CST - Complete the 2000-step direct-memory run
- Request: Evaluate the original direct historical-memory variant that reuses earlier-layer `K/V` without current-layer reprojection.
- Plan: Run `depth_memory` under the same 2000-step large-scale schedule and compare it against baseline and the reprojection-based methods.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and saved the artifact as `artifacts/wikitext103probe_depth_memory_bs8_2000.json`.
- Rationale: This isolates whether the user's later reprojection changes actually help compared with direct reuse of historical memory projections.
- Key details: Logged checkpoints were: step 300 `val_loss=8.6566`, step 600 `5.5777`, step 900 `4.6022`, step 1200 `4.2303`, step 1500 `4.0434`, step 1800 `3.9587`, step 2000 `3.9228`. Final metrics were `test_loss=4.2325`, `test_ppl=68.89`. This is slightly better than baseline (`69.14`) but worse than `value_reproj` (`66.67`) and `value_reproj_normed` (`66.26`).
- Side effects: Strengthens the case that the reprojection idea is doing useful work beyond simple direct historical-memory reuse.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Use the expanded comparison table to quantify the added value of direct memory reuse, reprojection, normalization, and residual-attention hybrids.

### [Step 073] - 2026-03-31 01:37 CST - Prepare the repository for push-and-pull synchronization
- Request: Switch from ad-hoc file copying to a cleaner Git-based sync workflow between local and the Windows server.
- Plan: Stop tracking datasets and artifacts, keep only code/docs/logs in the repository, and verify the latest experimental branches compile before committing.
- Files touched: `.gitignore`, `dev_log.md`, `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`
- Modification: Updated `.gitignore` to ignore `artifacts/` and `external_data/`, removed those paths from the Git index without deleting local files, and confirmed that the current attention variants including the new `depth_memory_qkv_reproj` branch run a local forward/backward smoke.
- Rationale: Pushing datasets and experiment outputs would make the repo noisy and brittle; the user wants a push/pull workflow focused on source files.
- Key details: A temporary index-lock issue appeared because multiple Git commands touched the same repo in parallel; this was resolved by switching Git index operations back to serialized commands.
- Side effects: Future syncs can be done by Git once the server directory is attached to the remote repository.
- Verification: Local pure-`torch` smoke verified `baseline`, `depth_memory`, `depth_memory_value_reproj`, `depth_memory_value_reproj_normed`, and `depth_memory_qkv_reproj`.
- Next step: Commit the cleaned repository state, push it to GitHub, then connect the server directory to the same remote so later updates only need `git pull`.

### [Step 074] - 2026-03-31 01:38 CST - Switch the Windows server to Git-based code synchronization
- Request: Replace ad-hoc SCP syncing with a cleaner push/pull workflow.
- Plan: Commit and push the cleaned local repository, then attach the existing server directory to the same Git remote and remove stale hand-copied duplicates.
- Files touched: `dev_log.md`
- Modification: Pushed commit `89686e8` to `origin/main`, initialized Git inside `D:\Projects\Layer-Depth-Attention` on the server, fetched and hard-reset it to `origin/main`, and removed stale untracked files (`lm_data.py`, `model.py`, `src\\layer_depth_attention\\dev_log.md`) that came from older manual copies.
- Rationale: This makes future server syncs reproducible and much less error-prone: local changes can now be pushed once and applied remotely with `git pull`.
- Key details: The server directory was not previously a Git repository; after initialization it now tracks `origin/main`. Local repository status is clean after the push.
- Side effects: Future experiments should be launched from the Git-tracked server tree, not by copying source files manually.
- Verification: Local `git push origin main` succeeded; server `git reset --hard origin/main` succeeded; server `git status --short --branch` is clean.
- Next step: Use `git pull` on the server for subsequent code updates, then continue experiments from the synchronized tree.

### [Step 075] - 2026-03-31 01:40 CST - Launch the QKV-reprojection benchmark via Git sync
- Request: Start the planned experiment for the variant that uses same-position historical `Q/K/V` together, then adapts them into current-layer memory entries.
- Plan: Use the new Git-based workflow to sync the current code to the server, then run `depth_memory_qkv_reproj` under the same large-scale 2000-step benchmark used for the other main comparisons.
- Files touched: `dev_log.md`
- Modification: Logged the experiment launch plan for the `depth_memory_qkv_reproj` branch.
- Rationale: This tests whether using richer same-position cross-layer features (`Q`, `K`, and `V` jointly) improves over the existing value-only reprojection variants.
- Key details: The target configuration is `wikitext-103-probe`, `d_model=384`, `num_layers=16`, `seq_len=256`, `batch_size=8`, `steps=2000`, with progress printed every 300 steps.
- Side effects: This run adds a new main comparator to the long-budget comparison table.
- Verification: Local compile check passed before launch.
- Next step: Pull the latest code on the server and run the 2000-step `depth_memory_qkv_reproj` benchmark.

### [Step 076] - 2026-03-31 01:45 CST - Complete the 2000-step QKV-reprojection run
- Request: Evaluate the new variant that uses same-position historical `Q/K/V` jointly before current-layer memory adaptation.
- Plan: Run `depth_memory_qkv_reproj` on the same 2000-step large-scale benchmark and compare it against the existing direct-memory and value-only reprojection variants.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and saved the artifact as `artifacts/wikitext103probe_depth_memory_qkv_reproj_bs8_2000.json`.
- Rationale: This tests whether richer same-position cross-layer features help beyond the simpler value-only reprojection.
- Key details: Logged checkpoints were: step 300 `val_loss=8.8302`, step 600 `5.6025`, step 900 `4.6562`, step 1200 `4.2777`, step 1500 `4.1052`, step 1800 `4.0198`, step 2000 `3.9940`. Final metrics were `test_loss=4.2220`, `test_ppl=68.17`. This is slightly better than direct `depth_memory` (`68.89`) and baseline (`69.14`), but worse than `value_reproj` (`66.67`) and `value_reproj_normed` (`66.26`).
- Side effects: The result suggests that adding historical `Q` and `K` features does not outperform the simpler value-focused reprojection path in the current setup.
- Verification: Remote 2000-step CUDA benchmark completed and saved metrics.
- Next step: Keep `qkv_reproj` as an explored but non-leading branch; use the accumulated comparison table to sharpen the mechanism-level conclusions.

### [Step 077] - 2026-03-31 01:58 CST - Correct the QKV-reprojection semantics
- Request: Adjust `depth_memory_qkv_reproj` to match the intended method definition rather than the earlier concatenated-feature adapter interpretation.
- Plan: Treat same-position historical `Q`, `K`, and `V` as separate memory slots, project each of them with the current layer's existing `K/V` projection path, and use the resulting `3(i-1)` slots in the attention bank.
- Files touched: `src/layer_depth_attention/model.py`, `dev_log.md`
- Modification: Rewrote `LayerDepthQKVReprojAttention` so it no longer concatenates historical `Q/K/V` into one wide feature for dedicated adapters. It now reconstructs each historical `Q`, `K`, and `V` slot in `d_model`, concatenates them along the memory-slot axis, and runs the combined slots through the current layer's `_kv_proj`.
- Rationale: This matches the user's intended mechanism: use current-layer projection for all same-position historical `Q/K/V` items rather than introducing separate memory projection matrices.
- Key details: The resulting memory-bank size is now `n + 3(i-1)` for each query position, and the current query still comes from the standard current-layer projection.
- Side effects: Earlier experimental results for the old `depth_memory_qkv_reproj` semantics should no longer be treated as valid for this name.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke on `depth_memory_qkv_reproj`.
- Next step: Push the corrected semantics via Git, pull on the server, and rerun the 2000-step benchmark before drawing any conclusion about this branch.

### [Step 078] - 2026-03-31 02:01 CST - Add normalization before current-layer QKV reprojection
- Request: Update the corrected `depth_memory_qkv_reproj` semantics so the reconstructed historical `Q/K/V` slots are normalized before applying the current layer's projection.
- Plan: Insert a parameter-free normalization right before `_kv_proj`, verify locally, and rerun the benchmark instead of using the now-obsolete unnormalized attempt.
- Files touched: `src/layer_depth_attention/model.py`, `dev_log.md`
- Modification: Added `layer_norm` over the reconstructed `d_model`-dimensional historical `Q/K/V` slot inputs immediately before the current-layer `_kv_proj` inside `LayerDepthQKVReprojAttention`.
- Rationale: The user expects the historical attention-space vectors to be normalized before current-layer reprojection, analogous to the earlier successful `value_reproj_normed` refinement.
- Key details: This changes the intended 2000-step QKV-reprojection experiment again; any partially run unnormalized corrected version should be disregarded in favor of the new normalized variant.
- Side effects: The final intended semantics for `depth_memory_qkv_reproj` are now: separate historical `Q/K/V` slots, current-layer shared `_kv_proj`, and pre-projection normalization.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke on `depth_memory_qkv_reproj`.
- Next step: Push this normalization update, pull on the server, and run the benchmark to completion.

### [Step 079] - 2026-03-31 02:06 CST - Diagnose the server's outbound GitHub connectivity issue
- Request: Resolve the recurring problem where the Windows server appears unable to reach the internet.
- Plan: Separate basic internet access from GitHub-specific failures by checking IP config, routing, DNS, ICMP reachability, and HTTPS connectivity to multiple domains.
- Files touched: `dev_log.md`
- Modification: Verified that the server has healthy general internet access but intermittent or blocked HTTPS connectivity specifically to `github.com`.
- Rationale: Recent `git pull` failures were blocking the new push/pull sync workflow and needed a concrete root-cause classification.
- Key details: The server has a valid public IPv4 route (`218.194.25.254` with lower metric than ZeroTier), DNS resolves `github.com` correctly, `ping 1.1.1.1` and `ping github.com` both succeed, and HTTPS to `www.baidu.com` and `www.microsoft.com` returns `200`. However, both `curl -4 https://github.com` and GitHub HTTPS fetches time out on port `443`. This indicates a GitHub-specific HTTPS path issue rather than total outbound network failure.
- Side effects: Future server syncs cannot reliably depend on direct GitHub access until a proxy, mirror, or alternate route is configured.
- Verification: Remote `ipconfig`, `route print`, `nslookup github.com`, `ping`, `Invoke-WebRequest`, and `curl -4` checks.
- Next step: Use a fallback sync path for experiments now, and if needed later set up a stable Git proxy/mirror or a local bare-remote hop over ZeroTier.

### [Step 080] - 2026-03-31 02:13 CST - Establish a ZeroTier-based local Git source for the server
- Request: Change the sync workflow so the server can pull code from the local machine over ZeroTier instead of relying on unstable GitHub HTTPS access.
- Plan: Expose a local bare repository over the ZeroTier IP, verify the server can reach it, add it as a remote on the server, and fast-forward the server workspace to that source.
- Files touched: `dev_log.md`
- Modification: Created a local bare repo mirror at `/Users/a/Projects/Layer-Depth-Attention-zt.git`, exported it via `git daemon`, verified reachability from the server at `git://10.147.20.35:9418/Layer-Depth-Attention-zt.git`, added `zerotier-local` as a server remote, and reset the server workspace to `zerotier-local/main` at commit `6aee88a`.
- Rationale: The server has general internet access but unreliable GitHub 443 connectivity; a ZeroTier local source avoids that bottleneck.
- Key details: The local machine's ZeroTier IPv4 is `10.147.20.35`; server connectivity to `10.147.20.35:9418` now succeeds. The server's `origin` remote can remain for reference, but active code sync can use `zerotier-local`.
- Side effects: The local `git daemon` must be running for the server to fetch from the ZeroTier source.
- Verification: Server `git ls-remote git://10.147.20.35:9418/Layer-Depth-Attention-zt.git` succeeded; server `git reset --hard zerotier-local/main` succeeded.
- Next step: Use `git push zerotier-local main` locally after code changes, then `git fetch zerotier-local && git reset --hard zerotier-local/main` on the server before launching experiments.

### [Step 081] - 2026-03-31 02:44 CST - Record the final normalized QKV-reprojection result
- Request: Rerun `depth_memory_qkv_reproj` after correcting its semantics and adding pre-projection normalization, then compare it against the value-only branches.
- Plan: Use the ZeroTier sync path to update the server to the corrected code and complete the full `2000`-step benchmark with `300`-step logging.
- Files touched: `dev_log.md`
- Modification: Recorded the finished normalized `depth_memory_qkv_reproj` run from `artifacts/wikitext103probe_depth_memory_qkv_reproj_bs8_2000_v4.json`.
- Rationale: Earlier `depth_memory_qkv_reproj` numbers referred to obsolete semantics and should not be used for the final comparison table.
- Key details: The corrected+normalized run logged validation losses `8.7440`, `5.6381`, `4.6261`, `4.2259`, `4.0336`, `3.9515`, `3.9154` at steps `300/600/900/1200/1500/1800/2000`, and finished with `test_loss=4.1991`, `test_ppl=66.63`. This essentially matches `value_reproj` (`66.67`) and remains slightly behind `value_reproj_normed` (`66.26`).
- Side effects: The current mechanism conclusion should use this normalized v4 result, not the earlier `68.17` run from the old semantics.
- Verification: Remote CUDA run completed and the artifact contents were inspected.
- Next step: Explore whether a better query design can improve the value-reprojection branch further without losing the projection-consistency benefit.

### [Step 082] - 2026-03-31 03:03 CST - Add a dual-query value-reprojection variant
- Request: Based on `value_reproj`, introduce two query projections: one for same-row token attention and one for same-column depth-memory attention, with the column query shared across all layers rather than recreated per layer.
- Plan: Add a new attention type that keeps the current layer's native query for token-to-token scores, adds a model-level shared `W_Q` for column/depth retrieval, wires the option through the training CLI, and verify the variant locally before any remote run.
- Files touched: `src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- Modification: Added `LayerDepthValueReprojDualQAttention`, extended `TransformerBlock` and `TinyDecoderLM` so a single `shared_column_q_proj` is created once at model construction time and reused by every block, and exposed the new option as `depth_memory_value_reproj_dualq` in `train_wikitext_lm.py`.
- Rationale: The user wants row attention and column/depth retrieval to use different query projections while ensuring the column-side query mapping is globally shared across depth.
- Key details: The current implementation keeps token scores on the existing layer-specific `q_row` from `qkv_proj`, uses the shared `column_q_proj` only for memory scores, and leaves the memory value path identical to the current non-normalized `value_reproj` branch.
- Side effects: This adds one new global projection matrix to the model only when the dual-query attention type is selected.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke on `TinyDecoderLM(attention_type='depth_memory_value_reproj_dualq')`.
- Next step: Commit the dual-query implementation, sync it to the server through ZeroTier, and run the large-scale benchmark to see whether separating row/column queries improves over `value_reproj`.

### [Step 083] - 2026-03-31 03:10 CST - Correct the dual-query memory path to direct historical K/V reuse
- Request: Update the new dual-query variant so the same-column branch uses previously computed historical `K/V` directly instead of reprojecting them with the current layer.
- Plan: Keep the dual-query design, but replace the memory-side reprojection path with direct same-position historical `K/V` reuse; then re-run local compile and smoke checks before any server experiment.
- Files touched: `src/layer_depth_attention/model.py`, `dev_log.md`
- Modification: Changed `LayerDepthValueReprojDualQAttention` so token scores still use the current layer's row query, memory scores use the shared column query, and the memory bank now consumes stacked historical `K/V` directly with no extra `_kv_proj` on the column branch.
- Rationale: The user clarified that the dual-query experiment should isolate query separation only; historical column memories should remain the original projected `K/V` from earlier layers.
- Key details: This makes the branch semantically closer to `depth_memory` plus a shared column-query projection, rather than a reprojection-based memory variant. The attention-type name is unchanged for continuity, but its intended meaning is now "dual-query direct depth memory."
- Side effects: Earlier expectations for this variant should no longer compare it directly to `value_reproj`; the current experiment now isolates the effect of using different row/column query projections.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke on `TinyDecoderLM(attention_type='depth_memory_value_reproj_dualq')`.
- Next step: Commit the corrected dual-query semantics, sync to the server, and run the large-scale benchmark.

### [Step 084] - 2026-03-31 09:09 CST - Complete the 2000-step dual-query direct-memory benchmark
- Request: Evaluate the corrected dual-query variant under the main long-budget benchmark.
- Plan: Sync commit `a90b77a` to the Windows server and run `depth_memory_value_reproj_dualq` for `2000` steps with progress printed every `300` steps.
- Files touched: `dev_log.md`
- Modification: Completed the remote CUDA run and recorded the metrics from `artifacts/wikitext103probe_depth_memory_value_reproj_dualq_bs8_2000.json`.
- Rationale: This is the first full test of the "row query layer-specific + column query globally shared + direct historical K/V reuse" design.
- Key details: Logged validation losses were `8.5664`, `5.5497`, `4.5682`, `4.1851`, `3.9969`, `3.9147`, `3.8850` at steps `300/600/900/1200/1500/1800/2000`. Final metrics were `test_loss=4.1970`, `test_ppl=66.49`. This slightly improves over `value_reproj` (`66.67`) and remains very close to `value_reproj_normed` (`66.26`), while still trailing `Attention Residuals` (`60.62`).
- Side effects: The current evidence suggests that separating row and column queries helps a little even without memory reprojection, but the gain is still modest.
- Verification: Remote 2000-step CUDA benchmark completed successfully and wrote the artifact file.
- Next step: Compare whether the shared-column-query idea combines better with normalization or with stronger residual-attention baselines, or stop here and consolidate the current mechanism findings for paper positioning.

### [Step 085] - 2026-03-31 09:18 CST - Switch the dual-query branch to per-layer column queries
- Request: Change the dual-query implementation so the same-column `W_Q^{col}` is no longer shared across all layers, and add a dedicated experiment-summary Markdown file.
- Plan: Remove the model-level shared column-query projection, let each `depth_memory_value_reproj_dualq` block instantiate its own `column_q_proj`, verify locally, and create a standalone notes file for method names and long-run benchmark results.
- Files touched: `src/layer_depth_attention/model.py`, `experiment_notes.md`, `dev_log.md`
- Modification: Reworked `LayerDepthValueReprojDualQAttention` so each block creates its own `nn.Linear(d_model, d_model)` for the column query, removed the `TinyDecoderLM` and `TransformerBlock` plumbing for a shared column-query projection, and added `experiment_notes.md` summarizing method definitions and the main `2000`-step comparison table.
- Rationale: The user wants the dual-query variant to use a separate same-column query mapping per layer rather than a single globally shared mapping.
- Key details: The current semantics of `depth_memory_value_reproj_dualq` are now: row query from each block's native `qkv_proj`, column query from a per-block `column_q_proj`, and direct reuse of historical same-position `K/V` with no memory reprojection.
- Side effects: The previously reported `66.49` result belongs to the older shared-column-query version and should not be reused as the result for the current implementation.
- Verification: `python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`; local pure-`torch` forward/backward smoke on `TinyDecoderLM(attention_type='depth_memory_value_reproj_dualq')`.
- Next step: If needed, rerun the long-budget benchmark for the updated per-layer-column-query version and compare it against the old shared-column-query result.

### [步骤 086] - 2026-03-31 09:24 CST - 切换为中文记录并准备重跑独立列查询版本
- 请求：从现在开始使用中文记录实验和操作，并启动“每层独立同列 `W_Q`”版本的正式实验。
- 计划：先把当前本地修改提交并同步到服务器，再用统一主配置重新运行 `depth_memory_value_reproj_dualq`，避免继续引用旧的共享列查询结果。
- 涉及文件：`dev_log.md`
- 修改内容：新增中文记录约定，并明确接下来要重跑的是“每层独立同列 `W_Q` + 直接复用历史 `K/V`”版本。
- 原因：用户要求后续实验与操作记录改为中文，同时当前代码语义已经变化，需要新结果覆盖旧结果。
- 关键信息：旧结果 `test_ppl=66.49` 只对应“全层共享同列 `W_Q`”实现，不能再代表当前版本。
- 影响：从这一条开始，后续新增实验记录优先使用中文；历史英文记录保留不改。
- 验证：尚未开始服务器重跑。
- 下一步：提交当前修改，推送到仓库，服务器拉取后启动 2000-step 实验。

### [步骤 087] - 2026-03-31 09:26 CST - 完成每层独立列查询版本的 2000 步实验
- 请求：重跑 `depth_memory_value_reproj_dualq`，但这次同列 `W_Q` 改为每层独立，不再全层共享。
- 计划：将提交 `00a0521` 同步到服务器，保持主配置不变，运行 `2000` 步并每 `300` 步输出一次验证日志。
- 涉及文件：`dev_log.md`
- 修改内容：记录服务器端实验 `artifacts/wikitext103probe_depth_memory_value_reproj_dualq_perlayer_bs8_2000.json` 的完整结果。
- 原因：当前代码语义已经从“共享同列 `W_Q`”切换到“每层独立同列 `W_Q`”，必须用新结果替换旧版本结论。
- 关键信息：验证集日志依次为：`step 300 val_loss=8.6385`、`600=5.5330`、`900=4.6575`、`1200=4.2935`、`1500=4.1117`、`1800=4.0180`、`2000=3.9862`。最终 `test_loss=4.2056`、`test_ppl=67.06`。
- 影响：该版本仍明显优于 `baseline (69.14)` 和原始 `depth_memory (68.89)`，但略弱于旧的共享列查询版本 `66.49`，也弱于 `value_reproj (66.67)` 与 `value_reproj_normed (66.26)`。
- 验证：服务器 CUDA 训练完成，结果文件已生成并检查。
- 下一步：更新实验总表，明确区分“共享列查询版本”和“每层独立列查询版本”，避免混淆。

### [步骤 088] - 2026-03-31 09:34 CST - 切换到 CIFAR100 视觉基准
- 请求：不要只停留在语言模型上，尝试用 CIFAR100 再跑一轮，看看方法在视觉任务上是否也有信号。
- 计划：先确认服务器上的 CIFAR100 数据路径，再补一个最小可比的小型 ViT 训练入口，用统一骨架比较 `baseline` 和候选注意力变体。
- 涉及文件：`dev_log.md`
- 修改内容：记录了 CIFAR100 数据位置与接下来的实现方向。
- 原因：用户希望把当前注意力机制放到行业常见的小型视觉分类基准上验证，而不是只看文本任务。
- 关键信息：服务器上已存在 `D:\Projects\data\cifar-100-python` 与 `D:\Projects\data\cifar-100-python.tar.gz`；`torchvision` 可用，因此不需要额外下载数据依赖。
- 影响：接下来会新增一个视觉训练入口；语言模型主线代码保持不变。
- 验证：已确认服务器存在 CIFAR100 数据目录，且 `torchvision` 可导入。
- 下一步：实现 CIFAR100 小型 ViT 训练脚本，并接入当前注意力变体做短程探测实验。

### [步骤 089] - 2026-03-31 09:41 CST - 补齐 CIFAR100 小型 ViT 训练入口
- 请求：在 CIFAR100 上快速验证现有注意力机制是否也能工作。
- 计划：新增一个最小可比的 ViT 训练脚本和视觉版 attention/block 实现，先只支持 `baseline` 与 `depth_memory_value_reproj_normed` 两条，足够跑首轮探测。
- 涉及文件：`src/layer_depth_attention/vision_model.py`, `train_cifar100_vit.py`, `dev_log.md`
- 修改内容：新增 `TinyVisionTransformer`、patch embedding、视觉版 baseline attention、视觉版 `value_reproj_normed` attention，以及对应的 CIFAR100 训练入口。
- 原因：现有语言模型训练代码带因果掩码，不适合直接用于 CIFAR100 图像分类；需要一个非因果的视觉序列骨架。
- 关键信息：当前视觉基线采用 `32x32` 输入、`4x4` patch、`[CLS]` 分类 token、小型 ViT 主干；首轮只比较最基础和最稳的一个自家变体，避免一次铺太多分支。
- 影响：项目现在同时具备文本与视觉两条实验入口，但两者代码相互独立，减少互相污染。
- 验证：`python -m py_compile src/layer_depth_attention/vision_model.py train_cifar100_vit.py`；本地前向/反向 smoke test 通过，`baseline` 与 `depth_memory_value_reproj_normed` 都能输出 `(2, 100)`。
- 下一步：提交当前改动，同步到服务器，在 CIFAR100 上先跑一轮短程 probe。

### [步骤 090] - 2026-03-31 09:46 CST - 完成 CIFAR100 首轮短程探测
- 请求：在 CIFAR100 上先跑一轮小规模实验，看看视觉基准能不能正常起量，以及当前变体有没有初步信号。
- 计划：在同一套小型 ViT 配置上并行运行 `baseline` 与 `depth_memory_value_reproj_normed`，各训 `5` 个 epoch，比较测试集 loss 和 accuracy。
- 涉及文件：`dev_log.md`
- 修改内容：记录了服务器端两条 CIFAR100 probe 的结果。
- 原因：先用短程实验判断这条视觉基准是否可用，再决定是否值得拉长训练预算。
- 关键信息：配置为 `CIFAR100`、`d_model=256`、`num_layers=6`、`num_heads=8`、`patch_size=4`、`batch_size=128`、`epochs=5`。结果如下：
  - `baseline`：`test_loss=2.5585`，`test_acc=0.3375`
  - `depth_memory_value_reproj_normed`：`test_loss=2.5765`，`test_acc=0.3336`
- 影响：视觉版 `value_reproj_normed` 能正常训练，但在当前短预算下还没有超过 baseline；需要更长训练或更合适的视觉超参后再下结论。
- 验证：服务器生成了 `artifacts/cifar100_baseline_probe.json` 与 `artifacts/cifar100_value_reproj_normed_probe.json`。
- 下一步：如果继续做 CIFAR100，优先把训练拉长到 `20~50` epoch 再看趋势，而不是直接下方法无效结论。

### [步骤 091] - 2026-03-31 10:36 CST - 完成 CIFAR100 的 50 epoch 对比
- 请求：把 `baseline` 和 `depth_memory_value_reproj_normed` 在 CIFAR100 上一起拉到 `50` 个 epoch，观察长训练下的真实趋势。
- 计划：保持与 probe 相同的小型 ViT 配置，只把训练轮数从 `5` 提到 `50`，并行运行两条实验。
- 涉及文件：`dev_log.md`
- 修改内容：记录两条 `50 epoch` 结果及其关键趋势。
- 原因：5 个 epoch 只能说明训练入口能跑通，不足以判断视觉任务上的最终趋势。
- 关键信息：配置保持为 `d_model=256`、`num_layers=6`、`num_heads=8`、`patch_size=4`、`batch_size=128`。最终结果：
  - `baseline`：`test_loss=2.6447`，`test_acc=0.5216`
  - `depth_memory_value_reproj_normed`：`test_loss=2.6788`，`test_acc=0.5145`
  两条方法都在约 `epoch 46~50` 达到各自最高测试精度区间，但最终仍是 baseline 略优。
- 影响：在当前 CIFAR100 小型 ViT 设定下，`value_reproj_normed` 还没有复现语言模型中的优势；至少在这组视觉配置上并不占优。
- 验证：服务器生成了 `artifacts/cifar100_baseline_50ep.json` 与 `artifacts/cifar100_value_reproj_normed_50ep.json`，两条训练进程均已结束。
- 下一步：如果继续深挖视觉任务，应优先调整视觉侧超参或结构适配，而不是直接照搬文本版结论。

### [步骤 092] - 2026-03-31 10:44 CST - 新增二维前缀文本注意力分支
- 请求：回到文本任务，改成更激进的二维注意力区域。用户的约束是：每个位置不仅看当前层左边 token，还看所有更浅层中该位置左边的所有元素，也就是 layer-token 平面上的“左边或下面”区域。
- 计划：新增一个文本 attention 变体，把当前层的标准 causal token 区和所有浅层的前缀 token 区拼成统一注意力空间，并沿用现有训练入口做后续实验。
- 涉及文件：`src/layer_depth_attention/model.py`, `train_wikitext_lm.py`, `dev_log.md`
- 修改内容：新增 `depth_memory_2d_prefix` 分支，实现当前 query 同时访问当前层 `1..k` 的 token，以及所有更浅层中 `1..k` 的 token；并把新分支加入训练脚本的 `attention_type` 选项。
- 原因：用户希望把 depth memory 从“同列检索”扩展成真正的二维左下三角检索区域。
- 关键信息：当前实现中，历史部分直接复用过去层缓存的 `K/V`，并用位置掩码保证第 `k` 个 query 只能看到各浅层中的前缀 `1..k`，不会越过右侧未来位置。
- 影响：这条分支的 memory 空间会从按层 `i-1` 个同列槽位，扩展到按层前缀累计的 `k(i-1)` 个槽位，计算量显著增加。
- 验证：`python -m py_compile src/layer_depth_attention/model.py train_wikitext_lm.py`；本地前向/反向 smoke test 通过，`TinyDecoderLM(attention_type='depth_memory_2d_prefix')` 能输出 `(2, 16, 128)`。
- 下一步：如果继续，就把这条分支同步到服务器，在文本基准上先跑一轮短程 probe。
