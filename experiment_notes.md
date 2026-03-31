# Experiment Notes

## Main Benchmark

- Dataset: `wikitext-103-probe`
- Model: decoder-only LM
- Core config: `d_model=384`, `num_layers=16`, `num_heads=8`, `seq_len=256`, `batch_size=8`
- Long-run budget: `2000` training steps

## Method Names

- `baseline`
  Standard decoder-only Transformer with causal self-attention.

- `depth_memory`
  Same-position cross-layer memory using historical `K/V` directly, with the current layer query attending over current-row tokens plus same-column history.

- `depth_memory_value_reproj`
  Same-position historical `V` is reconstructed to model space and then reprojected through the current layer `K/V` projection path before entering the memory bank.

- `depth_memory_value_reproj_normed`
  Same as `depth_memory_value_reproj`, but the reconstructed historical `V` is normalized before current-layer reprojection.

- `depth_memory_qkv_reproj`
  Same-position historical `Q/K/V` are treated as separate memory slots, normalized, and then passed through the current layer `K/V` projection path.

- `depth_memory_value_reproj_dualq`
  Dual-query direct-memory variant.
  Current-row token attention uses the layer-specific `W_Q` from the block's `qkv_proj`.
  Same-column depth-memory attention uses a separate per-layer `W_Q^{col}`.
  Historical same-column `K/V` are reused directly and are not reprojected.

- `attn_residuals`
  Attention Residuals baseline implemented in-repo for comparison.

- `attn_residuals_value_reproj_normed`
  Attention Residuals combined with normalized value reprojection.

## 2000-Step Results

| Method | Val Loss | Val PPL | Test Loss | Test PPL |
|---|---:|---:|---:|---:|
| `baseline` | 3.9203 | 50.41 | 4.2361 | 69.14 |
| `depth_memory` | 3.9228 | 50.54 | 4.2325 | 68.89 |
| `depth_memory_value_reproj` | 3.9107 | 49.93 | 4.1997 | 66.67 |
| `depth_memory_value_reproj_normed` | 3.9219 | 50.49 | 4.1936 | 66.26 |
| `depth_memory_qkv_reproj` | 3.9154 | 50.17 | 4.1991 | 66.63 |
| `depth_memory_value_reproj_dualq` | 3.8850 | 48.67 | 4.1970 | 66.49 |
| `attn_residuals` | 3.8722 | 48.05 | 4.1046 | 60.62 |
| `attn_residuals_value_reproj_normed` | 3.8653 | 47.72 | 4.1274 | 62.01 |

## Current Reading

- Projection-space alignment matters: direct `depth_memory` improves only slightly over `baseline`, while reprojection-based variants improve more consistently.
- Normalizing historical `V` before reprojection gives a small but repeatable gain over plain `value_reproj`.
- Adding historical `Q/K` into reprojection does not clearly beat the simpler value-only route.
- Dual queries help a little, but the current per-layer dual-query direct-memory variant still does not exceed `depth_memory_value_reproj_normed`.
- `Attention Residuals` remains the strongest comparator in the current benchmark.
