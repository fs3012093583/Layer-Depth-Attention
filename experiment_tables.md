# 实验对照表

## 1. 文本主配置对照表

配置：
- 数据集：`wikitext-103-probe`
- 模型：`d_model=384`, `num_layers=16`, `num_heads=8`, `seq_len=256`
- 训练：`batch_size=8`, `steps=2000`

| 方法 | Step 2000 Train Loss | Final Val Loss | Final Val PPL | Final Test Loss | Final Test PPL |
|---|---:|---:|---:|---:|---:|
| `baseline` | 3.1920 | 3.9203 | 50.41 | 4.2361 | 69.14 |
| `depth_memory_value_reproj` | 3.2543 | 3.9107 | 49.93 | 4.1997 | 66.67 |
| `depth_memory_value_reproj_normed` | 3.2477 | 3.9219 | 50.49 | 4.1936 | 66.26 |
| `attn_residuals` | 4.8545 | 3.8722 | 48.05 | 4.1046 | 60.62 |
| `depth_memory_directkv_dualq` | 4.7427 | 3.9869 | 53.89 | 4.2063 | 67.11 |

说明：
- `attn_residuals` 在 `probe` 主配置上最强。
- 自定义方法里，`depth_memory_value_reproj_normed` 是当前最强版本。
- `depth_memory_directkv_dualq` 有正收益，但没有超过 `depth_memory_value_reproj_normed`。

## 2. 更大模型 + 更大数据集对照表

配置：
- 数据集：`wikitext-103-raw-v1`
- 模型：`d_model=512`, `num_layers=20`, `num_heads=8`, `seq_len=256`
- 训练：`batch_size=2`, `grad_accum_steps=4`, `steps=500`

| 方法 | Step 100 Train Loss | Step 200 Train Loss | Step 300 Train Loss | Step 400 Train Loss | Step 500 Train Loss | Final Val Loss | Final Test Loss | Final Test PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 28.4340 | 21.5795 | 17.0777 | 15.1395 | 15.3578 | 15.5705 | 14.3806 | 1759555.61 |
| `depth_memory_value_reproj_normed` | 29.1647 | 21.2487 | 17.2492 | 15.1673 | 15.4297 | 15.6581 | 14.3122 | 1643285.39 |
| `depth_memory_directkv_dualq` | 27.6589 | 20.6273 | 16.2186 | 15.4550 | 15.7590 | 15.8767 | 14.5655 | 2116994.37 |
| `depth_memory_directkv_qmix` | 27.6764 | 20.6695 | 16.3302 | 15.3977 | 15.6442 | 16.0246 | 14.6302 | 2258400.96 |
| `attn_residuals` | 30.1476 | 21.4423 | 19.3461 | 16.6177 | 15.4684 | 15.7318 | 15.0075 | 3293707.52 |

说明：
- 当前大设定下，`depth_memory_value_reproj_normed` 的 `test` 指标最好。
- `baseline` 排第二。
- `depth_memory_directkv_dualq` 与 `depth_memory_directkv_qmix` 在大设定下没有维持住 `probe` 上的相对优势。
- `attn_residuals` 在这组大设定和当前训练配方下表现最差。

## 3. 大数据集 + 小模型 2000 Step 对照表

配置：
- 数据集：`wikitext-103-raw-v1`
- 模型：`d_model=384`, `num_layers=16`, `num_heads=8`, `seq_len=256`
- 训练：`batch_size=4`, `grad_accum_steps=4`, `steps=2000`

| 方法 | Step 300 Train Loss | Step 600 Train Loss | Step 900 Train Loss | Step 1200 Train Loss | Step 1500 Train Loss | Step 1800 Train Loss | Step 2000 Train Loss | Final Val Loss | Final Val PPL | Final Test Loss | Final Test PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 13.3365 | 8.5184 | 7.5543 | 6.9758 | 6.5944 | 6.7837 | 6.8358 | 6.5931 | 730.01 | 6.4540 | 635.25 |
| `depth_memory_directkv_dualq` | 13.9043 | 8.4032 | 7.7700 | 7.3070 | 6.9088 | 6.8826 | 7.0036 | 6.5711 | 714.16 | 6.4109 | 608.42 |

说明：
- 在“大数据集 + 小模型 + 2000 step”这组更充分训练的设定下，`depth_memory_directkv_dualq` 明显优于 `baseline`。
- 相比之前 `500 step` 的大设定结果，这说明 `directkv_dualq` 在完整 `WikiText-103` 上更依赖训练预算。
