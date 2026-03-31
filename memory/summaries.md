# Memory Summaries

## 2026-03-31 12:18 CST - 双 q 同时进入 attention 与 FFN
- Scope:
  文本主线新增一个显式新分支，把同行 `q_row` / 同列 `q_col` 的拆分同时应用到 attention 子层和 FFN 替代子层。
- Change:
  新增 `depth_memory_value_reproj_normed_dualq_ffn_qattn_dualq`，保留旧方法名和旧结果不变。
- Why it matters:
  这样可以单独验证“双 q”这个设计本身，而不是把它和已有 `value_reproj_normed` 或 `ffn_qattn` 的结果混在一起。
- Open items:
  还没有服务器实验结果，需要先跑一轮短程 probe。
- Re-read if:
  后续继续比较 `dualq` 在 attention/FFN 不同插入位置的效果时。

## 2026-03-31 13:08 CST - 论文实验从零散跑数转向标准矩阵
- Scope:
  把当前工作重构成毕业论文导向的实验设计，而不是继续追加零散单次实验。
- Change:
  激活 layered memory，补齐 `memory/events.jsonl`，并把当前任务收束为“标准基线 + 你的方法 + 必要消融 + 分阶段执行顺序”的实验矩阵设计。
- Why it matters:
  现有仓库已经有可跑的文本与合成任务入口，主要瓶颈变成实验组织是否能支撑论文论证。
- Open items:
  需要明确用户口述中的“稳定注意力/长上下文注意力”各自对应哪类具体基线，并决定主表只放哪些方法。
- Re-read if:
  后续开始补基线实现、写实验章节、或安排服务器批量跑数时。

## 2026-03-31 13:22 CST - 新主方法 dual_axis_memory 已接入并完成首轮 probe
- Scope:
  把用户定义的“横向 + 纵向双轴追忆”收敛成一个单独的方法名并接入当前实验框架。
- Change:
  新增 `dual_axis_memory`：行内分支保持标准 causal self-attention；同列分支用独立 `q_col` 检索前层同 token 输出，并在纵向分支上直接令 `K = V = x`。训练入口已支持该方法。
- Why it matters:
  现在主论文最小对比集 `baseline / attn_residuals / ours` 已经可以直接运行，不需要继续借用旧的 `depth_memory_*` 名字代指你的最终方法。
- Open items:
  小型 CPU probe 说明实现稳定，但结果暂时只和 `baseline` 接近、弱于 `attn_residuals`；需要更合适的规模与预算验证趋势，并视情况加入 gate 或进一步对齐归一化。
- Re-read if:
  后续决定 `ours` 的最终结构、整理主表、或解释为什么第一轮 probe 没有立刻超过 `attn_residuals` 时。
