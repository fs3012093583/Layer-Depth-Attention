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
