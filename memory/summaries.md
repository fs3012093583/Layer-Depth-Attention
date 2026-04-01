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

## 2026-04-01 01:38 CST - SwanLab 记录统一接入
- Scope:
  为当前项目的训练脚本增加统一的可选 SwanLab 实验记录能力，并创建一个未来可复用的全局 skill。
- Change:
  新增共享模块 `experiment_logging.py`；`train_wikitext_lm.py` 和 `train_assoc_recall.py` 增加 `--log-backend/--log-project/--log-experiment-name`；在 `~/.codex/skills` 新建 `swanlab-experiment-logging`。
- Why it matters:
  后续做实验时不需要反复手写监控逻辑，也不会因为缺少 SwanLab 环境而影响训练本身。
- Open items:
  目前只完成了 1-step smoke run；正式长训练仍需要按具体实验命令开启 `--log-backend swanlab`。
- Re-read if:
  后续继续扩展实验记录、统一不同训练脚本的监控接口，或把服务器训练接到同一看板时。

## 2026-04-01 06:24 CST - 论文主稿骨架落地
- Scope:
  将已有方法笔记与实验记录收敛成一份可继续扩写的论文初稿。
- Change:
  新增 `paper_draft.md`，写入题目、摘要、引言、方法、实验设置、结果、讨论、结论和补写清单。
- Why it matters:
  项目已经从“方法探索”进入“论文收束”阶段，需要一个稳定的主叙事承接后续 Related Work、表格和图。
- Open items:
  仍需补正式公式排版、参考文献、表格格式，以及决定是否把 `dual_axis_*` 系列并入正文。
- Re-read if:
  后续继续论文写作、改摘要、改标题，或将 Markdown 稿件迁移到 LaTeX 时。
