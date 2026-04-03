# 算法优化待办

## 背景

- 当前 `dual_axis_full` / `dual_axis_full_no_final_mix` 系列方法训练开销明显偏大。
- 目前观察到：
  - **每个训练 step 的耗时大约是 baseline 的 4 倍**
- 这已经成为后续实验推进的现实瓶颈。

## 当前目标

- 在尽量不破坏当前算法核心思想的前提下，优先降低训练时间和显存/计算开销。
- 后续所有结构改动，都需要同时考虑：
  - 性能是否提升
  - 训练速度是否明显变慢

## 待办项

- [ ] 统计并拆解当前 `dual_axis_full` 的耗时来源  
  先明确时间主要消耗在：
  - `_attn_res_dual_axis_mix`
  - `DualAxisMemoryAttention`
  - 还是数据/评估部分

- [ ] 对 `dual_axis_full` 和 `baseline` 做逐模块 profiler 对比  
  记录：
  - attention 前预混合耗时
  - 主 attention 耗时
  - MLP 耗时
  - eval 耗时

- [ ] 检查 `dual_axis` 预混合是否存在重复的大矩阵构造与拼接  
  特别关注：
  - row/depth 分支张量是否反复 reshape / concat
  - 是否能减少中间 tensor 创建

- [ ] 检查 `DualAxisMemoryAttention` 的 memory 路径是否可以做缓存优化  
  重点看：
  - `memory_v_proj`
  - 历史状态堆叠
  - 多头拆分与重排

- [ ] 评估是否可以限制 depth 候选长度  
  例如：
  - 只取最近若干层
  - block 化历史
  - 降低 depth 方向候选数量

- [ ] 评估是否需要把预混合从“全量 token × depth”降成更轻量的近似形式  
  例如：
  - 低秩近似
  - block summary
  - 更小的 query/key 维度

- [ ] 将“速度”加入后续实验对照表  
  以后实验记录中除 loss / ppl 外，至少补：
  - step time
  - 总训练耗时
  - 相对 baseline 倍率

## 当前结论

- 当前算法不是不能跑，而是**训练成本过高**。
- 如果不优先解决速度问题，后续大规模实验与多轮对照都会受到限制。
- 因此“降低 step 耗时”已经是和“提升指标”同等优先级的问题。
