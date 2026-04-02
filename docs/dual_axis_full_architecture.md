# Dual-Axis Full 模型架构详细设计蓝图

本文档提供了 `dual_axis_full` 模型的 **1:1 完美复刻级设计说明**。包含完整的数学公式、维度变换记录以及所有操作步骤。任何人根据本文档即可用任何框架从零实现该架构。

## 1. 宏观架构定义与输入层

模型是一个 **Decoder-only 语言模型**。核心特征是**没有任何传统的层级硬残差连接（如 $x = x+f(x)$ ），一切跨层信息全部由动态注意力机制控制**。

给定超参数：
*   $V$: 词汇表大小 (Vocab Size)
*   $S$: 最大序列长度 (Seq Len)
*   $D$: 隐藏层维度 (D_model)
*   $H$: 注意力头数 (Num Heads)
*   $d$: 单头维度 (Head Dim) = $D/H$
*   $L$: Transformer 块（层）的数量 (Num Layers)

### 1.1 嵌入层 (Embedding Layer)
给定输入序列的 token index $I \in \mathbb{N}^{B \times S}$ (Batch size $B$)：
*   **Token Embedding**: 查表获取词向量 $E_{token} = \text{Embedding}(I) \in \mathbb{R}^{B \times S \times D}$
*   **位置嵌入**: 采用绝对位置编码 $P_{idx} = [0, 1, \dots, S-1]$。
    $E_{pos} = \text{PosEmbedding}(P_{idx}) \in \mathbb{R}^{1 \times S \times D}$
*   **初始输入特征**: 
    $x_0 = E_{token} + E_{pos} \in \mathbb{R}^{B \times S \times D}$

系统维护一个历史列表 $\mathcal{H} = []$ 用于存储经历过的所有非线性输出。

---

## 2. 核心模块：双轴动态预混合 (`_attn_res_dual_axis_mix`)

在进入任何一个注意力块（Attention）或前馈网络块（FFN）之前，都要生成它的动态输入。假设当前我们需要为第 $l$ 个计算块生成输入 $\hat{x}_l$。

对于这一个计算块，模型持有一个**独立的可学习列查询向量** $q_{depth}^{l} \in \mathbb{R}^{D}$，以及一个**独立的行投影矩阵** $W_{row\_q}^{l} \in \mathbb{R}^{D \times D}$。

### 步骤 2.1 获取最新的状态
当前最新的状态定义为：$x_{current} = \mathcal{H}[-1]$。如果 $\mathcal{H}$ 为空，则 $x_{current} = x_0$。

### 步骤 2.2 横向预平滑 (Row-wise Mix)
仅在 $x_{current}$ 上计算一次短期的特征过滤：
1. **RMSNorm**: 对 $x_{current}$ 进行无偏置的 Root Mean Square Normalization。
   $x_{norm} = \text{RMSNorm}(x_{current}) $
2. **行查询映射**: 通过独占的线性层获取当前序列的查询：
   $Q_{row} = x_{norm} W_{row\_q}^{l} \in \mathbb{R}^{B \times S \times D}$ （随后拆分为多头 $B \times H \times S \times d$）
3. **因果自注意力与上下文**: 用 $x_{norm}$ 作为 Key 和 Value（在此步骤不经额外映射）：
   $K_{row} = V_{row} = x_{norm}$ （同样拆分为多头）
   $$\text{Scores}_{row} = \frac{Q_{row} K_{row}^T}{\sqrt{d}}$$
   应用上三角（对角线向上平移 1）的因果 Mask（赋值为 $-\infty$）。
   $$\text{Weights}_{row} = \text{Softmax}(\text{Scores}_{row})$$
   $$\text{Context}_{row} = \text{Weights}_{row} V_{row} \in \mathbb{R}^{B \times S \times D}$$

### 步骤 2.3 纵向全局混合 (Column-wise Mix / Attention Residuals)
> [!NOTE] 
> **📝 预混合归一化机制与构成探讨 (TODO)**
> - [ ] **跨层归一化有必要吗？** 
>       *(解答：有必要，横向对 $x_{current}$ 或纵向对 $V_{stack}$ 归一化可防止数值爆炸，确立点积公平性。)*
> - [ ] **x和y轴分别先算出注意力矩阵归一化再后拼起来 VS 先拼起来再softmax，哪种较好？**
>       *(解答：各有用途。此处的预载混合使用分别归一化后相加，让时间和深度各自稳定发力；而下一节的核心 Attention 层则会把分数拼起来一起做全局竞争。)*
> - [ ] **堆叠的 $V_{stack}$ 这里具体包括哪些历史元素？**
>       *(解答：包含最根源的词向量 $x_0$，以及在这之前产生过的每一个 Attention 输出和每一个 MLP 输出的逐层累加。)*
> - [ ] **这里有必要用 dropout 吗？**
> - [ ] **这一步的输出基本上是接近归一化的，还需要归一化吗？如果需要，归一化后再输出到下一个模块？**

1. **堆叠所有层**:
   $V_{stack} = [x_0, \mathcal{H}[0], \dots, \mathcal{H}[-1]] \in \mathbb{R}^{B \times S \times M \times D}$  （其中 $M$ 是历史元素的总数）
2. **RMSNorm**: 沿着维度 $D$ 做归一化。
   $V_{norm} = \text{RMSNorm}(V_{stack})$

> [!WARNING]
> **📝 修复项 (TODO)**
> - [x] 公式中计算点积打分时，遗漏了缩放因子。**应该除以 $\sqrt{D}$**，以此保证模型深网参数放缩时的稳定性。

1. **深度加权注意力**: 使用独立的查询向量 $q_{depth}^{l}$ 去与所有 $M$ 个特征计算点积打分。
   $$\text{Scores}_{depth} = V_{norm} \cdot q_{depth}^{l} \in \mathbb{R}^{B \times S \times M}$$
   $$\text{Weights}_{depth} = \text{Softmax}_{dim=2}(\text{Scores}_{depth})$$
   通过加权求和得出最终的跨层残差平滑值：
   $$\text{Context}_{depth} = \sum_{m=1}^{M} \text{Weights}_{depth}[..., m] \cdot V_{stack}[..., m, :] \in \mathbb{R}^{B \times S \times D}$$

### 步骤 2.4 最终混合输出
$$\hat{x}_l = \text{Context}_{row} + \text{Context}_{depth} \in \mathbb{R}^{B \times S \times D}$$

---

## 3. 双端完全体子层 (`DualAxisMemoryAttention`)

> [!NOTE]
> **📝 完全体双端注意力机制探讨 (TODO)**
> - [ ] **归一化方式保持一致，为什么这里换成了 LayerNorm？**
>       *(解答：预混合时经历相加后，特征均值存在偏差漂移。这里采用带常数的 LayerNorm 进行均值清零（Mean-Centering），为重度矩阵乘法投映提供扎实的平稳零基准。)*
> - [ ] **要不要让 K=V，并且多个头共享？**
>       *(解答：主力层绝对不建议把 K 和 V 强行等同，会严重限制模型语言理解能力上限。但是极度推荐“多个不同头共享少量的 K 和 V(即 GQA)”，从而大幅节省显存壁垒。)*
> - [ ] **行和列的 q 不是同一个 $Q_{row} = Q_{col}$ 应该用同一个吧？如果用同一个，列的对应元素也要用和行相同的投影方法乘。**
> - [ ] **而且列也需要先归一化，既然这样我们可以把左右的归一化放在模块内部的最后，这样可以避免重复归一化。**
> - [ ] **两步可以一起算，纯算法步骤上应该可以优化。**
> - [ ] **把归一化接在后面。**

1. **计算行自注意力分数 (Token Scores)**：一致的投影*
将步骤 2 生成的 $\hat{x}_l$ 作为 Attention 子层的输入。

1. **层归一化 (LayerNorm)**：应用标准带参数的 LayerNorm。$x' = \text{LayerNorm}(\hat{x}_l)$。
2. **生成独立三轴查询 ($Q, K, V, Q_{col}$)**：
   * 行查询，Key，Value 通过同一矩阵投影：$Q_{row}, K, V = \text{Linear}_{3d}(x')$
   * 特殊的列查询，独立矩阵投影：$Q_{col} = \text{Linear}_{1d}(x')$
3. **计算行自注意力分数 (Token Scores)**：
   $$\text{Scores}_{token} = \frac{Q_{row} K^T}{\sqrt{d}} \in \mathbb{R}^{B \times H \times S \times S}$$
   加入因果 Mask 遮蔽掉未来信息。
4. **计算列深度注意力分数 (Memory Scores)**：
   提取自第一层堆叠到现在的历史状态 $\mathcal{H} \in \mathbb{R}^{B \times S \times L_{past} \times D}$。
   先对其取普通 `LayerNorm`（注：非 RMSNorm）。
   维度调整为 $(B, H, S, L_{past}, d)$ 后，用 $Q_{col}$ 进行列点积检索：
   $$\text{Scores}_{memory} = \frac{Q_{col} \cdot \mathcal{H}_{norm}}{\sqrt{d}} \in \mathbb{R}^{B \times H \times S \times L_{past}}$$
5. **软连接竞争 (Softmax Fusion)**：
   直接在序列长度和历史层深度的维度上进行字符串接（Concat）。
   $$ \text{Scores}_{all} = \text{Concat}([\text{Scores}_{token}, \text{Scores}_{memory}], \text{dim}=-1) \in \mathbb{R}^{B \times H \times S \times (S + L_{past})} $$
   然后在这一个庞大的联合维度上计算 Softmax 并应用 Dropout：
   $$ \text{Weights}_{all} = \text{Dropout}(\text{Softmax}(\text{Scores}_{all})) $$
6. **上下文还原：**
   将 $\text{Weights}_{all}$ 切割还原回 $W_{token} \in \mathbb{R}^{S \times S}$ 和 $W_{memory} \in \mathbb{R}^{S \times L_{past}}$。
   $$\text{Out}_{token} = W_{token} V$$
   $$\text{Out}_{memory} = W_{memory} \cdot \mathcal{H}$$
   $$\text{Output}_{attn} = \text{Linear}_{out}(\text{Out}_{token} + \text{Out}_{memory}) \in \mathbb{R}^{B \times S \times D}$$

算出 $\text{Output}_{attn}$ 后，**不再与 $\hat{x}_l$ 相加**，而是直接被加入全局历史：$\mathcal{H}\text{.append}(\text{Output}_{attn})$。

---

## 4. 全局调度流程

了解了核心后，模型的完整生命周期如下：

> [!NOTE]
> **📝 全局调度与生命周期推演 (TODO)**
> - [ ] **这里没解释清晰，每个函数到底是什么功能？**
>       *(解答：`_attn_res_dual_axis_mix` 专职负责“浅层调取与信息拼盘搜集”，`DualAxisMemoryAttention` 专职负责核心深度非线性计算。双模块循环交替。)*
> - [ ] **最后一步预测的时候可能退化一点反而更好，第4步最后兜底其实可能会降低模型能力，考虑使用常规 LLM 最后的做法来做？**
>       *(解答：是的，主流通用架构往往抛弃最后的深度全回顾层。在 LLM 极后段直接取最后一次输出后连结 LM_Head，反而能保证文字判别维度的纯粹。)*
> - [ ] **考虑起始直接用归一化的 $x_0$。**

1. 设 $\mathcal{H} = []$, $x_0 = \text{Embed}(I)$。
2. 循环 $L$ 次，针对每一层 $i \in [0, L-1]$:
   * **注意力预备or信息聚合**：
     $in_{attn} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{attn\_res}^{i}, W_{attn\_row}^{i})$
   * **注意力计算**：
     $out_{attn} = \text{DualAxisMemoryAttention}(\text{LayerNorm}(in_{attn}), \text{past\_states}=\mathcal{H})$
   * 追加记录： $\mathcal{H}\text{.append}(out_{attn})$
   
   * **前馈预备or信息聚合**：
     $in_{mlp} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{mlp\_res}^{i}, W_{mlp\_row}^{i})$
   * **前馈计算** (标准的扩张四倍再收缩回去的 GELU MLP)：
     $out_{mlp} = \text{MLP}(\text{LayerNorm}(in_{mlp}))$
   * 追加记录： $\mathcal{H}\text{.append}(out_{mlp})$

3. **最终输出预测**：
4. 
   此时 $\mathcal{H}$ 中已经积攒了 $2 \times L$ 个状态层数据。
   * 进行最后一次收拢混合：
     $out_{final} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{final\_res}, W_{final\_row})$
   * 过归一化层并输出到词汇表：
     $\text{Logits} = \text{Linear}_{LM\_Head}(\text{LayerNorm}(out_{final}))$
   *(注：当 `tie_weights=True` 时，$\text{Linear}_{LM\_Head}$ 的权重与最初的 Token Embedding 矩阵绑定。)*

---

## 5. 当前代码对照后的问题清单与优化建议

> [!IMPORTANT]
> 本节不是抽象讨论，而是根据当前仓库中的真实实现
> `src/layer_depth_attention/model.py`
> 与训练入口
> `scripts/train_wikitext_lm.py`
> 对照得到的结论。也就是说，这里记录的是“当前实现已经确认存在的问题”和“下一步建议”，不是泛泛而谈。

### 5.1 文档与实现并非严格 1:1

当前这份蓝图不能再视为“已经和代码完全一致的最终规范”，原因至少包括：

1. **`DualAxisMemoryAttention` 的 memory score/value 路径没有严格按本文当前公式实现**
   - 本文当前写法是：
     \[
     \text{Scores}_{memory} \leftarrow \mathcal{H}_{norm},
     \qquad
     \text{Out}_{memory} \leftarrow \mathcal{H}
     \]
   - 但当前代码实现实际是：
     \[
     \text{Scores}_{memory} \leftarrow \mathcal{H}_{norm},
     \qquad
     \text{Out}_{memory} \leftarrow \mathcal{H}_{norm}
     \]
   - 也就是说，归一化后的历史张量同时被用于“匹配谁”和“读出什么”，这会改变 memory 分支的算法语义。

2. **预混合深度分支原先缺少缩放项**
   - `_attn_res_mix()` 的深度打分本质上也是点积注意力，因此理论上应该除以 \(\sqrt{D}\)。
   - 当前仓库在最近一次修复中已经补上了这一项，但历史版本和部分实验结果未必包含这个修复，因此阅读旧结果时必须标明。

3. **训练脚本里的 `attn_residual/ffn_residual` 对 `dual_axis_full` 实际无效**
   - `dual_axis_full` 在 `TinyDecoderLM.forward()` 中走的是单独的主路径，不经过标准 `TransformerBlock.forward()` 的 residual 开关逻辑。
   - 因此脚本里即使显式传了 `--attn-residual on --ffn-residual on`，也不会真的改变 `dual_axis_full` 的前向结构。
   - 这会误导实验记录，后续正式实验表格中不应再把这两个开关当成 `dual_axis_full` 的有效配置项。

### 5.2 已确认的高优先级结构问题

#### 问题 A：预混合深度分支的缩放项（已修复）

预混合深度分支：

\[
\text{Scores}_{depth} = V_{norm} \cdot q_{depth}^{l}
\]

如果不除以 \(\sqrt{D}\)，随着隐藏维度增大，点积方差会增大，softmax 更容易过早变尖，导致深度路由过硬、训练不稳定。

**当前建议**
- 这一项已经作为高优先级修复加入实现。
- 后续所有正式 `dual_axis_full` 对照实验应以“带缩放”的版本为准。

#### 问题 B：`DualAxisMemoryAttention` 的 score/value 没有拆路径（已修复）

当前更合理的写法应为：

\[
\mathcal{H}_{norm} = \text{LayerNorm}(\mathcal{H})
\]

\[
\text{Scores}_{memory} =
\frac{Q_{col} \cdot \mathcal{H}_{norm}}{\sqrt d}
\]

\[
\text{Out}_{memory} = W_{memory} \cdot \mathcal{H}
\]

而不是：

\[
\text{Scores}_{memory} \leftarrow \mathcal{H}_{norm},
\qquad
\text{Out}_{memory} \leftarrow \mathcal{H}_{norm}
\]

**为什么这是大问题**
- `Scores_memory` 决定“看谁”
- `Out_memory` 决定“看到什么”
- 如果两者都直接使用归一化后的历史，就相当于把 history 的内容本身也洗平了
- 这样 memory 分支更稳定，但也更可能损失原始历史状态中的幅值与内容差异

**当前建议**
- 后续应做一组明确对照：
  - `score 用 H_norm, value 用 H_norm`
  - `score 用 H_norm, value 用 H`
- 这组对照的目的不是微调数值，而是确认当前 memory 路径的算法语义到底该怎么定。

#### 问题 C：`embedding` 永久进入深度候选池

当前 `_attn_res_mix()` 每次都会把：

\[
[x_0, \mathcal{H}[0], \dots, \mathcal{H}[-1]]
\]

全部堆起来参与 softmax。

这意味着最初的 `embedding` 在每一层、每个子层、最终输出时，都会永远作为一个固定候选项参与深度选择。

**潜在风险**
- 训练早期模型更容易反复回退到 `x_0`
- 深层表征难以逐步占据主导
- 在更大数据集和更深网络下，可能拖慢真正的深层信息积累

**当前建议**
- 将其列为高优先级结构对照项：
  1. 永远保留 `x_0`
  2. 只在早期若干层保留 `x_0`
  3. 最终去掉 `x_0` 常驻候选

#### 问题 D：模型可能无法显式区分浅层历史与深层历史

当前 `dual_axis_full` 在做深度混合或列向 memory 检索时，历史状态主要以张量堆叠的形式进入计算：

\[
[x_0, \mathcal{H}[0], \mathcal{H}[1], \dots]
\]

但这些历史项本身并没有额外附带一个显式的“深度身份标记”或“层索引编码”。因此当前实现存在一个**待验证的风险点**：

> 模型在注意力计算时，可能并不能稳定地区分“这是来自浅层的历史状态”还是“这是来自深层的历史状态”。

这件事目前**还不能直接下结论**，因为存在两种可能：

1. **可能确实是问题**
   - 如果不同层的历史状态经过归一化或投影后变得过于相似，那么模型可能更难学会“应该偏向浅层还是深层”。
   - 这种情况下，显式加入层索引编码、深度偏置或分层参数，可能会改善深度检索质量。

2. **也可能不是问题**
   - 不同层的表征本身就可能已经带有足够的统计差异，模型未必需要额外的深度标签也能学会区分。
   - 如果是这样，额外加入 depth encoding 反而可能增加复杂度而收益有限。

**当前建议**
- 先把这一点作为“未证实但值得专门验证的设计风险”记录下来，不要当成已经确认的问题。
- 后续如果要做对照，可考虑：
  1. 维持当前无显式深度标记的版本
  2. 给历史状态加入简单的层索引编码或深度偏置
  3. 对比两者在 `dual_axis_full` 上的收敛速度与最终指标

也就是说，这一项当前的正确定位应是：

> **潜在结构问题 / 值得做 ablation 的假设**

而不是：

> **已经确认成立的实现缺陷**

### 5.3 当前更合理的训练与记录建议

1. **正式对照时应固定评估口径**
   - 若要与旧结果公平对比，`eval_batches` 需保持一致。
   - 更大的 `eval_batches` 虽然能降低偶然性，但会显著增加总训练耗时。

2. **长训练时记录累计用时**
   - 当前训练脚本已经补充 `elapsed_seconds` 和 `elapsed_minutes`。
   - 后续所有正式 `dual_axis_full` 长训练结果都应同时记录：
     - `step`
     - `train_loss`
     - `val_loss`
     - `val_ppl`
     - `elapsed_minutes`

3. **SwanLab 状态不能当作训练是否成功的唯一标志**
   - 之前已经出现过 SSL 正常时训练成功、SSL 异常时监控被禁用但训练仍然继续的情况。
   - 因此正式记录时，应同时检查：
     - 远端 python 进程
     - artifact JSON / checkpoint
     - SwanLab 是否成功连接

### 5.4 当前推荐的修复优先级

若按“最小改动、最大信息量”的原则，推荐顺序如下：

1. **先修 `_attn_res_mix()` 的缩放项**
   - 这是最小且理论最明确的修复。

2. **再拆 `DualAxisMemoryAttention` 的 score/value 路径**
   - 让 `H_norm` 只用于算分数，`Out_memory` 使用原始 `H`。

3. **最后再研究 `embedding` 是否应永久常驻候选池**
   - 这是更偏架构选择的问题，改动影响会更大，适合在前两项稳定后再做。

### 5.5 当前文档状态声明

因此，本文档当前更适合被理解为：

> `dual_axis_full` 的设计蓝图 + 当前实现偏差说明 + 下一步修复路线图

而不是：

> 已经和代码完全一致的最终算法规范

后续如果完成上述关键修复并重新验证，应再整理出一版真正可称为“实现对齐版”的正式方法文档。
