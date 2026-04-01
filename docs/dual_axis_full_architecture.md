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
针对包括初始特征 $x_0$ 以及 $\mathcal{H}$ 中所有历史输出的合集。
1. **堆叠所有层**:
   $V_{stack} = [x_0, \mathcal{H}[0], \dots, \mathcal{H}[-1]] \in \mathbb{R}^{B \times S \times M \times D}$  （其中 $M$ 是历史元素的总数）
2. **RMSNorm**: 沿着维度 $D$ 做归一化。
   $V_{norm} = \text{RMSNorm}(V_{stack})$
3. **深度加权注意力**: 使用独立的查询向量 $q_{depth}^{l}$ 去与所有 $M$ 个特征计算点积打分。
   $$\text{Scores}_{depth} = V_{norm} \cdot q_{depth}^{l} \in \mathbb{R}^{B \times S \times M}$$
   $$\text{Weights}_{depth} = \text{Softmax}_{dim=2}(\text{Scores}_{depth})$$
   通过加权求和得出最终的跨层残差平滑值：
   $$\text{Context}_{depth} = \sum_{m=1}^{M} \text{Weights}_{depth}[..., m] \cdot V_{stack}[..., m, :] \in \mathbb{R}^{B \times S \times D}$$

### 步骤 2.4 最终混合输出
$$\hat{x}_l = \text{Context}_{row} + \text{Context}_{depth} \in \mathbb{R}^{B \times S \times D}$$

---

## 3. 双端完全体子层 (`DualAxisMemoryAttention`)

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

1. 设 $\mathcal{H} = []$, $x_0 = \text{Embed}(I)$。
2. 循环 $L$ 次，针对每一层 $i \in [0, L-1]$:
   * **注意力预备**：
     $in_{attn} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{attn\_res}^{i}, W_{attn\_row}^{i})$
   * **注意力计算**：
     $out_{attn} = \text{DualAxisMemoryAttention}(\text{LayerNorm}(in_{attn}), \text{past\_states}=\mathcal{H})$
   * 追加记录： $\mathcal{H}\text{.append}(out_{attn})$
   
   * **前馈预备**：
     $in_{mlp} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{mlp\_res}^{i}, W_{mlp\_row}^{i})$
   * **前馈计算** (标准的扩张四倍再收缩回去的 GELU MLP)：
     $out_{mlp} = \text{MLP}(\text{LayerNorm}(in_{mlp}))$
   * 追加记录： $\mathcal{H}\text{.append}(out_{mlp})$

3. **最终输出预测**：
   此时 $\mathcal{H}$ 中已经积攒了 $2 \times L$ 个状态层数据。
   * 进行最后一次收拢混合：
     $out_{final} = \text{\_attn\_res\_dual\_axis\_mix}(x_0, \mathcal{H}, q_{final\_res}, W_{final\_row})$
   * 过归一化层并输出到词汇表：
     $\text{Logits} = \text{Linear}_{LM\_Head}(\text{LayerNorm}(out_{final}))$
   *(注：当 `tie_weights=True` 时，$\text{Linear}_{LM\_Head}$ 的权重与最初的 Token Embedding 矩阵绑定。)*
