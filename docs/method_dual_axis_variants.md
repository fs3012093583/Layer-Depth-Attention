# Dual-Axis 方法整理

本文档集中整理以下三种方法的数学实现，供后续论文继续扩写：

- `dual_axis_memory`
- `attn_residuals_dual_axis`
- `dual_axis_full`

后续可直接在本文档中继续补充：

- 方法动机
- 伪代码
- 复杂度分析
- 参数量对比
- 实验表格
- 论文段落

## 1. 符号约定

设：

- 序列长度为 `T`
- 网络层数为 `L`
- 隐藏维度为 `d`
- 注意力头数为 `H`
- 单头维度为 `d_h = d / H`

记：

- `x_l in R^{T x d}`：第 `l` 层 block 输入
- `h_l in R^{T x d}`：第 `l` 层 block 输出
- `x_{l,t} in R^d`：第 `l` 层第 `t` 个 token 的表示
- `LN(.)`：LayerNorm
- `MLP(.)`：前馈网络

标准 causal self-attention 写为：

`Q_l = LN(x_l) W_l^Q`

`K_l = LN(x_l) W_l^K`

`V_l = LN(x_l) W_l^V`

`R_l^{row} = Softmax((Q_l K_l^T / sqrt(d_h)) + M_causal) V_l`

其中 `M_causal` 为上三角因果 mask。

跨层同列历史记忆记为：

`C_{l,t} = {h_{1,t}, h_{2,t}, ..., h_{l-1,t}}`

也就是说，对位置 `t` 来说，同列历史是该 token 在前面所有层的输出。

---

## 2. `dual_axis_memory`

### 2.1 方法定义

`dual_axis_memory` 只修改 block 内的主注意力路径，不修改 residual-attention 聚合方式。

它把第 `l` 层的注意力拆成两条支路：

- 行内支路：同层 token 前缀上的标准 causal attention
- 同列支路：同 token 位置跨层历史上的 depth attention

### 2.2 数学形式

给定第 `l` 层输入 `x_l`，先归一化：

`X_l = LN(x_l)`

#### 行内支路

`Q_l^{row} = X_l W_l^Q`

`K_l^{row} = X_l W_l^K`

`V_l^{row} = X_l W_l^V`

`R_l^{row} = Softmax((Q_l^{row} (K_l^{row})^T / sqrt(d_h)) + M_causal) V_l^{row}`

#### 同列支路

定义单独的同列查询：

`Q_l^{col} = X_l W_l^{Q,col}`

对每个位置 `t`，构造跨层记忆库：

`K_{l,t}^{col} = V_{l,t}^{col} = [Norm(h_{1,t}), Norm(h_{2,t}), ..., Norm(h_{l-1,t})]`

即当前实现里同列分支直接令：

`K = V = x`

只是对前层输出做归一化后直接作为记忆，不再额外学习列向 `W^K` / `W^V`。

于是：

`R_{l,t}^{col} = Softmax(Q_{l,t}^{col} (K_{l,t}^{col})^T / sqrt(d_h)) V_{l,t}^{col}`

把所有位置拼回后得到 `R_l^{col}`。

#### 融合

两条支路相加后再经过输出投影：

`A_l = (R_l^{row} + R_l^{col}) W_l^O`

随后沿用标准 Transformer block 残差形式：

`x_l' = x_l + A_l`

`h_l = x_l' + MLP(LN(x_l'))`

### 2.3 直观解释

该方法把“横向上下文建模”和“纵向跨层追忆”合并进主注意力路径：

- 横向支路负责读当前层序列上下文
- 纵向支路负责读同 token 的跨层历史

因此它对应“只改因果注意力”的版本。

### 2.4 参数量

在当前 `WikiText-2` 主配置下：

- `dual_axis_memory = 18,130,688`

相对 `baseline = 17,735,936`

新增参数：

- `394,752`

主要来源是每层额外的 `W^{Q,col}`。

---

## 3. `attn_residuals_dual_axis`

### 3.1 方法定义

`attn_residuals_dual_axis` 保留 block 内标准 causal self-attention，不改主干 `Q/K/V` 注意力。

它只修改 residual-attention 聚合器：

- 原始 `attn_residuals` 只沿深度轴动态混合历史
- 本方法在 residual mixing 中额外加入一条行内 causal 检索支路

因此它对应“只改残差注意力”的版本。

### 3.2 原始 depth residual mixing

设 embedding 输出为 `e`，历史状态序列为：

`S = [e, z_1, z_2, ..., z_n]`

原始 residual-attention 用一个深度查询向量 `q_depth` 在这些历史上做 soft selection：

`R_depth = sum_i alpha_i z_i`

`alpha_i = Softmax(<Norm(z_i), q_depth>)`

这个 residual mixer 会被分别用于：

- attention 子层输入
- MLP 子层输入
- 最终输出聚合

### 3.3 双轴 residual mixing

本方法在 `R_depth` 之外，再额外引入一条行内 residual 检索：

`R_row = ResidualRowMix(z_last, W^{Q,row})`

其中 `z_last` 表示当前 residual mixer 能看到的最近状态。

这一项的形式可以写为：

`Q^{row} = Norm(z_last) W^{Q,row}`

`K^{row} = V^{row} = Norm(z_last)`

`R_row = Softmax((Q^{row} (K^{row})^T / sqrt(d_h)) + M_causal) V^{row}`

最后把横向和纵向 residual 检索直接相加：

`R_dual = R_row + R_depth`

### 3.4 子层输入形式

于是：

`u_l^{attn} = R_dual^{attn}`

`u_l^{mlp} = R_dual^{mlp}`

最终输出前也做同样的双轴 residual mixing：

`u^{final} = R_dual^{final}`

而每层 block 本身仍然是标准形式：

`a_l = Attn(LN(u_l^{attn}))`

`m_l = MLP(LN(u_l^{mlp}))`

`history <- history U {a_l, m_l}`

`y = LMHead(LN(u^{final}))`

### 3.5 直观解释

该方法没有碰主干 attention，只是把 residual-attention 从“纯 depth 检索”扩成了“row + depth 双轴检索”。

因此它更像是在修改“信息聚合/残差路由”机制，而不是修改标准 self-attention 核本身。

### 3.6 参数量

在当前 `WikiText-2` 主配置下：

- `attn_residuals_dual_axis = 18,594,560`

相对 `baseline = 17,735,936`

新增参数：

- `858,624`

主要来源是：

- attention residual mixer 的 `W^{Q,row}`
- MLP residual mixer 的 `W^{Q,row}`
- final residual mixer 的 `W^{Q,row}`

---

## 4. `dual_axis_full`

### 4.1 方法定义

`dual_axis_full` 是当前主力版本。

它同时修改两条路径：

1. block 内主注意力：使用 `dual_axis_memory`
2. residual-attention 聚合：使用 `attn_residuals_dual_axis`

因此它是“因果注意力和残差注意力都改”的 full dual-axis 版本。

### 4.2 数学形式

第 `l` 层的主干 attention 不再是标准 causal attention，而是：

`A_l = DualAxisMemory(x_l, h_{<l})`

其中 `DualAxisMemory(.)` 即第 2 节定义的双轴主注意力。

同时，residual-attention 输入也不再是纯 depth mixing，而是：

`u_l^{attn} = ResidualDualAxisMix(e, history, q_l^{attn}, W_l^{Q,row,attn})`

`u_l^{mlp} = ResidualDualAxisMix(e, history, q_l^{mlp}, W_l^{Q,row,mlp})`

其中：

`ResidualDualAxisMix = ResidualRowMix + ResidualDepthMix`

于是每层计算为：

`a_l = DualAxisMemory(LN(u_l^{attn}), history_states)`

`m_l = MLP(LN(u_l^{mlp}))`

`history <- history U {a_l, m_l}`

最终输出：

`u^{final} = ResidualDualAxisMix(e, history, q^{final}, W^{Q,row,final})`

`y = LMHead(LN(u^{final}))`

### 4.3 直观解释

如果说：

- `dual_axis_memory` 是主干注意力双轴化
- `attn_residuals_dual_axis` 是残差路由双轴化

那么 `dual_axis_full` 就是两者叠加。

这使它具有最强的双轴归纳偏置，但也最容易带来：

- 分支间干扰
- 优化难度上升
- 泛化不稳定

### 4.4 参数量

在当前 `WikiText-2` 主配置下：

- `dual_axis_full = 18,989,312`

相对 `baseline = 17,735,936`

新增参数：

- `1,253,376`

相对增幅约为：

- `7.07%`

---

## 5. 三种方法统一对比

| 方法 | 改动位置 | 主干 attention | residual mixing | 参数量 |
|---|---|---|---|---:|
| `dual_axis_memory` | 只改主干 | 双轴 | 原始 | 18,130,688 |
| `attn_residuals_dual_axis` | 只改 residual | 原始 | 双轴 | 18,594,560 |
| `dual_axis_full` | 两者都改 | 双轴 | 双轴 | 18,989,312 |

也可以更口语化地表述为：

- `dual_axis_memory`
  只改因果注意力
- `attn_residuals_dual_axis`
  只改残差注意力
- `dual_axis_full`
  因果注意力和残差注意力都改

---

## 6. 复杂度讨论

### 6.1 `dual_axis_memory`

若只看单层：

- 行内支路复杂度约为 `O(T^2 d)`
- 同列支路复杂度约为 `O(L T d)`

因此它可以看作一种二维检索的分解形式，而不是完整二维全注意力。

### 6.2 `attn_residuals_dual_axis`

其 residual dual-axis mixing 主要包含：

- depth residual mixing：`O(L T d)`
- row residual mixing：`O(T^2 d)`

因此其额外开销主要集中在 residual 聚合器，而非主干 QKV 路径。

### 6.3 `dual_axis_full`

由于它同时在：

- 主干 attention
- residual mixing

两条路径上都引入双轴检索，因此理论与实践开销都最大。

---

## 7. 当前可直接扩写成论文的段落方向

### 7.1 方法动机

- 标准 causal attention 只沿 token 维度检索
- 标准残差连接是固定加法，缺少动态追忆
- `attn_residuals` 只沿 depth 维度动态选择历史
- 本工作尝试把 token 轴与 depth 轴统一进动态检索框架

### 7.2 三个变体的研究意义

- `dual_axis_memory`
  用来回答：双轴机制放进主干 attention 是否有益
- `attn_residuals_dual_axis`
  用来回答：双轴机制放进 residual aggregation 是否有益
- `dual_axis_full`
  用来回答：两条路径同时双轴化是否进一步增强

### 7.3 当前实验现象的理论解释方向

- dual-axis 可能对 retrieval-style 任务更友好
- 在标准语言建模上，纵向历史可能干扰正常层级抽象
- full 版本的归纳偏置最强，但优化扰动也最大

---

## 8. 后续建议补充内容

建议后续直接在本文档继续补：

1. 伪代码
2. 参数量与复杂度表
3. `seq_len=128` 与 `seq_len=512` 实验对比表
4. associative recall 与 WikiText 的对照讨论
5. 方法结构图
6. 论文中的 Method 与 Discussion 小节
