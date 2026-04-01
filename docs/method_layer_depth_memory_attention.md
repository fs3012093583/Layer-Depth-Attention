# Layer-Depth Memory Attention



---

#  方法定义（Layer-Depth Memory Attention）
---
## 1. 记号说明
- 模型层数or 注意力层数：\(L\) 
- 序列长度：\(n\)
- 隐藏维度：\(d\)
- 注意力头数：\(H\)
- 每个 head 的维度：\(d_h = d / H\)

第 \(i\) 层第 \(k\) 个 token状态 表示为：

\[
x_k^{(i)} \in \mathbb{R}^d
\]

# 2. Multi-Head Query/ Key / Value

对第 \(h\) 个 attention head，第 \(k\) 个 token 在第 \(i\) 层的 query、key、value 定义为：

\[
k_{k,h}^{(i)} = W_{K,h}^{(i)} x_k^{(i)}, \quad W_{K,h}^{(i)} \in \mathbb{R}^{d_h \times d}
\]
\[
q_{k,h}^{(i)} = W_{Q,h}^{(i)} x_k^{(i)}, \quad W_{Q,h}^{(i)} \in \mathbb{R}^{d_h \times d}
\]
\[
v_{k,h}^{(i)} = W_{V,h}^{(i)} x_k^{(i)}, \quad W_{V,h}^{(i)} \in \mathbb{R}^{d_h \times d}
\]

其中 \(h = 1,2,\dots,H\)。所有 head 的输出最终通过输出投影矩阵
\[
W_O^{(i)} \in \mathbb{R}^{d \times d}
\]
映射回原始隐藏空间。

## 4. Token 级深度记忆机制（核心）

对每个 token \(j\)到模型的第i层，维护其跨层历史：

### Key memory

\[
\mathcal{K}_{j,h}^{(i)} =
\{k_{j,h}^{(1)}, k_{j,h}^{(2)}, \dots, k_{j,h}^{(i-1)}\}
\]

### Value memory

\[
\mathcal{V}_{j,h}^{(i)} =
\{v_{j,h}^{(1)}, v_{j,h}^{(2)}, \dots, v_{j,h}^{(i-1)}\}
\]

## 5. 扩展注意力空间（关键结构）

在第 \(i\) 层，对每个 head \(h\) 构造扩展 Key/Value 空间。

### Key

由两部分组成：

- 当前层 token keys（\(n\) 个）
- 历史 memory keys（当前查询 token \(k\) 在前 \(i-1\) 层的 keys）


因此，对于第 \(h\) 个 head，有
\[
K_{k,h}^{(i)} =
\left[
k_{1,h}^{(i)}, k_{2,h}^{(i)}, \dots, k_{n,h}^{(i)},
k_{k,h}^{(1)}, k_{k,h}^{(2)}, \dots, k_{k,h}^{(i-1)}
\right] \in \mathbb{R}^{(n+i-1)\times d_h}
\]
\[
|K_{k,h}^{(i)}| = n + (i - 1)
\]

同理：

\[
V_{k,h}^{(i)} =
\left[
v_{1,h}^{(i)}, v_{2,h}^{(i)}, \dots, v_{n,h}^{(i)},
v_{k,h}^{(1)}, v_{k,h}^{(2)}, \dots, v_{k,h}^{(i-1)}
\right] \in \mathbb{R}^{(n+i-1)\times d_h}
\]
\[
|V_{k,h}^{(i)}| = n + (i - 1)
\]

## 6. 注意力计算

### Attention 权重

对于第 \(i\) 层第 \(k\) 个 query token，引入对应掩码向量
\[
m_k^{(i)} \in \mathbb{R}^{n+i-1}
\]
用于约束扩展注意力空间中的可见位置。其中：

- 对当前层 token 部分，采用标准 causal mask，即只允许关注位置 \(1,\dots,k\)
- 对深度 memory 部分，默认全部可见，因为它们来自同一 token 在前面各层的历史表示

更具体地，记 \(m_k^{(i)}\) 的第 \(r\) 个分量为 \(m_{k,r}^{(i)}\)，则有
\[
m_{k,r}^{(i)} =
\begin{cases}
0, & 1 \le r \le k, \\
-\infty, & k < r \le n, \\
0, & n < r \le n+i-1.
\end{cases}
\]

其中：

- 当 \(1 \le r \le n\) 时，\(m_{k,r}^{(i)}\) 对应当前层第 \(r\) 个 token 的 key
- 当 \(n < r \le n+i-1\) 时，\(m_{k,r}^{(i)}\) 对应 depth memory 中的 \(k_{k,h}^{(r-n)}\)

因此，第 \(h\) 个 head 的注意力分数应写为：

\[
s_{k,h}^{(i)} =
\frac{
q_{k,h}^{(i)} (K_{k,h}^{(i)})^\top
}{
\sqrt{d_h}
} + m_k^{(i)}
\]

\[
\alpha_{k,h}^{(i)} =
\mathrm{softmax}
\left(s_{k,h}^{(i)}\right)
\]

### 输出

\[
\mathrm{head}_{k,h}^{(i)} =
\alpha_{k,h}^{(i)} V_{k,h}^{(i)}
\]

\[
\mathrm{Attn}_k^{(i)} =
\mathrm{Concat}
\left(
\mathrm{head}_{k,1}^{(i)},
\mathrm{head}_{k,2}^{(i)},
\dots,
\mathrm{head}_{k,H}^{(i)}
\right) W_O^{(i)}
\]

其中掩码 \(m_k^{(i)}\) 的非法位置取值为 \(-\infty\)，合法位置取值为 \(0\)。这里 depth memory 不受 causal 约束，因为它们不来自未来 token，而是来自当前 query token 在前面各层的历史表示。

## 7. 结构解释（重要）

扩展后的 attention 空间包含两类信息：

### （1）当前层 token（标准 attention）

- 数量：\(n\)
- 结构：标准 causal self-attention

### （2）历史 memory token（depth memory）

- 数量：\(i - 1\)
- 来源：同一 token 在前面各层的 projection

## 8. 关键性质

### 动态 attention 空间

\[
\mathcal{D}^{(i)} = \mathcal{T} \cup \mathcal{H}^{(i)},
\quad
|\mathcal{D}^{(i)}| = n + i - 1
\]

其中：

- \(\mathcal{T}\)：当前 token 空间
- \(\mathcal{H}^{(i)}\)：深度历史 memory

## 9. 核心直觉总结

该方法将 Transformer 从：

> “仅在 token 维度进行注意力计算”

扩展为：

> “在 token + depth memory 联合空间中进行注意力计算”

## 10. 一句话定义（论文可用）

> 本方法通过引入 token 级跨层记忆，使注意力机制从静态 token 交互扩展为随深度增长的动态记忆注意力空间。

---

## 附：当前实现中的 `value_reproj` 变体

为了保持实现可运行且避免对历史 hidden states 做高开销的重复投影，当前代码中的 `value_reproj` 并不直接使用
\[
W_{K,h}^{(i)} x_k^{(\ell)}, \qquad W_{V,h}^{(i)} x_k^{(\ell)}
\]
来构造历史 memory，而是将前面各层同位置 token 的历史 value 视为 memory feature，再通过当前层的 key/value 投影子矩阵进行一次适配变换。

具体地，设第 \(\ell\) 层该 token 的历史 value 为
\[
v_{k,h}^{(\ell)} \in \mathbb{R}^{d_h}, \qquad \ell < i.
\]
将所有 head 的历史 value 拼回隐藏空间后，记为
\[
z_k^{(\ell)} =
\mathrm{Concat}\left(
v_{k,1}^{(\ell)}, \dots, v_{k,H}^{(\ell)}
\right) \in \mathbb{R}^{d}.
\]

在第 \(i\) 层，使用当前层的 key/value 投影生成适配后的 memory entries：
\[
\tilde{k}_{k,h}^{(\ell \to i)} = A_{K,h}^{(i)} z_k^{(\ell)}, \qquad
\tilde{v}_{k,h}^{(\ell \to i)} = A_{V,h}^{(i)} z_k^{(\ell)},
\]
其中
\[
A_{K,h}^{(i)} \in \mathbb{R}^{d_h \times d}, \qquad
A_{V,h}^{(i)} \in \mathbb{R}^{d_h \times d}.
\]

因此，实现中的扩展 memory bank 写为
\[
K_{k,h}^{(i)} =
\left[
k_{1,h}^{(i)}, \dots, k_{n,h}^{(i)},
\tilde{k}_{k,h}^{(1 \to i)}, \dots, \tilde{k}_{k,h}^{(i-1 \to i)}
\right],
\]
\[
V_{k,h}^{(i)} =
\left[
v_{1,h}^{(i)}, \dots, v_{n,h}^{(i)},
\tilde{v}_{k,h}^{(1 \to i)}, \dots, \tilde{v}_{k,h}^{(i-1 \to i)}
\right].
\]

这里的关键点是：`value_reproj` 应被理解为一种 **memory adapter / reprojection** 机制，而不是严格意义上的“用当前层 \(W_K^{(i)}, W_V^{(i)}\) 对旧层 hidden state 重新投影”。这样定义与当前实现一致，同时避免了对所有历史 hidden states 做高成本重复计算。
