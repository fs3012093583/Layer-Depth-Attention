# Layer-Depth Memory Attention

下面给出该方法的严格中文 Markdown 论文版表达，可直接整理到论文的 Method 部分。

---

# 🧠 方法定义（Layer-Depth Memory Attention）

---

## 1. 记号说明

设：

- 模型层数：\(L\)
- 序列长度：\(n\)
- 隐藏维度：\(d\)

第 \(i\) 层第 \(k\) 个 token 表示为：

\[
x_k^{(i)} \in \mathbb{R}^d
\]

## 2. Query（不随深度变化）

Query 保持标准 Transformer 形式，不引入层依赖：

\[
q_k^{(i)} = W_Q x_k^{(i)}, \quad W_Q \in \mathbb{R}^{d \times d}
\]

## 3. Key / Value（层内生成）

Key 与 Value 仍由当前层表示生成：

\[
k_j^{(i)} = W_K x_j^{(i)}, \quad v_j^{(i)} = W_V x_j^{(i)}
\]

其中：

- \(W_K, W_V \in \mathbb{R}^{d \times d}\)
- 在所有层之间共享

## 4. Token 级深度记忆机制（核心）

对每个 token \(j\)，维护其跨层历史：

### Key memory

\[
\mathcal{K}_j^{(i)} =
\{k_j^{(1)}, k_j^{(2)}, \dots, k_j^{(i)}\}
\]

### Value memory

\[
\mathcal{V}_j^{(i)} =
\{v_j^{(1)}, v_j^{(2)}, \dots, v_j^{(i)}\}
\]

## 5. 扩展注意力空间（关键结构）

在第 \(i\) 层，构造扩展 Key/Value 空间。

### Key

由两部分组成：

- 当前层 token keys（\(n\) 个）
- 历史 memory keys（每个 token 的 \(i - 1\) 层）

因此：

\[
|K^{(i)}| = n + (i - 1)
\]

同理：

\[
|V^{(i)}| = n + (i - 1)
\]

## 6. 注意力计算

### Attention 权重

\[
\alpha_k^{(i)} =
\mathrm{softmax}
\left(
\frac{
q_k^{(i)} (K^{(i)})^\top
}{
\sqrt{d}
}
\right)
\]

### 输出

\[
\mathrm{Attn}_k^{(i)} =
\alpha_k^{(i)} V^{(i)}
\]

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
