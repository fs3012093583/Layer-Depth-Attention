# 基于层深度路由的 Transformer 注意力机制研究

**作者**  
范顺（fanshun@stu.scu.edu.cn）  
胡超浪（通讯作者，huchaolang@scu.edu.cn）

## 摘要

Transformer 在 token 维度上的信息传播已经非常高效，但对于同一位置在更早层形成的表示，当前层仍然只能通过残差路径间接访问。本文研究这样一个问题：当前层的一个 token 是否能够通过直接读取自身的层深度历史，而不仅仅依赖堆叠结构中的重复残差传播，从而获得收益。为此，我们提出一种层深度注意力路由机制，在标准自注意力之外增加一条同位置跨层记忆分支。本文在 decoder-only 语言模型中对该思想进行了实例化与评估。当前实现采用单一共享 query，同步执行行方向 token 注意力与深度方向记忆检索，并且每个 block 归档一组投影后的历史 K/V 表示。在 WikiText-103 上，针对 8 层与 16 层模型、不同序列长度以及较长训练预算的多组设置，本文方法相对于强 Transformer 基线在最终测试困惑度上取得了 1.3\% 到 3.1\% 的稳定提升。注意力分配分析进一步表明，深层会为 depth 分支分配显著注意力质量，并从少量高价值的早层记忆槽位中进行选择性检索，而不是忽略该额外路径。此外，row 分支与 depth 分支之间的相对注意力分配在不同 token 位置上整体保持稳定，这表明随着序列变长，新增的深度路径并不会自然失效。

## 1. 引言

Transformer 语言模型的核心建立在 token 级注意力之上，即每一层都可以从序列中其他位置读取上下文信息。一个有用的理解方式是，将该结构视为一个在 token 位置与网络层深度上展开的二维计算图。在这种视角下，循环模型主要依赖链式局部状态传递进行信息传播，而自注意力则通过允许 token 在单层内直接访问其前缀，从而显著缩短了 token 轴上的通信路径。然而，同样的视角也暴露出一个仍未解决的不对称性：在标准 decoder-only Transformer 中，同一 token 在更早层形成的表示，仍然只能被当前层间接访问。

残差连接确实提供了一条沿深度方向传播信息的路径，因此标准 Transformer 并非完全缺乏跨层通信能力。但这条路径本质上是统一、相加式的，而不是选择性的：早层信息是通过残差流持续向后混合，而不是在当前 token 需要时被显式检索。随着层数加深，有价值的早层信号可能在后续多次 attention 与 FFN 变换中被逐渐稀释。这就引出一个自然问题：如果在标准 token 注意力之外，额外加入一条直接的层深度记忆分支，decoder-only 语言模型是否能够因此受益？如果当前 token 能够同时对当前层的 row 方向上下文以及自身的跨层历史进行联合注意力，那么模型可能更有效地复用部分已形成、但尚未被完全替代的中间表示，而不必在整个层堆叠中反复重建这些信息。

这一问题与若干已有方向相关，但并不相同。一类工作通过扩展跨 segment 的记忆或缓存机制来提升长上下文建模能力，例如 Transformer-XL；另一类工作重新思考深度方向上的重复计算组织方式，例如 Universal Transformer；还有一些工作则讨论跨层聚合、残差增强或深度路由机制。本文的设定比这些方向更窄：我们并不重新设计整体递归结构，也不研究参数共享本身，而是仅仅在一个 otherwise standard 的 decoder 堆叠中，显式暴露一条“当前 token 访问其同位置早层历史”的路径。

本文围绕这一问题，构建并比较了一组用于 decoder-only Transformer 的跨层记忆机制。当前最终版本采用单 query 路由与投影后的历史 K/V 归档，使得每一步注意力从纯 token 交互扩展为一个联合的 token-depth 路由问题。从计算图视角来看，该机制在保留 token 轴短通信路径的同时，引入了一条更直接的 depth 轴访问路径。

本文方法带来的收益并不夸张，但具有稳定性。在 WikiText-103 的多组设置下，包括 8 层与 16 层模型、不同序列长度以及较长训练预算，本文方法相对于强 Transformer 基线在最终测试困惑度上取得了 1.3\% 到 3.1\% 的提升。同时，该方法也引入了非平凡的运行时开销，且实测 slowdown 大于纯理论计算量差距，说明当前实现更接近一个结构原型，而不是一个已完全优化的系统版本。

因此，本文支持一个克制但明确的结论：对于 decoder-only 语言建模而言，显式访问 depth history 是有价值的，即使在已经较强的基线上，它仍然能够带来可重复的改进；但与此同时，该机制目前更适合被视为一种有前景的结构方向，而不是标准自注意力的直接替代。

### 贡献

1. 将 decoder-only 注意力表述为一个**联合 row-depth 路由问题**，使当前 token 不仅可以读取前缀 token，也可以读取自身跨层历史。
2. 提出一种基于**单 query 路由与投影历史 K/V** 的实用设计，并证明显式的 depth-history 检索能够稳定优于强 Transformer 基线。
3. 在 WikiText-103 上展示了该方法在多种设置下带来的**1.3\% 到 3.1\% 的最终测试困惑度相对改进**。
4. 提供了**注意力分配层面的证据**，表明 depth 分支在中后层被积极使用，且检索集中在少量高价值历史槽位，而非退化为均匀混合。
5. 分析了该方法的代价，说明其理论附加计算量相对温和，但当前原型存在明显的系统级运行时开销，为后续优化指出方向。

## 2. 方法

### 2.1 问题设定

考虑一个具有 $L$ 层的 decoder-only Transformer。在第 $l$ 层，位置 $t$ 上的隐藏状态记为 $x_t^{(l)}$。标准因果自注意力允许 $x_t^{(l)}$ 从同层前缀位置 $\{1,\dots,t\}$ 中读取 row 方向上下文，但它并未显式暴露该 token 在更早层的跨层历史 $\{x_t^{(1)}, \dots, x_t^{(l-1)}\}$。

本文希望在标准自注意力之外增加一条 depth-memory 分支，使当前 token 可以直接检索自身在前面各层形成的同位置历史表示。

更一般地，可以把模型看成一个以隐藏状态节点 $x_t^{(l)}$ 为顶点的有向计算图。两个状态之间的通信距离，定义为信息从一个状态传播到另一个状态所需经过的最短边数。在该视角下，标准因果自注意力已经显著缩短了 token 轴上的通信距离，因为一个 token 可以在单层内直接访问其完整前缀。本文的目标，是以类似方式缩短对同位置早层状态的访问距离，因为在标准 decoder 中，这些状态主要只能通过连续层间的残差混合间接到达当前层。

这一动机在深层网络中更强。在较浅的堆叠中，早层有价值的特征还可能通过残差流较充分地传到高层；但在更深的 decoder 中，底层产生的同位置信号需要穿过更多层的 attention 与 FFN 变换，才能最终影响高层。残差连接虽然缓解了这一问题，但它是一种被动传播：它将早层信息相加式地带到后续层，而不是允许当前层在需要时主动读取某个特定早层表示。本文方法本质上正是对这一点的强化：不再只依赖 repeated residual mixing，而是允许当前层在需要时直接读取 earlier-layer 的同位置状态。后文的注意力分析也支持这一解释，因为深层对 depth 分支分配了更多注意力质量，并且往往将这些质量集中在少量非常早的历史条目上。

### 2.2 Row 注意力与 Depth 记忆

对于每一层，我们保留标准的 row 方向 token 注意力分支，同时维护一个由 earlier layers 形成的同位置 depth-memory archive。当前 token 由此面临两个路由决策：

- 面向当前层前缀 token 的 row 路由；
- 面向同位置历史记忆的 depth 路由。

这两个分支分别产生分数，然后进行拼接并通过同一个 softmax 统一归一化，使模型在一个共享竞争空间中同时决定对 token 上下文和 depth 历史的注意力分配。这个联合归一化设计是本文方法解释上的关键：模型面对的不是两个彼此独立的旁路，而是一个必须在 token-context 与 depth-history 之间分配的统一注意力预算。

这也带来了一个重要的自适应性质。如果在某一输入模式或某一模型配置下，depth 分支并不重要，那么共享 softmax 可以几乎把全部概率质量分配给 row 分支，此时模型行为就会退化回接近标准自注意力。反之，如果 depth history 有帮助，模型可以自然地把一部分概率质量转移到 depth 分支，而不需要人为规定任一轴必须获得固定比例的注意力。本文因此并不强制某个维度必须拿到一定量的分数，而是暴露两个轴上的候选源，让学习过程自行决定两者之间的分配关系。

### 2.3 本文采用的最终设计

当前方法包含两个关键设计选择。

**Single-Q。**  
row 注意力与 depth-memory lookup 使用同一个 query 表示。与 dual-query 变体相比，这样的设计更简单，也避免额外引入一条 query 分支。

**Projected K/V。**  
历史记忆通过投影后的 keys 与 values 归档，而不是直接保留原始 hidden state。在当前实现中，每个 block 只向记忆库输出一组 attention-side 的 memory entry。这样，当前层获得的是一个学习得到的 memory matching 空间，经验上比完全取消 projection 的设计更稳定。从概念上看，这一选择将两个经常被残差传播混在一起的角色分开了：一类是用于当前层持续计算的 hidden state，另一类是暴露给未来层检索的 memory representation。通过学习一个专门的 memory projection，模型可以在历史 archive 中保留更适合检索的结构，而不必强迫整个 hidden state 直接承担这一角色。

使用共享的 memory projection 也使跨层 archive 更一致。不同层的 hidden states 不一定天然处于相同的表示分布中，尤其是在深层模型中，多次 attention 与 FFN 变换会带来分布漂移。通过同一个共享投影矩阵把各层历史状态映射到统一检索空间，后续层就更容易用一致的准则比较并选择历史条目，而不是直接在一个异质的 raw hidden state 集合上做匹配。

### 2.4 层内计算

在第 $l$ 层，当前隐藏序列记为：

\[
X^{(l)} \in \mathbb{R}^{B \times S \times D}.
\]

row-attention 分支首先计算标准 token-attention 投影：

\[
[Q_{\text{row}}, K_{\text{row}}, V_{\text{row}}] = W_{\text{qkv}} X^{(l)}.
\]

在 reshape 为多头形式后，标准因果 token 注意力分数为：

\[
A_{\text{row}} = \frac{Q_{\text{row}} K_{\text{row}}^\top}{\sqrt{d}}.
\]

对于 depth-memory 分支，最终设计直接使用同一个 query：

\[
Q_{\text{depth}} = Q_{\text{row}}.
\]

memory archive 由前面各层 block 输出的历史状态构成。在当前实现中，每个 block 仅贡献一组 memory entry，因此第 $l$ 层可见的 archive 是由 earlier layers 归档的投影 K/V 组成：

\[
\mathcal{M}^{(l)} = \{(K_{\text{mem}}^{(1)}, V_{\text{mem}}^{(1)}), \ldots, (K_{\text{mem}}^{(m)}, V_{\text{mem}}^{(m)})\}.
\]

每个历史条目通过共享的 memory projection 生成：

\[
K_{\text{mem}} = W_k^{\text{mem}} H,\qquad V_{\text{mem}} = W_v^{\text{mem}} H,
\]

其中 $H$ 表示对应 block 的隐藏状态。在当前实现中，这两组 memory projection 在不同层之间共享，因此 archive 使用的是全局共享的 memory projection，而不是每层单独一套参数。

给定 memory archive，当前层通过将当前 token 的 query 与同位置历史 keys 做匹配来计算 depth-memory 分数：

\[
A_{\text{depth}} = \frac{Q_{\text{depth}} \cdot K_{\text{mem}}}{\sqrt{d}}.
\]

关键设计在于，row 方向 token scores 与 depth 方向 memory scores 会先拼接，再统一归一化：

\[
A = [A_{\text{row}}; A_{\text{depth}}],\qquad P = \operatorname{softmax}(A).
\]

这意味着模型并不是分别独立做两次注意力决策，而是在一个统一的预算下同时决定：

- 当前层 token 上下文应该获得多少注意力；
- 同位置历史 depth memory 应该获得多少注意力。

最终输出上下文是 row 分支上下文与 depth 分支上下文的和：

\[
O = O_{\text{row}} + O_{\text{depth}}.
\]

除新增的历史 K/V 投影与联合注意力分数张量外，block 的其余部分仍保持标准 Transformer 的计算形式，包括输出投影、残差路径与 FFN 结构。整体结构见图 1。

**图 1**：所提出的 layer-depth attention routing block 概览。标准的 row 方向因果注意力路径被完整保留，同时每个 block 向 layer-depth memory archive 导出一组投影后的 K/V entry。在下一层，当前 token 使用单一 query 同时对当前层 token 上下文和 earlier-layer 的同位置历史记忆进行联合路由。row scores 与 depth scores 在 shared softmax 前进行拼接，因此二者竞争同一个统一注意力预算。

### 2.5 计算复杂度与内存开销

设模型包含 $L$ 层，隐藏维度为 $D$，序列长度为 $S$，batch size 为 $B$，MLP ratio 为 $r$。标准 decoder-only Transformer 的主导计算量可以写成：

\[
C_{\text{base}} = L\bigl((4+2r)BSD^2 + 2BS^2D\bigr).
\]

在常用的 $r=4$ 设定下，上式变为：

\[
C_{\text{base}} = L\bigl(12BSD^2 + 2BS^2D\bigr).
\]

相较于 baseline，本文方法只改变了两部分计算：第一，增加了用于 depth-memory archive 的历史 K/V 投影；第二，将注意力分数张量从纯 token 路由扩展为联合的 row-depth 路由。具体来说，baseline 的 attention block 在 MLP 之外包含 $QKV$ projection 与 output projection，共贡献 $4BSD^2$。本文当前实现中，每层还会额外计算一组 memory K/V projection，因此每层再增加 $2BSD^2$，使总投影项变为 $(6+2r)BSD^2$。在当前实现下，每层只贡献一个历史 entry，因此第 $l$ 层可见的 depth-memory 槽位数为：

\[
M_l = l-1.
\]

代入层内复杂度并对所有层求和，可得整体模型复杂度的闭式形式：

\[
C_{\text{method}} \approx L\bigl((6+2r)BSD^2 + 2BS^2D\bigr) + BSD\,L(L-1).
\]

在当前 $r=4$ 设定下，上式变为：

\[
C_{\text{method}} \approx L\bigl(14BSD^2 + 2BS^2D\bigr) + BSD\,L(L-1).
\]

因此，与 baseline 相比，本文方法新增了 $2LBSD^2$ 的 projection 项，以及 $BSD\,L(L-1)$ 的 depth matching 项。关键点在于，该方法是在标准自注意力之上引入了一个中等规模但不可忽略的附加计算，而不是完全替换原有 Transformer 主干。

在内存层面，本文实验中关注的是**训练时峰值 GPU 显存**。相较于 baseline，所提方法需要额外保留历史 K/V 激活以及 depth 分支的中间张量，因此训练时显存会有所增加；但与引入一整块新的 token-token attention matrix 相比，它并不会以同样激烈的方式改变关于序列长度的增长形式。经验结果表明，峰值显存增幅相对温和，而运行时 slowdown 更明显，这说明当前原型更受系统级执行开销约束，而不仅仅是原始存储量的增加。

## 3. 实验

### 3.1 数据集与实验设置

本文主实验使用 **WikiText-103 raw-text** 数据集。所有实验统一采用 GPT-2 BPE tokenizer，并在无特别说明时使用 tied input/output embeddings。

主比较中的所有模型均基于同一套 decoder-only Transformer backbone 家族实现。除非某个消融实验明确改变 attention 机制本身，否则各方法在 hidden size、head 数、MLP ratio、位置编码以及 language-model head 结构上均保持一致。

当前已经完成、并能够构成连贯对比的主结果设置包括：

- 8 层，`seq_len=256`，训练 40000 与 80000 steps；
- 16 层，`seq_len=256`，训练 80000 steps；
- 8 层，`seq_len=512`，训练 80000 steps；
- projected K/V 与非 projected K/V 的对比。

多数实验使用如下默认配置：

- `d_model=384`
- `num_heads=8`
- `mlp_ratio=4`
- tied embeddings
- learned positional embeddings

这使得本文实验位于与以往 WikiText-103 decoder-only 语言模型研究相近的比较口径中。

训练过程中定期在验证集上评估，而测试集只在训练结束后评估一次。因此，本文统一报告 **Best Val PPL** 与 **Final Test PPL**，而不是 best test perplexity。每项实验结果取三次运行的中值；由于波动很小，主表中未单独列出方差列。

### 3.2 主结果

在目前已经完成的所有主要设置中，最终的 **single-q + projected K/V** 方法都稳定优于标准 Transformer baseline，无论是在 best validation perplexity 还是 final test perplexity 上。

在 8 层、`seq_len=256` 设置下，方法在 40000 steps 时将最终测试困惑度从 **34.53** 降低到 **33.53**；在 80000 steps 时，从 **28.47** 降低到 **27.58**。在 16 层、`seq_len=256` 设置下，最终测试困惑度进一步从 **26.28** 降低到 **25.55**。在更长上下文长度下，这一收益依然存在：在 8 层、`seq_len=512` 设置下，最终测试困惑度从 **24.03** 降低到 **23.46**。

综合来看，在目前已经完成的主设置中，本文方法在模型深度、上下文长度和训练预算变化下都表现出稳定的改进，最终测试困惑度的相对下降区间为 **1.3\% 到 3.1\%**。虽然绝对提升幅度不算巨大，但它在所有已完成主设置中都保持一致。验证损失曲线见附录。

**表 1**：WikiText-103 主结果与 cost-performance 对比。每个结果均为相同配置下三次运行的中值。所提方法在参数量几乎不变的情况下，在所有已完成设置中都稳定改善验证集和测试集困惑度。对于 80000-step 的实验，还报告了匹配训练设置下的 step time、throughput 与 peak GPU memory。

### 3.3 注意力分配与检索模式分析

除了困惑度之外，注意力分析也直接证明了新增 depth 分支不是一个被忽略的结构扰动，而是一个真正参与计算的功能性分支。最基本的统计量是：row 分支与 depth 分支分别分得了多少平均注意力质量。在 16 层模型中，早层仍然明显由 row 方向 token 注意力主导，但中后层会将显著更多的注意力质量分配给 depth retrieval；在最深层，平均 depth mass 已经与 row mass 接近，甚至略高，如图 2 所示。

这种按层增长的趋势需要谨慎解释。因为随着层数增加，可见的 memory slots 也会变多，因此总 depth mass 的上升部分上是由候选数量增多导致的。为此，本文进一步检查了 slot-normalized 的统计量以及 depth-slot heatmap。结果显示，depth 使用量的上升并不只是由槽位数增长带来的：即使考虑每个 slot 的平均使用情况，中后层仍保留了非平凡的 per-slot depth usage。更重要的是，slot-allocation heatmap 表明，模型并没有把注意力均匀分配到所有可见历史上，而是对少量早期 memory slots 表现出明显偏好，尤其偏向网络底部附近的那些条目。

**图 2**：16 层模型按层统计的 row-vs-depth 注意力质量。早层明显以 row 分支为主，中后层则逐步给 depth 分支分配更多质量。在最深层，depth retrieval 已经与 row token routing 接近，甚至略强。

这一趋势与 token 位置分析结合后更有解释力。图 3 展示了 depth branch 占总注意力预算的比例在 layer 与 token position 两个维度上的变化。随着层数加深，depth share 明显增加；但它并没有塌缩到少数几个晚位置 token 上。除去最早的一小段 prefix 区域后，depth share 在不同 token 位置上的模式整体较稳定。直观上，随着 token 位置增大，row 分支可见的 token 候选数会增加；但模型并未因此把全部注意力预算重新推回 token 轴。这说明即使 row 分支的候选数变多，depth retrieval 仍保持了功能相关性。

与联合注意力矩阵结合来看，这些观察支持一个更具体的解释。depth 分支不仅在深层被积极使用，而且往往会将显著注意力质量放在少数极早层的历史条目上。在当前模型中，这些条目对应于网络底部附近的状态。这说明所提出的结构确实为深层提供了一条更直接访问底层或较少变换表示的路径。标准残差连接同样会把这些信息向后传播，但本文结果表明，仅靠 residual mixing 可能不足以实现对早层信号的强选择性复用。

这些观察并不能直接证明该方法在更大模型或更长训练中一定继续有效，但它们提供了朝这个方向的积极证据。随着层数加深或 token 候选增多，depth 分支并没有自然消失，相反它在某些情形下变得更显著。这至少与如下假设一致：显式的 layer-depth retrieval 在更深的结构和更长的上下文中仍然具有作用。

**图 3**：16 层模型中 depth-attention share 随 layer 与 token position 的变化。横轴每隔 12 个位置采样一次。主要变化发生在 layer 维上而不是 token 维上：中后层的 depth retrieval 明显更强，而在除最早 prefix 区域外的大多数 token 位置上，其占比保持整体稳定。

单样本注意力矩阵进一步展示了 baseline 与本文方法之间的定性差异。对于 baseline，全部注意力预算只分配给 token 位置；而在本文方法中，row token pathway 被保留，但联合注意力矩阵中另外有一块清晰的 depth-slot 区域也获得了可见的注意力质量。这使最终注意力模式更有选择性：模型不再只能在 token 位置之间分配概率，而是可以把一部分预算路由给少量高价值历史状态。

**图 4**：第 13 层、序列长度为 64 时的联合注意力矩阵。baseline 的全部注意力预算都分布在 token 位置上；而本文方法在保留 row-token 路径的同时，还向 depth-slot 区域分配了清晰可见的注意力质量。这说明模型并未忽略新增分支，而是把一部分联合预算分配给少量高价值的 layer-depth 历史槽位。

### 3.4 消融总结

当前消融结果表明，layer-depth routing 这一总体思想是稳健的，但 memory 分支的最佳参数化方式仍然是一个开放的经验问题。尤其是，最终采用的 single-query projected-K/V 设计明显优于 baseline，但其附带的每个设计选择并不都同样重要。作为相关比较，本文也在相同训练框架下列出了 Attention Residuals 风格方法。

**表 2**：核心 8 层、`seq_len=256` 设置下的关键消融结果。最终的 single-query projected-K/V 设计在目前已完成的变体中取得了最好的核心结果。

## 4. 相关工作

现代 decoder-only 语言模型主要建立在 token 维度上的因果自注意力之上。以往工作早已指出，这种结构的关键优点之一，是相较于循环结构大幅缩短了序列轴上的通信路径。本文的出发点则是观察到：这一优势主要体现在 token 维，而同一 token 在 earlier layers 中形成的历史表示，在标准结构中并未获得同等直接的访问路径。

大量已有工作从记忆机制、循环状态、缓存历史或跨层交互等角度扩展了 Transformer。一部分方法通过跨 segment 或跨 decoding step 的记忆来扩大有效上下文范围，例如 Transformer-XL；还有一些方法重新思考重复计算在深度方向上的组织方式，例如 Universal Transformer；另一些方法则通过跨层聚合、残差增强或深度路由改进不同层之间的信息复用。这些方向与本文都有关联，但本文的目标更窄：关注的是**当前 token 如何显式检索自身在 earlier layers 中形成的同位置表示**。

另一个相关方向是：标准残差传播是否可以通过学习到的层间聚合或残差注意力路径得到改进。这类方法与本文的动机高度相关，因为它们同样承认，有价值的信息可能分布在深度轴上，而不仅仅在当前层的 token 轴上。然而，本文方法保留了标准 token-attention 分支，并将 retrieval 组织为同位置历史 memory entries 的检索，而不是把整个层堆叠视为一个泛化的 residual recombination pool。从这个意义上说，它也与近期将 Transformer 信息流显式分解为 sequence 轴与 depth 轴的尝试相近，但与 DeepNet 这类主要关注深层稳定训练的方法互补。ALBERT 一类参数共享工作也有一定间接相关性，因为它们表明，深度方向上重要的不只是“堆更多层”，还包括“如何在层与层之间复用信息和计算”；本文则从 retrieval 角度而不是参数共享角度处理这个问题。

## 5. 结论

本文研究了一个简单但重要的结构问题：decoder-only Transformer 是否能从显式读取同一 token 的 depth history 中获益，而不仅仅依赖残差路径在层堆叠中的传播？本文实验结果表明，答案是肯定的。在当前完成的 WikiText-103 设置下，layer-depth memory routing 机制在不同深度、不同序列长度和不同训练预算下，都稳定优于标准 Transformer baseline。

本文最主要的经验信息并不是“增益巨大”，而是“增益稳定”。这一点很重要，因为本文使用的 baseline 本身已经较强，因此 1.3\% 到 3.1\% 的最终测试困惑度改进，更应被理解为一种稳定的结构收益，而不是来自弱基线的偶然现象。

与此同时，本文也清楚暴露了一个 trade-off：当前原型存在明显的运行时开销，且实测 slowdown 大于纯理论计算增量。这说明当前系统更适合被视为一个可信但尚未完全优化的结构原型，而不是一个已经打磨完毕的最终实现。

更广义地说，本文支持一个观点：对于 decoder-only 语言模型而言，depth history 是一个真实、有价值的信息源。一个 token 不仅需要访问 earlier tokens，也可以从直接访问自身在 earlier layers 中形成的表示中获益。注意力分析进一步支持这一解释：训练后的模型在深层会给 depth 分支分配显著质量，并把检索集中在少量高价值历史槽位上，而不是忽略新增分支或对历史均匀平均。本文提出的 layer-depth routing 只是显式暴露这一信号的一种具体方式。未来工作需要进一步研究更好的 memory 参数化、更高效的运行时实现，以及这一设计原则在更大模型和更广泛语言建模设定中的适用性。

## 附录 A：默认实验设置

**表 3**：WikiText 实验脚本使用的默认训练配置。正文中的主实验只显式覆盖结果表中会变化的字段，尤其是层数、序列长度与训练总步数。

- 数据集：WikiText-103 raw
- Tokenizer：GPT-2 BPE
- Sequence length：256
- Hidden width $D$：384
- Layers $L$：8
- Attention heads：8
- MLP ratio $r$：4
- Dropout：0.1
- Tied input/output embeddings：开启
- Positional embeddings：开启
- Batch size：8
- Gradient accumulation steps：1
- Training steps：40000
- Evaluation interval：1000 steps
- Evaluation batches：100
- Learning rate：$3\times10^{-4}$
- Minimum LR scale：0.1
- Warmup steps：100
- Weight decay：0.01
- Gradient clipping：1.0
- Random seed：42

## 附录 B：训练曲线

**图 5**：主 WikiText-103 比较中、`seq_len=256` 的验证损失曲线。实线表示本文方法，虚线表示 baseline；颜色区分 8 层和 16 层设置。本文方法在早期训练阶段已具有竞争力，并在主要训练区间内保持低于 baseline 的验证损失。
