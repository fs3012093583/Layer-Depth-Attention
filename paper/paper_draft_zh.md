# 基于层深度信息路由的 Transformer 结构研究

## 摘要

Transformer 在 token 维度上的信息传播已经非常高效，但对同一 token 在更早层形成的表示的访问仍然较为间接。本文研究这样一个问题：当前层的 token 是否能够通过直接读取自身的层深度历史表示，而不仅仅依赖残差连接在层间逐步传播信息，从而获得更好的建模效果。为此，本文提出一种层深度注意力路由机制，在标准自注意力之外引入一条同位置跨层记忆分支，使模型能够在序列维度和层深度维度之间联合分配注意力预算。本文在 decoder-only 语言模型上对这一思路进行了实例化和验证。当前实现采用单一共享查询向量，同时用于 row-wise token attention 与 depth-wise memory retrieval，并为每个 block 维护一组投影后的历史 K/V 记忆。在 WikiText-103 数据集上，针对 8 层、16 层、不同序列长度和较长训练预算等多种设置，所提方法相较于强 Transformer baseline 在最终测试困惑度上取得了 1.3\% 至 3.1\% 的稳定改进。进一步的注意力分配分析表明，深层网络确实为 depth 分支分配了较大比例的注意力质量，并倾向于检索少量高价值的早期层历史槽位，而不是忽略这条附加路径。此外，随着 token 位置增大，row 分支与 depth 分支之间的相对注意力分配总体保持稳定，说明 depth 路径在较长序列条件下并未明显失效。整体而言，这些结果表明显式的层深度历史是一种有价值的信息来源，但同时也显示出明显的代价：当前原型方法存在可观的运行时开销。本文据此认为，基于层深度信息访问的 Transformer 改进方向具有较好的研究潜力，但在记忆路径优化和更强参数化设计方面仍有进一步研究空间。

## 1 引言

Transformer 已成为当前自然语言处理中的核心基础架构，其成功很大程度上来自于自注意力机制在序列维度上的高效信息传播能力。相比循环神经网络中依赖多步状态传递的方式，Transformer 允许当前 token 在单层中直接访问其上下文，从而显著缩短序列维度上的通信路径。

从计算图角度看，Transformer 的表示演化可以理解为同时沿着两个维度展开：一是 token 维度，即序列中不同位置之间的信息交互；二是 layer 维度，即同一 token 在不同层中的表示逐步演化。现有 Transformer 的优势主要体现在前者，而在后者上，当前层对于 earlier-layer 表示的访问仍然较为间接。标准 Transformer 中，浅层信息主要通过残差连接向后传播。虽然残差连接确实提供了跨层信息通路，但这种传播方式是被动的、加法式的，而非当前层根据需要主动选择某个 earlier-layer 表示进行读取。

这一问题在模型加深时会更加突出。对于一个深层 decoder-only Transformer 而言，底层产生的词法特征、局部模式或中间语义状态，需要穿过多层 attention 和 FFN 变换才能影响最上层决策。残差连接能够缓解信息退化问题，但不保证深层能够高效、选择性地访问对当前决策仍然重要的 earlier-layer 表示。由此引出一个自然问题：**Transformer 是否可以在标准 token attention 之外，增加一条针对同位置层深度历史的主动检索路径，从而改善层维度上的信息传播？**

本文围绕这一问题展开研究，提出一种层深度注意力路由机制。该方法保留标准 row-wise token attention 主干，同时引入一条 same-position history retrieval 分支，并通过联合 softmax 将两条路径放入同一个注意力预算中竞争。与将层间信息作为额外旁路附加到网络中不同，本文希望让模型自行学习：在何时更应关注当前层 token 上下文，何时更应检索 earlier-layer history。

与现有工作相比，本文的目标更聚焦于 **信息传播路径的重新组织**。长上下文方法主要扩展的是 token 维度上的可见范围；深层稳定化方法关注的是网络在更深层数下的训练可行性；跨层聚合方法则尝试融合不同层表示。本文更强调：对于某一 token 而言，是否存在一种更直接的路径，使其能够读取自己在 earlier layers 中形成的高价值中间表示。

本文的实验结果表明，这一思路在 WikiText-103 上能够带来稳定收益。尽管性能提升幅度不是特别大，但其跨设置的一致性、注意力行为上的可解释性以及对层间信息利用规律的揭示，说明该方向具有进一步研究的价值。

本文的主要贡献包括：

1. 从信息传播视角重新审视 Transformer 的结构特点，指出其在序列维度和层深度维度上的信息访问存在不对称性。
2. 提出一种层深度注意力路由机制，使当前 token 能够联合访问当前层 prefix context 和自身 earlier-layer history。
3. 在 WikiText-103 上验证了该机制在多组主实验设置下的稳定改进。
4. 通过注意力质量统计、联合注意力矩阵和 layer-position 热图分析，验证了模型确实在主动使用 depth branch。
5. 分析了方法的理论复杂度和实际代价，指出该方向仍然存在明显的系统实现优化空间。

## 2 方法

### 2.1 问题设定

考虑一个具有 $L$ 层的 decoder-only Transformer。记第 $l$ 层第 $t$ 个 token 的隐藏状态为 $x_t^{(l)}$。标准 causal self-attention 允许 $x_t^{(l)}$ 直接访问同层中的 prefix token 集合 $\{1,\dots,t\}$，但不会显式暴露其在 earlier layers 中形成的同位置历史表示 $\{x_t^{(1)},\dots,x_t^{(l-1)}\}$ 供当前层检索。

本文希望增强标准 Transformer 在 layer 维度上的信息传播能力，使当前层 token 除了可以读取序列上下文外，还能够主动访问 earlier-layer 表示。

### 2.2 信息传播视角下的动机

从有向计算图角度看，Transformer 的节点是各层各位置的隐藏状态，边表示信息影响路径。标准 causal self-attention 已经显著缩短了 token 维度上的通信距离，因为当前 token 可以在单层内直接读取 prefix 中的其他 token。而在层深度维度上，当前层对 earlier-layer 同位置表示的访问仍主要依赖残差连接的逐层传递。

这一点在深层模型中尤为重要。浅层特征如果要影响深层输出，往往需要穿过多层 attention 和 FFN 变换。残差连接虽然提供了稳定传播路径，但它更多是被动传输，而不是显式读取。本文的方法可以看作对残差传播的一种主动补充：与其仅仅依赖早层信息在残差流中被保留下来，不如允许当前层在需要时直接读取 earlier-layer 历史。

### 2.3 Row Attention 与 Depth Memory

在每一层中，本文同时保留两种信息来源：

1. **row branch**：标准的当前层 token attention，用于读取同层 prefix token；
2. **depth branch**：同位置跨层历史记忆，用于读取 earlier-layer same-position states。

当前 token 产生的查询表示同时用于这两条路径，最后将 row 分支和 depth 分支的注意力得分拼接起来，并通过同一个 softmax 进行归一化。于是，模型不是分别做两个独立注意力决定，而是在一个统一预算中学习如何在两种信息来源之间分配注意力。

这一定义带来一个重要性质：如果 depth branch 在某种设置下并不重要，联合 softmax 可以自动将绝大多数注意力质量分配给 row branch，使模型行为退化到接近标准自注意力；如果 depth history 确实有价值，则模型可以自适应地将部分预算转移到 depth branch。也就是说，本文的方法不会强制某个轴必须占据固定比例的注意力质量，而是让模型根据数据和训练过程自行决定两个维度的相对重要性。

### 2.4 最终实现

本文当前采用的具体实现包含以下两点关键设计。

#### 2.4.1 Single-Q

对于 row attention 和 depth memory retrieval，本文使用同一组查询表示，而不额外引入独立的 depth query 分支。这样做的原因在于：depth branch 的作用并不是构造一个完全不同的检索任务，而是让当前 token 在“继续看上下文”与“回读自身历史”之间进行统一路由。共享查询可以使两种检索决策建立在同一个当前状态之上，从而更符合统一注意力预算的设计目标。

#### 2.4.2 Projected Historical K/V

历史 memory 并不是直接存储为原始 hidden state，而是通过一组共享的线性映射投影到统一的 memory K/V 空间中。这样做有两层考虑：

第一，隐藏状态在网络内部承担的是持续计算的角色，而用于历史检索的 memory 表示未必需要与 ongoing hidden state 完全一致。通过专门的 memory projection，可以将“用于当前层计算的表示”和“用于未来层检索的表示”解耦。

第二，不同层的 hidden state 分布并不一定一致，尤其在深层网络中，各层经过的 attention 和 FFN 变换不同。如果直接把各层 raw hidden state 混合到一起作为历史库，模型需要面对一个异质性较强的表示集合。共享的 memory projection 提供了一种统一映射方式，使不同层的历史信息能够被放到一个相对一致的检索空间中，降低 later layers 在历史库中进行匹配的难度。

### 2.5 层内计算过程

在第 $l$ 层，记当前隐藏序列为

\[
X^{(l)} \in \mathbb{R}^{B\times S\times D}.
\]

row branch 先通过标准的 token-attention 投影得到

\[
[Q_{\text{row}},K_{\text{row}},V_{\text{row}}] = W_{\text{qkv}}X^{(l)}.
\]

然后在 causal mask 下计算同层 prefix attention 分数：

\[
A_{\text{row}} = \frac{Q_{\text{row}}K_{\text{row}}^\top}{\sqrt d}.
\]

depth branch 使用同一个查询：

\[
Q_{\text{depth}} = Q_{\text{row}}.
\]

历史记忆由 earlier-layer block states 构成，每层向历史库中贡献一组 projected K/V：

\[
K_{\text{mem}} = W_k^{\text{mem}}H,\qquad
V_{\text{mem}} = W_v^{\text{mem}}H.
\]

因此，第 $l$ 层可见的历史库可写为

\[
\mathcal{M}^{(l)}=\{(K_{\text{mem}}^{(1)},V_{\text{mem}}^{(1)}),\dots,(K_{\text{mem}}^{(m)},V_{\text{mem}}^{(m)})\}.
\]

当前层与历史库之间的 depth matching 记为

\[
A_{\text{depth}}=\frac{Q_{\text{depth}}\cdot K_{\text{mem}}}{\sqrt d}.
\]

最后，两条路径的得分被拼接并统一归一化：

\[
A=[A_{\text{row}};A_{\text{depth}}],\qquad
P=\operatorname{softmax}(A).
\]

模型据此在 row-context aggregation 与 memory-context aggregation 之间分配一个共享的注意力预算，并将两部分上下文相加得到最终输出：

\[
O=O_{\text{row}}+O_{\text{depth}}.
\]

除额外历史投影和 depth branch 外，其余部分仍保持标准 Transformer 结构，包括输出投影、残差连接和 FFN 主干。

### 2.6 复杂度分析

设模型层数为 $L$，隐藏维度为 $D$，序列长度为 $S$，batch size 为 $B$，MLP ratio 为 $r$。标准 decoder-only Transformer 的主要计算量可写为：

\[
C_{\text{base}} = L\bigl((4+2r)BSD^2 + 2BS^2D\bigr).
\]

在本文方法中，相比 baseline 主要增加两部分开销：

1. 历史 memory 的额外 K/V 投影；
2. depth branch 的 matching 与 aggregation。

由于当前实现中每层只向历史库追加一组 memory entry，因此第 $l$ 层可见历史槽位数为：

\[
M_l=l-1.
\]

于是整个模型的近似计算量可写为：

\[
C_{\text{method}} \approx L\bigl((6+2r)BSD^2 + 2BS^2D\bigr)+BSD\,L(L-1).
\]

在本文主要实验所用的 $r=4$ 设置下，有：

\[
C_{\text{method}} \approx L\bigl(14BSD^2 + 2BS^2D\bigr)+BSD\,L(L-1).
\]

可以看到，方法并没有重新定义 Transformer 的主干计算，而是在标准自注意力之上增加了一条与层深度历史相关的附加路径。

在空间开销方面，实验中更直接的指标是训练阶段的峰值显存。相较 baseline，本文方法需要保留额外的历史 K/V 激活以及 depth branch 的中间张量，因此训练显存会有所增加。但从主实验结果来看，显存增长相对温和，而更明显的代价主要体现在训练吞吐下降上。这说明当前方法的瓶颈不仅来自额外的理论计算量，也来自原型实现的系统执行开销。

## 3 实验设计

### 3.1 数据集与实验设置

本文主要在 WikiText-103 原始文本设置上开展实验。所有模型均采用 GPT-2 BPE tokenizer，并使用 tied embedding。主体比较围绕 decoder-only Transformer baseline 与本文方法展开，控制 hidden size、head 数、MLP ratio、position embedding 等主干结构一致，仅在跨层信息建模方式上做区别。

目前已完成的主实验设置包括：

- 8 层、`seq_len=256`，训练 40000 与 80000 steps；
- 16 层、`seq_len=256`，训练 80000 steps；
- 8 层、`seq_len=512`，训练 80000 steps；
- 16 层、`seq_len=512`，训练 80000 steps。

主要指标包括：

- Best Validation PPL
- Final Test PPL
- Step Time
- Tokens/s
- Peak GPU Memory

### 3.2 主要结果

实验结果表明，本文方法在多组主实验设置下均优于标准 Transformer baseline。在 8 层、`seq_len=256` 设置下，40000 steps 时最终测试困惑度由 34.53 降至 33.53；80000 steps 时由 28.47 降至 27.58。在 16 层、`seq_len=256` 设置下，最终测试困惑度由 26.28 降至 25.55。在更长上下文的 8 层、`seq_len=512` 设置下，也由 24.03 降至 23.46。

综合各组已完成主实验，本文方法在最终测试困惑度上取得了约 1.3\% 至 3.1\% 的相对改进。虽然提升幅度并非特别大，但其稳定性较好，说明该方法带来的不是偶然的实验波动，而是一种具有重复性的结构收益。

### 3.3 消融实验

为分析不同设计选择的作用，本文进一步进行了关键消融比较，包括：

- block-level history 与更细粒度历史建模方式的比较；
- projected historical K/V 与 raw history 方案的比较；
- 与注意力残差类相关方法的对比。

消融结果表明，显式的 depth-history retrieval 本身是有效的，而不同历史表示方式和参数化策略会影响收益大小。其中，使用投影后的历史 K/V 通常比直接使用原始历史表示更稳定。

### 3.4 效率分析

在训练效率方面，本文方法并不以减少计算为目标。实验结果表明，在参数规模几乎不变、显存增幅相对温和的情况下，方法带来了稳定的困惑度改进；但与此同时，训练吞吐有所下降，说明该方法存在较明显的 quality-efficiency trade-off。

因此，本文更倾向于将当前系统视为一个有效但尚未充分优化的结构原型，而不是一个已经完成系统效率优化的最终方案。

## 4 模型行为分析

除了性能结果之外，本文还关注模型是否真实利用了所引入的层深度历史路径。

### 4.1 Row 与 Depth 注意力分配

通过统计联合 softmax 后分配给 row branch 和 depth branch 的注意力质量，可以观察到：浅层仍主要依赖 row attention，而随着层数加深，depth branch 所占注意力质量逐步提高。在最深的若干层中，depth 分支甚至能够与 row 分支相当，说明 depth history 在 deep layers 中并不是可有可无的附加结构。

### 4.2 历史槽位选择模式

进一步的联合注意力可视化显示，depth branch 的注意力并不会均匀地分配到所有历史槽位上，而是倾向于集中在少数高价值早期槽位，尤其是靠近网络底部的历史表示。这说明本文方法更像一种“主动检索”机制，而不是简单的跨层平均混合。

### 4.3 Token 位置与 Layer 深度的联合规律

从 layer × token position 的热图可以看出，depth 分支占比的主要变化维度是 layer，而不是 token 位置。随着层数增加，depth share 明显增大；而随着 token 位置增大，depth branch 并没有被持续压缩。这表明该方法的作用不只局限于浅层或短上下文场景，而在更深层、更长序列条件下仍然保持一定稳定性。

## 5 相关工作

与本文最相关的工作主要包括以下几类。

第一类是长上下文和记忆增强方法，如 Transformer-XL。这类方法主要关注在 token 维度上扩展模型的可见范围，即让模型能够处理更长距离的上下文依赖。

第二类是从深层结构组织角度改进 Transformer 的工作，如 Universal Transformer、DeepNet 以及残差增强相关方法。这些工作表明，随着层数增加，结构中的信息传播与稳定训练问题会逐渐显现。

第三类是跨层特征聚合、跨层注意力和残差路径增强方法。这些研究说明，不同层的表示可能具有互补信息，later layers 直接利用 earlier-layer 表示是一个具有潜力的方向。

本文与这些工作的共同点在于都关注深层 Transformer 中的信息流动问题；不同之处在于，本文更强调从信息传播角度统一理解 token 维与 layer 维的作用，并聚焦于同一 token 在 earlier layers 中形成的历史表示是否值得被显式读取。

## 6 结论

本文从信息传播视角研究了 Transformer 中的跨层信息利用问题，提出了一种层深度注意力路由机制，使当前 token 在标准 row-wise token attention 之外，还能够主动访问自身的 earlier-layer history。实验结果表明，所提方法在 WikiText-103 上可以稳定改善最终测试困惑度；行为分析进一步表明，模型在深层中确实会显著利用 depth branch，并倾向于检索少量高价值的早期层历史表示。

总体而言，本文的结论并不是该方法已经成为标准 Transformer 的完全替代，而是：**层深度历史确实是一种有价值的信息来源，显式增强 Transformer 在 layer 维度上的信息访问能力是值得继续研究的方向。** 未来工作可进一步围绕更高效的 memory 参数化、更低开销的实现方式以及更大规模模型中的适用性展开。
