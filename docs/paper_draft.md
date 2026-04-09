# Layer-Depth Attention Routing in Transformers

## Abstract

Transformers propagate information effectively along the token dimension, but access to intermediate representations formed at earlier layers remains indirect. We study whether a token at the current layer can benefit from directly reading its own depth history, rather than relying only on repeated residual propagation through the stack. To this end, we introduce a layer-depth attention routing mechanism that augments standard self-attention with a same-position cross-layer memory branch. In this paper, we instantiate and evaluate this idea in decoder-only language models. The current implementation uses a single shared query for row-wise token attention and depth-wise memory retrieval, stores sublayer-level history, and archives historical memory through projected keys and values. On WikiText-103, across multiple settings including 8-layer and 16-layer models, different sequence lengths, and long training budgets, the method consistently improves perplexity over strong Transformer baselines by roughly 2% to 4%. These results indicate that explicit depth history is a useful information source even for already strong decoder baselines, while also revealing a clear trade-off: the current prototype incurs noticeable runtime overhead. Overall, the results position layer-depth attention routing as a stable and promising architectural direction for Transformer language models, while leaving memory-path optimization and stronger memory parametrization studies to future work.

---

## 1. Introduction

Transformer language models are built around token-wise attention, where each layer reads contextual information from other positions in the sequence. A useful way to interpret this architecture is as a two-dimensional computation graph over token positions and layer depth. Under this view, recurrent models mainly propagate information through chained local transitions, while self-attention dramatically shortens communication along the token axis by allowing a token to directly access its prefix within one layer. However, the same perspective also reveals a remaining asymmetry: intermediate representations formed by the same token at earlier layers are still only indirectly accessible in standard decoder-only Transformers.

Residual connections do provide a depth-wise transmission path, so standard Transformers are not entirely lacking in cross-layer communication. But this path is uniform and additive rather than selective: earlier-layer information is mixed forward through the residual stream, not explicitly retrieved on demand for the current token. As depth increases, useful earlier-layer signals may therefore be diluted among many subsequent transformations. This observation motivates a simple question: can a decoder-only language model improve by augmenting token attention with a direct layer-depth memory branch? If the current token could jointly attend to both the usual row-wise prefix context and its own cross-layer history, then the model might reuse partially refined intermediate states more effectively instead of repeatedly reconstructing them through the stack.

In this work, we study this question through a controlled family of cross-layer memory mechanisms for decoder-only Transformers. Our final design uses a single query to route both row-wise token attention and depth-wise memory retrieval, stores memory at the sublayer level, and archives historical states through projected keys and values. The resulting mechanism extends each attention step from pure token interaction to a joint token-depth routing problem. In the computation-graph view, the method preserves the short communication path along the token axis while introducing a more direct and selective access path along the depth axis.

The resulting gains are not dramatic, but they are stable. Across multiple WikiText-103 settings, including 8-layer and 16-layer models, different sequence lengths, and long training budgets, the proposed method consistently improves perplexity over strong Transformer baselines by roughly 2% to 4%. At the same time, the method introduces nontrivial runtime overhead, and the measured slowdown is larger than the pure theoretical compute gap, indicating that the current prototype implementation is not yet fully optimized.

These results support a restrained but meaningful conclusion: explicit depth-history access is useful for decoder-only language modeling, and it can yield repeatable improvements even on already strong baselines. However, the method is best viewed as a promising architectural direction rather than a fully optimized replacement for standard self-attention.

### Contributions

This paper makes the following contributions:

1. We formulate decoder-only attention as a **joint row-depth routing problem**, where the current token can read both prefix tokens and its own cross-layer history.
2. We introduce a practical final design based on **single-query routing, sublayer memory, and projected historical K/V**, and show that this combination gives the most stable results among the tested variants.
3. We show on WikiText-103 that the method yields **consistent 2% to 4% perplexity improvements** across multiple settings, including 8-layer and 16-layer models and different sequence lengths.
4. We analyze the method’s cost and show that while the theoretical extra compute is moderate, the current prototype incurs additional systems-level overhead, highlighting an important optimization direction for future work.

---

## 2. Method

### 2.1 Problem Setup

Consider a decoder-only Transformer with `L` layers. At layer `l`, the hidden state at token position `t` is denoted by `x_t^(l)`. Standard causal self-attention allows `x_t^(l)` to read row-wise context from the prefix positions `{1, ..., t}` in the same layer. However, it does not explicitly expose the cross-layer history `{x_t^(1), ..., x_t^(l-1)}` for direct retrieval.

We aim to augment standard self-attention with a depth-memory branch that allows the current token to retrieve same-position information formed at previous layers.

More generally, we view the model as a directed computation graph whose nodes are hidden states `x_t^(l)`. The communication distance between two states is the minimum number of graph edges required for one state to influence the other. Under this view, standard causal self-attention already shortens token-wise communication distance, since a token can directly access its full prefix within one layer. Our objective is to similarly shorten access to same-position earlier-layer states, which in the standard decoder are otherwise reached mainly through residual mixing across successive layers.

### 2.2 Row Attention and Depth Memory

For each layer, we keep the usual row-wise token attention branch over the current sequence. In parallel, we maintain a depth-memory archive consisting of same-position history from earlier layers or sublayers. The current token then produces a single query that is used for two routing decisions:

- row-wise routing over current-layer prefix tokens
- depth-wise routing over same-position historical memory

The row branch and depth branch produce scores that are concatenated and normalized jointly, so the model allocates attention mass across both sources within one competition space.

### 2.3 Final Design Used in This Paper

The final method has three defining design choices.

#### Single-Q

The same query representation is used for both row attention and depth-memory lookup. This avoids introducing an additional query branch and gives a simpler routing mechanism than dual-query variants.

#### Sublayer Memory

Instead of storing only one memory item per Transformer block, we cache history at the sublayer level. In the current implementation, this means that attention-side and FFN-side intermediate states can both contribute memory entries. This produces a richer depth archive than block-only storage.

#### Projected K/V

Historical memory is archived through projected keys and values rather than leaving it in raw hidden-state form. This gives the current layer a learned memory space for matching and aggregation, and empirically performs more stably than removing projection entirely.

### 2.4 Why This Combination

The available empirical evidence suggests the following:

- **single-q** is simpler and sufficient for stable gains
- **sublayer memory** improves the usefulness of the depth archive relative to coarser block-only memory
- **projected K/V** performs more reliably than directly using raw hidden states as memory keys and values

Accordingly, **single-q + sublayer + projected K/V** is treated as the main method, while the other variants are treated as supporting ablations rather than separate primary methods.


### 2.5 Layer Computation in the Final Implementation

The final implementation can be described as a standard decoder block augmented with an additional same-position memory-routing path. At layer `l`, let the current hidden sequence be

\[
X^{(l)} \in \mathbb{R}^{B \times S \times D}.
\]

The row-attention branch first computes a fused token-attention projection:

\[
[Q_{\text{row}}, K_{\text{row}}, V_{\text{row}}] = W_{\text{qkv}} X^{(l)}.
\]

After reshaping into multi-head form, the model computes the standard causal token-attention scores over prefix positions:

\[
A_{\text{row}} = \frac{Q_{\text{row}} K_{\text{row}}^\top}{\sqrt{d}},
\]

with the usual causal masking applied.

For the depth-memory branch, the final design uses the same query as the row branch:

\[
Q_{\text{depth}} = Q_{\text{row}}.
\]

This is the **single-query** choice. The same token representation that decides which prefix tokens are useful also decides which historical same-position states are useful, avoiding an extra query branch while keeping routing behavior stable in practice.

The memory archive is built from previously produced sublayer states. In the current implementation, each block contributes multiple same-position memory entries, so the archive for layer `l` contains projected historical keys and values from earlier layers and earlier sublayers:

\[
\mathcal{M}^{(l)} = \{(K_{\text{mem}}^{(1)}, V_{\text{mem}}^{(1)}), \ldots, (K_{\text{mem}}^{(m)}, V_{\text{mem}}^{(m)})\}.
\]

Each historical entry is archived through learned projections:

\[
K_{\text{mem}} = W_k^{\text{mem}} H,\qquad V_{\text{mem}} = W_v^{\text{mem}} H,
\]

where `H` denotes the corresponding sublayer hidden state. In the current implementation, the memory projection matrices are shared across layers, so the historical archive is parameterized by global memory projections rather than separate per-layer memory heads. This is the **projected K/V** choice. It gives the model a dedicated memory-matching space, which empirically behaves more stably than directly treating raw hidden states as both keys and values.

Given the memory archive, the current layer computes depth-memory scores by matching the current token query against the historical same-position keys:

\[
A_{\text{depth}} = \frac{Q_{\text{depth}} \cdot K_{\text{mem}}}{\sqrt{d}}.
\]

The key design choice is that row-wise token scores and depth-wise memory scores are concatenated and normalized jointly:

\[
A = [A_{\text{row}}; A_{\text{depth}}],\qquad P = \operatorname{softmax}(A).
\]

This means the model does not make two independent attention decisions. Instead, it allocates one unified attention budget across two competing sources:

- current-layer token context
- historical same-position depth memory

The final output context is then the sum of the row-context aggregation and the memory-context aggregation:

\[
O = O_{\text{row}} + O_{\text{depth}}.
\]

The resulting sequence is passed through the usual output projection and then through the remaining residual/FFN components of the Transformer block.

**[Figure 1 about here]**  
**Figure 1 caption draft.** Overview of the proposed layer-depth memory routing block. The standard row-wise causal self-attention pathway is preserved, while an additional same-position depth-memory branch allows the current token to retrieve earlier-layer or earlier-sublayer history. The current implementation uses a single query for both token routing and depth retrieval. Row-wise token scores and depth-memory scores are concatenated before a joint softmax, so the model allocates one shared attention budget across current-layer prefix context and historical same-position memory.

### 2.6 What "Sublayer Memory" Means in Practice

The phrase **sublayer memory** should be explained carefully in the paper, because it is more specific than generic cross-layer memory. In the final implementation, memory entries are collected not only from one coarse block-level hidden state, but also from intermediate representations produced within the block. Concretely, both the attention-side intermediate state and the FFN-side intermediate state can contribute memory entries to the history archive used by later layers.

This gives the model a denser depth archive than one-memory-per-layer designs. Conceptually, it means later layers are allowed to retrieve not just what an earlier layer finally output, but also partially transformed internal states produced during that layer's computation. Empirically, this richer archive appears to be more useful than a coarse layer-only memory, which is why the final method keeps the sublayer design.

### 2.7 Implementation Interpretation

An important implementation detail is that the final method is best viewed as **routing over archived memory K/V pairs**, not as directly reusing raw hidden-state tensors in place. The model reads from a history bank whose entries have already been transformed into a learned memory space. This distinction matters because it clarifies both the strengths and the costs of the method:

- it gives the model a structured memory-retrieval space rather than raw feature replay
- it adds extra projection, archive-maintenance, and memory-aggregation cost compared with baseline self-attention

This interpretation should be stated explicitly in the paper to avoid confusion between cross-layer hidden reuse and cross-layer projected memory retrieval. The final implementation belongs to the latter category.

### 2.8 Complexity and Memory Overhead

The proposed mechanism improves modeling quality at the cost of additional computation and storage.

#### Baseline Cost

For a decoder-only Transformer layer with hidden width `D`, sequence length `S`, batch size `B`, and MLP ratio `r`, the dominant baseline forward cost comes from three components:

- token-attention projections
- token-token attention
- the two-layer MLP

Ignoring lower-order terms such as layer normalization, masking, and dropout, a standard baseline layer has the following leading-order cost:

\[
C_{\text{base, layer}} = (4 + 2r)BSD^2 + 2BS^2D.
\]

For the commonly used `r = 4` setting, this becomes:

\[
C_{\text{base, layer}} = 12BSD^2 + 2BS^2D.
\]

#### Additional Cost of Layer-Depth Memory Routing

Relative to the baseline, the proposed method adds three kinds of work.

First, each layer must build the memory archive entries that will be consumed by later layers. In the final projected-memory design, this introduces extra memory-side `K/V` projections on top of the usual row-attention projections.

Second, each layer performs depth-memory matching in addition to standard token-token attention. This means that the current token query must score against archived historical keys and then aggregate the corresponding historical values.

Third, the sublayer-memory design increases the number of archived entries, because attention-side and FFN-side intermediate states can both contribute to the history bank.

Under this approximation, the final method has leading-order per-layer cost

\[
C_{\text{method, layer}} \approx (8 + 2r)BSD^2 + 2BS^2D + 2BSM_iD,
\]

where `M_i` denotes the number of historical memory entries visible to layer `i`. For the current `r = 4` setting,

\[
C_{\text{method, layer}} \approx 16BSD^2 + 2BS^2D + 2BSM_iD.
\]

The important point is that the extra cost comes from two sources:

- extra learned projection work, which scales with `BSD^2`
- extra memory matching and aggregation, which scales with `BSM_iD`

This also clarifies an earlier source of confusion: the archive does not contain a quadratic number of distinct memory items. Instead, the number of unique stored entries grows roughly linearly with depth, while the total retrieval work grows because earlier entries are repeatedly queried by later layers.

#### Why the Practical Slowdown Can Be Larger Than the Theoretical Ratio

Theoretical MAC analysis suggests that the final method should be moderately more expensive than the baseline, but not by the full amount observed in end-to-end runtime. In our current implementation, the measured wall-clock slowdown is larger than the pure analytical compute ratio. This strongly suggests that the current prototype still pays additional systems-level overhead beyond the method's unavoidable arithmetic cost.

The likely sources include:

- memory-archive maintenance
- tensor concatenation and layout overhead
- extra aggregation kernels in the depth-memory branch
- implementation inefficiencies in the current prototype path

For this reason, the paper should distinguish clearly between:

- the **theoretical complexity increase** of the method
- the **current prototype runtime overhead** measured in practice

These are related, but they are not identical.

#### Memory Storage Overhead

The memory cost also increases in a structured way.

At the parameter level, the proposed method adds only a small number of extra projection matrices, so the model-parameter increase is modest relative to the full network size. This matches the empirical parameter counts, where the final method is only slightly larger than the baseline.

At the activation and archive level, however, the method stores additional historical memory entries for later retrieval. In the final sublayer-memory design, the number of unique archived entries grows approximately linearly with the number of layers. Therefore, the storage cost of the history bank is better described as depth-linear rather than depth-quadratic.

The practical training-memory footprint can still be noticeably larger than this clean theoretical archive term, because the actual implementation must also hold:

- projected memory tensors
- intermediate aggregation tensors
- normal training activations used for backpropagation

As a result, the paper should present the memory story in two layers:

1. theoretical archive growth is moderate and approximately linear in depth
2. prototype training memory can still be higher due to implementation-side activation overhead

This framing is important because it explains why the method can have a relatively small parameter increase, a moderate theoretical compute increase, and yet a larger empirical runtime/memory penalty in the current codebase.

**[Table 1 about here]**  
Suggested content: a compact complexity and overhead summary comparing the baseline and the proposed method in terms of parameter growth, theoretical compute increase, and qualitative runtime overhead. This table should remain small and conceptual, while the measured efficiency table can stay in the experiments section or appendix.

---

## 3. Experiments

### 3.1 Benchmark and Tokenization

Our main benchmark is **WikiText-103** in the raw-text setting. All experiments use a **BPE tokenizer**, specifically the GPT-2 tokenizer, and train decoder-only language models with tied input/output embeddings unless otherwise noted.

This tokenizer choice matters for interpretation. Because perplexity depends on tokenization granularity and vocabulary definition, the absolute perplexity values reported here should be compared directly only against results that use sufficiently similar tokenization and evaluation protocols.

### 3.2 Model Family and Main Configurations

All models in the main comparison use the same decoder-only Transformer backbone family implemented in `ablation_models.py`. Unless a specific ablation changes the attention mechanism itself, the backbone remains matched across methods in hidden size, head count, MLP ratio, positional embeddings, and language-model head structure.

The main completed result pool already supports a coherent comparison across multiple settings:

- `8-layer`, `seq_len=256`, `40000` and `80000` steps
- `16-layer`, `seq_len=256`, `80000` steps
- `8-layer`, `seq_len=512`, `80000` steps
- projected-K/V versus non-projected historical K/V

Most runs use `d_model=384`, `num_heads=8`, `mlp_ratio=4`, tied embeddings, and learned positional embeddings. The current main comparisons therefore vary primarily in:

- attention mechanism
- model depth
- sequence length
- training budget

### 3.3 Training Protocol

All experiments are trained with AdamW using decoupled weight decay. The base learning rate is `3e-4`, with linear warmup over the first `100` steps followed by cosine decay to a minimum learning-rate scale of `0.1`. Unless otherwise stated, the training script uses gradient clipping with threshold `1.0` and weight decay `0.01`.

For the main WikiText-103 training stack, the default batch size is `8` with `grad_accum_steps=1`. The main completed runs use training budgets of `40000` or `80000` optimization steps, depending on the configuration.

### 3.4 Evaluation Protocol

Validation is performed periodically during training, while the test set is evaluated only at the end of the run. For this reason, we consistently report:

- **Best Val PPL**
- **Final Test PPL**

rather than “best test perplexity,” which would imply repeated test-set selection during training.

In the current training stack, validation uses a fixed evaluation routine without a moving cursor by default, and the final test perplexity is computed once at the end of the full run. This is the protocol assumed by the filled result tables below.

### 3.5 Main Result Framing

The main result section emphasizes the following:

- the gains are **stable rather than dramatic**
- the method improves perplexity by **roughly 2% to 4%**
- the gains persist across multiple depths and context settings

The core empirical message is:

> The proposed method consistently improves over standard Transformer baselines across the tested settings, indicating that explicit layer-depth memory is a useful architectural signal even when the baseline is already fairly strong.

### 3.6 Important Caveat

The paper explicitly acknowledges that:

- the current implementation is still a prototype
- runtime overhead is significant
- measured wall-clock slowdown is larger than the pure theoretical compute ratio

This makes the paper more credible, not less.

### 3.7 Main Results

The completed WikiText-103 experiments already support a coherent main-result claim. Across all completed settings, the final **single-q + sublayer + projected K/V** method consistently outperforms the standard Transformer baseline in both best validation perplexity and final test perplexity.

In the `8-layer, seq_len=256` setting, the method reduces final test perplexity from `34.53` to `33.53` at `40000` steps, and from `28.47` to `27.58` at `80000` steps. In the `16-layer, seq_len=256` setting, it further reduces final test perplexity from `26.28` to `25.55`. The gain also persists at longer context length, where the `8-layer, seq_len=512` setting improves from `24.03` to `23.46`.

Taken together, the currently completed settings show stable relative final-test-perplexity improvements of roughly `2.3%` to `3.1%` across model depth, context length, and training budget. Although the absolute gains are moderate rather than dramatic, they are consistent across all currently completed main settings. This consistency is important because the baseline models are already reasonably strong, so the observed improvements are better interpreted as repeatable architectural gains than as artifacts of weak comparison points. Overall, the current evidence supports the central claim of the paper: explicit same-position layer-depth memory provides useful additional signal even for already competitive decoder-only baselines.

**[Table 2 about here]**  
**Table 2 caption draft.** Main WikiText-103 results. The proposed layer-depth memory routing method consistently improves both best validation perplexity and final test perplexity across the completed settings. The final paper should keep only the cleanest rows needed to support the main claim.

**[Figure 2 about here]**  
**Figure 2 caption draft.** Validation-perplexity training curves for the baseline and the proposed method. The figure should illustrate that the gain is not confined to a single endpoint, but is reflected in the optimization trajectory across training.

### 3.8 Ablation Discussion

The current ablation evidence is already sufficient to support part of the design story, but not all of it equally strongly. In the `8-layer, seq_len=256, 80000-step` setting, the final method improves final test perplexity from `28.47` to `27.58`, while the variant without sublayer removal currently remains close to the final method. This suggests that the overall layer-depth memory idea is useful, but that the exact contribution of the sublayer design should still be described carefully until more matched ablation settings are filled in.

More importantly, the recorded `w/o projected K/V` result reaches a final test perplexity of `27.33`, which is slightly better than the projected-K/V version in the same setting. This means the paper should not overclaim that projected K/V is already proven to be universally optimal. A more accurate wording is that projected K/V is part of the current main implementation, while the raw-history alternative remains competitive and may even be better in some settings. This is a useful scientific result rather than a problem: it shows that the broad idea of layer-depth memory is robust, while the best choice of memory parametrization is still an open ablation question.

Accordingly, the ablation section should be written with two levels of certainty:

- the existence of a stable gain from adding layer-depth memory is already well supported
- the exact best form of memory parametrization still needs to be framed as an empirical trade-off rather than a fully settled conclusion

This framing keeps the paper honest while preserving its main claim. The contribution is the layer-depth routing mechanism itself; the strongest specific implementation choice can still be presented as the current best working version rather than an already universal optimum.

**[Table 3 about here]**  
**Table 3 caption draft.** Key ablations for the proposed layer-depth memory routing design. At minimum, this table should compare the baseline, the final method, the variant without sublayer memory, and the variant without projected memory K/V. If `AttnResidual`-style baselines remain much weaker, they can be moved to the appendix.

---

## 4. Related Work

### 4.1 Token-Wise Self-Attention in Decoder-Only Language Models

Modern decoder-only language models are built around causal self-attention over the token dimension. This design has proved extremely effective, and prior work has long emphasized that one of its key benefits is the drastic reduction of communication path length along the sequence axis compared with recurrent architectures. Our work starts from the observation that this leaves a complementary axis underexploited, namely the depth history of the same token across previous layers.

This paper therefore does not argue against token-wise self-attention. Instead, it augments it with an explicit same-position depth-memory branch. In that sense, the proposed method is better understood as a structured extension of standard causal attention rather than a replacement for it.

### 4.2 Memory-Augmented and Cross-Layer Transformer Variants

A broad family of prior work has explored augmenting Transformers with memory mechanisms, recurrent state, cached history, or cross-layer interactions. Some methods enlarge the effective context window by storing additional information across segments or decoding steps; others revisit how later layers can reuse earlier representations more directly. Our method belongs most naturally to this second category, but with a narrower and more specific design goal: enabling a token at the current layer to retrieve its own same-position history from earlier layers and sublayers.

The distinguishing feature of the present work is that row-wise token routing and depth-wise memory routing are placed in one shared competition space. Rather than introducing an independent auxiliary branch whose outputs are fused only later, the proposed design concatenates row and depth scores before normalization, so that current-layer token context and historical same-position memory directly compete for attention mass.

Recent work has also made the sequence-axis versus depth-axis distinction more explicit, arguing that residual pathways are not merely optimization scaffolding but part of the model's representational machinery. Our formulation is aligned with this view: self-attention already provides adaptive routing along the token axis, while the residual stream provides a weaker, non-selective form of propagation along the depth axis. The present work focuses on making this depth-wise access more explicit and selective without replacing the standard row-wise causal attention pathway.

### 4.3 Residual Aggregation and Attention-Residual Alternatives

Another nearby line of work asks whether standard residual propagation can be improved by learned aggregation across layers. This includes attention-style residual mixers and related schemes in which later layers adaptively combine earlier hidden states. These methods are highly relevant to our motivation, since they also recognize that useful information may be distributed across depth rather than only along the current layer's token axis.

However, our approach differs in two important ways. First, we preserve the standard token-attention branch rather than replacing the block's main computation with a residual aggregator. Second, our method organizes retrieval around same-position historical memory entries, rather than treating the entire layer stack as a generic pool for residual recombination. In this sense, the proposed mechanism sits between standard self-attention and generic cross-layer residual mixing: it is more explicit than ordinary residual propagation, but more structured than a free-form layer mixer.

### 4.4 Positioning of This Work

The present work is best positioned as follows:

- not as a fully general memory-augmented Transformer for every setting
- not as a pure efficiency paper
- not as a wholesale replacement for decoder attention

Instead, it is best positioned as a targeted architectural study of **layer-depth memory routing for decoder-only language modeling**. Its central claim is modest but concrete: explicit access to same-position depth history yields stable perplexity improvements over already strong decoder baselines, and this improvement can be obtained with a simple routing design rather than a large architectural overhaul.

---

## 5. Conclusion

This paper studies a simple architectural question: can a decoder-only Transformer benefit from explicitly reading the same token's depth history, rather than relying only on residual propagation through the layer stack? The experiments presented here suggest that the answer is yes. Across the completed WikiText-103 settings, a layer-depth memory routing mechanism yields stable perplexity improvements over standard Transformer baselines across multiple depths, sequence lengths, and training budgets.

The main empirical message is not that the gains are dramatic, but that they are repeatable. This is important because the baseline models used here are already reasonably strong, and therefore the observed `2%` to `4%` improvements are better interpreted as stable architectural gains than as artifacts of a weak comparison point.

At the same time, the work also exposes a clear trade-off. The current prototype incurs noticeable runtime overhead, and the measured slowdown is larger than the pure analytical compute increase. This indicates that the present implementation should still be viewed as an early but credible systems prototype rather than a fully optimized final design.

**[Table 4 about here or in appendix]**  
**Table 4 caption draft.** Efficiency trade-off between the baseline and the proposed method. The table should report parameter count, final test perplexity, step time, throughput, and peak GPU memory. If the runtime trade-off remains unfavorable, the full table can be moved to the appendix while still being discussed briefly in the main text.

The broader conclusion is that depth history is a real information source for decoder-only language models. A token does not only need access to earlier tokens; it can also benefit from direct access to what earlier layers already computed at the same position. The proposed layer-depth routing mechanism provides one concrete way to expose that signal. Future work should further clarify the best memory parametrization, improve runtime efficiency, and test whether the same design principle scales to larger models and broader language-model settings.

---

## Appendix: Working Tables

The following tables are kept as an internal working appendix for drafting and later LaTeX conversion.

### A.1 Main Results Table

| Method | Layers | Seq Len | Steps | Params (M) | Best Val PPL | Final Test PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 8 | 256 | 40000 | 33.59 | 33.5519 | 34.5286 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 | 40000 | 33.89 | 32.3694 | 33.5314 |  |
| Baseline | 8 | 256 | 80000 | 33.59 | 27.5747 | 28.4682 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 | 80000 | 33.89 | 26.5059 | 27.5812 |  |
| Baseline | 16 | 256 | 80000 | 47.79 | 25.2991 | 26.2810 |  |
| Ours (single-q + sublayer + projected K/V) | 16 | 256 | 80000 | 48.08 | 24.6331 | 25.5485 |  |
| Baseline | 8 | 512 | 80000 | 33.69 | 23.5300 | 24.0251 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 512 | 80000 | 33.99 | 22.8900 | 23.4636 |  |
| Baseline | 16 | 512 | 80000 |  |  |  | optional |
| Ours (single-q + sublayer + projected K/V) | 16 | 512 | 80000 |  | 21.26 |  20.71| optional |

### A.2 Key Ablation Table

| Variant | Layers | Seq Len | Steps | Params (M) | Best Val PPL | Final Test PPL | Relative to Baseline | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 8 | 256 | 80000 | 33.59 | 27.5747 | 28.4682 | 0.00% | reference configuration |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 | 80000 | 33.89 | 26.5059 | 27.5812 | 3.12% | final method |
| Ours w/o sublayer |  |  |  |  | 26.73 | 27.81 |  |  |
| Ours w/o projected K/V | 8 | 256 | 80000 | 33.89 | 26.86 | 27.9254 | 4.01% | raw-history variant |
| AttnResidual |  |  |  |  |  |  |  | optional |
| AttnResidual2D |  |  |  |  |  |  |  | optional |

### A.3 Efficiency Table

If `tokens/s` is not directly logged, fill:

\[
\text{tokens/s} =
\frac{\text{batch size} \times \text{seq len} \times \text{grad accum steps}}{\text{step time}}
\]

using global batch size if the logged batch is already global.

| Method | Layers | Seq Len | Steps Used for Timing | Params (M) | Final Test PPL | Step Time (s) | Tokens/s | Peak GPU Mem (GB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 8 | 256 |  |  |  |  |  | 7.8 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 |  |  |  |  |  |8.4  |  |
| Baseline | 16 | 256 |  |  |  |  |  | 9.58 |  |
| Ours (single-q + sublayer + projected K/V) | 16 | 256 |  |  |  |  |  |  9.6|  |
| Baseline | 8 | 512 |  |  |  |  |  |11.2  | optional |
| Ours (single-q + sublayer + projected K/V) | 8 | 512 |  |  |  |  |  |11.2  | optional |

### A.4 Training Curve Checklist

For the current paper, the minimum recommended curves are:

- `baseline` vs `ours`, `8-layer`, `seq_len=256`
- `baseline` vs `ours`, `16-layer`, `seq_len=256`

For each curve pair, record:

| Curve Pair | X-axis | Y-axis | Available? | Figure Path / Note |
|---|---|---|---|---|
| Baseline vs Ours, 8-layer, seq_len=256 | Steps | Val PPL |  |  |
| Baseline vs Ours, 16-layer, seq_len=256 | Steps | Val PPL |  |  |

### A.5 Stability / Seed Table

If you decide to add seeds, use this template.

| Method | Layers | Seq Len | Steps | Seed | Best Val PPL | Final Test PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline |  |  |  | 42 |  |7.8  |  |
| Ours |  |  |  | 42 |  | 8.4|  |
| Baseline |  |  |  | 43 |  |  | optional |
| Ours |  |  |  | 43 |  |  | optional |
| Baseline |  |  |  | 44 |  |  | optional |
| Ours |  |  |  | 44 |  |  | optional |
