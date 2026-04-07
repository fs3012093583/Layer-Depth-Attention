# Layer-Depth Memory Routing for Decoder-Only Transformers

## Draft Status

This file is the **mother draft** for both:

- a **submission version** with a narrower main story and tighter evidence
- an **arXiv version** with fuller appendices, extra curves, and more ablation details

The current final method assumed in this draft is:

- **single-q**
- **sublayer memory**
- **projected K/V**

That is, the model uses one query for both row-token attention and depth-memory lookup, stores sublayer-level history, and archives historical memory using projected K/V rather than raw hidden states.

---

## 1. Working Title Options

### Option A
**Layer-Depth Memory Routing for Decoder-Only Transformers**

### Option B
**Single-Query Layer-Depth Memory for Decoder-Only Language Models**

### Option C
**Improving Decoder-Only Transformers with Layer-Depth Memory Routing**

Recommended current choice:

> **Layer-Depth Memory Routing for Decoder-Only Transformers**

This is broad enough for both the submission version and the arXiv version, while still matching the current experimental scope.

---

## 2. Abstract Draft

Decoder-only Transformers propagate information effectively along the token dimension, but access to intermediate representations formed at earlier layers remains indirect. We study whether a token at the current layer can benefit from directly reading its own depth history, rather than relying only on residual propagation through the stack. To this end, we introduce a layer-depth memory routing mechanism that augments standard causal self-attention with a same-position cross-layer memory branch. Our final design uses a single shared query for row-wise token attention and depth-wise memory retrieval, stores sublayer-level history, and archives historical states through projected keys and values. Across multiple WikiText-103 settings, including 8-layer and 16-layer models, different sequence lengths, and long training budgets, the proposed method consistently improves perplexity over strong Transformer baselines by roughly 2% to 4%. These results suggest that depth history is a useful information source even for already strong decoder baselines, but also reveal a clear efficiency trade-off: the current prototype incurs higher runtime cost than standard self-attention. Overall, the results position layer-depth memory routing as a stable and promising direction for improving decoder-only language models, while leaving memory-path optimization as an important future step.

---

## 3. One-Paragraph Paper Positioning

This paper should be positioned as a **mechanism paper with stable but moderate gains**, not as a sweeping replacement for standard Transformer attention.

The main claim should be:

> A decoder-only Transformer can benefit from explicitly reading same-token cross-layer history, and a single-query, sublayer-level, projected-K/V memory design yields stable perplexity improvements across multiple settings.

The paper should **not** claim:

- that the method dominates all possible alternatives
- that the method is already fully optimized in runtime
- that the gains are large in absolute terms

Instead, the honest and credible framing is:

- the gains are **small-to-moderate but stable**
- the effect persists across depth and training budget
- the efficiency gap is real, but current runtime appears worse than the pure theoretical complexity gap, suggesting implementation overhead still matters

---

## 4. Introduction Draft

Transformer language models are built around token-wise attention, where each layer reads contextual information from other positions in the sequence. This design has proved extremely successful, but it leaves a separate source of information only indirectly accessible: the intermediate representations that the same token has formed across previous layers. In standard decoder-only Transformers, such depth information can only reach later layers through repeated residual propagation. The model therefore has no explicit mechanism for saying: *for this token, retrieve what earlier layers already computed at the same position*.

This observation motivates a simple question: can a decoder-only language model improve by augmenting token attention with a direct layer-depth memory branch? If the current token could jointly attend to both the usual row-wise prefix context and its own cross-layer history, then the model might reuse partially refined intermediate states more effectively instead of repeatedly reconstructing them through the stack.

In this work, we study this question through a controlled family of cross-layer memory mechanisms for decoder-only Transformers. Our final design uses a single query to route both row-wise token attention and depth-wise memory retrieval, stores memory at the sublayer level, and archives historical states through projected keys and values. The resulting mechanism extends each attention step from pure token interaction to a joint token-depth routing problem.

The resulting gains are not dramatic, but they are stable. Across multiple WikiText-103 settings, including 8-layer and 16-layer models, different sequence lengths, and long training budgets, the proposed method consistently improves perplexity over strong Transformer baselines by roughly 2% to 4%. At the same time, the method introduces nontrivial runtime overhead, and the measured slowdown is larger than the pure theoretical compute gap, indicating that the current prototype implementation is not yet fully optimized.

These results support a restrained but meaningful conclusion: explicit depth-history access is useful for decoder-only language modeling, and it can yield repeatable improvements even on already strong baselines. However, the method should currently be viewed as a promising architectural direction rather than a fully optimized replacement for standard self-attention.

### Contributions

The current paper should claim the following contributions:

1. We formulate decoder-only attention as a **joint row-depth routing problem**, where the current token can read both prefix tokens and its own cross-layer history.
2. We introduce a practical final design based on **single-query routing, sublayer memory, and projected historical K/V**, and show that this combination gives the most stable results among the tested variants.
3. We show on WikiText-103 that the method yields **consistent 2% to 4% perplexity improvements** across multiple settings, including 8-layer and 16-layer models and different sequence lengths.
4. We analyze the method’s cost and show that while the theoretical extra compute is moderate, the current prototype incurs additional systems-level overhead, highlighting an important optimization direction for future work.

---

## 5. Method Draft

### 5.1 Problem Setup

Consider a decoder-only Transformer with `L` layers. At layer `l`, the hidden state at token position `t` is denoted by `x_t^(l)`. Standard causal self-attention allows `x_t^(l)` to read row-wise context from the prefix positions `{1, ..., t}` in the same layer. However, it does not explicitly expose the cross-layer history `{x_t^(1), ..., x_t^(l-1)}` for direct retrieval.

We aim to augment standard self-attention with a depth-memory branch that allows the current token to retrieve same-position information formed at previous layers.

### 5.2 Row Attention and Depth Memory

For each layer, we keep the usual row-wise token attention branch over the current sequence. In parallel, we maintain a depth-memory archive consisting of same-position history from earlier layers or sublayers. The current token then produces a single query that is used for two routing decisions:

- row-wise routing over current-layer prefix tokens
- depth-wise routing over same-position historical memory

The row branch and depth branch produce scores that are concatenated and normalized jointly, so the model allocates attention mass across both sources within one competition space.

### 5.3 Final Design Used in This Paper

The final method used in the current draft has three defining design choices.

#### Single-Q

The same query representation is used for both row attention and depth-memory lookup. This avoids introducing an additional query branch and gives a simpler routing mechanism than dual-query variants.

#### Sublayer Memory

Instead of storing only one memory item per Transformer block, we cache history at the sublayer level. In the current implementation, this means that attention-side and FFN-side intermediate states can both contribute memory entries. This produces a richer depth archive than block-only storage.

#### Projected K/V

Historical memory is archived through projected keys and values rather than leaving it in raw hidden-state form. This gives the current layer a learned memory space for matching and aggregation, and empirically performs more stably than removing projection entirely.

### 5.4 Why This Combination

The current experimental evidence suggests the following:

- **single-q** is simpler and sufficient for stable gains
- **sublayer memory** improves the usefulness of the depth archive relative to coarser block-only memory
- **projected K/V** performs more reliably than directly using raw hidden states as memory keys and values

This is why the final paper should treat **single-q + sublayer + projected K/V** as the main method, and treat the other variants as supporting ablations rather than separate primary methods.

---

## 6. Experiments Section Plan

### 6.1 Benchmark

The main benchmark should be positioned as:

- **WikiText-103**
- **BPE tokenizer**
- decoder-only language modeling

The paper should explicitly note that perplexity values are tokenizer-dependent, so cross-paper absolute comparisons are only meaningful when tokenizer and evaluation protocol are sufficiently aligned.

### 6.2 Main Result Axes Already Available

The current result pool already supports a coherent main-experiment story:

- `8-layer`, `seq_len=256`, `40000` and `80000` steps
- `16-layer`, `seq_len=256`, `80000` steps
- `seq_len=512` comparisons
- projected-K/V versus non-projected historical K/V

This is already stronger than a single-point benchmark story because the gains appear across multiple settings rather than only one run.

### 6.3 Main Result Narrative

The main result section should emphasize:

- the gains are **stable rather than dramatic**
- the method improves perplexity by **roughly 2% to 4%**
- the gains persist across multiple depths and context settings

The best current wording is:

> The proposed method consistently improves over standard Transformer baselines across the tested settings, indicating that explicit layer-depth memory is a useful architectural signal even when the baseline is already fairly strong.

### 6.4 Important Caveat

The paper should explicitly state that:

- the current implementation is still a prototype
- runtime overhead is significant
- measured wall-clock slowdown is larger than the pure theoretical compute ratio

This makes the paper more credible, not less.

---

## 7. Minimum Tables and Figures

The minimum viable **submission version** should contain:

### Table 1: Main Results

Columns:

- Method
- Layers
- Seq Len
- Steps
- Params
- Best Val PPL
- Best Test PPL

Rows should include at least:

- `baseline`
- `attn_residual`
- `attn_residual_2d`
- `single-q + sublayer + projected K/V` (ours)

Only keep the strongest, cleanest settings in the main paper.

### Table 2: Key Ablations

Recommended ablations:

- final method
- without sublayer memory
- without projected K/V
- baseline

The goal is to justify why the final method is the final method.

### Table 3: Efficiency Trade-off

Columns:

- Method
- Params
- Test PPL
- Step Time
- Tokens/s
- Peak GPU Memory

This table does not need to show that the method is efficient. It only needs to honestly quantify the trade-off.

### Figure 1: Training Curves

At least one plot:

- baseline vs final method
- validation PPL versus training steps

This is especially useful because earlier observations suggested that different methods can differ meaningfully in optimization trajectory.

---

## 8. Submission Version vs arXiv Version

### Submission Version

Keep it focused:

- one main method
- limited but clean main results
- one key ablation table
- one efficiency table
- one training-curve figure

Avoid dumping every exploratory branch into the main text.

### arXiv Version

Add more material in appendix or supplementary sections:

- more training curves
- more experimental settings
- extra variants that worked but are not the final method
- more implementation discussion
- theoretical complexity derivation
- more detailed efficiency discussion

The arXiv version should be broader, but still have the same main claim as the submission version.

---

## 9. Recommended Next Writing Steps

### Immediate

1. Replace placeholders in the main results section with the actual selected numbers.
2. Write the experimental setup subsection with the exact tokenizer, optimizer, batch size, sequence length, and evaluation protocol.
3. Draft the efficiency paragraph with careful wording that distinguishes theoretical overhead from prototype implementation overhead.

### Next

4. Add the key ablation subsection.
5. Add one training-curve figure.
6. Draft Related Work around:
   - cross-layer attention
   - memory-augmented Transformers
   - residual aggregation / Attention Residuals

### Before Submission

7. Split this mother draft into:
   - a **submission draft**
   - an **arXiv-expanded draft**

---

## 10. Current Risks To Address In Writing

These should be handled explicitly in the paper:

- gains are moderate, not huge
- runtime overhead is real
- current implementation is not fully optimized
- tokenizer choice makes absolute perplexity comparisons delicate

If these are written openly and carefully, they do not weaken the paper. They make it look more rigorous.

---

## 11. One-Sentence Paper Summary

> We propose a single-query, sublayer-level layer-depth memory routing mechanism for decoder-only Transformers, and show that it delivers stable 2% to 4% perplexity improvements across multiple WikiText-103 settings while introducing a measurable but still-optimizable runtime cost.

---

## 12. Fill-In Experiment Tables

Use this section as the working table area for both the submission version and the arXiv version. Fill the numbers here first, then later move the final selected rows into the polished paper tables.

### 12.1 Main Results Table

| Method | Layers | Seq Len | Steps | Params (M) | Last Step | Last Val PPL | Last Test PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 8 | 256 | 40000 | 33593472 |  |33.5519  | 34.5286 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 | 40000 | 3.3889e+7 |  |32.3694  | 33.5314 |  |
| Baseline | 8 | 256 | 80000 |  |  | 27.5747 | 28.4682 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 | 80000 |  |  |  26.5059 |27.5812 |  |
| Baseline | 16 | 256 | 80000 | 4.7789e+7 |  | 25.2991 | 26.281 |  |
| Ours (single-q + sublayer + projected K/V) | 16 | 256 | 80000 |4.8085e  |  |24.6331  | 25.5485 |  |
| Baseline | 8 | 512 | 80000 | 3.3692e+7 |  |23.53  | 24.0251 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 512 | 80000 |  |  | 22.89 |23.4636  |  |
| Baseline | 16 | 512 | 80000 |  |  |  |  | optional |
| Ours (single-q + sublayer + projected K/V) | 16 | 512 | 80000 |  |  |  |  | optional |

### 12.2 Key Ablation Table

| Variant | Layers | Seq Len | Steps | Params (M) | Best Val PPL | Best Test PPL | Relative to Baseline | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |  |  |  |
| Ours (single-q + sublayer + projected K/V) |  |  |  |  |  |  |  | final method |
| Ours w/o sublayer |  |  |  |  |  |  |  |  |
| Ours w/o projected K/V | 8 |  256|80000  |3.3889e+7  | 26.3196 | 27.3254 |  | raw-history variant |
| AttnResidual |  |  |  |  |  |  |  | optional |
| AttnResidual2D |  |  |  |  |  |  |  | optional |

### 12.3 Efficiency Table

If `tokens/s` is not directly logged, fill:

\[
\text{tokens/s} =
\frac{\text{batch size} \times \text{seq len} \times \text{grad accum steps}}{\text{step time}}
\]

using global batch size if the logged batch is already global.

| Method | Layers | Seq Len | Steps Used for Timing | Params (M) | Best Test PPL | Step Time (s) | Tokens/s | Peak GPU Mem (GB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 8 | 256 |  |  |  |  |  | 7.8 |  |
| Ours (single-q + sublayer + projected K/V) | 8 | 256 |  |  |  |  |  |8.4  |  |
| Baseline | 16 | 256 |  |  |  |  |  | 9.58 |  |
| Ours (single-q + sublayer + projected K/V) | 16 | 256 |  |  |  |  |  |  9.6|  |
| Baseline | 8 | 512 |  |  |  |  |  |11.2  | optional |
| Ours (single-q + sublayer + projected K/V) | 8 | 512 |  |  |  |  |  |11.2  | optional |

### 12.4 Training Curve Checklist

For the current paper, the minimum recommended curves are:

- `baseline` vs `ours`, `8-layer`, `seq_len=256`
- `baseline` vs `ours`, `16-layer`, `seq_len=256`

For each curve pair, record:

| Curve Pair | X-axis | Y-axis | Available? | Figure Path / Note |
|---|---|---|---|---|
| Baseline vs Ours, 8-layer, seq_len=256 | Steps | Val PPL |  |  |
| Baseline vs Ours, 16-layer, seq_len=256 | Steps | Val PPL |  |  |

### 12.5 Stability / Seed Table

If you decide to add seeds, use this template.

| Method | Layers | Seq Len | Steps | Seed | Best Val PPL | Best Test PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline |  |  |  | 42 |  |7.8  |  |
| Ours |  |  |  | 42 |  | 8.4|  |
| Baseline |  |  |  | 43 |  |  | optional |
| Ours |  |  |  | 43 |  |  | optional |
| Baseline |  |  |  | 44 |  |  | optional |
| Ours |  |  |  | 44 |  |  | optional |
