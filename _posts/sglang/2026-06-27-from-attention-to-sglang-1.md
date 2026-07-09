---
layout: post
title: From Attention to SGLang (1) -- Attention Mechanism
date: 2026-06-27 15:59 +0800

categories: [sglang]
tags: [sglang, infra]
pin: false
math: true
mermaid: true
---

注意力机制 (Attention Mechanism) [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3) 是自然语言处理领域的重要概念之一。
Attention 使模型能够在**不受输入或输出序列中元素距离影响的情况下**建模它们之间的依赖关系。
Bahdanau 等人 [[2]](#ref-2) 首次将 Attention 引入自然语言处理，随后由 Transformer [[1]](#ref-1) 发扬光大，成为当代大语言模型的核心技术之一。

随着大语言模型规模不断增长，Attention 的计算与存储开销逐渐成为推理阶段的主要瓶颈。围绕这一问题，研究者提出了一系列 Attention 变体。
- MHA (Multi-Head Attention)：多头注意力
- MQA (Multi-Query Attention)：多查询注意力
- GQA (Grouped-Query Attention)：分组查询注意力
- MLA (Multi-Latent Attention)：多潜变量注意力
- ...

这些方法虽然都属于 Attention 的不同实现形式，但它们在计算复杂度、KV Cache 占用以及推理效率方面存在显著差异，直接影响了vLLM、SGLang等推理框架的系统设计。

作为 “From Attention to SGLang” 系列的第一篇文章，本文将从最基础的 Scaled Dot-Product Attention 出发，逐步介绍 MHA、MQA、GQA 和 MLA 的设计思想、数学原理及实现方式，并分析它们在推理场景中的优缺点。
在理解Attention机制之后，将逐步引入 SGLang 的 KV Cache、Schedule 以及 Kernel 等核心组件。

## Scaled Dot-Product Attention

Attention 本质是在计算

输入：词序 $X=(x_1, x_2, ..., x_n)$ 

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$


## GQA


## MLA


## reference
<span id="ref-1"></span>[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in neural information processing systems, 2017, 30.

<span id="ref-2"></span>[2] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.

<span id="ref-3"></span>[3]  Kim Y, Denton C, Hoang L, et al. Structured attention networks[J]. arXiv preprint arXiv:1702.00887, 2017.
