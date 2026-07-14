---
layout: post
title: Attention (2) - FA & Other Variants
date: 2026-04-29 15:59 +0800

categories: [sglang]
tags: [Attention]
pin: false
math: true
mermaid: true
---

注意力机制 (Attention Mechanism) [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3) 是自然语言处理领域的重要概念之一。
Attention 使模型能够在**不受输入或输出序列中元素距离影响的情况下**建模它们之间的依赖关系。
Bahdanau 等人 [[2]](#ref-2) 首次将 Attention 引入自然语言处理，随后由 Transformer [[1]](#ref-1) 发扬光大，成为当代大语言模型的核心技术之一。


## reference
<span id="ref-1"></span>[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in neural information processing systems, 2017, 30.

<span id="ref-2"></span>[2] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.

<span id="ref-3"></span>[3]  Kim Y, Denton C, Hoang L, et al. Structured attention networks[J]. arXiv preprint arXiv:1702.00887, 2017.
