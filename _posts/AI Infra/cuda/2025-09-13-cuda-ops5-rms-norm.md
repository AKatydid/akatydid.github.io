---
layout: post
title: CUDA-Operators-5-RMSNorm
date: 2025-09-13 08:49 +0000
categories: [CUDA]
tags: [CUDA]
pin: false
math: true
mermaid: true
---

## RMSNorm

RMSNorm 是一种归一化操作，使用 均方根（Root Mean Square, RMS）实现归一化，计算公式为：$RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2} +\epsilon }$，代码表示如下。

```python
x = x / torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim=True) + 1e-5)
```

### Naive 实现



## Reference
[1] DefTruth, Many Others. LeetCUDA: A Modern CUDA Learn Notes with PyTorch for Beginners. 2025. https://github.com/xlite-dev/LeetCUDA.git.
