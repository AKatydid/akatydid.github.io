---
layout: post
title: CUDA-Operators-1-sgemv
date: 2025-09-13 00:45 +0000
categories: [CUDA]
tags: [CUDA]
pin: false
math: true
mermaid: true
---

本系列文章重点阐述了各类算子的逐步优化过程，涵盖 CUDA 常用算子，并对不同算子的性能瓶颈进行分析。各类算子完整代码请参考个人仓库 [OpenKernels](https://github.com/AKatydid/OpenKernels.git)。

## 1.SGEMV
SGEMV 表达式为：$Y = A * X$，其中 A(M, K)，X(K, 1)，Y(M,1)，具体计算流程如图 *Fig 1* 所示。首先，以 `K=16` 和 `K=128` 两种特殊情况，展示如何优化 GEMV 算子。
最 Naive 的优化为一个线程算 Y 的一个数，即每个线程从 global mem 读取 A 的一行向量，再读取 X 向量，两个向量进行内积。数据利用率低，访存效率低。

![Desktop View](/assets/img/blog/CUDA/gemv1.png){: width="200" height="350" }
<center>Fig 1. GEMV 计算流程</center>

### 1.1 K=16


### 1.2 K=128


### 1.3 Large K




## reference
[1] DefTruth, Many Others. LeetCUDA: A Modern CUDA Learn Notes with PyTorch for Beginners. 2025. https://github.com/xlite-dev/LeetCUDA.git.
[2] 深入浅出GPU优化系列：gemv优化. https://zhuanlan.zhihu.com/p/494144694
