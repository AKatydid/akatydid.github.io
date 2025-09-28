---
layout: post
title: CUDA-Operators-6-Cumsum
date: 2025-09-28 06:57 +0000
categories: [CUDA]
tags: [CUDA]
pin: false
math: true
mermaid: true
---

## 1.Cumsum 理论
Prefix Sum 定义为：$y_i = \sum_0^i x_i$，在 CPU 上简单实现如下所示。

```c
void PrefixSum(const int32_t* input, size_t n, int32_t* output) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}
```

Cumsum 计算过程中，每个元素依赖上一个元素的值，如何并行计算呢？

![Desktop View](/assets/img/blog/CUDA/cumsum-t.png){: width="800" height="500" }
<center>Fig 1. cumsum 算法示例</center>

图 1 展示了当代前缀扫描算法相关的典型扫描网络结构。复杂度来源于“深度 + 规模”。具体内容可参考论文 [1]。
* 图 1(a) 没有并行机会，但最小 n-1 的规模在处理器计算能力远大于输入规模时，极具吸引力。通过增加线程计算粒度（每个线程多处理几个数）来减少线程间通信，是提升处理器利用率的常用技术。
* 图 1(b) 规模为低效的 $O(nlog_2n)$，但较浅的深度和简单的共享内存地址计算，是 warp 实现的理想策略。
* 图 1(c) 同样以 $O(nlog_2n)$ 规模实现了最小深度 $log_2n$。
* 图 1(d) 是一种高效的策略，具有$2log_2n$ 的深度和 $O(n)$ 的规模，其数据流在视觉上呈现"沙漏"形态，包含：(1) 并行度逐渐减小的归约阶段，(2) 并行度逐渐增加的向下传播。
* 图 1(e) 将向下传播的行为从传播转换为了扫描，先归约后扫描，需要一次额外的 $O(n)$ 开销。

当输入规模不超过处理器一次能并行处理的宽度时（如输入长度 ≤ 32），性能取决于**深度**。而对于大规模问题，每层涉及大量同步、内存访问、调度，最小**规模**网络则更为关键。

### 1.1 blcok 级扫描策略
基于上述算法和分析，block 级的策略一般有如下 3 种，具体如图 2 所示，图中以 warp = 4 展示了 blcok 级的扫描策略，虚线表示基于 shared mem 实现的同步。

![Desktop View](/assets/img/blog/CUDA/cumsum-b.png){: width="800" height="500" }
<center>Fig 2. cumsum block 级策略</center>


### 1.2 global 级扫描策略

Reduce-then-scan 策略，先调度 block 级归约内核，再调度 block 级扫描网络内核。如图 3 所示，输入被均匀划分为 G 个 block（G为处理器可同时驻留的块数，与n无关），分为 3 个内核实现。
* upsweep 中，各线程块以迭代串行方式归约其分块，随后在 root scan 中对 G 个块聚合结果进行扫描。
* downsweep 中，各线程块基于块聚合扫描生成的块前缀，迭代计算分块的前缀扫描。

![Desktop View](/assets/img/blog/CUDA/cumsum-g1.png){: width="800" height="500" }
<center>Fig 3. 对 G 个 block 采用 3 内核实现 “reduce-then-scan” 并行化方式（~ 3n 次全局访存） </center>

如图 4 所示，链式扫描并行化中，每个线程块被分配一个输入数据块，且线程块之间存在串行依赖链。**各线程块需等待其前驱的 block 前缀计算完成。** 链式扫描的性能受限于线程块之间的信号传播延迟，会严重限制吞吐。

解决方案是增加 block size，但是这又会导致片上资源紧张，难以权衡。

![Desktop View](/assets/img/blog/CUDA/cumsum-g2.png){: width="800" height="500" }
<center>Fig 4. 一遍的 "chained-scan" 前缀扫描 (~ 2n 次全局访存)</center>

如图 5 所示，NVIDIA 提出的方法对 block 规模不敏感，能适配多样化的架构配置。
![Desktop View](/assets/img/blog/CUDA/cumsum-g3.png){: width="800" height="500" }
<center>Fig 5. </center>


## 2.CUDA 实现




## reference
[1] [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)

[2] [CUB](https://github.com/NVIDIA/cub)

[3] [CUDA高性能计算经典问题（二）—— 前缀和（Prefix Sum）](https://zhuanlan.zhihu.com/p/423992093)
