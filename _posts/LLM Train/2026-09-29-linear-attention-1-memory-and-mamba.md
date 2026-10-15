---
layout: post
title: 'Linear Attention (1)：有限状态记忆、线性复杂度与 Mamba'
date: 2026-09-29 09:00 +0000
categories: [LLM]
tags: [LLM, Linear Attention, Mamba, SSM]
pin: false
math: true
---

> **系列导读**：本文介绍 Linear Attention 的基本递推，以及 Mamba 与它的关系。第二篇讲 DeltaNet、Gated DeltaNet（GDN）和 Qwen3-Next；第三篇讲 Kimi Delta Attention（KDA）；第四篇分析 SGLang CUDA 路径，并以 Qwen3-Next 做 roofline 估算。

## 核心结论

* **问题**：softmax attention 的计算随上下文长度按 $T^2$ 增长，KV cache 按 $T$ 增长。
* **动机**：降低长上下文推理成本，不能只优化 kernel；必须避免每步都携带、访问完整历史。
* **方法**：Linear Attention 将历史关联压缩到固定形状状态 $S_t$，每个 token 更新状态，由当前 query 读取：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad
o_t\propto S_t^\top\phi(q_t).
$$

**核心 tradeoff：用固定大小的记忆降低长上下文成本，同时接受压缩带来的信息损失。**

这样生成时维护的状态大小不随历史增长，训练也可以按块并行计算。需要注意的是，这份摘要会丢失细节：相似或冲突的信息可能混在一起，主题变化时旧内容也不会自动清除。后续方法分别尝试修正冲突信息、清理过时记忆；需要精确回查时，混合模型仍保留全局 attention。

## 从二次交互到矩阵状态

对一个 head，标准 causal attention 可写为

$$
o_t=\sum_{i=1}^{t}\operatorname{softmax}_i(q_t^\top k_i) v_i.
$$

每个位置都要看历史位置，显式 attention score 矩阵的规模是 $T\times T$。FlashAttention 大幅降低了中间矩阵的读写，但没有改变精确 attention 对历史 KV 的依赖。

Linear Attention 的经典出发点（Katharopoulos 等，2020）是用 feature map $\phi$ 替换 softmax kernel，并利用乘法结合律：

$$
\sum_{i\le t} \phi(q_t)^\top\phi(k_i)v_i
=\phi(q_t)^\top\left(\sum_{i\le t}\phi(k_i)v_i^\top\right).
$$

定义累计状态 $S_t=\sum_{i\le t}\phi(k_i)v_i^\top$，分母另用 $z_t=\sum_{i\le t}\phi(k_i)$ 做归一化，就得到

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),\qquad
o_t=\frac{S_t^\top\phi(q_t)}{z_t^\top\phi(q_t)}.
$$

于是，历史不再以长度为 $t$ 的 token 列表进入下一步，而被汇总到固定形状的矩阵 $S_t$ 和向量 $z_t$。对固定 head dimension，序列方向的工作量随长度线性增长；每步的状态读写大小不随上下文长度增长。投影、head 数和状态矩阵本身的维数仍然有成本，不能把“linear”误读成每一步只做常数 FLOPs。

![Linear Transformers 论文中的时间与内存复杂度对比](/assets/img/blog/Linear%20Attnetion/linear-transformers-figure-1.jpg){: width="700" }
<center>图 1. 时间与显存复杂度对比。<a href="#ref-1">[1]</a></center>

## 状态表示与关联记忆

暂时忽略 feature map 和归一化，令 $S_t\in\mathbb{R}^{d_k\times d_v}$，最简单的关联记忆更新为

$$
S_t=S_{t-1}+k_t v_t^\top,\qquad o_t=S_t^\top q_t.
$$

每个 token 把一条 $k_t\mapsto v_t$ 关联以外积方式写入矩阵。查询 $q_t$ 与各个 key 方向的匹配强度，决定读取多少对应 value。

这个视角也揭示了第一个核心问题：**状态容量有限，而写入默认只增不减。** 多个关联会叠加在同一矩阵中，久而久之发生干扰；模型很难精确覆盖旧记忆或清理已经无关的信息。线性复杂度只是预算，记忆管理才决定这个预算能否被有效利用。

下面的朴素实现只为展示递推语义；它在 Python 中逐 token 循环，不是高性能实现。实际训练需要把序列改写为并行或 chunkwise 矩阵运算。

```python
import torch


def linear_attention_recurrent(q, k, v):
    """Single-head, unnormalized feature-map linear attention."""
    seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(key_dim, value_dim, device=q.device, dtype=q.dtype)
    outputs = []

    for token in range(seq_len):
        state = state + torch.outer(k[token], v[token])
        outputs.append(q[token] @ state)

    return torch.stack(outputs)
```

这里省略了常见的 $\phi$ feature map、分母归一化、head batching 和数值稳定处理。它的目的只有一个：让“把历史写进 state，再由 query 读取”一眼可见。

## 分块训练与递归推理

递推形式天然适合 autoregressive decode：每个新 token 只需要上一步状态，所以状态大小固定。但若训练也严格按时间步串行，GPU 就很难充分并行。

一个关键工程解法是 **chunkwise recurrence**：把序列切成块，块内把多个更新展开成矩阵乘法并行算，块与块之间只传递边界状态。对块大小 $C$，实现会在“块内矩阵工作”和“块间状态递推”之间折中。它不是近似地忽略块内因果关系，而是把递推代数化、分块执行。

这也是为什么比较架构时不能只看一个“序列复杂度”数字。Prefill 的目标是把长输入吞吐喂满，常用 chunk kernel；decode 的目标是低延迟地更新已有状态，常用 fused recurrent kernel。两者数学上对应同一递推，但 kernel、并行维度和数据搬运并不相同。

## Mamba 与线性注意力的关系

**严格分类：Mamba 是 selective State Space Model（SSM），不是经典 kernelized Linear Attention。** 二者都可以用固定大小的递归状态处理序列，生成时都具有线性于序列长度的总成本，因此宽泛讨论高效序列模型时常被放在一起。

差异在于状态更新的来源和结构：Linear Attention 从 $Q/K/V$ 关联记忆出发；Mamba 从状态空间方程出发，让离散化后的 SSM 参数依赖输入，以输入选择性地传播或遗忘状态。Mamba 的重点是 selective scan 和硬件友好的扫描算法，不是把 softmax attention 的核函数替换成 $\phi$。

![Mamba 论文中的选择性状态空间模型结构](/assets/img/blog/Linear%20Attnetion/mamba-paper-figure-1.jpg){: width="800" }
<center>图 2. Mamba 的选择性状态空间结构。<a href="#ref-4">[4]</a></center>

![Mamba 论文中的 H3、Gated MLP 与 Mamba block 对比](/assets/img/blog/Linear%20Attnetion/mamba-paper-figure-3.jpg){: width="800" }
<center>图 3. Mamba block 架构。<a href="#ref-4">[4]</a></center>

Mamba-2 的 **State Space Duality（SSD）** 进一步展示了结构化 SSM 与某些注意力形式之间的数学联系，并把核心计算整理成易于使用矩阵乘法的块算法 [<a href="#ref-6">6</a>]。因此论文会说“Transformers are SSMs”或把某些 SSM 写成线性注意力形式；这指的是特定结构下的等价/对偶，不代表 Mamba-1 就是经典线性注意力。

可以用这张表快速定位术语：

| 名称 | 固定状态递推 | 关联记忆式 Q/K/V | 关键机制 |
| --- | --- | --- | --- |
| Kernelized Linear Attention | 是 | 是 | feature map 将历史关联累积到矩阵状态 |
| DeltaNet / GDN / KDA | 是 | 是 | 对关联记忆做有目标的修正，并逐步增加遗忘控制 |
| Mamba | 是 | 通常不按经典 Q/K/V 关联记忆定义 | 输入选择性 SSM 与 selective scan |
| Mamba-2 / SSD | 是 | 存在重要数学对偶与块分解联系 | 用结构化状态转移实现高效块矩阵计算 |

## 技术发展脉络

| 时间 | 代表工作 | 核心推进 |
| --- | --- | --- |
| 2020 | Katharopoulos 等，Linear Transformer | 用 feature map 和结合律把 kernel attention 写成递归状态 |
| 2021 | RetNet；Schlag 等，Fast Weight / DeltaNet | retention 提供并行、递归、chunkwise 三种视角；fast-weight 工作把状态更新解释成在线关联学习，并引出 delta rule |
| 2023 | Mamba；GLA | Mamba 发展输入选择性 SSM；GLA 在线性注意力状态上加入细粒度数据依赖门控，并研究硬件友好的 chunk 训练 |
| 2024 | Mamba-2 / SSD；高效 DeltaNet chunk 算法 | 建立结构化 SSM 与 attention 的对偶框架；DeltaNet 的三角/WY 分块算法让更复杂的更新可在 GPU 上并行 |
| 2025 | Gated DeltaNet；Qwen3-Next；Kimi Linear / KDA | 把全局遗忘与定向修正结合，再细化到通道级遗忘，并通过混合层保留全局注意力 |

这张时间线是按**机制问题**串联的研究脉络，不是严格的单一祖先树：Mamba 与 GLA 是并行发展的相邻路线，SSD 提供数学连接，而 GDN/KDA 直接属于带 Q/K/V 状态更新的线性注意力谱系。

## 核心动机

* **问题**：softmax attention 的计算随上下文长度按 $T^2$ 增长，KV cache 按 $T$ 增长。
* **动机**：降低长上下文推理成本，不能只优化 kernel；必须避免每步都携带、访问完整历史。
* **方法**：Linear Attention 将历史关联压缩到固定形状状态 $S_t$，每个 token 做 rank-one 更新，由当前 query 读取：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad
o_t\propto S_t^\top\phi(q_t).
$$

生成时维护的状态大小不随历史增长，训练时可以按块并行计算。不过摘要有损：相似或冲突的信息可能混在一起，主题变化时旧内容也不会自动清除。后续方法尝试改正冲突记忆、清理过时信息；需要精确回查远处内容时，混合模型仍保留全局 attention。

## References

<a id="ref-1"></a>[1] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

<a id="ref-2"></a>[2] Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers*. ICML 2021. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)

<a id="ref-3"></a>[3] Yutao Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models*. 2023. [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)

<a id="ref-4"></a>[4] Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

<a id="ref-5"></a>[5] Songlin Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training*. ICML 2024. [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

<a id="ref-6"></a>[6] Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:10041–10071, 2024. [会议论文](https://proceedings.mlr.press/v235/dao24a.html)

<a id="ref-7"></a>[7] Songlin Yang et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
