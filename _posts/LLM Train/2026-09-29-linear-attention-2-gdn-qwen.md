---
layout: post
title: 'Linear Attention（二）：Delta Rule、Gated DeltaNet 与 Qwen3-Next'
date: 2026-09-29 09:05 +0000
categories: [LLM]
tags: [LLM, Linear Attention, Gated DeltaNet, Qwen]
pin: false
math: true
---

上一篇把 Linear Attention 看成固定大小的关联记忆 $S$。本篇聚焦两个互补动作：**Delta rule 负责“修正这条记忆”，forget gate 负责“清理整份记忆”**。Gated DeltaNet（GDN）把它们合到一起，Qwen3-Next 则把 GDN 放进线性/全局注意力混合架构中。

## Delta Rule：定向覆写与全局遗忘

朴素线性注意力不断累加 $k_tv_t^\top$。如果新 token 提供一条与旧记忆冲突的关联，简单累加会让新旧内容相互叠加，而不是明确替换。

考虑用状态预测当前 key 对应的 value：$\hat v_t=S_{t-1}^\top k_t$。预测误差为

$$
e_t=v_t-S_{t-1}^\top k_t.
$$

Delta rule 用这个误差修正状态：

$$
S_t=S_{t-1}+\beta_t k_t e_t^\top
= (I-\beta_t k_t k_t^\top)S_{t-1}+\beta_t k_t v_t^\top.
$$

若旧状态已经把 $k_t$ 映射到正确 value，$e_t$ 很小，就无需重复写入；若旧映射错了，更新会沿着 $k_t$ 对应方向修正，而不是无差别地在矩阵上再叠一份 value。$\beta_t\in(0,1)$ 可理解为写入强度或在线学习率。

这解决了“旧值怎么被覆盖”的问题，但没完全解决“旧主题何时清空”。当上下文发生主题切换時，逐条关联修正可能不够快。反过来，如果只用一个全局 forget gate 清空状态，又可能把仍然有用的记忆一起删掉。

## Gated DeltaNet：门控衰减与 Delta 更新

Gated DeltaNet 在 delta 更新中加入输入依赖的遗忘率 $\alpha_t$：

$$
S_t=\alpha_t(I-\beta_t k_tk_t^\top)S_{t-1}+\beta_t k_tv_t^\top.
$$

![Gated Delta Networks 论文中的模型结构与 Gated DeltaNet block](/assets/img/blog/Linear%20Attnetion/gated-delta-networks-figure-1.jpg){: width="530" }
<center>图 1. Gated DeltaNet 模块与混合架构。<a href="#ref-1">[1]</a></center>

这里按 $S\in\mathbb{R}^{d_k\times d_v}$ 记状态。公式可以拆成两步理解：

1. $\alpha_t$ 缩放旧状态，降低整份历史记忆的强度；
2. $I-\beta_t k_tk_t^\top$ 对当前 key 方向上的旧映射做 delta 修正，然后写入当前 value。

当 $\alpha_t$ 较小，模型能快速衰减旧上下文；当 $\alpha_t$ 接近 1，delta rule 保留更有针对性的覆盖能力。GDN 的关键 insight 是：**全局擦除和局部纠错互补，分别处理记忆干扰的两个尺度。**

注意 $\alpha_t$ 常由 log-space gate、softplus/sigmoid 等参数化得到，实际实现还涉及 Q/K 归一化、短卷积、输出门和 fused RMSNorm。上式描述的是状态递推核心，不是完整模型 block。

下面是说明状态更新的单步 PyTorch 参考代码。它按数学公式写，主要用于阅读/验证；生产实现会把 gate、归一化、state load/store 和输出计算融合进设备 kernel。

```python
import torch
import torch.nn.functional as F


def gdn_step(state, q, k, v, alpha, beta, eps=1e-6):
    """One head, one token; state shape is [key_dim, value_dim]."""
    q = F.normalize(q.float(), dim=-1, eps=eps)
    k = F.normalize(k.float(), dim=-1, eps=eps)
    state = state.float()

    decayed_state = alpha * state
    old_prediction = decayed_state.transpose(-1, -2) @ k
    residual = v.float() - old_prediction
    state = decayed_state + beta * torch.outer(k, residual)
    output = state.transpose(-1, -2) @ q
    return output, state
```

代码用了误差形式 $S+\beta k(v-S^\top k)^\top$，再乘遗忘率。具体实现中 gate 与 delta 操作的数学排列要和训练定义保持一致；本例的两行展开用于展示核心动机，不能直接代替任意 checkpoint 的门控约定或 kernel oracle。

## Chunkwise 训练与递归推理

decode 一次通常只有一个 token，递推很自然。Prefill 则一次处理长序列，逐 token 执行会浪费 GPU 并行能力。GDN 的 chunk 算法把每段内的递推展开成因果三角系统和矩阵乘法，再沿 chunk 边界传递状态。

SGLang 当前实现的执行结构能从调用顺序看出来：

```python
# SGLang FLA-style chunk prefill: abbreviated to show the stages.
g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, ...)
w, u, aux = chunk_gated_delta_rule_fwd_intra(
    k=k, v=v, g=g, beta=beta, ...
)
h, v_new = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g, initial_state=state, ...
)
out = chunk_gated_delta_rule_fwd_o(q=q, v=v_new, g=g, h=h, ...)
```

第一阶段处理 chunk 内的关联与三角变换；第二阶段读入 chunk 起始状态并推进状态；第三阶段合成 chunk 输出。SGLang 的 GDN dispatcher 会按 decode、prefill、speculative verify 分派不同 kernel，而不是用一个 kernel 硬扛所有形状。CUDA 上也可以在 Triton、FlashInfer、CuTe DSL 等 backend 之间按能力与硬件选择。

源码入口：[`gdn_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/gdn_backend.py)、[`gdn_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py)、[`chunk.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk.py) 和 [`chunk_fwd.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk_fwd.py)。上面是按这些调用阶段整理的缩略示意，不是逐字拷贝某个完整 kernel。

## Qwen3-Next：GDN 与 Gated Attention 的混合架构

需要区分**算法提出者**和**模型采用者**：Gated DeltaNet 的机制由 Songlin Yang、Jan Kautz、Ali Hatamizadeh 等人提出 [<a href="#ref-1">1</a>]；Qwen3-Next 采用 Gated DeltaNet，并不是 GDN 的最初提出者。

Qwen3-Next-80B-A3B 的公开配置是 48 层，层型按每 4 层一个周期排列：3 层 Gated DeltaNet，再接 1 层 Gated Attention。模型卡将其概括为 $12\times(3\times\text{GDN}+1\times\text{Gated Attention})$。线性层负责高效吸收长输入，周期性全局 attention 则保留显式跨位置检索通路。

SGLang 的 [`qwen3_next.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py) 展示了 GDN block 侧的 Q/K/V/Z 与 gate 投影、短卷积、状态 cache 和 gated norm；[Hugging Face 配置](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) 可核对 `full_attention_interval: 4`、16 个 key heads、32 个 value heads 与 head dimension 128 等字段。模型卡列出的原生 context length 为 262,144；这一数字属于该 checkpoint 的配置，不是 GDN 的理论上限。

混合不是对线性注意力缺点的否认，而是架构层面的资源分配：绝大多数层用固定状态降低长上下文的缓存成本，同时周期性保留全局检索能力。具体效果仍取决于模型训练数据、层比例、head/state 维数、kernel 和推理 batch，不能只凭一个层型比例推导质量或速度。

## References

<a id="ref-1"></a>[1] Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

<a id="ref-2"></a>[2] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training*. ICML 2024. [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

<a id="ref-3"></a>[3] Qwen Team. *Qwen3-Next: Towards Ultimate Training & Inference Efficiency*. 2025. [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

<a id="ref-4"></a>[4] SGLang. Qwen3-Next model and GDN kernels: [`qwen3_next.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py), [`gdn_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/gdn_backend.py), [`gdn_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py)
