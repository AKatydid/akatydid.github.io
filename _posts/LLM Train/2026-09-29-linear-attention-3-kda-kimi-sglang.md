---
layout: post
title: 'Linear Attention（三）：Kimi Delta Attention、逐通道记忆与 SGLang'
date: 2026-09-29 09:10 +0000
categories: [LLM]
tags: [LLM, Linear Attention, Kimi, SGLang]
pin: false
math: true
---

第二篇的 GDN 用每个 head 一个遗忘率 $\alpha_t$ 衰减整块状态。Kimi Delta Attention（KDA）进一步追问：一个 head 内的所有 key 特征，真的应该用同一个遗忘速率吗？KDA 的答案是**逐通道遗忘**：每个 key 维度拥有自己的 gate。

## KDA：逐通道遗忘门控

把状态记为 $S_t\in\mathbb{R}^{d_k\times d_v}$。GDN 中 $\alpha_t$ 是标量，旧状态的所有 key 方向同步衰减。KDA 把它换成 $\boldsymbol{\alpha}_t\in(0,1)^{d_k}$，并令 $D_t=\operatorname{Diag}(\boldsymbol{\alpha}_t)$：

$$
S_t=(I-\beta_t k_t k_t^\top)D_tS_{t-1}
+\beta_t k_t v_t^\top,\qquad
o_t=S_t^\top q_t.
$$

先按通道衰减，再沿当前 $k_t$ 对应的方向做 delta-rule 修正。某些特征方向可以保留更久，另一些方向可以更快淡出。**粗粒度的“整个 head 忘多少”升级为细粒度的“哪些记忆方向忘多少”。**

若令 $D_t=\alpha_tI$，就退化回标量门控的 GDN 形式（忽略模型外围参数化差异）。这是两者关系最简洁的表述。

Kimi Linear 报告将这种细粒度门控与位置建模联系起来：不同通道的累计衰减可以形成不同的时间尺度。它不是把 RoPE 简单塞进线性注意力，而是让状态转移本身具备输入依赖的方向性衰减。

## KDA 分块并行算法

如果把上面的单步公式原样循环一百万次，数学正确却不是理想的 prefill kernel。KDA 的挑战是逐通道 gate 让 chunk 内每个 key 维度都有自己的累计衰减；实现需避免昂贵的不稳定倒数，还要尽量把工作整理成 GPU 擅长的矩阵乘法。

Kimi Linear 的 chunkwise 推导使用 WY/UT 一类变换，把一个 chunk 中连续的低秩更新组织为辅助矩阵 $W,U$ 和三角系统。直观上分三层：

1. **Chunk prologue**：计算逐通道 log-decay 的 chunk 内前缀量，并据此缩放 Q/K；
2. **Chunk 内并行**：构造局部 key-key 关系，解因果三角系统，得到等价的 delta 修正项；
3. **Chunk 间递推**：读取上一个 chunk 的 state，更新边界 state，并计算当前 chunk 输出。

SGLang 的 KDA Triton prefill 代码也体现了这个分层：`chunk_kda_fwd_intra` 先产生 `w`、`u` 及缩放后的 `kg`；`chunk_gated_delta_rule_fwd_h` 推进状态；随后输出阶段融合 chunk 内部与跨 chunk 的贡献。逐通道衰减被前缀和及缩放后的操作数吸收，避免把整套通用 DPLR 计算直接搬进每一步矩阵运算。

对应的 SGLang 入口可在 [`kda.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/kda.py) 看到。其关键路径可缩写为：

```python
# Abbreviated from SGLang's chunk_kda prefill path.
w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
    q=q,
    k=k,
    v=v,
    gk=g,
    beta=beta,
    scale=scale,
    chunk_size=chunk_size,
    safe_gate=lower_bound is not None,
)

state, v_new = chunk_gated_delta_rule_fwd_h(
    k=kg,
    w=w,
    u=u,
    gk=g,
    initial_state=initial_state,
    use_exp2=True,
)
```

这里刻意省略 variable-length metadata、chunk index、state pool 的索引、output kernel 和中间 buffer 生命周期；上面只呈现计算顺序。SGLang 的 `kda.py` 和其 kernel wrapper 才是完整语义的依据。

## Kimi Linear：KDA 与 MLA 的混合架构

Kimi Linear 技术报告的模型结构不是“所有层都换成 KDA”，而是以 **3 层 KDA + 1 层全局 MLA** 周期性交错。KDA 以固定大小 state 提供低缓存成本，MLA 保留全局显式检索路径。

![Kimi Linear 论文中的 KDA 与 MLA 混合架构](/assets/img/blog/Linear%20Attnetion/kimi-linear-figure-3.jpg){: width="346" }
<center>图 1. Kimi Linear 的 KDA/MLA 混合架构。<a href="#ref-1">[1]</a></center>

这里有两个不同的“比例/混合”概念，容易混淆：

* KDA **单层内部**：每个 key 通道独立衰减，配合 delta update；
* Kimi Linear **跨层**：3 层 KDA 后接 1 层全局 MLA。

全局层用于补充固定状态模型在精确长程检索、复制等能力上的短板。论文报告的质量与吞吐结果属于论文指定的模型规模、硬件、训练 recipe 和 benchmark；它们不是任何 KDA 实现都自动获得的保证。

## SGLang 中的 KDA 推理实现

Prefill 走 chunk kernel；autoregressive decode 通常一次只需处理每个请求的下一个 token，因此使用 recurrent update。SGLang 的 dispatcher 把 KDA 的 decode、prefill 与 target verify 分开，常见入口包括：

* 模型投影与 gate 参数化：[`kimi_linear.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_linear.py)；
* backend 元数据与请求状态池：[`kda_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py)；
* Triton recurrent decode / chunk prefill wrapper：[`kda_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py)；
* chunk 三角变换与逐通道 gate：[`kda.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/kda.py)；
* Blackwell prefill pipeline：[`kda_blackwell`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/attention/linear/kda_blackwell)。

实现中最值得注意的是状态形状：每层、每请求、每个 value head 都要维护大致为 $[d_v,d_k]$ 的矩阵状态；KDA 还需处理每 key 通道 gate。decode 的算术量虽不再随总历史长度累积，但状态池的读写带宽、state dtype、batch、head dimension、融合 kernel 的适用范围都会影响实际 token latency。

SGLang 当前同时保留不同硬件/形状的实现路径。例如 KDA 的 Triton 路径是通用的重要基线，CuTe DSL Blackwell prefill 将 gate 前缀 prologue、chunk 内矩阵变换、state update 与 output 分成专用阶段；decode 又可采用融合 recurrent kernel。源码中的 dispatch/fallback 逻辑正说明：**算法的渐近复杂度并不等于跨硬件恒定的性能。**

## GDN 与 KDA 的机制比较

| 机制 | 遗忘粒度 | 写入/覆写 | 核心收益 | 仍需面对的问题 |
| --- | --- | --- | --- | --- |
| Linear Attention | 无或较弱 | 累加关联 | 极简固定状态递推 | 旧关联干扰、缺少遗忘策略 |
| DeltaNet | 当前 key 方向 | 依据预测误差修正 | 能有针对性地纠正关联 | 大范围清理旧记忆不够直接 |
| GDN | 每 head 一个标量 gate | Delta 修正 | 整体记忆寿命可自适应 | head 内特征共用遗忘强度 |
| KDA | 每 key 通道一个 gate | Delta 修正 | 更精细的记忆时间尺度 | 更复杂的 gate 与 chunk kernel |

KDA 没有消除固定状态的信息容量上限，也没有让线性模型自动拥有完整 KV cache 的逐 token 检索能力。它改善的是压缩记忆的**管理粒度和状态转移表达力**。混合 MLA/全局 attention 则从模型结构层补上显式检索通路。

## References

<a id="ref-1"></a>[1] Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture*. 2025. [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

<a id="ref-2"></a>[2] Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

<a id="ref-3"></a>[3] SGLang source: [`kimi_linear.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_linear.py), [`kda_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py), [`kda_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py), [`kda.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/kda.py), [`kda_blackwell`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/attention/linear/kda_blackwell)
