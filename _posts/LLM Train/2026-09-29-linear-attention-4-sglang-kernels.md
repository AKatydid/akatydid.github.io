---
layout: post
title: 'Linear Attention (4)：SGLang CUDA Path GDN/KDA Kernel Analysis'
date: 2026-09-29 09:15 +0000
categories: [LLM]
tags: [LLM, Linear Attention, SGLang, CUDA, Triton]
pin: false
math: true
---

> **系列导读**：第一篇介绍固定状态记忆与 Mamba 的关系；第二篇讲 DeltaNet、Gated DeltaNet 和 Qwen3-Next；第三篇讲 KDA 与 Kimi Linear；本篇从 SGLang 源码看 GDN/KDA 如何落到 prefill、decode 和硬件 kernel，并分析实际性能瓶颈。

本文只讨论 NVIDIA CUDA 路径，不覆盖 ROCm、XPU、NPU 或 CPU。例子选 Qwen3-Next-80B-A3B，在 H100 SXM 上做理论 roofline 分析；不运行 benchmark，也不把理论下界写成实测性能。

## 核心结论

线性注意力的“线性复杂度”不会自动变成低延迟。实际性能取决于请求处于 prefill 还是 decode、状态怎样读写、kernel 是否融合、矩阵形状是否适合 Tensor Core，以及输入是否需要 padding 或 gather/scatter。

* **Prefill**：一个请求一次有很多 token。目标是并行处理 chunk，把递推转成矩阵乘法。
* **Decode**：每个请求通常只有一个新 token。目标是用一次尽量融合的 kernel 读取并更新已有状态。
* **工程结论**：同一组递推公式必须有不同执行路径；融合可以省 kernel launch 和中间张量，但状态矩阵读写仍在，仍可能成为带宽瓶颈。

## SGLang 的调用分层

以 GDN 为例，SGLang 的 model layer 产生 Q/K/V、门控输入和短卷积结果；`GDNAttnBackend` 从请求对应的 Mamba cache 取出卷积状态与 SSM 状态；`GDNKernelDispatcher` 再依据运行模式和配置选择 decode、prefill 或 verify kernel。

```text
Qwen3 GDN model
  projection -> short Conv1D -> Q/K/V + gates
                                  |
                                  v
                         GDNAttnBackend
                     state pool + metadata
                       /       |       \
                 prefill    decode    verify
                 chunk      recurrent  draft states
```

这不是普通 softmax attention backend 的简单注册路径。SGLang 文档说明，GDN/KDA 使用 `LinearAttnKernelBase` 和专用 dispatcher；它们还要处理每个 request 的 state slot、变长序列边界、CUDA graph 元数据与 speculative state 回滚。

源码入口：

* [Qwen3-Next GDN model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py)
* [GDN backend / dispatcher](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/gdn_backend.py)
* [KDA backend / dispatcher](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py)
* [Linear attention backend API](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention/linear/kernels)

## Decode：每步更新状态，瓶颈转到状态流量

每个 value head 的状态可以看作 $H\in\mathbb{R}^{V\times K}$。一次 decode 对一个 token 读入 $H$，执行门控衰减、预测当前 key 对应的旧 value、计算残差并写回 rank-one 更新，最后用 query 从新状态读出输出。

下面是 SGLang recurrent kernel 的等价缩写，省略了指针计算、mask、向量化和 Q/K 归一化细节：

```python
# Conceptual excerpt of fused_recurrent.py's recurrent update.
state *= decay[None, :]             # per-key-channel decay for KDA
residual = value - (state * key).sum(dim=-1)
delta = beta * residual
state += delta[:, None] * key[None, :]
output = (state * query[None, :]).sum(dim=-1)
```

SGLang 的 [fused_recurrent.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/fused_recurrent.py) 将状态按 tile 分配给 CTA，使用 fp32 累加，并直接从 state pool 按请求 slot 读写。代码中的 KDA packed decode 把 Q/K/V 打包输入直接送入 kernel；门控、delta update 和输出计算在同一 kernel 中完成。GDN 也有对应的 packed decode 路径。

### 为什么 packed/fused decode 有帮助

未融合的 serving 路径通常要：拆分 packed QKV、reshape/view，另行计算 gate，再 launch recurrent update。虽然 `view` 本身未必复制数据，`split` 后的布局整理、临时张量和多次 launch 仍会带来调度及内存流量。packed kernel 尽量让一份投影结果直接被消费，减少中间结果和 launch 数。

但融合并没有让 state 消失。以每个 head 的 $V\times K$ 状态为例，每 token 至少需要读取旧状态并写回新状态，状态流量近似正比于

$$
2\,H_vVK\times \text{bytes-per-element}.
$$

例如 $H_v=32$、$K=V=128$、fp32 state 时，这个理想化读写量约为每层每请求每生成 token $4$ MiB（32 heads × $128^2$ × 4 bytes × 读写各一次）。这是估算下界，不是 profiler 实测：state 可能以其他 dtype 存储，cache 命中、CTA 切分、额外临时量也会改变 HBM 流量。它说明为什么 decode 的瓶颈常从“算多少”转为“状态搬多少”。

当 batch 增大时，多个 request 的 state 读写会持续占用显存带宽；当 batch 很小时，kernel launch 与并行度不足又可能占主导。因此 fused kernel 的收益依赖 batch、head 数、$K/V$ 维度、状态 dtype、硬件和上游投影是否也已融合，不能由“少一个 kernel”直接推出固定倍数的加速。

SGLang 的 [GDN decode benchmark](https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_gdn_decode.py) 将两条路径放在一起比较：baseline 包含 split/view 后调用 recurrent update；packed 路径直接消费 mixed QKV，并检查输出与最终 state 的误差。复现时应使用目标模型的 head/state 形状，报告 batch sweep、dtype、CUDA graph 状态与 p50/p95 latency，而不只报单个最好结果。

## Prefill：把递推改造成矩阵乘法

Prefill 的长度维度大，逐 token 更新会串行化 GPU 工作。chunkwise 算法把一个 chunk 内的依赖整理为小型因果三角系统，再在 chunk 边界递推状态。

### GDN 的 chunk 内部

SGLang [chunk_fwd.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk_fwd.py) 的 GDN intra 阶段将下面的步骤组合起来：

1. 计算 chunk 内 key-key 相关矩阵 $K K^\top$，并纳入 gate 和写入强度；
2. 求解对应的下三角系统，得到 delta 更新的等价系数；
3. 由变换后的系数生成 $w,u$，供后续状态更新与输出计算使用。

源码中的 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 把 KKT 构造与 `solve_tril` 放进同一个 Triton kernel，避免把中间矩阵 $A$ 写入 HBM 后再读回。后续仍有 `recompute_w_u`、chunk state update 和输出阶段，因此不是“整层只有一个 kernel”；它针对的是一个具体的中间读写和 launch 边界。

GDN 主要涉及 head 级标量 gate；KDA 的 gate 是每个 key 通道一份，形状随 $K$ 扩展。chunk 内需要先计算逐通道累计 gate，并构造缩放后的 key/query 操作数，再求解局部三角关系。SGLang 的 [kda.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/kda.py) 组织 Triton 路径；Blackwell 的 [kda_blackwell pipeline](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/attention/linear/kda_blackwell) 则拆为 gate prologue、`kkt_inv_uw`、state update 和 output 几个专用阶段。

固定 chunk 长度 $C$ 时，Kimi Linear 论文给出的单 head KDA attention 计算量近似为（见 [论文全文](https://arxiv.org/html/2510.26692v2)）：

$$
\operatorname{FLOPs}_{\text{KDA}}(T,C,d_h)
=6Td_h^2+3TCd_h+TC^2,
\qquad
\operatorname{FLOPs}_{\text{global}}(T,d_h)\approx 2T^2d_h.
$$

忽略投影、MoE 和其他层，这说明固定 $C$ 后 KDA 的序列长度项为线性，而 global attention 有二次项；但 $6Td_h^2$ 并不小，head dimension、chunk 大小和 Tensor Core 利用率仍会决定常数项。GDN 的 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 使用更小的子块构造 chunk 内三角关系，并融合 KKT 与求解，减少一个中间矩阵的 HBM round-trip。融合降低数据搬运，却会增加单 kernel 的寄存器/共享内存压力；tile 过大可能降低 occupancy，或触发寄存器 spill。

```python
# Conceptual outline of the SGLang KDA chunk prefill path.
g_cumsum = kda_gate_chunk_cumsum(g, chunk_size=64, ...)
w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(q, k, v, g_cumsum, beta, ...)
chunk_state, v_new = chunk_gated_delta_rule_fwd_h(kg, w, u, g_cumsum, state, ...)
output = chunk_kda_fwd_o(q, v_new, g_cumsum, Aqk, chunk_state, ...)
```

这是计算阶段示意，不是可直接执行的 API 调用。变长序列还需要 `cu_seqlens`、chunk indices 和 state indices；初始/最终状态则供相邻 chunk 与后续 decode 衔接。

### CuTe DSL / Blackwell 路径的性能含义

SGLang 为 GDN/KDA 提供 Triton、FlashInfer，以及特定 GPU 上的 CuTe DSL 等实现。dispatcher 会检查硬件和 shape 能力；例如部分 GDN/KDA CuTe prefill 要求 SM100（Blackwell），不满足时回退 Triton。专用路径通常能用 Tensor Core、TMA 和更贴合固定 tile 的布局，但代价是编译特化更多、支持的形状更窄，并需要保留 fallback。

KDA Blackwell prefill 的 pipeline 将繁重的逐通道 gate 预处理独立出来，再执行矩阵型 KKT/三角变换、chunk state 更新和输出。把工作组织成矩阵乘法有利于提高 Tensor Core 利用率；然而较小的 batch/sequence 数、短 chunk 或大量 padding 会使 tile 利用率降低，prologue 和多个阶段的 launch 开销也可能盖过计算节省。

## 性能分析：该测什么、怎么解释

SGLang 的 [GDN decode benchmark](https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_gdn_decode.py) 与 [KDA prefill CuTeDSL benchmark](https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py) 都同时做 correctness 与 latency/吞吐比较。阅读或复现结果时，至少区分以下因素：

| 路径 | 主要工作 | 常见瓶颈 | 有意义的对比 |
| --- | --- | --- | --- |
| 单 token decode | 读取/写回每请求 state，执行递归更新 | state 带宽、低 batch 并行度、launch 开销 | split + recurrent baseline 对比 packed/fused；扫 batch 与 state dtype |
| 长序列 prefill | chunk 内 KKT/三角求解，chunk 间状态递推 | Tensor Core 利用率、临时矩阵流量、padding 与 launch 数 | 固定模型形状，扫 $T$、head 数、chunk size 和变长序列分布 |
| Speculative verify | 为候选 token 计算输出与中间状态，接受后提交 | 中间 state 容量、分支布局、写回与 rollback | 分开量 verify 和 commit；比较 chain 与 tree 支持范围 |

论文或 benchmark 的 FLOPs/TB/s 只是近似指标。不同实现若一个把 metadata 预计算放在计时区外、另一个算在区内，或者一个包含投影/卷积而另一个只计核心 recurrence，就不能直接比较数字。应同步报告 GPU 型号、CUDA/框架版本、dtype、batch、序列长度、warmup、CUDA graph、计时边界、correctness 误差与 p50/p95；对长序列吞吐和单 token decode latency 分开作图。

## Roofline 算例：Qwen3-Next on H100

公开配置中，Qwen3-Next 有 48 层、每 4 层一个 global-attention 层，因此按该周期共有 36 个 GDN 层；GDN 配置为 16 个 key heads、32 个 value heads，$d_k=d_v=128$。以下选两个输入点：prefill 为 batch 1、prompt 8192 tokens；decode 为 batch 1、每请求生成 1 token。H100 SXM 标称 FP32 峰值 67 TFLOP/s、显存带宽 3.35 TB/s，对应 FP32 ridge point 约 $20$ FLOP/byte。roofline 是硬件上限模型，不代表 kernel 可以达到峰值。

配置来源：[Qwen3-Next config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)；硬件规格：[NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)。

### Decode：state 更新是 memory-bound

每个 GDN value head 的 recurrent state 是 $128\times128$。32 个 value heads 合计 $32\times128\times128=524{,}288$ 个元素：若 state 用 fp32，每层每请求约 2 MiB。每个 token 至少读旧 state、写新 state，因此：

* 每个 GDN 层约 4 MiB state traffic / request / token；
* 36 个 GDN 层合计约 144 MiB / request / generated token；
* batch 16 时，整步约 2.25 GiB state traffic。

用 H100 SXM 标称带宽做理想化下界，batch 1 的 144 MiB 至少需要约 45 微秒，batch 16 的 2.25 GiB 至少约 0.72 毫秒。它们是假设 state 流量全部经过 HBM 且带宽完全饱和的下界；L2 命中、state dtype、调度和其他层都会改变真实时间。

再估算 recurrent core 的算术强度。对每个 state 元素，粗略计入一次 decay、一次 delta 更新；旧值预测和 query 读取各是一个长度为 $K$ 的点积，合计约 $7H_vVK$ FLOPs。于是 fp32 state 读写下：

$$
I_{\text{decode}}\approx
\frac{7H_vVK}{2H_vVK\times4}\approx0.875\ \text{FLOP/byte}.
$$

该值远低于 H100 FP32 ridge point（约 20 FLOP/byte），所以在 HBM 流量占主导时，GDN decode core 明确不是算力受限，而是 state 数据搬运受限；小 batch 还可能叠加 occupancy 和 launch latency。packed decode 省掉拆分、临时布局和额外 launch 有帮助，但不消除主要的 $V\times K$ state 读写。

### Prefill：矩阵计算变重，受限项随 shape 改变

同一条 8192-token prompt 按 $C=64$ 分 chunk，共 128 个 chunk。chunk 内 KKT 等矩阵操作的工作量随 chunk size 增长，状态更新则将跨 token 递推变成块矩阵计算，因此比 decode 更能利用并行和 Tensor Core。以单个 GDN head 的 KKT 子问题估算，$2C^2K$ 在 $C=64,K=128$ 时约 1.05 MFLOPs；仅计两份 BF16 key tile 读取和一个 FP32 $C\times C$ 输出写入，理想算术强度约 21 FLOP/byte。

这个估算只是一个子 kernel 的下界：真实路径还要搬运 gate、beta、$w/u$、state 和输出。它接近 H100 FP32 CUDA-core ridge，但远低于 dense BF16 Tensor Core 理论 ridge（约 $989/3.35\approx295$ FLOP/byte）。因此不能只凭“有大矩阵乘法”就断言 prefill 是 compute-bound：若矩阵运算没有充分落到 Tensor Core，计算吞吐可能限制性能；若 Tensor Core 峰值可用，临时矩阵流量、padding、chunk 尾部和 kernel launch 又可能成为限制。对 8192-token 输入，128 chunks × 32 value heads 提供约 4096 个 chunk/head 工作单元，有利于提供并行度，但实际瓶颈仍须由 Nsight 计数器或目标机 benchmark 判定。

**对这个模型的判断**：单 token decode 的 GDN recurrence 有低算术强度，优先看 state 带宽和 batch；长 prefill 有大量块内矩阵工作，性能更依赖 tile 利用率、Tensor Core 路径和中间量流量。两者不能用同一组延迟数字或一个“linear”标签概括。

## 从源码得到的三点结论

1. **先选对执行模式**：prefill 是 chunk 并行问题，decode 是 state 更新问题；verify 还要处理候选状态提交与回滚。
2. **融合主要省数据搬运和调度**：packed decode 与 KKT/三角求解融合减少中间张量和 launch，但不能消除 state pool 流量。
3. **专用 kernel 以覆盖范围换峰值效率**：CuTe DSL/Blackwell 路径更贴合目标硬件和 tile，shape 不满足时必须回退；基准结论要限定设备、形状与计时口径。

## References

<a id="ref-1"></a>[1] SGLang GDN backend: [`gdn_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/gdn_backend.py), [`gdn_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py)

<a id="ref-2"></a>[2] SGLang KDA backend: [`kda_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py), [`kda_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py)

<a id="ref-3"></a>[3] SGLang recurrent GDN/KDA kernels: [`fused_recurrent.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/fused_recurrent.py)

<a id="ref-4"></a>[4] SGLang chunkwise GDN kernels: [`chunk.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk.py), [`chunk_fwd.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk_fwd.py), [`chunk_delta_h.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py)

<a id="ref-5"></a>[5] SGLang KDA chunkwise implementation: [`kda.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fla/kda.py), [`kda_blackwell`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/attention/linear/kda_blackwell)

<a id="ref-6"></a>[6] SGLang benchmark: [`bench_gdn_decode.py`](https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_gdn_decode.py), [`bench_kda_prefill_cutedsl.py`](https://github.com/sgl-project/sglang/blob/main/benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py)
