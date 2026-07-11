---
layout: post
title: Parallelism Schemes for LLMs
date: 2026-07-09 15:50 +0800
categories: [basic/algorithm]
tags: [Basic, LLMs]
pin: false
math: true
mermaid: true

image:
    assets/img/blog/paral_schemes/cover_parallelism_schemes_for_llms.png
---

近年来，随着大语言模型（Large Language Models, LLMs）参数不断提升，传统的单机单卡模式已经无法满足超大模型进行训练的要求。
分布式训练通过多个计算设备协同完成模型训练，其核心思想是将训练任务按照数据、模型或计算过程进行划分，并通过不同的并行策略实现高效计算。
下面将介绍分布式训练中几种典型并行方式的基本原理及其适用场景。

## 1.数据并行

在训练过程中，需要考虑到 forward 和 backward 两个过程。因此，不同数据并行的实现方式在数据传递的处理上会有所不同
而在推理中，数据并行是指请求并行，即：将模型参数复制到不同卡上，不同的卡推理不同的请求

下面主要说明训练过程中的数据并行方式

### 1.1 DP 实现

`torch.nn.DataParallel` 是 PyTorch 最早提供的并行方式。它使用**一个进程来计算模型权重**，在每个批处理期间将数据分发到每个GPU。

```python
model = MLP().cuda(0)
model = nn.DataParallel(model, device_ids=[0, 1])   # 使用两张 GPU，一行代码实现并行

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(64, 1024).cuda(0)
targets = torch.randint(0, 10, (64,)).cuda(0)

"""
    训练过程（每次更新参数都要重复以下过程）
    1. 【通信】拆分 Batch：inputs (64, 1024) -> Scatter -> GPU 0 (32, 1024) | GPU 1 (32, 1024)
    2. 【通信】复制模型参数：GPU 0 和 1 都会持有一份当前的 MLP 完整参数
    3. 【计算】每张 GPU 独立执行 forward() -> Output 0 | Output 1
    4. 【通信】Gather Output 到 master GPU 0
    Output0
        \
          ---> GPU0
        /
    Output1
"""
outputs = model(inputs)
"""
    5. 【计算】在 GPU 0 上计算 Loss
"""
loss = criterion(outputs, targets)
optimizer.zero_grad()
"""
    6.【通信】分发 Loss
    7.【计算】每张 GPU 根据自己的 batch 计算 grid -> grid 0 | grid 1
"""
loss.backward()
"""
    8.【通信】Gather grid 至 GPU 0
    9.【计算】在 GPU0 上更新主模型参数
"""
optimizer.step()
```

DP 的缺点十分明显，具体如下。
* 单进程多线程问题：**受限于 Python GIL**，线程通信开销大，且**只能在单机多卡进行（不支持分布式）**。同时，**无法使用 Apex 进行混合精度训练**。
* 效率问题：GPU 利用率低，且负载不均衡（Master GPU 可能占 10GB HBM，其他卡才 2GB），Batch Size 一大 Master GPU 就会 OOM。
* 由于固定`replicate(model)`，无法与拆解模型的并行方法共同使用


### 1.2 DDP 实现

`torch.nn.DistributedDataParallel` 基于多进程实现，每个进程都有独立的优化器，每个进程都执行相同的任务，并且每个进程都与所有其他进程通信，不同进程之间只传递梯度。

流程示意图如下。
```
Process0              Process1
 GPU0                  GPU1
   │                     │
Forward               Forward
   │                     │
Loss                  Loss
   │                     │
Backward             Backward
   │                     │
      ◄── All-Reduce ──►
          Gradient
   │                     │
optimizer.step()    optimizer.step()
```

代码示意如下所示。
```python
torch.distributed.init_process_group(backend="nccl")    # 初始化通信
local_rank = int(os.environ["LOCAL_RANK"])              # 当前进程对应的 GPU 编号，通过 torchrun --nproc_per_node=0 train.py 来指定 RANK
torch.cuda.set_device(local_rank)

model = MLP().cuda(local_rank)
model = nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank],
    find_unused_parameters=False                        # 关闭检查，无需每个参数都收到梯度更新
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = inputs.cuda(local_rank)
targets = targets.cuda(local_rank)
"""
    训练过程（每次更新参数都要重复以下过程）
    1. 【计算】每个进程读取自己的 Mini-Batch
        Process0 -> GPU0 -> Batch0
        Process1 -> GPU1 -> Batch1
    2. 【计算】每张 GPU 独立执行 forward()
    3. 【计算】每个进程独立计算 Loss
"""
outputs = model(inputs)
loss = criterion(outputs, targets)

"""
    4.【计算】每张 GPU 独立执行 backward() -> 得到本地梯度 grad0、grad1 ...
    5.【通信】NCCL All-Reduce 梯度 -> 所有 GPU 对梯度求平均/求和
"""
optimizer.zero_grad()
loss.backward()
"""
    6.【计算】每个进程使用同步后的梯度独立执行 optimizer.step()
"""
optimizer.step()
torch.distributed.destroy_process_group()               # 手动销毁进程组
```

### 1.3 FSDP 实现

torch.nn.DistributedDataParallel（DDP） 通过多个进程分别维护一份完整模型，并在反向传播阶段同步梯度，实现了高效的数据并行。然而，DDP 要求每张 GPU 都保存完整的模型参数、梯度以及优化器状态，当模型规模不断增大时，GPU 显存很快成为训练瓶颈。因此，需要一种能够进一步降低显存占用的数据并行方案，这便是 Fully Sharded Data Parallel（FSDP）<a href="#ref-1">[1]</a>。

FSDP 借鉴了 Microsoft 提出的 DeepSpeed ZeRO <a href="#ref-2">[2]</a> 思想，对模型状态进行分片存储，从而显著降低每张 GPU 的显存占用。

训练过程中，GPU 并不仅仅保存模型参数，还需要维护优化器状态、梯度、激活值以及临时缓存。其中模型状态（Model States）通常占据了绝大部分显存。
* Model Parameters ≈ 23%
* Model States ≈ 77%
    * (~5%) Temporary Buffers
    * (~8%) Graidents
    * (~9%) Activations
    * (~55%) Optimizer States

而对于混合精度训练而言，仅模型状态（Model Parameters + Gradients + Optimizer States）便占据了约 85% 的 GPU 显存，因此，减少模型状态的冗余存储成为降低显存占用的关键。

ZeRO 根据模型状态分片程度不同，共分为三个阶段：
* ZeRO-1 : 优化器状态分片（ Optimizer States Sharding）
* ZeRO-2 : 优化器状态与梯度分片（Optimizer States & Gradients Sharding）
* ZeRO-3 : 优化器状态、梯度和模型权重参数分片（Optimizer States & Gradients & Parameters Sharding）

![Desktop View](assets/img/blog/paral_schemes/zero.jpg){: width="700"}
<center>Fig 1. 不同ZeRO 分块比较 <a href="#ref-3">[3]</a></center>

PyTorch FSDP 可以看作是 PyTorch 官方对于 ZeRO-3 的实现。与 DDP 不同，FSDP 将模型参数、梯度以及优化器状态全部按照数据并行组进行分片，每张 GPU 仅保存自身负责的一部分数据，而不是完整副本。

![Desktop View](assets/img/blog/paral_schemes/fsdp.jpg){: width="600"}
<center>Fig 2. FSDP 算法概述 <a href="#ref-2">[2]</a></center>

与 DDP 使用 All-Reduce 同步梯度不同，FSDP 需要在训练过程中动态收集参数并重新分片，因此 All-Reduce 已无法满足需求。

> PS —— All-Reduce = Reduce-Scatter + All-Gather
> ![Desktop View](assets/img/blog/paral_schemes/all_reduce.jpg){: width="550"}
> <center>Fig 3. All Reduce 示例</center>

如 *Fig 4* 所示，与 DDP 不同，FSDP 并不会在每张 GPU 上保存完整模型，而是将模型参数按照数据并行组进行分片。
每张 GPU 仅维护属于自己的参数分片，在真正需要计算时才临时恢复完整参数，计算完成后立即释放，因此能够显著降低训练过程中的峰值显存占用。

此外，FSDP 支持 CPU Offload。当 GPU 显存仍然不足时，可以将暂时不参与计算的参数分片、梯度甚至优化器状态 **Offload 到 CPU 内存**，在需要时再重新加载回 GPU，以进一步降低 GPU 的显存压力。
**代价则是增加了 CPU 与 GPU 之间的数据传输开销。**

![Desktop View](assets/img/blog/paral_schemes/fsdp_process.png){: width="800" height="600" }
<center>Fig 4. FSDP 具体过程</center>

具体流程为：

1. **加载模型分片**：训练开始时，每张 GPU 仅加载属于自己的模型参数分片，因此任意一张 GPU 均无法单独完成 forward。若开启 CPU Offload，参数分片可以存放在 CPU 内存中，仅在**即将计算时**才加载到 GPU。
2. **All-Gather 收集完整参数**：进入当前 FSDP 模块之前，每张 GPU 会通过 All-Gather 收集其它 GPU 保存的参数分片，**临时恢复**当前模块所需的完整参数（Full Weights）。
需要注意的是，恢复的仅仅是当前 FSDP 包装模块的完整参数，而不是整个模型，因此通信和显存开销都能够得到有效控制。
> PS —— 实际过程中，**FSDP 包装的是某个或多个 Layer。**若 FSDP 包装整个 model，当需要计算时，还是存了整个模型，要么存不下，存得下也多了通信开销，不如 DDP。

3. **执行 Forward 并释放完整参数**：获得完整参数后，每张 GPU 独立完成当前模块的 Forward 计算。Forward 结束后，完整参数立即释放（Free Full Weights），GPU 中仅保留自己的参数分片。

4. **Backward 重新收集参数**：由于完整参数已经在 Forward 结束后释放，因此进入 Backward 时，FSDP 会再次执行一次 All-Gather，重新恢复当前模块的完整参数，然后完成 Backward 计算，得到对应的梯度。

5. **Reduce-Scatter 同步梯度**：梯度计算完成后，FSDP 并不像 DDP 那样执行 All-Reduce，**而是采用 Reduce-Scatter。**，Reduce 后再 Scatter，最终每张 GPU 仅保留自己的梯度分片。如果开启了 CPU Offload，梯度分片会在同步完成后立即卸载到 CPU，等待后续参数更新时再重新加载，以降低 GPU 显存占用。

6. **更新参数：**最后，每张 GPU 使用本地保存的参数分片和梯度分片执行 optimizer.step() 完成参数更新。如果启用了 CPU Offload，则需要先将梯度重新加载到 GPU，再完成参数更新。

整个训练过程中，完整参数仅在 Forward 和 Backward 期间短暂存在，其余时间 GPU 中仅保存参数分片；而在启用 CPU Offload 后，参数和梯度还可以进一步迁移到 CPU 保存，使 FSDP 能够支持训练更大规模的模型。不过，这种方式会增加 CPU 与 GPU 之间的数据传输，因此通常仅在 GPU 显存成为瓶颈时才会启用。

PyTorch FSDP 中支持了自动包装和手动包装两种方式。自动包装会递归遍历整个模型，并按照预设策略自动使用 FSDP 对各个子模块进行包装，例如，对于 Transformer 模型，通常会以 Transformer Block 为单位进行自动包装，示例代码如下所示。

``` python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
)

model = torch.nn.DistributedDataParallel(model())
fsdp_model = FullyShardedDataParallel(
   model(),
   fsdp_auto_wrap_policy=default_auto_wrap_policy,
   cpu_offload=CPUOffload(offload_params=True),
)
```

## 2.模型并行

上一章节介绍的数据并行主要通过复制模型来提升训练吞吐量，但每张 GPU 仍然需要保存完整模型（或模型分片）。
随着模型参数规模不断增长，即使采用 FSDP 等参数分片技术，单个模型仍可能无法容纳在有限数量 GPU 的显存中。

一种更直接的思路是不再复制模型，而是拆分模型。
模型并行正是基于这一思想：将模型划分为多个部分，并分别部署到不同 GPU 上，每张 GPU 仅负责模型中的一部分计算，从而突破单卡显存限制。

如 *Fig 5* 所示，模型并行主要分为两类：
* 张量并行（Tensor Parallelism, TP）：对单个 Layer 内部切分，**层内并行**。
* 流水线并行（Pipeline Parallelism, PP）：对 Layer 切分，将不同 Layer 部署到不同 GPU 上，**层间并行**。

![Desktop View](assets/img/blog/paral_schemes/pp_instance.png){: width="800"}
<center>Fig 5. 两种流水线并行模式</center>

### 2.1 流水线并行

PP 是一种典型的层间并行。它将模型按照网络层划分为多个 Stage，并部署到不同 GPU 上，每张 GPU 仅负责部分 Layer 的计算。
不同 Stage 之间通过传递 Activation 和 Gradient 协同完成一次完整的训练。

根据 PP 策略不同，目前主流的流水线并行可以分为两类，如 *Fig 6* 所示：
* `F-then-B`：先完成所有 Forward，然后统一执行 Backward；
* `1F1B`：流水线填满后，每个 Stage 交替执行一次 Forward 和一次 Backward，从而减少激活缓存并降低 Pipeline Bubble。

![Desktop View](assets/img/blog/paral_schemes/pp_method.png){: width="800" }
<center>Fig 6. 两种流水线并行策略</center>

本章节将按照这一发展脉络，依次介绍朴素流水线、GPipe<a href="#ref-4">[4]</a> 以及非交错式 1F1B<a href="#ref-5">[5]</a> 等典型流水线调度策略。

#### 2.1.1 朴素流水线并行
朴素流水线并行是流水线并行最直接的实现方式。
如 *Fig 7* 所示，它将模型按照层间切分成多个部 Stage，每个 Stage 部署到不同 GPU 上，每张 GPU 仅负责本阶段对应层的计算。

![Desktop View](assets/img/blog/paral_schemes/pp_naive.png){: width="600"}
<center>Fig 7. 朴素流水线并行模式</center>

例如，对于四层网络 `output=L4(L3(L2(L1(input))))`，在两张 GPU 上，可以划分为：
* GPU_0 computes: `intermediate=L2(L1(input))`
* GPU_1 computes: `output=L4(L3(intermediate))`

整个前向传播过程如下：

1. GPU_0 完成 L1、L2 的计算，得到中间激活（Activation）。
2. GPU_0 将中间激活发送至 GPU_1。
3. GPU_1 接收激活后继续完成 L3、L4 的计算，并计算最终 Loss。

反向传播过程则完全相反：

4. GPU_1 首先完成 L4、L3 的梯度计算。
5. GPU_1 将中间梯度发送给 GPU0。
6. GPU_0 根据收到的梯度继续完成 L2、L1 的反向传播。

整个过程如 *Fig 8* 所示。朴素流水线仅涉及相邻 Stage 之间的点对点通信。Forward 时传输 Activation，Backward 时传输 Gradient，整个过程无需进行 All-Reduce 等集体通信。

![Desktop View](assets/img/blog/paral_schemes/pp_naive_process.png){: width="700" }
<center>Fig 8. 朴素流水线并行示例</center>

朴素流水线存在以下 3 个局限，几乎不会在实际训练中直接使用。
1. **GPU 利用率极低：**当流水线划分的 Stage 数量为 $K$ 时，其 Bubble 比例约为 $\frac{K-1}{K}$，随着 GPU 数量增加，Bubble 所占比例也不断增大。当 $K \rightarrow \infty$ 时，Bubble 比例趋近于 1，大部分 GPU 都处于空闲状态，导致硬件利用率极低。
2. **无法实现通信与计算重叠：**朴素流水线采用严格的串行执行方式，当前 Stage 完成计算后，需要等待中间激活发送完成，下一个 Stage 才能继续执行；反向传播同样需要等待梯度传输结束。
3. **激活缓存占用较大：** forward 过程中，每个 Stage 都需要保存对应层产生的激活值（Activation），直到反向传播完成后才能释放。尤其是最前面的 Stage，需要缓存整个 Mini-Batch 的激活，因此随着 Batch Size 增大，激活缓存可能成为新的显存瓶颈。

#### 2.1.2 GPipe

GPipe 为 Micro-Batch 版本的 F-then-B 并行策略，流程如 *Fig 9* 所示。GPipe 引入 Micro-Batch 思想，将一个 Mini-Batch 划分为多个更小的 Micro-Batch，显著降低了 Pipeline Bubble。

> PS — Mini-Batch 即为广义上的 Batch，表示一次参数更新 `optimizer.step()` 所使用的全部训练样本。

![Desktop View](assets/img/blog/paral_schemes/pp_gpip.png){: width="700" }
<center>Fig 9. GPip 流水线示意图</center>

若有 K 张 GPU，划分成 M 个 Micro-Batch，则 $Bubble\ ratio = \frac{K-1}{K+M-1}$。当 $M >> K$ 时，bubble 时间可以忽略不计。
但是，当 M 足够大时，activation 的占比会急速上升，成为一个新瓶颈。

GPip 通过重计算（Re-materialization）来降低显存消耗。
在模型训练过程中的前向传播时，会记录每一个算子的计算结果，用于反向传播时的梯度计算。
而 Re-materialization 可以不用保存中间层输出的激活值，在计算梯度的时候会重新计算出来这些激活值从而可以计算梯度，具体如 *Fig 10* 所示。

![Desktop View](assets/img/blog/paral_schemes/pp_gpip_activation.png){: width="700" }
<center>Fig 10. GPip 重计算示意图</center>

#### 2.1.3 PipeDream - 非交错式 1F1B

PipeDream 通过 1F1B 策略解决了 GPipe 中激活过大的问题，具体如 *Fig 11* 所示，$Bubble\ ratio = \frac{K-1}{K+M-1}$。

![Desktop View](assets/img/blog/paral_schemes/pp_pipe_dream.jpg){: width="650" }
<center>Fig 11. PipeDream 流水示意图</center>

### 2.2 张量并行

张量并行是一种典型的层内并行，将计算图中的层内的参数（张量）切分到不同设备，每个设备只拥有模型的一部分，以减少内存负荷，具体如 *Fig 12* 所示。

![Desktop View](assets/img/blog/paral_schemes/tp_intro.png){: width="250" }
<center>Fig 12. 张量并行示意图</center>

张量并行从数学原理上来看，就是对 Linear 层的矩阵进行分块计算，然后把结果合并，而对非 Linear 层不做额外设计。

#### 2.2.1 张量并行方式

Matmul 算子的计算公式为：$C = AB$ 。行并行（Row Parallelism）就是将 B 矩阵按行切成多部分，列并行则将 B 矩阵按列切成多部分，如 *Fig 13* 所示。

![Desktop View](assets/img/blog/paral_schemes/tp_1d_method.png){: width="750" }
<center>Fig 13. 两种张量切分方式：行并行与列并行</center>

#### 2.2.2 一维张量并行

目前，大模型训练最广泛采用的张量并行方案由 Megatron-LM <a href="#ref-6">[6]</a> 提出。
它采用用一维（1D）张量并行，即仅沿张量的某一个维度进行切分，并将不同分片分配到不同 GPU 上执行计算。

对于 Dense LLMs 而言，主要由一个 N 层 Transformer 块组成，除此之外还有输入和输出 Embedding 层，如 *Fig 14* 所示。

![Desktop View](assets/img/blog/paral_schemes/model_overview.png){: width="200" }
<center>Fig 14. Transformer 模型结构</center>

而 Transformer Block 主要由 MLP 和 Self-Attention 组成，因此，1D 张量并行主要针对这两部分进行切分。

**（1）MLP张量并行**

Transformer Block 内的 MLP 一般由 GEMM 和 GeLU 两部分组成如下式所示：

$$
output = GeLU(XA)B
$$

其中，$X$ 是输入矩阵，A 和 B 是权重矩阵。

如 *Fig 15(a)* 所示，在 1D 张量并行中，对于 MLP 块一般按**照先列并行，再行并行的方式进行拆分**。

在 $GeLU(XA)$ 中，如果矩阵 A 按 Row 并行，则 $Y = GeLU(X_1A_1 + X_2A_2)$，而由于 GeLU 是非线性的，所以在计算 GeLU 算子前需要同步一次。
而 A 按照 Column 并行，**则能节省一次同步**，即：$Y = \[GeLU(X_1A_1), GeLU(X_2A_2)\]$。

![Desktop View](assets/img/blog/paral_schemes/tp_1d.jpg){: width="350" }
<center>Fig 15. Transformer Block 内的模型并行方式</center>

**（2）Self-attnetion 张量并行**

利用 MHA 多头注意力天然具有的并行性，对 Q、K 和 V 对应的 GEMM 采用 Column 并行，使 Attention Head 的矩阵计算都能够在单个 GPU 上独立完成。
这种划分方式将不同 Attention Head 的参数和计算任务分配到不同 GPU，因此完成 Self-Attention 本身无需进行任何通信。
随后，Self-Attention 输出后的线性层采用 Row 并行进行划分，无需在进入该线性层之前进行通信。

在 Megatron-LM 1D 并行方案中，对于 MLP 和 Self-Attention 都采用了这种 "Column + Row" 的 GEMM 并行方式。这样可以将两个连续的矩阵乘法融合在一起，消除中间的一次同步。


![Desktop View](assets/img/blog/paral_schemes/tp_transformer_communication.jpg){: width="425" }
<center>Fig 16. Transformer Block 中 TP 并行通信开销</center>


#### 2.2.3 多维张量并行

虽然 Megatron-LM 的 1D 张量并行已经成为当前大模型训练的主流方案，但随着 GPU 数量不断增加，其通信复杂度也随之提升。此外，由于 1D 张量并行中输入 Activation 在多个 GPU 上保持完整副本，因此 Activation Memory 无法继续下降。

为了解决这一问题，研究者提出了 2D、2.5D 以及 3D 张量并行，通过同时切分权重矩阵和输入 Activation，降低显存占用的同时减少通信开销，Colossal-AI <a href="#ref-7">[7]</a>训练框架已经集成了这些并行方式。

**（1）2D 张量并行**

2D TP 最早由 **SUMMA（Scalable Universal Matrix Multiplication Algorithm）**<a href="#ref-8">[8]</a> 引入，其核心思想是将参与矩阵乘法的两个矩阵同时按照二维 Block 进行划分，并将 GPU 组织成一个二维 Processor Grid。

设矩阵乘法为：

$$
C = AB,
$$

其中

$$
A \in \mathbb{R}^{M\times K},\qquad
B \in \mathbb{R}^{K\times N}.
$$

对于一个 $2\times2$ Processor Grid，可将矩阵划分为：

$$
A=
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
B_{11} & B_{12}\\
B_{21} & B_{22}
\end{bmatrix}.
$$

> PS — 每个设备仅存储一部分，比如 GPU 0 仅存 A_{11}，每次计算都需要冗余通信

最终输出矩阵各 Block 的计算公式为

$$
\begin{aligned}
C_{11}&=A_{11}B_{11}+A_{12}B_{21},\\
C_{12}&=A_{11}B_{12}+A_{12}B_{22},\\
C_{21}&=A_{21}B_{11}+A_{22}B_{21},\\
C_{22}&=A_{21}B_{12}+A_{22}B_{22}.
\end{aligned}
$$

与 1D 张量并行不同，2D 张量并行不仅切分权重，同时也切分输入 Activation。

例如，对于一个$(B,S,8192)\rightarrow(B,S,32768)$ 的 Linear 层，在 **TP=4** 时：

- **1D Tensor Parallel：**
  - 每张 GPU 保存完整输入 Activation：$(B,S,8192)$；
  - 权重仅保存 $1/4$。

- **2D Tensor Parallel：**
  - GPU 组成 $2\times2$ 网格；
  - 输入 Activation 被切分为 $(B,S,4096)$；
  - 权重同时沿行、列两个维度切分；
  - 每张 GPU 仅保存部分 Activation 与部分权重。

因此，相比 1D 张量并行，2D 张量并行进一步降低了 Activation Memory，同时通信复杂度由
$O(P)$ 降低为 $O(\sqrt{P})$。

**（2）2.5D 张量并行**

2D 张量并行虽然降低了通信复杂度，但每轮矩阵乘法仍需要在二维 Processor Grid 内进行 **Row Broadcast** 和 **Column Broadcast**，随着 GPU 数量增加，这部分通信仍然占据较大的比例。

2.5D TP 进一步在第三个维度引入**冗余副本（Replication）**，相当于用一次预取，替代了多次重复通信。

假设有$P=q^2d$ 张 GPU，其中：

- $q\times q$ 为二维 Processor Grid；
- $d$ 为复制维度（Replication Factor）。

与 2D 张量并行不同，每个矩阵 Block 会复制到 $d$ 个 GPU 上：

$$
A_{ij}
\longrightarrow
\left\{
A_{ij}^{(1)},
A_{ij}^{(2)},
\cdots,
A_{ij}^{(d)}
\right\}.
$$

于是，每个副本组分别负责一部分乘法计算：

$$
C^{(t)}
=
A^{(t)}B^{(t)},
\qquad
t=1,\cdots,d.
$$

最终，仅需对所有副本执行一次 Reduce：

$$
C
=
\sum_{t=1}^{d}
C^{(t)}.
$$

由于矩阵已经在第三维进行了复制，同一个副本组中的 GPU 可以直接访问本地矩阵，而无需像 2D 张量并行那样在每一轮迭代都广播矩阵 Block。因此，通信模式由

$$
\underbrace{\text{Broadcast}+\text{Broadcast}}_{\text{每轮}}
$$

变为

$$
\underbrace{\text{Local GEMM}}_{\text{重复计算}}
+
\underbrace{\text{Reduce}}_{\text{最终一次}}.
$$

也就是说，2.5D 张量并行**使用少量额外显存换取通信减少**：矩阵复制带来了更多本地计算，但显著降低了跨 GPU 的 Broadcast 开销。

**（3）3D 张量并行**

3D TP 进一步将矩阵划分到三个维度，并将 GPU 组织成立方体 Processor Grid，Transformer Block 中的3D TP 如 *Fig 17* 所示。

![Desktop View](assets/img/blog/paral_schemes/tp_3d.jpg){: width="625" }
<center>Fig 17. Colossal-AI 框架中的 3D 张量并行方案</center>

假设共有$P=q^3$ 张 GPU，则每个 GPU 仅负责$\frac1q$的输入、$\frac1q$的权重以及$\frac1q$的输出。

例如，对于 **64 张 GPU**，

3D Tensor Parallel 可以组织为

$$
4\times4\times4
$$

的 Processor Grid。

相比 2D 张量并行：

- 输入 Activation 进一步切分；
- 权重进一步切分；
- 通信复杂度进一步下降为

$$
O(\sqrt[3]{P}).
$$

理论上，3D 张量并行具有最佳的扩展能力，但由于实现复杂，需要维护多个通信组，并且需要更加复杂的 Broadcast、Reduce 与同步机制，因此目前主要用于学术研究，在工业界应用较少。

## 3. MoE 并行

模型规模的扩展会导致训练成本显著增加，计算资源的限制成为了大规模密集模型训练的瓶颈。
为了解决这一问题，一种基于稀疏 MoE 层<a href="#ref-9">[9]</a>的深度学习模型架构被提出，即将大模型拆分成多个小模型（即 Expert），每轮迭代根据样本决定激活一部分专家用于计算，达到了节省计算资源的效果。

![Desktop View](assets/img/blog/paral_schemes/moe.jpg){: width="500" }
<center>Fig 18. MoE 网络结构</center>

，模型包含 3 个模型层，如(a)到(b)所示，将中间层扩展为具有 n 个 expert 的 MoE 结构，并引入 Gating network 和 Top_k 机制，MoE 细节如下图(c)所示


## Reference
<span id="ref-1"></span>[1] Zhao Y, Gu A, Varma R, et al. Pytorch FSDP: Experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

<span id="ref-2"></span>[2] Rasley J, Rajbhandari S, Ruwase O, et al. DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters[C]//Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020: 3505-3506.

<span id="ref-3"></span>[3] https://github.com/liguodongiot/llm-action.

<span id="ref-4"></span>[4] Huang Y, Cheng Y, Bapna A, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism[C]. Advances in neural information processing systems, 2019, 32.

<span id="ref-5"></span>[5] Narayanan D, Harlap A, Phanishayee A, et al. PipeDream: Generalized pipeline parallelism for DNN training[C]//Proceedings of the 27th ACM symposium on operating systems principles. 2019: 1-15.

<span id="ref-6"></span>[6] Shoeybi M, Patwary M, Puri R, et al. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

<span id="ref-7"></span>[7] Li S, Liu H, Bian Z, et al. Colossal-AI: A unified deep learning system for large-scale parallel training[C]//Proceedings of the 52nd International Conference on Parallel Processing. 2023: 766-775.

<span id="ref-8"></span>[8] Van De Geijn R A, Watts J. SUMMA: Scalable universal matrix multiplication algorithm[J]. Concurrency: Practice and Experience, 1997, 9(4): 255-274.

<span id="ref-9"></span>[9] Shazeer N, Mirhoseini A, Maziarz K, et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
