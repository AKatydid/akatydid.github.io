---
layout: post
title: SGLang 源码学习（1）——  Cache & Shcheduler & Request
date: 2026-07-14 15:59 +0800

categories: [sglang]
tags: [sglang, infra]
pin: false
math: true
mermaid: true
---

SGLang 是目前最常用的工业级推理引擎之一。
为了深入理解其高性能推理背后的实现机制，本文将从源码角度，对 SGLang 的整体架构和执行流程进行系统梳理。

文章将沿着一次推理请求的执行路径，从框架入口开始，逐步分析 KV Cache & Request & Shcheduler 核心模块的职责，以及请求如何在不同组件之间流转，并在后续章节进一步深入到NVIDIA GPU & CPU 算子实现与性能优化细节。

> **SGLang Version: 0.5.13**

## 1. KV Cache 相关模块

如 Fig 1 所示，SGLang 主要由 Tokenizer/Detokenizer、Scheduler、KV Cache Manager、Model Runner 等模块组成。
其中，KV Cache 作为全局资源，与其，是推理框架中最重要的组件之一，本文主要关注 KV Cache 管理部分。

![Desktop View](assets/img/sglang/sglang-architecture.svg){: width="800" height="600" }
<center>Fig 1. SGLang 框架概述图 <a href="#ref-1">[1]</a></center>

然而，由于 KV Cache 作为全局资源，遍布推理全流程。为便于阅读，本章在此列出相关代码目录。

* KV Cache 流程相关代码
  * **初始化：**位于 [`class Scheduler::__init__`](https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/managers/scheduler.py#L433-L452) 中。其中，Scheduler 类并不是在主进程中直接创建，而是在 `Engine._launch_subprocesses(...)` 启动子进程时，调用函数 `_launch_scheduler_processes(...)` ，通过 [`mp.Process`](https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/entrypoints/engine.py#L614-L646) 创建 Scheduler 进程；子进程启动后，会执行 srt/managers/scheduler.py 中的函数 [`run_scheduler_process(...)`](https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/managers/scheduler.py#L4046-L4056)，在其中实例化 Scheduler，并完成 KV Cache、Memory Pool 等核心组件的初始化。
  * **Batch 构建：**每轮调度时，`Scheduler.event_loop()` 会调用 [`get_next_batch_to_run()`](https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/managers/scheduler.py#L2415-L2540) 选择当前需要执行的请求，并构建一个 ScheduleBatch。在这一过程中，会完成与 KV Cache 相关的核心工作，例如 Prefix Cache 命中检查、KV Block 分配与复用、请求状态更新 等。随后，ScheduleBatch 会进一步转换为 ModelWorkerBatch（位于 tp_worker），补充模型执行所需的元数据，最后封装为 ForwardBatch 并发送给 ModelRunner 执行一次 Forward。
  ```
    The following is the flow of data structures for a batch:

    ScheduleBatch -> ModelWorkerBatch -> ForwardBatch
    - ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU
  ```

* Cache 类相关代码




## 总结


## Reference
<span id="ref-1"></span>[1] [code-walk-through.](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md)


