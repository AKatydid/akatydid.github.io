---
layout: post
title: SGLang Source Code Analysis (1)
date: 2026-06-27 15:59 +0800

categories: [sglang]
tags: [sglang, infra]
pin: false
math: true
mermaid: true
---

SGLang 是目前最常用的工业级推理引擎之一。
为了深入理解其高性能推理背后的实现机制，本文将从源码角度，对 SGLang 后端的整体架构和执行流程进行系统梳理。

文章将沿着一次推理请求的执行路径，从框架入口开始，逐步分析各个核心模块的职责，以及请求如何在不同组件之间流转，并在后续章节进一步深入到 CUDA 算子实现与性能优化细节。

> **本文环境：**
> - SGLang v0.5.10
> - CUDA Version: 12.8

## 1.SGLang 框架概述

![Desktop View](assets/img/sglang/sglang-architecture.svg){: width="800" height="600" }
<center>SGLang 框架概述图 <a href="#ref-1">[1]</a></center>

### 1.1 初始化过程

SGLang 核心初始化函数 [Engine._launch_subprocess(...)](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/entrypoints/engine.py#L620-L754)，初始化了 SRT (SGLang Runtime) Server。
* HTTP Server: FastAPI 组件
* Engine Subprocess
    * TokenizerManager
    * Scheduler
    * DetokenizerManager

#### 1.1.1 Scheduler Init 过程
源码：[_launch_scheduler_processes(...)](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/entrypoints/engine.py#L513-L594)

```
                _launch_scheduler_processes()
                           │
            ┌──────────────┴──────────────┐
            │                             │
      dp_size == 1                  dp_size > 1
            │                             │
      启动 Scheduler                 启动 DP Controller
            │                             │
      每个 Scheduler                  Controller 再启动
      对应一个 GPU                    多个 Scheduler
            │
            ▼
      建立 Pipe 通信
            │
            ▼
      返回 SchedulerInitResult
```


### 1.2 


## Reference
<span id="ref-1"></span>[1] [code-walk-through.](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md)


