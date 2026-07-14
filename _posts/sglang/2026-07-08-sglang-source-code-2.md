---
layout: post
title: SGLang 源码学习（2）—— Forward Process on CPU 
date: 2026-07-14 15:59 +0800

categories: [sglang]
tags: [sglang, infra]
pin: false
math: true
mermaid: true
---

SGLang 是目前最常用的工业级推理引擎之一。
为了深入理解其高性能推理背后的实现机制，本文将从源码角度，对 SGLang 的整体架构和执行流程进行系统梳理。

文章将沿着一次推理请求的执行路径，从框架入口开始，逐步分析 kv-cache & request & shcheduler 核心模块的职责，以及请求如何在不同组件之间流转，并在后续章节进一步深入到NVIDIA GPU & CPU 算子实现与性能优化细节。

> **SGLang Version: 0.5.13**


![Desktop View](assets/img/sglang/sglang-architecture.svg){: width="800" height="600" }
<center>Fig 1. SGLang 框架概述图 <a href="#ref-1">[1]</a></center>


## Reference
<span id="ref-1"></span>[1] [code-walk-through.](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md)


