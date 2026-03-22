---
# the default layout is 'page'
title: ABOUT ME
icon: fas fa-info-circle
order: 4
---

<style>
	.resume-page {
		--resume-accent: #4870ad;
		--resume-text: #353a42;
		--resume-soft: #6f7785;
		--resume-border: #dbe4ee;
		--resume-bg: #ffffff;
		font-family: "Source Han Sans SC", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
		color: var(--resume-text);
		max-width: 920px;
		margin: 0 auto;
		padding: 1rem 1.25rem 1.5rem;
		background: var(--resume-bg);
		border: 1px solid var(--resume-border);
		border-radius: 12px;
		box-shadow: 0 8px 28px rgba(33, 54, 84, 0.08);
	}

	.resume-page h1,
	.resume-page h2,
	.resume-page h3,
	.resume-page p,
	.resume-page ul,
	.resume-page ol,
	.resume-page li {
		margin-top: 0;
		margin-bottom: 0;
	}

	.resume-page h1 {
		font-family: "Source Han Serif SC", "Noto Serif SC", "STSong", serif;
		font-size: 2rem;
		line-height: 1.25;
		letter-spacing: 0.02em;
		margin-bottom: 0.4rem;
	}

	.resume-contact {
		margin-bottom: 0.9rem;
		font-family: "JetBrains Mono", "SFMono-Regular", "Consolas", monospace;
		color: #2a3648;
		font-size: 0.95rem;
		line-height: 1.75;
	}

	.resume-contact span {
		margin-right: 1rem;
		white-space: nowrap;
		display: inline-block;
	}

	.resume-intent {
		margin-bottom: 0.7rem;
		color: var(--resume-soft);
		font-size: 0.95rem;
	}

	.resume-page h2 {
		color: var(--resume-accent);
		font-size: 1.08rem;
		font-weight: 700;
		padding: 0.42rem 0 0.34rem;
		margin-top: 0.95rem;
		margin-bottom: 0.56rem;
		border-bottom: 1px solid color-mix(in srgb, var(--resume-accent), transparent 65%);
	}

	.resume-page h3 {
		font-size: 1rem;
		font-weight: 700;
		line-height: 1.5;
		margin-bottom: 0.2rem;
	}

	.resume-meta {
		font-size: 0.92rem;
		color: var(--resume-soft);
		margin-bottom: 0.25rem;
	}

	.resume-page ul,
	.resume-page ol {
		padding-left: 1.2rem;
		margin-bottom: 0.4rem;
		line-height: 1.72;
		font-size: 0.96rem;
	}

	.resume-page p {
		font-size: 0.96rem;
		line-height: 1.72;
		margin-bottom: 0.4rem;
	}

	.resume-grid {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0.5rem 1rem;
		align-items: baseline;
		margin-bottom: 0.2rem;
	}

	.resume-date {
		font-size: 0.9rem;
		color: var(--resume-soft);
		white-space: nowrap;
	}

	.resume-page a {
		color: #2e5f9f;
		text-decoration: none;
	}

	.resume-page a:hover {
		text-decoration: underline;
	}

	@media (max-width: 780px) {
		.resume-page {
			padding: 0.9rem 0.9rem 1.2rem;
			border-radius: 10px;
		}

		.resume-grid {
			grid-template-columns: 1fr;
			gap: 0.15rem;
		}

		.resume-date {
			margin-bottom: 0.2rem;
		}
	}
</style>

<div class="resume-page" markdown="1">

# 朱鑫国

<p class="resume-intent">求职意向：算法工程师 | AI Infra</p>

<p class="resume-contact">
	<span>☎ (+86) 19168158265</span>
	<span>✉ xinguojoe@gmail.com</span>
	<span>GitHub: <a href="https://github.com/Akatydid">Akatydid</a></span>
</p>

## 教育经历

<div class="resume-grid">
	<h3>中国科学院大学（硕士）· 人工智能专业</h3>
	<span class="resume-date">2023.09 - 2026.06</span>
</div>

- GPA：3.72 / 4.00
- 研究方向：基于大模型的 GPU 内核代码自动生成方法研究
- 奖项：校二等奖学金

<div class="resume-grid">
	<h3>华北电力大学（本科）· 软件工程专业</h3>
	<span class="resume-date">2019.09 - 2023.06</span>
</div>

- GPA：3.51 / 4.00（排名 6/64，Top 10%）
- 奖项：校二等奖学金、校三好学生、系学优奖学金

## 科研成果

1. QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation（AAAI 26'，CCF-A，第一作者）
2. 基于大语言模型的多智能体强化学习环境自动构建方法及系统（发明专利，第一作者）
3. Think as Compiler, Write as Human: Compiler-Trained LLM for Search-Free High-Performance CUDA Generation（OSDI 26'，CCF-A，第四作者，在投）
4. QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives（IJCAI 25'，CCF-A，第七作者）

## 工作经历

<div class="resume-grid">
	<h3>百度（实习）· 大模型基建组</h3>
	<span class="resume-date">2025.11 - 2026.03</span>
</div>

- 相关项目：[PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet)、[PaddlePaddle/AI4C](https://github.com/PaddlePaddle/AI4C)
- 项目简介：AI4C / GraphNet 项目，旨在以 AI for System 的方式评估并加速国产芯片上 AI 软件栈的发展进程。
- 负责内容：参与 GraphNet 数据集建设、轨迹数据蒸馏，并深度参与 AI4C。
- 代表 PR：
	- 数据集建设：[ai4c#10](https://github.com/PaddlePaddle/ai4c/pull/10)
	- 经典子图拆分算法：[GraphNet#328](https://github.com/PaddlePaddle/GraphNet/pull/328)、[GraphNet#351](https://github.com/PaddlePaddle/GraphNet/pull/351)
	- AI4C Agent：[ai4c#1](https://github.com/PaddlePaddle/ai4c/pull/1)、[ai4c#8](https://github.com/PaddlePaddle/ai4c/pull/8)
	- 防止 Hack 机制与 PassManager：[ai4c#32](https://github.com/PaddlePaddle/ai4c/pull/32)
- 核心产出：GraphNet: Towards Automated Graph Compiler Passes Generation via LLMs（ASE 26'，CCF-A，在投）

## 项目经历

<div class="resume-grid">
	<h3>基于大语言模型的高性能内核代码自动生成</h3>
	<span class="resume-date">2024.09 - 至今</span>
</div>

项目简介：面向硬件迭代与算子重构需求，利用大模型自动生成高性能 GPU 内核，兼顾开发效率与执行性能。

主导项目：[QiMeng-Kernel](https://github.com/QiMeng-IPRC/QiMeng-Kernel)

- 项目简介：“宏观思维”模块通过强化学习引导轻量级 LLM 学习语义化优化策略，“微观编码”模块由通用 LLM 多步迭代实现。宏观-微观分层解耦生成框架能够显著降低 GPU 内核生成错误率，赋能大模型生成正确且高性能的内核代码。
- 成果：
	- KernelBench：L1/L2 准确率 100%，L3 准确率 70%，较 SOTA 提升超过 50%；性能较通用 LLM 提升 7.3 倍，较专家优化的 PyTorch Eager 提升 2.2 倍。
	- TritonBench：准确率 59.64%，性能提升 34 倍。

深度参与项目：

- QiMeng-TensorOp：负责矩阵乘算子（Tensor Core）生成，基于大模型 + 蒙特卡洛树搜索生成 CUTE 代码。在规整维度下较 Ansor 提升 1.03 - 1.55 倍，接近 cuBLAS（平均达到 95% 性能），相关成果发表于 IJCAI 25'。
- NEUPILER：训练 Qwen-8B 作为张量编译器，search-free 生成高性能 CUDA 内核。负责语料收集（源于 Ansor、MetaSchedule）、清洗与 SFT；生成内核性能最高达 1.41 倍 Ansor，且无需昂贵搜索过程。

## 专业技能

- 熟练使用 Python、C++；熟练使用 CUDA、Triton。
- 掌握常见强化学习算法；掌握 LoRA 微调和全量微调；具备多卡微调大模型经验。
- 掌握 TorchInductor 原理，了解 PyTorch 深度学习框架；熟练使用 TVM 张量编译器。

</div>
