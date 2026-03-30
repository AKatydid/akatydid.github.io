---
layout: post
title: RL-Algo-for-LLMs
date: 2026-02-24 06:44 +0000
categories: [LLM]
tags: [LLM, RL, Training]
pin: true
math: true
mermaid: true
picture:
  path: 
---

## 算法汇总

| 名称          | 机构与时间       | 详情                                                                                                                                                                                                                                                                                        |
| ------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [PPO](#ppo)   | (17.07)OpenAI    | 通过裁剪重要性采样比限制策略更新范围，稳定训练；<br> 但是， 在 LLM 训练中 Value Model 很难学，自然语言噪声太大，很难根据当前片段预测未来得分；<br> 此外，由于奖励模型 $RM$ 是在 SFT 输出分布上训练的，RL 后的输出分布可能与差异显著，造成**分布偏移问题**，导致$RM$对新分布的预测可靠性下降 |
| [DPO](#dpo)   | (23.05)斯坦福    | 将偏好优化转化为分类问题，无需显式的RL采样，直接在固定的偏好数据集上训练，完全避免了在线分布偏移问题。<br> 但是，代价是无法利用训练过程中策略改善产生的新信息。                                                                                                                             |
| [GRPO](#grpo) | (25.01)DeepSeek  | 提出组优化思路，在组内对奖励做归一化，从而摆脱 Value Model 的依赖。 <br> 但仍停留在 token 级，方差依旧较大，且组内奖励存在**奖励坍塌**的问题                                                                                                                                                |
| [DAPO](#dapo) | (25.03)字节,清华 | 在 GRPO 基础上加入大量工程改进（如 Clip-Higher、Dynamic Sampling 等），一定程度缓解大模型 RL 的训练瓶颈，但仍停留在 token 级。                                                                                                                                                              |
| [GSPO](#gspo) | (25.07)阿里      | 实现范式转变，将 off-policy 与 clip 全部提升到**序列级**，显著降低方差，兼具算法简洁性与性能表现，已成为 Qwen3 RL 的核心实践框架。                                                                                                                                                          |
| [GFPO](#gfpo) | (25.08)微软      | 针对同时优化多个所需属性的目标进行优化，加入数据过滤操作。                                                                                                                                                                                                                                  |
| [GDPO](#gdpo) | (26.01)NVIDIA    | 针对“正确性/格式/长度/安全/偏好...”的多目标奖励混合，计算 advantage 时做解耦。                                                                                                                                                                                                              |

## 一、RL 如何与 LLM 结合？

强化学习（RL）的核心思想是：通过**采样与试错**，在未知环境中优化长期回报。

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/case_rl_for_llm.png){: width="800" height="500" }

在传统场景（如游戏）中，agent 通过选择动作（上/下/左/右）与环境交互，并根据最终奖励调整策略，如下图 (a) 所示，其目标可以形式化为：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

而在 LLM 中，如下图(b)所示，一个关键的建模转化是：**将“生成文本”视为一个序列决策过程**。

- 状态：
$s_t = (x, y_{<t})$，其中，$y_{<t}=(y_1,y_2,...,y_{t-1})$ 代表第 t 个 token 之前所有 token （历史动作 / token序列）。

- 动作：
$
a_t = y_t
$

- 策略（token-level）：
$
\pi_\phi(y_t \mid x, y_{<t})
$

虽然策略是逐 token 定义的，但整个序列的概率可以通过链式法则表示为：
$$
\pi_\phi(y|x) = \prod_{t=1}^{T} \pi_\phi(y_t \mid x, y_{<t})
$$。
其中，一条生成序列 \(y\) 就对应 RL 中的一条 trajectory \($\tau$\)。

LLM 的 post-training 阶段通常从一个监督微调模型（SFT）出发，优化目标如下所示：

$$
\mathrm{objective}(\phi)
= \mathbb{E}_{(x,y)\sim \mathcal{D}_{\pi_\phi^{\mathrm{RL}}}}
\left[
r_\theta(x,y)
- \beta \log \frac{\pi_\phi^{\mathrm{RL}}(y \mid x)}{\pi^{\mathrm{SFT}}(y \mid x)}
\right]
$$

依据上式，RL for LLMs 就是在 SFT LLMs 分布附近，通过 reward 对生成分布进行有约束的重加权。

## 二、PPO

PPO 算法优化的是整个 sequence 的质量，但梯度是在 token 级别分解计算的。
优化目标如下所示：


$$
J^{\text{PPO}}(\theta) =
\mathbb{E}_{t} \left[
\min \left(
\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_t,\;
\mathrm{clip}(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

其中，优势函数 $A_t = Q(s_t,a_t) - V(s_t)$ 用于表示在当前状态 $s_t$ 采取动作 $a_t$ 相对 $s_t$ 的平均收益高多少，决定哪些采样行为应该被强化，哪些应该被抑制；$V(s_t)$ 相当于提供了一个 Baseline，避免一次参数更新过大的情况，实现**稳定训练**。

但是，由于自然语言固有的高维复杂性，分布极其脆弱，Value Model 难以根据当前状态 $s_t$ 的 token 片段去预测未来奖励。

## 三、DPO



## 四、GRPO

GRPO 移除了 Value Model，通过组内奖励归一化估计优势函数（Advantage）。

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/ppo_and_grpo.png){: width="600" height="500" }

但是，GRPO 在训练中经常会出现稳定性问题，具体表现为以下三个方面：

1. 多种任务奖励不均匀导致的任务崩溃问题
2. 长时间训练导致的后期稳定性问题
3. 熵坍塌

### 4.1 多任务奖励均匀性问题

实际使用中会对不同的任务进行训练，目前 policy 基本上还是抽卡奖励的行为。
无论是针对不同难度任务使用不同奖励，还是基于性能表现采用同一套奖励规则，**不同难度的任务之间的奖励分布存在显著差异。**

- 简单任务：reward 均值较高，例如，在 ReLU Kernel 上很容易取得正确性和性能加速比的奖励，模型很快就能泛化到不同输入维度上。
- 困难任务：reward 均值较低，例如，在 Convolution 这类较难的任务上，对应的 token 模式优化速度较慢；Cumsum（SFT 学到的是双重 for 循环的解法）在 RL 阶段也无法从算法层面改进。

### 4.2 长时间训练导致的后期稳定性问题



### 4.3 熵坍塌问题

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/entropy_collapse.png){: width="600" height="500" }


## 五、DAPO

### DAPO Motivation

GRPO 存在以下几个问题：
* Token 级别的 Clip 容易导致**熵崩溃**：模型很快收敛到少量固定答案，导致多样性和探索能力不足。
* Batch 采样中出现**奖励极端化**：部分样本的奖励可能全部为 1 或 0，从而产生「零梯度」问题，削弱训练信号。
* 长序列训练的梯度分布失衡：权重分布让极少数 token 的梯度占据主导，导致许多高质量的长序列样本被忽视。

### DAPO 解决方案

1. Clip-Higher 机制：将 Clip 的上下限分开 ，研究者将较低和较高的剪辑范围解耦为 ε_low 和 ε_high，研究者增加了 ε_high 的值，以便为低概率 token 的增加留出更多空间，能够显著提升模型训练早期的熵。
2. 动态采样：进行过度采样，过滤掉奖励等于 1 和 0 的提示语，只保留有效梯度的样本，提高训练效率。
3. Token 级策略梯度损失：对所有 token 一起求平均，保证长序列的所有
token 都公平地为 batch loss 做贡献，并防止长序列的优化梯度被过度缩小。
4. 超长奖励调整：针对超长样本，当响应长度超过预定义的最大值时，研究者定义一个「soft 罚分」。在这个区间内，响应越长，受到的惩罚就越大，以此避免过长的响应。



## 六、GSPO

- 将 off-policy 与 clip 提升到序列级，降低优化方差。
- 在简洁性与效果上较平衡，适合大规模 RL 实践。

## 七、GFPO

- 面向多属性联合优化。
- 通过数据过滤配合训练目标，增强对多维指标的可控性。

## 八、GDPO

- 面向多目标奖励混合场景。
- 在 advantage 计算阶段做解耦，减少不同目标之间的相互干扰。
