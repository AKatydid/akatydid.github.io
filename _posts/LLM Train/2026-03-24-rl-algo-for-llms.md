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

| 名称          | 机构与时间       | 详情                                                                                                                                                                                                                        |
| ------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [PPO](#ppo)   | (17.07)OpenAI    | 标准的 RL 训练范式，通过 clip、KL散度等措施限制策略更新幅度，实现稳定训练。<br> 但是LLM中 Value Model 难训，自然语言噪声大，难以根据当前片段预测未来得分。<br> 此外奖励模型基于 SFT 分布训练，RL 后易造成**分布偏移问题**。 |  |
| [GRPO](#grpo) | (25.01)DeepSeek  | 提出组优化思路，在组内对奖励做归一化，无需训练 Value Model。 <br>  但仅在序列末尾分配奖励会导致梯度信号稀疏，且组内奖励存在**奖励坍塌问题**。                                                                               |
| [DAPO](#dapo) | (25.03)字节,清华 | 在 GRPO 基础上加入大量工程改进（如 Clip-Higher、Dynamic Sampling 等），一定程度缓解大模型 RL 的训练瓶颈，但仍停留在 token 级。                                                                                              |
| [GSPO](#gspo) | (25.07)阿里      | 实现范式转变，将 off-policy 与 clip 全部提升到**序列级**，显著降低方差，兼具算法简洁性与性能表现，已成为 Qwen3 RL 的核心实践框架。                                                                                          |
| [GFPO](#gfpo) | (25.08)微软      | 针对同时优化多个所需属性的目标进行优化，加入数据过滤操作。                                                                                                                                                                  |
| [GDPO](#gdpo) | (26.01)NVIDIA    | 针对“正确性/格式/长度/安全/偏好...”的多目标奖励混合，计算 advantage 时做解耦。                                                                                                                                              |

## 一、RL 如何与 LLM 结合？

强化学习（RL）的核心思想是：通过**采样与试错**，在未知环境中优化长期回报。

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/case_rl_for_llm.png){: width="800" height="500" }

在传统场景（如游戏）中，agent 通过选择动作（上/下/左/右）与环境交互，并根据最终奖励调整策略，如下图 (a) 所示，其目标可以形式化为：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

而在 LLM 中，如上图(b)所示，一个关键的建模转化是：**将“生成文本”视为一个序列决策过程**。

- 状态：
$(q, o_{<t})$，其中，$o_{<t}=(o_1,o_2,...,o_{t-1})$ 代表第 t 个 token 之前采样的所有 token。

- 动作：
$
o_t
$

- 策略（token-level）：
$
\pi_\phi(o_t \mid q, o_{<t})
$

虽然策略是逐 token 定义的，但整个序列的概率可以通过链式法则表示为：
$$
\pi_\phi(o|q) = \prod_{t=1}^{T} \pi_\phi(o_t \mid q, o_{<t})
$$。
其中，一条生成序列 \(o\) 就对应 RL 中的一条 trajectory \($\tau$\)。

LLM 的 post-training 阶段通常从一个监督微调模型（SFT）出发，优化目标如下所示：

$$
\mathrm{objective}(\phi)
= \mathbb{E}_{(q,o)\sim \mathcal{D}_{\pi_\phi^{\mathrm{RL}}}}
\left[
r_\theta(q,o)
- \beta \log \frac{\pi_\phi^{\mathrm{RL}}(o \mid q)}{\pi^{\mathrm{SFT}}(o \mid q)}
\right]
$$

依据上式，RL for LLMs 本质是在 SFT LLMs 分布附近，通过 reward 对生成分布进行有约束的重加权。

> 补充说明：
> 
> SFT 阶段，模型通过最大化标注数据中标准答案的 token logits 进行训练。
> 给定来自数据分布 $P_{sft}(Q, O)$ 的问题-答案对 $(q, o)$，模型的目标是学习条件概率 $\pi_\theta(o \mid q)$。
>
> 具体而言，SFT 采用 token-level 的最大似然训练目标：

> $$J_{SFT}(\theta) = \mathbb{E}_{\substack{q, o \sim P_{sft}(Q,O)}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \log \pi_\theta(o_t \mid q, o_{<t}) \right]$$

## 二、PPO

对于 LLM 中的 PPO 算法，优化的是整个序列（Sequence）的质量，但优势（Advantage）的计算和策略梯度是在 token 级别分解的。

具体而言，为了约束策略分布不偏离预训练的 SFT 模型太远并保证稳定性，PPO 在 LLM 的实践中通常会将 KL 散度作为**内在惩罚项**直接加到每个 token 的环境奖励上：

$$
r'_t = r_t - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{ref}}(o_t|q, o_{<t})}
$$

在这个修改后的奖励 $r'_t$ 基础上，PPO 会利用一个额外的 Value Model $V_\phi$ 预估每个状态的价值 $V_\phi(s_t)$，从而计算出优势估计 $A_t$。最常用的优势计算方法是 **GAE (Generalized Advantage Estimation)**。GAE 引入了时间差分（TD）误差 $\delta_t$ 和平滑参数 $\lambda$，能在偏差与方差之间取得较好的平衡：

单步的 TD 误差为：
$$
\delta_t = r'_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

当前时间步的 GAE 优势为后续所有 TD 误差的指数加权和：
$$
A_t = \sum_{l=0}^{|o|-t} (\gamma \lambda)^l \delta_{t+l}
$$

这里的优势 $A_t$ 用于衡量在给定上下文 $s_t = (q, o_{<t})$ 下，生成当前 token $o_t$ 相比平均预估价值（Baseline）要高出多少。

随后，通过优化以下 Clip 目标函数来更新策略模型：

$$
J_{\text{PPO}}(\theta) =
\mathbb{E}_{(q,o)} \left[ \frac{1}{|o|}\sum_{t=1}^{|o|}
\min \left(
\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})} A_t,\;
\mathrm{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon\right) A_t
\right)
\right]
$$

尽管引入了 Clipping 和 KL 散度等机制来约束策略偏移，PPO 在 LLM 场景下仍面临一个根本性的训练瓶颈：由于自然语言空间的高维复杂性与极其脆弱的分布特性，Value Model 很难仅凭当前上下文 $(q, o_{<t})$ 这种局部片段去准确预测未来的全局奖励。
这导致 Advantage 充满噪声，严重影响了训练稳定性。

## 三、GRPO

由于 PPO 中使用的 Value Model 通常是一个与 Policy Model 规模相当的模型，这会带来**巨大的内存占用与计算开销**。
此外，在强化学习训练过程中，Value Model 被用作 Advantage 计算的基准值，以降低梯度估计的方差。
但在大语言模型场景下，奖励模型通常仅为最后一个 token 分配奖励分数，这会让需要对每个 token 都保持精准的 Value Model 训练变得更加困难。

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/ppo_and_grpo.png){: width="600" height="500" }

因此，如上图所示，GRPO 移除了 Value Model，通过组内奖励归一化估计优势函数（Advantage）。

GRPO 针对同一个问题，将模型多次采样输出的奖励平均值作为基准值。
具体而言，对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$中采样一组输出 $$\{o_1,o_2,...,o_G\}$$，并通过最大化下述目标函数来优化策略模型：

$$
\small{
\begin{align*}
\mathcal{J}_{\text{GRPO}}(\theta) &= \mathbb{E}\left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right] \nonumber \\
&\quad \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg\{
\min\Bigg[
\frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},\,
\mathrm{clip}\Bigg(
\frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})},\,
1-\varepsilon,\, 1+\varepsilon
\Bigg) \hat{A}_{i,t}
\Bigg]
- \beta \mathrm{D}_{\mathrm{KL}}\big[\pi_{\theta} \parallel \pi_{\text{ref}}\big]
\Bigg\}
\end{align*}
}
$$

其中，$\epsilon$ 和 $\beta$ 为超参数。
优势函数 $\hat{A}_{i,t}$ 仅基于组内输出的相对奖励计算得到。
具体而言，对于每个问题 $q$，旧模型 $$\pi_{\theta_{old}}$$ 中采样一组输出 $$\{o_1,o_2,...,o_G\}$$，得到对应的 G 个奖励值 $$r=\{r_1,r_2,...,r_G\}$$，通过减去组内均值、除以组内标准差实现归一化，然后输出所有 Token 的优势函数：

$$
\hat{A}_{i,t}=\tilde{r}_i=\frac{r_i-mean(r)}{std{r}}
$$

需要注意，GRPO 为避免优势函数 $\hat{A}_{i,t}$计算变得复杂，**KL 散度未直接加在奖励中**，而是直接在损失函数中引入训练策略与参考策略之间的 KL 散度作为正则项。

GRPO 虽然通过组内平均的方法移除了对 Value Model 的依赖， 但是在训练中经常会出现稳定性问题，具体表现为以下三个方面：

1. 多种任务奖励不均匀导致的任务崩溃问题
2. 长时间训练导致的后期稳定性问题
3. 熵坍塌问题

### 3.1 多任务奖励均匀性问题

实际使用中会对不同的任务进行训练，目前 policy 基本上还是抽卡奖励的行为。
无论是针对不同难度任务使用不同奖励，还是基于性能表现采用同一套奖励规则，**不同难度的任务之间的奖励分布存在显著差异。**

- 简单任务：reward 均值较高，例如，在 ReLU Kernel 上很容易取得正确性和性能加速比的奖励，模型很快就能泛化到不同输入维度上。
- 困难任务：reward 均值较低，例如，在 Convolution 这类较难的任务上，对应的 token 模式优化速度较慢；Cumsum（SFT 学到的是双重 for 循环的解法）在 RL 阶段也无法从算法层面改进。

**解决方案：**
* **平衡奖励。**
实际训练前开始测试，统计不同难度任务在部分样本上的均值，然后对奖励函数进行调整，将不同难度任务的奖励均值拉到同一水平。

* **预奖励过滤。**
使用初始的模型进行样本生成，将获得满分的样本从数据集中按照一定比例剔除（通常不会全部删除）。

* 【只对回归属性的任务有效】**高奖励死区**。
对困难任务设置高奖励内容的死区，认定在一定的偏差内可以直接获得满分或者另一个均值接近满分的分段奖励。

### 3.2 长时间训练导致的后期稳定性问题

Token 级的 RL 微调在原理上均会面对后期训练稳定性问题：高奖励 token 会在优化过程中逐渐占据主导地位，强烈冲击并改写模型原有的分布结构。
实际中，数据集很难做到完美，因此这种“优势token”的存在不可避免。
这些 token 会持续放大分布偏移，逐步破坏模型原本相对均匀、稳定的概率分布；随着这种偏移的不断累积，最终可能导致模型在训练后期发生突发性的崩溃。

**解决方案：**
* **增大 batch_size**。在高batch的情况下，尖峰奖励会被平滑，但需要更多计算资源。
* **改善信度分配**。
对整个样本进行奖励然后广播奖励到 token 可以比较好的改善“优势token”累积过度的问题 => GSPO 的解决方案。
* **保持训练过程的高熵性**同样能有效缓解长时间训练的崩溃问题。

### 3.3 熵坍塌问题

熵过早的衰减导致rollout的过程无法有效的获取多样性的样本，从而使模型陷入到难以优化的情况，常出现如下图所示的情况。

![Desktop View](assets/img/blog/RL_Algo_for_LLMs/entropy_collapse.png){: width="400" height="300" }

**解决方案：**
* 进行熵抑制。通过**裁剪策略梯度保持高熵**，这种方式可以保证熵不会过早的衰减；但是同时也会进一步限制模型的收敛性
* **动态采样**。如果一个采样组的奖励没有区别时，动态的再进行额外的采样，让模型在足够多的采样中获取可能的不同的结果 => DAPO 的解决方案。

## 四、DAPO

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



## 五、GSPO

- 将 off-policy 与 clip 提升到序列级，降低优化方差。
- 在简洁性与效果上较平衡，适合大规模 RL 实践。

## 六、GFPO

- 面向多属性联合优化。
- 通过数据过滤配合训练目标，增强对多维指标的可控性。

## 七、GDPO

- 面向多目标奖励混合场景。
- 在 advantage 计算阶段做解耦，减少不同目标之间的相互干扰。


## Reference
[1] 知乎：如何改善GRPO的稳定性. https://zhuanlan.zhihu.com/p/1972697530946041490.


