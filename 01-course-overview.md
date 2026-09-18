# CS329A 第 1 讲：课程概览

> 课程：Stanford CS329A — Self-Improving AI Agents（2025 年秋季）  
> 主讲：Aakanksha Chowdhery、Azalia Mirhoseini  
> [视频（约 70 分钟）](https://www.youtube.com/watch?v=6YnLB0XbTnI) · [课程官网与阅读材料](https://cs329a.stanford.edu/)  
> 本文按讲座内容归纳，不是逐字稿；时间链接可跳回视频对应位置。

## 一句话主线

大模型先通过扩大**训练计算量、数据量和参数量**获得基础能力，再经**高质量数据微调、指令微调与人类反馈**变得易用；随后可通过**推理时计算与验证**提高解题能力。Agent 则进一步把模型置于“目标 → 计划 → 行动/工具 → 反馈/验证 → 调整或停止”的闭环里。课程研究的核心，是怎样让这个闭环更可靠，并把有效反馈用于下一轮改进。

## 1. 大模型能力从哪里来？[02:23](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=143s)

- **预训练扩展（scaling）**：讲座展示训练计算量、数据集规模、参数规模增加时，测试损失下降的趋势。这解释了为什么更大的基础模型通常有更强的语言与任务能力；它是经验规律，并不保证所有任务都以同样幅度受益。
- **零样本与少样本学习**：[05:39](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=339s) 零样本只提供任务说明，少样本额外提供几个输入输出示例。两者都不需要针对这个任务重新训练模型参数。
- **思维链（chain of thought, CoT）**：[07:03](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=423s) 在示例中加入中间推理步骤，可能帮助模型解决新问题。讲座用数学题说明这一点，并指出所展示的较小模型不一定能从 CoT 提示受益。这里的“涌现”描述的是讲座引用的实验现象，不应理解为存在通用的参数门槛。

## 2. 从基础模型到会遵循指令的助手 [11:20](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=680s)

| 阶段 | 主要做法 | 作用 |
| --- | --- | --- |
| 预训练 | 在大量文本等数据上预测下一个 token | 获得广泛的语言与知识能力 |
| 高质量数据微调 | 继续用筛选过的数据训练 | 改善输出质量 |
| 指令微调 | 学习“指令／问题 → 合适回答”的示例，也可包含推理步骤 | 学会按请求完成任务 |
| RLHF | 收集人类对多个回答的偏好，训练奖励模型，再据此优化回答 | 更贴近人类偏好，例如有用性与安全性 |

讲座强调：模型在预训练中学到大量统计规律，并不等于天然知道如何遵循用户意图。后训练的数据质量、覆盖范围以及反馈目标，都会改变实际使用体验。[14:26](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=866s) [17:01](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1021s)

## 3. 推理时计算与自我改进 [19:25](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1165s)

**推理时扩展（test-time / inference-time scaling）**是在模型参数固定时，投入更多生成、搜索或思考计算。讲座用 *Large Language Monkeys* 说明一种简单形式：对同一道题采样多个候选答案，再用验证器挑选。[21:01](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1261s)

```text
问题 → 固定模型生成多个候选 → 验证或排序 → 选出答案
```

- 数学答案、代码单元测试等提供相对明确的验证信号。开放式写作等任务较难建立可靠验证器，可能需要人类反馈或模型评审。[26:43](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1603s)
- **覆盖率／pass@k**问“k 个候选中是否至少一个正确”；**pass@1**问“一次输出是否正确”。讲座的重复采样实验主要展示前者；若无法识别正确候选，覆盖率高也不等于用户会收到正确答案。[22:23](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1343s) [36:37](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2197s)
- 计算成本会随采样增加；并行生成可缓解部分等待时间，但不能消除算力成本。采样多样性也需控制，温度过高会降低答案质量。[26:18](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1578s)
- **自我改进闭环**：用模型产生候选解，用可靠反馈筛出高质量推理或代码，再把这些结果作为训练数据，改善之后的一次输出能力。这是讲座从“推理时扩展”过渡到“自我改进”的关键。[28:28](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1708s)

推理模型还会分析问题、拆解任务、尝试方案、读取反馈、回退并修正。课程把这些能力与训练、推理时搜索和奖励信号联系起来；其确切贡献机制仍有研究空间。[31:22](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1882s) [51:20](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3080s)

## 4. 从聊天模型到 Agent [40:58](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2458s)

讲座给出的实用定义是：Agent 接收一个**目标**，规划并与环境交互，依据反馈调整行动，最后判断目标是否完成或报告无法完成。[42:20](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2540s)

```text
理解目标 → 规划 → 调用工具并观察结果 → 验证 → 修正/继续 → 结束
```

课程区分两种实现形态：

| 形态 | 特点 | 例子 |
| --- | --- | --- |
| 预先设计的 agentic workflow | 人设计步骤与分支，模型在各节点执行任务 | 提示链、任务路由、并行研究与汇总 |
| 更开放的 Agent 循环 | 模型依据环境反馈动态决定下一步 | 编码 Agent 搜索仓库、修改文件、运行测试、再修改 |

常见组件包括工具调用、上下文或记忆、协调器、评审器与验证器。**评审器**可用模型判断答案质量；**验证器**则尽可能利用可检查的外部结果，例如运行代码和单元测试。能否获得可靠反馈，是系统继续改进的关键瓶颈。[44:50](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2690s) [47:11](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2831s) [50:34](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3034s)

**讲座案例**：编码 Agent 进行文件检索、编辑与测试；研究 Agent 搜集资料、拟定提纲并综合报告；客户支持系统完成转录、知识检索与回复；科研 Agent 辅助产生想法、迭代实验和写作。这些案例展示了可用方向，不代表所有环节都已能稳定全自动完成。[48:04](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2884s) [54:34](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3274s) [55:49](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3349s)

## 5. 课程路线与项目 [01:00:32](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3632s)

课程官网把后续内容组织为：推理时计算扩展、可靠验证、从工具与代码反馈中学习、多步推理与规划、训练时强化学习、开放式演化、搜索与深度研究、软件工程 Agent、记忆、长任务评估和机器人多模态 Agent。学习形式包括论文阅读、作业和原创研究项目。

2025 年秋季开课安排包含 **3 次作业与 1 个研究项目**；官网给出的总评比例为作业 50%，项目各阶段合计 50%。项目鼓励提出可检验的研究问题，例如新基准、Agent 可靠性评估，或对已有方法的改进；单纯拼装应用或文献综述不符合讲座描述的研究目标。官网列出的日期与政策属于 **2025 年秋季学期**，以后修读应以当期课程页面为准。[01:02:17](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3737s) [01:03:39](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=3819s) [课程官网](https://cs329a.stanford.edu/)

## 复习时回答这 4 个问题

1. 为什么参数规模之外，还要讨论数据、训练计算量和后训练？
2. pass@k 提升为什么不能直接证明用户拿到的单个答案也更好？
3. 哪些任务能提供可靠验证信号？验证困难时系统会遇到什么瓶颈？
4. 一个完成端到端任务的 Agent，需要哪些状态、工具、反馈和停止条件？

## 资料与核对

- [原始讲座视频](https://www.youtube.com/watch?v=6YnLB0XbTnI)
- [Stanford CS329A 课程官网（2025 年秋季）](https://cs329a.stanford.edu/)
- [带时间戳的公开视频字幕索引](https://ainotes.us/summary/550)：用于定位讲座内容；自动字幕可能有识别错误。课程名、讲师、安排及评分以官网为准。
