# CS329A 第 3 讲：Robust Verification

> 课程：Stanford CS329A — Self-Improving AI Agents（2025 年秋季）<br>
> 主讲：Azalia Mirhoseini（2025 年 9 月 29 日课堂；视频由 Stanford Online 于 2026 年发布）<br>
> [原视频（约 72 分钟）](https://www.youtube.com/watch?v=p7TdPUcPoik) · [课程官网与本讲阅读材料](https://cs329a.stanford.edu/)<br>
> 本文按讲座归纳，不是逐字稿。时间链接来自发布者提供的英文 CC；关键机制与数字已对照论文，文中截图取自原视频对应幻灯片。听辨不清的课堂问答不作确定转述。

## 一句话主线

Repeated sampling 可以让正确解出现在候选集合中，却不能保证系统能把它挑出来。Robust verification 要解决的是这个 **generation–verification gap（生成—验证落差）**：先学习给完整解答打分的 verifier，继而监督每一步的 process reward model（PRM），再用自动 rollout 代替昂贵的人类逐步标注，最后通过多个有误差的 verifier 组成更强的选择器。这里既有**推理时选答案**，也有把 verifier 当作 reward 改进生成模型的**训练时循环**，两者不能混为一谈。[00:05](https://www.youtube.com/watch?v=p7TdPUcPoik&t=5s) [01:06:29](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3989s)

## 1. 为什么仅仅生成更多答案还不够？[00:14](https://www.youtube.com/watch?v=p7TdPUcPoik&t=14s)

给定问题 `q`，generator 产生 `k` 个候选 `s₁ … sₖ`。**Coverage／pass@k** 问的是“其中是否至少有一个正确”；实际系统的 **selected-answer accuracy** 问的是“verifier 排序后输出的那一个是否正确”。前者随候选增加通常有上升空间，后者却可能因 verifier 误判而停滞甚至下降。Oracle selector 假定总能认出正确候选，只是上界，不是可部署的 verifier。讲师回顾上一讲的 majority voting（按最终答案出现频率投票）：它不需要训练 verifier，但候选继续增加时也可能无法跟上 coverage。[14:40](https://www.youtube.com/watch?v=p7TdPUcPoik&t=880s) [01:00:30](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3630s)

```mermaid
flowchart LR
    Q[问题] --> G[generator 采样多个解答]
    G --> V[verifier 评分与排序]
    V --> O[输出选中的一个]
    G -.-> C["pass@k：增加候选覆盖率"]
    V -.-> D["generation–verification gap：误判会留下落差"]
```

本讲的四篇论文不是同一套模型、数据和算力预算下的统一排行榜；它们依次改变了**反馈粒度、标签来源和验证计算分配方式**。

| 方法 | 主要评分对象 | 标签／信号来源 | 推理时怎么用 | 主要代价或风险 |
| --- | --- | --- | --- | --- |
| [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168) | 整条数学解答 | 已知最终答案对生成解答作对错标注 | best-of-`k` | 错误过程可能碰巧得出正确答案；排序器会误判 |
| [Lightman et al., 2023](https://arxiv.org/abs/2305.20050) | 每个推理步骤 | 人工逐步标注 | 汇总 step scores 后选择 | 人工标注昂贵，步骤定义与评分策略影响结果 |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | 每个步骤到正确终点的潜力 | 从该步骤继续采样，以最终答案自动标注 | PRM 排序；还可作 RL reward | 需要可检查的最终答案和额外 rollout；潜力不等于步骤正确 |
| [Weaver](https://arxiv.org/abs/2506.18203) | 完整候选的多路 verifier 分数 | 多个 reward model／LLM judge，辅以少量标签作筛选 | 标准化、估计可靠性、加权选择 | verifier 相关错误、低质成员和推理成本 |

## 2. 从 GSM8K 到训练完整解答的 verifier [01:15](https://www.youtube.com/watch?v=p7TdPUcPoik&t=75s)

《[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)》同时引入 **GSM8K（Grade School Math 8K）**：约 8,500 道需要多步自然语言推理的小学数学应用题。它提供已知的最终答案，因此能给模型生成的解答自动贴“最终答案对／错”的二元标签；这与请人类逐步审查推理不是同一种监督。[01:50](https://www.youtube.com/watch?v=p7TdPUcPoik&t=110s) [03:48](https://www.youtube.com/watch?v=p7TdPUcPoik&t=228s)

训练流程是先 fine-tune generator，再对每题采样约 100 条 completion，将结果与 ground truth 比较以建立 verifier 训练集。论文的 verifier 是带额外标量预测头的语言模型，结合 correctness prediction 与 language-modeling loss；可在 token／句子位置预测，但用于整条解答排序的是**末尾的预测分数**。测试时重新采样多个候选，由 verifier 选出最高分者。讲座明确区分了人写一次标准答案与模型多次采样、自动标注的两个环节。[04:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=242s) [05:28](https://www.youtube.com/watch?v=p7TdPUcPoik&t=328s) [08:25](https://www.youtube.com/watch?v=p7TdPUcPoik&t=505s) [16:23](https://www.youtube.com/watch?v=p7TdPUcPoik&t=983s)

对照实验显示，在该实验设置中，verifier 有足够训练数据时，generate-and-rank 相比只做 generator 的 supervised fine-tuning 更有利；标签少时优势并不稳定。另一项消融显示“大 generator＋小 verifier”好于相反配置，但讲师把最佳规模比例留作研究问题，不能外推为所有模型的定律。[11:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=662s) [12:55](https://www.youtube.com/watch?v=p7TdPUcPoik&t=775s)

讲座展示的 selected-answer accuracy 在候选数约 400 附近达到峰值，更多样本时回落；原因是搜索规模越大，越可能找到 verifier 给高分的错误解答，**这不是 pass@k 本身下降**。原论文还测试了一个缓解办法：不只采用排名第一的 completion，而让 verifier 排名前几的候选按最终答案投票；100 个候选时让 top 3–5 投票较好，3,200 个候选时约为 top 30。讲师说主要实验采用约 100 次采样，以取得大部分收益并限制成本；`100`、`400` 与 `3,200` 分别对应默认设置、单一最高分选择的峰值和论文探索的更大采样规模。[14:25](https://www.youtube.com/watch?v=p7TdPUcPoik&t=865s) [17:18](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1038s) [论文 §5.1](https://arxiv.org/html/2110.14168#S5.SS1)

![课程幻灯片：候选数增加时 verifier 选答准确率先升后降](assets/03-robust-verification/test-time-compute.png)

*读图：横轴是每题 completion 数，从 25 增至 3,200；纵轴是 6B verifier 系统最终选中答案的 test solve rate，而非 oracle coverage。曲线约在 400 个候选处见顶，随后下降，说明扩大搜索也扩大了撞到 verifier false positive 的机会。截图取自[原视频 14:20](https://www.youtube.com/watch?v=p7TdPUcPoik&t=860s)，对应 Cobbe et al. Figure 7(a)；top-ranked voting 的缓解实验见论文 Figure 7(b)。*

## 3. Outcome supervision 与 process supervision [21:10](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1270s)

《[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)》把比较转向更难的 **MATH** 题。**Outcome reward model（ORM）** 对完成的解答给分；**process reward model（PRM）** 看问题和前文，逐步判断下一步是否可接受。ORM 的最终答案标签较容易取得，却可能奖励“中间推导错误、最后碰巧答对”的解答；PRM 给 credit assignment（贡献归因）提供更局部的信号，但需要大量逐步反馈。[21:57](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1317s) [23:08](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1388s) [25:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1502s)

论文收集的 **PRM800K** 是约 80 万条**人类给出的 step-level feedback 标签**，覆盖约 12,000 道数学题，不是 80 万道独立题目。讲座用分数题说明：前面列式可以正确，最后一次算术失误仍应标为错误。为提高标注效率，研究者用当前 PRM 找出 **convincing wrong-answer solutions**：PRM 评分很高、但最终答案错误的解答，再优先交给人标注，因为其中必有当前 PRM 没发现的错误步骤。论文报告这种 active learning 在其代理实验中带来约 **2.6 倍**的数据效率提升。[26:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1560s) [27:08](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1628s) [28:01](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1681s)

![课程幻灯片：PRM800K 的 active learning 与 step-level labels](assets/03-robust-verification/prm800k-active-learning.png)

*读图：幻灯片把 `convincing wrong` 定义为“当前 PRM 高分、最终答案错误”的生成解答，每一步由人工标为 positive／negative／neutral。截图取自[原视频 27:20](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1640s)。课堂在前一分钟口头说成“最终答案正确但中间步骤错误”，与幻灯片及论文定义相反；此处以论文 §2.4 为准。*

本实验中，ORM 用完成时的分数，PRM 可将各步骤的正确概率**相乘**形成整条解答的分数，再从候选中选择；这是一种具体聚合方式，不是 PRM 的唯一定义。论文在 MATH 的 500 题代表性测试子集上报告 process supervision 优于 outcome supervision，并公开 PRM800K；为扩大 reward-model 训练题量，研究者把原 MATH test split 中 4,500 题加入训练，仅将随机抽取的 500 题保留评测。因此论文的 `78.2%` 属于这 500 题和特定采样、选择设置，不是完整 MATH test set 上的一次输出正确率。[29:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1740s) [论文 Appendix C](https://arxiv.org/html/2305.20050#A3)

讲师还展示 PRM 在这些实验里优于 ORM 与 majority voting、对少数正确候选的辨别以及部分 distribution shift（分布偏移）的表现。但比较两种监督时必须说明预算口径：一条解答的 outcome label 与同一解答的多个 step labels 并非相同人工标注工作量。课堂问答亦指出，跳步、看似合理却无贡献的步骤、PRM 过高评分及后续用 reward 训练 generator 时的 **reward hacking** 都不能靠“逐步打分”自动消除；最终结果与过程信号可以互补。[29:25](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1765s) [31:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1862s) [32:10](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1930s) [34:04](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2044s)

## 4. Math-Shepherd：不用人工逐步标注，代价是什么？[37:30](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2250s)

《[Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)》从某个中间步骤继续做多次 **rollout（续写采样）**，并检查最终答案。它用“从这里走到正确终点的可能性”做代理标签，而不是请人直接判断该步推理是否正确：[39:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2342s)

```mermaid
flowchart LR
    P[当前问题与已写步骤] --> R[采样 N 条完整后续]
    R --> A[与已知最终答案核对]
    A --> H[hard estimate：至少一条正确则为 1，否则为 0]
    A --> S[soft estimate：正确后续占 N 条的比例]
```

![课程幻灯片：Math-Shepherd 的 hard 与 soft automatic annotation](assets/03-robust-verification/math-shepherd-automatic-annotation.png)

*读图：hard estimate 只问 `N` 条 rollout 中是否至少一条到达正确答案；soft estimate 使用成功频率。截图取自[原视频 39:52](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2392s)。图中的 `N` 是一般定义；论文构造训练标签时实际使用 LLemma-7B 作 completer、`N=8`。*

若三条续写中两条到达正确终点，hard 标签为 1，soft 标签为 `2/3`。讲座说实验选择较简便的 hard estimate；论文实际以 **8 条 rollout** 自动标注每个步骤。要留意：**它估计的是当前前缀在指定 completer 与采样预算下的可解性，不直接证明当前步骤逻辑正确**。困难或罕见但有效的思路可能在有限 `N` 下全部采样失败，因而被误标；错误步骤也可能因后来“碰巧答对”而获得正标签。它节省了人工逐步标注，却仍依赖可核查的最终答案、足够的 rollout 和训练 PRM 的计算。[40:03](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2403s) [41:22](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2482s) [42:30](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2550s) [论文 §3.3](https://arxiv.org/html/2312.08935v3#S3.SS3)

候选解答的整体验证分数也与上一节不同：Math-Shepherd 取一条解答所有 step scores 的**最小值**，再选择 best-of-`N`；Lightman et al. 的讲座设置则把 step probabilities 相乘。Step aggregation 本身是系统设计选择，比较 PRM 结果时不能只看“是否逐步评分”，还要核对如何把步骤分数合成 solution score。[论文 §3.2](https://arxiv.org/html/2312.08935v3#S3.SS2)

训练好的 PRM 有两种用途：**推理时**从多条生成解答中挑最高分；**训练时**作为 reward，借助 proximal policy optimization（PPO）更新 generator。论文在 Mistral-7B 的 GSM8K／MATH 实验报告：逐步 PPO 后分别由 `77.9% → 84.1%`、`28.6% → 33.0%`；再加 Math-Shepherd verification 分别达到 `89.1%`、`43.5%`。前一组是训练改进，后一组还包含推理时的选答，不能称作相同预算下的 pass@1 提升。[43:12](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2592s) [46:08](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2768s) [论文摘要](https://arxiv.org/abs/2312.08935)

课堂讨论提出 calculator、SymPy、规则 rubric 等可执行或外部反馈来补充模型评分；这是设计方向，不是上述论文已证明能普遍修复 PRM 错误的实验结论。讲师也说，跨 ORM／PRM 比较固定总推理 compute 不容易，标签和训练数据来源同样影响公平性。[48:15](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2895s) [49:15](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2955s)

## 5. Weaver：组合不完美的 verifiers [51:25](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3085s)

《[Shrinking the Generation-Verification Gap with Weak Verifiers](https://arxiv.org/abs/2506.18203)》提出 **Weaver**：不要求某一个 verifier 无误，而组合 reward models 与 **LLM-as-judge（用语言模型评审）** 等不同信号。“Weak”指各自不完美，并非故意选最差的模型。直接平均可能让较差或高度同质的 verifier 拖累系统；有标签时可学习不同成员的权重，而 Weaver 借助 **weak supervision（弱监督）**在标签有限的情况下估计可靠性。[52:04](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3124s) [53:04](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3184s) [54:04](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3244s)

```mermaid
flowchart LR
    Q[问题] --> K[k 条候选]
    K --> M[m 个 verifiers 分别评分]
    M --> N[标准化分数并筛去低质量成员]
    N --> W[估计可靠性并组合分数]
    W --> O[选择候选]
```

![课程幻灯片：Weaver 的 score、weight、select 三阶段](assets/03-robust-verification/weaver-score-weight-select.png)

*读图：`Score` 统一不同 verifier 的输出尺度并滤掉低质量成员；`Weight` 用 weak supervision 从极少量标签估计可靠性；`Select` 合成加权分数并挑选最高置信候选。截图取自[原视频 56:15](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3375s)。这张图描述的是 inference-time selection，没有更新 generator 参数。*

讲座称之为“score, weight, select”。每个 verifier 的输出尺度可能不同，低质量成员需筛选；弱监督利用各评分器对大量候选的同意／分歧模式估计权重，核心依赖它们的错误不完全同步、在模型假设下可作**条件独立**处理。如果所有评分器复制同一种偏见，简单地加人或加模型不会凭空创造真值。讲座的对比包括 naive ensemble、majority voting 与 oracle 上界；所报告的提升是**选中答案的准确率**，不是只有 coverage 增加。[55:10](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3310s) [57:40](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3460s) [59:02](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3542s) [01:03:05](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3785s)

更多 verifiers、更多候选、更大的 generator／judge 都会增加计算开销。论文还把 ensemble 信号 **distill（蒸馏）**到约 400M 参数的 cross-encoder，以便推理时只跑较小模型；这仍是该任务分布上的成本—准确率取舍，并不等于在所有任务上保持原集成的能力。[01:00:08](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3608s) [01:04:10](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3850s) [论文摘要](https://arxiv.org/abs/2506.18203)

**版本提醒**：课堂是 2025 年 9 月 29 日，而 arXiv 页面现显示 Weaver 论文后续修订到 2026 年。讲座中提到的平均准确率与当前摘要中的 `87.7%` 不宜强行当成同一实验数字；此处以讲座解释机制，论文链接供核查具体版本、模型组合和基准测试。[01:03:35](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3815s) [arXiv 版本记录](https://arxiv.org/abs/2506.18203)

## 6. 适用边界与收束 [01:06:20](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3980s)

- **可检查的目标决定自动验证上限**：GSM8K／MATH 有标准答案，代码可运行 unit tests；开放式研究、写作与真实环境任务往往需要多种证据、人类反馈或外部工具，不能照搬“核对最终数值”的标签机制。课堂问答把生成代码测试作为另一条 verifier 路径，不是这四篇数学实验的已实现结果。[01:08:15](https://www.youtube.com/watch?v=p7TdPUcPoik&t=4095s)
- **选择与学习是两个循环**：best-of-`k`／Weaver 不改 generator 参数；用 PRM 进行 PPO 则会改变 generator。后者可能追逐 verifier 的漏洞，因此更需要独立的外部评估。[43:12](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2592s) [01:07:03](https://www.youtube.com/watch?v=p7TdPUcPoik&t=4023s)
- **不要只报 pass@k**：同时说明候选数、最终选择规则、被选答案准确率、训练标签量与总推理预算。追求更强 pass@1 可以省推理成本，但讲师提醒过度收缩生成分布也可能损失多样性；这是研究判断而非已被本讲证明的必然规律。[01:10:28](https://www.youtube.com/watch?v=p7TdPUcPoik&t=4228s)

## 复习时回答这 5 个问题

1. 为什么增加 `k` 可能提高 pass@k，却降低 selected-answer accuracy？
2. ORM 和 PRM 的标签粒度、人工成本与误判方式分别是什么？
3. Math-Shepherd 的 hard／soft rollout 标签衡量“步骤正确”还是“从此处可解”？举一例说明二者差异。
4. Weaver 为什么先筛选和校准 verifiers，再估计权重？相关错误会破坏什么假设？
5. 把 verifier 用于 best-of-`k` 与用于 PPO 有何不同风险？要怎样公平报告计算和最终质量？

## 资料与核对

- [原始视频：Stanford Online, Part 3 — Robust Verification](https://www.youtube.com/watch?v=p7TdPUcPoik)：时间戳依据视频发布者提供的英文 CC（在 YouTube 视频中可选择 English - CC）；本文所用课程幻灯片截图已回看原视频对应画面。
- [Stanford CS329A 课程官网：2025 年秋季，9 月 29 日第 3 讲及指定阅读](https://cs329a.stanford.edu/)。
- [Cobbe et al., *Training Verifiers to Solve Math Word Problems*](https://arxiv.org/abs/2110.14168)。
- [Lightman et al., *Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050)。
- [Wang et al., *Math-Shepherd*](https://arxiv.org/abs/2312.08935)。
- [Saad-Falcon et al., *Shrinking the Generation-Verification Gap with Weak Verifiers*](https://arxiv.org/abs/2506.18203)：论文当前修订版与课堂讲述可能并非同一版本。
- [onehr, *CS329A 学习伴侣·第 3 讲：鲁棒验证*](https://github.com/onehr/cs329-notes/blob/main/companion/L03-robust-verification-zh.md)：二手笔记，用于发现 Figure 7 的 top-ranked voting、`convincing wrong` 口径冲突、MATH500 与 step aggregation 等核查线索；本文相关内容已回查原视频和论文。
- [My Learning Wiki, *Stanford CS329A Part 3 — Robust Verification*](https://weihaoqu.github.io/learnAIDoc/wiki/cs329a-part-03-robust-verification/)：二手 teaching companion，用于检查 failure modes 与 verifier-error correlation 是否表达完整；其教学归纳未作为论文结论引用。
