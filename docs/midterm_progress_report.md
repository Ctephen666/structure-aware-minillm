# 中期实验进度报告

## 一、课题概述

本课题围绕“结构感知语言模型”展开，目标是在常规 Decoder-only Transformer 的基础上，引入文本结构信息，使模型在 JSON、Markdown、代码片段以及中文问答等任务中具备更好的格式理解、结构保持和生成稳定性。

当前项目并不是直接调用现成大模型，而是在本地实现并训练一个约 9,268 万参数的结构感知 Transformer。模型训练流程包括三个阶段：

1. 通用中文与结构混合语料预训练；
2. 知识型语料继续预训练；
3. 中文问答数据监督微调。

截至中期阶段，项目已经完成模型主体实现、结构标注器实现、数据构建脚本、训练脚本、解码脚本、预训练 checkpoint、知识继续训练 checkpoint 以及一轮 SFT checkpoint，整体进度已达到中期要求的 30% 以上。

## 二、已完成工作

### 2.1 项目代码结构

当前仓库已经形成较完整的实验工程结构：

| 目录 | 作用 |
| --- | --- |
| `model/` | Transformer 与结构感知 Transformer 模型实现 |
| `parser/` | JSON / Markdown 结构状态标注器 |
| `train/` | 数据集封装、流式加载与训练入口 |
| `decode/` | checkpoint 加载与生成解码 |
| `configs/` | 预训练、继续训练、SFT 配置文件 |
| `scripts/` | 数据准备、训练启动、采样评估脚本 |
| `data/` | 预训练、继续训练、SFT 数据 |
| `checkpoints/` | 已保存模型权重 |
| `logs/` | 训练日志与错误日志 |

### 2.2 结构感知模型实现

核心模型位于 `model/struct_transformer.py`。当前模型是在 GPT 风格 Decoder-only Transformer 基础上加入结构信息，输入表示由以下几部分相加：

```text
token embedding
+ position embedding
+ depth embedding
+ state embedding
```

其中：

- `depth_embedding` 表示当前 token 所处的结构嵌套深度；
- `state_embedding` 表示当前 token 所处的结构状态；
- `lm_head` 预测下一个 token；
- `depth_head` 辅助预测下一个 token 的结构深度；
- `state_head` 辅助预测下一个 token 的结构状态。

训练损失为：

```text
total_loss = lm_loss + lambda_depth * depth_loss + lambda_state * state_loss
```

这一设计使模型不仅学习语言建模目标，也同时学习结构边界和结构状态。

### 2.3 结构状态标注器实现

结构标注器位于 `parser/structure_annotator.py` 和 `parser/structure_states.py`。

当前已实现的结构状态包括：

| 状态 | 含义 |
| --- | --- |
| `PLAIN` | 普通文本 |
| `JSON_OBJECT` | JSON 对象内部 |
| `JSON_ARRAY` | JSON 数组内部 |
| `JSON_STRING` | JSON 字符串内部 |
| `JSON_ESCAPE` | JSON 转义状态 |
| `MD_TEXT` | Markdown 文本 |
| `MD_OUTER_FENCE` | Markdown 外层代码块 |
| `MD_INNER_FENCE` | Markdown 嵌套代码块 |
| `MD_INLINE_CODE` | Markdown 行内代码 |

最大结构深度设置为 `32`，状态总数为 `9`。标注器能够基于 token 序列追踪 JSON 的 `{}`、`[]`、字符串转义，以及 Markdown 代码围栏等结构。

### 2.4 数据集与训练管线

当前已经实现两类训练数据封装：

1. 普通语言模型数据集；
2. 结构感知语言模型数据集。

结构感知数据集位于 `train/structure_dataset.py`，支持：

- 普通文本预训练格式：`{"text": "..."}`
- SFT 问答格式：`{"prompt": "...", "answer": "..."}`
- 对 SFT prompt 部分做 loss mask，只在 answer 部分计算语言模型损失；
- 对结构辅助目标同步使用 `IGNORE_INDEX=-100` 屏蔽 prompt 部分；
- 支持大文件流式训练；
- 支持多源数据加权流式混合。

训练入口位于 `train/train_struct.py`，已经支持：

- YAML 配置加载；
- checkpoint 初始化与恢复；
- AdamW 优化器；
- cosine learning rate schedule；
- warmup；
- gradient accumulation；
- AMP 半精度训练；
- 定期验证；
- 保存 best checkpoint；
- 最大训练时长限制。

## 三、实际训练过程

### 3.1 第一阶段：结构感知预训练

预训练配置文件为 `configs/struct_pretrain_80m.yaml`。

主要配置如下：

| 项目 | 配置 |
| --- | --- |
| tokenizer | `uer/gpt2-chinese-cluecorpussmall` |
| vocab size | 21128 |
| block size | 512 |
| layer | 16 |
| head | 8 |
| embedding dim | 640 |
| dropout | 0.1 |
| batch size | 2 |
| gradient accumulation | 8 |
| effective batch | 16 sequences |
| learning rate | `3.0e-4` |
| min lr | `3.0e-5` |
| max steps | 100000 |
| AMP | float16 |

训练数据：

| 文件 | 大小 |
| --- | ---: |
| `data/pretrain/pretrain_80m_train.jsonl` | 约 1.59 GB |
| `data/pretrain/pretrain_80m_valid.jsonl` | 约 16.15 MB |

预训练结果：

| checkpoint | step | best val loss |
| --- | ---: | ---: |
| `checkpoints/struct_pretrain_80m.pt` | 100000 | 2.7782 |

该阶段完成了模型的基础语言建模能力训练，并得到后续继续训练和 SFT 的初始化权重。

### 3.2 第二阶段：知识语料继续预训练

继续训练配置文件为 `configs/struct_continue_80m_knowledge.yaml`。

主要配置如下：

| 项目 | 配置 |
| --- | --- |
| init checkpoint | `checkpoints/struct_pretrain_80m.pt` |
| learning rate | `3.0e-5` |
| min lr | `1.0e-5` |
| max steps | 10000 |
| batch size | 2 |
| gradient accumulation | 8 |
| AMP | float16 |

训练数据：

| 文件 | 大小 |
| --- | ---: |
| `data/pretrain/continue_knowledge_train.jsonl` | 约 1.06 GB |
| `data/pretrain/continue_knowledge_valid.jsonl` | 约 10.87 MB |

知识继续训练混合数据主要由以下部分构成：

- 教育/知识类语料；
- wiki 知识语料；
- 少量结构化样本；
- JSON、YAML、Python、Markdown 等结构样本。

继续训练结果：

| checkpoint | step | best val loss |
| --- | ---: | ---: |
| `checkpoints/struct_pretrain_80m_knowledge_continued.pt` | 10000 | 2.3462 |

与第一阶段相比，验证损失进一步下降，说明继续训练对知识型文本建模有正向作用。

### 3.3 第三阶段：中文问答 SFT

SFT 配置文件为 `configs/struct_sft_80m_qa.yaml`。

主要配置如下：

| 项目 | 配置 |
| --- | --- |
| init checkpoint | `checkpoints/struct_pretrain_80m.pt` |
| learning rate | `1.0e-5` |
| min lr | `3.0e-6` |
| max steps | 30000 |
| max train seconds | 18000 秒，约 5 小时 |
| batch size | 2 |
| gradient accumulation | 8 |
| lambda depth | 0.02 |
| lambda state | 0.05 |

当前实际保存的 SFT checkpoint 信息如下：

| checkpoint | step | best val loss |
| --- | ---: | ---: |
| `checkpoints/struct_sft_80m_qa.pt` | 21575 | 3.3564 |

当前 SFT 数据如下：

| 文件 | 样本数 |
| --- | ---: |
| `data/sft/sft_train.jsonl` | 31024 |
| `data/sft/sft_valid.jsonl` | 1632 |

后续又准备了更大的 8 小时实验数据：

| 文件 | 样本数 |
| --- | ---: |
| `data/sft/sft_train_8h_fixed.jsonl` | 124530 |
| `data/sft/sft_valid_8h_fixed.jsonl` | 6554 |

日志中还记录了一次配置为 8 小时的 SFT 尝试。该实验从 `checkpoints/struct_pretrain_80m.pt` 的 step 100000 checkpoint 初始化，日志记录到约 step 1603。验证损失变化如下：

| step | val loss |
| ---: | ---: |
| 1 | 4.7553 |
| 500 | 3.8527 |
| 1000 | 3.7006 |
| 1500 | 3.6291 |

这说明 SFT 早期训练过程中验证损失有明显下降。不过从后续生成效果看，仅降低验证损失还不足以保证问答质量稳定。

## 四、阶段性实验结果

### 4.1 训练指标

从 checkpoint 记录看，当前三个主要阶段均已产出可加载的模型：

| 阶段 | checkpoint | step | best val loss |
| --- | --- | ---: | ---: |
| 结构感知预训练 | `struct_pretrain_80m.pt` | 100000 | 2.7782 |
| 知识继续训练 | `struct_pretrain_80m_knowledge_continued.pt` | 10000 | 2.3462 |
| QA SFT | `struct_sft_80m_qa.pt` | 21575 | 3.3564 |

其中继续预训练阶段验证损失最低，说明知识语料继续训练对语言建模能力有提升。SFT 阶段由于数据格式、回答风格和监督目标发生变化，loss 数值不能直接与预训练阶段横向比较。

### 4.2 生成效果观察

目前使用 `scripts/sample_sft_80m.py` 对 SFT checkpoint 进行采样。测试问题包括：

- 水的沸点是多少？
- 太阳主要由什么组成？
- 光合作用是什么？
- 牛顿第一定律说明什么？
- 请用 JSON 表示一个用户信息。
- 请写一个读取 JSONL 的 Python 函数。
- 如果你不知道答案，应该怎么回答？

观察到的典型问题包括：

1. 部分回答只复述问题片段，例如“水的沸点是多少？”生成结果只保留“沸点是多少？”；
2. 出现明显重复，例如“太阳主要由太阳和太阳组成”反复出现；
3. 概念解释类问题会生成循环句式，例如“光合作用是指光合作用……”；
4. 代码生成任务可能输出为空；
5. 不知道答案类问题可能退化成无关长文本。

这些现象说明模型已经学习到一定问答形式，但还没有稳定掌握指令遵循、事实回答和停止生成能力。

### 4.3 解码策略调整

针对重复生成问题，已经对 `scripts/sample_sft_80m.py` 做了初步修改：

```text
--repetition-penalty 默认值从 1.0 调整为 1.2
--no-repeat-ngram-size 默认值从 0 调整为 4
```

同时加入参数校验：

- `repetition_penalty` 必须大于等于 1.0；
- `no_repeat_ngram_size` 必须大于等于 0。

这属于解码层面的快速缓解措施，后续仍需要从数据质量、训练策略和模型初始化方面继续优化。

## 五、当前问题分析

### 5.1 SFT 后生成质量不稳定

当前 SFT checkpoint 的生成结果存在重复、空回答和无关回答。这可能由以下因素共同导致：

1. 模型规模较小，约 92.68M 参数，事实记忆和指令泛化能力有限；
2. SFT 数据来源较杂，公开中文问答数据中存在长回答、噪声回答和风格不一致问题；
3. 当前 SFT 初始化使用的是 `struct_pretrain_80m.pt`，尚未充分比较从知识继续训练 checkpoint 初始化的效果；
4. 解码阶段原先重复惩罚关闭，容易放大模型的局部循环倾向；
5. 验证 loss 下降不等价于人工问题上的回答质量提升，需要补充专门的生成评测集。

### 5.2 训练过程中的工程问题

日志中记录了一次数据标准化失败：

```text
PermissionError: Permission denied: data/sft/sft_train_fixed.jsonl
```

这说明部分数据文件可能在训练、编辑器或其他进程中被占用。后续需要规范数据生成流程，避免在文件被占用时覆盖写入。

日志中还出现 Hugging Face 在线请求失败：

```text
ProxyError / WinError 10061
```

这说明训练环境存在网络访问不稳定问题。后续应尽量使用本地缓存，并在配置中启用 `local_files_only` 或提前下载 tokenizer 和数据集，减少训练时对网络的依赖。

### 5.3 评测体系还不完整

当前已有训练 loss 和人工观察样例，但还缺少系统性评测，例如：

- JSON 结构合法率；
- Markdown 代码块闭合率；
- 中文问答准确率；
- 重复率；
- 平均输出长度；
- 与 baseline Transformer 的对比；
- 去掉 depth/state 辅助目标的消融实验。

这些内容将作为下一阶段重点补充。

## 六、进度评估

根据当前完成情况，本课题已经达到并超过 30% 中期进度要求。

| 模块 | 完成情况 | 进度判断 |
| --- | --- | --- |
| 文献与方案设计 | 已明确结构感知 Transformer 思路 | 已完成基础部分 |
| 模型实现 | 已完成结构 embedding 与辅助预测头 | 已完成 |
| 结构标注 | 已完成 JSON / Markdown 状态标注 | 已完成 |
| 预训练数据 | 已构建 GB 级训练数据 | 已完成阶段性版本 |
| 预训练实验 | 已训练到 100000 step | 已完成阶段性版本 |
| 知识继续训练 | 已训练到 10000 step | 已完成阶段性版本 |
| SFT 实验 | 已训练并保存 checkpoint | 已完成初步版本 |
| 生成测试 | 已完成固定 prompt 人工观察 | 初步完成 |
| 系统评测 | 仍需补充自动指标 | 未完成 |
| baseline / 消融 | 仍需补充 | 未完成 |
| 论文式总结 | 本报告为阶段性整理 | 进行中 |

综合判断：当前工作量和实验深度可以支撑中期报告，合理进度约为 35% 到 40%。后续重点应从“能训练、能生成”转向“可评测、可对比、可解释地优化”。

## 七、下一阶段计划

### 7.1 数据与训练优化

1. 固定 SFT 模板，统一使用：

```text
问题：{prompt}
回答：{answer}
```

2. 对 SFT 数据做进一步清洗，减少超长回答、低质量问答和重复样本；
3. 对比两种初始化方式：

```text
struct_pretrain_80m.pt
struct_pretrain_80m_knowledge_continued.pt
```

4. 针对 8 小时数据集重新进行稳定训练，记录完整 loss 曲线和 checkpoint；
5. 保存中间 checkpoint，用固定测试集比较不同 step 的生成质量，避免只看最终 checkpoint。

### 7.2 解码与生成优化

1. 系统比较不同解码参数：

```text
temperature: 0.3 / 0.5 / 0.7
top_p: 0.8 / 0.85 / 0.9
repetition_penalty: 1.1 / 1.2 / 1.3
no_repeat_ngram_size: 3 / 4 / 5
```

2. 增加 EOS 停止策略和最大长度控制；
3. 对常见任务分别设置测试 prompt，包括事实问答、JSON 生成、Python 函数生成、未知问题拒答等。

### 7.3 自动评测与对比实验

后续计划补充以下评测：

| 评测项 | 目的 |
| --- | --- |
| validation loss | 衡量训练过程 |
| repetition rate | 衡量重复退化 |
| JSON parse success rate | 衡量结构生成能力 |
| Markdown fence closure rate | 衡量 Markdown 结构保持 |
| exact / keyword match | 衡量基础知识问答 |
| baseline vs structure-aware | 验证结构信息是否有效 |
| no depth/state ablation | 验证辅助结构目标贡献 |

### 7.4 报告与论文整理

下一阶段将把实验结果整理为：

1. 方法结构图；
2. 训练流程图；
3. 数据统计表；
4. loss 曲线；
5. 生成案例对比表；
6. 失败案例分析；
7. baseline 与消融实验结果。

## 八、阶段性结论

本阶段已经完成结构感知语言模型的主要工程实现，并完成预训练、知识继续训练和 SFT 的初步实验。checkpoint 记录显示，预训练模型已训练到 100000 step，知识继续训练模型已训练到 10000 step，SFT 模型已保存到 21575 step。实验表明模型能够完成基本训练闭环，并在验证集上取得可记录的 loss 下降。

同时，当前生成结果也暴露出小模型 SFT 中常见的问题，包括重复生成、指令遵循不足、事实回答不稳定和代码任务空输出。针对这些问题，已经开始从解码重复惩罚入手进行修正。后续将重点加强数据清洗、SFT 初始化对比、自动评测和 baseline / 消融实验，使实验结果从“阶段性可运行”推进到“可量化比较和可论文呈现”。

因此，当前项目进展可以满足中期 30% 的要求，并具备继续推进到完整实验报告的基础。
