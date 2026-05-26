# CS336 Assignment 5：第 8 节 GRPO 实验任务要求整理

> 来源：`cs336_assignment5_alignment_zh.pdf` 第 8 节“GRPO 实验”。本文档只整理第 8 节实验要求；第 9 节“排行榜”不包含在内。

---

## 0. 第 8 节实验的共同前提

第 8 节的目标是：在已经实现完整 `GRPO train loop` 的基础上，系统比较不同超参数和算法修改对 MATH 推理强化学习效果的影响。

所有第 8 节实验默认需要 **2 个 GPU**：

- 一个 GPU 用于 `vLLM instance`，负责 rollout / evaluation generation；
- 一个 GPU 用于 `policy model`，负责训练。

如果某个 run 在 **200 个 GRPO steps** 之前已经表现出明显差异，例如发散、明显劣于其他配置，允许提前停止，把算力留给后续实验。文档中标注的 H100 小时数只是粗略估计。

---

## 1. 默认起点与通用训练设置

第 8 节的第一个实验要求“从上面建议的超参数开始”。这里的默认起点来自前一节 GRPO train loop：

```python
n_grpo_steps: int = 200
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 1      # On-policy
train_batch_size: int = 256            # On-policy
gradient_accumulation_steps: int = 128 # microbatch size = 2, fits on H100
gpu_memory_utilization: float = 0.85
loss_type = "reinforce_with_baseline"
use_std_normalization: bool = True

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),
)
```

需要满足的基本约束：

```python
assert train_batch_size % gradient_accumulation_steps == 0
micro_train_batch_size = train_batch_size // gradient_accumulation_steps

assert rollout_batch_size % group_size == 0
n_prompts_per_rollout_batch = rollout_batch_size // group_size

assert train_batch_size >= group_size
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
```

通用提示与记录要求：

- 使用 `r1_zero` prompt，除非实验明确要求改成 `question_only` prompt。
- 使用 vLLM 时，generation 应在第二个 `</answer>` 处停止，与前面实验一致。
- 建议使用 `typer` 做命令行参数解析。
- 使用 gradient clipping，clip value 为 `1.0`。
- 定期记录 validation rewards，例如每 5 或 10 steps 记录一次。
- 为了比较超参数，validation 至少应在 1024 个 examples 上评估，因为 CoT / RL evaluation 可能较 noisy。
- 按照本作业的 loss 实现，`GRPO-Clip` 只应在 off-policy 设置中使用，因为它需要 `old_log_probs`。
- 在 off-policy 的多个 epochs / gradient updates per rollout batch 设置中，不要为每个 epoch 重新计算 `old_log_probs`；应在每次 rollout batch generation phase 后计算一次，然后复用。
- 不应对 `old_log_probs` 求导，建议用 `torch.inference_mode()` 计算。
- 每次 optimizer update 时，建议记录：
  - loss；
  - gradient norm；
  - token entropy；
  - off-policy 时的 clip fraction；
  - train rewards，包括 total、format、answer；
  - 其他有助于调试的指标。

---

## 2. 实验总览清单

| 顺序 | 问题名 | 分值 | 估计算力 | 核心比较对象 |
|---:|---|---:|---:|---|
| 1 | `grpo_learning_rate` | 2 分 | 6 H100 小时 | 多个 learning rates |
| 2 | `grpo_baselines` | 2 分 | 2 H100 小时 | `reinforce_with_baseline` vs `no_baseline` |
| 3 | `think_about_length_normalization` | 1 分 | 无需跑实验 | `masked_mean` vs `masked_normalize` 的理论比较 |
| 4 | `grpo_length_normalization` | 2 分 | 2 H100 小时 | 端到端比较 `masked_mean` vs `masked_normalize` |
| 5 | `grpo_group_standard_deviation` | 2 分 | 2 H100 小时 | `use_std_normalization=True` vs `False` |
| 6 | `grpo_off_policy` | 未单独给分 | 实现要求 | 实现 off-policy GRPO training |
| 7 | `grpo_off_policy_sweep` | 4 分 | 12 H100 小时 | `epochs_per_rollout_batch` 与 `train_batch_size` sweep |
| 8 | `grpo_off_policy_clip_ablation` | 2 分 | 2 H100 小时 | GRPO-Clip vs GRPO-No-Clip |
| 9 | `grpo_prompt_ablation` | 2 分 | 2 H100 小时 | R1-Zero prompt vs question-only prompt |

---

## 3. `grpo_learning_rate`：调节 learning rate

### 目标

从默认 GRPO 超参数出发，对不同 learning rates 做 sweep，观察 learning rate 对 validation answer reward / validation accuracy 的影响。

### 必做设置

- 使用默认起点超参数。
- 只改变 learning rate，其他设置尽量固定。
- 如果某个 optimizer 发散，需要明确注明 divergence。
- 后续实验可以使用本实验中表现最好的 learning rate。

### 交付物

- 多个 learning rates 对应的 validation reward curves。
- 一个在 MATH 上达到至少 **25% validation accuracy** 的模型。
- 2 句简短讨论，说明其他 logged metrics 上观察到的趋势，例如 entropy、response length、gradient norm、train rewards 等。

---

## 4. `grpo_baselines`：Baselining 的影响

### 目标

在 on-policy 设置下比较是否使用 group-normalized reward baseline 的效果。

### 必做设置

使用上一节选出的最佳 learning rate，并比较以下两种 `loss_type`：

- `no_baseline`
- `reinforce_with_baseline`

默认 `use_std_normalization=True`。

### 交付物

- 每种 loss type 对应的 validation reward curves。
- 2 句简短讨论，说明其他 logged metrics 上观察到的趋势。

### 后续约定

接下来的几个实验使用本实验中表现最好的 `loss_type`。

---

## 5. `think_about_length_normalization`：思考长度归一化

### 目标

先不运行实验，只从机制上比较两种把 per-token losses 聚合到 response-level loss 的方式。

比较对象：

1. `masked_mean`：对每个 sequence 中未被 mask 的 response tokens 求平均。
2. `masked_normalize`：对每个 sequence 中未被 mask 的 response tokens 求和，然后除以固定常数，例如 `max_gen_len`。

### 需要回答的问题

- 两种方法各自的优点是什么？
- 两种方法各自的缺点是什么？
- 是否存在某些具体设置或例子，使其中一种方法看起来更合理？

### 交付物

- 一段比较性讨论，不需要跑实验。

---

## 6. `grpo_length_normalization`：长度归一化的影响

### 目标

通过端到端 GRPO training run，实证比较 `masked_mean` 与 `masked_normalize` 两种 length normalization 方法。

### 必做设置

- 继承前面实验中选出的较优 learning rate 和较优 `loss_type`。
- 分别使用 `masked_mean` 与 `masked_normalize` 聚合 response token 上的 per-token losses。
- 注意观察稳定性相关指标，尤其是 gradient norm。

### 交付物

- `masked_mean` 与 `masked_normalize` 两种设置对应的 validation answer reward curves。
- 对发现进行评论，包括其他 logged metrics 上是否有明显趋势，例如 gradient norm、entropy、response length 等。

### 后续约定

后续实验固定使用本实验中表现更好的 length normalization 方法。

---

## 7. `grpo_group_standard_deviation`：Group standard deviation 归一化的影响

### 背景

标准 GRPO 会在组内计算 advantage 时除以 group standard deviation；但 Dr. GRPO 指出，这可能给训练引入不希望的偏差：组内 reward 方差较低的问题，例如太容易或太难的问题，可能获得过高权重。因此需要比较是否使用标准差归一化。

### 目标

比较 `compute_group_normalized_rewards` 中是否除以 group standard deviation 的效果。

### 必做设置

比较：

- `use_std_normalization == True`
- `use_std_normalization == False`

继承前面实验中选出的较优 learning rate、较优 `loss_type` 和较优 length normalization 方法。

### 交付物

- 两种设置对应的 validation answer reward curves。
- 对发现进行评论，包括其他 logged metrics 上是否有明显趋势。
- 建议特别关注稳定性相关指标，例如 gradient norm。

### 后续约定

后续实验固定使用本实验中表现更好的 group normalization 方法。

---

## 8. `grpo_off_policy`：实现 off-policy GRPO training

### 目标

将前面 on-policy 的 GRPO 训练改成 off-policy 版本：每个 rollout batch 不只做一个 gradient step，而是执行多个 gradient steps，甚至多个 epochs。

### 必做实现

如果完整的 GRPO train loop 还不支持 off-policy，需要补充以下能力：

1. 对每个 rollout batch 执行多个 epochs 的 gradient steps。
2. 由以下参数共同控制 off-policy 程度：
   - `rollout_batch_size`
   - `epochs_per_rollout_batch`
   - `train_batch_size`
3. 编辑主训练循环：在每次 rollout batch generation phase 之后、gradient steps 的 inner loop 之前，从 policy 获取 response logprobs。
4. 上一步得到的 response logprobs 就是 `old_log_probs`。
5. 建议用 `torch.inference_mode()` 计算 `old_log_probs`。
6. 使用 `"GRPO-Clip"` loss type。

### 交付物

- 实现 off-policy GRPO training。

---

## 9. `grpo_off_policy_sweep`：Off-policy GRPO 超参数 sweep

### 目标

系统比较 off-policy 程度对训练效果和效率的影响。

### 必做设置

- 固定 `rollout_batch_size = 256`。
- 选择一组 `epochs_per_rollout_batch` 和 `train_batch_size` 的范围做 sweep。
- 第一步：在有限 GRPO steps，即小于 50 steps 上做 broad sweep，用于了解 performance landscape。
- 第二步：在更多 GRPO steps，即 200 steps 上做更聚焦的 sweep。
- 提供简短 experiment log，说明为什么选择这些 sweep 范围。
- 与 on-policy run 比较，其中 on-policy 设置为：
  - `epochs_per_rollout_batch = 1`
  - `train_batch_size = 256`
- 需要改变 `gradient_accumulation_steps`，以保持 memory usage 恒定。

### 交付物

- 与 on-policy run 的比较图：
  - 随 validation steps 变化的 plots；
  - 随 wall-clock time 变化的 plots。
- validation answer reward curves。
- 对发现进行评论，包括其他有明显趋势的 metrics，例如 entropy 和 response length。
- 将训练过程中模型 response 的 entropy 与 EI 实验中的观察进行比较。
- 一份简短 experiment log，说明 sweep 范围选择理由。

---

## 10. `grpo_off_policy_clip_ablation`：Off-policy GRPO-Clip 消融

### 背景

GRPO-Clip 中 clipping 的目的，是在单个 rollout batch 上执行多个 gradient steps 时，防止当前 policy 离旧 policy 太远。本实验要测试 clipping 在 off-policy 设置中是否真的必要。

### 目标

在 off-policy 设置中消融 clipping，把 GRPO-Clip 与不加 clipping 的版本进行比较。

### 必做实现

实现 unclipped per-token loss，作为新的 loss type：

```text
GRPO-No-Clip
```

其 per-token loss 对应：

```text
- [πθ(o_t | q, o_<t) / πθ_old(o_t | q, o_<t)] * A_t
```

### 必做设置

- 使用上一题中表现最好的 off-policy hyperparameters。
- 运行 unclipped 版本，即 `GRPO-No-Clip`。
- 与对应的 `GRPO-Clip` run 比较。

### 交付物

- `GRPO-No-Clip` 的 validation answer reward curves。
- 与 `GRPO-Clip` run 的对比。
- 对发现进行评论，包括其他 logged metrics 上是否有明显趋势，例如 entropy、response length 和 gradient norm。

---

## 11. `grpo_prompt_ablation`：Prompt ablation

### 背景

本实验研究 prompt 对 RL 训练效果的影响。文档指出，RL 使用的 prompt 会显著影响模型性能，这与模型的预训练方式有关。

### 目标

比较 R1-Zero prompt 与 question-only prompt 对 GRPO 训练和验证效果的影响。

### 必做设置

原设置使用：

```text
cs336_alignment/prompts/r1_zero.prompt
```

本实验改用 question-only prompt：

```text
cs336_alignment/prompts/question_only.prompt
```

该 prompt 内容为：

```text
{question}
```

训练和验证时都使用 question-only prompt。

reward function 也要对应更换为：

```text
cs336_alignment/drgrpo_grader.py 中的 question_only_reward_fn
```

训练和验证都使用 `question_only_reward_fn`。

### 交付物

- R1-Zero prompt 与 question-only prompt 的 validation answer reward curves。
- 比较各项 metrics 的差异，包括有明显趋势的 metrics，例如 entropy、response length 和 gradient norm。
- 尝试解释观察到的现象。

---

## 12. 最终提交自查清单

完成第 8 节时，至少应当有以下材料：

- `grpo_learning_rate`
  - 多个 learning rates 的 validation reward curves。
  - 至少一个达到 25% MATH validation accuracy 的模型。
  - 2 句关于其他 metrics 趋势的讨论。
- `grpo_baselines`
  - `no_baseline` 与 `reinforce_with_baseline` 的 validation reward curves。
  - 2 句 metrics 趋势讨论。
- `think_about_length_normalization`
  - `masked_mean` 与 `masked_normalize` 的理论比较，不需要实验。
- `grpo_length_normalization`
  - `masked_mean` 与 `masked_normalize` 的端到端 validation answer reward curves。
  - 对趋势的评论，尤其关注 gradient norm。
- `grpo_group_standard_deviation`
  - `use_std_normalization=True` 与 `False` 的 validation answer reward curves。
  - 对趋势的评论，尤其关注 gradient norm。
- `grpo_off_policy`
  - 支持多个 epochs / gradient updates per rollout batch 的 off-policy GRPO training 实现。
  - 能正确计算并复用 `old_log_probs`。
  - 使用 `GRPO-Clip` loss type。
- `grpo_off_policy_sweep`
  - 固定 `rollout_batch_size=256` 的 broad sweep 与 focused sweep。
  - sweep 范围选择的 experiment log。
  - 与 on-policy run 的 step-based plots 和 wall-clock-time-based plots。
  - validation answer reward curves。
  - 对 entropy、response length 等 metrics 的评论。
  - 与 EI 实验中的 entropy 观察进行比较。
- `grpo_off_policy_clip_ablation`
  - 新 loss type `GRPO-No-Clip`。
  - 与最佳 off-policy GRPO-Clip run 的 validation answer reward curve 对比。
  - 对 entropy、response length、gradient norm 等趋势的评论。
- `grpo_prompt_ablation`
  - R1-Zero prompt 与 question-only prompt 的 validation answer reward curves。
  - 使用正确的 prompt 文件与 reward function。
  - 对 entropy、response length、gradient norm 等 metrics 的比较与解释。

---

## 13. 推荐实验顺序

为了避免重复浪费算力，建议严格按以下顺序推进：

1. 先确保 `grpo_train_loop` 能跑通，并看到 validation rewards 提升。
2. 跑 learning rate sweep，确定后续 learning rate。
3. 跑 baseline ablation，确定后续 `loss_type`。
4. 先写 `think_about_length_normalization` 的文字讨论。
5. 跑 length normalization ablation，确定后续 length normalization。
6. 跑 group standard deviation normalization ablation，确定后续 group normalization。
7. 实现 off-policy GRPO。
8. 跑 off-policy hyperparameter sweep，先 broad sweep，再 focused sweep。
9. 在最佳 off-policy 设置上做 clip ablation。
10. 最后做 prompt ablation。

这样每一步都会把“当前最佳设置”传递给后续实验，符合原文要求。
