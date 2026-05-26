# CS336 Assignment 5：第 5 节 MATH 专家迭代任务整理

## 任务目标

在 MATH 数据集上实现并运行 **Expert Iteration，专家迭代，EI**。

核心思想是：

1. 用当前策略模型为一批数学问题生成多个 reasoning responses；
2. 用奖励函数判断哪些 responses 得到正确答案；
3. 过滤掉错误 responses；
4. 只用正确的 question-response pairs 对当前模型继续做 SFT；
5. 重复上述过程，逐步提升模型推理能力。

---

## 一、算法流程

输入：

```text
初始策略模型 πθinit
奖励函数 R
任务问题数据集 D
```

输出：

```text
更新后的策略模型 πθ
```

专家迭代流程：

```text
1. 初始化策略模型：πθ ← πθinit

2. 对 step = 1, ..., n_ei_steps：

   2.1 从数据集 D 中采样一批问题 Db

   2.2 保存旧策略模型：πθold ← πθ

   2.3 对 Db 中每个问题 q，使用旧策略模型采样 G 个输出：
       {o(i)}_{i=1}^G ~ πθold(· | q)

   2.4 对每个输出 o(i)，运行奖励函数：
       r(i) = R(q, o(i))

   2.5 过滤掉错误输出，即 r(i) = 0 的样例
       保留正确的 question-response pairs，组成 Dsft

   2.6 用 Dsft 对当前策略模型执行 SFT：
       πθ ← SFT(πθ, Dsft)

3. 返回 πθ
```

---

## 二、实现提示

### 1. vLLM generation 参数

采样时应设置 `min_tokens`，避免生成空字符串导致下游出现 NaN。

示例：

```python
sampling_min_tokens = 4

sampling_params = SamplingParams(
    temperature=sampling_temperature,
    max_tokens=sampling_max_tokens,
    min_tokens=sampling_min_tokens,
    n=G,
    seed=seed,
)
```

其中：

```text
G = 每个问题采样的 rollout 数量
```

### 2. generation 停止条件

需要确保 vLLM 在第二个 answer tag 处终止生成：

```text
</answer>
```

这一点应与 SFT 部分保持一致。

### 3. 梯度裁剪

与 SFT 一样，训练时使用 gradient clipping：

```text
clip value = 1.0
```

---

## 三、实验要求

问题名称：

```text
expert_iteration_experiment
```

分值与资源：

```text
2 分
约 6 H100 小时
```

使用模型：

```text
Qwen 2.5 Math 1.5B Base
```

训练数据：

```text
/data/a5-alignment/MATH/train.jsonl
```

固定设置：

```text
n_ei_steps = 5
```

需要改变的超参数：

```text
1. 每个问题的 rollout 数量 G
2. 每个 EI step 中 SFT 使用的 epochs 数
3. 每个 EI step 的 batch size，即 Db 的大小
```

EI batch size 候选值：

```text
512, 1024, 2048
```

注意：

```text
不需要尝试所有超参数组合；
只需要尝试足够多的配置，以便能判断每个因素的影响。
```

---

## 四、至少需要完成的实验

### 1. 不同 rollout counts

至少尝试 2 种不同的 rollout 数量 G。

例如：

```text
G = 4
G = 8
```

或根据显存与时间调整。

### 2. 不同 SFT epoch counts

至少尝试 2 种不同的 SFT epochs 数。

例如：

```text
epochs = 1
epochs = 2
```

### 3. 不同 EI batch size

从以下值中选择若干个进行比较：

```text
512, 1024, 2048
```

---

## 五、训练过程中需要记录的指标

训练过程中需要记录：

```text
1. validation accuracy
2. train reward / rollout reward
3. response entropy
4. 每个 EI step 的表现变化
5. 不同 rollout configuration 的性能曲线
```

特别要求记录：

```text
模型 response 的 entropy 随训练变化的曲线
```

---

## 六、交付物

需要提交以下内容：

### 1. Validation accuracy curves

提交与不同 rollout configurations 相关的 validation accuracy curves。

要求：

```text
至少包含 2 种不同 rollout counts
至少包含 2 种不同 epoch counts
```

### 2. 达标模型

提交或报告一个在 MATH 上达到至少：

```text
15% validation accuracy
```

的模型。

### 3. 简短讨论

写 2 句简短讨论，内容包括：

```text
1. 与 SFT performance 的比较
2. 不同 EI steps 之间 performance 的比较
```

### 4. Entropy plot

提交模型 responses 的 entropy 随训练变化的 plot。

---

## 七、建议实现顺序

```text
1. 先复用 4.2 和 4.3 中已经实现的 SFT 组件。
2. 实现 rollout generation：对每个问题采样 G 个 responses。
3. 使用 reward function 对 responses 打分。
4. 过滤 reward = 0 的错误 responses。
5. 将正确 responses 组成临时 SFT 数据集 Dsft。
6. 对当前 policy model 执行若干 epoch 的 SFT。
7. 每个 EI step 后在 validation set 上评估 accuracy。
8. 记录 response entropy。
9. 跑不同 G、epochs、batch size 配置。
10. 整理 validation accuracy curves 和 entropy plot。
```

---

## 八、最终完成清单

```text
1. Expert Iteration 训练脚本
2. Rollout generation 逻辑
3. Reward filtering 逻辑
4. 用正确 responses 构造 Dsft 的逻辑
5. 每个 EI step 内部的 SFT 更新逻辑
6. Validation evaluation 逻辑
7. Response entropy 记录逻辑
8. 不同 rollout configurations 的实验结果
9. 达到至少 15% validation accuracy 的模型
10. Validation accuracy curves
11. Entropy plot
12. 2 句实验比较讨论
```
