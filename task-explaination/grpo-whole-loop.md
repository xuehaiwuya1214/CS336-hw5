1. 第一层：把 GRPO 算法翻译成“数据流”

先不要想代码，先问一句：

每一步输入什么，输出什么，shape 是什么？

GRPO 的核心训练链路其实就这一条：

train_examples
  ↓
采样 prompts
  ↓
每个 prompt 生成 G 个 rollout responses
  ↓
reward_fn(response, answer)
  ↓
raw_rewards, advantages
  ↓
tokenize(prompt, response)
  ↓
input_ids, labels, response_mask
  ↓
计算 old_log_probs
  ↓
重新前向，计算 current policy_log_probs
  ↓
算 policy gradient loss
  ↓
用 response_mask 聚合
  ↓
loss.backward()
  ↓
gradient accumulation
  ↓
clip grad + optimizer.step()
  ↓
log / eval / checkpoint

你写训练代码时，脑子里要一直维护这几个核心张量：

R = rollout_batch_size
G = group_size
B = train_batch_size
m = micro_train_batch_size
T = sequence_length

于是常见 shape 是：

rollout_responses        list[str], 长度 R
repeated_prompts         list[str], 长度 R
repeated_ground_truths   list[str], 长度 R

raw_rewards              (R,)
advantages               (R,) 或 (R, 1)

input_ids                (R, T)
labels                   (R, T)
response_mask            (R, T)

old_log_probs            (R, T)
policy_log_probs         (m, T)

per_token_loss           (m, T)
per_example_loss         (m,)
scalar_loss              ()

只要这条数据流清楚，训练 loop 就不会乱。

2. 哪些东西值得单独提出来写？

一个很实用的判断标准是：

只要一个步骤有清晰的数学定义、固定输入输出、容易单测，就应该抽成函数。

所以这些应该单独写：

compute_group_normalized_rewards(...)
compute_naive_policy_gradient_loss(...)
compute_grpo_clip_loss(...)
compute_policy_gradient_loss(...)
masked_mean(...)
grpo_microbatch_train_step(...)

因为它们满足三个特点：

第一，它们是“数学原语”。比如 GRPO-Clip loss、group-normalized advantage、masked mean，本身就是公式翻译成代码。作业文档也把这些作为单独问题列出来，让你分别测试。GRPO 的 group advantage 来自同一题的多个 rollout，先算 reward，再减组均值，可选择除以组内标准差；GRPO-Clip 则要用当前 policy 和 old policy 的 log-prob ratio 做 clipping。

第二，它们不应该知道外部世界。比如 compute_grpo_clip_loss 不需要知道 tokenizer、vLLM、dataset、wandb、optimizer，它只需要：

advantages
policy_log_probs
old_log_probs
cliprange

这类函数越纯越好。

第三，它们容易错，但容易测。比如：

ratio = torch.exp(policy_log_probs - old_log_probs)
loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)

这类地方如果写在大训练循环里，debug 会非常痛苦；单独抽出来可以直接用小 tensor 测。

3. 哪些可以留在 train loop 里？

训练 loop 的职责不是做数学，而是调度。

也就是：

采样数据
生成 rollout
组织 batch
搬到 device
调用函数算 loss
控制 optimizer
记录日志
周期评估
保存结果

所以像下面这些可以留在 grpo_train_loop 里：

for grpo_step in range(...):
    prompt_records = rng.sample(...)
    prompts = [...]
    ground_truths = [...]
for rollout_epoch in range(config.epochs_per_rollout_batch):
    rng.shuffle(epoch_indices)
    train_indices = ...
optimizer.zero_grad(...)
...
clip_grad_norm_(...)
optimizer.step()

这些不是稳定的数学模块，而是训练过程的“胶水代码”。你当前代码的 grpo_train_loop 注释其实已经非常接近正确思路：先根据 config 推导 microbatch 与 prompt 数量，然后每个 GRPO step 采样 prompt、生成 rollout、算 reward/advantage、tokenize、算 old log-probs、分 microbatch 训练，最后做 optimizer、日志、评估。

4. 一个最有用的拆分原则

可以用这个表判断。

类型	是否抽函数	例子
公式本身	必须抽	advantage、GRPO loss、masked mean
会被多个地方复用	应该抽	tokenize batch、compute old log probs、evaluate
有复杂 shape 约束	应该抽	microbatch train step
和外部库强绑定	可以抽薄封装	vLLM rollout、transformers generate
只出现一次的流程胶水	留在 loop	sample prompts、shuffle indices
日志字段拼装	可留在 loop	metrics dict
简单 config 推导	可留在 loop	micro_train_batch_size = train_batch_size // grad_acc

一句话：

“会被测试、会被复用、会独立犯错”的东西抽出来；“只负责把各模块按顺序连起来”的东西留在训练循环。

5. 参数类型多变时，怎么快速理清？

你不要先看参数名，要先把参数分成五类。

第一类：模型对象
policy
tokenizer
optimizer
reward_fn

它们不是数据，而是“工具”。

第二类：原始文本数据
prompts: list[str]
rollout_responses: list[str]
ground_truths: list[str]

这些还没有进入模型训练的 tensor 世界。

第三类：tokenized tensor
input_ids: Tensor      # (R, T)
labels: Tensor         # (R, T)
response_mask: Tensor  # (R, T)

它们来自 tokenizer，是模型 forward 的输入。

第四类：奖励和 advantage
raw_rewards: Tensor    # (R,) or (R, 1)
advantages: Tensor     # (R,) or (R, 1)

它们来自 reward_fn，不来自模型 forward。

第五类：log-probs 和 loss
old_log_probs: Tensor     # (R, T)
policy_log_probs: Tensor  # (m, T)
per_token_loss: Tensor    # (m, T)

它们来自模型 forward 和 loss 公式。

写代码前，你可以在纸上写：

文本世界：
prompts / responses / answers

token 世界：
input_ids / labels / response_mask

reward 世界：
raw_rewards / advantages

policy 世界：
old_log_probs / policy_log_probs

optimization 世界：
loss / backward / optimizer.step

很多人写乱，就是因为把这几个世界混在一起了。

6. 写训练 loop 的正确顺序

不要从空白文件开始写完整训练。应该按这个顺序来。

第一步：只写 shape 推导和 asserts
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
n_prompts_per_rollout_batch = rollout_batch_size // group_size

这一步先不训练，只保证所有 batch size 合法。作业文档也强调了这些 sanity check：train_batch_size 要能被 gradient_accumulation_steps 整除，rollout_batch_size 要能被 group_size 整除，并由此得到 microbatch size 和每个 rollout batch 的 prompt 数量。

第二步：写 rollout 部分

目标是得到：

rollout_responses: list[str]  # 长度 R
repeated_prompts: list[str]   # 长度 R
repeated_ground_truths        # 长度 R

这里不要碰 loss。

第三步：写 reward / advantage

目标是得到：

raw_rewards: Tensor      # (R,)
advantages: Tensor       # (R,)

这个时候先打印：

reward_mean
answer_reward_mean
advantage_mean
advantage_std

如果 reward 全是 0，后面的训练基本没意义，应该先检查生成格式和 reward_fn。

第四步：写 tokenize

目标是得到：

input_ids
labels
response_mask

这里重点检查：

input_ids.shape == labels.shape == response_mask.shape
response_mask.sum() > 0
第五步：写 old_log_probs

如果是 grpo_clip，必须在 policy 更新前缓存：

old_log_probs = _compute_old_log_probs(...)

因为 GRPO-Clip 用的是：

ratio = current_policy_prob / old_policy_prob

你的代码里 _compute_old_log_probs 就是这个用途：在 policy 更新前缓存 old log-probs，后面多个训练 step 可以复用。

第六步：写一个 microbatch 的训练

先只跑一个 microbatch：

output = get_response_log_probs(...)
loss, metadata = grpo_microbatch_train_step(...)

这里检查：

loss 是标量
loss.requires_grad == True
backward 后参数有 grad
第七步：包上 gradient accumulation

再写：

optimizer.zero_grad()

for microbatch in microbatches:
    loss.backward()

clip_grad_norm_(...)
optimizer.step()

作业文档对 grpo_microbatch_train_step 的要求也是：计算 policy-gradient loss，用 mask 聚合，针对 gradient accumulation 缩放，并调用 loss.backward()。

第八步：加日志、eval、checkpoint

这些永远最后加。不要一开始就写 wandb、eval、save，否则 bug 会藏得很深。

7. 你可以记一个“GRPO 训练伪代码模板”

以后看到类似 RLHF / GRPO / PPO 代码，都可以先还原成这个骨架：

for step in range(n_grpo_steps):

    # 1. sample questions
    prompts, answers = sample_questions(train_set)

    # 2. rollout with old/current policy
    responses = generate(policy, prompts, group_size=G)

    # 3. reward and advantage
    raw_rewards, advantages = compute_rewards_and_advantages(
        responses,
        answers,
        group_size=G,
    )

    # 4. tokenize prompt-response pairs
    batch = tokenize(prompts, responses)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    response_mask = batch["response_mask"]

    # 5. cache old log-probs if needed
    old_log_probs = compute_old_log_probs(policy, input_ids, labels)

    # 6. train on this rollout batch
    for epoch in range(epochs_per_rollout_batch):
        for microbatch in split(batch):

            policy_log_probs = get_log_probs(policy, microbatch)

            loss = compute_grpo_loss(
                policy_log_probs,
                old_log_probs,
                advantages,
                response_mask,
            )

            loss = loss / gradient_accumulation_steps
            loss.backward()

        clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    # 7. log / eval
    evaluate_periodically()

你真正写代码时，就是不断把这里每一行替换成具体函数。

8. 你当前这份代码的结构其实已经挺标准

它已经有几个好的设计：

一是有 GRPOConfig，把训练超参数集中起来。这样 train loop 不会被几十个散乱参数淹没。

二是有 PolicyGradientLossType，把三种 loss 类型限制为：

"no_baseline"
"reinforce_with_baseline"
"grpo_clip"

这比随便传字符串更安全。

三是把 compute_policy_gradient_loss 做成 wrapper。这样 train loop 不用关心当前到底是哪种 loss，只需要传入：

loss_type
raw_rewards
advantages
old_log_probs
cliprange

四是把 grpo_microbatch_train_step 抽出来。这个函数正好是“数学 loss”和“工程 backward”之间的边界：它接收 log-probs、mask、reward/advantage，产出 scalar loss 并 backward。这种边界设计是合理的。

9. 最后给你一个写这类代码的工作法

可以按“四张表”来写。

第一张表：算法步骤表。

sample questions
generate responses
compute rewards
compute advantages
tokenize
compute old log probs
train microbatches
optimizer step
eval

第二张表：数据 shape 表。

responses: R
advantages: R or R x 1
input_ids: R x T
log_probs: R x T
microbatch log_probs: m x T
loss: scalar

第三张表：函数边界表。

compute_xxx: 纯数学
_tokenize_xxx: 文本 -> tensor
_generate_xxx: 模型生成
_train_loop: 调度

第四张表：debug checkpoint 表。

rollout 数量对不对？
reward 是否全 0？
response_mask 是否为空？
old_log_probs shape 是否等于 policy_log_probs？
loss 是否为 NaN？
grad norm 是否正常？
eval reward 是否有变化？

把这四张表写出来以后，训练代码基本就是“照表填空”。






0. 五个量分别是什么意思
R = rollout_batch_size      # 一次 rollout 总共生成多少条 response
G = group_size              # 每个 prompt 生成多少条 response
B = train_batch_size        # 一次 optimizer step 用多少条 rollout 样本训练
m = micro_train_batch_size  # 每个 microbatch 放多少条样本
T = sequence_length         # tokenized 后的序列长度

其中有几个关系：

n_prompts = R // G
m = B // gradient_accumulation_steps
num_microbatches = B // m

比如作业默认：

R = 256
G = 8
B = 256
gradient_accumulation_steps = 128

那么：

n_prompts = 256 // 8 = 32
m = 256 // 128 = 2
num_microbatches = 256 // 2 = 128

意思是：

一次 GRPO step 抽 32 道题，每题生成 8 个回答，一共 256 条回答。训练时用这 256 条回答，但显存放不下，所以每次只喂 2 条，喂 128 次，累积梯度后更新一次参数。

1. 第一步：抽 prompt

先从训练集中抽题。

n_prompts = R // G

比如：

R = 256
G = 8

那么抽：

n_prompts = 32

此时数据形状是：

prompts:       长度 32
ground_truths: 长度 32

也就是：

q1, q2, q3, ..., q32
a1, a2, a3, ..., a32
2. 第二步：每个 prompt 生成 G 个 response

GRPO 的关键是：同一道题生成多个回答，然后在组内比较好坏。

所以每道题生成 G 个回答。

如果 G = 8，那么：

q1 -> o11, o12, ..., o18
q2 -> o21, o22, ..., o28
...
q32 -> o32,1, ..., o32,8

总 response 数量是：

R = n_prompts * G = 32 * 8 = 256

此时：

rollout_responses: 长度 R

即：

rollout_responses: 长度 256

同时，为了让每个 response 都能和自己的 prompt、answer 对齐，需要把 prompt 和 answer 也重复 G 次：

repeated_prompts:
q1, q1, q1, q1, q1, q1, q1, q1,
q2, q2, q2, q2, q2, q2, q2, q2,
...
q32 repeated 8 times
repeated_ground_truths:
a1, a1, a1, a1, a1, a1, a1, a1,
a2, a2, a2, a2, a2, a2, a2, a2,
...
a32 repeated 8 times

所以现在有三列一一对应：

repeated_prompts          长度 R
rollout_responses         长度 R
repeated_ground_truths    长度 R

比如第 17 个元素可能是：

prompt:   q3
response: q3 的第 1 个回答
answer:   a3
3. 第三步：算 reward

现在对每条 response 打分：

reward_fn(response, ground_truth)

得到：

raw_rewards.shape == (R,)

比如：

raw_rewards: (256,)

里面每个值通常是 0 或 1：

[0, 1, 0, 0, 1, 0, 0, 1, ...]

但 GRPO 不直接用 raw reward，而是先按题目分组。

因为每个题有 G 个回答，所以 reshape 成：

grouped_rewards.shape == (n_prompts, G)

也就是：

grouped_rewards: (32, 8)

一行对应一道题的 8 个回答：

q1: [0, 1, 0, 0, 1, 0, 0, 1]
q2: [0, 0, 0, 1, 0, 0, 0, 0]
...
4. 第四步：组内算 advantage

GRPO 的 advantage 是组内比较来的。

比如某道题 8 个回答的 reward 是：

[0, 1, 0, 0, 1, 0, 0, 1]

平均值是：

mean = 3 / 8 = 0.375

那么每条 response 的 advantage 大概是：

[-0.375, 0.625, -0.375, -0.375, 0.625, -0.375, -0.375, 0.625]

如果使用标准差归一化，就是：

advantage = (reward - group_mean) / (group_std + eps)

如果不用标准差归一化，就是：

advantage = reward - group_mean

所以 advantage 先是：

advantages.shape == (n_prompts, G)

也就是：

(32, 8)

然后 flatten 回：

advantages.shape == (R,)

也就是：

(256,)

训练时一般再变成：

advantages.shape == (R, 1)

为什么要变成 (R, 1)？

因为后面要和每个 token 的 log-prob 相乘：

policy_log_probs.shape == (R, T)
advantages.shape       == (R, 1)

这样 advantage 会自动 broadcast 到每个 token：

第 i 条 response 的所有 token 使用同一个 advantage。
5. 第五步：tokenize prompt + response

现在把：

prompt + response

拼起来，丢进 tokenizer。

得到三个核心张量：

input_ids.shape     == (R, T)
labels.shape        == (R, T)
response_mask.shape == (R, T)

比如：

input_ids:      (256, T)
labels:         (256, T)
response_mask:  (256, T)

这里的 T 是当前 rollout batch 中 padding 后的统一长度。

每一行是一条：

prompt + response

例如：

第 0 行：q1 + o11
第 1 行：q1 + o12
第 2 行：q1 + o13
...
第 8 行：q2 + o21

response_mask 的作用是：只在 response 部分算 loss，不在 prompt 部分算 loss。

例如：

prompt tokens:   0 0 0 0 0
response tokens: 1 1 1 1 1 1 1
padding tokens:  0 0 0

所以一行 mask 可能长这样：

[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
6. 第六步：算 old_log_probs

如果用的是 grpo_clip，需要先固定旧策略。

在模型更新之前，计算：

old_log_probs.shape == (R, T)

也就是：

old_log_probs: (256, T)

含义是：

这些 response 是旧 policy 生成的，所以要记录旧 policy 对每个 token 的 log probability。

后面训练时会算：

ratio = exp(policy_log_probs - old_log_probs)

也就是：

当前 policy 给这个 token 的概率 / 旧 policy 给这个 token 的概率

这个 ratio 是 GRPO-Clip 的核心。

7. 第七步：从 R 个 rollout 里取 B 个训练

通常作业默认：

B = R

也就是一次 rollout 得到的 256 条 response 全部拿来训练。

但一般也允许：

B < R

比如：

R = 512
B = 256

那就是先生成 512 条 rollout，但每次 optimizer step 只抽 256 条训练。

所以进入训练时，张量从：

input_ids.shape       == (R, T)
response_mask.shape   == (R, T)
advantages.shape      == (R, 1)
old_log_probs.shape   == (R, T)

取出 B 条：

batch_input_ids.shape      == (B, T)
batch_response_mask.shape  == (B, T)
batch_advantages.shape     == (B, 1)
batch_old_log_probs.shape  == (B, T)

如果 B = R = 256，那就是：

(256, T)
8. 第八步：把 B 切成 microbatch

显存一般放不下整个 B，所以要切成 microbatch。

m = micro_train_batch_size

比如：

B = 256
gradient_accumulation_steps = 128

所以：

m = 256 // 128 = 2

于是训练时不是一次喂 256 条，而是：

第 1 个 microbatch:   2 条
第 2 个 microbatch:   2 条
...
第 128 个 microbatch: 2 条

每个 microbatch 的形状是：

micro_input_ids.shape      == (m, T)
micro_labels.shape         == (m, T)
micro_response_mask.shape  == (m, T)
micro_advantages.shape     == (m, 1)
micro_old_log_probs.shape  == (m, T)

如果 m = 2：

micro_input_ids:      (2, T)
micro_response_mask:  (2, T)
micro_advantages:     (2, 1)
micro_old_log_probs:  (2, T)
9. 第九步：当前 policy 前向，得到 policy_log_probs

对每个 microbatch 做 forward：

output = get_response_log_probs(
    model=policy,
    input_ids=micro_input_ids,
    labels=micro_labels,
)

得到：

policy_log_probs.shape == (m, T)

比如：

policy_log_probs: (2, T)

含义是：

当前 policy 对这 2 条 response 中每个 token 的 log probability。

10. 第十步：算 per-token GRPO loss

现在几个东西形状是：

policy_log_probs.shape   == (m, T)
old_log_probs.shape      == (m, T)
advantages.shape         == (m, 1)
response_mask.shape      == (m, T)

GRPO-Clip 做的是：

ratio = exp(policy_log_probs - old_log_probs)

所以：

ratio.shape == (m, T)

然后：

unclipped = ratio * advantages
clipped = clip(ratio, 1 - eps, 1 + eps) * advantages

由于：

advantages.shape == (m, 1)

会广播成：

(m, T)

所以：

per_token_loss.shape == (m, T)

也就是每个 token 都有一个 loss。

11. 第十一步：用 response_mask 聚合 loss

注意，prompt 部分不能算 loss，所以用：

response_mask.shape == (m, T)

做 masked mean：

per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

此时：

per_example_loss.shape == (m,)

意思是：

每条 response 得到一个 loss。

然后对 batch 维度取平均：

unscaled_loss = per_example_loss.mean()

得到标量：

unscaled_loss.shape == ()

再除以梯度累积步数：

scaled_loss = unscaled_loss / gradient_accumulation_steps

然后：

scaled_loss.backward()
12. 第十二步：gradient accumulation

因为每次只喂 m 条，但是希望等价于一次喂 B 条，所以要累积梯度。

比如：

B = 256
m = 2
gradient_accumulation_steps = 128

那么：

microbatch 1: backward，但不 step
microbatch 2: backward，但不 step
...
microbatch 128: backward，仍然不立即 step

等 128 个 microbatch 都 backward 完之后：

clip_grad_norm_(policy.parameters(), max_grad_norm)
optimizer.step()
optimizer.zero_grad()

所以一次 optimizer step 实际看到了：

128 * 2 = 256

条样本，也就是 B 条。

13. 整体形状流动总结

完整写成一条线就是：

抽题：
prompts: (R/G,)
answers: (R/G,)

生成：
rollout_responses: (R,)

重复对齐：
repeated_prompts:       (R,)
repeated_ground_truths: (R,)

奖励：
raw_rewards: (R,)

组内 advantage：
grouped_rewards: (R/G, G)
advantages:      (R/G, G)
flatten:
advantages:      (R,)
unsqueeze:
advantages:      (R, 1)

tokenize：
input_ids:      (R, T)
labels:         (R, T)
response_mask:  (R, T)

old policy：
old_log_probs:  (R, T)

取训练 batch：
input_ids:      (B, T)
labels:         (B, T)
response_mask:  (B, T)
advantages:     (B, 1)
old_log_probs:  (B, T)

切 microbatch：
input_ids:      (m, T)
labels:         (m, T)
response_mask:  (m, T)
advantages:     (m, 1)
old_log_probs:  (m, T)

当前 policy 前向：
policy_log_probs: (m, T)

GRPO loss：
per_token_loss:   (m, T)

mask 聚合：
per_example_loss: (m,)

取均值：
loss: scalar

梯度累积：
重复 B/m 次 backward

参数更新：
optimizer.step()
最简直觉版

可以这么记：

G：每道题生成几个回答。
R：这一轮一共生成多少个回答。
B：这一轮拿多少个回答训练。
m：显存一次只能吃几个回答。
T：每个回答 token 化后有多长。

所以：

R/G 道题
每题 G 个回答
合起来 R 条回答
从 R 条里拿 B 条训练
每次只喂 m 条
每条长度 T

如果用默认例子：

32 道题
每题 8 个回答
一共 256 条回答
256 条都拿来训练
每次只喂 2 条
喂 128 次
最后更新一次模型