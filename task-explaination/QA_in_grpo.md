# 作业PDF中问题的回答与具体实现
## 原理回顾（详情见中文pdf，由于中文为gpt翻译，推荐中英文辅助对照阅读，下面页码为中文文档位置，便于快速定位与初步理解）
### 6 策略梯度(13/32)
- 一些关键概念：
- Vannila Gradient Policy:把LM处理为一种策略
- on-/off- Policy:对一种策略进行更新还是经过重要性采样后保持原策略不变对现有策略进行更新。
### 7 GRPO（17/32）
  原理部分在作业文档中写的已经较为清楚，可以参考作业文档加以理解。
## 问题1：GRPO组件(Page 18/32):
这部分在理解grpo的整个流程后实现起来就比较容易，在grpo-scaffold.py中有脚手架代码，实现起来就相对容易了。
最后一个grpo完整train-loop，可见grpo.md中笔者与gpt的博弈，有助于回顾与实现整个训练流程。
## 问题2：GRPO实验（Page24/32）
- 最终采用的配置
```bash
n_grpo_steps = 50
learning_rate = 2e-5

rollout_batch_size = 256
group_size = 8
n_prompts_per_rollout_batch = 32

epochs_per_rollout_batch = 2
train_batch_size = 256
gradient_accumulation_steps = 128
micro_train_batch_size = 2

loss_type = grpo_clip
cliprange = 0.2

use_std_normalization = True
advantage_eps = 1e-6

sampling_temperature = 1.0
sampling_min_tokens = 4
sampling_max_tokens = 768

eval_every = 20
eval_limit = 1000
eval_max_new_tokens = 768

max_seq_len = 2048
max_grad_norm = 1.0
length_normalization = masked_mean
```
评估点step 10, 20, 30, 40, 50,最终再来一次完整测试
命令如下（假设在sft中已经成功载入环境和模型）：
```bash
cd ~/All-code/AI-lessons/CS336/assignment5-alignment-main/assignment5-alignment-main(cd进自己的目录)

rsync -av -e "ssh -p 34191" \
  scripts/train_grpo.py \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/scripts/

rsync -av -e "ssh -p 34191" \
  cs336_alignment/grpo.py \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/cs336_alignment/

连接到服务器中运行：
cd /root/workspace/assignment5-alignment-main
mkdir -p /data/outputs/grpo/offpolicy_50step_fast /data/cache /data/tmp
tmux new -s grpo
CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=/data/cache/huggingface \
TRANSFORMERS_CACHE=/data/cache/huggingface \
XDG_CACHE_HOME=/data/cache \
TMPDIR=/data/tmp \
PYTHONPATH=. python scripts/train_grpo.py \
  --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --train-path /data/math/train.jsonl \
  --val-path /data/math/val.jsonl \
  --output-dir /data/outputs/grpo/offpolicy_50step_fast \
  --run-name offpolicy_50step_fast \
  --experiment grpo_off_policy \
  --final-full-eval \
  --gradient-checkpointing \
  2>&1 | tee /data/outputs/grpo/offpolicy_50step_fast/train.log

按crtl+b后在按d退出（可通过 tmux attach -t grpo 再进入）
新开一个终端连接服务器
watch -n 5 nvidia-smi 可查看显卡运行情况
```

### 总结：
实际训练过程中，大约3分半1个step，由于采用的是mean_masked，可能会鼓励长回答，随着模型训练加深，速度可能会变慢，但实测也相差不多。
这里可以简单计算一下每个step的FLOP：
首先给出一些假设：
- 模型参数量 ($P$)：Qwen2.5-1.5B，即 $P = 1.5 \times 10^9$。
- 平均序列长度 ($N$)：不妨假设题目（Prompt）为 256 tokens，模型生成的回答（Response）为 768 tokens，则$N \approx 1024$ tokens(严格意义上应该是小于)
- 按照经典的算法，前向传播为2P，反向传播为4P，大约共6P次浮点运算
基于这些假设，可以如下计算：
- 每个rollout batch：生成256个回答，每个回答生成768token，计算公式为$2 \times P \times \text{Batch} \times N_{gen}$，算力消耗：$$2 \times 1.5 \times 10^9 \times 256 \times 768 \approx 5.9 \times 10^{14} \text{ FLOPs}$$
- 旧模型轨迹计算量：此时生成的N=1024，算力消耗：$$2 \times 1.5 \times 10^9 \times 256 \times 1024 \approx 7.8 \times 10^{14} \text{ FLOPs}$$
- training阶段：没一步需要处理2个micro_batch，也就是$2*256=512$条序列，$N = 1024$，消耗：$$6 \times 1.5 \times 10^9 \times 512 \times 1024 \approx 4.7 \times 10^{15} \text{ FLOPs}$$
- 三者加起来，共消耗$$(0.59 + 0.78 + 4.7) \times 10^{15} \approx 6.07 \times 10^{15} \text{ FLOPs}$$
笔者实际用的是1张A40用于训练，1张用于评估，算例大约$1.5 \times 10^{14}$FLOPS，实际MFU折损综合几个前向、反向等几个过程能到30%就不错了，大约$$150 \text{ TFLOPs} \times 30\% \approx 45 \text{ TFLOPs/sec}$$，估算出$$\frac{6.07 \times 10^{15}}{4.5 \times 10^{13}} \approx 135 \text{ 秒} \text{（2 分 15 秒）}$$。如此来看，如果估计没有错的话，大约实际训练时间与理论时间要多了1分半左右（1/3）。最终大约跑了3个小时。
结果如下图，其余结果见/grpo-outputs
![alt text](image-1.png)


### 更新：In-Context Reinforcement Learning for Tool Use in Large Language Models 简单复现
基于这个作业，正好可以用来简单在non-tooluse的场景下简单复现一下icrl的结果（论文链接；https://arxiv.org/pdf/2603.08068）。
- 原理：

这里做的是一个 no-tool ICRL-lite 对比实验，不使用工具、不做 SFT、不改 reward function，也不改 GRPO 的 loss / advantage / logprob / mask / optimizer 等核心实现。  
唯一变化是 rollout 阶段的 prompt：在原始 `r1_zero` prompt 前拼接少量 few-shot demonstrations；而 validation/eval 仍然使用原始 zero-shot `r1_zero` prompt。

固定 schedule：

```text
step 1-10: 3-shot
step 11-20: 2-shot，使用后 2 个 demo
step 21-30: 1-shot，使用最后 1 个 demo
step 31-50: 0-shot
```

demo bank 放在：

```text
data/icrl/fewshot_r1_zero.jsonl
```

实现文件：

```text
cs336_alignment/icrl.py
scripts/train_grpo_icrl_lite.py
```

参数检查：`train_grpo_icrl_lite.py` 复用 `train_grpo.py` 的 `parse_args()` 和 `build_config()`，因此除 `--icrl-demo-path` 与 rollout prompt 构造外，其余训练参数与 50-step GRPO baseline 保持一致。也就是说，`n_grpo_steps`、`learning_rate`、`rollout_batch_size`、`group_size`、`epochs_per_rollout_batch`、`train_batch_size`、`gradient_accumulation_steps`、`loss_type`、`cliprange`、eval 设置等都沿用同一套参数。

- 训练命令：

先从本地同步新增文件：

```bash
cd ~/All-code/AI-lessons/CS336/assignment5-alignment-main/assignment5-alignment-main

rsync -av -e "ssh -p 34191" \
  cs336_alignment/icrl.py \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/cs336_alignment/

rsync -av -e "ssh -p 34191" \
  scripts/train_grpo_icrl_lite.py \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/scripts/

ssh -p 34191 root@ssh-cn-huabei1.ebcloud.com "mkdir -p /root/workspace/assignment5-alignment-main/data/icrl"

rsync -av -e "ssh -p 34191" \
  data/icrl/fewshot_r1_zero.jsonl \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/data/icrl/
```

服务器中运行：

```bash
cd /root/workspace/assignment5-alignment-main
mkdir -p /data/outputs/grpo/icrl_lite_50step /data/cache /data/tmp
tmux new -s grpo_icrl

CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=/data/cache/huggingface \
TRANSFORMERS_CACHE=/data/cache/huggingface \
XDG_CACHE_HOME=/data/cache \
TMPDIR=/data/tmp \
PYTHONPATH=. python scripts/train_grpo_icrl_lite.py \
  --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --train-path /data/math/train.jsonl \
  --val-path /data/math/val.jsonl \
  --output-dir /data/outputs/grpo/icrl_lite_50step \
  --run-name icrl_lite_50step \
  --experiment grpo_off_policy \
  --n-grpo-steps 50 \
  --learning-rate 2e-5 \
  --rollout-batch-size 256 \
  --group-size 8 \
  --epochs-per-rollout-batch 2 \
  --train-batch-size 256 \
  --gradient-accumulation-steps 128 \
  --loss-type grpo_clip \
  --cliprange 0.2 \
  --use-std-normalization \
  --sampling-temperature 1.0 \
  --sampling-min-tokens 4 \
  --sampling-max-tokens 512 \
  --eval-every 10 \
  --eval-limit 512 \
  --eval-max-new-tokens 512 \
  --max-seq-len 2048 \
  --max-grad-norm 1.0 \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --rollout-backend vllm \
  --eval-backend vllm \
  --cache-dir /data/cache \
  --tmp-dir /data/tmp \
  --tmp-model-dir /data/tmp/grpo_vllm_policy \
  --icrl-demo-path data/icrl/fewshot_r1_zero.jsonl \
  --final-full-eval \
  --gradient-checkpointing \
  2>&1 | tee /data/outputs/grpo/icrl_lite_50step/train.log
```

查看结果：

```bash
cat /data/outputs/grpo/icrl_lite_50step/summary.json
cat /data/outputs/grpo/icrl_lite_50step/final_full_eval.summary.json
tail -20 /data/outputs/grpo/icrl_lite_50step/metrics.jsonl
```
