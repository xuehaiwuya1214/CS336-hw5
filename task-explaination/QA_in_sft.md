# 作业PDF中问题的回答与具体实现（详情见中文pdf，由于中文为gpt翻译，推荐中英文辅助对照阅读，下面页码为中文文档位置，便于快速定位与初步理解）
## 问题1：测试math_baseline(Page 5/32):
- （1）下载Qwen 1.5B Math到本地
- （2）用\data\gsm8k\test.jsonl来进行测试
- （3）把一条条数据统一为r1-promt的格式
- （4）用vllm进行本地推理（实际运行中占用显存约6G，4060 8G可跑通 3-5分钟）
- （5）整理评测结果————格式正确、结果正确

- 运行命令：.venv/bin/python scripts/math_baseline.py \
  --input-path data/gsm8k/test.jsonl \
  --output-path outputs/gsm8k_qwen25_math_15b_vllm_full.jsonl \
  --generation-mode vllm \
  --model-name-or-path /home/xuehaiwuya/models/Qwen2.5-Math-1.5B \
  --num-gpus 1 \
  --max-tokens 1024 \
  --analysis-examples-per-bucket 10 \
  --seed 0

- 运行结果：{
  "num_examples": 1319,
  "mean_reward": 0.17968157695223655,
  "mean_format_reward": 0.5109931766489765,
  "mean_answer_reward": 0.17968157695223655,
  "format_1_answer_1": 237,
  "format_1_answer_0": 437,
  "format_0_answer_0": 645,
  "input_path": "data/gsm8k/test.jsonl",
  "output_path": "outputs/gsm8k_qwen25_math_15b_vllm_full_1024.jsonl",
  "model_name_or_path": "/home/xuehaiwuya/models/Qwen2.5-Math-1.5B",
  "generation_mode": "vllm",
  "limit": null,
  "max_tokens": 1024,
  "temperature": 0.0,
  "top_p": 1.0
}
- 之后sft全参数采用hiyouga/math12k的数据集，故加入另外的测试脚本scripts/prepare_math12k.py

## 问题2：SFT
### 4.2 sft组件
- 按照文档要求，在cs336_alignment/sft.py中实现要求的六个组件。分别如下：
  - tokenize_prompt_and_output
    把一批 prompt 和 response 分别 tokenize，然后拼接成 causal LM 训练样例。返回：

    input_ids：模型输入
    labels：右移后的 next-token 目标
    response_mask：只标记 response 部分，prompt 和 padding 不参与 loss
  - compute_entropy
    对模型 logits 计算每个 token 位置的 vocabulary entropy，用于观察模型输出分布的不确定性。

  - get_response_log_probs
    用模型算出 labels 每个 token 的 log-prob。训练时会再配合 response_mask 只取 response 部分。

  - masked_normalize
    对 mask 为 1 的位置求和，再除以固定归一化常数。

  - sft_microbatch_train_step
    实现一个 SFT microbatch 的 loss 和 backward：

    loss = response token 上的 negative log likelihood
    prompt token 不参与 loss
    支持 gradient accumulation
  - log_generations
    记录生成样例、reward、平均长度、平均 entropy 等，给后面训练/eval 分析用。

- 运行测试：.venv/bin/python -m pytest tests/test_sft.py::test_compute_entropy \
  tests/test_sft.py::test_masked_normalize_dim0 \
  tests/test_sft.py::test_masked_normalize_dim1 \
  tests/test_sft.py::test_masked_normalize_dimlast \
  tests/test_sft.py::test_masked_normalize_dimNone \
  tests/test_sft.py::test_sft_microbatch_train_step \
  tests/test_sft.py::test_sft_microbatch_train_step_normalize \
  tests/test_sft.py::test_sft_microbatch_train_step_10_steps \
  -q
  （其中最后一个test_sft需要在fixture中更改Qwen模型位置为自己的路径，或者建立软连接如
  sudo mkdir -p /data/a5-alignment/models
  sudo ln -s /home/xuehaiwuya/models/Qwen2.5-Math-1.5B \
  /data/a5-alignment/models/Qwen2.5-Math-1.5B，之后运行.venv/bin/python -m pytest -k test_sft -q即可）

### 4.3 sft实验
Hugging Face 文件	对应作业里的什么	用途
val.jsonl	/data/a5-alignment/MATH/validation.jsonl	做 baseline evaluation 和 SFT validation
sft_gpt-oss-120b.jsonl	/data/a5-alignment/MATH/sft.jsonl 的替代版	未过滤 SFT 数据，包含正确和错误 reasoning traces
sft_gpt-oss-120b_filtered.jsonl	4.3 中“过滤后 SFT 数据”	只保留能得到正确答案的 reasoning traces，README 也标为 recommended
train.jsonl	/data/a5-alignment/MATH/train.jsonl 的替代版	后面 Expert Iteration / GRPO 用的训练问题，不是 4.3 SFT 的主数据
baseline_results.jsonl	3.2 baseline 实验输出	已经跑好的 baseline 结果，不是训练数据
r1_zero.prompt	cs336_alignment/prompts/r1_zero.prompt	baseline / validation generation 的 prompt 模板
math_results.jsonl	原始 validation 结果/数据来源	一般不用直接训练
train_data_4_batchinference_gpt-oss-120b.jsonl	生成 SFT 数据的中间输入	一般不用训练
batch-infer-math-train-outputs_gpt-oss-120b.jsonl	GPT-OSS-120B 批量推理原始输出	一般不用训练

具体要求参考任务书，主要实现未过滤和已过滤数据集的sft。训练脚本为scripts/train_sft.py scripts/run_sft_experiment，用两个gpu，一张跑训练，一张搁固定的step后跑vllm测试

```bash
mkdir -p /data/a5-alignment/models

HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
  --local-dir /data/a5-alignment/models/Qwen2.5-Math-1.5B

  MODEL_PATH=/data/a5-alignment/models/Qwen2.5-Math-1.5B \
bash scripts/run_sft_experiments.sh smoke

bash scripts/run_sft_experiments.sh unfiltered_128
bash scripts/run_sft_experiments.sh unfiltered_256
bash scripts/run_sft_experiments.sh unfiltered_512
bash scripts/run_sft_experiments.sh unfiltered_1024
bash scripts/run_sft_experiments.sh unfiltered_full
bash scripts/run_sft_experiments.sh filtered_full

MODEL_PATH=/path/to/Qwen2.5-Math-1.5B \
DATA_DIR=/data/math \
OUTPUT_ROOT=outputs/sft \
bash scripts/run_sft_experiments.sh unfiltered_128
```

输出在outputs/sft/<experiment_name>/，包含
```bash
run_config.json
checkpoints/
eval/step_xxxxxx.jsonl
eval/step_xxxxxx.summary.json(里面含有eval/answer_accuracy
eval/reward
eval/format_accuracy)
```

## 问题3：专家迭代
基本思想是采样模型对同一个问题的多个输出，筛选其中正确的轨迹，用来做sft（STaR）。
代码实现在scripts/train_expert_iteration.py
scripts/run_expert_iteration_experiments.sh
读取train.jsonl的问题数据
命令：
```bash
bash scripts/run_expert_iteration_experiments.sh smoke
bash scripts/run_expert_iteration_experiments.sh g4_e1_b512
bash scripts/run_expert_iteration_experiments.sh g8_e1_b512
bash scripts/run_expert_iteration_experiments.sh g4_e2_b512
bash scripts/run_expert_iteration_experiments.sh g4_e1_b1024
bash scripts/run_expert_iteration_experiments.sh g4_e1_b2048
```
输出在outputs/expert_iteration/<experiment_name>/，包含
```bash
run_config.json
ei_step_01/rollouts.jsonl
ei_step_01/correct_sft.jsonl
ei_step_01/metrics.json(内含loss、accuracy等)
checkpoints/ei_step_01/
...
```


## 实际服务器训练

这里记录一次真实服务器复现实验流程，目标是在英博云 2 张 A40 上跑通 4.3 的 filtered SFT。

### 服务器与目录约定

本次服务器配置：

- Ubuntu 22.04
- 2 张 NVIDIA A40，每张约 48GB 显存
- 项目目录：`/root/workspace/assignment5-alignment-main`
- 模型目录：`/data/a5-alignment/models/Qwen2.5-Math-1.5B`
- 数据目录：`/data/math`
- filtered SFT 数据：`/data/math/sft_gpt-oss-120b_filtered.jsonl`
- validation 数据：`/data/math/val.jsonl`

连接服务器：

```bash
ssh -p 34191 root@ssh-cn-huabei1.ebcloud.com
```

检查 GPU：

```bash
nvidia-smi
```

### 上传代码

注意本地仓库有两层 `assignment5-alignment-main`，真实代码目录是：

```bash
~/All-code/AI-lessons/CS336/assignment5-alignment-main/assignment5-alignment-main
```

第一次上传整个项目：

```bash
cd ~/All-code/AI-lessons/CS336/assignment5-alignment-main/assignment5-alignment-main

rsync -av \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude "outputs" \
  --exclude "checkpoints" \
  -e "ssh -p 34191" \
  ./ \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/
```

如果服务器没有 `rsync`，先在服务器上装：

```bash
apt update
apt install -y rsync
```

后续只同步刚改过的训练脚本：

```bash
cd ~/All-code/AI-lessons/CS336/assignment5-alignment-main/assignment5-alignment-main

rsync -av \
  -e "ssh -p 34191" \
  scripts/train_sft.py \
  scripts/run_sft_experiments.sh \
  root@ssh-cn-huabei1.ebcloud.com:/root/workspace/assignment5-alignment-main/scripts/
```

### 环境配置

进入服务器项目目录：

```bash
cd /root/workspace/assignment5-alignment-main
```

需要保证能导入项目本身，所以运行脚本时都加：

```bash
PYTHONPATH=.
```

否则可能遇到：

```text
ModuleNotFoundError: No module named 'cs336_alignment'
```

安装必要依赖。若网络慢，优先使用国内源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers tqdm xopen wandb pylatexenc latex2sympy2-extended \
  "math-verify[antlr4-13-2]"
```

vLLM 也需要安装。如果镜像里没有：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm
```

如果使用 `uv`，慢的时候也可以改源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv
```

### vLLM 版本问题

作业文档原始方案要求在每次 evaluation 前，把当前训练中的 policy weights 热加载到已有 vLLM instance：

```python
llm.llm_engine.model_executor.driver_worker.model_runner.model
```

但本次服务器上的新版 vLLM 已经没有这个内部路径，实际报错包括：

```text
ImportError: cannot import name 'set_random_seed' from 'vllm.model_executor'
ModuleNotFoundError: No module named 'vllm.worker'
TypeError: EngineArgs.__init__() got an unexpected keyword argument 'device'
AttributeError: 'LLMEngine' object has no attribute 'model_executor'
RuntimeError: Engine core initialization failed
```

因此 `scripts/train_sft.py` 做了兼容处理：

- `set_random_seed` 改为多路径 fallback；
- `vllm.worker.worker` 不存在时跳过 profiling patch；
- 新版 vLLM 不支持 `device=` 时，通过临时 `CUDA_VISIBLE_DEVICES` 指定评估 GPU；
- 默认使用 `--vllm-sync-mode checkpoint`，即每次评估先把当前 policy 保存成临时 Hugging Face checkpoint，再用 vLLM 从该 checkpoint 启动。

`checkpoint` 模式比热加载慢一些，但在当前服务器上更稳定。

### smoke test

先跑一个很小的 filtered smoke test，验证训练、checkpoint、vLLM 评估、reward 统计、文件输出都能跑通：

```bash
cd /root/workspace/assignment5-alignment-main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python scripts/train_sft.py \
  --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --train-path /data/math/sft_gpt-oss-120b_filtered.jsonl \
  --val-path /data/math/val.jsonl \
  --output-dir outputs/sft/filtered_smoke_vllm \
  --run-name filtered_smoke_vllm \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --eval-backend vllm \
  --vllm-sync-mode checkpoint \
  --num-train-examples 16 \
  --max-steps 5 \
  --eval-every 5 \
  --eval-limit 16 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-seq-len 1024 \
  --normalize-constant 1024 \
  --eval-max-new-tokens 256 \
  --gradient-checkpointing \
  2>&1 | tee logs/filtered_smoke_vllm.log
```

本次 smoke test 已经成功，日志中出现：

```json
{"eval_step": 5, "num_examples": 16, "eval/format_accuracy": 0.75, "eval/answer_accuracy": 0.5625, "eval/reward": 0.5625}
```

检查输出：

```bash
find outputs/sft/filtered_smoke_vllm -maxdepth 3 -type f | sort
cat outputs/sft/filtered_smoke_vllm/eval/*.summary.json
```

如果看到 `ProcessGroupNCCL.cpp` 的 `destroy_process_group()` warning，可以先忽略；只要 vLLM 完成 generation，并且 summary JSON 正常输出，就说明链路是通的。

### filtered SFT 正式训练：稳妥版

这个版本更接近作业要求：训练中周期性跑完整 validation。缺点是耗时较长，因为每次 validation 有 5000 条，并且 `checkpoint` 模式会重新启动 vLLM。

```bash
cd /root/workspace/assignment5-alignment-main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=. \
MODEL_PATH=/data/a5-alignment/models/Qwen2.5-Math-1.5B \
DATA_DIR=/data/math \
VAL_PATH=/data/math/val.jsonl \
OUTPUT_ROOT=outputs/sft \
EVAL_BACKEND=vllm \
VLLM_SYNC_MODE=checkpoint \
TRAIN_DEVICE=cuda:0 \
EVAL_DEVICE=cuda:1 \
BATCH_SIZE=2 \
GRAD_ACCUM=4 \
LR=1e-5 \
MAX_SEQ_LEN=2048 \
NORMALIZE_CONSTANT=2048 \
EVAL_EVERY=200 \
EVAL_MAX_NEW_TOKENS=768 \
SAVE_EVERY=200 \
bash scripts/run_sft_experiments.sh filtered_full \
  2>&1 | tee logs/filtered_full.log
```

核心参数：

- `BATCH_SIZE=2`
- `GRAD_ACCUM=4`
- 有效 batch size 为 8，和保守版 `BATCH_SIZE=1, GRAD_ACCUM=8` 一致；
- `LR=1e-5`
- `MAX_SEQ_LEN=2048`
- `EVAL_EVERY=200`
- `EVAL_MAX_NEW_TOKENS=768`
- `TRAIN_DEVICE=cuda:0`
- `EVAL_DEVICE=cuda:1`
- `VLLM_SYNC_MODE=checkpoint`

filtered 数据约 3496 条，一轮训练大约 437 个 optimizer steps。`EVAL_EVERY=200` 通常会在 step 200、400 和最终 step 进行 validation。

### filtered SFT 正式训练：

如果只是为了粗略实现和快速观察趋势，可以训练过程中只评估 1000 条 validation，最后再单独做一次完整 validation。这样不完全等价于作业要求的完整 validation curve，但能明显缩短时间。

```bash
cd /root/workspace/assignment5-alignment-main
mkdir -p /data/outputs/sft logs

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python scripts/train_sft.py \
  --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --train-path /data/math/sft_gpt-oss-120b_filtered.jsonl \
  --val-path /data/math/val.jsonl \
  --output-dir /data/outputs/sft/filtered_full_fast_subset_eval \
  --run-name filtered_full_fast_subset_eval \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --eval-backend vllm \
  --vllm-sync-mode checkpoint \
  --num-train-examples -1 \
  --num-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-seq-len 2048 \
  --normalize-constant 2048 \
  --eval-every 200 \
  --eval-limit 1000 \
  --eval-max-new-tokens 768 \
  --save-every 200 \
  --gradient-checkpointing \
  2>&1 | tee logs/filtered_full_fast_subset_eval.log
```

### 运行管理

正式训练建议放在 `tmux` 里，避免 SSH 断开导致训练中断：

```bash
tmux new -s sft_filtered
```

恢复会话：

```bash
tmux attach -t sft_filtered
```

查看日志：

```bash
tail -f logs/filtered_full_fast_subset_eval.log
```

查看 GPU：

```bash
watch -n 5 nvidia-smi
```

如果中途出错，先检查是否有残留训练或 vLLM 进程：

```bash
ps -ef | grep -E "train_sft|vllm" | grep -v grep
```

必要时手动杀掉残留 PID：

```bash
kill -9 <PID>
```

### 检查训练结果

训练中和训练后都可以看 summary：

```bash
cat outputs/sft/filtered_full_fast_subset_eval/eval/*.summary.json
```

查看文件结构：

```bash
find outputs/sft/filtered_full_fast_subset_eval -maxdepth 3 -type f | sort | head -100
```

输出目录中关键文件：

```text
run_config.json
checkpoints/step_xxxxxx/
eval/step_xxxxxx.jsonl
eval/step_xxxxxx.summary.json
tmp_vllm_policy/step_xxxxxx/
```

最重要的指标：

```text
eval/answer_accuracy
eval/format_accuracy
eval/reward
```

其中 `eval/answer_accuracy` 可以作为 validation accuracy 报告。

### 最终完整 validation

如果训练时采用快速兴趣版，即 `--eval-limit 1000`，最后需要对完整 validation set 单独跑一次评估。先找到最后 checkpoint：

```bash
ls outputs/sft/filtered_full_fast_subset_eval/checkpoints | sort | tail -1
```

假设最后 checkpoint 是 `step_000437`，运行：

```bash
mkdir -p outputs/sft/filtered_full_fast_subset_eval/final_full_eval

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/math_baseline.py \
  --input-path /data/math/val.jsonl \
  --output-path outputs/sft/filtered_full_fast_subset_eval/final_full_eval/results.jsonl \
  --model-name-or-path outputs/sft/filtered_full_fast_subset_eval/checkpoints/step_000437 \
  --generation-mode vllm \
  --num-gpus 1 \
  --max-tokens 768 \
  2>&1 | tee logs/filtered_full_final_eval.log
```

查看最终完整 validation 结果：

```bash
cat outputs/sft/filtered_full_fast_subset_eval/final_full_eval/results.summary.json
```

报告：

- 训练过程曲线使用 1000 条 validation subset 追踪趋势；
- 最终 reported validation accuracy 使用完整 validation set；
- filtered 数据集大小约 3496 条；
- 因为没有原始 MATH 数据集，本实验使用 `/data/math` 中的替代数据集。

### 本次 4.3 SFT 实验结果汇总

本次实际采用的模型和数据集与参考博客
`https://huggingface.co/blog/garg-aayush/building-sft-from-ground-up`
基本一致：

- base model：`Qwen2.5-Math-1.5B`
- SFT 数据：`/data/math/sft_gpt-oss-120b_filtered.jsonl`
- validation 数据：`/data/math/val.jsonl`
- filtered 数据规模：约 3496 条 reasoning traces
- prompt 模板：`cs336_alignment/prompts/r1_zero.prompt`
- 评估 reward：`r1_zero_reward_fn`

由于 vLLM 新版本不兼容作业 starter code 中的热加载路径，本实验使用
`--vllm-sync-mode checkpoint`：每次评估前先保存当前 policy checkpoint，
再让 vLLM 从该 checkpoint 加载并生成。这个方法慢一些，但在当前服务器上稳定。

#### 实验 A：1 epoch filtered SFT

训练命令中的关键参数：

```text
num_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
effective batch size = 8
learning_rate = 1e-5
max_seq_len = 2048
eval_every = 200
eval_limit = 1000
eval_max_new_tokens = 768
```

由于 filtered 数据约 3496 条，1 epoch 大约对应 437 个 optimizer steps。
因此训练中得到 3 个 validation subset 曲线点：

| step | subset size | reward / answer accuracy | format accuracy |
| ---: | ---: | ---: | ---: |
| 200 | 1000 | 0.715 | 0.943 |
| 400 | 1000 | 0.734 | 0.945 |
| 437 | 1000 | 0.726 | 0.948 |

这个实验中，1000 条 validation subset 上的最好结果出现在 step 400：

```text
best subset reward accuracy = 0.734
best subset format accuracy around = 0.945
```

随后使用 `step_000437` checkpoint 在完整 5000 条 validation set 上重新评估：

```json
{
  "num_examples": 5000,
  "mean_reward": 0.5966,
  "mean_format_reward": 0.8824,
  "mean_answer_reward": 0.5966
}
```

因此，1 epoch filtered SFT 的完整验证集结果为：

```text
full validation answer/reward accuracy = 59.66%
full validation format accuracy = 88.24%
```

#### 实验 B：2 epoch filtered SFT 曲线实验

为了得到更像样的训练曲线，第二次实验改为 2 epochs，并提高评估频率：

```text
num_epochs = 2
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
effective batch size = 8
learning_rate = 1e-5
max_seq_len = 2048
eval_every = 150
eval_limit = 1000
eval_max_new_tokens = 768
```

预期完整训练约 874 optimizer steps，但由于服务器 SSH/实例中途断开，
训练实际跑到约 step 806，最后保留下来的 eval checkpoint 到 step 750。
已经得到的 validation subset 曲线点如下：

| step | subset size | reward / answer accuracy | format accuracy |
| ---: | ---: | ---: | ---: |
| 150 | 1000 | 0.692 | 0.944 |
| 300 | 1000 | 0.728 | 0.940 |
| 450 | 1000 | 0.734 | 0.960 |
| 600 | 1000 | 0.719 | 0.958 |
| 750 | 1000 | 0.724 | 0.960 |

这个实验的最高 subset reward accuracy 同样是：

```text
best subset reward accuracy = 0.734 at step 450
```

2 epoch 曲线没有明显超过 1 epoch，反而在后半段略有回落。可能原因包括：

- filtered 数据规模不大，1 epoch 后已经接近饱和；
- 训练中只使用 1000 条 validation subset 评估，曲线存在抽样波动；
- learning rate 使用 cosine decay，后期学习率已经非常小；
- 第二次实验没有完成最后 full validation，因此不能和 1 epoch 的完整验证结果完全等价比较。

#### 曲线图与本地结果文件

两次实验的曲线图已经整理到本地：

```text
sft-outputs/4.3_sft/figures/sft_two_runs_accuracy_curves.svg
```

对应的 CSV 和摘要：

```text
sft-outputs/4.3_sft/figures/sft_two_runs_accuracy_curves.csv
sft-outputs/4.3_sft/figures/sft_two_runs_summary.md
```
![alt text](image.png)
图中含义：

- 左图：reward / answer accuracy；
- 右图：format accuracy；
- 蓝线：1 epoch，在 1000 条 validation subset 上评估；
- 绿线：2 epoch 实验中断前，在 1000 条 validation subset 上评估；
- 星标：1 epoch `step_000437` checkpoint 的完整 validation 结果；
- 灰色虚线：参考博客中的 Qwen2.5-Math-1.5B baseline，reward 约 2.9%，format 约 14.4%。
- 由此也可见，基本300个step的时候两种奖励就已经收敛。

#### 结论

在 `/data/math` 替代数据集上，filtered reasoning SFT 明显提升了
Qwen2.5-Math-1.5B 的格式遵循能力和答案正确率。1 epoch 后，在 1000 条
validation subset 上可以达到约 72% 到 73% 的 reward accuracy；在完整
5000 条 validation set 上，最终得到：

```text
answer / reward accuracy = 59.66%
format accuracy = 88.24%
```

相比参考（https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/blob/main/sft-reason/README.md）中的 baseline（reward 约 2.9%，format 约 14.4%），SFT 后提升非常明显。
不过，本实验的 subset accuracy 明显高于 full validation accuracy，说明 1000 条 subset
可能比完整验证集更容易。最终 full validation 为 59.66% 。训练时间也不太长，好像是1个小时左右（时间有点长了记不清了，sorry~）
