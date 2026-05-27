"""
No-tool ICRL-lite GRPO 实验入口。

这个脚本不修改 cs336_alignment/grpo.py 的核心数学实现；它只改变 rollout
阶段的 prompt 构造：

- rollout: 按 step schedule 在 r1_zero prompt 前加入 few-shot demos；
- old_log_probs / policy_log_probs: 使用 rollout 时实际构造的完整 prompt；
- eval: 始终使用原始 zero-shot r1_zero prompt；
- reward / advantage / loss / microbatch train step / optimizer: 全部复用现有实现。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cs336_alignment.grpo import (  # noqa: E402
    GRPOConfig,
    _compute_old_log_probs,
    _evaluate_grpo_policy,
    _generate_rollouts_transformers,
    _normalize_math_examples,
    _tokenize_grpo_batch,
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)
from cs336_alignment.icrl import (  # noqa: E402
    build_icrl_prompt,
    load_icrl_demos,
    num_icrl_shots_for_step,
    select_icrl_demos_for_step,
)
from cs336_alignment.sft import get_response_log_probs  # noqa: E402
from train_grpo import (  # noqa: E402
    build_config,
    build_rollout_and_eval_fns,
    configure_data_disk_env,
    init_policy,
    init_tokenizer,
    normalize_math_records,
    parse_args as parse_grpo_args,
    read_records,
    resolve_prompt_and_reward,
    run_final_full_eval,
    save_outputs,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    复用 train_grpo.py 的参数，并新增 ICRL demo bank 路径。

    这样 baseline 与 ICRL-lite 可以保持完全相同的训练参数，只替换 rollout
    prompt 构造。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--icrl-demo-path", default="data/icrl/fewshot_r1_zero.jsonl")
    known_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    args = parse_grpo_args()
    args.icrl_demo_path = known_args.icrl_demo_path
    return args


def grpo_train_loop_icrl_lite(
    *,
    policy: torch.nn.Module,
    reward_fn,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    config: GRPOConfig,
    r1_zero_template: str,
    icrl_demo_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    ICRL-lite 专用训练循环。

    这段循环基本沿用现有 grpo_train_loop 的工程流程，唯一关键差异是：
    rollout prompt 根据当前 step 构造为 few-shot prompt，并且后续 logprob
    tokenization 使用同一份 rollout prompt。
    """
    tokenizer = kwargs.get("tokenizer")
    optimizer = kwargs.get("optimizer")
    rollout_fn = kwargs.get("rollout_fn")
    eval_fn = kwargs.get("eval_fn")
    log_fn = kwargs.get("log_fn")
    seed = int(kwargs.get("seed", 0))
    top_p = float(kwargs.get("top_p", 1.0))
    eval_every = int(kwargs.get("eval_every", 10))
    eval_limit = int(kwargs.get("eval_limit", 1024))
    eval_max_new_tokens = int(kwargs.get("eval_max_new_tokens", config.sampling_max_tokens))
    max_seq_len = kwargs.get("max_seq_len")
    num_logged_rollouts = int(kwargs.get("num_logged_rollouts", 4))

    if tokenizer is None:
        raise ValueError("tokenizer is required for ICRL-lite training.")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.train_batch_size % config.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    if config.rollout_batch_size % config.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size

    if config.train_batch_size < config.group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size.")
    if config.train_batch_size > config.rollout_batch_size:
        raise ValueError("train_batch_size must be less than or equal to rollout_batch_size.")
    if config.rollout_batch_size % micro_train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by micro_train_batch_size.")

    n_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size

    device = kwargs.get("device")
    if device is None:
        device = next(policy.parameters()).device
    device = torch.device(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

    train_records = _normalize_math_examples(
        train_examples,
        r1_zero_template,
        require_answer=True,
    )
    validation_records = _normalize_math_examples(
        validation_examples,
        r1_zero_template,
        require_answer=True,
    )
    if not train_records:
        raise ValueError("No usable train examples after normalization.")

    demos = load_icrl_demos(icrl_demo_path)
    rng = random.Random(seed)
    metrics_history: list[dict[str, float]] = []
    rollout_examples: list[dict[str, Any]] = []
    optimizer_step = 0

    derived = {
        "micro_train_batch_size": micro_train_batch_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "n_microbatches_per_rollout_batch": n_microbatches_per_rollout_batch,
        "icrl_demo_path": str(icrl_demo_path),
        "icrl_schedule": "1-10:3shot,11-20:2shot,21-30:1shot,31-50:0shot",
    }

    def emit(metrics: dict[str, float]) -> None:
        metrics_history.append(metrics)
        if log_fn is not None:
            log_fn(metrics)
        else:
            logger.info("%s", metrics)

    for grpo_step in range(1, config.n_grpo_steps + 1):
        if n_prompts_per_rollout_batch <= len(train_records):
            prompt_records = rng.sample(train_records, n_prompts_per_rollout_batch)
        else:
            prompt_records = rng.choices(train_records, k=n_prompts_per_rollout_batch)

        selected_demos = select_icrl_demos_for_step(demos, grpo_step)
        num_shots = num_icrl_shots_for_step(grpo_step)
        rollout_prompts = [
            build_icrl_prompt(
                question=record["problem"],
                demos=selected_demos,
                r1_zero_template=r1_zero_template,
            )
            for record in prompt_records
        ]
        ground_truths = [record["answer"] for record in prompt_records]

        if rollout_fn is None:
            rollout_responses = _generate_rollouts_transformers(
                policy=policy,
                tokenizer=tokenizer,
                prompts=rollout_prompts,
                group_size=config.group_size,
                config=config,
                device=device,
                top_p=top_p,
            )
        else:
            generated = rollout_fn(
                policy=policy,
                tokenizer=tokenizer,
                prompts=rollout_prompts,
                group_size=config.group_size,
                config=config,
                step=grpo_step,
            )
            if generated and isinstance(generated[0], list):
                rollout_responses = [response for group in generated for response in group]
            else:
                rollout_responses = list(generated)

        expected_rollouts = n_prompts_per_rollout_batch * config.group_size
        if len(rollout_responses) != expected_rollouts:
            raise ValueError(f"Expected {expected_rollouts} rollout responses, got {len(rollout_responses)}.")

        repeated_rollout_prompts = [prompt for prompt in rollout_prompts for _ in range(config.group_size)]
        repeated_ground_truths = [answer for answer in ground_truths for _ in range(config.group_size)]

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        tokenized = _tokenize_grpo_batch(
            tokenizer=tokenizer,
            prompts=repeated_rollout_prompts,
            responses=rollout_responses,
            max_seq_len=max_seq_len,
        )
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        response_mask = tokenized["response_mask"]

        old_log_probs = None
        if config.loss_type == "grpo_clip":
            old_log_probs = _compute_old_log_probs(
                policy=policy,
                input_ids=input_ids,
                labels=labels,
                microbatch_size=micro_train_batch_size,
                device=device,
            )

        advantages = advantages.unsqueeze(1)
        raw_rewards = raw_rewards.unsqueeze(1)
        all_indices = list(range(config.rollout_batch_size))

        for rollout_epoch in range(config.epochs_per_rollout_batch):
            epoch_indices = all_indices.copy()
            rng.shuffle(epoch_indices)
            train_indices = epoch_indices[: config.train_batch_size]

            optimizer.zero_grad(set_to_none=True)
            microbatch_metadata: dict[str, Tensor] = {}
            entropy_values: list[Tensor] = []

            for start in range(0, config.train_batch_size, micro_train_batch_size):
                batch_indices = train_indices[start : start + micro_train_batch_size]
                batch_input_ids = input_ids[batch_indices].to(device)
                batch_labels = labels[batch_indices].to(device)
                batch_response_mask = response_mask[batch_indices].to(device)

                output = get_response_log_probs(
                    model=policy,
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                    return_token_entropy=True,
                )

                old_log_probs_batch = None
                if old_log_probs is not None:
                    old_log_probs_batch = old_log_probs[batch_indices].to(device)

                _, microbatch_metadata = grpo_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=batch_response_mask,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    loss_type=config.loss_type,
                    raw_rewards=raw_rewards[batch_indices].to(device),
                    advantages=advantages[batch_indices].to(device),
                    old_log_probs=old_log_probs_batch,
                    cliprange=kwargs.get("cliprange", 0.2),
                )

                entropy_values.append(
                    masked_mean(
                        output["token_entropy"].detach(),
                        batch_response_mask,
                        dim=None,
                    ).detach()
                )

            grad_norm = clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_step += 1

            metrics: dict[str, float] = {
                "grpo_step": float(grpo_step),
                "rollout_epoch": float(rollout_epoch),
                "optimizer_step": float(optimizer_step),
                "icrl/num_shots": float(num_shots),
                "train/loss": float(microbatch_metadata["scaled_loss"].item()),
                "train/unscaled_loss": float(microbatch_metadata["unscaled_loss"].item()),
                "train/grad_norm": float(grad_norm),
                "train/token_entropy": float(torch.stack(entropy_values).mean().item()),
                "train/num_response_tokens": float(microbatch_metadata["num_response_tokens"].item()),
                "rollout/reward": float(reward_metadata["reward_mean"]),
                "rollout/reward_std": float(reward_metadata["reward_std"]),
                "rollout/format_reward": float(reward_metadata["format_reward_mean"]),
                "rollout/answer_reward": float(reward_metadata["answer_reward_mean"]),
            }
            if "clip_fraction" in microbatch_metadata:
                metrics["train/clip_fraction"] = float(microbatch_metadata["clip_fraction"].item())
            emit(metrics)

        if len(rollout_examples) < num_logged_rollouts:
            for response, answer in zip(rollout_responses, repeated_ground_truths):
                rollout_examples.append(
                    {
                        "grpo_step": grpo_step,
                        "icrl_num_shots": num_shots,
                        "response": response,
                        "ground_truth": answer,
                        "reward": reward_fn(response, answer),
                    }
                )
                if len(rollout_examples) >= num_logged_rollouts:
                    break

        if eval_every > 0 and grpo_step % eval_every == 0:
            if eval_fn is not None:
                eval_metrics = eval_fn(
                    policy=policy,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    validation_examples=validation_records,
                    config=config,
                    step=grpo_step,
                )
            else:
                eval_metrics = _evaluate_grpo_policy(
                    policy=policy,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    validation_records=validation_records,
                    limit=eval_limit,
                    max_new_tokens=eval_max_new_tokens,
                    device=device,
                )
            if eval_metrics:
                emit(
                    {
                        "grpo_step": float(grpo_step),
                        "optimizer_step": float(optimizer_step),
                        "icrl/num_shots": float(num_shots),
                        **{key: float(value) for key, value in eval_metrics.items()},
                    }
                )

    return {
        "metrics": metrics_history,
        "rollout_examples": rollout_examples,
        "config": config,
        "derived": derived,
        "optimizer_steps": optimizer_step,
    }


def main() -> None:
    args = parse_args()
    configure_data_disk_env(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))

    prompt_template, reward_fn, prompt_path = resolve_prompt_and_reward(args)
    config = build_config(args)

    run_config = vars(args) | {
        "resolved_prompt_path": prompt_path,
        "reward_fn": reward_fn.__name__,
        "grpo_config": config.__dict__,
        "icrl_lite": {
            "demo_path": args.icrl_demo_path,
            "schedule": "step 1-10: 3-shot; step 11-20: 2-shot; step 21-30: 1-shot; step 31-50: 0-shot",
            "rollout_prompt_only": True,
            "eval_uses_zero_shot": True,
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_records = normalize_math_records(
        read_records(args.train_path),
        prompt_template=prompt_template,
    )
    val_records = normalize_math_records(
        read_records(args.val_path),
        prompt_template=prompt_template,
    )
    logger.info("Loaded %d train records and %d validation records.", len(train_records), len(val_records))

    tokenizer = init_tokenizer(args.model_path)
    policy = init_policy(
        model_path=args.model_path,
        device=args.train_device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_fn, eval_fn = build_rollout_and_eval_fns(args)
    metrics_path = output_dir / "metrics.jsonl"

    def log_fn(metrics: dict[str, float]) -> None:
        logger.info("%s", json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    result = grpo_train_loop_icrl_lite(
        policy=policy,
        reward_fn=reward_fn,
        train_examples=train_records,
        validation_examples=val_records,
        config=config,
        r1_zero_template=prompt_template,
        icrl_demo_path=args.icrl_demo_path,
        tokenizer=tokenizer,
        optimizer=optimizer,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
        log_fn=log_fn,
        seed=args.seed,
        device=args.train_device,
        eval_every=args.eval_every,
        eval_limit=args.eval_limit,
        eval_max_new_tokens=args.eval_max_new_tokens,
        max_seq_len=args.max_seq_len,
        num_logged_rollouts=args.num_logged_rollouts,
        cliprange=args.cliprange,
        top_p=args.top_p,
    )

    # val_records 保持 zero-shot prompt，因此 final full eval 不加 few-shot。
    run_final_full_eval(args, policy, tokenizer, reward_fn, val_records)
    save_outputs(args, result, policy, tokenizer)
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
