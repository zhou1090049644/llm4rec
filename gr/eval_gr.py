import argparse
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from gr.train_gr import parse_args_from_json
    from gr.utils import build_semantic_tokens, load_item2token_dict
except ModuleNotFoundError:
    from train_gr import parse_args_from_json
    from utils import build_semantic_tokens, load_item2token_dict


HIST_START = "<|hist_clk_start|>"
HIST_END = "<|hist_clk_end|>"


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        timeout_hours = float(os.environ.get("GR_EVAL_DISTRIBUTED_TIMEOUT_HOURS", "12"))
        dist.init_process_group(backend=backend, timeout=timedelta(hours=timeout_hours))
    return rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def stream_user_sequences(json_path):
    """Stream legacy user sequences or explicit history-target JSONL samples."""
    with open(json_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line in ("{", "}"):
                continue
            if line.endswith(","):
                line = line[:-1]
            if line.startswith("{"):
                sample = json.loads(line)
                if "history" in sample and "target" in sample:
                    history = [str(item) for item in sample["history"]]
                    target = str(sample["target"])
                    user_id = str(sample.get("user_id", sample.get("sample_id", "")))
                    sample_id = str(sample.get("sample_id", user_id))
                    if history and target:
                        yield sample_id, user_id, history, target
                else:
                    user_id, item_sequence = next(iter(sample.items()))
                    if len(item_sequence) >= 2:
                        yield str(user_id), str(user_id), [str(item) for item in item_sequence[:-1]], str(item_sequence[-1])
            else:
                user_data = json.loads("{" + line + "}")
                user_id, item_sequence = next(iter(user_data.items()))
                if len(item_sequence) >= 2:
                    yield str(user_id), str(user_id), [str(item) for item in item_sequence[:-1]], str(item_sequence[-1])


def load_candidate_items(candidate_path):
    if not candidate_path:
        return None

    candidate_path = Path(candidate_path)
    candidates = set()
    with candidate_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                sample = json.loads(line)
                if "item_id" in sample:
                    candidates.add(str(sample["item_id"]))
                elif "item" in sample:
                    candidates.add(str(sample["item"]))
                else:
                    candidates.update(str(value) for value in sample.values())
            elif line.startswith("["):
                candidates.update(str(item) for item in json.loads(line))
            else:
                candidates.add(line.split()[0])
    print("length of candidate_items is:", len(candidates))
    return candidates


def build_token2items(item2token_dict):
    token2items = defaultdict(list)
    for item_id, token in item2token_dict.items():
        token2items[token].append(str(item_id))
    return dict(token2items)


def load_eval_model_and_tokenizer(model_args, checkpoint_path):
    tokenizer_path = model_args.tokenizer_path or checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": build_semantic_tokens(model_args.se_id_space_width)}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


def make_prompt(history_items, item2token_dict, max_seq_length):
    history_tokens = [
        item2token_dict[item_id]
        for item_id in history_items
        if item_id in item2token_dict
    ][-max_seq_length:]
    if not history_tokens:
        return None
    return HIST_START + "".join(history_tokens) + HIST_END


def decode_generated_tokens(tokenizer, generated_ids, prompt_len):
    generated_part = generated_ids[prompt_len:]
    decoded = tokenizer.decode(
        generated_part,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    for token in (tokenizer.eos_token, tokenizer.pad_token):
        if token:
            decoded = decoded.replace(token, "")
    return decoded.strip()


def generated_token_to_items(generated_token, token2items):
    if generated_token in token2items:
        return token2items[generated_token]

    # Some tokenizers may decode spaces around adjacent special tokens.
    compact_token = "".join(generated_token.split())
    return token2items.get(compact_token, [])


def collect_recommendations(decoded_tokens, token2items, history_set, candidate_top_k, candidate_items=None):
    recommendations = []
    seen = set()
    for generated_token in decoded_tokens:
        for item_id in generated_token_to_items(generated_token, token2items):
            if item_id in seen or item_id in history_set:
                continue
            if candidate_items is not None and item_id not in candidate_items:
                continue
            recommendations.append(item_id)
            seen.add(item_id)
            if len(recommendations) >= candidate_top_k:
                return recommendations
    return recommendations


@torch.no_grad()
def generate_recommendations(
    model,
    tokenizer,
    prompt,
    token2items,
    history_set,
    generate_num,
    max_new_tokens,
    candidate_top_k,
    candidate_items,
    device,
):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=generate_num,
        num_return_sequences=generate_num,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded_tokens = [
        decode_generated_tokens(tokenizer, output_ids, prompt_len)
        for output_ids in outputs
    ]
    return collect_recommendations(decoded_tokens, token2items, history_set, candidate_top_k, candidate_items)


@torch.no_grad()
def generate_recommendations_batch(
    model,
    tokenizer,
    prompts,
    token2items,
    history_sets,
    generate_num,
    max_new_tokens,
    candidate_top_k,
    candidate_items,
    device,
):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=generate_num,
        num_return_sequences=generate_num,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    batch_outputs = outputs.reshape(len(prompts), generate_num, outputs.shape[-1])
    batch_recommendations = []
    for output_group, history_set in zip(batch_outputs, history_sets):
        decoded_tokens = [
            decode_generated_tokens(tokenizer, output_ids, prompt_len)
            for output_ids in output_group
        ]
        batch_recommendations.append(
            collect_recommendations(decoded_tokens, token2items, history_set, candidate_top_k, candidate_items)
        )
    return batch_recommendations


def reduce_counts(values, world_size, device):
    counts = torch.tensor(values, dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return [int(value) for value in counts.cpu().tolist()]


def warmup_distributed(world_size, device):
    if world_size <= 1:
        return
    warmup = torch.zeros(1, dtype=torch.long, device=device)
    dist.all_reduce(warmup, op=dist.ReduceOp.SUM)


def write_rank_predictions(output_path, rank, result_rows):
    rank_output_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.jsonl")
    with rank_output_path.open("w", encoding="utf-8") as fp:
        for row in result_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rank_output_path


def evaluate(args):
    rank, local_rank, world_size = init_distributed()
    model_args, data_args, training_args = parse_args_from_json(args.config)
    metric_ks = sorted({int(k.strip()) for k in args.metric_ks.split(",") if k.strip()})
    if args.metric_k not in metric_ks:
        metric_ks.append(args.metric_k)
        metric_ks = sorted(set(metric_ks))
    max_metric_k = max(metric_ks)

    user_cache_path = Path(os.environ.get("USER_CACHE_PATH", Path.cwd() / "outputs"))
    train_ckpt_path = Path(os.environ.get("TRAIN_CKPT_PATH", user_cache_path / "gr" / "checkpoints"))
    output_dir = args.output_dir or training_args.output_dir
    checkpoint_path = Path(args.checkpoint_path or model_args.checkpoint_path or train_ckpt_path / output_dir)
    test_file = Path(args.test_file or data_args.test_file or data_args.train_file)
    if not test_file.is_absolute():
        test_file = user_cache_path / test_file
    item2token_path = user_cache_path / data_args.item2token_dict
    candidate_path = Path(args.candidate_file or data_args.candidate_file) if (args.candidate_file or data_args.candidate_file) else None
    if candidate_path is not None and not candidate_path.is_absolute():
        candidate_path = user_cache_path / candidate_path

    item2token_dict = load_item2token_dict(str(item2token_path))
    token2items = build_token2items(item2token_dict)
    candidate_items = load_candidate_items(str(candidate_path)) if candidate_path else None
    model, tokenizer = load_eval_model_and_tokenizer(model_args, str(checkpoint_path))

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    warmup_distributed(world_size, device)

    total = 0
    skipped = 0
    target_not_in_candidate = 0
    hit_counts = {k: 0 for k in metric_ks}
    result_rows = []
    eval_batch_size = max(1, args.eval_batch_size)
    batch_rows = []

    def flush_batch():
        nonlocal total, target_not_in_candidate, result_rows, batch_rows
        if not batch_rows:
            return

        prompts = [row["prompt"] for row in batch_rows]
        history_sets = [row["history_set"] for row in batch_rows]
        batch_recommendations = generate_recommendations_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token2items=token2items,
            history_sets=history_sets,
            generate_num=args.generate_num,
            max_new_tokens=data_args.token_depth,
            candidate_top_k=max(args.candidate_top_k, max_metric_k),
            candidate_items=candidate_items,
            device=device,
        )

        for row, recommendations in zip(batch_rows, batch_recommendations):
            total += 1
            if candidate_items is not None and row["target_item"] not in candidate_items:
                target_not_in_candidate += 1

            row_hits = {}
            for metric_k in metric_ks:
                hit = int(row["target_item"] in recommendations[:metric_k])
                hit_counts[metric_k] += hit
                row_hits[f"hit@{metric_k}"] = hit

            if args.save_predictions:
                result_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "user_id": row["user_id"],
                        "target_item": row["target_item"],
                        "recommendations": recommendations,
                        "candidate_filtered": candidate_items is not None,
                        **row_hits,
                    }
                )
        batch_rows = []

    iterator = enumerate(stream_user_sequences(test_file))
    progress = tqdm(
        iterator,
        desc=f"Evaluating GR rank {rank}/{world_size}",
        disable=not is_main_process(rank),
    )
    for row_idx, (sample_id, user_id, history_items, target_item) in progress:
        if args.max_eval_rows is not None and row_idx >= args.max_eval_rows:
            break

        if row_idx % world_size != rank:
            continue

        if target_item not in item2token_dict:
            skipped += 1
            continue

        prompt = make_prompt(history_items, item2token_dict, data_args.max_seq_length)
        if prompt is None:
            skipped += 1
            continue

        batch_rows.append(
            {
                "sample_id": sample_id,
                "user_id": user_id,
                "target_item": target_item,
                "prompt": prompt,
                "history_set": set(history_items),
            }
        )
        if len(batch_rows) >= eval_batch_size:
            flush_batch()

    flush_batch()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_files = []
    if args.save_predictions:
        rank_prediction_path = write_rank_predictions(output_path, rank, result_rows)
        prediction_files = [str(rank_prediction_path)]
        if world_size > 1:
            gathered_prediction_files = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_prediction_files, str(rank_prediction_path))
            prediction_files = gathered_prediction_files

    reduced_values = reduce_counts(
        [total, skipped, target_not_in_candidate] + [hit_counts[k] for k in metric_ks],
        world_size,
        device,
    )
    total, skipped, target_not_in_candidate = reduced_values[:3]
    reduced_hits = dict(zip(metric_ks, reduced_values[3:]))

    metrics = {
        "total_users": total,
        "skipped_users": skipped,
        "target_not_in_candidate": target_not_in_candidate,
        "generate_num": args.generate_num,
        "candidate_top_k": max(args.candidate_top_k, max_metric_k),
        "candidate_file": str(candidate_path) if candidate_path else None,
        "candidate_size": len(candidate_items) if candidate_items is not None else None,
        "eval_batch_size": eval_batch_size,
        "max_eval_rows": args.max_eval_rows,
        "world_size": world_size,
    }
    for metric_k in metric_ks:
        value = reduced_hits[metric_k] / total if total else 0.0
        metrics[f"hit@{metric_k}"] = value
        metrics[f"recall@{metric_k}"] = value

    if is_main_process(rank):
        payload = {"metrics": metrics}
        if args.save_predictions:
            payload["prediction_files"] = prediction_files
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Wrote evaluation results to {output_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GR with leave-one-last-item-out generation.")
    parser.add_argument("--config", type=str, default="gr/gr_train.json")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="outputs/gr/eval_result.json")
    parser.add_argument("--generate_num", type=int, default=100)
    parser.add_argument("--candidate_top_k", type=int, default=20)
    parser.add_argument("--metric_k", type=int, default=20)
    parser.add_argument("--metric_ks", type=str, default="5,10,20")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_eval_rows", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--candidate_file", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
