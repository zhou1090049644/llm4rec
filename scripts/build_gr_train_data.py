import argparse
import json
import pickle
from pathlib import Path
from typing import List

import pyarrow.dataset as ds
from tqdm import tqdm


def convert_item2token(input_file: Path, output_dir: Path, id_offset: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / input_file.name
    with input_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue
            item_id, token = line.split("\t", 1)
            dst.write(f"{int(item_id) + id_offset}\t{token}\n")
    return output_file


def write_legacy_train_data(seq_dir: Path, output_file: Path, max_items_per_user: int, batch_size: int) -> None:
    dataset = ds.dataset(str(seq_dir), format="parquet")
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)
    total_rows = dataset.count_rows()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept_users = 0
    skipped_users = 0
    first = True

    with output_file.open("w", encoding="utf-8") as fp:
        fp.write("{\n")
        with tqdm(total=total_rows, desc="Building train_data.json", unit=" users") as pbar:
            for batch in scanner.to_batches():
                user_ids = batch.column("user_id").to_pylist()
                seqs = batch.column("seq").to_pylist()
                for user_id, seq in zip(user_ids, seqs):
                    items = [
                        (int(event["timestamp"] or 0), str(int(event["item_id"])))
                        for event in seq
                        if event is not None and event.get("item_id") is not None
                    ]
                    if len(items) < 2:
                        skipped_users += 1
                        continue
                    items.sort(key=lambda x: x[0])
                    item_ids = [item_id for _, item_id in items[-max_items_per_user:]]
                    if len(item_ids) < 2:
                        skipped_users += 1
                        continue

                    if not first:
                        fp.write(",\n")
                    first = False
                    fp.write(json.dumps(str(user_id), ensure_ascii=False))
                    fp.write(": ")
                    fp.write(json.dumps(item_ids, ensure_ascii=False))
                    kept_users += 1
                pbar.update(batch.num_rows)
        fp.write("\n}\n")

    print(f"Wrote {output_file}")
    print(f"Users kept: {kept_users}")
    print(f"Users skipped: {skipped_users}")


def iter_user_item_ids(seq_dir: Path, max_items_per_user: int, batch_size: int):
    dataset = ds.dataset(str(seq_dir), format="parquet")
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)
    total_rows = dataset.count_rows()

    with tqdm(total=total_rows, desc="Reading user sequences", unit=" users") as pbar:
        for batch in scanner.to_batches():
            user_ids = batch.column("user_id").to_pylist()
            seqs = batch.column("seq").to_pylist()
            for user_id, seq in zip(user_ids, seqs):
                items = [
                    (int(event["timestamp"] or 0), str(int(event["item_id"])))
                    for event in seq
                    if event is not None and event.get("item_id") is not None
                ]
                if len(items) < 2:
                    continue
                items.sort(key=lambda x: x[0])
                item_ids = [item_id for _, item_id in items[-max_items_per_user:]]
                if len(item_ids) >= 2:
                    yield str(user_id), item_ids
            pbar.update(batch.num_rows)


def write_jsonl_record(fp, sample_id: str, user_id: str, history: List[str], target: str) -> None:
    fp.write(
        json.dumps(
            {
                "sample_id": sample_id,
                "user_id": user_id,
                "history": history,
                "target": target,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def write_gr_splits(
    seq_dir: Path,
    train_output: Path,
    valid_output: Path,
    test_output: Path,
    max_seq_length: int,
    max_items_per_user: int,
    batch_size: int,
    min_items_for_valid_test: int,
) -> None:
    for output_file in (train_output, valid_output, test_output):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    kept_users = 0
    skipped_users = 0
    train_samples = 0
    valid_samples = 0
    test_samples = 0

    with (
        train_output.open("w", encoding="utf-8") as train_fp,
        valid_output.open("w", encoding="utf-8") as valid_fp,
        test_output.open("w", encoding="utf-8") as test_fp,
    ):
        for user_id, item_ids in iter_user_item_ids(seq_dir, max_items_per_user, batch_size):
            if len(item_ids) < min_items_for_valid_test:
                skipped_users += 1
                continue

            kept_users += 1
            train_end = len(item_ids) - 2
            for target_idx in range(1, train_end):
                history = item_ids[:target_idx][-max_seq_length:]
                target = item_ids[target_idx]
                write_jsonl_record(train_fp, f"{user_id}#train#{target_idx}", user_id, history, target)
                train_samples += 1

            valid_history = item_ids[: len(item_ids) - 2][-max_seq_length:]
            valid_target = item_ids[-2]
            write_jsonl_record(valid_fp, f"{user_id}#valid", user_id, valid_history, valid_target)
            valid_samples += 1

            test_history = item_ids[: len(item_ids) - 1][-max_seq_length:]
            test_target = item_ids[-1]
            write_jsonl_record(test_fp, f"{user_id}#test", user_id, test_history, test_target)
            test_samples += 1

    print(f"Wrote train split: {train_output} ({train_samples} samples)")
    print(f"Wrote valid split: {valid_output} ({valid_samples} samples)")
    print(f"Wrote test split: {test_output} ({test_samples} samples)")
    print(f"Users kept for split: {kept_users}")
    print(f"Users skipped for split: {skipped_users}")


def load_item_indexer(indexer_file: Path):
    with indexer_file.open("rb") as fp:
        indexer = pickle.load(fp)
    if not isinstance(indexer, dict) or "i" not in indexer:
        raise ValueError(f"Expected {indexer_file} to contain an item indexer under key 'i'.")
    return indexer["i"]


def write_candidate_items(
    candidate_dir: Path,
    output_file: Path,
    batch_size: int,
    indexer_file: Path,
    candidate_id_space: str,
) -> None:
    dataset = ds.dataset(str(candidate_dir), format="parquet")
    scanner = dataset.scanner(columns=["item_id"], batch_size=batch_size)
    total_rows = dataset.count_rows()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    item_indexer = load_item_indexer(indexer_file) if candidate_id_space == "internal" else None

    count = 0
    skipped = 0
    with output_file.open("w", encoding="utf-8") as fp:
        with tqdm(total=total_rows, desc="Writing candidate items", unit=" items") as pbar:
            for batch in scanner.to_batches():
                item_ids = batch.column("item_id").to_pylist()
                for item_id in item_ids:
                    if item_id is None:
                        continue
                    raw_item_id = int(item_id)
                    if item_indexer is not None:
                        mapped_item_id = item_indexer.get(raw_item_id)
                        if mapped_item_id is None:
                            skipped += 1
                            continue
                        fp.write(f"{mapped_item_id}\n")
                    else:
                        fp.write(f"{raw_item_id}\n")
                    count += 1
                pbar.update(batch.num_rows)

    print(f"Wrote candidate items: {output_file} ({count} items)")
    if skipped:
        print(f"Candidate items skipped because they were missing from indexer: {skipped}")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build GR train_data.json from sasrec seq parquet.")
    parser.add_argument("--seq_dir", type=Path, default=project_root / "data" / "seq")
    parser.add_argument("--output_file", type=Path, default=project_root / "outputs" / "train_data.json")
    parser.add_argument("--train_output", type=Path, default=project_root / "outputs" / "gr_train_split.jsonl")
    parser.add_argument("--valid_output", type=Path, default=project_root / "outputs" / "gr_valid_split.jsonl")
    parser.add_argument("--test_output", type=Path, default=project_root / "outputs" / "gr_test_split.jsonl")
    parser.add_argument("--candidate_dir", type=Path, default=project_root / "data" / "candidate")
    parser.add_argument("--candidate_output", type=Path, default=project_root / "outputs" / "candidate_items.txt")
    parser.add_argument("--indexer_file", type=Path, default=project_root / "data" / "indexer.pkl")
    parser.add_argument("--candidate_id_space", choices=["internal", "raw"], default="internal")
    parser.add_argument(
        "--item2token_input",
        type=Path,
        default=project_root / "outputs" / "rqvae" / "emb_infer" / "sinkhorn50" / "worker_0_output.txt",
    )
    parser.add_argument(
        "--item2token_output_dir",
        type=Path,
        default=project_root / "outputs" / "rqvae" / "emb_infer" / "sinkhorn50_gr",
    )
    parser.add_argument("--id_offset", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--max_items_per_user", type=int, default=102)
    parser.add_argument("--min_items_for_valid_test", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--write_legacy_train_data", action="store_true")
    parser.add_argument("--skip_item2token", action="store_true")
    parser.add_argument("--skip_splits", action="store_true")
    parser.add_argument("--skip_candidate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_item2token:
        converted = convert_item2token(args.item2token_input, args.item2token_output_dir, args.id_offset)
        print(f"Wrote {converted}")
    if args.write_legacy_train_data:
        write_legacy_train_data(args.seq_dir, args.output_file, args.max_items_per_user, args.batch_size)
    if not args.skip_splits:
        write_gr_splits(
            seq_dir=args.seq_dir,
            train_output=args.train_output,
            valid_output=args.valid_output,
            test_output=args.test_output,
            max_seq_length=args.max_seq_length,
            max_items_per_user=args.max_items_per_user,
            batch_size=args.batch_size,
            min_items_for_valid_test=args.min_items_for_valid_test,
        )
    if not args.skip_candidate:
        write_candidate_items(
            candidate_dir=args.candidate_dir,
            output_file=args.candidate_output,
            batch_size=args.batch_size,
            indexer_file=args.indexer_file,
            candidate_id_space=args.candidate_id_space,
        )


if __name__ == "__main__":
    main()
