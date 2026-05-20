import argparse
import json
from pathlib import Path

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


def write_train_data(seq_dir: Path, output_file: Path, max_items_per_user: int, batch_size: int) -> None:
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


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build GR train_data.json from sasrec seq parquet.")
    parser.add_argument("--seq_dir", type=Path, default=project_root / "data" / "seq")
    parser.add_argument("--output_file", type=Path, default=project_root / "outputs" / "train_data.json")
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
    parser.add_argument("--max_items_per_user", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    converted = convert_item2token(args.item2token_input, args.item2token_output_dir, args.id_offset)
    print(f"Wrote {converted}")
    write_train_data(args.seq_dir, args.output_file, args.max_items_per_user, args.batch_size)


if __name__ == "__main__":
    main()
