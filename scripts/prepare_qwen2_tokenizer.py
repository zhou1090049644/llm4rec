import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def semantic_tokens(widths):
    prefixes = ["a", "b", "c", "d"]
    tokens = []
    for prefix, size in zip(prefixes, widths):
        tokens.extend(f"<{prefix}_{idx}>" for idx in range(size))
    tokens.extend(["<|hist_clk_start|>", "<|hist_clk_end|>"])
    return tokens


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare Qwen2 tokenizer/config for GR.")
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--output_dir", type=Path, default=project_root / "outputs" / "qwen_init2")
    parser.add_argument("--se_id_space_width", default="2048,2048,1024")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    widths = [int(x) for x in args.se_id_space_width.split(",") if x]
    tokens = semantic_tokens(widths)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    tokenizer.add_tokens(tokens, special_tokens=True)
    tokenizer.save_pretrained(args.output_dir)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    config.vocab_size = len(tokenizer)
    config.save_pretrained(args.output_dir)

    print(f"Saved tokenizer/config to {args.output_dir}")
    print(f"Added semantic tokens: {len(tokens)}")
    print(f"Final vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
