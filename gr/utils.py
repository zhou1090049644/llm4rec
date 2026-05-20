import copy
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    Qwen2ForCausalLM,
    AutoConfig
)
import os


def build_semantic_tokens(se_id_space_width):
    widths = [int(width) for width in se_id_space_width.split(",") if width]
    prefixes = ["a", "b", "c", "d"]
    tokens = ["<|hist_clk_start|>", "<|hist_clk_end|>"]
    for prefix, width in zip(prefixes, widths):
        tokens.extend(f"<{prefix}_{idx}>" for idx in range(width))
    return tokens


def load_model_and_tokenizer(model_args):
    dummy_model_path = os.path.join(os.environ.get("USER_CACHE_PATH"), "qwen_init2")
    tokenizer = AutoTokenizer.from_pretrained(dummy_model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": build_semantic_tokens(model_args.se_id_space_width)}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(dummy_model_path)
    config.vocab_size = len(tokenizer)
    # 未加载预训练权重：
    # model = Qwen2ForCausalLM.from_pretrained(dummy_model_path)
    model = Qwen2ForCausalLM(config)
    return model, tokenizer

# load semantic id dict related function
def load_item2token_dict(item2token_dict_path):
    item2token_dict = dict()

    for file in os.listdir(item2token_dict_path):
        path = os.path.join(item2token_dict_path,file)
        with open(path, 'r') as f:
            for line in f.readlines():
                item_id, token = line.strip().split('\t')  # 去除每一行最后的换行符，否则解码的时候会有'\n'
                item2token_dict[item_id] = token

    print("length of item2token_dict is:", len(item2token_dict))
    return item2token_dict

def load_token2item_dict(item2token_dict_path):
    token2item_dict = dict()
    with open(item2token_dict_path, 'r') as f:
        for line in f.readlines():
            token, item_list = line.strip().split('\t')  # 去除每一行最后的换行符，否则解码的时候会有'\n'
            item_list = item_list[1:-1].split(',')
            token2item_dict[token] = item_list
    print("length of token2item_dict is:", len(token2item_dict))
    return token2item_dict
