from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import json
from arguments import *
from utils import *
import argparse
try:
    from gr.custom_dataset import *
except ModuleNotFoundError:
    from custom_dataset import *
import torch

def parse_args_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config_dict={}
    for keys,value_dict in config.items():
        for k,v in value_dict.items():
            config_dict[k] = v
    print(config_dict)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(config_dict)
    return model_args, data_args, training_args

def main(model_args, data_args, training_args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args)
    # 加载item2token dict
    item2token_dict = load_item2token_dict(os.path.join(os.environ.get("USER_CACHE_PATH"), data_args.item2token_dict))
    # if model_args.from_pretrained and local_rank == '0':
    tokenizer.save_pretrained(os.path.join(os.environ.get("TRAIN_CKPT_PATH"), training_args.output_dir))
    # 加载数据集
    train_dataset = CustomTrainDataset(
        json_path=os.path.join(os.environ.get("USER_CACHE_PATH"), data_args.train_file),
        item2token_dict=item2token_dict,
        tokenizer=tokenizer,
        data_args=data_args
    )
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    # data_collator = Collator(data_args,tokenizer=tokenizer)

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 保存模型和tokenizer
    model.save_pretrained(os.path.join(os.environ.get("TRAIN_CKPT_PATH"), training_args.output_dir))
    tokenizer.save_pretrained(os.path.join(os.environ.get("TRAIN_CKPT_PATH"), training_args.output_dir))

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run SFT modeling with DeepSpeed")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args, unknown = parser.parse_known_args()  # 使用 parse_known_args 处理未知参数

    # 从配置文件中解析参数
    model_args, data_args, training_args = parse_args_from_json(args.config)

    # 主函数
    main(model_args, data_args, training_args)
