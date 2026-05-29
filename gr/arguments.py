from dataclasses import dataclass, field
from typing import Optional

# 定义数据类
@dataclass
class ModelArguments:
    from_pretrained: bool = field(default=True, metadata={"help": "whether to load the pretrained model"})
    from_checkpoint: bool = field(default=False, metadata={"help": "whether to load the trained model checkpoint"})
    pretrained_path: Optional[str] = field(
        default=None,
        metadata={"help": "The pretrained model path"})
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "The trained model checkpoint path"})
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "The used tokenizer path"})
    se_id_space_width: Optional[str] = field(
        default='256,256,256,256',
        metadata={"help": "dimension of se id for each depth"})


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None,metadata={"help": "A csv or a json file containing the training data."})
    max_seq_length: int = field(default=150, metadata={
        "help": "The maximum total input user behavior sequence length before tokenization."})
    max_imp_seq_length: int = field(default=20, metadata={
        "help": "The maximum total input user in page imp sequence length before tokenization."})
    max_click_seq_length: int = field(default=5, metadata={
        "help": "The maximum total input user in page click length before tokenization."})
    item2token_dict: Optional[str] = field(default="", metadata={"help": "item to token file path,txt format"})
    token2item_dict: Optional[str] = field(default="", metadata={"help": "token to item file path,txt format"})
    token2ads_dict: Optional[str] = field(default="", metadata={"help": "token to ads file path,txt format"})
    token_depth: int = field(default=4, metadata={
        "help": "se id depth."})
    padding_side: Optional[str] = field(default="right", metadata={"help": "The padding side for training"})
    response_flag: bool = field(default=True, metadata={"help": "whether to train on the target response tokens only"})
    test_file: Optional[str] = field(default=None,metadata={"help": "A csv or a json file containing the evaluation data.","required":False})
    valid_file: Optional[str] = field(default=None,metadata={"help": "A json/jsonl file containing validation samples.","required":False})
    candidate_file: Optional[str] = field(default=None,metadata={"help": "A text/json/jsonl file containing globally valid candidate item ids.","required":False})
