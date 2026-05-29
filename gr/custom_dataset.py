import copy
import json

import numpy as np
from torch.utils.data import IterableDataset


class CustomTrainDataset(IterableDataset):
    def __init__(self, json_path, item2token_dict, tokenizer, data_args):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.token_depth = data_args.token_depth
        self.tokenizer.padding_side = data_args.padding_side
        self.max_seq_length = data_args.max_seq_length
        self.max_imp_seq_length = data_args.max_imp_seq_length
        self.max_click_seq_length = data_args.max_click_seq_length
        self.item2token_dict = item2token_dict
        self.response_flag = data_args.response_flag

        self.total_seq_length = (
            self.max_seq_length * self.token_depth
            + self.max_imp_seq_length * self.token_depth
            + self.max_click_seq_length * self.token_depth
            + self.token_depth
            + 2
        )

    def _stream_json_data(self):
        """Stream legacy user sequences or explicit history-target JSONL samples."""
        with open(self.json_path, "r", encoding="utf-8") as fp:
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
                        item_sequence = history + [target]
                        user_id = str(sample.get("user_id", sample.get("sample_id", "")))
                    else:
                        user_id, item_sequence = next(iter(sample.items()))
                else:
                    user_data = json.loads("{" + line + "}")
                    user_id, item_sequence = next(iter(user_data.items()))

                if len(item_sequence) < 2:
                    continue
                yield {
                    "user_id": user_id,
                    "user_behavior_list": np.array(item_sequence),
                    "main_imp_item_list": np.array([]),
                    "main_click_item_list": np.array([]),
                    "similar_imp_item_list": np.array([]),
                    "similar_click_item_list": np.array([]),
                    "guess_imp_item_list": np.array([]),
                    "guess_click_item_list": np.array([]),
                    "user_imp_item_list": np.array([]),
                    "user_click_item_list": np.array([]),
                }

    def _process_data_model_log(self, example):
        user_behavior_list = np.array(example["user_behavior_list"])
        if len(user_behavior_list) == 0:
            return None

        target_item = ""
        for i in range(len(user_behavior_list) - 1, -1, -1):
            item = str(user_behavior_list[i])
            if item in self.item2token_dict:
                target_item = item
                break
        if not target_item:
            return None

        input_user_behavior_list = []
        for user_behavior in user_behavior_list:
            user_behavior_key = str(user_behavior)
            if user_behavior_key == target_item:
                continue
            if user_behavior_key in self.item2token_dict:
                input_user_behavior_list.append(self.item2token_dict[user_behavior_key])

        input_user_behavior_list = input_user_behavior_list[-self.max_seq_length :]
        target_item_seid = self.item2token_dict[target_item]
        history_behavior_input = (
            "<|hist_clk_start|>" + "".join(input_user_behavior_list) + "<|hist_clk_end|>"
        )
        full_inputs = history_behavior_input + target_item_seid
        target = history_behavior_input

        result = self.tokenizer(
            text=full_inputs,
            text_target=target,
            padding="max_length",
            max_length=self.total_seq_length,
            truncation=True,
        )

        labels = copy.deepcopy(result["input_ids"])
        labels = [
            -100
            if labels[i] == self.tokenizer.pad_token_id or result["labels"][i] != self.tokenizer.pad_token_id
            else labels[i]
            for i in range(len(labels))
        ]
        result["labels"] = labels
        return result

    def __iter__(self):
        for example in self._stream_json_data():
            output = self._process_data_model_log(example)
            if output:
                yield output
