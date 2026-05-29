# GR 本地划分与 Candidate 评估

## 生成数据

先确保 `data/candidate` 已从 `TAAC2025/TencentGR-1M` 下载。若还没有下载，可以在项目根目录执行：

```bash
python data/download_tencentgr_parquet.py --include_candidate
```

然后构造 GR 所需文件：

```bash
python scripts/build_gr_train_data.py
```

默认会生成：

- `outputs/gr_train_split.jsonl`
- `outputs/gr_valid_split.jsonl`
- `outputs/gr_test_split.jsonl`
- `outputs/candidate_items.txt`

`candidate_items.txt` 默认会通过 `data/indexer.pkl` 把官方 candidate 的原始 `item_id` 映射到当前 GR/RQ-VAE 使用的内部 item id 空间，这样才能和模型生成的推荐 id 对齐。

划分逻辑为每个用户保留最后两个行为做验证和测试：

- train: 用前 `n-2` 个行为构造滑窗样本
- valid: 用前 `n-2` 个行为预测第 `n-1` 个行为
- test: 用前 `n-1` 个行为预测第 `n` 个行为

每个样本的历史长度最多保留最近 `100` 个 item。

## 评估

验证集：

```bash
bash gr/run_eval_valid.sh
```

测试集：

```bash
bash gr/run_eval_test.sh
```

测试脚本默认会保存推荐结果，并在 candidate 约束下计算：

- `hit@5` / `recall@5`
- `hit@10` / `recall@10`
- `hit@20` / `recall@20`

由于每个样本只有一个目标 item，所以这里的 Recall@K 与 Hit@K 数值相同。
