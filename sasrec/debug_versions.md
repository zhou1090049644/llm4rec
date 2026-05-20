# SASRec Debug Versions

This file records manual debug versions because git is not available in the
current Windows shell. Append a new section for every future debug edit so a
specific version can be reverted by hand.

## nan-debug-v1 - 2026-05-06

Goal:
- Locate the first non-finite value that causes training loss to become NaN
  after the first optimizer step.

Modified files:
- `sasrec/main.py`
- `sasrec/model.py`

Added behavior:
- `sasrec/main.py`
  - Opens `TRAIN_LOG_PATH/nan_debug.jsonl`.
  - Records empty supervised batches as `empty_valid_mask`.
  - Records finite statistics for all/valid positive and negative logits.
  - Records `nonfinite_forward_output` if model outputs contain NaN or Inf.
  - Records `nonfinite_loss_skip_backward` and skips backward/optimizer step
    when the scalar loss is non-finite.
  - After `loss.backward()`, records `nonfinite_grad_after_backward` and raises
    `RuntimeError` if the first non-finite gradient is found.
  - After `optimizer.step()`, records `nonfinite_param_after_optimizer_step`
    and raises `RuntimeError` if the first non-finite parameter is found.
- `sasrec/model.py`
  - Adds `_debug_tensor_stats`.
  - Stores `model.debug_last_forward` for each forward pass, including stats
    for `log_feats`, `pos_embs`, `neg_embs`, logits before mask, and loss mask
    counts.

Expected debug output:
- `outputs/sasrec/log/nan_debug.jsonl` when using `sasrec/run_train_linux.sh`.
- Each line is one JSON event and can be searched by `event`.

Rollback notes:
- Remove helper functions `_tensor_debug_stats`, `_write_debug`,
  `_first_nonfinite_named_tensor`, `_first_nonfinite_param`, and
  `_first_nonfinite_grad` from `sasrec/main.py`.
- Remove the `debug_file = open(...)` line and the final `debug_file.close()`.
- Remove all `_write_debug(...)`, gradient check, parameter check, and
  non-finite-loss skip blocks from the training loop.
- Remove `_debug_tensor_stats` and `self.debug_last_forward = {...}` from
  `sasrec/model.py`.

## nan-fix-v2 - 2026-05-06

Reason from debug output:
- `global_step=0` had finite loss, but `log_feats` already contained NaN on
  exactly 151 positions.
- `pos_logits_valid` and `neg_logits_valid` were finite, while all logits had
  151 NaNs.
- `loss_mask_true_count=1481` and `loss_mask_numel=1632`; therefore
  `1632 - 1481 = 151` non-supervised or padding positions produced NaN.
- Backward then produced non-finite `item_emb.weight` gradients, consistent
  with masked-out `0 * NaN` values still poisoning autograd.

Modified files:
- `sasrec/model.py`

Added behavior:
- In `BaselineModel.log2feats`, after each attention call:
  - Convert NaN/Inf in `mha_outputs` to zero with `torch.nan_to_num`.
  - Zero out padding token rows with `attention_mask_pad.unsqueeze(-1)`.
- After each feed-forward block:
  - Convert NaN/Inf in `seqs` to zero.
  - Zero out padding token rows again.
- Before returning `log_feats`:
  - Convert NaN/Inf to zero.
  - Zero out padding token rows.

Expected result:
- `model_forward.log_feats.finite` should become `true`.
- `pos_logits_all` and `neg_logits_all` should become finite.
- `nonfinite_grad_after_backward` should no longer fire at step 0.

Rollback notes:
- Revert the added `torch.nan_to_num(...)` calls in `BaselineModel.log2feats`.
- Revert the added `seqs = seqs * attention_mask_pad.unsqueeze(-1)` lines.
- Revert the added `log_feats = log_feats * attention_mask_pad.unsqueeze(-1)`
  line.

## nan-fix-v3 - 2026-05-07

Reason from debug output:
- After `nan-fix-v2`, all forward tensors became finite, including
  `log_feats`, all logits, and valid logits.
- `loss` was finite, but `loss.backward()` still produced non-finite
  `item_emb.weight` gradients at step 0.
- This points to a backward-only NaN path inside attention. Padding query rows
  can be fully masked in `scaled_dot_product_attention`; cleaning the returned
  tensor is not enough because the internal backward path can still emit NaN.

Modified files:
- `sasrec/model.py`

Added behavior:
- In `BaselineModel.log2feats`, before attention:
  - Zero out padding token embeddings with
    `seqs = seqs * attention_mask_pad.unsqueeze(-1)`.
  - Make attention masks safe for padding query rows by allowing each invalid
    query to attend to its own zeroed position.

Expected result:
- `nonfinite_grad_after_backward` should no longer fire at step 0.
- Forward tensors should remain finite.

Rollback notes:
- Remove the pre-attention padding clear line:
  `seqs = seqs * attention_mask_pad.unsqueeze(-1)`.
- Remove `invalid_query_mask`, `self_attention_mask`, and the line that ORs
  them into `attention_mask`.

## cleanup-v4 - 2026-05-07

Reason:
- Training loss is now normal.
- Remove heavy debug checks and `nan_debug.jsonl` output to restore training
  speed.

Modified files:
- `sasrec/main.py`
- `sasrec/model.py`

Removed behavior:
- `sasrec/main.py`
  - Removed `_tensor_debug_stats`, `_write_debug`,
    `_first_nonfinite_named_tensor`, `_first_nonfinite_param`, and
    `_first_nonfinite_grad`.
  - Removed `TRAIN_LOG_PATH/nan_debug.jsonl` creation.
  - Removed per-step forward tensor stats.
  - Removed non-finite loss skip debug output.
  - Removed full-model gradient and parameter finite checks.
- `sasrec/model.py`
  - Removed `_debug_tensor_stats`.
  - Removed `self.debug_last_forward` collection.

Kept behavior:
- Empty supervised batches are still skipped.
- The padding-safe attention mask and padding-row cleanup from `nan-fix-v2` and
  `nan-fix-v3` remain in place.

Rollback notes:
- To restore debug output, re-apply the behavior described in `nan-debug-v1`.
- To remove the actual NaN fix, follow the rollback notes for `nan-fix-v2` and
  `nan-fix-v3`.
