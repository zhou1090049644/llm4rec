import json
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover - reported at runtime with a clearer message.
    pa = None
    ds = None


ITEM_SPARSE_FEATS = [
    "100",
    "117",
    "118",
    "101",
    "102",
    "119",
    "120",
    "114",
    "112",
    "121",
    "115",
    "122",
    "116",
]
USER_SPARSE_FEATS = ["103", "104", "105", "109"]
USER_ARRAY_FEATS = ["106", "107", "108", "110"]
MM_EMB_SHAPES = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}


def _require_pyarrow() -> None:
    if ds is None:
        raise RuntimeError("pyarrow is required to read TencentGR Parquet data. Install pyarrow first.")


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def _first_existing(base: Path, candidates: Iterable[str]) -> Path:
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    names = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find any of [{names}] under {base}")


def _parquet_dataset(path: Path):
    _require_pyarrow()
    return ds.dataset(str(path), format="parquet")


def _available_columns(path: Path) -> set[str]:
    dataset = _parquet_dataset(path)
    return set(dataset.schema.names)


def _to_python(v: Any) -> Any:
    if hasattr(v, "as_py"):
        v = v.as_py()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _extract_feature_value(v: Any) -> Any:
    v = _to_python(v)
    if _is_null(v):
        return None
    if isinstance(v, dict):
        # HF feature cells are commonly structs like {"cold_start": ..., "feature_value": ...}.
        if "feature_value" in v:
            return _extract_feature_value(v["feature_value"])
        if "value" in v:
            return _extract_feature_value(v["value"])
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v if not _is_null(_to_python(x))]
    return v


def _coerce_sparse(v: Any) -> Optional[int]:
    v = _extract_feature_value(v)
    if _is_null(v):
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        v = v[0]
    try:
        return int(v)
    except Exception:
        return 0


def _coerce_array(v: Any) -> Optional[List[int]]:
    v = _extract_feature_value(v)
    if _is_null(v):
        return None
    if not isinstance(v, (list, tuple)):
        v = [v]
    out: List[int] = []
    for x in v:
        if _is_null(x):
            continue
        try:
            out.append(int(x))
        except Exception:
            out.append(0)
    return out or None


def _safe_int(v: Any, default: int = 0) -> int:
    if _is_null(v):
        return default
    try:
        return int(v)
    except Exception:
        return default


def _index_lookup(indexer: Dict[str, Dict[Any, int]], group: str, raw: Any) -> Any:
    mapping = indexer.get(group, {})
    if raw in mapping:
        return mapping[raw]
    raw_s = str(raw)
    if raw_s in mapping:
        return mapping[raw_s]
    try:
        raw_i = int(raw)
        if raw_i in mapping:
            return mapping[raw_i]
    except Exception:
        pass
    return raw


def load_feat_dict_from_parquet_folder(
    src: Path,
    id_col: str,
    sparse_ids: List[str],
    array_ids: Optional[List[str]] = None,
    indexer: Optional[Dict[str, Dict[Any, int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    dataset = _parquet_dataset(src)
    available = set(dataset.schema.names)
    feat_ids = [fid for fid in sparse_ids + (array_ids or []) if fid in available]
    missing = sorted(set(sparse_ids + (array_ids or [])) - available)
    if missing:
        print(f"Warning: feature columns missing in {src}: {missing}")
    cols = [id_col] + feat_ids
    table = dataset.to_table(columns=cols)
    data = table.to_pydict()

    out: Dict[str, Dict[str, Any]] = {}
    for i in tqdm(range(len(data[id_col])), desc=f"Loading {src.name}"):
        raw_id = data[id_col][i]
        if _is_null(raw_id):
            continue
        key_id = _index_lookup(indexer or {}, "i" if id_col == "item_id" else "u", raw_id)
        feat: Dict[str, Any] = {}
        for fid in feat_ids:
            if fid in sparse_ids:
                value = _coerce_sparse(data[fid][i])
            else:
                value = _coerce_array(data[fid][i])
            if value is not None:
                feat[fid] = value
        out[str(key_id)] = feat
        out.setdefault(str(raw_id), feat)
    return out


def load_mm_emb_from_parquet_folder(src: Path) -> Dict[str, np.ndarray]:
    dataset = _parquet_dataset(src)
    table = dataset.to_table(columns=["anonymous_cid", "emb"])
    data = table.to_pydict()
    out: Dict[str, np.ndarray] = {}
    for cid, emb in tqdm(zip(data["anonymous_cid"], data["emb"]), total=len(data["anonymous_cid"]), desc=f"Loading {src.name}"):
        if _is_null(cid) or _is_null(emb):
            continue
        emb = _to_python(emb)
        if isinstance(emb, str):
            emb = json.loads(emb)
        out[str(cid)] = np.asarray(emb, dtype=np.float32)
    return out


def load_mm_emb_v2(data_dir: Path, feat_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    mm_emb_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for feat_id in tqdm(feat_ids, desc="Loading mm_emb"):
        shape = MM_EMB_SHAPES[feat_id]
        src = _first_existing(
            data_dir,
            [
                f"mm_emb/emb_{feat_id}_{shape}_parquet",
                f"mm_emb_{feat_id}_{shape}",
                f"emb_{feat_id}_{shape}_parquet",
                f"emb_{feat_id}_{shape}",
            ],
        )
        mm_emb_dict[feat_id] = load_mm_emb_from_parquet_folder(src)
        if not mm_emb_dict[feat_id]:
            raise RuntimeError(f"No embeddings loaded for mm_emb_id={feat_id} from {src}")
    return mm_emb_dict


def load_seq_as_list(seq_dir: Path, batch_size: int = 100000) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    dataset = _parquet_dataset(seq_dir)
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)
    event_type = np.dtype([("item_id", np.int32), ("action_type", np.int8), ("timestamp", np.int64)])

    all_events_list: List[np.ndarray] = []
    user_indices: Dict[int, Tuple[int, int]] = {}
    cur_global_idx = 0
    total_rows = dataset.count_rows()

    with tqdm(total=total_rows, desc="Loading seq", unit=" rows") as pbar:
        for batch in scanner.to_batches():
            uids = batch.column("user_id").to_pylist()
            seqs = batch.column("seq")

            if not pa.types.is_list(seqs.type) and not pa.types.is_large_list(seqs.type):
                raise ValueError(f"Expected seq to be list<struct>, got {seqs.type}")

            flat_structs = seqs.values
            batch_events = np.empty(len(flat_structs), dtype=event_type)
            batch_events["item_id"] = np.asarray(flat_structs.field("item_id").to_pylist(), dtype=np.int32)
            batch_events["action_type"] = np.asarray(
                [_safe_int(v) for v in flat_structs.field("action_type").to_pylist()], dtype=np.int8
            )
            batch_events["timestamp"] = np.asarray(
                [_safe_int(v) for v in flat_structs.field("timestamp").to_pylist()], dtype=np.int64
            )
            all_events_list.append(batch_events)

            offsets = seqs.offsets.to_numpy()
            lengths = np.diff(offsets)
            for i, uid in enumerate(uids):
                if not _is_null(uid):
                    user_indices[int(uid)] = (int(cur_global_idx + offsets[i]), int(lengths[i]))

            cur_global_idx += len(batch_events)
            pbar.update(batch.num_rows)

    if not all_events_list:
        return np.empty(0, dtype=event_type), user_indices
    return np.concatenate(all_events_list), user_indices


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        indexer_path = self.data_dir / "indexer.pkl"
        if indexer_path.exists():
            with open(indexer_path, "rb") as f:
                self.indexer = pickle.load(f)
        else:
            print("Warning: indexer.pkl not found; falling back to max-id based sizes.")
            self.indexer = {"i": {}, "u": {}, "f": {}}

        seq_src = _first_existing(self.data_dir, ["seq"])
        item_feat_src = _first_existing(self.data_dir, ["item_feat"])
        user_feat_src = _first_existing(self.data_dir, ["user_feat"])

        self.full_events, self.user_indices = load_seq_as_list(seq_src)
        self.seq_usernum = len(self.user_indices)
        self.item_feat_dict = load_feat_dict_from_parquet_folder(
            item_feat_src, "item_id", ITEM_SPARSE_FEATS, indexer=self.indexer
        )
        self.user_feat_dict = load_feat_dict_from_parquet_folder(
            user_feat_src, "user_id", USER_SPARSE_FEATS, USER_ARRAY_FEATS, indexer=self.indexer
        )
        self.mm_emb_dict = load_mm_emb_v2(self.data_dir, self.mm_emb_ids)

        self.itemnum = len(self.indexer.get("i", {})) or self._infer_itemnum()
        self.usernum = len(self.indexer.get("u", {})) or max(self.user_indices.keys(), default=0)
        self.indexer_i_rev = {v: k for k, v in self.indexer.get("i", {}).items()}
        self.indexer_u_rev = {v: k for k, v in self.indexer.get("u", {}).items()}
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _infer_itemnum(self) -> int:
        max_seq_item = int(self.full_events["item_id"].max()) if len(self.full_events) else 0
        max_feat_item = max((int(k) for k in self.item_feat_dict.keys() if str(k).isdigit()), default=0)
        return max(max_seq_item, max_feat_item)

    def _feature_cardinality(self, feat_id: str, values: Iterable[Any]) -> int:
        indexed = self.indexer.get("f", {}).get(feat_id)
        if indexed is not None:
            return len(indexed)
        max_value = 0
        for value in values:
            if isinstance(value, list):
                vals = value
            else:
                vals = [value]
            for v in vals:
                try:
                    max_value = max(max_value, int(v))
                except Exception:
                    pass
        return max_value + 1

    def _init_feat_info(self):
        feat_default_value: Dict[str, Any] = {}
        feat_statistics: Dict[str, int] = {}
        feat_types = {
            "user_sparse": USER_SPARSE_FEATS,
            "item_sparse": ITEM_SPARSE_FEATS,
            "item_array": [],
            "user_array": USER_ARRAY_FEATS,
            "item_emb": self.mm_emb_ids,
            "user_continual": [],
            "item_continual": [],
        }

        all_feat_dicts = list(self.item_feat_dict.values()) + list(self.user_feat_dict.values())
        for feat_id in feat_types["user_sparse"] + feat_types["item_sparse"]:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = self._feature_cardinality(
                feat_id, (feat.get(feat_id, 0) for feat in all_feat_dicts)
            )
        for feat_id in feat_types["user_array"] + feat_types["item_array"]:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = self._feature_cardinality(
                feat_id, (feat.get(feat_id, [0]) for feat in all_feat_dicts)
            )
        for feat_id in feat_types["item_emb"]:
            first = next(iter(self.mm_emb_dict[feat_id].values()))
            feat_default_value[feat_id] = np.zeros(first.shape[0], dtype=np.float32)
        return feat_default_value, feat_types, feat_statistics

    def new_load_user_data(self, uid: int):
        if uid not in self.user_indices:
            return None
        start_idx, length = self.user_indices[uid]
        user_data = self.full_events[start_idx : start_idx + length]
        user_feat = self.user_feat_dict.get(str(uid), {})
        data = []
        last_timestamp = 0
        for row in user_data:
            item_id = int(row["item_id"])
            action_type = int(row["action_type"])
            timestamp = int(row["timestamp"])
            item_feat = self.item_feat_dict.get(str(item_id), {})
            data.append((uid, item_id, None, item_feat, action_type, timestamp))
            last_timestamp = timestamp
        data.append((uid, None, user_feat, None, None, last_timestamp))
        return data

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        uid += 1
        user_sequence = self.new_load_user_data(uid)
        if not user_sequence:
            raise IndexError(f"user id {uid} is not available in seq parquet")

        ext_user_sequence = []
        for u, i, user_feat, item_feat, action_type, _ in user_sequence:
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat is not None:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        if len(ext_user_sequence) < 2:
            return self._empty_sample()

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen
        ts = {record[0] for record in ext_user_sequence if record[2] == 1 and record[0]}

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict.get(str(neg_id), {}), neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def _empty_sample(self):
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        feat = np.empty([self.maxlen + 1], dtype=object)
        feat[:] = self.feature_default_value
        return seq, seq.copy(), seq.copy(), seq.copy(), seq.copy(), seq.copy(), feat, feat.copy(), feat.copy()

    def __len__(self):
        return self.seq_usernum

    def fill_missing_feat(self, feat, item_id):
        if feat is None:
            feat = {}
        filled_feat = dict(feat)
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        for feat_id in set(all_feat_ids) - set(filled_feat.keys()):
            filled_feat[feat_id] = self.feature_default_value[feat_id]

        raw_item_id = self.indexer_i_rev.get(item_id, item_id)
        for feat_id in self.feature_types["item_emb"]:
            emb = self.mm_emb_dict[feat_id].get(str(raw_item_id))
            if emb is None:
                emb = self.mm_emb_dict[feat_id].get(str(item_id))
            if isinstance(emb, np.ndarray):
                filled_feat[feat_id] = emb
        return filled_feat

    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        return (
            torch.from_numpy(np.array(seq)),
            torch.from_numpy(np.array(pos)),
            torch.from_numpy(np.array(neg)),
            torch.from_numpy(np.array(token_type)),
            torch.from_numpy(np.array(next_token_type)),
            torch.from_numpy(np.array(next_action_type)),
            list(seq_feat),
            list(pos_feat),
            list(neg_feat),
        )


class MyTestDataset(MyDataset):
    def __getitem__(self, uid):
        uid += 1
        user_sequence = self.new_load_user_data(uid)
        if not user_sequence:
            raise IndexError(f"user id {uid} is not available in seq parquet")

        user_id = self.indexer_u_rev.get(uid, uid)
        ext_user_sequence = []
        for u, i, user_feat, item_feat, _, _ in user_sequence:
            if u and user_feat:
                ext_user_sequence.insert(0, (u, self._process_cold_start_feat(user_feat), 2))
            if i and item_feat is not None:
                if i > self.itemnum:
                    i = 0
                ext_user_sequence.append((i, self._process_cold_start_feat(item_feat), 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        idx = self.maxlen
        for i, feat, type_ in reversed(ext_user_sequence[:-1]):
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        return seq, token_type, seq_feat, str(user_id)

    def _process_cold_start_feat(self, feat):
        processed = {}
        for feat_id, value in feat.items():
            if isinstance(value, list):
                processed[feat_id] = [0 if isinstance(v, str) else v for v in value]
            elif isinstance(value, str):
                processed[feat_id] = 0
            else:
                processed[feat_id] = value
        return processed

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        return torch.from_numpy(np.array(seq)), torch.from_numpy(np.array(token_type)), list(seq_feat), user_id


def save_emb(emb, save_path):
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f"saving {save_path}")
    with open(Path(save_path), "wb") as f:
        f.write(struct.pack("II", num_points, num_dimensions))
        emb.tofile(f)
