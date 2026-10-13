import re
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from .tokenizer import SQLTokenizer

def serialize_schema(raw_schema: str) -> str:
    """
    Compact Clean Schema Serialization.
    Extracts table names and column names, stripping all DDL keywords (CREATE, TABLE,
    VARCHAR, INT, PRIMARY KEY, etc.) and constraints.
    Supports both formal DDL statements and shorthand table(col1, col2).
    Output format:
      table1 : col1 , col2 | table2 : col1 , col2 , col3
    """
    if not raw_schema or not raw_schema.strip():
        return ""

    text = " ".join(raw_schema.split())
    tables = []

    # 1. Match formal CREATE TABLE statements
    ddl_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_]+)\s*\((.*?)\)',
        re.IGNORECASE
    )
    matches = ddl_pattern.findall(text)

    if matches:
        for tbl_name, col_defs in matches:
            cols = []
            for col_def in col_defs.split(","):
                col_def = col_def.strip()
                if not col_def:
                    continue
                # Ignore table-level constraints
                if re.match(r'^(PRIMARY\s+KEY|FOREIGN\s+KEY|CONSTRAINT|UNIQUE|CHECK)\b', col_def, re.IGNORECASE):
                    continue
                parts = col_def.split()
                if parts:
                    col_name = parts[0].strip('`"[]')
                    cols.append(col_name.lower())
            if cols:
                tables.append(f"{tbl_name.lower()} : {', '.join(cols)}")
    else:
        # 2. Match shorthand pattern: tablename (col1, col2)
        shorthand_pattern = re.compile(r'([A-Za-z0-9_]+)\s*\((.*?)\)')
        sh_matches = shorthand_pattern.findall(text)
        for tbl_name, col_defs in sh_matches:
            cols = []
            for c in col_defs.split(","):
                c = c.strip()
                if c:
                    col_name = c.split()[0].strip('`"[]')
                    cols.append(col_name.lower())
            if cols:
                tables.append(f"{tbl_name.lower()} : {', '.join(cols)}")

    if not tables:
        return text.strip().lower()
    return " | ".join(tables)

def normalize_sql(sql: str) -> str:
    """
    Canonical SQL Normalizer.
    Normalizes SQL keywords to uppercase, standardizes spacing around operators
    and commas, and formats queries to a canonical representation.
    """
    if not sql or not sql.strip():
        return ""

    text = " ".join(sql.strip().split())

    # Standardize operators and punctuation spacing
    text = re.sub(r'\s*([,;()])\s*', r' \1 ', text)
    text = re.sub(r'\s*([=><!]+)\s*', r' \1 ', text)
    text = re.sub(r'\s*([><!=])\s*=\s*', r' \1= ', text)

    tokens = text.split()
    normalized = []

    # Keyword lookup dictionary (case-insensitive)
    kw_set = {k.lower(): k for k in [
        "select", "from", "where", "join", "inner", "left", "right", "cross",
        "on", "group", "by", "order", "having", "limit", "offset", "as",
        "and", "or", "not", "in", "like", "between", "is", "null", "count",
        "sum", "avg", "min", "max", "distinct", "desc", "asc", "union",
        "except", "intersect", "case", "when", "then", "else", "end",
        "delete", "update", "set", "create", "table", "true", "false"
    ]}

    for t in tokens:
        low = t.lower()
        if low in kw_set:
            normalized.append(kw_set[low].upper() if low not in ("true", "false") else low)
        else:
            normalized.append(low)

    res = " ".join(normalized)
    # Clean up spaces before punctuation
    res = re.sub(r'\s+([,;.()])', r'\1', res)
    res = re.sub(r'([(])\s+', r'\1', res)
    res = re.sub(r'\s*;\s*$', ';', res)
    if not res.endswith(";"):
        res += ";"
    return res

def format_input_prompt(question: str, context: str) -> str:
    """
    Format input sequence using Clean Schema Serialization.
    Structure: [Schema] <compact_schema> [Question] <question>
    """
    clean_schema = serialize_schema(context)
    return f"[Schema] {clean_schema.strip()} [Question] {question.strip()}"

class Text2SQLDataset(Dataset):
    """
    PyTorch Dataset for Text-to-SQL tasks.
    Takes pairs of serialized prompts (+ schema) and canonical SQL queries.
    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        src_tokenizer: SQLTokenizer,
        tgt_tokenizer: SQLTokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 192
    ):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        src_text = item["src"]
        tgt_text = item["tgt"]

        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)

        # Truncate if exceeding max length (always preserve EOS token)
        if len(src_ids) > self.max_src_len:
            src_ids = src_ids[:self.max_src_len - 1] + [self.src_tokenizer.EOS_IDX]
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[:self.max_tgt_len - 1] + [self.tgt_tokenizer.EOS_IDX]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_raw": src_text,
            "tgt_raw": tgt_text
        }

def collate_fn(batch: List[Dict], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Pad batch of variable-length sequences to the max sequence length in the batch."""
    src_list = [item["src_ids"] for item in batch]
    tgt_list = [item["tgt_ids"] for item in batch]

    src_lengths = [len(s) for s in src_list]
    tgt_lengths = [len(t) for t in tgt_list]

    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    batch_size = len(batch)
    padded_src = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((batch_size, max_tgt_len), pad_idx, dtype=torch.long)

    for i in range(batch_size):
        padded_src[i, :len(src_list[i])] = src_list[i]
        padded_tgt[i, :len(tgt_list[i])] = tgt_list[i]

    # Attention mask: 1 for valid tokens, 0 for padding
    src_mask = (padded_src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)

    return {
        "src": padded_src,
        "tgt": padded_tgt,
        "src_mask": src_mask,
        "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
        "tgt_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
        "src_raw": [item["src_raw"] for item in batch],
        "tgt_raw": [item["tgt_raw"] for item in batch]
    }

def prepare_and_cache_dataset(
    output_dir: str = "data",
    num_samples: int = 15000,
    val_split: float = 0.1,
    force_rebuild: bool = False
) -> Tuple[str, str]:
    """
    Downloads a subset of gretelai/synthetic_text_to_sql, extracts prompt+schema,
    applies Clean Schema Serialization and Canonical SQL Normalization,
    and saves train.json and val.json to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    if not force_rebuild and os.path.exists(train_path) and os.path.exists(val_path):
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                cached_train = json.load(f)
            # Verify if cached dataset uses new schema format
            if len(cached_train) >= int(num_samples * (1 - val_split)) and "[Schema]" in cached_train[0].get("src", ""):
                print(f"[Dataset] Found existing serialized datasets at {train_path} ({len(cached_train)} samples)")
                return train_path, val_path
            else:
                print(f"[Dataset] Cached dataset needs updating to clean schema format or sample count. Re-fetching...")
        except Exception:
            pass

    print(f"[Dataset] Loading {num_samples} samples from gretelai/synthetic_text_to_sql...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train", streaming=True)

    records = []
    for item in ds:
        q = item.get("sql_prompt", "")
        ctx = item.get("sql_context", "")
        sql = item.get("sql", "")
        if not q or not sql or not ctx:
            continue

        src_text = format_input_prompt(q, ctx)
        tgt_text = normalize_sql(sql)

        if not src_text or not tgt_text:
            continue

        records.append({
            "src": src_text,
            "tgt": tgt_text
        })
        if len(records) >= num_samples:
            break

    val_count = int(len(records) * val_split)
    train_count = len(records) - val_count

    train_data = records[:train_count]
    val_data = records[train_count:]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"[Dataset] Saved {len(train_data)} train samples to {train_path}")
    print(f"[Dataset] Saved {len(val_data)} val samples to {val_path}")
    return train_path, val_path
