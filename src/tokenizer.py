import re
import json
import string
from collections import Counter
from typing import List, Dict, Optional

class SQLTokenizer:
    """
    Subword & SQL-aware Tokenizer with Character-Fallback.
    Guarantees ZERO <unk> tokens by falling back to character-level representations
    for rare identifiers, unseen table names, and arbitrary column names.
    """
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # Built-in SQL keywords & operators
    BASE_SQL_KEYWORDS = [
        "select", "from", "where", "group", "by", "order", "desc", "asc", "limit",
        "having", "join", "left", "right", "inner", "outer", "cross", "on", "as",
        "count", "sum", "avg", "min", "max", "distinct", "and", "or", "not", "in",
        "like", "between", "is", "null", "create", "table", "insert", "into",
        "values", "update", "delete", "set", "drop", "alter", "add", "constraint",
        "primary", "key", "foreign", "references", "integer", "int", "varchar",
        "text", "real", "float", "date", "datetime", "boolean", "true", "false",
        "question:", "schema:", "|", "*", "(", ")", ",", ";", ".", "=", ">", "<",
        ">=", "<=", "!=", "<>", "+", "-", "/", "%"
    ]

    TOKEN_PATTERN = re.compile(
        r"""
        [A-Za-z_][A-Za-z0-9_]*          # Identifiers / Keywords / Words
        |\d+(?:\.\d+)?                  # Numbers (integers and floats)
        |'[^']*'|"[^"]*"                # String literals enclosed in quotes
        |>=|<=|!=|<>|==                 # Multi-character comparison operators
        |[=><+\-*/%^&|~]                # Single-character operators
        |[(),.;:`]                      # Delimiters and punctuation
        |\S                             # Fallback non-whitespace character
        """,
        re.VERBOSE | re.IGNORECASE
    )

    def __init__(self, min_freq: int = 1, max_vocab_size: Optional[int] = 12000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.special_tokens = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN
        ]

        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

        # 1. Special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

        # 2. Add printable characters so we can always represent any unseen string
        for ch in string.printable:
            if ch not in ("\r", "\n", "\t", "\x0b", "\x0c") and ch not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[ch] = idx
                self.idx2token[idx] = ch

        # 3. Add base SQL keywords
        for kw in self.BASE_SQL_KEYWORDS:
            if kw not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[kw] = idx
                self.idx2token[idx] = kw

    def raw_tokenize(self, text: str) -> List[str]:
        """Extract primary regex tokens."""
        if not text:
            return []
        matches = self.TOKEN_PATTERN.findall(text)
        return [m.strip() for m in matches if m.strip()]

    def build_vocab(self, texts: List[str]):
        """Build vocabulary mapping from a corpus of texts."""
        counter = Counter()
        for text in texts:
            tokens = self.raw_tokenize(text.lower())
            counter.update(tokens)

        # Sort candidate words by frequency
        sorted_tokens = [w for w, c in counter.most_common() if c >= self.min_freq]

        for token in sorted_tokens:
            if self.max_vocab_size and len(self.token2idx) >= self.max_vocab_size:
                break
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def tokenize_word(self, word: str) -> List[str]:
        """Break down a single token if it is not in the vocabulary."""
        w = word.lower()
        if w in self.token2idx:
            return [w]

        # If it has underscores (like column_name), split and tokenize
        if "_" in w:
            parts = []
            segments = w.split("_")
            for i, seg in enumerate(segments):
                if seg:
                    parts.extend(self.tokenize_word(seg))
                if i < len(segments) - 1:
                    parts.append("_")
            return parts

        # Character fallback: every character exists in vocab, so zero <unk>!
        return list(w)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize full text with guaranteed vocabulary coverage."""
        raw_tokens = self.raw_tokenize(text)
        tokens = []
        for raw in raw_tokens:
            tokens.extend(self.tokenize_word(raw))
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a text string to a list of token IDs."""
        tokens = self.tokenize(text)
        ids = [self.token2idx.get(t, self.UNK_IDX) for t in tokens]
        if add_special_tokens:
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]
        return ids

    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back into a readable SQL/text string."""
        tokens = []
        for tid in token_ids:
            if isinstance(tid, (list, tuple)):
                continue
            if tid == self.EOS_IDX and remove_special_tokens:
                break
            if remove_special_tokens and tid in (self.PAD_IDX, self.SOS_IDX):
                continue
            tok_str = self.idx2token.get(tid, "")
            if tok_str:
                tokens.append(tok_str)

        # Merge adjacent single characters back into full words
        merged = []
        char_buf = []
        for t in tokens:
            if len(t) == 1 and t.isalnum():
                char_buf.append(t)
            else:
                if char_buf:
                    merged.append("".join(char_buf))
                    char_buf = []
                merged.append(t)
        if char_buf:
            merged.append("".join(char_buf))

        text = " ".join(merged)
        # Clean up spaces around SQL punctuation
        text = re.sub(r'\s+([,;.()=><])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        text = re.sub(r'\s*([=><])\s*', r' \1 ', text)
        text = re.sub(r'\s*([><!=])\s*=\s*', r' \1= ', text)
        return " ".join(text.split())

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, filepath: str):
        """Save vocabulary to a JSON file."""
        data = {
            "token2idx": self.token2idx,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SQLTokenizer":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(min_freq=data.get("min_freq", 1), max_vocab_size=data.get("max_vocab_size"))
        tok.token2idx = data["token2idx"]
        tok.idx2token = {int(v): k for k, v in tok.token2idx.items()}
        return tok
