import torch
import sqlite3
from typing import List, Optional, Tuple
from .tokenizer import SQLTokenizer
from .dataset import format_input_prompt
from .models import TransformerSeq2Seq

def greedy_decode_transformer(
    model: TransformerSeq2Seq,
    src_text: str,
    src_tokenizer: SQLTokenizer,
    tgt_tokenizer: SQLTokenizer,
    max_len: int = 150,
    device: str = "cpu"
) -> str:
    """
    Greedy autoregressive decoding for Transformer Seq2Seq.
    Generates target SQL tokens one-by-one by choosing the highest probability token at each step.
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_tokenizer.encode(src_text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = model.make_src_mask(src_tensor)

        # 1. Encode source
        memory = model.encode(src_tensor, src_mask)

        # 2. Autoregressive decoding starting with <sos>
        tgt_ids = [tgt_tokenizer.SOS_IDX]

        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)

            out = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
            # Logits for the latest token
            next_token_logits = out[0, -1, :]
            next_token_id = next_token_logits.argmax().item()

            if next_token_id == tgt_tokenizer.EOS_IDX:
                break

            tgt_ids.append(next_token_id)

        # Remove <sos>
        generated_sql = tgt_tokenizer.decode(tgt_ids[1:], remove_special_tokens=True)
        return generated_sql

def beam_search_decode_transformer(
    model: TransformerSeq2Seq,
    src_text: str,
    src_tokenizer: SQLTokenizer,
    tgt_tokenizer: SQLTokenizer,
    beam_width: int = 3,
    max_len: int = 150,
    device: str = "cpu"
) -> str:
    """
    Beam search decoding to explore multiple candidate hypotheses in parallel.
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_tokenizer.encode(src_text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = model.make_src_mask(src_tensor)
        memory = model.encode(src_tensor, src_mask)

        # List of tuples: (sequence_of_ids, cumulative_log_prob)
        beams = [([tgt_tokenizer.SOS_IDX], 0.0)]
        completed_beams = []

        for step in range(max_len):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == tgt_tokenizer.EOS_IDX:
                    completed_beams.append((seq, score))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = model.make_tgt_mask(tgt_tensor)
                out = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
                log_probs = torch.log_softmax(out[0, -1, :], dim=-1)

                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    next_id = topk_indices[k].item()
                    candidate_seq = seq + [next_id]
                    candidate_score = score + topk_probs[k].item()
                    all_candidates.append((candidate_seq, candidate_score))

            if not all_candidates:
                break

            # Length normalization
            ordered = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
            beams = ordered[:beam_width]

            if len(completed_beams) >= beam_width:
                break

        final_candidates = completed_beams if completed_beams else beams
        best_seq, _ = max(final_candidates, key=lambda x: x[1] / (len(x[0]) ** 0.7))
        # Skip <sos> and stop at <eos>
        res_ids = [tok for tok in best_seq[1:] if tok != tgt_tokenizer.EOS_IDX]
        return tgt_tokenizer.decode(res_ids, remove_special_tokens=True)

def predict_sql(
    model: TransformerSeq2Seq,
    question: str,
    schema: str,
    src_tokenizer: SQLTokenizer,
    tgt_tokenizer: SQLTokenizer,
    use_beam_search: bool = False,
    beam_width: int = 3,
    device: str = "cpu"
) -> str:
    """End-to-end inference from question and schema string to SQL statement."""
    src_text = format_input_prompt(question, schema)
    if use_beam_search:
        return beam_search_decode_transformer(model, src_text, src_tokenizer, tgt_tokenizer, beam_width, device=device)
    return greedy_decode_transformer(model, src_text, src_tokenizer, tgt_tokenizer, device=device)

def validate_sql_syntax(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the generated SQL query is syntactically valid in SQLite.
    Returns (is_valid, error_message).
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        # EXPLAIN checks SQL syntax without needing existing tables
        cursor.execute(f"EXPLAIN {sql}")
        return True, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        # "no such table" means syntax is fine, only table doesn't exist in dummy memory DB
        if "no such table" in msg.lower() or "no such column" in msg.lower():
            return True, None
        return False, msg
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()
