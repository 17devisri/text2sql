import json
from typing import List, Dict
from tqdm import tqdm
from .tokenizer import SQLTokenizer
from .models import TransformerSeq2Seq
from .inference import greedy_decode_transformer, validate_sql_syntax

def compute_exact_match(pred: str, ground_truth: str) -> bool:
    """Check normalized exact match between predicted and target SQL."""
    def clean(s: str) -> str:
        s = s.strip().lower().rstrip(";")
        return " ".join(s.split())
    return clean(pred) == clean(ground_truth)

def compute_simple_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute sentence-level 1-gram & 2-gram overlap precision."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    # 1-gram precision
    pred_1 = set(pred_tokens)
    ref_1 = set(ref_tokens)
    p1 = len(pred_1 & ref_1) / len(pred_1) if pred_1 else 0

    # 2-gram precision
    pred_2 = set(zip(pred_tokens[:-1], pred_tokens[1:]))
    ref_2 = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    p2 = len(pred_2 & ref_2) / len(pred_2) if pred_2 else 0

    return 0.5 * (p1 + p2)

def evaluate_model_on_test_set(
    model: TransformerSeq2Seq,
    test_data: List[Dict[str, str]],
    src_tokenizer: SQLTokenizer,
    tgt_tokenizer: SQLTokenizer,
    max_eval_samples: int = 200,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate trained Text-to-SQL model across Exact Match, Syntax Validity, and BLEU.
    """
    samples = test_data[:max_eval_samples]
    exact_matches = 0
    valid_syntaxes = 0
    bleu_scores = 0.0

    print(f"\n[Evaluating {len(samples)} samples]")
    results = []

    for item in tqdm(samples, desc="Evaluating"):
        src = item["src"]
        tgt = item["tgt"]

        pred = greedy_decode_transformer(model, src, src_tokenizer, tgt_tokenizer, device=device)
        is_em = compute_exact_match(pred, tgt)
        is_valid, _ = validate_sql_syntax(pred)

        pred_tokens = tgt_tokenizer.tokenize(pred.lower())
        tgt_tokens = tgt_tokenizer.tokenize(tgt.lower())
        bleu = compute_simple_bleu(pred_tokens, tgt_tokens)

        if is_em:
            exact_matches += 1
        if is_valid:
            valid_syntaxes += 1
        bleu_scores += bleu

        results.append({
            "src": src,
            "target": tgt,
            "predicted": pred,
            "exact_match": is_em,
            "syntax_valid": is_valid
        })

    n = len(samples)
    em_rate = (exact_matches / n) * 100
    syntax_rate = (valid_syntaxes / n) * 100
    avg_bleu = (bleu_scores / n) * 100

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY ({n} samples):")
    print(f"Exact Match Accuracy: {em_rate:.2f}%")
    print(f"SQL Syntax Validity:  {syntax_rate:.2f}%")
    print(f"Average Token Overlap (BLEU-style): {avg_bleu:.2f}%")
    print(f"{'='*50}\n")

    # Display 5 qualitative samples
    print("Sample Model Predictions vs Ground Truth:")
    for idx, r in enumerate(results[:5]):
        print(f"\n[Sample {idx+1}]")
        print(f"Prompt:    {r['src'][:110]}...")
        print(f"Expected:  {r['target']}")
        print(f"Predicted: {r['predicted']}")
        print(f"EM: {r['exact_match']} | Syntax Valid: {r['syntax_valid']}")

    return {
        "exact_match_rate": em_rate,
        "syntax_valid_rate": syntax_rate,
        "bleu": avg_bleu
    }
