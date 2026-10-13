import os
import json
import torch
import argparse
from src.tokenizer import SQLTokenizer
from src.models import TransformerSeq2Seq
from src.inference import predict_sql, validate_sql_syntax

def load_trained_model(checkpoint_dir: str = "checkpoints", device: str = "cpu"):
    src_vocab = os.path.join(checkpoint_dir, "src_vocab.json")
    tgt_vocab = os.path.join(checkpoint_dir, "tgt_vocab.json")
    weights = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model checkpoint not found at {weights}. Please run `python run_train.py` first.")

    src_tok = SQLTokenizer.load(src_vocab)
    tgt_tok = SQLTokenizer.load(tgt_vocab)

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        d_model = cfg.get("d_model", 256)
        num_heads = cfg.get("num_heads", 8)
        layers = cfg.get("layers", 3)
        d_ff = cfg.get("d_ff", 512)
        dropout = cfg.get("dropout", 0.1)
    else:
        d_model, num_heads, layers, d_ff, dropout = 256, 8, 3, 512, 0.1

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_tok),
        tgt_vocab_size=len(tgt_tok),
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=src_tok.PAD_IDX
    )

    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, src_tok, tgt_tok

def run_interactive_demo(device_choice: str = "auto"):
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    print("=" * 60)
    print("        TEXT-TO-SQL PYTORCH MODEL DEMO")
    print("=" * 60)

    try:
        model, src_tok, tgt_tok = load_trained_model("checkpoints", device=device)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"\n[Notice] {e}")
        print("Tip: Run `python run_train.py --samples 2000 --epochs 5` to quickly train a model.\n")
        return

    # Pre-canned examples for immediate test
    examples = [
        {
            "q": "What is the total salary of employees in Engineering?",
            "schema": "CREATE TABLE employees (emp_id INT, name TEXT, department TEXT, salary REAL);"
        },
        {
            "q": "Find all students whose age is greater than 20 ordered by grade descending",
            "schema": "CREATE TABLE students (id INT, name TEXT, age INT, grade TEXT);"
        },
        {
            "q": "Count how many orders were placed by customer 105",
            "schema": "CREATE TABLE orders (order_id INT, customer_id INT, total_price REAL, order_date DATE);"
        }
    ]

    print("\n--- Running Demo on Pre-built Examples ---")
    for idx, ex in enumerate(examples, 1):
        print(f"\n[Example {idx}]")
        print(f"Question: {ex['q']}")
        print(f"Schema:   {ex['schema']}")

        sql_greedy = predict_sql(model, ex['q'], ex['schema'], src_tok, tgt_tok, use_beam_search=False, device=device)
        sql_beam = predict_sql(model, ex['q'], ex['schema'], src_tok, tgt_tok, use_beam_search=True, beam_width=3, device=device)
        is_valid, err = validate_sql_syntax(sql_greedy)

        print(f"Predicted SQL (Greedy): {sql_greedy}")
        print(f"Predicted SQL (Beam):   {sql_beam}")
        print(f"Syntax Valid in SQLite: {'YES' if is_valid else f'NO ({err})'}")

    print("\n" + "=" * 60)
    print("Interactive Mode (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            q = input("\nEnter Question (or 'exit'): ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            schema = input("Enter Table Schema (e.g. CREATE TABLE t (col1 INT, col2 TEXT);): ").strip()
            if not schema:
                schema = "CREATE TABLE records (id INT, name TEXT, value REAL);"

            sql = predict_sql(model, q, schema, src_tok, tgt_tok, use_beam_search=True, beam_width=3, device=device)
            is_valid, _ = validate_sql_syntax(sql)

            print(f"\nGenerated SQL: {sql}")
            print(f"SQLite Valid:  {'YES' if is_valid else 'NO'}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Text-to-SQL Interactive Demo")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device: auto, cuda, or cpu")
    args = parser.parse_args()
    run_interactive_demo(device_choice=args.device)
