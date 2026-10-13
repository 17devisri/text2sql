import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import prepare_and_cache_dataset, Text2SQLDataset, collate_fn
from src.tokenizer import SQLTokenizer
from src.models import TransformerSeq2Seq
from src.train import train_model
from src.evaluate import evaluate_model_on_test_set

def main():
    parser = argparse.ArgumentParser(description="Train custom Seq2Seq Text-to-SQL Model in PyTorch")
    parser.add_argument("--samples", type=int, default=5000, help="Number of dataset samples to download and use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding & model dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of multi-head attention heads")
    parser.add_argument("--layers", type=int, default=10, help="Number of encoder and decoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum token frequency in vocabulary")
    parser.add_argument("--max_vocab", type=int, default=12000, help="Max vocabulary size")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device: auto, cuda, or cpu")
    parser.add_argument("--rebuild_data", action="store_true", help="Force rebuild dataset cache with new clean schema format")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] Using {device.upper()}")

    # 1. Prepare and cache data
    train_path, val_path = prepare_and_cache_dataset(
        output_dir="data",
        num_samples=args.samples,
        val_split=0.1,
        force_rebuild=args.rebuild_data
    )

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"[Data] Train count: {len(train_data)}, Val count: {len(val_data)}")

    # 2. Build or load unified shared tokenizer
    os.makedirs(args.save_dir, exist_ok=True)
    vocab_path = os.path.join(args.save_dir, "vocab.json")

    # If forcing rebuild or new dataset size, build fresh
    print("[Vocab] Building unified shared vocabulary from both schemas and SQL targets...")
    all_texts = [d["src"] for d in train_data] + [d["tgt"] for d in train_data]

    tokenizer = SQLTokenizer(min_freq=args.min_freq, max_vocab_size=args.max_vocab)
    tokenizer.build_vocab(all_texts)
    tokenizer.save(vocab_path)
    # Also save as src and tgt for backward compatibility
    tokenizer.save(os.path.join(args.save_dir, "src_vocab.json"))
    tokenizer.save(os.path.join(args.save_dir, "tgt_vocab.json"))

    src_tokenizer = tokenizer
    tgt_tokenizer = tokenizer

    print(f"[Vocab] Unified Vocabulary Size: {len(tokenizer):,} (Zero <unk> guaranteed)")

    # 3. Create Datasets and DataLoaders
    train_dataset = Text2SQLDataset(train_data, src_tokenizer, tgt_tokenizer)
    val_dataset = Text2SQLDataset(val_data, src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_tokenizer.PAD_IDX)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_tokenizer.PAD_IDX)
    )

    # 4. Instantiate Transformer Model
    config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "layers": args.layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "src_vocab_size": len(src_tokenizer),
        "tgt_vocab_size": len(tgt_tokenizer)
    }
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=src_tokenizer.PAD_IDX
    )

    best_checkpoint_path = os.path.join(args.save_dir, "best_model.pt")

    if args.eval_only:
        print(f"[Evaluation] Loading weights from {best_checkpoint_path}...")
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        evaluate_model_on_test_set(model, val_data, src_tokenizer, tgt_tokenizer, max_eval_samples=150, device=device)
        return

    # 5. Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device
    )

    # 6. Evaluate Best Checkpoint
    print(f"\n[Final Evaluation] Testing best checkpoint...")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    evaluate_model_on_test_set(model, val_data, src_tokenizer, tgt_tokenizer, max_eval_samples=150, device=device)

if __name__ == "__main__":
    main()
