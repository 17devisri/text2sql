import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .models import TransformerSeq2Seq, LSTMSeq2Seq
from .tokenizer import SQLTokenizer

def train_epoch_transformer(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_norm: float = 1.0,
    device: str = "cpu"
) -> Tuple[float, float]:
    """Train Transformer for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        src = batch["src"].to(device)      # (batch, src_len)
        tgt = batch["tgt"].to(device)      # (batch, tgt_len)

        # Teacher forcing inputs and targets
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)     # (batch, tgt_len - 1, tgt_vocab_size)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        # Token accuracy (ignoring pad tokens)
        preds = output.argmax(dim=-1)
        mask = tgt_target != model.pad_idx
        num_valid = mask.sum().item()
        num_correct = ((preds == tgt_target) & mask).sum().item()

        total_tokens += num_valid
        correct_tokens += num_correct
        total_loss += loss.item() * num_valid

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(num_correct / (num_valid + 1e-8)):.3f}"})

    avg_loss = total_loss / (total_tokens + 1e-8)
    token_acc = correct_tokens / (total_tokens + 1e-8)
    return avg_loss, token_acc

def evaluate_epoch_transformer(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float, float]:
    """Evaluate Transformer on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))

            preds = output.argmax(dim=-1)
            mask = tgt_target != model.pad_idx
            num_valid = mask.sum().item()
            num_correct = ((preds == tgt_target) & mask).sum().item()

            total_tokens += num_valid
            correct_tokens += num_correct
            total_loss += loss.item() * num_valid

    avg_loss = total_loss / (total_tokens + 1e-8)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent overflow
    token_acc = correct_tokens / (total_tokens + 1e-8)
    return avg_loss, perplexity, token_acc

def train_model(
    model: TransformerSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    save_dir: str = "checkpoints",
    device: str = "cpu"
) -> Dict[str, list]:
    """Full training pipeline with checkpoint saving and history tracking."""
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(save_dir, "best_model.pt")

    print(f"\n{'='*60}")
    print(f"Starting Training on {device.upper()} for {epochs} epochs")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch_transformer(
            model, train_loader, optimizer, criterion, clip_norm=clip_norm, device=device
        )
        val_loss, val_ppl, val_acc = evaluate_epoch_transformer(
            model, val_loader, criterion, device=device
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%, PPL: {val_ppl:.2f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {val_loss:.4f})")

    # Also save the final epoch model
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[Training Complete] Best model saved to {best_checkpoint_path}")
    return history
