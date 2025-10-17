from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
import os
from tqdm import tqdm
from datasets.ticket_dataset import TicketDataset
from sklearn.metrics import accuracy_score, f1_score


def map3(y_true, y_pred):
    m = (y_true.reshape((-1,1)) == y_pred)
    return np.mean(np.where(m.any(axis=1), m.argmax(axis=1)+1, np.inf)**(-1))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions_sorted = np.argsort(-logits, axis=1)[:, :3]
    preds = logits.argmax(axis=1)
    return {'map3': map3(labels, predictions_sorted),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='weighted'),}

# --- Utility ---
def format_float(val, sci_thresh=1e-3):
    """Format floats in scientific notation if small, else 3 decimal places."""
    if val == 0:
        return "0"
    if abs(val) < sci_thresh:
        return f"{val:.0e}"  # e.g. 3e-05
    return f"{val:.3f}".rstrip("0").rstrip(".")  # e.g. 0.01 → 0.01, 0.1 → 0.1


def generate_run_name(
    model_name="deberta",
    task="ticketclass",
    batch_size=16,
    lr=2e-5,
    num_epochs=3,
    gradient_accumulation=1,
    weight_decay=0.01,
    scheduler="cosine",
    **kwargs
):
    now = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = (
        f"{model_name}_{task}"
        f"_bs{batch_size}"
        f"_lr{format_float(lr)}"
        f"_ep{num_epochs}"
        f"_ga{gradient_accumulation}"
        f"_wd{format_float(weight_decay)}"
        f"_{scheduler}"
        f"_{now}"
    )
    return run_name

def evaluate(model, val_loader, device, compute_metrics, global_step, use_wandb=False):
    model.eval()
    val_logits, val_labels = [], []

    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f"Eval @ step {global_step}"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            val_outputs = model(**val_batch)
            val_logits.append(val_outputs.logits.cpu().numpy())
            val_labels.append(val_batch["labels"].cpu().numpy())

    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    metrics = compute_metrics((val_logits, val_labels))
    metrics["step"] = global_step

    if use_wandb:
        import wandb
        wandb.log(metrics)

    return metrics


# --- Main training function ---
def train(
    train_csv=None,
    val_csv=None,
    model_name="microsoft/deberta-v3-base",
    task="ticketclass",
    project="ticket-classifier",
    batch_size=8,
    lr=3e-5,
    weight_decay=0.01,
    gradient_accumulation=4,
    num_epochs=3,
    scheduler="cosine",
    warmup_ratio=0.05,
    save_steps=2250,
    checkpoint_dir="model_checkpoints",
    use_wandb=False,
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_csv is None or val_csv is None:
        raise ValueError("Must provide --train_csv and --val_csv")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = train_df["label"].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    scaler = GradScaler()
    rolling_losses = deque(maxlen=500)
    global_step = 0
    best_map3 = -float("inf")

    train_dataset = TicketDataset(train_df, tokenizer)
    val_dataset = TicketDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = (num_epochs * len(train_loader)) // gradient_accumulation
    lr_scheduler = get_scheduler(
        scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )

    if use_wandb:
        import wandb
        wandb.init(
            project=project,
            name=generate_run_name(**locals()),  # locals() is fine here since args are explicit
        )


    # --- Training Loop ---
    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        processed = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation

            total_loss += loss.item()
            rolling_losses.append(loss.item() * gradient_accumulation)

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                processed += 1

                current_lr = optimizer.param_groups[0]["lr"]

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/step_loss": total_loss / processed,
                        "train/rolling_loss": np.mean(rolling_losses),
                        "train/lr": current_lr,
                        "step": global_step,
                    })

                progress_bar.set_postfix(
                    loss=total_loss / processed,
                    rolling_loss=f"{(sum(rolling_losses)/len(rolling_losses)):.4f}",
                )
                global_step += 1

                # --- Eval + Save ---
                if global_step % save_steps == 0:
                    metrics = evaluate(
                        model, val_loader, device, compute_metrics, global_step, use_wandb=use_wandb
                    )
                    print(f"[Step {global_step}] val_acc={metrics['accuracy']:.4f}, "
                          f"val_f1={metrics['f1']:.4f}, val_map3={metrics['map3']:.4f}")

                    if metrics["map3"] > best_map3:
                        best_map3 = metrics["map3"]
                        save_path = f"{checkpoint_dir}/best"
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(model.state_dict(), f"{save_path}/model.pt")
                        torch.save(optimizer.state_dict(), f"{save_path}/optimizer.pt")
                        torch.save(lr_scheduler.state_dict(), f"{save_path}/scheduler.pt")
                        torch.save(scaler.state_dict(), f"{save_path}/scaler.pt")
                        print(f"✅ Saved new best model with map3={best_map3:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        if use_wandb:
            import wandb
            wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})
