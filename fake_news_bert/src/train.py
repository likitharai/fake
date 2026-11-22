"""End-to-end training pipeline: EDA -> clean -> split -> fine-tune -> evaluate.

Run with:  python -m src.train
"""
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE, BERT_NAME, EPOCHS, LEARNING_RATE, MAX_LENGTH, METRICS_DIR,
    MODEL_DIR, PATIENCE, PROCESSED_DIR, RANDOM_SEED, SAMPLE_SIZE, TEST_SIZE,
    VAL_SIZE, WARMUP_RATIO, WEIGHT_DECAY,
)
from src.data import (
    clean_dataframe, load_raw_dataset, maybe_subsample, run_eda, stratified_split,
)
from src.dataset import NewsDataset, make_collate_fn
from src.evaluate import compute_metrics, save_json
from src.model import get_device, load_model_for_training, load_trained


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, shuffle: bool) -> DataLoader:
    dataset = NewsDataset(df["input_text"], df["label"])
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer, MAX_LENGTH),
    )


def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        n = batch["labels"].size(0)
        total_loss += loss.item() * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        n = batch["labels"].size(0)
        total_loss += outputs.loss.item() * n
        total_n += n
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return total_loss / max(total_n, 1), y_true, y_pred


def run_training() -> dict:
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")
    for d in (METRICS_DIR, PROCESSED_DIR, MODEL_DIR.parent):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load + EDA on raw data
    raw = load_raw_dataset()
    eda = run_eda(raw)

    # 2. Clean, dedupe, optional subsample, stratified split (test untouched until the end)
    clean_df, duplicates_removed = clean_dataframe(raw)
    clean_rows = len(clean_df)
    clean_df = maybe_subsample(clean_df, SAMPLE_SIZE, RANDOM_SEED)
    train_df, val_df, test_df = stratified_split(clean_df, RANDOM_SEED, TEST_SIZE, VAL_SIZE)

    stats = {
        **eda,
        "clean_rows": int(clean_rows),
        "duplicates_removed": duplicates_removed,
        "sample_size_used": int(len(clean_df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "label_counts_clean": {
            "FAKE": int((clean_df["label"] == 0).sum()),
            "REAL": int((clean_df["label"] == 1).sum()),
        },
        "avg_words_clean": round(float(clean_df["input_text"].str.split().str.len().mean()), 1),
    }
    save_json(stats, METRICS_DIR / "dataset_stats.json")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    # 3. Model + data loaders
    tokenizer, model = load_model_for_training(BERT_NAME)
    model.to(device)
    train_loader = make_loader(train_df, tokenizer, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps
    )

    # 4. Fine-tune with validation-based checkpoint selection and early stopping
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    patience_left = PATIENCE
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, y_true, y_pred = run_inference(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = PATIENCE
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  -> saved best model to {MODEL_DIR}")
        else:
            patience_left -= 1
            if patience_left < 0:
                print("Early stopping: validation loss stopped improving.")
                break

    save_json(history, METRICS_DIR / "training_history.json")

    # 5. Final evaluation of the best checkpoint on unseen test data
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_tokenizer, best_model = load_trained(MODEL_DIR, device)
    test_loader = make_loader(test_df, best_tokenizer, shuffle=False)
    _, y_true, y_pred = run_inference(best_model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["epochs_run"] = len(history["epoch"])
    save_json(metrics, METRICS_DIR / "test_metrics.json")

    print(
        f"TEST  accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    run_training()
