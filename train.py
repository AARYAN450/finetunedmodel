"""Fine-tune roberta-base to classify meeting sentences as action items.

Plain PyTorch loop on purpose: no Trainer API churn, and the dataset is small
enough that CPU training takes a few minutes.

Usage: python train.py
Saves: action_item_model/ , results_roberta.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import SEED, load_split

MODEL_NAME = "roberta-base"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
OUT_DIR = Path(__file__).parent / "action_item_model"
RESULTS = Path(__file__).parent / "results_roberta.json"


def encode(tokenizer, df):
    enc = tokenizer(
        list(df.sentence),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    labels = torch.tensor(df.label.values, dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for input_ids, mask, _ in loader:
        logits = model(
            input_ids=input_ids.to(device), attention_mask=mask.to(device)
        ).logits
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def report(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", pos_label=1, zero_division=0
    )
    return precision, recall, f1, pred


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df = load_split()
    print(f"device: {device}")
    print(f"train {len(train_df)} rows ({train_df.label.sum()} pos)")
    print(f"test  {len(test_df)} rows ({test_df.label.sum()} pos)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    train_loader = DataLoader(
        encode(tokenizer, train_df), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(encode(tokenizer, test_df), batch_size=32)

    # balanced class weights, same idea as class_weight="balanced"
    counts = np.bincount(train_df.label.values, minlength=2)
    weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float).to(device)
    print(f"class weights: {weights.tolist()}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps)
    )

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for i, (input_ids, mask, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids.to(device), attention_mask=mask.to(device)
            ).logits
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if i % 20 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_loader)} loss {running / i:.4f}")
        print(f"epoch {epoch} done, mean loss {running / len(train_loader):.4f}, "
              f"{time.time() - start:.0f}s elapsed")

    probs = predict_probs(model, test_loader, device)
    y_true = test_df.label.values

    precision, recall, f1, pred = report(y_true, probs, 0.5)
    print("\n" + classification_report(y_true, pred, digits=3, zero_division=0))
    print("confusion matrix [[tn fp] [fn tp]]:")
    print(confusion_matrix(y_true, pred))
    print(f"\nACTION-ITEM CLASS @0.50  precision={precision:.3f} "
          f"recall={recall:.3f} f1={f1:.3f}")

    # threshold sweep -- no retraining, just a different operating point
    print("\nthreshold sweep:")
    best = (f1, 0.5, precision, recall)
    for threshold in np.arange(0.2, 0.85, 0.05):
        p, r, f, _ = report(y_true, probs, threshold)
        marker = ""
        if f > best[0]:
            best = (f, float(threshold), p, r)
            marker = "  <- best so far"
        print(f"  {threshold:.2f}  p={p:.3f} r={r:.3f} f1={f:.3f}{marker}")
    print(f"\nbest f1={best[0]:.3f} at threshold {best[1]:.2f} "
          f"(p={best[2]:.3f} r={best[3]:.3f})")

    OUT_DIR.mkdir(exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    RESULTS.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_len": MAX_LEN,
                "lr": LR,
                "device": str(device),
                "train_seconds": round(time.time() - start, 1),
                "at_0.5": {
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                },
                "best_threshold": {
                    "threshold": round(best[1], 2),
                    "precision": round(float(best[2]), 4),
                    "recall": round(float(best[3]), 4),
                    "f1": round(float(best[0]), 4),
                },
                "test_rows": int(len(test_df)),
                "test_positives": int(test_df.label.sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {OUT_DIR.name}/ and {RESULTS.name}")


if __name__ == "__main__":
    main()
