import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import get_data_loaders, CLASS_NAMES
from src.model import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 for Bottle Defect Detection")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--variant", type=str, default="small", choices=["small", "large"], help="MobileNetV3 variant")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    
    # Class-specific metrics for Defective (class 1)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(all_targets, all_preds, labels=[0, 1], zero_division=0)

    return val_loss, val_acc, precision, recall, f1, (p_cls, r_cls, f1_cls)

def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # F1 / Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b--", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r--", label="Val Acc")
    plt.plot(epochs, history["val_f1"], "g-o", label="Val F1 (Macro)")
    plt.title("Accuracy & F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # 1. Load Data
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"[*] Loaded {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} valid samples.")
    print(f"[*] Class Weights (Normal vs Defective): {class_weights.numpy()}")
    
    # 2. Build Model
    model = create_model(num_classes=2, variant=args.variant, pretrained=True, dropout=0.2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "val_precision": [], "val_recall": []
    }
    
    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^8} | {'Val Recall':^10} | {'Val F1':^8}")
    print("=" * 70)
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, (p_cls, r_cls, f1_cls) = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        
        print(f"{epoch:^7d} | {tr_loss:^10.4f} | {val_loss:^10.4f} | {val_acc:^8.2%} | {val_rec:^10.2%} | {val_f1:^8.4f}")
        
        # Save best model based on F1-Score
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "variant": args.variant,
                "class_names": CLASS_NAMES,
                "img_size": args.img_size
            }, best_checkpoint_path)
            
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"[*] Training finished in {total_time:.1f}s. Best Validation F1: {best_val_f1:.4f}")
    print(f"[*] Saved best model checkpoint to: {best_checkpoint_path}")
    
    # Save training curves & metrics
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_history(history, curves_path)
    print(f"[*] Saved training curves to: {curves_path}")
    
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[*] Saved training metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
