import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import get_data_loaders, CLASS_NAMES
from src.model import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pure TensorFlow MobileNetV3 for Bottle Defect Detection")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--variant", type=str, default="small", choices=["small", "large"], help="MobileNetV3 variant")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for leak-free split")
    return parser.parse_args()

def plot_history(history, save_path):
    epochs = range(1, len(history["loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("TensorFlow Training & Validation Loss", fontsize=12, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], "b--", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r-o", label="Val Accuracy")
    plt.title("TensorFlow Training & Validation Accuracy", fontsize=12, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"[*] TensorFlow Version: {tf.__version__}")
    
    # 1. Load Leak-Free Grouped Split Dataset
    train_ds, val_ds, test_ds, class_weights, split_paths, split_labels = get_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed
    )
    
    print(f"[*] Leak-Free Dataset Split Statistics:")
    print(f"    - Train Set:      {len(split_paths['train'])} samples (Normal: {sum(1 for l in split_labels['train'] if l==0)}, Defective: {sum(1 for l in split_labels['train'] if l==1)})")
    print(f"    - Validation Set: {len(split_paths['valid'])} samples (Normal: {sum(1 for l in split_labels['valid'] if l==0)}, Defective: {sum(1 for l in split_labels['valid'] if l==1)})")
    print(f"    - Test Set:       {len(split_paths['test'])} samples (Normal: {sum(1 for l in split_labels['test'] if l==0)}, Defective: {sum(1 for l in split_labels['test'] if l==1)})")
    print(f"[*] Balanced Class Weights: {class_weights}")
    
    # Save split manifest for reproducible evaluation
    split_info_path = os.path.join(args.output_dir, "split_manifest.json")
    with open(split_info_path, "w") as f:
        json.dump({
            "train_samples": len(split_paths["train"]),
            "valid_samples": len(split_paths["valid"]),
            "test_samples": len(split_paths["test"]),
            "split_paths": split_paths,
            "split_labels": split_labels
        }, f, indent=2)
    print(f"[*] Saved Leak-Free Split Manifest to: {split_info_path}")

    # 2. Build Model
    model = create_model(
        num_classes=len(CLASS_NAMES),
        variant=args.variant,
        dropout=0.25,
        input_shape=(args.img_size, args.img_size, 3)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_accuracy", mode="max"):
        super().__init__()
        self.filepath = os.path.abspath(filepath)
        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        is_best = (val >= self.best_score) if self.mode == "max" else (val <= self.best_score)
        if is_best:
            self.best_score = val
            tmp_path = self.filepath + ".tmp.keras"
            try:
                self.model.save(tmp_path)
                if os.path.exists(self.filepath):
                    try:
                        os.remove(self.filepath)
                    except Exception:
                        pass
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, self.filepath)
                print(f"\nEpoch {epoch+1}: {self.monitor} reached {val:.5f}, saved best checkpoint.")
            except Exception as e:
                try:
                    self.model.save(self.filepath)
                except Exception as ex:
                    print(f"\n[!] Warning: Model save exception: {ex}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"[*] TensorFlow Version: {tf.__version__}")
    
    # 1. Load Dataset
    train_ds, val_ds, test_ds, class_weights, split_paths, split_labels = get_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        use_direct_split=True
    )
    
    print(f"[*] Dataset Split Statistics:")
    print(f"    - Train Set:      {len(split_paths['train'])} samples (Normal: {sum(1 for l in split_labels['train'] if l==0)}, Defective: {sum(1 for l in split_labels['train'] if l==1)})")
    print(f"    - Validation Set: {len(split_paths['valid'])} samples (Normal: {sum(1 for l in split_labels['valid'] if l==0)}, Defective: {sum(1 for l in split_labels['valid'] if l==1)})")
    print(f"    - Test Set:       {len(split_paths['test'])} samples (Normal: {sum(1 for l in split_labels['test'] if l==0)}, Defective: {sum(1 for l in split_labels['test'] if l==1)})")
    print(f"[*] Balanced Class Weights: {class_weights}")
    
    # Save split manifest for reproducible evaluation
    split_info_path = os.path.join(args.output_dir, "split_manifest.json")
    with open(split_info_path, "w") as f:
        json.dump({
            "train_samples": len(split_paths["train"]),
            "valid_samples": len(split_paths["valid"]),
            "test_samples": len(split_paths["test"]),
            "split_paths": split_paths,
            "split_labels": split_labels
        }, f, indent=2)
    print(f"[*] Saved Split Manifest to: {split_info_path}")

    # 2. Build Model
    model = create_model(
        num_classes=len(CLASS_NAMES),
        variant=args.variant,
        dropout=0.25,
        input_shape=(args.img_size, args.img_size, 3)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
    # 3. Callbacks
    best_model_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        SafeModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 4. Train
    print("\n" + "=" * 65)
    print("      TRAINING TENSORFLOW MOBILENETV3 ON BOTTLE DEFECT DATASET")
    print("=" * 65)
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    total_time = time.time() - start_time
    print(f"\n[*] Training finished in {total_time:.1f}s.")
    
    # Ensure final model is saved if callback did not trigger
    if not os.path.exists(best_model_path):
        model.save(best_model_path)
    print(f"[*] Best model checkpoint confirmed at: {best_model_path}")
    
    # 5. Plot & Save History
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_history(history.history, curves_path)
    print(f"[*] Saved training curves to: {curves_path}")
    
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=4)
    print(f"[*] Saved training metrics to: {metrics_path}")

    # 6. Auto-evaluate on Test and Valid Sets
    from src.evaluate import run_evaluation
    print("\n[*] Running post-training evaluation across splits...")
    run_evaluation(
        model_path=best_model_path,
        dataset_root=args.dataset_root,
        split="all",
        output_dir="metrics",
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()
