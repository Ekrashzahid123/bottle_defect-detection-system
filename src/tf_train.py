import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tf_dataset import get_tf_data_loaders, CLASS_NAMES
from src.tf_model import create_tf_mobilenet_v3

def parse_args():
    parser = argparse.ArgumentParser(description="Train TensorFlow MobileNetV3 for Bottle Defect Detection")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--variant", type=str, default="small", choices=["small", "large"], help="MobileNetV3 variant")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Root directory of dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Input image dimension")
    return parser.parse_args()

def plot_tf_history(history, save_path):
    epochs = range(1, len(history["loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("TensorFlow Training & Validation Loss", fontsize=12, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse Categorical Cross-Entropy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], "b--", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r-o", label="Val Accuracy")
    plt.title("Training & Validation Accuracy", fontsize=12, weight="bold")
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
    
    print(f"[*] TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[*] Available GPUs: {len(gpus)}")
    
    # 1. Load Data
    train_ds, val_ds, test_ds, class_weights, (tr_p, val_p, test_p) = get_tf_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"[*] Loaded {len(tr_p)} train samples, {len(val_p)} valid samples, {len(test_p)} test samples.")
    print(f"[*] Class Weights: {class_weights}")
    
    # 2. Build MobileNetV3 Model
    model = create_tf_mobilenet_v3(
        num_classes=len(CLASS_NAMES),
        variant=args.variant,
        dropout=0.2,
        input_shape=(args.img_size, args.img_size, 3)
    )
    
    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # 4. Callbacks
    best_model_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 5. Train
    print("\n" + "=" * 60)
    print("      STARTING TENSORFLOW MOBILENETV3 TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.1f}s.")
    print(f"[*] Best model saved to: {best_model_path}")
    
    # 6. Save Learning Curves & Metrics
    curves_path = os.path.join(args.output_dir, "tf_training_curves.png")
    plot_tf_history(history.history, curves_path)
    print(f"[*] Saved TF training curves to: {curves_path}")
    
    metrics_path = os.path.join(args.output_dir, "tf_training_metrics.json")
    with open(metrics_path, "w") as f:
        # Convert float32 numpy to float for JSON
        serializable_hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(serializable_hist, f, indent=4)
    print(f"[*] Saved TF training metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
