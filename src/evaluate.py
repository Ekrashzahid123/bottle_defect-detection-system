import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_and_split_dataset, CLASS_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pure TensorFlow MobileNetV3 Model")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to .keras model")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train"], help="Split to evaluate")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Directory to save evaluation reports")
    parser.add_argument("--img_size", type=int, default=224, help="Input size")
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 15, "weight": "bold"})
    plt.title(title, fontsize=12, pad=12, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, weight="semibold")
    plt.ylabel("Actual Ground Truth", fontsize=11, weight="semibold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}. Run training first.")
        
    print(f"[*] Loading TensorFlow model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    manifest_path = "models/split_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        paths = manifest["split_paths"][args.split]
        targets = manifest["split_labels"][args.split]
    else:
        split_paths, split_labels = load_and_split_dataset(args.dataset_root)
        paths = split_paths[args.split]
        targets = split_labels[args.split]

    print(f"[*] Evaluating {len(paths)} samples from {args.split.upper()} set (Leak-Free Holdout).")
    
    all_preds = []
    all_targets = []
    all_probs = []
    sample_details = []
    error_samples = []
    
    for img_path, target in zip(paths, targets):
        raw_img = tf.io.read_file(img_path)
        img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [args.img_size, args.img_size])
        img = tf.cast(img, tf.float32)
        img_batch = tf.expand_dims(img, axis=0)
        
        probs = model.predict(img_batch, verbose=0)[0]
        pred = int(np.argmax(probs))
        
        all_preds.append(pred)
        all_targets.append(target)
        all_probs.append(float(probs[1]))
        
        info = {
            "filename": os.path.basename(img_path),
            "image_path": img_path,
            "true_class": CLASS_NAMES[target],
            "predicted_class": CLASS_NAMES[pred],
            "confidence": float(probs[pred]),
            "correct": bool(pred == target)
        }
        sample_details.append(info)
        
        if pred != target:
            info["error_type"] = "False Positive" if pred == 1 else "False Negative"
            error_samples.append(info)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0
        
    print("\n" + "=" * 65)
    print(f"    TENSORFLOW EVALUATION REPORT: {args.split.upper()} SET ({len(paths)} Samples)")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nDetailed Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (Normal classified as Normal):       {cm[0, 0]}")
    print(f"  • False Positives (Normal misclassified as Defective): {cm[0, 1]}")
    print(f"  • False Negatives (Defective misclassified as Normal): {cm[1, 0]}")
    print(f"  • True Positives  (Defective classified as Defective):  {cm[1, 1]}")
    print("=" * 65)
    
    cm_path = os.path.join(args.output_dir, f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix: {args.split.capitalize()} Set ({len(paths)} Samples)", cm_path)
    print(f"[*] Saved Confusion Matrix to: {cm_path}")
    
    results = {
        "framework": "TensorFlow 2.x",
        "split": args.split,
        "sample_count": len(paths),
        "metrics": report,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "error_analysis": {
            "total_errors": len(error_samples),
            "false_positives": len([e for e in error_samples if e["error_type"] == "False Positive"]),
            "false_negatives": len([e for e in error_samples if e["error_type"] == "False Negative"]),
            "error_samples": error_samples
        },
        "all_predictions": sample_details
    }
    
    report_json_path = os.path.join(args.output_dir, f"{args.split}_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Saved Evaluation Report & Error Analysis to: {report_json_path}")
    
    if error_samples:
        print(f"\n[!] Error Analysis ({len(error_samples)} misclassifications):")
        for err in error_samples:
            print(f"  - [{err['error_type']}] File: {err['filename']} | True={err['true_class']} | Pred={err['predicted_class']} (Conf: {err['confidence']:.2%})")
    else:
        print(f"\n[+] Zero Classification Errors on {args.split.capitalize()} Set!")

if __name__ == "__main__":
    main()
