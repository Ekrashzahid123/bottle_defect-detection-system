import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_and_split_dataset, find_dataset_root, resolve_image_path, CLASS_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pure TensorFlow MobileNetV3 Model")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to .keras model")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train", "all"], help="Split to evaluate")
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

def evaluate_single_split(model, split_name: str, paths: list, targets: list, output_dir: str, dataset_root: str, img_size: int = 224):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[*] Evaluating {len(paths)} samples from {split_name.upper()} set (Leak-Free Holdout)...")
    
    all_preds = []
    all_targets = []
    all_probs = []
    sample_details = []
    error_samples = []
    
    for img_path, target in zip(paths, targets):
        resolved_path = resolve_image_path(img_path, dataset_root)
        if not os.path.exists(resolved_path):
            print(f"[!] Warning: Image not found at {img_path} (resolved: {resolved_path}), skipping.")
            continue

        try:
            raw_img = tf.io.read_file(resolved_path)
            img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32)
            img_batch = tf.expand_dims(img, axis=0)
            
            probs = model.predict(img_batch, verbose=0)[0]
            pred = int(np.argmax(probs))
            
            all_preds.append(pred)
            all_targets.append(target)
            all_probs.append(float(probs[1]))
            
            info = {
                "filename": os.path.basename(resolved_path),
                "image_path": resolved_path,
                "true_class": CLASS_NAMES[target],
                "predicted_class": CLASS_NAMES[pred],
                "confidence": float(probs[pred]),
                "correct": bool(pred == target),
                "is_defective": bool(pred == 1),
                "prob_normal": float(probs[0]),
                "prob_defective": float(probs[1])
            }
            sample_details.append(info)
            
            if pred != target:
                info["error_type"] = "False Positive" if pred == 1 else "False Negative"
                error_samples.append(info)
        except Exception as e:
            print(f"[!] Error processing {resolved_path}: {e}")
            
    if not all_preds:
        print(f"[!] No valid predictions generated for split: {split_name}")
        return None

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0
        
    print("=" * 65)
    print(f"    TENSORFLOW EVALUATION REPORT: {split_name.upper()} SET ({len(all_preds)} Valid Samples)")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nDetailed Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (Normal classified as Normal):       {cm[0, 0]}")
    print(f"  • False Positives (Normal misclassified as Defective): {cm[0, 1]}")
    print(f"  • False Negatives (Defective misclassified as Normal): {cm[1, 0]}")
    print(f"  • True Positives  (Defective classified as Defective):  {cm[1, 1]}")
    print("=" * 65)
    
    # Save Confusion Matrix Plot
    cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix: {split_name.capitalize()} Set ({len(all_preds)} Samples)", cm_path)
    print(f"[*] Saved Confusion Matrix to: {cm_path}")
    
    # Save CSV of All Predictions
    csv_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    pd.DataFrame(sample_details).to_csv(csv_path, index=False)
    print(f"[*] Saved Predictions CSV to: {csv_path}")

    results = {
        "framework": "TensorFlow 2.x",
        "split": split_name,
        "sample_count": len(all_preds),
        "accuracy": float(np.mean(all_preds == all_targets)),
        "metrics": report,
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "error_analysis": {
            "total_errors": len(error_samples),
            "false_positives": len([e for e in error_samples if e.get("error_type") == "False Positive"]),
            "false_negatives": len([e for e in error_samples if e.get("error_type") == "False Negative"]),
            "error_samples": error_samples
        },
        "all_predictions": sample_details
    }
    
    report_json_path = os.path.join(output_dir, f"{split_name}_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Saved Evaluation Report & Error Analysis to: {report_json_path}")
    
    return results

def run_evaluation(model_path: str = "models/best_model.keras",
                   dataset_root: str = "dataset",
                   split: str = "test",
                   output_dir: str = "metrics",
                   img_size: int = 224):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset_root = find_dataset_root(dataset_root)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run training first.")
        
    print(f"[*] Loading TensorFlow model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    manifest_path = "models/split_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        split_paths = manifest["split_paths"]
        split_labels = manifest["split_labels"]
    else:
        split_paths, split_labels = load_and_split_dataset(dataset_root)
    
    splits_to_eval = ["test", "valid"] if split == "all" else [split]
    
    eval_results = {}
    for s in splits_to_eval:
        if s in split_paths:
            res = evaluate_single_split(
                model=model,
                split_name=s,
                paths=split_paths[s],
                targets=split_labels[s],
                output_dir=output_dir,
                dataset_root=dataset_root,
                img_size=img_size
            )
            eval_results[s] = res

    return eval_results

def main():
    args = parse_args()
    run_evaluation(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        split=args.split,
        output_dir=args.output_dir,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()
