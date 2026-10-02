import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BottleDataset, get_transforms, CLASS_NAMES
from src.model import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained MobileNetV3 Model on Test / Valid Sets")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train"], help="Split to evaluate")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Directory to save evaluation reports")
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 15, "weight": "bold"})
    plt.title(title, fontsize=13, pad=12, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, weight="semibold")
    plt.ylabel("Actual Ground Truth", fontsize=11, weight="semibold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_split(model, dataset, split_name, device, output_dir):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    sample_details = []
    error_samples = []
    
    with torch.no_grad():
        for img, label, img_path in loader:
            img = img.to(device)
            output = model(img)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            true_label = int(label.item())
            
            all_preds.append(pred)
            all_targets.append(true_label)
            all_probs.append(probs[1]) # Prob of defective
            
            sample_info = {
                "filename": os.path.basename(img_path[0]),
                "image_path": img_path[0],
                "true_class": CLASS_NAMES[true_label] if true_label in [0, 1] else "Unknown",
                "predicted_class": CLASS_NAMES[pred],
                "confidence": float(probs[pred]),
                "prob_normal": float(probs[0]),
                "prob_defective": float(probs[1]),
                "correct": bool(pred == true_label)
            }
            sample_details.append(sample_info)
            
            if pred != true_label and true_label in [0, 1]:
                sample_info["error_type"] = "False Positive" if pred == 1 else "False Negative"
                error_samples.append(sample_info)
                
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
    print(f"       OFFICIAL EVALUATION REPORT: {split_name.upper()} SET ({len(dataset)} Samples)")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nDetailed Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (Normal classified as Normal):       {cm[0, 0]}")
    print(f"  • False Positives (Normal misclassified as Defective): {cm[0, 1]}")
    print(f"  • False Negatives (Defective misclassified as Normal): {cm[1, 0]}")
    print(f"  • True Positives  (Defective classified as Defective):  {cm[1, 1]}")
    print("=" * 65)
    
    cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix: {split_name.capitalize()} Set ({len(dataset)} Bottles)", cm_path)
    print(f"[*] Saved Confusion Matrix to: {cm_path}")
    
    results = {
        "split": split_name,
        "sample_count": len(dataset),
        "metrics": report,
        "roc_auc": auc,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "error_analysis": {
            "total_errors": len(error_samples),
            "false_positives": len([e for e in error_samples if e["error_type"] == "False Positive"]),
            "false_negatives": len([e for e in error_samples if e["error_type"] == "False Negative"]),
            "error_samples": error_samples
        },
        "all_predictions": sample_details
    }
    
    json_path = os.path.join(output_dir, f"{split_name}_evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Saved Evaluation Report & Error Analysis to: {json_path}")
    
    if error_samples:
        print(f"\n[!] Error Analysis ({len(error_samples)} misclassifications):")
        for err in error_samples:
            print(f"  - [{err['error_type']}] File: {err['filename']} | True={err['true_class']} | Pred={err['predicted_class']} (Conf: {err['confidence']:.2%})")
    else:
        print(f"\n[+] Zero Classification Errors on {split_name.capitalize()} Set! (100% Generalization)")
        
    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.model_path}. Please train model first.")
        
    checkpoint = torch.load(args.model_path, map_location=device)
    variant = checkpoint.get("variant", "small")
    img_size = checkpoint.get("img_size", 224)
    
    model = create_model(num_classes=2, variant=variant, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    _, eval_tf = get_transforms(img_size)
    
    # Evaluate requested split
    dataset = BottleDataset(args.dataset_root, split=args.split, transform=eval_tf)
    evaluate_split(model, dataset, args.split, device, args.output_dir)

if __name__ == "__main__":
    main()
