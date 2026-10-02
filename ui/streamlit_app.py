import os
import sys
import glob
import json
import time
from PIL import Image
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predictor import BottleDefectPredictor
from src.dataset import find_dataset_root, resolve_image_path, CLASS_NAMES

# Page Setup
st.set_page_config(
    page_title="Bottle Defect Inspection System (TensorFlow 2.x)",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-normal {
        background-color: #DCFCE7;
        color: #15803D;
        border: 2px solid #86EFAC;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: bold;
    }
    .status-defective {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 2px solid #FCA5A5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    model_path = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "best_model.keras"))
    return BottleDefectPredictor(model_path=model_path)

predictor = get_model()
dataset_dir = find_dataset_root(os.path.join(PROJECT_ROOT, "dataset"))
metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(metrics_dir, exist_ok=True)

# Sidebar
st.sidebar.title("🍾 System Monitor")
st.sidebar.success("✅ Framework: TensorFlow 2.x")
st.sidebar.info(f"⚙️ Backbone: MobileNetV3 ({predictor.device})")
st.sidebar.caption("Leak-Free Group Split & Pure YOLO Label Parsing")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Inspection Classes")
st.sidebar.markdown("""
- **Normal (0)**: Good cap, properly sealed bottle.
- **Defective (1)**: Open cap, missing cap, defective seal.
- **Processing Time**: < 20 ms / product.
""")

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 Dataset Directory: `{os.path.basename(dataset_dir)}`")
st.sidebar.caption(f"📁 Metrics Saved: `{os.path.relpath(metrics_dir, PROJECT_ROOT)}`")

# Main Title
st.markdown('<div class="main-title">Industrial Bottle Visual Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% TensorFlow 2.x & MobileNetV3 Computer Vision Quality Control</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Live Single Inspection", "📦 Leak-Free Test Evaluation", "📊 Model Performance & Analytics"])

# TAB 1: Live Single Inspection
with tabs[0]:
    st.markdown("### Inspect Bottle Product")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        input_source = st.radio("Select Image Source:", ["Upload Custom Image", "Pick Sample from Dataset"], horizontal=True)
        selected_img = None
        
        if input_source == "Upload Custom Image":
            uploaded_file = st.file_uploader("Upload a bottle image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_img = Image.open(uploaded_file).convert("RGB")
        else:
            sample_candidates = []
            for sp in ["test", "valid", "train"]:
                sp_img_dir = os.path.join(dataset_dir, sp, "images")
                if os.path.exists(sp_img_dir):
                    sample_candidates.extend(glob.glob(os.path.join(sp_img_dir, "*.*")))

            if sample_candidates:
                sample_map = {os.path.basename(p): p for p in sample_candidates[:50]}
                chosen_name = st.selectbox("Choose Sample Image from Dataset:", list(sample_map.keys()))
                selected_path = sample_map[chosen_name]
                selected_img = Image.open(selected_path).convert("RGB")
                st.caption(f"File: `{chosen_name}` | Path: `{os.path.relpath(selected_path, PROJECT_ROOT)}`")
            else:
                st.warning("No sample images found in dataset folder.")

        if selected_img is not None:
            st.image(selected_img, caption="Product Inspection View", use_column_width=True)

    with col2:
        st.markdown("### Quality Inspection Result")
        if selected_img is not None:
            with st.spinner("Running TensorFlow inference..."):
                result = predictor.predict(selected_img)

            # Prediction Status Banner
            if result["is_defective"]:
                st.markdown(f'<div class="status-defective">🚨 DEFECTIVE PRODUCT DETECTED<br><span style="font-size: 1rem; font-weight: normal;">Confidence: {result["confidence"]:.1%}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-normal">✅ NORMAL / PASSED QUALITY INSPECTION<br><span style="font-size: 1rem; font-weight: normal;">Confidence: {result["confidence"]:.1%}</span></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted State", result["prediction"])
            m2.metric("Confidence", f"{result['confidence']:.2%}")
            m3.metric("Latency", f"{result['inference_time_ms']} ms")

            st.markdown("#### Class Probability Distribution")
            probs_df = pd.DataFrame({
                "Class": list(result["probabilities"].keys()),
                "Probability": [v * 100 for v in result["probabilities"].values()]
            })
            st.bar_chart(probs_df.set_index("Class"), height=200)

            with st.expander("View Raw Inference Payload (FastAPI Standard)"):
                st.json(result)
        else:
            st.info("👈 Upload an image or select a sample from the left panel to begin inspection.")

# TAB 2: Batch Leak-Free Test Evaluation
with tabs[1]:
    st.markdown("### Holdout Test Set Evaluation (Leak-Free)")
    
    manifest_path = os.path.join(models_dir, "split_manifest.json")
    saved_csv_path = os.path.join(metrics_dir, "test_predictions.csv")
    saved_json_path = os.path.join(metrics_dir, "test_evaluation_report.json")
    
    # Auto-load persisted results from disk into session_state if available
    if "test_results_df" not in st.session_state:
        if os.path.exists(saved_csv_path):
            st.session_state["test_results_df"] = pd.read_csv(saved_csv_path)
            st.session_state["evaluation_saved_source"] = "Loaded from saved disk cache"
        else:
            st.session_state["test_results_df"] = None

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        test_images = manifest["split_paths"]["test"]
        test_labels = manifest["split_labels"]["test"]
        st.write(f"Dataset split contains **{len(test_images)}** holdout test samples with **zero data leakage**.")
    else:
        test_images = []
        test_labels = []

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1, 1])
    with c_btn1:
        run_batch = st.button("🚀 Run / Refresh Batch Test Evaluation", use_container_width=True, type="primary")
    
    if run_batch and test_images:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        for idx, (img_p, true_lbl) in enumerate(zip(test_images, test_labels)):
            real_path = resolve_image_path(img_p, dataset_dir)
            if not os.path.exists(real_path):
                continue
            res = predictor.predict(real_path)
            pred_lbl = res["class_id"]
            
            item = {
                "filename": os.path.basename(real_path),
                "image_path": real_path,
                "true_class": "Defective" if true_lbl == 1 else "Normal",
                "predicted_class": res["prediction"],
                "confidence": res["confidence"],
                "correct": bool(pred_lbl == true_lbl),
                "is_defective": bool(res["is_defective"]),
                "prob_normal": res["probabilities"].get("Normal", 0.0),
                "prob_defective": res["probabilities"].get("Defective", 0.0),
                "inference_time_ms": res["inference_time_ms"]
            }
            results_list.append(item)
            progress_bar.progress((idx + 1) / len(test_images))
            status_text.caption(f"Evaluating {idx + 1}/{len(test_images)}: `{os.path.basename(real_path)}`")
            
        status_text.empty()
        progress_bar.empty()
        
        df_new = pd.DataFrame(results_list)
        st.session_state["test_results_df"] = df_new
        st.session_state["evaluation_saved_source"] = "Evaluated & saved just now"
        
        # Save to disk permanently
        df_new.to_csv(saved_csv_path, index=False)
        
        # Save JSON evaluation report
        acc = float(df_new["correct"].mean())
        json_report = {
            "framework": "TensorFlow 2.x",
            "split": "test",
            "sample_count": len(df_new),
            "accuracy": acc,
            "defects_detected": int(df_new["is_defective"].sum()),
            "all_predictions": results_list
        }
        with open(saved_json_path, "w") as jf:
            json.dump(json_report, jf, indent=4)
            
        st.success(f"✅ Evaluation completed and results saved to `{saved_csv_path}` and `{saved_json_path}`!")

    # Display results if available in session state or disk
    df_results = st.session_state.get("test_results_df")
    
    if df_results is not None and not df_results.empty:
        total = len(df_results)
        correct_count = int(df_results["correct"].sum())
        acc = (correct_count / total) * 100
        defective_count = int(df_results["is_defective"].sum())
        avg_latency = float(df_results["inference_time_ms"].mean()) if "inference_time_ms" in df_results else 0.0
        misclassified_count = total - correct_count

        st.markdown(f"*{st.session_state.get('evaluation_saved_source', 'Persisted Evaluation Data')}*")
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Tested Samples", total)
        b2.metric("Overall Accuracy", f"{acc:.2f}%")
        b3.metric("Defects Flagged", defective_count)
        b4.metric("Avg Latency / Sample", f"{avg_latency:.2f} ms" if avg_latency > 0 else "< 20 ms")

        # Download options
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        with c_btn2:
            st.download_button("📥 Download Results (CSV)", data=csv_bytes, file_name="holdout_test_predictions.csv", mime="text/csv", use_container_width=True)
        
        if os.path.exists(saved_json_path):
            with open(saved_json_path, "r") as jf:
                json_str = jf.read()
            with c_btn3:
                st.download_button("📥 Download Report (JSON)", data=json_str, file_name="holdout_test_report.json", mime="application/json", use_container_width=True)

        st.markdown("---")
        
        # Gallery with interactive filtering
        st.markdown("#### Sample Predictions Gallery & Inspection")
        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            filter_choice = st.selectbox(
                "Filter Samples:",
                ["All Samples", "Defective Only", "Normal Only", f"Misclassified / Errors Only ({misclassified_count})"]
            )
        
        filtered_df = df_results.copy()
        if filter_choice == "Defective Only":
            filtered_df = filtered_df[filtered_df["predicted_class"] == "Defective"]
        elif filter_choice == "Normal Only":
            filtered_df = filtered_df[filtered_df["predicted_class"] == "Normal"]
        elif "Misclassified" in filter_choice:
            filtered_df = filtered_df[filtered_df["correct"] == False]

        st.caption(f"Showing **{len(filtered_df)}** images ({filter_choice}):")
        
        gallery_cols = st.columns(4)
        for i, (_, r) in enumerate(filtered_df.iterrows()):
            col = gallery_cols[i % 4]
            with col:
                img_path = resolve_image_path(str(r["image_path"]), dataset_dir)
                if os.path.exists(img_path):
                    thumb = Image.open(img_path).resize((220, 220))
                    status_icon = "✅" if r["correct"] else "🚨 MISCLASSIFIED"
                    caption = f"{r['filename']}\n{status_icon} | Pred: {r['predicted_class']} ({float(r['confidence']):.1%})\nTrue: {r['true_class']}"
                    st.image(thumb, caption=caption, use_column_width=True)
                else:
                    st.warning(f"Image not found: {r['filename']}")
    else:
        st.info("👆 Click **'🚀 Run / Refresh Batch Test Evaluation'** to evaluate the holdout test set. Results will be automatically saved and displayed here across all sessions.")

# TAB 3: Model Performance & Analytics
with tabs[2]:
    st.markdown("### Model Performance & Evaluation Metrics")
    
    col_sel, col_act = st.columns([2, 1])
    with col_sel:
        split_choice = st.radio("Select Dataset Split to Inspect:", ["Holdout Test Set", "Validation Set"], horizontal=True)
    selected_split = "test" if "Test" in split_choice else "valid"

    with col_act:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"⚡ Re-generate Metrics for {selected_split.upper()}", use_container_width=True):
            with st.spinner(f"Evaluating {selected_split.upper()} split..."):
                from src.evaluate import run_evaluation
                run_evaluation(
                    model_path=os.path.join(models_dir, "best_model.keras"),
                    dataset_root=dataset_dir,
                    split=selected_split,
                    output_dir=metrics_dir
                )
            st.success(f"Metrics and confusion matrix for {selected_split.upper()} generated successfully!")
            st.rerun()

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"#### Confusion Matrix ({selected_split.capitalize()} Set)")
        cm_path = os.path.join(metrics_dir, f"{selected_split}_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix ({selected_split.capitalize()} Set)", use_column_width=True)
        else:
            st.info(f"No confusion matrix plot found at `{os.path.relpath(cm_path, PROJECT_ROOT)}`. Click the button above to generate it.")
            
    with col_b:
        st.markdown("#### Training & Validation Learning Curves")
        curves_path = os.path.join(models_dir, "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="TensorFlow Loss & Accuracy Curves", use_column_width=True)
        else:
            st.info(f"Training curves not found at `{os.path.relpath(curves_path, PROJECT_ROOT)}`.")

    st.markdown("---")
    st.markdown(f"#### Detailed Classification Report ({selected_split.capitalize()} Set)")
    eval_json_path = os.path.join(metrics_dir, f"{selected_split}_evaluation_report.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
        
        if "metrics" in eval_data:
            m_df = pd.DataFrame(eval_data["metrics"]).transpose()
            st.dataframe(m_df.style.format(precision=4), use_container_width=True)
            
        if "roc_auc" in eval_data:
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC Score", f"{eval_data['roc_auc']:.4f}")
            c2.metric("Accuracy", f"{eval_data.get('accuracy', 0.0):.2%}" if 'accuracy' in eval_data else f"{eval_data.get('sample_count', 0)} samples")
        
        err_data = eval_data.get("error_analysis", {})
        st.markdown(f"**Error Analysis Breakdown:** Total Errors: `{err_data.get('total_errors', 0)}` (False Positives: `{err_data.get('false_positives', 0)}`, False Negatives: `{err_data.get('false_negatives', 0)}`)")
        
        if err_data.get("error_samples"):
            with st.expander(f"Inspect Misclassified Samples ({len(err_data['error_samples'])} items)"):
                for err in err_data["error_samples"]:
                    st.write(f"- **{err['filename']}**: True: `{err['true_class']}` ➔ Pred: `{err['predicted_class']}` (Conf: {err['confidence']:.1%}) [{err.get('error_type', 'Error')}]")
        else:
            st.success("🎉 Zero classification errors on this split!")
    else:
        st.info(f"Evaluation report will be displayed once generated by clicking the button above.")
