import os
import sys
import glob
import json
import time
from PIL import Image
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import BottleDefectPredictor

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
    model_path = os.getenv("MODEL_PATH", "models/best_model.keras")
    return BottleDefectPredictor(model_path=model_path)

predictor = get_model()

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
            sample_images = glob.glob("dataset/test/images/*.*") + glob.glob("dataset/valid/images/*.*")
            if sample_images:
                sample_names = [os.path.basename(p) for p in sample_images[:20]]
                chosen_idx = st.selectbox("Choose Sample Image:", range(len(sample_names)), format_func=lambda x: sample_names[x])
                selected_path = sample_images[chosen_idx]
                selected_img = Image.open(selected_path).convert("RGB")
                st.caption(f"File: `{os.path.basename(selected_path)}`")
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
    
    manifest_path = "models/split_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        test_images = manifest["split_paths"]["test"]
        test_labels = manifest["split_labels"]["test"]
        st.write(f"Found **{len(test_images)}** holdout test samples with **zero data leakage**.")
        
        if st.button("🚀 Run Batch Inspection on Holdout Test Set"):
            progress_bar = st.progress(0)
            results_list = []
            
            for idx, (img_path, true_lbl) in enumerate(zip(test_images, test_labels)):
                res = predictor.predict(img_path)
                res["filename"] = os.path.basename(img_path)
                res["image_path"] = img_path
                res["true_class"] = "Defective" if true_lbl == 1 else "Normal"
                res["is_correct"] = bool(res["class_id"] == true_lbl)
                results_list.append(res)
                progress_bar.progress((idx + 1) / len(test_images))
            
            df_results = pd.DataFrame(results_list)
            
            total = len(df_results)
            correct_count = int(df_results["is_correct"].sum())
            acc = (correct_count / total) * 100
            defective_count = int(df_results["is_defective"].sum())
            avg_latency = df_results["inference_time_ms"].mean()

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Tested", total)
            b2.metric("Accuracy", f"{acc:.1f}%")
            b3.metric("Defects Found", defective_count)
            b4.metric("Avg Latency", f"{avg_latency:.2f} ms")
            
            st.markdown("---")
            st.markdown("#### Test Sample Predictions Gallery")
            cols = st.columns(4)
            for i, r in enumerate(results_list):
                col = cols[i % 4]
                with col:
                    thumb = Image.open(r["image_path"]).resize((200, 200))
                    status_icon = "✅" if r["is_correct"] else "❌"
                    badge = f"{status_icon} Pred: {r['prediction']} (True: {r['true_class']})"
                    st.image(thumb, caption=f"{r['filename']}\n{badge} ({r['confidence']:.1%})", use_column_width=True)
    else:
        st.info("Run `python src/train.py` to generate the leak-free split manifest.")

# TAB 3: Model Performance & Analytics
with tabs[2]:
    st.markdown("### Model Performance & Evaluation Metrics")
    
    split_choice = st.radio("Select Dataset Split to Inspect:", ["Holdout Test Set", "Validation Set"], horizontal=True)
    selected_split = "test" if "Test" in split_choice else "valid"

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"#### Confusion Matrix ({selected_split.capitalize()} Set)")
        cm_path = f"metrics/{selected_split}_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix ({selected_split.capitalize()} Set)", use_column_width=True)
        else:
            st.info(f"Run `python src/evaluate.py --split {selected_split}` to generate Confusion Matrix.")
            
    with col_b:
        st.markdown("#### Training & Validation Learning Curves")
        curves_path = "models/training_curves.png"
        if os.path.exists(curves_path):
            st.image(curves_path, caption="TensorFlow Loss & Accuracy Curves", use_column_width=True)
        else:
            st.info("Training curves will appear after running `python src/train.py`.")

    st.markdown("---")
    st.markdown(f"#### Detailed Classification Report ({selected_split.capitalize()} Set)")
    eval_json_path = f"metrics/{selected_split}_evaluation_report.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
        
        m_df = pd.DataFrame(eval_data["metrics"]).transpose()
        st.dataframe(m_df.style.format(precision=4))
        
        err_data = eval_data.get("error_analysis", {})
        st.markdown(f"**Error Analysis Breakdown:** Total Errors: `{err_data.get('total_errors', 0)}` (False Positives: `{err_data.get('false_positives', 0)}`, False Negatives: `{err_data.get('false_negatives', 0)}`)")
        
        if err_data.get("error_samples"):
            with st.expander(f"Inspect Misclassified Samples ({len(err_data['error_samples'])} items)"):
                for err in err_data["error_samples"]:
                    st.write(f"- **{err['filename']}**: True: `{err['true_class']}` ➔ Pred: `{err['predicted_class']}` (Conf: {err['confidence']:.1%}) [{err.get('error_type', 'Error')}]")
    else:
        st.info(f"Evaluation report will be displayed once generated by `evaluate.py --split {selected_split}`.")
