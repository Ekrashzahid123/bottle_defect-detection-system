import os
import sys
import glob
import json
import time
from PIL import Image
import streamlit as st
import pandas as pd

# Path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import BottleDefectPredictor

# Page Setup
st.set_page_config(
    page_title="Bottle Defect Inspection System",
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
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-normal {
        background-color: #DCFCE7;
        color: #15803D;
        border: 2px solid #86EFAC;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .status-defective {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 2px solid #FCA5A5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    if os.path.exists(model_path):
        return BottleDefectPredictor(model_path=model_path)
    return None

predictor = get_model()

# Sidebar
st.sidebar.title("🍾 System Status")
if predictor is not None:
    st.sidebar.success("✅ Model: MobileNetV3 (Loaded)")
    st.sidebar.info(f"⚙️ Device: `{predictor.device}`")
    st.sidebar.caption("Backbone: Pretrained MobileNetV3-Small Fine-Tuned for Industrial Bottle Defect Inspection")
else:
    st.sidebar.error("⚠️ Model checkpoint not found.")
    st.sidebar.warning("Run `python src/train.py` to train the model first.")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Inspection Guidelines")
st.sidebar.markdown("""
- **Normal**: Good cap, properly sealed bottle.
- **Defective**: Open cap, misaligned, or missing cap.
- **Target Response Time**: < 30ms / product image.
""")

# Main Title
st.markdown('<div class="main-title">Industrial Bottle Visual Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Automated Quality Assurance System powered by MobileNetV3 & PyTorch</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Live Single Inspection", "📦 Batch Test Inspection", "📊 Evaluation & Model Analytics"])

# TAB 1: Live Single Inspection
with tabs[0]:
    st.markdown("### Inspect Bottle Product")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        input_source = st.radio("Select Image Source:", ["Upload Custom Image", "Select from Test Set Sample"], horizontal=True)
        selected_img = None
        
        if input_source == "Upload Custom Image":
            uploaded_file = st.file_uploader("Upload a bottle image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_img = Image.open(uploaded_file).convert("RGB")
        else:
            sample_images = glob.glob("dataset/test/images/*.*") + glob.glob("dataset/valid/images/*.*")
            if sample_images:
                sample_names = [os.path.basename(p) for p in sample_images[:15]]
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
            if predictor is None:
                st.error("Please train the model first to perform inference.")
            else:
                with st.spinner("Analyzing image features..."):
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

                # JSON Output Expander
                with st.expander("View Raw Inference Payload (FastAPI Standard)"):
                    st.json(result)
        else:
            st.info("👈 Upload an image or select a sample from the left panel to begin inspection.")

# TAB 2: Batch Inspection
with tabs[1]:
    st.markdown("### Batch Test Set Evaluation")
    test_images = glob.glob("dataset/test/images/*.*")
    
    if not test_images:
        st.warning("No images found in `dataset/test/images/`.")
    else:
        st.write(f"Found **{len(test_images)}** unannotated test images in manufacturing batch.")
        
        if st.button("🚀 Run Batch Inspection on All Test Samples"):
            if predictor is None:
                st.error("Model not loaded. Please train model first.")
            else:
                progress_bar = st.progress(0)
                results_list = []
                
                for idx, img_path in enumerate(test_images):
                    res = predictor.predict(img_path)
                    res["filename"] = os.path.basename(img_path)
                    res["image_path"] = img_path
                    results_list.append(res)
                    progress_bar.progress((idx + 1) / len(test_images))
                
                df_results = pd.DataFrame(results_list)
                
                # Batch Summary Metrics
                total = len(df_results)
                defective_count = int(df_results["is_defective"].sum())
                normal_count = total - defective_count
                defect_rate = (defective_count / total) * 100
                avg_latency = df_results["inference_time_ms"].mean()

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Inspected", total)
                b2.metric("Normal (Passed)", normal_count)
                b3.metric("Defective (Rejected)", defective_count)
                b4.metric("Defect Rate", f"{defect_rate:.1f}%")
                
                st.markdown(f"**Average Latency per Product:** `{avg_latency:.2f} ms`")
                st.markdown("---")
                
                # Image Gallery Grid
                st.markdown("#### Inspection Gallery")
                cols = st.columns(4)
                for i, r in enumerate(results_list):
                    col = cols[i % 4]
                    with col:
                        thumb = Image.open(r["image_path"]).resize((200, 200))
                        badge = "🚨 DEFECTIVE" if r["is_defective"] else "✅ NORMAL"
                        st.image(thumb, caption=f"{r['filename']}\n{badge} ({r['confidence']:.1%})", use_column_width=True)

# TAB 3: Evaluation & Model Analytics
with tabs[2]:
    st.markdown("### Model Performance & Evaluation Metrics")
    
    split_choice = st.radio("Select Evaluation Dataset to Inspect:", ["Holdout Test Set (30 Images)", "Validation Set (40 Images)"], horizontal=True)
    selected_split = "test" if "Test" in split_choice else "valid"

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"#### Confusion Matrix ({selected_split.capitalize()} Set)")
        cm_path = f"metrics/{selected_split}_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix ({selected_split.capitalize()} Set)", use_column_width=True)
        else:
            st.info(f"Run `python src/evaluate.py --split {selected_split}` to generate the Confusion Matrix.")
            
    with col_b:
        st.markdown("#### Training & Validation Learning Curves")
        curves_path = "models/training_curves.png"
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training & Validation Loss / F1 Curves", use_column_width=True)
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
