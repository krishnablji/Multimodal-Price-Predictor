"""Streamlit Web Application for Multimodal Product Price Prediction."""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.price_predictor.pipeline import MultimodalPricePredictorPipeline
from src.price_predictor.data import build_submission

# Page configuration
st.set_page_config(
    page_title="Multimodal Price Predictor",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(37, 99, 235, 0.05);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return MultimodalPricePredictorPipeline()


pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Architecture-SigLIP2%20%2B%20MiniLM%20%2B%20Ensemble-blue?style=for-the-badge", use_container_width=True)
    st.markdown("### ⚙️ Pipeline Configuration")
    st.markdown("""
    - **Vision Encoder:** Google SigLIP 2 (`1,024-dim`)
    - **Text Encoder:** SigLIP 2 + MiniLM (`1,408-dim`)
    - **Catalogue Rules:** 17 Engineered Attributes
    - **Combined Width:** **2,450 Features**
    - **Models:** LightGBM, XGBoost, CatBoost, PyTorch MLP
    - **Optimization:** Constrained SMAPE Loss
    """)
    st.divider()
    st.markdown("👨‍💻 **Author:** Krishna Balaji")
    st.markdown("📦 **System:** Multimodal Pricing Engine")

# App Header
st.markdown('<div class="main-header">🏷️ Multimodal Product Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict optimal e-commerce product prices using state-of-the-art vision & text deep learning models.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Single Product Estimation", "📁 Batch CSV Predictor", "🔬 Architecture & Insights"])

# ----------------- TAB 1: SINGLE PRODUCT -----------------
with tab1:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("📝 Product Listing Details")
        
        # Sample selector
        sample_choice = st.selectbox(
            "Load Sample Example:",
            ["Custom Input", "Sample 1: Dark Roast Coffee Beans", "Sample 2: Insulated Water Bottle", "Sample 3: Wireless Headphones"]
        )

        default_title = "Item Name: Premium Arabica Coffee Beans\nBrand: Roasters Choice\nDescription: Whole bean dark roast coffee, 100% Arabica. Pack of 2.\nPack of 2, 500 g each. Total 1.0 kg."
        default_img = "https://images.unsplash.com/photo-1559056199-641a0ac8b55e"

        if sample_choice == "Sample 2: Insulated Water Bottle":
            default_title = "Item Name: Stainless Steel Insulated Water Bottle\nBrand: HydroPro\nDescription: Double-walled vacuum insulated flask, keeps cold for 24h.\nVolume: 750 ml."
            default_img = "https://images.unsplash.com/photo-1602143407151-7111542de6e8"
        elif sample_choice == "Sample 3: Wireless Headphones":
            default_title = "Item Name: Wireless Noise Cancelling Headphones\nBrand: AudioMax\nDescription: Bluetooth over-ear headphones with 30-hour battery life and quick charge.\nPack of 1."
            default_img = "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"

        catalog_text = st.text_area(
            "Catalogue Content (Title, Bullets, Specs, Description):",
            value=default_title if sample_choice != "Custom Input" else "",
            height=160,
            placeholder="Item Name: ...\nBrand: ...\nDescription: ..."
        )

        st.markdown("**Product Image**")
        img_source = st.radio("Image Source:", ["Upload Local Image", "Provide Image URL"], horizontal=True)

        uploaded_file = None
        img_url = ""
        pil_image = None

        if img_source == "Upload Local Image":
            uploaded_file = st.file_uploader("Upload product photo (PNG/JPG):", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                pil_image = Image.open(uploaded_file)
        else:
            img_url = st.text_input("Image URL:", value=default_img if sample_choice != "Custom Input" else "")
            if img_url:
                pil_image = img_url

        predict_btn = st.button("🚀 Predict Product Price", type="primary", use_container_width=True)

    with col_output:
        st.subheader("📊 Price Prediction & Analysis")

        if predict_btn or (sample_choice != "Custom Input" and catalog_text):
            if not catalog_text.strip():
                st.warning("Please provide product catalogue content.")
            else:
                with st.spinner("Analyzing text semantics, image embeddings, and unit rules..."):
                    result = pipeline.predict_single(catalog_text, pil_image)

                # Metric Box
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-label">Estimated Optimal Price</div>
                    <div class="price-value">${result['predicted_price']:.2f}</div>
                    <div style="color: #64748B; font-size: 0.95rem; margin-top: 4px;">
                        Expected Range: <strong>${result['price_range_low']:.2f}</strong> — <strong>${result['price_range_high']:.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                # Attributes grid
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric("Pack Quantity", f"{result['pack_quantity']} unit(s)")
                with col_a2:
                    st.metric("Image-Text Alignment", f"{result['image_text_alignment']}%")

                st.info(f"📏 **Parsed Measurement / Unit:** `{result['parsed_measurement']}`")

                # Model Breakdown chart
                st.markdown("##### 🤖 Individual Model Estimates")
                breakdown_df = pd.DataFrame({
                    "Model": list(result["model_breakdown"].keys()),
                    "Predicted Price ($)": list(result["model_breakdown"].values())
                })
                st.bar_chart(breakdown_df.set_index("Model"), color="#2563EB")
        else:
            st.info("👈 Enter product details or select a sample on the left and click **Predict** to view real-time estimates.")


# ----------------- TAB 2: BATCH CSV -----------------
with tab2:
    st.subheader("📁 Batch Product Dataset Prediction")
    st.markdown("Upload a product catalog CSV (`sample_id`, `catalog_content`, `image_link`) to generate predictions in bulk.")

    batch_file = st.file_uploader("Upload test CSV:", type=["csv"])
    if batch_file:
        test_df = pd.read_csv(batch_file)
        st.write(f"Loaded **{len(test_df)}** records. Preview:")
        st.dataframe(test_df.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Prediction", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            predictions = []
            total = len(test_df)
            
            for idx, row in test_df.iterrows():
                res = pipeline.predict_single(str(row["catalog_content"]), row.get("image_link"))
                predictions.append(res["predicted_price"])
                
                if (idx + 1) % max(1, total // 20) == 0 or idx == total - 1:
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"Processing product {idx + 1} of {total}...")
            
            sub_df = build_submission(test_df["sample_id"], predictions)
            st.success("✅ Batch prediction complete!")
            st.dataframe(sub_df.head(10), use_container_width=True)

            csv_buffer = io.StringIO()
            sub_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download predictions.csv",
                data=csv_buffer.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary"
            )


# ----------------- TAB 3: ARCHITECTURE -----------------
with tab3:
    st.subheader("🔬 Multimodal System Architecture")
    st.markdown("""
    This system implements a state-of-the-art multimodal deep learning and ensemble architecture:

    #### 1. Feature Representation (2,450 Dimensions)
    - **SigLIP 2 Vision Projection (1,024 dims):** Encodes visual shape, packaging, and high-level product appearance.
    - **SigLIP 2 Text Projection (1,024 dims):** Encodes title and specifications into the shared multimodal space.
    - **Cross-Modal Cosine Similarity (1 dim):** Measures direct alignment between the image and description.
    - **MiniLM Embeddings (384 dims):** Compact sentence-level semantic representations.
    - **Domain Catalogue Rules (17 dims):** Quantities, unit normalization (weight/volume/count), text statistics, and brand frequency encoding.

    #### 2. Ensemble & Optimization
    - **Cross-Validation:** 5-Fold split on `log1p(price)`.
    - **Models:** Leaf-wise Gradient Boosting (LightGBM), Histogram-based Trees (XGBoost), Symmetric Trees (CatBoost), and Dense PyTorch MLP.
    - **Metric:** Blended using constrained **SMAPE** (Symmetric Mean Absolute Percentage Error) minimization.
    """)
