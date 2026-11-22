"""Streamlit entry point. Run with:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.styles import inject_css  # noqa: E402
from app.views import (  # noqa: E402
    render_about, render_analytics, render_batch, render_dashboard, render_detect,
)

st.set_page_config(
    page_title="Fake News Detector (BERT)",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

PAGES = {
    "🏠 Dashboard": render_dashboard,
    "🔍 Fake News Detection": render_detect,
    "📂 Batch Detection": render_batch,
    "📊 Model Analytics": render_analytics,
    "ℹ️ About": render_about,
}

with st.sidebar:
    st.markdown('<div class="brand">📰 FakeNews BERT</div>', unsafe_allow_html=True)
    st.caption("Transformer-based misinformation detection")
    st.divider()
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Fine-tuned bert-base-uncased · PyTorch")

PAGES[choice]()
