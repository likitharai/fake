"""Custom CSS, color palette and reusable UI helpers."""
import streamlit as st

COLORS = {
    "FAKE": "#EF4444",
    "REAL": "#10B981",
    "primary": "#6366F1",
    "accent": "#06B6D4",
    "warning": "#F59E0B",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%); }
section[data-testid="stSidebar"] * { color: #E0E7FF; }
section[data-testid="stSidebar"] .brand { color: #FFFFFF !important; font-size: 1.35rem; font-weight: 800; }
.hero { background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%); color: #FFFFFF; padding: 2rem 2.2rem; border-radius: 18px; box-shadow: 0 10px 30px rgba(99,102,241,0.25); margin-bottom: 1.5rem; }
.hero h1 { color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 800; }
.hero p { margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }
.metric-card { background: #FFFFFF; border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: 0 4px 14px rgba(15,23,42,0.08); border-left: 5px solid var(--accent, #6366F1); min-height: 110px; }
.metric-label { font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #0F172A; margin-top: 0.3rem; }
.metric-sub { font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem; }
.verdict { background: #FFFFFF; border-radius: 16px; padding: 1.4rem 1.6rem; border: 2px solid; box-shadow: 0 6px 20px rgba(15,23,42,0.08); margin: 1rem 0; }
.verdict-label { font-size: 2rem; font-weight: 800; }
.verdict-sub { color: #475569; font-size: 1rem; margin-top: 0.2rem; }
.info-card { background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.4rem; box-shadow: 0 4px 14px rgba(15,23,42,0.06); margin-bottom: 1rem; }
.info-card h4 { margin-top: 0; color: #312E81; }
.stButton > button { border-radius: 10px; font-weight: 600; }
@media (max-width: 768px) {
  .hero h1 { font-size: 1.5rem; }
  .metric-value { font-size: 1.4rem; }
  .verdict-label { font-size: 1.5rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = COLORS["primary"], icon: str = "", sub: str = "") -> None:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card" style="--accent:{color}">'
        f'<div class="metric-label">{icon} {label}</div>'
        f'<div class="metric-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def verdict_card(label: str, confidence: float) -> None:
    color = COLORS[label]
    icon = "🚨" if label == "FAKE" else "✅"
    st.markdown(
        f'<div class="verdict" style="border-color:{color}">'
        f'<div class="verdict-label" style="color:{color}">{icon} {label}</div>'
        f'<div class="verdict-sub">Model confidence: <b>{confidence * 100:.2f}%</b></div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1E293B"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig
