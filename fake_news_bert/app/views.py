"""Page renderers for the Streamlit app. All outputs come from saved artifacts or live model inference."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from config import BATCH_SIZE, BERT_NAME, EPOCHS, LEARNING_RATE, MAX_BATCH_ROWS, MAX_LENGTH
from app.services import get_predictor, load_dataset_stats, load_history, load_metrics
from app.styles import COLORS, metric_card, style_fig, verdict_card
from src.preprocessing import clean_text

TEXT_COLUMN_HINTS = ["text", "content", "article", "body", "news", "headline", "title"]
NO_MODEL_MSG = "Trained model not found. Run `python -m src.train` first, then reload this page."


def _fmt_pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _fmt_int(value) -> str:
    return "—" if value is None else f"{value:,}"


# ---------------------------------------------------------------- Dashboard
def render_dashboard():
    st.markdown(
        '<div class="hero"><h1>📰 Fake News Detection with BERT</h1>'
        '<p>A fine-tuned bert-base-uncased classifier that labels news headlines and articles as FAKE or REAL, '
        'with batch processing and full model analytics.</p></div>',
        unsafe_allow_html=True,
    )

    stats = load_dataset_stats() or {}
    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Raw articles", _fmt_int(stats.get("raw_rows")), COLORS["primary"], "🗂️", "WELFake dataset")
    with c2:
        metric_card("Clean articles", _fmt_int(stats.get("clean_rows")), COLORS["accent"], "🧹",
                    f"{stats.get('duplicates_removed', 0):,} duplicates removed" if stats else "")
    with c3:
        metric_card("Test accuracy", _fmt_pct(metrics["accuracy"]) if metrics else "—", COLORS["REAL"], "🎯",
                    "Unseen test split" if metrics else "Train the model to populate")
    with c4:
        metric_card("Test F1 (FAKE)", _fmt_pct(metrics["f1"]) if metrics else "—", COLORS["warning"], "⚖️",
                    "Positive class = FAKE" if metrics else "")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Model information")
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        st.markdown(
            f"""
- **Architecture:** BERT encoder with a sequence-classification head (2 labels)
- **Pretrained checkpoint:** `{BERT_NAME}`
- **Max sequence length:** {MAX_LENGTH} tokens
- **Training:** up to {EPOCHS} epochs, learning rate {LEARNING_RATE}, batch size {BATCH_SIZE}
- **Inference device:** {device}
"""
        )
    with right:
        st.markdown("#### Label distribution")
        counts = stats.get("label_counts_clean") or stats.get("label_counts_raw")
        if counts:
            fig = go.Figure(go.Pie(
                labels=list(counts.keys()), values=list(counts.values()), hole=0.55,
                marker=dict(colors=[COLORS["FAKE"], COLORS["REAL"]]), textinfo="label+percent",
            ))
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            st.info("Dataset statistics not found. Run training to generate them.")

    if stats.get("train_size"):
        st.markdown("#### Data splits")
        split_fig = go.Figure(go.Bar(
            x=[stats["train_size"], stats["val_size"], stats["test_size"]],
            y=["Train", "Validation", "Test"], orientation="h",
            marker_color=[COLORS["primary"], COLORS["accent"], COLORS["REAL"]],
            text=[f"{v:,}" for v in [stats["train_size"], stats["val_size"], stats["test_size"]]],
            textposition="outside",
        ))
        st.plotly_chart(style_fig(split_fig, 240), use_container_width=True)


# ---------------------------------------------------------------- Detection
def _clear_input():
    st.session_state["news_input"] = ""


def render_detect():
    st.markdown("## 🔍 Fake News Detection")
    st.caption("Paste a headline or a full article. The fine-tuned BERT model returns a verdict and probabilities.")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.text_area(
        "News headline or article",
        key="news_input",
        height=220,
        placeholder="e.g. Scientists confirm that drinking coffee cures all known diseases...",
    )
    b1, b2, _ = st.columns([1, 1, 4])
    predict_clicked = b1.button("Predict", type="primary", use_container_width=True)
    b2.button("Clear", on_click=_clear_input, use_container_width=True)

    if predict_clicked:
        text = st.session_state.get("news_input", "")
        if not clean_text(text):
            st.warning("Please enter a headline or article before predicting.")
            return
        with st.spinner("Running BERT inference..."):
            try:
                result = predictor.predict([text])[0]
            except Exception as exc:  # surface runtime failures without crashing the app
                st.error(f"Prediction failed: {exc}")
                return
        _render_result(result)


def _render_result(result: dict):
    verdict_card(result["label"], result["confidence"])
    fig = go.Figure(go.Bar(
        x=["FAKE", "REAL"],
        y=[result["prob_fake"], result["prob_real"]],
        marker_color=[COLORS["FAKE"], COLORS["REAL"]],
        text=[f"{result['prob_fake']:.1%}", f"{result['prob_real']:.1%}"],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 1.12], tickformat=".0%", title="Probability"),
                      title="Probability distribution")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)


# ---------------------------------------------------------------- Batch
def _guess_text_columns(df: pd.DataFrame) -> list:
    text_cols = [c for c in df.columns if df[c].dtype == object]
    preferred = [c for c in text_cols if str(c).lower() in TEXT_COLUMN_HINTS]
    return preferred or text_cols[:1]


def render_batch():
    st.markdown("## 📂 Batch News Detection")
    st.caption(f"Upload a CSV with a title, text or article column. Up to {MAX_BATCH_ROWS:,} rows are processed.")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        st.info("Waiting for a CSV file. Each row will be classified as FAKE or REAL.")
        return

    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if df.empty:
        st.error("The uploaded CSV is empty.")
        return

    text_options = [c for c in df.columns if df[c].dtype == object]
    if not text_options:
        st.error("No text columns found in the CSV.")
        return

    selected = st.multiselect("Text columns to analyse (combined per row)",
                              options=text_options, default=_guess_text_columns(df))
    if not selected:
        st.warning("Select at least one text column.")
        return

    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"File has {len(df):,} rows; only the first {MAX_BATCH_ROWS:,} will be processed.")
    work = df.head(MAX_BATCH_ROWS).copy()
    texts = work[selected].apply(
        lambda row: " . ".join(t for t in (clean_text(v) for v in row) if t), axis=1
    ).tolist()

    if st.button("Run batch prediction", type="primary"):
        preds = []
        progress = st.progress(0.0, text="Running BERT inference...")
        chunk = 64
        for start in range(0, len(texts), chunk):
            preds.extend(predictor.predict(texts[start:start + chunk]))
            done = min(start + chunk, len(texts))
            progress.progress(done / len(texts), text=f"Processed {done:,} / {len(texts):,}")
        progress.empty()

        results = pd.DataFrame(preds)
        results.insert(0, "row", range(1, len(results) + 1))
        results.insert(1, "text_preview", [t[:120] + ("…" if len(t) > 120 else "") for t in texts])
        st.session_state["batch_results"] = results

    results = st.session_state.get("batch_results")
    if results is None:
        return
    _render_batch_results(results)


def _render_batch_results(results: pd.DataFrame):
    valid = results[results["label"].notna()]
    fake_n = int((valid["label"] == "FAKE").sum())
    real_n = int((valid["label"] == "REAL").sum())
    invalid_n = len(results) - len(valid)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Processed", f"{len(results):,}", COLORS["primary"], "📄")
    with c2:
        metric_card("Predicted FAKE", f"{fake_n:,}", COLORS["FAKE"], "🚨")
    with c3:
        metric_card("Predicted REAL", f"{real_n:,}", COLORS["REAL"], "✅")
    with c4:
        avg_conf = valid["confidence"].mean() if len(valid) else None
        metric_card("Avg confidence", _fmt_pct(avg_conf), COLORS["accent"], "📈",
                    f"{invalid_n} empty rows skipped" if invalid_n else "")

    left, right = st.columns([1, 1.3])
    with left:
        if len(valid):
            pie = go.Figure(go.Pie(labels=["FAKE", "REAL"], values=[fake_n, real_n], hole=0.5,
                                   marker=dict(colors=[COLORS["FAKE"], COLORS["REAL"]])))
            st.plotly_chart(style_fig(pie, 300, ), use_container_width=True)
    with right:
        if len(valid):
            hist = go.Figure(go.Histogram(x=valid["confidence"], nbinsx=20, marker_color=COLORS["primary"]))
            hist.update_layout(xaxis_title="Confidence", yaxis_title="Articles", title="Confidence distribution")
            st.plotly_chart(style_fig(hist, 300), use_container_width=True)

    st.markdown("#### Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "prob_fake": st.column_config.NumberColumn("P(FAKE)", format="%.3f"),
            "prob_real": st.column_config.NumberColumn("P(REAL)", format="%.3f"),
        },
    )
    st.download_button(
        "Download predictions CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fake_news_predictions.csv",
        mime="text/csv",
        type="primary",
    )


# ---------------------------------------------------------------- Analytics
def render_analytics():
    st.markdown("## 📊 Model Analytics")
    metrics = load_metrics()
    history = load_history()
    if metrics is None:
        st.info("No evaluation results found. Train the model with `python -m src.train` to generate them.")
        return

    st.caption(f"Metrics computed on {metrics['n_samples']:,} held-out test articles. Positive class: FAKE.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", _fmt_pct(metrics["accuracy"]), COLORS["primary"], "🎯")
    with c2:
        metric_card("Precision", _fmt_pct(metrics["precision"]), COLORS["accent"], "🔎")
    with c3:
        metric_card("Recall", _fmt_pct(metrics["recall"]), COLORS["REAL"], "📡")
    with c4:
        metric_card("F1-score", _fmt_pct(metrics["f1"]), COLORS["warning"], "⚖️")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Confusion matrix")
        cm = metrics["confusion_matrix"]
        heat = go.Figure(go.Heatmap(
            z=cm, x=["Predicted FAKE", "Predicted REAL"], y=["Actual FAKE", "Actual REAL"],
            colorscale=[[0, "#EEF2FF"], [1, "#4338CA"]], showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm], texttemplate="%{text}",
            textfont=dict(size=18),
            hovertemplate="%{y} → %{x}<br>Count: %{z}<extra></extra>",
        ))
        heat.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(heat, 340), use_container_width=True)
    with right:
        st.markdown("#### Classification report")
        report = pd.DataFrame(metrics["classification_report"]).T.round(3)
        st.dataframe(report, use_container_width=True)

    st.markdown("#### Training and validation curves")
    if history:
        epochs = history["epoch"]
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Train loss",
                                      line=dict(color=COLORS["primary"], width=3)))
        loss_fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Validation loss",
                                      line=dict(color=COLORS["FAKE"], width=3)))
        loss_fig.update_layout(title="Loss per epoch", xaxis_title="Epoch", yaxis_title="Loss",
                               xaxis=dict(dtick=1))
        acc_fig = go.Figure(go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines+markers",
                                       name="Validation accuracy", line=dict(color=COLORS["REAL"], width=3)))
        acc_fig.update_layout(title="Validation accuracy per epoch", xaxis_title="Epoch",
                              yaxis=dict(tickformat=".0%"), xaxis=dict(dtick=1))
        c1, c2 = st.columns(2)
        c1.plotly_chart(style_fig(loss_fig, 330), use_container_width=True)
        c2.plotly_chart(style_fig(acc_fig, 330), use_container_width=True)
    else:
        st.info("Training history not found.")


# ---------------------------------------------------------------- About
def render_about():
    st.markdown("## ℹ️ About")
    st.markdown(
        f"""
<div class="info-card">
<h4>BERT model</h4>
BERT (Bidirectional Encoder Representations from Transformers) reads text in both directions at once,
so each token representation depends on its full context. This project fine-tunes <code>{BERT_NAME}</code>
with a two-class classification head. The uncased variant lowercases input text.
</div>
<div class="info-card">
<h4>Application workflow</h4>
1. Load the WELFake dataset and run exploratory analysis.<br>
2. Clean text (HTML, URLs, Unicode, whitespace), drop empties and duplicates before splitting.<br>
3. Encode labels (0 = FAKE, 1 = REAL) and make a stratified 70/15/15 train/validation/test split.<br>
4. Tokenize with truncation and padding; fine-tune with AdamW, linear warm-up and early stopping on validation loss.<br>
5. Save the best checkpoint, then evaluate once on the untouched test split.<br>
6. Serve predictions through this Streamlit app for single text, batch CSV and analytics.
</div>
<div class="info-card">
<h4>Technology stack</h4>
Python · PyTorch · Hugging Face Transformers · Pandas · NumPy · Scikit-learn · Streamlit · Plotly
</div>
<div class="info-card">
<h4>Limitations</h4>
- Predictions reflect patterns in the training data and can be wrong, especially for new events, satire, opinion pieces or sources unlike WELFake.<br>
- The model judges text style and content, not whether a claim is true. It cannot fact-check against live sources.<br>
- Inputs are truncated to the maximum sequence length, so the decision may ignore the end of long articles.<br>
- Confidence is the model's softmax probability, not a calibrated guarantee of correctness.<br>
- Use this tool as an assistant for triage, not as the final word on whether a story is true.
</div>
""",
        unsafe_allow_html=True,
    )
