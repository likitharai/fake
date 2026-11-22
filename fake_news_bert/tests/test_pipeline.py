"""Unit tests for preprocessing, data handling and metrics (no model download needed)."""
import numpy as np
import pandas as pd

from src.data import clean_dataframe, stratified_split
from src.evaluate import compute_metrics
from src.preprocessing import build_input_text, clean_text


def test_clean_text_removes_html_and_urls():
    assert clean_text("<b>Breaking</b> news at https://x.com now") == "Breaking news at now"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_build_input_text_combines_title_and_body():
    assert build_input_text("Title", "Body text") == "Title. Body text"
    assert build_input_text(None, "Only body") == "Only body"


def test_clean_dataframe_drops_empty_and_duplicate_rows():
    raw = pd.DataFrame({
        "title": ["A", "A", None, "B"],
        "text": ["x", "x", "", "y"],
        "label": [0, 0, 1, 1],
    })
    clean, removed = clean_dataframe(raw)
    assert len(clean) == 2
    assert removed == 1
    assert set(clean.columns) == {"input_text", "label"}


def test_clean_dataframe_removes_conflicting_labels():
    raw = pd.DataFrame({"title": ["A", "A", "B"], "text": ["x", "x", "y"], "label": [0, 1, 1]})
    clean, _ = clean_dataframe(raw)
    assert "A. x" not in clean["input_text"].values
    assert len(clean) == 1


def test_stratified_split_is_disjoint_and_complete():
    n = 200
    df = pd.DataFrame({
        "input_text": [f"article {i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
    })
    train, val, test = stratified_split(df, seed=42, test_size=0.15, val_size=0.15)
    assert len(train) + len(val) + len(test) == n
    assert set(train.input_text).isdisjoint(test.input_text)
    assert set(val.input_text).isdisjoint(test.input_text)
    assert set(train.input_text).isdisjoint(val.input_text)
    assert abs(test["label"].mean() - 0.5) < 0.05


def test_compute_metrics_matches_hand_calculation():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert np.isclose(m["accuracy"], 0.75)
    # FAKE is positive: TP=1, FP=0, FN=1
    assert np.isclose(m["precision"], 1.0)
    assert np.isclose(m["recall"], 0.5)
    assert np.isclose(m["f1"], 2 / 3)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
