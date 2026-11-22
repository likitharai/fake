"""Cached loaders for the trained model and saved evaluation artifacts."""
import json

import streamlit as st

from config import METRICS_DIR


def _read_json(name: str):
    path = METRICS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_metrics():
    return _read_json("test_metrics.json")


@st.cache_data
def load_history():
    return _read_json("training_history.json")


@st.cache_data
def load_dataset_stats():
    return _read_json("dataset_stats.json")


@st.cache_resource(show_spinner="Loading BERT model...")
def get_predictor():
    from src.predict import FakeNewsPredictor

    return FakeNewsPredictor()
