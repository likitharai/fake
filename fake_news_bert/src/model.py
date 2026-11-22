"""Model and tokenizer loading helpers."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABEL_MAP


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_training(model_name: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=dict(LABEL_MAP),
        label2id={v: k for k, v in LABEL_MAP.items()},
    )
    return tokenizer, model


def load_trained(model_dir, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device).eval()
    return tokenizer, model
