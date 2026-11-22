"""Inference API used by the Streamlit app and batch processing."""
from pathlib import Path

import numpy as np
import torch

from config import BATCH_SIZE, LABEL_MAP, MAX_LENGTH, MODEL_DIR
from src.model import get_device, load_trained
from src.preprocessing import clean_text


class FakeNewsPredictor:
    def __init__(self, model_dir=MODEL_DIR, device=None,
                 max_length: int = MAX_LENGTH, batch_size: int = BATCH_SIZE):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No trained model found at {model_dir}. Run `python -m src.train` first."
            )
        self.device = device or get_device()
        self.tokenizer, self.model = load_trained(model_dir, self.device)
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        """Return an (n, 2) array of [P(FAKE), P(REAL)] for already-cleaned texts."""
        if len(texts) == 0:
            return np.empty((0, 2))
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.vstack(chunks)

    def predict(self, texts) -> list:
        """Classify raw strings. Empty inputs return label=None instead of guessing."""
        cleaned = [clean_text(t) for t in texts]
        valid = [i for i, t in enumerate(cleaned) if t]
        results = [
            {"label": None, "confidence": None, "prob_fake": None, "prob_real": None}
            for _ in texts
        ]
        if valid:
            probs = self.predict_proba([cleaned[i] for i in valid])
            for row, i in zip(probs, valid):
                pred = int(np.argmax(row))
                results[i] = {
                    "label": LABEL_MAP[pred],
                    "confidence": float(row[pred]),
                    "prob_fake": float(row[0]),
                    "prob_real": float(row[1]),
                }
        return results
