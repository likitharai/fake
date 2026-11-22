"""PyTorch dataset with lazy, dynamic-padded tokenization."""
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


def make_collate_fn(tokenizer, max_length: int):
    """Tokenize per batch (truncation + padding) to avoid storing huge encodings."""

    def collate(batch):
        enc = tokenizer(
            [item["text"] for item in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return enc

    return collate
