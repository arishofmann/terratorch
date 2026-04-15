# Copyright contributors to the Terratorch project

"""Dataset for pre-extracted embedding tensors (.pt)."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Loads embeddings and labels from .pt files in a directory.

    Expects:
        - embeddings.pt: shape (N, embed_dim)
        - labels.pt: shape (N, num_classes) multi-hot
        - patch_ids.pt: list of N patch ID strings
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.embeddings = torch.load(self.data_root / "embeddings.pt", weights_only=True)
        self.labels = torch.load(self.data_root / "labels.pt", weights_only=True)
        self.patch_ids = torch.load(self.data_root / "patch_ids.pt", weights_only=False)

        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                f"embeddings ({len(self.embeddings)}) and labels ({len(self.labels)}) size mismatch"
            )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.embeddings[index].float(),
            "label": self.labels[index].float(),
            "filename": self.patch_ids[index],
        }
