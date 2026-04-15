"""DataModule for pre-extracted embedding tensors (.pt) with multi-label classification."""

import lightning
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from terratorch.datasets.embedding_dataset import EmbeddingDataset


class EmbeddingClassificationDataModule(lightning.LightningDataModule):
    """DataModule that loads pre-extracted embeddings for classification tasks.

    Expects each data root directory to contain:
        - embeddings.pt: Tensor of shape (N, embed_dim)
        - labels.pt: Tensor of shape (N, num_classes), multi-hot
        - patch_ids.pt: List of N patch ID strings
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        train_data_root: str | None = None,
        val_data_root: str | None = None,
        test_data_root: str | None = None,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_root = train_data_root
        self.val_data_root = val_data_root
        self.test_data_root = test_data_root
        self.drop_last = drop_last

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str) -> None:
        if stage in ["fit"] and self.train_data_root:
            self.train_dataset = EmbeddingDataset(self.train_data_root)
        if stage in ["fit", "validate"] and self.val_data_root:
            self.val_dataset = EmbeddingDataset(self.val_data_root)
        if stage in ["test"] and self.test_data_root:
            self.test_dataset = EmbeddingDataset(self.test_data_root)

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
