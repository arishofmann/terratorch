# Copyright contributors to the Terratorch project

import logging

import numpy as np
import torch
from torch import Tensor

from terratorch.models.decoders.rf_decoder import RandomForestDecoder
from terratorch.tasks.multilabel_classification_tasks import MultiLabelClassificationTask

logger = logging.getLogger("terratorch")


class RandomForestClassificationTask(MultiLabelClassificationTask):
    """MultiLabelClassificationTask backed by a RandomForestDecoder.

    Because RF training is not gradient-based, this task disables automatic
    optimization. Instead it collects labels over one epoch and calls
    decoder.fit() at epoch end. Validation and test use the inherited
    MultiLabelClassificationTask steps with proper multi-label metrics
    (F1, AUROC, AveragePrecision, etc.).

    The RF decoder returns logits (log-odds), so BCEWithLogitsLoss and
    sigmoid-based metrics work correctly.
    """

    automatic_optimization: bool = False

    def __init__(self, *args, **kwargs) -> None:
        kwargs["freeze_backbone"] = True
        kwargs["freeze_decoder"] = True
        kwargs.setdefault("loss", "bce")
        super().__init__(*args, **kwargs)

        self._y_buf: list[np.ndarray] = []
        self._validate_decoder()

    def _validate_decoder(self) -> None:
        decoder = self.model.decoder
        if not isinstance(decoder, RandomForestDecoder):
            raise TypeError(
                f"RandomForestClassificationTask requires a RandomForestDecoder, "
                f"got {type(decoder).__name__}"
            )

    def configure_optimizers(self):  # type: ignore[override]
        return []

    def training_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        with torch.no_grad():
            self(x, **rest)

        self._y_buf.append(y.detach().cpu().numpy())
        return torch.tensor(0.0)

    def on_train_epoch_end(self) -> None:
        if not self._y_buf:
            logger.warning("no labels accumulated, skipping fit")
            super().on_train_epoch_end()
            return

        y_all = np.concatenate(self._y_buf, axis=0).astype(int)
        self._y_buf = []

        self.model.decoder.fit(y_all)
        super().on_train_epoch_end()
