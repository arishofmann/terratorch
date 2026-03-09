# Copyright contributors to the Terratorch project

import logging

import numpy as np
import torch
from torch import Tensor, nn

from terratorch.models.decoders.rf_decoder import RandomForestDecoder
from terratorch.models.model import AuxiliaryHead
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

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: nn.Module | None = None,
        loss: str | list[str] | dict[str, float] | nn.Module = "bce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = -100,
        custom_loss: bool = False,
        custom_loss_kwargs: dict = None,
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = True,
        freeze_decoder: bool = True,
        freeze_head: bool = False,
        plot_on_val: bool | int = False,
        class_names: list[str] | None = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        path_to_record_metrics: str = None,
    ) -> None:
        super().__init__(
            model_args=model_args,
            model_factory=model_factory,
            model=model,
            loss=loss,
            aux_heads=aux_heads,
            aux_loss=aux_loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            custom_loss=custom_loss,
            custom_loss_kwargs=custom_loss_kwargs,
            lr=lr,
            optimizer=optimizer,
            optimizer_hparams=optimizer_hparams,
            scheduler=scheduler,
            scheduler_hparams=scheduler_hparams,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            freeze_head=freeze_head,
            plot_on_val=plot_on_val,
            class_names=class_names,
            test_dataloaders_names=test_dataloaders_names,
            lr_overrides=lr_overrides,
            path_to_record_metrics=path_to_record_metrics,
        )

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
