# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Inference script."""

import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from atsc.counting.training import get_pretrained_model_checkpoint

from .data import TrafficCountDataset
from .models.baseline import TARGET_CLASSES

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def inference(config: DictConfig) -> pd.DataFrame:
    """Perform inference with a trained model."""
    log.info(f"Site: {config.site}")

    log.info(f"Inference configuration\n{OmegaConf.to_yaml(config, resolve=True)}")

    inference_config = config.inference

    # Output path
    output_path = Path(config.inference.output_path).with_suffix(".csv")
    if output_path.exists() and not inference_config.get("overwrite", False):
        raise FileExistsError(f"Output file already exists at: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load inference index to make sure it exists
    df_predictions = pd.read_csv(inference_config.dataset.index)[["path"]]

    # Load model from checkpoint
    ckpt_path = get_pretrained_model_checkpoint(
        site=config.site,
        pretrained_model=inference_config.alias,
        work_folder=config.env.work_folder,
    )
    model: LightningModule = hydra.utils.get_class(config.model._target_).load_from_checkpoint(ckpt_path)

    # Load prediction dataset
    inference_dataset = TrafficCountDataset(**inference_config.dataset)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_config.batch_size,
        shuffle=False,
        persistent_workers=False,
        num_workers=inference_config.num_workers,
    )

    log.info(f"Inference dataset: {len(inference_dataset)} segments")

    # Predict
    trainer = Trainer(logger=False, accelerator=inference_config.accelerator)
    predictions = trainer.predict(
        model=model,
        dataloaders=inference_dataloader,
        return_predictions=True,
    )

    # Store predictions
    for label in TARGET_CLASSES:
        df_predictions[label] = torch.cat([batch[label] for batch in predictions])

    # Store results
    df_predictions.to_csv(output_path, index=False)
    log.info(f"Predictions stored at: {output_path}")

    return df_predictions


if __name__ == "__main__":
    inference()
