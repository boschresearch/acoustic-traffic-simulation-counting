# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Train an acoustic traffic counting model."""
import logging
import os
from pathlib import Path

import coolname
import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, NeptuneLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


def get_pretrained_model_checkpoint(*, site: str, pretrained_model: str | Path, work_folder: str | Path) -> Path:
    """
    Compute pre-trained model checkpoint path.

    Args:
        site: Site name.
        pretrained_model: Pre-trained model name or path
        work_folder: Work folder.

    Returns:
        path to a pre-trained model checkpoint.
    """
    if Path(pretrained_model).is_file():
        return Path(pretrained_model)
    return Path(work_folder) / "counting" / site / pretrained_model / "checkpoints" / "best.ckpt"


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def train(config: DictConfig) -> Trainer:
    """
    Training function for acoustic traffic counting model.

    Args:
        config: Hydra configuration object.
    """
    log.info(f"Site: {config.site}")

    training_config = config.training
    if training_config.alias is None:
        alias = coolname.generate_slug(2)
        training_config.alias = alias
    else:
        alias = training_config.alias

    log.info(f"Training alias: {alias}")

    output_dir: str = training_config.output_folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Output folder: {output_dir}")

    config_path = Path(output_dir) / "config.yaml"
    if config_path.exists() and not training_config.get("overwrite", False):
        raise FileExistsError(
            f"Configuration file already exists at {config_path}. Please set `training.overwrite=True` to overwrite."
        )
    OmegaConf.save(config, config_path, resolve=True)

    log.info(f"Training configuration\n{OmegaConf.to_yaml(config, resolve=True)}")

    seed_everything(training_config.seed)
    log.info(f"Seed: {training_config.seed}")

    train_dataset: Dataset = instantiate(training_config.train_dataset)
    log.info(f"Train dataset: {len(train_dataset)} segments")

    val_dataset: Dataset = instantiate(training_config.val_dataset)
    log.info(f"Val dataset: {len(val_dataset)} segments")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        persistent_workers=training_config.num_workers > 0,
        num_workers=training_config.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        persistent_workers=training_config.num_workers > 0,
        num_workers=training_config.num_workers,
    )

    training_config.trainer["log_every_n_steps"] = min(
        training_config.trainer["log_every_n_steps"], len(train_dataloader)
    )

    model: LightningModule = hydra.utils.instantiate(config.model)

    if training_config.get("pretrained_model", None) is not None:
        pretrained_ckpt_path = get_pretrained_model_checkpoint(
            site=config.site,
            pretrained_model=training_config.pretrained_model,
            work_folder=config.env.work_folder,
        )
        log.info(f"Loading pretrained model from {pretrained_ckpt_path}")
        state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys):
            log.warning(f"Missing keys: {', '.join(missing_keys)}")
        if len(unexpected_keys):
            log.warning(f"Unexpected keys: {', '.join(unexpected_keys)}")

    # Preparing loggers
    loggers = []

    try:
        proxies = {
            key: os.getenv(key)
            for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
            if key in os.environ
        }
        neptune_tags = training_config.get("tags", None)
        loggers.append(
            NeptuneLogger(
                proxies=proxies,
                name=alias,
                log_model_checkpoints=False,
                tags=OmegaConf.to_container(neptune_tags) if neptune_tags is not None else None,
            )
        )
        log.info("Neptune is installed. Adding logger.")
    except ModuleNotFoundError:
        pass
    try:
        if training_config.get("tags", None) is not None:
            mlflow_tags = {tag: tag for tag in training_config.tags}
        else:
            mlflow_tags = None
        loggers.append(MLFlowLogger(run_name=alias, tags=mlflow_tags))
        log.info("MLFlow is installed. Adding logger.")
    except ModuleNotFoundError:
        pass

    if len(loggers) == 0:
        log.info("No loggers found. Adding CSVLogger.")
        loggers.append(CSVLogger(save_dir=output_dir, name=alias))

    hparams = OmegaConf.to_container(config, resolve=True)
    hparams["output_dir"] = output_dir
    for logger in loggers:
        logger.log_hyperparams(hparams)

    # Prepare callbacks
    early_stop = EarlyStopping(**training_config.callbacks.get("early_stopping", {}))

    model_checkpoint_params = OmegaConf.to_container(
        training_config.callbacks.get("model_checkpoint", {}), resolve=True
    )
    model_checkpoint_params.setdefault("dirpath", Path(output_dir) / "checkpoints")
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)

    # Prepare trainer and fit
    trainer = Trainer(
        **training_config.get("trainer", {}),
        logger=loggers,
        callbacks=[model_checkpoint, early_stop],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        **training_config.get("fit", {}),
    )

    log.info(f"Best model checkpoint at: {model_checkpoint.best_model_path}")

    return trainer


if __name__ == "__main__":
    load_dotenv()
    train()
