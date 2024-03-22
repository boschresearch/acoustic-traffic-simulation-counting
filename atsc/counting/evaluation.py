# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluation script."""
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import kendalltau

from .models.baseline import TARGET_CLASSES

log = logging.getLogger(__name__)

METRICS = ["Kendall's Tau Corr", "RMSE"]
SHIFT_TOLERANCE = 1


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def evaluate(config: DictConfig) -> pd.DataFrame:
    """Evaluate inference output against ground truth and compute metrics."""
    log.info(f"Site: {config.site}")

    log.info(f"Evaluation configuration\n{OmegaConf.to_yaml(config)}")

    # Output path
    output_path = Path(config.evaluation.output_path).with_suffix(".csv")
    if output_path.exists():
        raise FileExistsError(f"Output file already exists at: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load predictions and ground truth
    df_pred = pd.read_csv(config.inference.output_path)
    df_gt = pd.read_csv(config.inference.dataset.index)

    if len(df_pred) != len(df_gt):
        raise ValueError("Predictions and ground truth must contain the same number of samples.")

    if not all([col in df_pred.columns for col in TARGET_CLASSES]):
        raise ValueError("Missing columns in predictions.")
    if not all([col in df_gt.columns for col in TARGET_CLASSES]):
        raise ValueError("Missing columns in groudn truth.")

    df_gt.sort_values("path", inplace=True)
    df_pred.sort_values("path", inplace=True)

    ktau_corr_dict, rmse_dict = {}, {}
    for label in TARGET_CLASSES:
        gt_scores = df_gt[label].values
        pred_scores = df_pred[label].values

        # RMSE score
        rmse = np.sqrt(1.0 / len(gt_scores) * np.sum((gt_scores - pred_scores) ** 2.0))

        # correlation-based metrics
        ktau_corr = kendalltau(gt_scores, pred_scores).correlation

        # output results
        ktau_corr_dict[label] = round(ktau_corr, 3)
        rmse_dict[label] = round(rmse, 3)

    results = pd.DataFrame([ktau_corr_dict, rmse_dict], index=METRICS)
    log.info(results)

    # Store in desired directory
    results.to_csv(output_path, index=True, index_label="Metric")
    log.info(f"Evaluation results stored at: {output_path}")

    return results


if __name__ == "__main__":
    evaluate()
