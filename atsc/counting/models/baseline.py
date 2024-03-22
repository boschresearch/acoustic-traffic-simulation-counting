# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Baseline neural network for traffic counting."""
from typing import Any, Literal

import torch
import torchaudio.transforms
import torchmetrics
from lightning import LightningModule
from torch import Tensor

from .gcc import GCCPhat

TARGET_CLASSES = ["car_left", "car_right", "cv_left", "cv_right"]


class Baseline(LightningModule):
    """Baseline model for traffic counting."""

    class ConvBlock(torch.nn.Module):
        """
        Simple convolutional block.

        Layers:
            - Conv2d
            - ReLU
            - BatchNorm2d
        """

        def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple[int, int] | int,
            stride: tuple[int, int] | int,
            padding: tuple[int, int] | int,
        ):
            """
            Initialize ConvBlock layer.

            Args:
                in_channels: number of input channels
                out_channels: number of output channels
                kernel_size: size of the convolutional kernel
                stride: stride of the convolution
                padding: padding of the convolution
            """
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.non_linearity = torch.nn.ReLU()
            self.norm = torch.nn.BatchNorm2d(num_features=out_channels)

        def forward(self, inputs: Tensor) -> Tensor:
            """Forward step for ConvBlock layer."""
            return self.norm(self.non_linearity(self.conv(inputs)))

    class LinearBlock(torch.nn.Module):
        """
        Simple linear block.

        Layers:
            - Linear
            - ReLU
            - BatchNorm1d
        """

        def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
        ):
            """
            Initialize LinearBlock.

            Args:
                in_features: number of input features
                out_features: number of output features
            """
            super().__init__()
            self.conv = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
            )
            self.non_linearity = torch.nn.ReLU()
            self.norm = torch.nn.BatchNorm1d(num_features=out_features)

        def forward(self, inputs: Tensor) -> Tensor:
            """
            Forward pass of the layer.

            Args:
                inputs: (batch_size, n_time_frames, n_in_features)

            Returns: (batch_size, n_time_frames, n_out_features)
            """
            outputs = self.non_linearity(self.conv(inputs))
            outputs = torch.swapdims(outputs, 1, 2)
            outputs = self.norm(outputs)
            outputs = torch.swapdims(outputs, 1, 2)
            return outputs

    class ConvToTime(torch.nn.Module):
        """
        Convolution to time layer.

        Turns a (batch_size, n_channels, n_features, n_time_frames) tensor into a
        (batch_size, n_time_frames, n_out_features) tensor by aggregating over channels and features and swapping axes.
        """

        def __init__(self, mode: Literal["avg", "max", "cat"]):
            """
            Initialize ConvToTime layer.

            Args:
                mode: aggregation mode.
                      avg: average features across channels. n_out_features = n_features
                      max: max features across channels. n_out_features = n_features
                      cat: concatenate features across channels. n_out_features = n_channels * n_features
            """
            super().__init__()
            self.mode = mode

        def forward(self, inputs: Tensor) -> Tensor:
            """
            Forward pass of ConvToTime layer.

            Args:
                inputs: (batch_size, n_channels, n_features, n_time_frames) tensor

            Returns: (batch_size, n_time_frames, n_out_features)
                     if mode is avg or max, n_out_features = n_features
                    if mode is cat, n_out_features = n_channels * n_features

            """
            if self.mode == "avg":
                outputs = torch.mean(inputs, dim=1)
            elif self.mode == "max":
                outputs = torch.max(inputs, dim=1)[0]
            elif self.mode == "cat":
                outputs = torch.flatten(inputs, start_dim=1, end_dim=2)
            else:
                raise ValueError(f"mode {self.mode} not supported.")
            outputs = torch.swapdims(outputs, 1, 2)
            return outputs

    @staticmethod
    def _build_branch(
        *,
        n_channels: int,
        n_features: int,
        extractor_filters: tuple[int, ...],
        branch_td_filters: tuple[int, ...],
        conv2time_mode: Literal["avg", "max", "cat"],
    ) -> torch.nn.Module:
        """Build a feature extractor branch."""
        return torch.nn.Sequential(
            *[
                Baseline.ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
                for in_channels, out_channels in zip(
                    (n_channels, *extractor_filters[:-1]), extractor_filters, strict=True
                )
            ],
            Baseline.ConvToTime(mode=conv2time_mode),
            *[
                Baseline.LinearBlock(
                    in_features=in_features,
                    out_features=out_features,
                )
                for in_features, out_features in zip(
                    (
                        (extractor_filters[-1] if conv2time_mode == "cat" else 1)
                        * (n_features // (2 ** len(extractor_filters))),
                        *branch_td_filters[:-1],
                    ),
                    branch_td_filters,
                    strict=True,
                )
            ],
        )

    def __init__(
        self,
        *,
        n_channels: int = 4,
        n_mels: int,
        n_gcc: int,
        stft_params: dict[str, Any],
        melscale_params: dict[str, Any],
        extractor_filters: tuple[int, ...] = (32, 32, 64),
        conv2time_mode: Literal["avg", "max", "cat"] = "cat",
        branch_td_filters: tuple[int, ...] = (128, 128),
        merger_td_filters: tuple[int, ...] = (128, 128, 128),
        recurrent_layers: int = 2,
        recurrent_hidden_units: int = 128,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
    ):
        """
        Initialize the baseline model.

        Args:
            n_channels: number of channels in the input audio
            n_mels: number of mel coefficients
            n_gcc: number of GCC-PHAT coefficients to preserve
            stft_params: parameters for `torch.stft`
            melscale_params: parameters for `torchaudio.transforms.MelScale`
            extractor_filters: number of filters in each layer of the feature extractor branch
            conv2time_mode: aggregation mode for the ConvToTime layer
            branch_td_filters: number of filters in each layer of the time domain branch
            merger_td_filters: number of filters in each layer of the merger branch
            recurrent_layers: number of recurrent layers
            recurrent_hidden_units: number of hidden units in the recurrent layer
            optimizer: optimizer class to use
        """
        super().__init__()
        self.save_hyperparameters()
        self._stft_params = stft_params
        self._stft_params.setdefault("return_complex", True)
        self._stft_params.setdefault("n_fft", 1024)
        self._stft_params.setdefault("hop_length", 160)
        self._stft_params.setdefault("center", False)
        self.register_buffer(
            "_stft_window",
            torch.hann_window(self._stft_params["n_fft"]),
            persistent=False,
        )

        melscale_params.setdefault("sample_rate", 16000)
        melscale_params.setdefault("n_stft", 513)
        self.logmel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(**melscale_params),
            torchaudio.transforms.AmplitudeToDB(),
        )

        self.gcc = GCCPhat(n_gcc)
        gcc_channels = (n_channels * (n_channels - 1)) // 2
        self.logmel_norm = torch.nn.BatchNorm1d(num_features=n_channels * n_mels)
        self.gcc_norm = torch.nn.BatchNorm1d(num_features=gcc_channels * n_gcc)

        self.logmel_extractor = Baseline._build_branch(
            n_channels=n_channels,
            n_features=n_mels,
            extractor_filters=extractor_filters,
            branch_td_filters=branch_td_filters,
            conv2time_mode=conv2time_mode,
        )
        self.gcc_extractor = Baseline._build_branch(
            n_channels=gcc_channels,
            n_features=n_gcc,
            extractor_filters=extractor_filters,
            branch_td_filters=branch_td_filters,
            conv2time_mode=conv2time_mode,
        )

        self.merger_td = torch.nn.Sequential(
            *[
                Baseline.LinearBlock(in_features=in_features, out_features=out_features)
                for in_features, out_features in zip(
                    (branch_td_filters[-1] * 2, *merger_td_filters[:-1]), merger_td_filters, strict=True
                )
            ],
        )

        self.recurrent = torch.nn.GRU(
            input_size=merger_td_filters[-1],
            hidden_size=recurrent_hidden_units,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(in_features=recurrent_hidden_units, out_features=len(TARGET_CLASSES)),
            torch.nn.ReLU(),
        )

        self.metrics: dict[str, dict[str, dict[str, torchmetrics.Metric]]] = {
            "train": {
                "mae": {label: torchmetrics.regression.mae.MeanAbsoluteError() for label in TARGET_CLASSES},
            },
            "val": {
                "mae": {label: torchmetrics.regression.mae.MeanAbsoluteError() for label in TARGET_CLASSES},
            },
        }

        # Register metrics to allow automatic device placement
        for split in self.metrics:
            for metric_name, metric_dict in self.metrics[split].items():
                for label, metric in metric_dict.items():
                    self.register_module(f"metric_{split}_{metric_name}_{label}", metric)

        self.optimizer_partial = optimizer

    def forward(self, audio: Tensor) -> dict[str, Tensor]:
        """
        Forward pass of the baseline model.

        Args:
            audio: (batch_size, n_channels, n_samples) raw waveform

        Returns:
            dictionary containing the counts for each class, e.g.
            {
                'car_left': Tensor([1,2,1]),
                'car_right': Tensor([4,5,1]),
                'cv_left': Tensor([0,1,3]),
                'cv_right': Tensor([1,0,0])
            }

        """
        batch_size, n_channels, _ = audio.shape

        # Output: (batch_size, n_channels, n_fft/2+1, n_time_frames) complex
        audio_flat = audio.view(batch_size * n_channels, -1)
        stft_flat = torch.stft(audio_flat, **self._stft_params, window=self._stft_window)
        n_time_frames = stft_flat.shape[-1]
        stft = stft_flat.view(batch_size, n_channels, -1, n_time_frames)

        # Output: (batch_size, n_channels, n_mels, n_time_frames) real
        logmel_out = self.logmel(torch.abs(stft))

        # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
        gcc_out = self.gcc(stft)

        # Output: (batch_size, n_channels, n_mels, n_time_frames) real
        logmel_norm = self.logmel_norm(logmel_out.reshape(batch_size, -1, n_time_frames)).reshape(*logmel_out.shape)

        # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
        gcc_norm = self.gcc_norm(gcc_out.reshape(batch_size, -1, n_time_frames)).reshape(*gcc_out.shape)

        # Output: (batch_size, n_time_frames, n_logmel_features) real
        logmel_branch_out = self.logmel_extractor(logmel_norm)

        # Output: (batch_size, n_time_frames, n_gcc_features) real
        gcc_branch_out = self.gcc_extractor(gcc_norm)

        # Output: (batch_size, n_time_frames, n_mel_features+n_gcc_features) real
        concat_out = torch.cat((logmel_branch_out, gcc_branch_out), dim=2)

        # Output: (batch_size, n_time_frames, n_merger_features) real
        merger_out = self.merger_td(concat_out)

        # Output: (batch_size, n_time_frames, n_recurrent_features) real
        recurrent_out, _ = self.recurrent(merger_out)

        # Output: (batch_size, 4) real
        count_out = self.regression(recurrent_out[:, -1, :])

        return dict(
            car_right=count_out[:, 0],
            car_left=count_out[:, 1],
            cv_right=count_out[:, 2],
            cv_left=count_out[:, 3],
        )

    def _compute_loss(self, *, split: str, output: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        """
        Compute the loss for the given split.

        Args:
            split: split name, used for logging purpose
            output: model output containing the counts for each class, e.g.
                {
                    'car_left': Tensor([1,2,1]),
                    'car_right': Tensor([4,5,1]),
                    'cv_left': Tensor([0,1,3]),
                    'cv_right': Tensor([1,0,0])
                }
            batch: batch containing the ground truth counts for each class, e.g.
                {
                    'car_left': Tensor([1,2,1]),
                    'car_right': Tensor([4,5,1]),
                    'cv_left': Tensor([0,1,3]),
                    'cv_right': Tensor([1,0,0])
                }

        Returns:
            the loss, computed as the average loss for across all classes

        """
        batch_size = batch["audio"].shape[0]

        loss_dict = {
            f"{split}/loss/{label}": torch.nn.functional.mse_loss(output[label], batch[label].float())
            for label in TARGET_CLASSES
        }
        self.log_dict(
            loss_dict,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        loss = torch.mean(torch.stack(list(loss_dict.values())))
        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def _update_metrics(self, *, split: str, output: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        """Update the metrics for the given split."""
        split_metrics = self.metrics[split]
        for metric_dict in split_metrics.values():
            for label, metric in metric_dict.items():
                metric.update(output[label], batch[label].float())

    def _log_metrics(self, *, split: str) -> None:
        """Log the metrics for the given split."""
        split_metrics = self.metrics[split]
        for metric_name, metric_dict in split_metrics.items():
            for label, metric in metric_dict.items():
                self.log(
                    f"{split}/{metric_name}/{label}",
                    metric.compute(),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
                metric.reset()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the baseline model.

        Args:
            batch: dictionary with keys:
                audio: (batch_size, num_channels, num_samples) float tensor, multichannel raw audio
                wn: (batch_size,) long tensor, week number
                dow: (batch_size,) long tensor, day of the week
                hour: (batch_size,) long tensor, hour of the day
                minute: (batch_size,) long tensor, minute of the hour
                car_left: (batch_size,) long tensor, ground truth count for left lane cars
                car_right: (batch_size,) long tensor, ground truth count for right lane cars
                cv_left: (batch_size,) long tensor, ground truth count for left lane commercial vehicles
                cv_right: (batch_size,) long tensor, ground truth count for right lane commercial vehicles
            batch_idx: index of the batch
        """
        output = self(batch["audio"])
        self._update_metrics(split="train", output=output, batch=batch)
        return self._compute_loss(split="train", output=output, batch=batch)

    def on_train_epoch_end(self) -> None:
        """Hook at the end of the training epoch."""
        self._log_metrics(split="train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """
        Validation step for the baseline model.

        Args:
            batch: dictionary with keys:
                audio: (batch_size, num_channels, num_samples) float tensor, multichannel raw audio
                wn: (batch_size,) long tensor, week number
                dow: (batch_size,) long tensor, day of the week
                hour: (batch_size,) long tensor, hour of the day
                minute: (batch_size,) long tensor, minute of the hour
                car_left: (batch_size,) long tensor, ground truth count for left lane cars
                car_right: (batch_size,) long tensor, ground truth count for right lane cars
                cv_left: (batch_size,) long tensor, ground truth count for left lane commercial vehicles
                cv_right: (batch_size,) long tensor, ground truth count for right lane commercial vehicles
            batch_idx: index of the batch
        """
        output = self(batch["audio"])
        self._update_metrics(split="val", output=output, batch=batch)
        self._compute_loss(split="val", output=output, batch=batch)

    def on_validation_epoch_end(self) -> None:
        """Hook at the end of the validation epoch."""
        self._log_metrics(split="val")

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """
        Prediction step for the baseline model.

        Args:
            batch: dictionary with keys:
                audio: (batch_size, num_channels, num_samples) float tensor, multichannel raw audio
                wn: (batch_size,) long tensor, week number
                dow: (batch_size,) long tensor, day of the week
                hour: (batch_size,) long tensor, hour of the day
                minute: (batch_size,) long tensor, minute of the hour
            batch_idx: index of the batch
        """
        output = self(batch["audio"])
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the baseline model."""
        return self.optimizer_partial(self.parameters())
