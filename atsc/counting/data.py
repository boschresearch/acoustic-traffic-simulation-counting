# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Data loader for traffic counting."""
import logging
from copy import copy
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

log = logging.getLogger(__name__)


class TrafficCountDataset(Dataset):
    """Traffic count dataset."""

    def __init__(
        self,
        *,
        root: str,
        index: str,
        sample_rate: int = 16000,
        duration: float = 60.0,
        split: str | None = None,
    ):
        """Initialize the traffic counting dataset."""
        super().__init__()
        self._root = Path(root)
        self._sample_rate = sample_rate
        self._expected_samples = int(sample_rate * duration)
        self._expected_channels = 4

        # Replace by CSV loader
        _index_dict = pd.read_csv(index).to_dict(orient="records")
        self._segments = [segment for segment in _index_dict if split is None or segment["split"] == split]

    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        return len(self._segments)

    def __getitem__(self, item: int) -> dict[str, Tensor | int]:
        """Load audio and return a single segment from the dataset."""
        segment = copy(self._segments[item])
        path = segment.pop("path")
        segment["audio"], _ = librosa.load(self._root.joinpath(path), sr=self._sample_rate, mono=False)

        if segment["audio"].shape[1] != self._expected_samples:
            raise RuntimeError(
                f"Audio segment at '{path}' has {segment['audio'].shape[1]} samples, "
                f"but {self._expected_samples} were expected."
            )

        if segment["audio"].shape[0] != self._expected_channels:
            raise RuntimeError(
                f"Audio segment at '{path}' has {segment['audio'].shape[0]} channels, "
                f"but {self._expected_channels} were expected."
            )

        return segment


class TrafficEntry(TypedDict):
    """Traffic entry as generated by the traffic model."""

    timestamp: pd.Timestamp
    vehicle_type: str
    direction: str
    speed: float


def _load_traffic_model(path: str | Path) -> list[TrafficEntry]:
    """
    Load traffic model from file.

    Args:
        path: Path to the traffic model CSV file.

    Returns:
        List of traffic entries, e.g.
            [{'timestamp': pd.Timestamp, 'vehicle_type': str, 'direction': str, 'speed': float},...]

    """
    df_events = pd.read_csv(path)
    df_events["timestamp"] = df_events["timestamp"].astype("datetime64[ns]")
    traffic_model = df_events.to_dict(orient="records")
    return traffic_model


def _load_passby_indices(
    site_sim_events_root: str | Path,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Path]]]:
    """
    Load pass-by indices for the given site.

    Args:
        site_sim_events_root: Path to the root folder of the simulated pass-by events for the specific site.

    Returns: (source_df_map,source_folder_map)
        source_df_map: A dictionary of dataframes containing the pass-by indices for each vehicle type and direction.
        source_folder_map: A dictionary of paths to the root folder of the simulated pass-by events
                           for each vehicle type and direction.

    """
    source_df_map: dict[str, dict[str, pd.DataFrame]] = {}
    source_folder_map: dict[str, dict[str, Path]] = {}
    for vehicle_type in ["car", "cv"]:
        source_df_map[vehicle_type] = {}
        source_folder_map[vehicle_type] = {}
        for direction in ["left", "right"]:
            source_folder = Path(site_sim_events_root) / vehicle_type / direction
            source_folder_map[vehicle_type][direction] = source_folder
            source_files = list(sorted(source_folder.glob("events-*.csv")))
            if len(source_files) == 0:
                raise ValueError(
                    f"Cannot synthesize dataset without source events."
                    f" No single pass-bys indices found in {source_folder}."
                )
            source_df_map[vehicle_type][direction] = pd.concat(
                [pd.read_csv(path) for path in source_files], ignore_index=True
            )
            if len(source_df_map[vehicle_type][direction]) < 1:
                raise ValueError(
                    f"Cannot synthesize dataset without source events."
                    f" No single pass-bys events found in {source_folder}."
                )
            log.debug(
                f"Loaded {len(source_df_map[vehicle_type][direction])} single pass-bys for"
                f" {vehicle_type}/{direction} from {source_folder}."
            )
    return source_df_map, source_folder_map


class SyntheticTrafficDataset(Dataset):
    """Synthetic traffic dataset."""

    def __init__(
        self,
        *,
        traffic_model_path: Path | str,
        segment_duration: float = 60.0,
        event_duration: float,
        sample_rate: int = 16000,
        random: bool,
        sim_events_root: Path | str,
        speed_tolerance: float = 10.0,
    ):
        """
        Initialize the synthetic traffic dataset.

        Args:
            traffic_model_path: Path to the traffic model CSV file
            segment_duration: Duration of the audio segment in seconds
            event_duration: Duration of source audio events in seconds
            sample_rate: Sample rate of the audio segment
            random: Set to randomly extract segments
            sim_events_root: Path to the root folder of the simulated pass-by events for the specific site
            speed_tolerance: Tolerance for speed matching in the source events [km/h]

        """
        super().__init__()
        self._traffic_model = _load_traffic_model(traffic_model_path)
        self._segment_duration = pd.Timedelta(segment_duration, "s")
        self._event_duration = pd.Timedelta(event_duration, "s")
        self._sample_rate = sample_rate
        self._random = random
        self._speed_tolerance = speed_tolerance

        # Define sequential segments for non-random extraction
        self._sequential_segments_start = pd.date_range(
            start=self._traffic_model[0]["timestamp"] - self._event_duration / 2,
            end=self._traffic_model[-1]["timestamp"] + self._event_duration / 2,
            freq=self._segment_duration,
        )
        self._num_segments = len(self._sequential_segments_start)

        # Load source events indices
        self._source_df_map, self._source_folder_map = _load_passby_indices(site_sim_events_root=sim_events_root)

    def __len__(self) -> int:
        """Number of segments in the dataset."""
        return len(self._sequential_segments_start)

    def __getitem__(self, item: int) -> dict[str, Tensor | int]:
        """Return a synthesized audio segment."""
        if self._random:
            segment_start = pd.Timestamp(
                np.random.uniform(
                    (self._traffic_model[0]["timestamp"] - self._event_duration / 2).to_datetime64(),
                    (self._traffic_model[-1]["timestamp"] + self._event_duration / 2).to_datetime64(),
                )
            )
        else:
            segment_start = self._sequential_segments_start[item]

        # Allocate audio buffer
        audio_segment = torch.zeros(
            (4, int(self._segment_duration.total_seconds() * self._sample_rate)),
            dtype=torch.float32,
        )

        # Retrieve events in the segment
        events_in_segment = [
            event
            for event in self._traffic_model
            if (event["timestamp"] >= segment_start) and (event["timestamp"] < (segment_start + self._segment_duration))
        ]

        # Loop through event to synthesize the segment
        for event in events_in_segment:
            # Retrieve index of pass-by events for the vehicle type and direction
            df_vehicle_direction = self._source_df_map[event["vehicle_type"]][event["direction"]]

            # Retrieve the pool of pass-by events with similar speed
            df_vehicle_direction_speed_pool = df_vehicle_direction[
                (df_vehicle_direction["speed"] >= (event["speed"] - self._speed_tolerance))
                & (df_vehicle_direction["speed"] <= (event["speed"] + self._speed_tolerance))
            ]

            if len(df_vehicle_direction_speed_pool) < 1:
                raise ValueError(f"Cannot find similar speed pass-by events in the source events.\nEvent: {event}")

            event_path = (
                self._source_folder_map[event["vehicle_type"]][event["direction"]]
                / df_vehicle_direction_speed_pool["path"].sample().iloc[0]
            )
            event_audio, _ = librosa.load(str(event_path), sr=None, mono=False)

            event_start_idx = int(
                self._sample_rate * (event["timestamp"] - self._event_duration / 2 - segment_start).total_seconds()
            )

            if event_start_idx < 0:
                # Event starts before the beginning of the segment
                event_audio = event_audio[:, event_start_idx:]
                audio_segment[:, : event_audio.shape[-1]] += event_audio
            elif event_start_idx + event_audio.shape[-1] > audio_segment.shape[-1]:
                # Event ends after the end of the segment
                event_audio = event_audio[:, : audio_segment.shape[-1] - event_start_idx]
                audio_segment[:, event_start_idx:] += event_audio
            else:
                # Event fits in the segment
                audio_segment[:, event_start_idx : event_start_idx + event_audio.shape[-1]] += event_audio

        # Create segment entry
        segment = {
            "wn": segment_start.isocalendar()[1],
            "dow": segment_start.day_of_week,
            "hour": segment_start.hour,
            "minute": segment_start.minute,
            "car_left": sum(
                [1 for event in events_in_segment if event["vehicle_type"] == "car" and event["direction"] == "left"]
            ),
            "car_right": sum(
                [1 for event in events_in_segment if event["vehicle_type"] == "car" and event["direction"] == "right"]
            ),
            "cv_left": sum(
                [1 for event in events_in_segment if event["vehicle_type"] == "cv" and event["direction"] == "left"]
            ),
            "cv_right": sum(
                [1 for event in events_in_segment if event["vehicle_type"] == "cv" and event["direction"] == "right"]
            ),
            "audio": audio_segment,
        }

        return segment
