# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Single pass-by events simulation script."""

import json
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import pandas as pd
import pyroadacoustics as pyroad
import soundfile as sf
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from atsc.modeling.traffic import _generate_speed_distribution

from .source_synthesis import SourceSynthesis

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VehicleSimulator:
    """Simulator for pass-by events."""

    def __init__(
        self,
        sample_rate: int = 16000,
        event_duration: float = 30.0,
        array_height: float = 2.7,
        array_distance_to_street_side: float = 4.0,
        lane_width: float = 3.5,
        source_model: Literal["hm", "hm+bd"] = "hm+bd",
        engine_samples_directory: str | None = None,
        atmospheric_params: dict | None = None,
    ) -> None:
        """
        Initialize the simulator.

        Assumes array is parallel to the road.

        Args:
            sample_rate: sample rate for signals generation [Hz]
            event_duration: duration of the event [s]
            array_height: height of the microphone array from the ground [m]
            array_distance_to_street_side: distance of the microphone array to the street side [m]
            lane_width: width of the road lane [m]
            source_model: source model type. Either "hm" for harmonoise or "hm+bd" for harmonoise + Baldan.
                          See source_synthesis.py for details
            engine_samples_directory: directory containing the engine sound samples
            atmospheric_params: atmospheric parameters,
                                including temperature [°C], pressure [atm], and relative humidity [%]

        """
        self.sample_rate = sample_rate
        self.event_duration = event_duration
        self.source_model = source_model
        self.atmospheric_params = {"T": 20, "p": 1, "rel_hum": 50} if atmospheric_params is None else atmospheric_params

        self.microphone_array = np.array(
            [
                [0.12, 0, array_height],
                [0.04, 0, array_height],
                [-0.04, 0, array_height],
                [-0.12, 0, array_height],
            ]
        )

        self.array_distance_to_street_side = array_distance_to_street_side
        self.lane_width = lane_width

        self.source_synthesis_module = SourceSynthesis(
            self.sample_rate,
            self.event_duration,
            self.source_model,
            engine_samples_directory,
        )

    def run_simulation(
        self, source_signal: np.ndarray, vehicle_speed: float, vehicle_trajectory: np.ndarray
    ) -> np.ndarray:
        """
        Run simulation for a single source.

        Args:
            source_signal: (num_samples,) mono source signal
            vehicle_speed: vehicle speed, assumed to be constant [km/h]
            vehicle_trajectory: (2, 3) start and end point of the trajectory, e.g.
                                [(x_start, y_start, z_start), (x_end, y_end, z_end)]

        Returns: (4, num_samples,) simulated signal

        """
        env = pyroad.Environment(
            fs=self.sample_rate,
            temperature=self.atmospheric_params["T"],
            pressure=self.atmospheric_params["p"],
            rel_humidity=self.atmospheric_params["rel_hum"],
        )
        env.set_simulation_params(
            interp_method="Allpass",
            include_reflection=True,
            include_air_absorption=True,
        )

        env.add_microphone_array(mic_locs=self.microphone_array)
        env.add_source(
            position=vehicle_trajectory[0],
            signal=source_signal,
            trajectory_points=vehicle_trajectory,
            source_velocity=np.array([vehicle_speed]),
        )
        return env.simulate()

    def simulate_passby(self, vehicle_type: str, speed: float, direction: str) -> np.ndarray:
        """
        Simulate a single pass-by event.

        Args:
            vehicle_type: type of vehicle, either "car" or "cv"
            speed: vehicle speed [km/h], constant during the event
            direction: direction of the vehicle, either "left" or "right"

        Returns:
            (4, num_samples) simulated signal
        """
        # Convert speed to m/s
        speed = speed / 3.6

        # Set height of higher source based on vehicle type
        higher_src_height = 0.3 if vehicle_type == "car" else 0.75

        # Define trajectory for right-to-left vehicle direction
        horizontal_distance = self.array_distance_to_street_side + self.lane_width
        traj_higher_src = np.array(
            [
                [
                    self.event_duration / 2 * speed,
                    horizontal_distance,
                    higher_src_height,
                ],
                [
                    -self.event_duration / 2 * speed,
                    horizontal_distance,
                    higher_src_height,
                ],
            ]
        )
        traj_lower_src = np.array(
            [
                [self.event_duration / 2 * speed, horizontal_distance, 0.01],
                [-self.event_duration / 2 * speed, horizontal_distance, 0.01],
            ]
        )

        # If vehicle direction is left-to-right, invert direction
        if direction == "right":
            traj_higher_src[:, 1] = horizontal_distance - self.lane_width
            traj_lower_src[:, 1] = horizontal_distance - self.lane_width
            traj_higher_src[:, 0] = -traj_higher_src[:, 0]
            traj_lower_src[:, 0] = -traj_lower_src[:, 0]

        log.debug("generate source signal")
        # Generate source signal
        (
            higher_src_signal,
            lower_src_signal,
        ) = self.source_synthesis_module.generate_harmonoise_source_signals(vehicle_type, speed)

        # Simulate motion along trajectory
        log.debug("simulate higher")
        sim_higher = self.run_simulation(higher_src_signal, speed, traj_higher_src)

        log.debug("simulate lower")
        sim_lower = self.run_simulation(lower_src_signal, speed, traj_lower_src)

        # Sum signals using harmonoise mixture
        signal = 2 / np.sqrt(5) * sim_higher + 1 / np.sqrt(5) * sim_lower

        return signal


def _simulate_and_save(
    args: tuple[int, float],
    vehicle_type: str,
    direction: str,
    simulator: VehicleSimulator,
    folder: Path,
    overwrite: bool,
) -> dict:
    """
    Simulate a single pass by and save the event to file.

    Args:
        args: tuple containing the index and speed of the event
        vehicle_type: type of vehicle
        direction: direction of the vehicle
        simulator: vehicle simulator
        folder: destination folder
        overwrite: overwrite existing files

    Return:
        dictionary containing: path, index, speed, vehicle_type, and direction of the event

    """
    index, speed = args

    file_path = folder / f"event-{index:04d}_speed-{int(speed):03d}.flac"
    if overwrite or not file_path.exists():
        event_audio = simulator.simulate_passby(vehicle_type=vehicle_type, speed=speed, direction=direction)
        sf.write(
            file=str(file_path),
            data=event_audio.T,
            samplerate=simulator.sample_rate,
            format="flac",
            subtype="PCM_16",
        )

    return {
        "path": file_path.relative_to(folder),
        "index": index,
        "speed": int(speed),
        "vehicle_type": vehicle_type,
        "direction": direction,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def simulate_events(config: DictConfig) -> None:
    """Simulate a pool of single pass-by events."""
    log.info(f"Events simulation configuration\n{OmegaConf.to_yaml(config)}")

    events_config = config.simulation
    overwrite = events_config.get("overwrite", False)

    # Read site metadata
    with (Path(config.env.real_root) / config.site / "meta.json").open("r") as fp_meta:
        site_meta = json.load(fp_meta)
        # site_meta has the following structure
        # {
        #   "geometry": {
        #     "array-height": 2.7,
        #     "distance-to-street-side": 4.0
        #   },
        #   "traffic": {
        #     "max-pass-by-speed": 100.0,
        #     "max-traffic-density": 1000
        #   }
        # }

    simulator = VehicleSimulator(
        event_duration=events_config.event_duration,
        array_height=site_meta["geometry"]["array-height"],
        array_distance_to_street_side=site_meta["geometry"]["distance-to-street-side"],
        lane_width=events_config.lane_width,
        source_model=events_config.source_model,
        engine_samples_directory=config.env.engine_sound_dir,
    )

    # Prepare events generation
    distribution = _generate_speed_distribution(site_meta["traffic"]["max-pass-by-speed"])
    speeds = distribution.rvs(size=events_config.num_events, random_state=events_config.seed)
    indexes = range(
        events_config.init_counter,
        events_config.init_counter + events_config.num_events,
    )

    with Pool(events_config.num_workers) as pool:
        for vehicle_type in events_config.vehicle_types:
            for direction in events_config.directions:
                folder = Path(events_config.output_folder) / vehicle_type / direction

                events_list_path = (
                    folder / f"events-{events_config.init_counter:04d}-"
                    f"{events_config.init_counter + events_config.num_events - 1:04d}.csv"
                )
                if events_list_path.exists() and not overwrite:
                    log.info(f"Events list already exists: {events_list_path}. Skipping simulation.")
                    continue

                log.info(f"Simulating single pass-by events for {config.site} {vehicle_type} {direction}")

                # Create destination folder
                folder.mkdir(exist_ok=True, parents=True)

                # Create partial function for pool
                simulate_and_save_event = partial(
                    _simulate_and_save,
                    simulator=simulator,
                    vehicle_type=vehicle_type,
                    direction=direction,
                    folder=folder,
                    overwrite=events_config.get("overwrite", False),
                )

                # Simulate
                events_list = list(
                    tqdm(
                        pool.imap(simulate_and_save_event, zip(indexes, speeds, strict=True)),
                        desc=f"Simulating {vehicle_type} {direction} events",
                        total=events_config.num_events,
                    )
                )

                # Store events list
                df_events = pd.DataFrame(events_list)
                log.info(f"Saving events list to: {events_list_path}")
                df_events.to_csv(events_list_path, index=False)


if __name__ == "__main__":
    simulate_events()
