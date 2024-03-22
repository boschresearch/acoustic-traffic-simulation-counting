# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""
Traffic modeling module.

Starting from traffic statistic of a specific site, generate a list of pass-by events for a given time period.
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import rv_continuous, truncnorm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrafficModel:
    """
    Traffic modeling class.

    Given traffic statistics and some additional parameters, generate a list of pass-by events for a given time period.

    This is a very simplistic model of traffic, that does not take into account the day of week,
    and models traffic density as a single gaussian with a peek at 12.
    """

    def __init__(
        self,
        min_time_between_consecutive_events: pd.Timedelta,
        num_hours: int,
        start_datetime: pd.Timestamp,
        max_cars_per_hour: int,
        max_cvs_per_hour: int,
        speed_limit: float,
        distribution_variance: float = 4.5,
    ):
        """
        Initialize the traffic model.

        Args:
            min_time_between_consecutive_events: minimum time between consecutive pass-bys in the same direction
            num_hours: total number of hours to generate events for
            start_datetime: start datetime for events generation
            max_cars_per_hour: maximum number of cars per hour, in either directions
            max_cvs_per_hour:  maximum number of commercial vehicles per hour, in either directions
            speed_limit: street speed limit [km/h]
            distribution_variance: variance of the normal distribution used to model the number of vehicles per hour
        """
        self.min_time_between_consecutive_events = min_time_between_consecutive_events
        self.num_hours = num_hours
        self.start_datetime = start_datetime

        self.max_cars_per_hour = max_cars_per_hour
        self.max_cvs_per_hour = max_cvs_per_hour

        self.speed_distribution = _generate_speed_distribution(speed_limit)

        self.distribution_variance = distribution_variance

        self.vehicles_per_hour = {
            "car": self._generate_day_distribution(self.max_cars_per_hour),
            "cv": self._generate_day_distribution(self.max_cvs_per_hour),
        }

    def _generate_day_distribution(self, max_vehicles: int) -> np.ndarray:
        """
        Generate traffic distribution for a day.

        Generate normal distribution centered at 12, with peak corresponding to the maximum number of vehicles per hour.
        The value of the distribution denotes the number of vehicles passing in each hour of a whole day.

        Args:
            max_vehicles: maximum number of vehicles per hour

        Returns:
            (24,) array of number of vehicles per hour
        """
        avg_num_vehicles_per_hour = np.zeros(24)
        variance = self.distribution_variance

        for i in range(24):
            avg_num_vehicles_per_hour[i] = max_vehicles * np.exp(-((i - 12) ** 2) / (2 * variance**2))

        return avg_num_vehicles_per_hour

    def _generate_event(self, vehicle_type: str, current_hour: pd.Timestamp) -> dict:
        """
        Generate a single pass-by event.

        Args:
            vehicle_type: vehicle type (car or cv)
            current_hour: timestamp of hour for which the event is generated.
                          A random time within the hour will be chosen.

        Returns:
            event: dictionary containing the details of the event, e.g.
                   {"vehicle_type": "car", "timestamp": "2024-01-01 12:34:56", "direction": "left", "speed": 50}

        """
        timestamp = current_hour + pd.Timedelta(np.random.rand(), unit="hour").floor("s")
        direction = np.random.choice(["left", "right"])
        speed = self.speed_distribution.rvs()

        event = {
            "vehicle_type": vehicle_type,
            "timestamp": timestamp,
            "direction": direction,
            "speed": int(speed),
        }

        return event

    def _check_minimum_distance_between_events(self, timestamp: pd.Timestamp, events: list[dict]) -> bool:
        """
        Check if the minimum time between consecutive events is respected.

        Args:
            timestamp: timestamp of a candidate event
            events: list of existing events

        Returns:
            bool: True if the minimum time between consecutive events is respected, False otherwise

        """
        time_diff = [
            (max(el["timestamp"], timestamp) - min(el["timestamp"], timestamp))
            < self.min_time_between_consecutive_events
            for el in events
        ]
        return any(time_diff)

    def _draw_number_of_vehicles(self, avg_num_vehicles: int) -> int:
        """
        Draw the number of vehicles for a given hour.

        The number of vehicles is drawn from a normal distribution
        with a variance of 15% of the average number of vehicles.

        Args:
            avg_num_vehicles: average number of vehicles per hour

        Returns:
            number of vehicles for the given hour
        """
        num_vehicles = np.random.normal(avg_num_vehicles, 0.15 * avg_num_vehicles)
        num_vehicles = int(np.max([2, num_vehicles]))  # Always assume at least 2 vehicles per hour
        return num_vehicles

    def generate_taffic_flow(self) -> pd.DataFrame:
        """
        Generate traffic flow for the given time period.

        Generate a list of pass-by events based on the initialization parameters

        Returns:
            dataframe containing the list of pass-by events with columns: vehicle_type, direction, timestamp, speed

        """
        start_ts = self.start_datetime

        event_list: dict[str, list[dict]] = {"left": [], "right": []}

        for hour_idx in range(self.num_hours):
            start_current_hour = start_ts + hour_idx * pd.Timedelta("1h")

            for vehicle_type in ["car", "cv"]:
                num_vehicles = self._draw_number_of_vehicles(
                    self.vehicles_per_hour[vehicle_type][int(start_current_hour.hour)]
                )

                for _ in range(num_vehicles):
                    while True:
                        event = self._generate_event(vehicle_type, start_current_hour)
                        if not self._check_minimum_distance_between_events(
                            event["timestamp"], event_list[event["direction"]]
                        ):
                            break
                    event_list[event["direction"]].append(event)

        # Generate dataframe
        combined_event_list = event_list["left"] + event_list["right"]
        df_events = pd.DataFrame(combined_event_list).sort_values("timestamp").reset_index(drop=True)
        return df_events


def _generate_speed_distribution(speed_limit: float) -> rv_continuous:
    """
    Generate speed distribution.

    Args:
        speed_limit: speed limit on the road

    Returns:
        truncated normal continuous random variable
    """
    max_speed = speed_limit * 1.1  # Assume vehicles are driving at most 10% above speed limit
    min_speed = speed_limit * 0.5  # Assume vehicles are driving at least at 50% the speed limit
    std_dev = 0.2 * speed_limit  # Assume 20% of the speed limit as standard deviation
    mean = 0.9 * speed_limit  # Assume 90% of the speed limit as mean

    lower_bound = (min_speed - mean) / std_dev
    upper_bound = (max_speed - mean) / std_dev

    # Generate continuous random variable for the truncated normal distribution
    rv = truncnorm(lower_bound, upper_bound, loc=mean, scale=std_dev)
    return rv


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def model(config: DictConfig) -> None:
    """Generate traffic model for a given site."""
    log.info(f"Traffic modeling configuration\n{OmegaConf.to_yaml(config)}")

    traffic_config = config.traffic
    overwrite = traffic_config.get("overwrite", False)

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

    np.random.seed(traffic_config.seed)

    for split in ["train", "val"]:
        split_config = traffic_config[split]
        split_output_path = split_config.output_path
        if Path(split_output_path).exists() and not overwrite:
            log.info(f"Traffic model for {split} already exists at {split_output_path}.")
            continue
        log.info("Generating traffic model for {split}.")
        simulator = TrafficModel(
            min_time_between_consecutive_events=pd.Timedelta(traffic_config.min_time_between_consecutive_events),
            num_hours=split_config.num_hours,
            start_datetime=pd.Timestamp(split_config.start_datetime),
            max_cars_per_hour=site_meta["traffic"]["max-traffic-density"] * traffic_config.car_max_fraction,
            max_cvs_per_hour=site_meta["traffic"]["max-traffic-density"] * traffic_config.cv_max_fraction,
            speed_limit=site_meta["traffic"]["max-pass-by-speed"],
        )

        df_events = simulator.generate_taffic_flow()

        Path(split_output_path).parent.mkdir(parents=True, exist_ok=True)
        df_events.to_csv(split_output_path, index=False)
        log.info(f"Traffic model saved in {split_output_path}.")


if __name__ == "__main__":
    model()
