# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Source synthesis module."""

import os
import random
import warnings
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

from . import pyoctaveband

# Coefficients specified by the Harmonoise model for the sound pressure level (SPL) produced by the rolling and
# propulsion noise components, expressed in one-third octave frequency bands between 20Hz and 12.5kHz.
#
# Coefficients are specified for cars (i.e., category 1 vehicles) and trucks (i.e., category 3 vehicles), as from
#
# [1] H. Jonasson, U. Sandberg, G. van Blokland, J. Ejsmont, G. Watts, and M. Luminari,
# “Source modelling of road vehicles,”
# Statens Provningsanstalt, 2004. Available: https://www.diva-portal.org/smash/record.jsf?pid=diva2:674007

HARMONOISE_COEFFS = {
    "rolling": {
        "a": np.array(
            [
                # Car
                [
                    69.9,
                    69.9,
                    69.9,
                    74.9,
                    74.9,
                    74.9,
                    79.3,
                    82.5,
                    81.3,
                    80.9,
                    78.9,
                    78.8,
                    80.5,
                    85.7,
                    87.7,
                    89.2,
                    90.6,
                    89.9,
                    89.4,
                    87.6,
                    85.6,
                    82.5,
                    79.6,
                    76.8,
                    74.5,
                    71.9,
                    69.0,
                    69.0,
                    69.0,
                ],
                # CV
                [
                    79.5,
                    79.5,
                    79.5,
                    81.5,
                    82.5,
                    82.5,
                    82.5,
                    87.3,
                    87.7,
                    87.3,
                    90.4,
                    91.2,
                    95.0,
                    97.1,
                    96.8,
                    97.4,
                    95.2,
                    92.6,
                    91.9,
                    89.5,
                    86.1,
                    84.1,
                    82.2,
                    80.3,
                    80.3,
                    80.3,
                    80.3,
                    80.3,
                    80.3,
                ],
            ]
        ),
        "b": np.array(
            [
                # Car
                [
                    33.0,
                    33.0,
                    33.0,
                    30.0,
                    30.0,
                    30.0,
                    41.0,
                    41.2,
                    42.3,
                    41.8,
                    38.6,
                    35.5,
                    31.7,
                    21.5,
                    21.2,
                    23.5,
                    29.1,
                    33.5,
                    34.1,
                    35.1,
                    36.4,
                    37.4,
                    38.9,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                ],
                # CV
                [
                    33.0,
                    33.0,
                    33.0,
                    30.0,
                    30.0,
                    30.0,
                    41.0,
                    41.2,
                    42.3,
                    41.8,
                    38.6,
                    35.5,
                    31.7,
                    21.5,
                    21.2,
                    23.5,
                    29.1,
                    33.5,
                    34.1,
                    35.1,
                    36.4,
                    37.4,
                    38.9,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                    39.7,
                ],
            ]
        ),
    },
    "propulsion": {
        "a": np.array(
            [
                # Car
                [
                    85.8,
                    87.6,
                    97.5,
                    87.5,
                    96.6,
                    97.2,
                    91.5,
                    86.7,
                    86.8,
                    84.9,
                    86.0,
                    86.0,
                    85.9,
                    80.6,
                    80.2,
                    77.8,
                    78.0,
                    81.4,
                    82.3,
                    82.6,
                    81.5,
                    80.7,
                    78.8,
                    77.0,
                    76.0,
                    74.0,
                    72.0,
                    72.0,
                    72.0,
                ],
                # CV
                [
                    97.7,
                    97.3,
                    98.2,
                    103.3,
                    109.5,
                    105.3,
                    100.8,
                    101.2,
                    99.9,
                    102.3,
                    103.5,
                    104.0,
                    101.6,
                    99.2,
                    99.4,
                    95.1,
                    95.8,
                    95.3,
                    93.8,
                    93.9,
                    92.7,
                    91.6,
                    90.9,
                    87.9,
                    87.9,
                    81.8,
                    80.2,
                    80.2,
                    80.2,
                ],
            ]
        ),
        "b": np.array(
            [
                # Car
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                    8.2,
                ],
                # CV
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    8.2,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                    8.5,
                ],
            ]
        ),
    },
}


class SourceSynthesis:
    """
    Source synthesis class.

    This class implements the generation of the source signal emitted by moving vehicles. A moving vehicle (car or CV)
    is represented by two vertically stacked point sources, whose height depends on the vehicle type, as described
    by Harmonoise model [1]. Each of the two sources emits a mixture of rolling noise and propulsion noise.
    The rolling noise is generated via the Harmonoise model, as described in [1]. The propulsion noise can be
    generated either via the Harmonoise model [1] or the Baldan physical model of the 4-stroke engine:

    [2] S. Baldan, H. Lachambre, S. Delle Monache, and P. Boussard, “Physically informed car engine sound synthesis
    for virtual and augmented environments,” in Proceedings of the 2015 IEEE 2nd VR Workshop on Sonic Interactions
    for Virtual Environments (SIVE), Arles, France, 2015, pp. 1-6.

    This class does not implement the Baldan model, but can take as input previously generated audio samples of
    engine sounds generated using the Baldan model. An implementation of the Baldan model is provided
    in https://github.com/DasEtwas/enginesound and can be used to generate engine sounds to be used by this class
    in the synthesis of the source signals.

    Attributes:
    ----------
    sample_rate: int
        sampling frequency used for the generation of source signals
    event_duration: float
        duration in seconds of the events to be generated
    source_model: Literal["hm", "hm+bd"]
        Model used for the generation of the source signal. If "hm" is selected, Harmonoise model is used for
        the generation of both propulsion and rolling noise. If "hm+bd" (default) is selected, Harmonoise model is
        used for the generation of rolling noise, whereas Baldan model is used for the generation of the engine noise
    engine_sound_root: str | None
        Directory where engine sound samples to be used for the generation of source signals are stored. Required
        if source_model == "hm+bd".

    Methods:
    ----------
    generate_harmonoise_source_signals(vehicle_type: str, vehicle_speed: float):
        Returns the synthetic signal of the two point sources that represent the vehicle according to the
        Harmonoise model [1], depending on the type (vehicle_type) and speed (vehicle_speed) of the vehicle.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        event_duration: float = 30.0,
        source_model: Literal["hm", "hm+bd"] = "hm+bd",
        engine_sound_root: str | None = None,
    ) -> None:
        """
        Initialize the source synthesis class.

        Args:
            sample_rate: audio sampling rate
            event_duration: duration of the event in seconds
            source_model: model used for the generation of the source signal.
                If "hm" is selected, Harmonoise model is used for the generation of both propulsion and rolling noise.
                If "hm+bd" (default) is selected, Harmonoise model is used for the generation of rolling noise,
                whereas Baldan model is used for the generation of the engine noise
            engine_sound_root: directory where engine sound samples are read from. Required if source_model == "hm+bd"
        """
        self.sample_rate = sample_rate
        self.event_duration = event_duration

        if source_model not in ["hm", "hm+bd"]:
            raise ValueError("Invalid source signal model. Choose between 'hm' and 'hm+bd'")
        if source_model == "hm+bd":
            if engine_sound_root is None:
                raise ValueError("Specify 'engine_sound_root' in order to use hm+bd model")

            self._engine_sounds = {
                vehicle_type: sorted(Path(engine_sound_root, vehicle_type).glob("*.wav"))
                for vehicle_type in ["car", "cv"]
            }
            for vehicle_type, sounds in self._engine_sounds.items():
                if len(sounds) < 1:
                    raise FileNotFoundError(
                        f"No engine sounds found in {os.path.join(engine_sound_root, vehicle_type, '*.wav')}"
                    )

        self.source_model = source_model

        self.v_ref = 70  # Harmonoise reference speed (70km/h)
        self.p_ref = 0.000020  # Reference sound pressure (Pa)

        self.limits = [
            20,
            12500,
        ]  # Frequency range considered in the generation of source signals
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            self.states, _, _ = pyoctaveband._genfreqs(self.limits, 3, self.sample_rate)

    def _generate_filtered_noise(self, noise_type: str, source_type: str, source_speed: float) -> np.ndarray:
        source_type_idx = 0 if source_type == "car" else 1
        a_coeffs = HARMONOISE_COEFFS[noise_type]["a"][source_type_idx]
        b_coeffs = HARMONOISE_COEFFS[noise_type]["b"][source_type_idx]
        input_signal = self._white_noise()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            _, _, subband_noise = pyoctaveband.octavefilter(
                input_signal,
                self.sample_rate,
                fraction=3,
                order=4,
                limits=self.limits,
                show=0,
                sigbands=1,
            )

        subband_noise_res = np.zeros((len(subband_noise), len(input_signal)))
        for i in range(len(subband_noise)):
            if len(subband_noise[i]) < len(input_signal):
                res = np.zeros(len(input_signal))
                res[: len(subband_noise[i])] = subband_noise[i]
                subband_noise_res[i] = res
            elif len(subband_noise[i]) > len(input_signal):
                subband_noise_res[i] = subband_noise[i][: len(input_signal)]
            else:
                subband_noise_res[i] = subband_noise[i]

        # Synthesis in the frequency domain
        L_coeff = np.zeros(len(self.states))
        for i in range(len(self.states)):
            L_coeff[i] = a_coeffs[i] + b_coeffs[i] * np.log(source_speed / self.v_ref)

        # Synthesis in the time domain
        filtered_noise = np.zeros_like(input_signal)
        for i in range(len(self.states)):
            filtered_noise += self.p_ref * (10 ** (L_coeff[i] / 20)) * subband_noise_res[i]

        return filtered_noise

    def _white_noise(self) -> np.ndarray:
        noise = np.random.normal(0, 1, self.sample_rate * self.event_duration)
        return noise / max(abs(noise))

    def generate_harmonoise_source_signals(
        self, vehicle_type: str, vehicle_speed: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate Harmonoise source signals.

        Args:
            vehicle_type: car or cv
            vehicle_speed: km/h

        Returns: (sig_higher, sig_lower)
            sig_higher: (int(self.sample_rate * self.event_duration),) flot ndarray, higher source signal
            sig_lower: (int(self.sample_rate * self.event_duration),) flot ndarray, lower source signal
        """
        # Generate rolling sound
        rolling_sound = self._generate_filtered_noise("rolling", vehicle_type, vehicle_speed)

        # Generate engine sound
        propulsion_noise = self._generate_filtered_noise("propulsion", vehicle_type, vehicle_speed)

        if self.source_model == "hm+bd":
            engine_file = random.choice(self._engine_sounds[vehicle_type])  # noqa: S311
            engine_signal, _ = librosa.load(
                engine_file,
                sr=self.sample_rate,
                mono=True,
            )

            if len(engine_signal) / self.sample_rate < self.event_duration:
                engine_signal = np.tile(engine_signal, 2)

            engine_signal = engine_signal[: int(self.sample_rate * self.event_duration)]
            propulsion_noise = engine_signal * np.sqrt(np.sum(propulsion_noise**2) / np.sum(engine_signal**2))

        # Generate source signals and return
        sig_higher = 2 / np.sqrt(5) * propulsion_noise + 1 / np.sqrt(5) * rolling_sound
        sig_lower = 1 / np.sqrt(5) * propulsion_noise + 2 / np.sqrt(5) * rolling_sound

        return sig_higher, sig_lower
