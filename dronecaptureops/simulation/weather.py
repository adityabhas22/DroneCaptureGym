"""Weather helpers."""

from __future__ import annotations

import random

from dronecaptureops.core.models import WeatherState


def sample_weather(rng: random.Random) -> WeatherState:
    """Sample deterministic weather for a scenario seed."""

    return WeatherState(wind_mps=round(rng.uniform(1.0, 6.5), 2), visibility=round(rng.uniform(0.82, 1.0), 2))
