"""Session-owned, causal replay of the real detector; no truth inputs."""
from pathlib import Path

import numpy as np
import pandas as pd

from ai.detect import LeakDetector
from ai.train import load_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/sensor_study"
SCENARIOS = {
    "Normal": DATA / "normal/normal.csv",
    "Small leak": DATA / "leak_pipe12/leak.csv",
    "Rupture": DATA / "rupture_pipe12/rupture.csv",
    "Demand spike": DATA / "demand_spike/demand_spike.csv",
}


class Replay:
    def __init__(self, csv_path, model_path, inp_path):
        self.frame = load_data(csv_path)
        self.detector = LeakDetector(model_path, inp_path)
        columns = ["timestamp", *self.detector.sensor_cols]
        self.readings = self.frame[columns].copy()
        if self.frame.empty or not np.isfinite(self.readings.to_numpy(dtype=float)).all():
            raise ValueError("Readings must be nonempty and finite.")
        if not self.readings.timestamp.is_unique:
            raise ValueError("Timestamps must be unique.")
        self.results = []

    def at(self, index):
        if not 0 <= index < len(self.frame):
            raise IndexError(index)
        # Backward seeks read stored results. Forward seeks process every skipped
        # row exactly once, preserving the detector's consecutive anomaly streak.
        history = 2 * self.detector.window - 1
        while len(self.results) <= index:
            i = len(self.results)
            self.results.append(self.detector.detect(
                self.readings.iloc[max(0, i - history + 1):i + 1]
            ))
        return self.results[index]
