"""
LoadProfileSampler — loads the PowerGym 24-hr CSV load-shape files and
provides episode-level multipliers that replicate PowerGym's daily profile.

Usage
-----
sampler = LoadProfileSampler('13Bus')        # loads all CSV files for that system
muls    = sampler.sample(rng)                # array of 24 hourly multipliers
mul_t   = muls[step]                         # scalar for step t in [0,23]

PowerGym CSV format: one float per line, 8760 lines (one year of hourly values).
Multiple CSV files are concatenated so every distinct 24-hr window becomes a
potential episode.
"""

from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


class LoadProfileSampler:
    """
    Reads the PowerGym load-shape CSV files for a given system and samples
    random 24-hour load-multiplier windows for each episode.

    Parameters
    ----------
    system : '13Bus' | '34Bus' | '123Bus'
    """

    def __init__(self, system: str):
        folder = os.path.join(_HERE, "powergym_systems", system, "loadshape")
        assert os.path.isdir(folder), f"Loadshape folder not found: {folder}"

        # Read all *.CSV / *.csv files (excludes sub-folders)
        multipliers: list[float] = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith('.csv') and os.path.isfile(os.path.join(folder, fname)):
                path = os.path.join(folder, fname)
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                multipliers.append(float(line))
                            except ValueError:
                                pass

        assert len(multipliers) >= 24, f"Not enough load-shape data for {system}"

        # Trim to a multiple of 24 so every window is a complete day
        n = (len(multipliers) // 24) * 24
        self._data = np.array(multipliers[:n], dtype=np.float32)
        self._n_episodes = n // 24   # number of distinct 24-hr windows

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Return a random 24-hour multiplier array (shape [24]) drawn from the
        pool of available daily profiles.
        """
        idx = int(rng.integers(0, self._n_episodes))
        return self._data[idx * 24 : idx * 24 + 24].copy()
