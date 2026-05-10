"""Sensor and setting column names for the NASA turbofan dataset."""

SETTINGS: tuple[str, ...] = ("setting1", "setting2", "setting3")
SENSORS: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 22))
ALL_NUMERIC: tuple[str, ...] = SETTINGS + SENSORS

READING_COLUMNS: tuple[str, ...] = ("id", "cycle", *ALL_NUMERIC)
"""26 columns of the raw train_FDxxx.txt files (space-delimited)."""

TRUTH_COLUMNS: tuple[str, ...] = ("id", "rul_at_maxcycle")
