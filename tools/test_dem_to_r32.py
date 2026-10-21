"""Checks for dem_to_r32's shoreline smoothing: python3 tools/test_dem_to_r32.py (numpy)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dem_to_r32 import smooth_shore  # noqa: E402


def staircase(n=80, angle=12.0):
    """A beach at +0.05 m and sea from -0.7 m, split along a slanting line, cell by cell (as
    land and bathymetry tiles meet), and the line's offset per column."""
    t = math.tan(math.radians(angle))
    g = np.zeros((n, n), np.float32)
    for y in range(n):
        for x in range(n):
            g[y, x] = 0.05 if x + 0.5 < 30 + (y + 0.5) * t else -0.7 - 0.05 * (x - 30 - y * t)
    return g, t


def contour_x(g, y):
    """Where row y crosses zero, linearly between cells (cell centres at x + 0.5)."""
    row = g[y]
    for x in range(len(row) - 1):
        if row[x] > 0 >= row[x + 1]:
            return x + 0.5 + row[x] / (row[x] - row[x + 1])
    return None


def test_waterline_follows_the_coast_not_the_cells():
    g, t = staircase()
    s = smooth_shore(g)
    # The raw staircase wanders ~half a cell either side of the true line; smoothed, it doesn't.
    raw = [contour_x(g, y) - (30 + (y + 0.5) * t) for y in range(10, 70)]
    smooth = [contour_x(s, y) - (30 + (y + 0.5) * t) for y in range(10, 70)]
    wander = lambda v: max(v) - min(v)
    assert wander(raw) > 0.8, wander(raw)
    assert wander(smooth) < 0.35, wander(smooth)


def test_waterline_stays_put_on_average():
    # Smoothing must not pull the beach under the sea (the step is lopsided: +0.05 against -0.7).
    g, t = staircase()
    s = smooth_shore(g)
    shift = np.mean([contour_x(s, y) - (30 + (y + 0.5) * t) for y in range(10, 70)])
    assert abs(shift) < 0.25, shift
    assert abs(int((s > 0).sum()) - int((g > 0).sum())) < 0.02 * (g > 0).sum()


def test_only_the_shoreline_band_changes():
    g, _ = staircase()
    s = smooth_shore(g)
    far_land = g[:, :20], s[:, :20]
    far_sea = g[:, 60:], s[:, 60:]
    assert np.array_equal(*far_land)
    assert np.array_equal(*far_sea)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
