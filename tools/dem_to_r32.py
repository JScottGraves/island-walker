"""Convert USGS 3DEP 1/3 arc-second GeoTIFF tiles to islandgen's float32 heightfield.

    python3 tools/dem_to_r32.py OUT.r32 SIZE CELL_METRES LAT0 LON0 TILE.tif [TILE.tif ...]

Samples a SIZE x SIZE grid of square CELL_METRES cells centred on (LAT0, LON0) with a
local equirectangular projection (x east, y north, row 0 at the south edge, as Grid
expects), bilinear from the geographic tiles. Ocean and nodata become -1 m (sea level 0).
Needs numpy and tifffile (+ imagecodecs for the LZW tiles).
"""
import math
import sys

import numpy as np
import tifffile


def load(path):
    with tifffile.TiffFile(path) as t:
        g = t.geotiff_metadata
        scale = g["ModelPixelScale"][0]
        lon0, lat0 = g["ModelTiepoint"][3], g["ModelTiepoint"][4]
        return t.pages[0].asarray(), lon0, lat0, scale


def main():
    out, size, cell, lat_c, lon_c = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    m_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat_c))
    m_lon = 111412.84 * math.cos(math.radians(lat_c))
    offsets = (np.arange(size) + 0.5 - size / 2) * cell
    lats = lat_c + offsets / m_lat           # row index → latitude (south to north)
    lons = lon_c + offsets / m_lon
    result = np.full((size, size), -1.0, dtype=np.float32)
    for path in sys.argv[6:]:
        data, lon0, lat0, scale = load(path)
        rows, cols = data.shape
        # Pixel centres: tiepoint is the corner of pixel (0, 0); row 0 is the north edge.
        fy = (lat0 - lats) / scale - 0.5
        fx = (lons - lon0) / scale - 0.5
        ry = (fy >= 0) & (fy < rows - 1)
        cx = (fx >= 0) & (fx < cols - 1)
        if not ry.any() or not cx.any():
            continue
        yi, xi = np.nonzero(ry)[0], np.nonzero(cx)[0]
        y0 = np.floor(fy[yi]).astype(int); ty = (fy[yi] - y0)[:, None]
        x0 = np.floor(fx[xi]).astype(int); tx = (fx[xi] - x0)[None, :]
        a = data[np.ix_(y0, x0)]; b = data[np.ix_(y0, x0 + 1)]
        c = data[np.ix_(y0 + 1, x0)]; d = data[np.ix_(y0 + 1, x0 + 1)]
        valid = (a > -1000) & (b > -1000) & (c > -1000) & (d > -1000)
        v = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
        block = result[np.ix_(yi, xi)]
        block[valid] = v[valid]
        result[np.ix_(yi, xi)] = block
        del data
    result[result <= 0] = -1.0
    result.astype("<f4").tofile(out)
    land = (result > 0).sum()
    print(f"{out}: {size}x{size} at {cell} m, land {land * cell * cell / 1e6:.1f} km², max {result.max():.1f} m")


if __name__ == "__main__":
    main()
