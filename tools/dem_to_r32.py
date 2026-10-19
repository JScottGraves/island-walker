"""Convert USGS 3DEP 1/3 arc-second GeoTIFF tiles to islandgen's float32 heightfield.

    python3 tools/dem_to_r32.py OUT.r32 SIZE CELL_METRES LAT0 LON0 TILE.tif [TILE.tif ...]
        [--bathymetry SEA.tif [SEA.tif ...]]

Samples a SIZE x SIZE grid of square CELL_METRES cells centred on (LAT0, LON0) with a
local equirectangular projection (x east, y north, row 0 at the south edge, as Grid
expects), bilinear from the geographic tiles. Without bathymetry, ocean and nodata become
-1 m (sea level 0).

--bathymetry takes topobathy tiles (NOAA NCEI CUDEM Hawaii, metres on LMSL). Land stays
the land tiles' wherever they are above 0; everywhere else takes the sea tiles' depth,
kept at least 0.2 m under the sea so the two datums can't make new land along the shore, and
land at least 5 cm above it, so the bundle's height steps (~7 cm) can't flip either.
Sea cells no tile covers are filled smoothly from their neighbours. Tiles listed earlier
win where tiles overlap, so list the finest first.
Needs numpy and tifffile (+ imagecodecs for the LZW tiles).
"""
import math
import sys
import warnings

import numpy as np
import tifffile


def load(path):
    with tifffile.TiffFile(path) as t:
        g = t.geotiff_metadata
        scale = g["ModelPixelScale"][0]
        lon0, lat0 = g["ModelTiepoint"][3], g["ModelTiepoint"][4]
        nodata = t.pages[0].tags.get("GDAL_NODATA")
        nodata = float(nodata.value.strip("\x00 ")) if nodata else None
        return t.pages[0].asarray(), lon0, lat0, scale, nodata


def sample(paths, lats, lons):
    """Bilinear samples of the tiles at the grid's cells; NaN where no tile covers a cell."""
    result = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    for path in paths:
        data, lon0, lat0, scale, nodata = load(path)
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
        def ok(q):
            return (q > -11000) & (q < 9000) & ((q != nodata) if nodata is not None else True)
        valid = ok(a) & ok(b) & ok(c) & ok(d)
        v = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
        block = result[np.ix_(yi, xi)]
        fresh = valid & np.isnan(block)
        block[fresh] = v[fresh]
        result[np.ix_(yi, xi)] = block
        del data
    return result


def fill(grid):
    """Fills NaN cells from their neighbours: coarse-to-fine, so wide gaps get smooth values."""
    holes = np.isnan(grid)
    if not holes.any() or holes.all():
        return grid
    n = grid.shape[0]
    if n <= 2:
        grid[holes] = np.nanmean(grid)
        return grid
    # Mean of each 2x2 block's known cells, filled recursively, then upsampled into the holes.
    h = n // 2 * 2
    blocks = grid[:h, :h].reshape(h // 2, 2, h // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
        coarse = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)
    coarse = fill(coarse)
    up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    if up.shape != grid.shape:
        up = np.pad(up, ((0, n - h), (0, n - h)), mode="edge")
    grid[holes] = up[holes]
    # Soften the block edges the upsampling left in the holes.
    for _ in range(3):
        p = np.pad(grid, 1, mode="edge")
        blur = sum(p[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
        grid[holes] = blur[holes]
    return grid


def main():
    args = sys.argv[6:]
    land_paths, sea_paths = (args[:args.index("--bathymetry")], args[args.index("--bathymetry") + 1:]) \
        if "--bathymetry" in args else (args, [])
    out, size, cell, lat_c, lon_c = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    m_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat_c))
    m_lon = 111412.84 * math.cos(math.radians(lat_c))
    offsets = (np.arange(size) + 0.5 - size / 2) * cell
    lats = lat_c + offsets / m_lat           # row index → latitude (south to north)
    lons = lon_c + offsets / m_lon
    result = sample(land_paths, lats, lons)
    sea = np.isnan(result) | (result <= 0)
    if sea_paths:
        depth = sample(sea_paths, lats, lons)
        depth[~sea] = np.nan
        covered = ~np.isnan(depth)
        print(f"bathymetry covers {covered.sum() / max(sea.sum(), 1) * 100:.1f}% of the sea cells")
        depth = np.minimum(depth, -0.2)
        result[sea] = depth[sea]
        result = fill(result)
        result[sea] = np.minimum(result[sea], -0.2)
        # Land at least 5 cm up: a bundle's 16-bit heights step ~7 cm over this range, and land a
        # centimetre or two above the sea would otherwise round under it into a film of water.
        result[~sea] = np.maximum(result[~sea], 0.05)
    else:
        result[sea] = -1.0
    result.astype("<f4").tofile(out)
    land = (result > 0).sum()
    print(f"{out}: {size}x{size} at {cell} m, land {land * cell * cell / 1e6:.1f} km², "
          f"max {result.max():.1f} m, min {result.min():.1f} m")


if __name__ == "__main__":
    main()
