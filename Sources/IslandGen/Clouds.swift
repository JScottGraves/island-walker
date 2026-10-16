import Foundation
import IslandCore

/// Where trade-wind clouds sit over the island: a terrain-locked coverage map baked next to
/// the rain map, for the renderer's cloud layer and cloud shadows (research/clouds.md §3).
///
/// Air arriving from the open ocean is followed downwind over the smoothed terrain. Rising air
/// condenses part of its vapour into cloud water and descending air evaporates it again;
/// cloud water also rains out or mixes away over a trade-wind travel time, faster once it is
/// back over the sea. What comes out is the standing orographic cloud: thick over windward
/// slopes and crests, thinning a short way into the lee. Drifting ocean cumulus and the
/// daily cycle belong to the runtime, which modulates this map rather than replacing it.
public enum Clouds {
    public struct Params: Sendable {
        /// Direction the wind blows towards, in radians (same convention as `Climate`).
        public var windDirection: Double = 5 * Double.pi / 4
        /// Trade-wind speed in m/s: about 7.5 for normal trades, 11 for strong trades.
        public var windSpeed: Double = 7.5
        /// Metres of ascent that would condense all of the air's vapour.
        public var condensationHeight: Double = 900
        /// Metres of descent that evaporate all cloud water at 7.5 m/s. Scales with wind
        /// speed: faster air carries cloud further into the lee before it dries out.
        public var evaporationDescent: Double = 60
        /// Seconds for cloud water to rain out or mix away over land.
        public var lifetime: Double = 900
        /// How many times faster cloud decays once it is back over the sea.
        public var seaDecay: Double = 4
        /// Distance over open sea for depleted air to recover its moisture, in metres.
        public var oceanRecovery: Double = 20_000
        /// Cloud water (as a fraction of the incoming vapour) that gives 63% coverage.
        public var coverageScale: Double = 0.08
        /// How far upwind each cell's air is followed, in metres.
        public var reach: Double = 40_000
        /// March step along the wind, in metres.
        public var step: Double = 250
        /// Output cell size in metres. Clouds need no finer map than this.
        public var cellSize: Double = 500
        /// Terrain smoothing radius in metres: the air does not follow every gully.
        public var smoothing: Double = 750
        public init() {}

        public static var strongTrades: Params {
            var p = Params()
            p.windSpeed = 11
            return p
        }
    }

    /// Orographic cloud coverage in 0...1 on a coarse grid covering the same domain.
    public static func coverage(height: Grid, seaLevel: Float, params p: Params = Params()) -> Grid {
        var surface = height
        for i in 0..<surface.count { surface.values[i] = Swift.max(surface.values[i], seaLevel) }
        let r = Swift.max(1, Int(p.smoothing / height.cellSize))
        let smooth = Climate.boxBlur(Climate.boxBlur(Climate.boxBlur(surface, radius: r), radius: r), radius: r)

        let domain = Double(height.width) * height.cellSize
        let side = Swift.max(8, Int((domain / p.cellSize).rounded()))
        let cell = domain / Double(side)
        var out = Grid(width: side, height: side, cellSize: cell)

        let wx = cos(p.windDirection), wy = sin(p.windDirection)
        let steps = Swift.max(1, Int(p.reach / p.step))
        let stepCells = p.step / height.cellSize
        let decayPerStep = p.step / (p.windSpeed * p.lifetime)
        let descent = p.evaporationDescent * p.windSpeed / 7.5
        let recoveryPerStep = p.step / p.oceanRecovery
        let sea = Double(seaLevel) + 1
        let toFine = cell / height.cellSize

        out.values.withUnsafeMutableBufferPointer { buf in
            let result = SharedBuffer(buf)
            Parallel.forRanges(side) { rows in
                for cy in rows {
                    for cx in 0..<side {
                        let fx = (Double(cx) + 0.5) * toFine - 0.5, fy = (Double(cy) + 0.5) * toFine - 0.5
                        func surfaceAt(_ k: Int) -> Double {
                            let d = stepCells * Double(k)
                            return Double(smooth.sample(x: fx - wx * d, y: fy - wy * d))
                        }
                        // Follow the air from `reach` upwind to this cell. q is vapour as a
                        // fraction of what open-ocean air carries; c is cloud water.
                        var q = 1.0, c = 0.0
                        var previous = surfaceAt(steps)
                        for k in stride(from: steps - 1, through: 0, by: -1) {
                            let z = surfaceAt(k)
                            let dz = z - previous
                            previous = z
                            let condensed = q * Swift.min(1, Swift.max(dz, 0) / p.condensationHeight)
                            let evaporated = c * Swift.min(1, Swift.max(-dz, 0) / descent)
                            let overSea = z <= sea
                            let decayed = c * Swift.min(1, decayPerStep * (overSea ? p.seaDecay : 1))
                            q += evaporated - condensed
                            c += condensed - evaporated - decayed
                            if overSea { q += (1 - q) * recoveryPerStep }
                        }
                        result[cy * side + cx] = Float(1 - exp(-Swift.max(c, 0) / p.coverageScale))
                    }
                }
            }
        }
        return out
    }

    /// Mean coverage over land, and over the land on each side of the island's centre
    /// relative to the wind. Windward should be clearly the cloudier side.
    public static func stats(coverage: Grid, height: Grid, seaLevel: Float, windDirection: Double) -> (land: Double, windward: Double, leeward: Double) {
        let wx = cos(windDirection), wy = sin(windDirection)
        let scale = coverage.cellSize / height.cellSize
        var cx = 0.0, cy = 0.0, n = 0.0
        for y in 0..<height.height {
            for x in 0..<height.width where height[x, y] > seaLevel {
                cx += Double(x); cy += Double(y); n += 1
            }
        }
        guard n > 0 else { return (0, 0, 0) }
        cx /= n; cy /= n
        var sum = (all: 0.0, windward: 0.0, leeward: 0.0), count = (windward: 0.0, leeward: 0.0)
        for y in 0..<height.height {
            for x in 0..<height.width where height[x, y] > seaLevel {
                let v = Double(coverage.sample(x: (Double(x) + 0.5) / scale - 0.5, y: (Double(y) + 0.5) / scale - 0.5))
                sum.all += v
                // Upwind of the centre means against the wind direction.
                if (Double(x) - cx) * wx + (Double(y) - cy) * wy < 0 {
                    sum.windward += v; count.windward += 1
                } else {
                    sum.leeward += v; count.leeward += 1
                }
            }
        }
        return (sum.all / n, sum.windward / Swift.max(count.windward, 1), sum.leeward / Swift.max(count.leeward, 1))
    }
}

extension Preview {
    /// Cloud coverage drawn as white over an image of the same domain, such as the hillshade.
    public static func clouds(_ coverage: Grid, over base: Image) -> Image {
        var out = base
        let sx = Double(coverage.width) / Double(base.width), sy = Double(coverage.height) / Double(base.height)
        for row in 0..<base.height {
            let y = base.height - 1 - row // images have north at the top
            for x in 0..<base.width {
                let a = 0.85 * Double(coverage.sample(x: (Double(x) + 0.5) * sx - 0.5, y: (Double(y) + 0.5) * sy - 0.5))
                let o = (row * base.width + x) * 3
                for k in 0..<3 {
                    let v = Double(base.rgb[o + k]) * (1 - a) + 255 * a
                    out.rgb[o + k] = UInt8(Swift.max(0, Swift.min(255, v.rounded())))
                }
            }
        }
        return out
    }
}
