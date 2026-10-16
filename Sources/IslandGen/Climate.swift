import Foundation
import IslandCore

/// Annual rainfall from prevailing wind and terrain.
///
/// This is a simple upslope model: air pushed uphill by the trade winds rains out on
/// windward slopes, and air that has already crossed high ground is drier. It is a
/// stand-in for the FFT-based linear orographic model (Smith & Barstad 2004) in the plan.
public enum Climate {
    public struct Params: Sendable {
        /// Direction the wind blows towards, in radians. Default: from the north-east.
        public var windDirection: Double = 5 * Double.pi / 4
        /// Rainfall over open ocean, in metres per year.
        public var baseRain: Double = 0.6
        /// Extra rain per unit of windward slope.
        public var upslopeGain: Double = 40
        /// Moisture lost per (metre of height × kilometre) of land already crossed.
        /// 6e-5 leaves about 30% of the rain behind a 1,000 m, 20 km wide range.
        public var shadowStrength: Double = 6e-5
        /// Terrain smoothing radius before measuring slopes, in metres.
        public var smoothing: Double = 2000
        public init() {}
    }

    public static func precipitation(height: Grid, seaLevel: Float, params: Params = Params()) -> Grid {
        let w = height.width, n = height.height
        // Air flows over the sea surface, not the sea floor: clamp before smoothing, or the
        // steep submarine flanks pull the rain maximum out over the ocean.
        var surface = height
        for i in 0..<surface.count { surface.values[i] = Swift.max(surface.values[i], seaLevel) }
        let radius = Swift.max(1, Int(params.smoothing / height.cellSize))
        let smooth = boxBlur(surface, radius: radius)
        let wx = cos(params.windDirection), wy = sin(params.windDirection)

        // Rain shadow: on a coarse grid (~1 km cells), march each cell's ray back upwind to
        // the domain edge, summing (land height × distance) already crossed. Marching whole
        // rays on a coarse grid is cheap and free of the banding a cell-to-cell sweep shows.
        let coarseSide = Swift.max(16, Int((Double(w) * height.cellSize / 1000).rounded()))
        let coarseCell = Double(w) * height.cellSize / Double(coarseSide)
        let toFine = coarseCell / height.cellSize
        var crossed = Grid(width: coarseSide, height: coarseSide, cellSize: coarseCell)
        let stepKm = 0.5
        let stepFine = stepKm * 1000 / height.cellSize
        for cy in 0..<coarseSide {
            for cx in 0..<coarseSide {
                var fx = (Double(cx) + 0.5) * toFine - 0.5, fy = (Double(cy) + 0.5) * toFine - 0.5
                var sum = 0.0
                while true {
                    fx -= wx * stepFine
                    fy -= wy * stepFine
                    if fx < 0 || fy < 0 || fx > Double(w - 1) || fy > Double(n - 1) { break }
                    sum += Double(Swift.max(0, smooth.sample(x: fx, y: fy) - seaLevel)) * stepKm
                }
                crossed[cx, cy] = Float(sum)
            }
        }

        var rain = Grid(width: w, height: n, cellSize: height.cellSize)
        for y in 0..<n {
            for x in 0..<w {
                let gx = Double(smooth.sample(x: Double(x + 1), y: Double(y)) - smooth.sample(x: Double(x - 1), y: Double(y))) / (2 * height.cellSize)
                let gy = Double(smooth.sample(x: Double(x), y: Double(y + 1)) - smooth.sample(x: Double(x), y: Double(y - 1))) / (2 * height.cellSize)
                let upslope = Swift.max(0, gx * wx + gy * wy)
                let c = Double(crossed.sample(x: (Double(x) + 0.5) / toFine - 0.5, y: (Double(y) + 0.5) / toFine - 0.5))
                let shadow = exp(-params.shadowStrength * c)
                rain[x, y] = Float(params.baseRain * (1 + params.upslopeGain * upslope) * shadow)
            }
        }
        return rain
    }

    /// Separable running-sum box blur with clamped edges.
    static func boxBlur(_ g: Grid, radius r: Int) -> Grid {
        var tmp = g
        var out = g
        let w = g.width, h = g.height
        let norm = Float(2 * r + 1)
        for y in 0..<h {
            var sum: Float = 0
            for k in -r...r { sum += g[Swift.min(Swift.max(k, 0), w - 1), y] }
            for x in 0..<w {
                tmp[x, y] = sum / norm
                sum += g[Swift.min(x + r + 1, w - 1), y] - g[Swift.max(x - r, 0), y]
            }
        }
        for x in 0..<w {
            var sum: Float = 0
            for k in -r...r { sum += tmp[x, Swift.min(Swift.max(k, 0), h - 1)] }
            for y in 0..<h {
                out[x, y] = sum / norm
                sum += tmp[x, Swift.min(y + r + 1, h - 1)] - tmp[x, Swift.max(y - r, 0)]
            }
        }
        return out
    }
}
