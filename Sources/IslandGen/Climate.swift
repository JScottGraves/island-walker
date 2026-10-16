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
        let radius = Swift.max(1, Int(params.smoothing / height.cellSize))
        let smooth = boxBlur(height, radius: radius)
        let wx = cos(params.windDirection), wy = sin(params.windDirection)
        var rain = Grid(width: w, height: n, cellSize: height.cellSize)

        // Moisture budget: "crossed" accumulates (height above sea × distance) along the
        // wind. Cells are visited in upwind-to-downwind order so each one can read the
        // already-computed value one cell upwind (semi-Lagrangian, O(n log n)).
        var crossed = Grid(width: w, height: n, cellSize: height.cellSize)
        let order = (0..<(w * n)).sorted { a, b in
            let pa = Double(a % w) * wx + Double(a / w) * wy
            let pb = Double(b % w) * wx + Double(b / w) * wy
            return pa < pb || (pa == pb && a < b)
        }
        let km = height.cellSize / 1000
        for i in order {
            let x = i % w, y = i / w
            let ux = Double(x) - wx, uy = Double(y) - wy
            var upwind: Float = 0
            if ux >= 0, uy >= 0, ux <= Double(w - 1), uy <= Double(n - 1) {
                upwind = crossed.sample(x: ux, y: uy)
            }
            let land = Swift.max(0, smooth.values[i] - seaLevel)
            crossed.values[i] = upwind + Float(Double(land) * km)
        }

        for y in 0..<n {
            for x in 0..<w {
                let i = height.index(x, y)
                let gx = Double(smooth.sample(x: Double(x + 1), y: Double(y)) - smooth.sample(x: Double(x - 1), y: Double(y))) / (2 * height.cellSize)
                let gy = Double(smooth.sample(x: Double(x), y: Double(y + 1)) - smooth.sample(x: Double(x), y: Double(y - 1))) / (2 * height.cellSize)
                let upslope = Swift.max(0, gx * wx + gy * wy)
                let shadow = exp(-params.shadowStrength * Double(crossed.values[i]))
                rain.values[i] = Float(params.baseRain * (1 + params.upslopeGain * upslope) * shadow)
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
