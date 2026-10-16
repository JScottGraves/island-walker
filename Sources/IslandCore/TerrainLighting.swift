import Foundation

/// Per-cell sky visibility and sun visibility from the terrain itself, baked once per
/// heightfield and sun direction: the cheap core of terrain GI (research/
/// global-illumination.md §4). Valley floors see a strip of sky instead of the whole
/// dome, and ridges cast shadows, which is where most of the depth in a ground view
/// comes from. Brute-force horizon marching; the sweep algorithm and 16-sector horizon
/// maps from the plan replace it when the sun starts to move.
public struct TerrainLighting: Sendable {
    /// Fraction of open sky seen from each cell, cosine-weighted (1 on open flat ground).
    public var sky: Grid
    /// 1 where the sun is visible, 0 in terrain shadow, soft in between.
    public var sun: Grid

    public static let sectors = 16
    /// How far to look for the horizon: sky visibility only needs nearby terrain, but
    /// long ridge shadows at a low sun reach much further.
    public static let skyRange = 6000.0
    public static let sunRange = 25_000.0

    public static func bake(height h: Grid, seaLevel: Float,
                            sunDirection s: (x: Double, y: Double, z: Double) = Look.sunDirection) -> TerrainLighting {
        var sky = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        var sun = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        let cell = h.cellSize
        let sea = Double(seaLevel)
        let sunFlat = (s.x * s.x + s.y * s.y).squareRoot()
        let sunElevation = atan2(s.z, sunFlat)
        let sunDir = sunFlat > 0 ? (s.x / sunFlat, s.y / sunFlat) : (0.0, 0.0)
        let directions = (0..<sectors).map { k -> (Double, Double) in
            let a = (Double(k) + 0.5) * 2 * Double.pi / Double(sectors)
            return (cos(a), sin(a))
        }

        @inline(__always) func ground(_ x: Double, _ y: Double) -> Double {
            Swift.max(Double(h.sample(x: x, y: y)), sea)
        }
        /// Steepest rise (tan of the horizon angle) along a direction, in grid units.
        func horizon(_ gx: Double, _ gy: Double, _ z0: Double, _ dir: (Double, Double), _ range: Double) -> Double {
            var best = 0.0
            var d = 1.0
            let maxSteps = range / cell
            while d <= maxSteps {
                let x = gx + dir.0 * d, y = gy + dir.1 * d
                if x < 0 || y < 0 || x > Double(h.width - 1) || y > Double(h.height - 1) { break }
                best = Swift.max(best, (ground(x, y) - z0) / (d * cell))
                d = Swift.max(d + 1, d * 1.15)
            }
            return best
        }

        sky.values.withUnsafeMutableBufferPointer { skyBuf in
            sun.values.withUnsafeMutableBufferPointer { sunBuf in
                let skyOut = SharedBuffer(skyBuf), sunOut = SharedBuffer(sunBuf)
                Parallel.forRanges(h.height) { rows in
                    for y in rows {
                        for x in 0..<h.width {
                            let i = y * h.width + x
                            let gx = Double(x), gy = Double(y)
                            // Lift the eye a little so a cell doesn't occlude itself on slopes.
                            let z0 = ground(gx, gy) + 2
                            var visible = 0.0
                            for dir in directions {
                                let t = horizon(gx, gy, z0, dir, skyRange)
                                visible += 1 / (1 + t * t) // cos² of the horizon angle
                            }
                            skyOut[i] = Float(visible / Double(sectors))
                            let t = horizon(gx, gy, z0, sunDir, sunRange)
                            // Soft edge over about a degree: the sun's disc plus some blur.
                            let margin = sunElevation - atan(t)
                            let v = Swift.min(1, Swift.max(0, (margin + 0.01) / 0.02))
                            sunOut[i] = Float(v * v * (3 - 2 * v))
                        }
                    }
                }
            }
        }
        return TerrainLighting(sky: sky, sun: sun)
    }
}
