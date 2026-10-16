import Foundation

/// Per-cell sky visibility and sun visibility from the terrain itself, baked once per
/// heightfield and sun direction: the cheap core of terrain GI (research/
/// global-illumination.md §4). Valley floors see a strip of sky instead of the whole
/// dome, and ridges cast shadows, which is where most of the depth in a ground view
/// comes from. Brute-force horizon marching; the sweep algorithm replaces it if bake time
/// starts to matter.
public struct TerrainLighting: Sendable {
    /// Fraction of open sky seen from each cell, cosine-weighted (1 on open flat ground).
    public var sky: Grid
    /// 1 where the sun is visible, 0 in terrain shadow, soft in between.
    public var sun: Grid
    /// Optional horizon map: for each cell, `sectors` bytes, the horizon's elevation angle
    /// in each azimuth sector out to `sunRange` (0 = flat, 255 = straight up). Sector k is
    /// centred on the direction (k + 0.5) · 360° / sectors, counter-clockwise from east.
    /// Lets a renderer without ray tracing shadow terrain for any sun position.
    public var horizon: [UInt8]

    public static let sectors = 16
    /// How far to look for the horizon: sky visibility only needs nearby terrain, but
    /// long ridge shadows at a low sun reach much further.
    public static let skyRange = 6000.0
    public static let sunRange = 25_000.0

    public static func bake(height h: Grid, seaLevel: Float,
                            sunDirection s: (x: Double, y: Double, z: Double) = Look.sunDirection,
                            horizons: Bool = false) -> TerrainLighting {
        var sky = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        var sun = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        var horizon = [UInt8](repeating: 0, count: horizons ? h.count * sectors : 0)
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
        /// Steepest rise (tan of the horizon angle) along a direction, in grid units: within
        /// `near` metres, and within `range` metres (the same samples, so the near value
        /// is what a march stopping at `near` would give).
        func march(_ gx: Double, _ gy: Double, _ z0: Double, _ dir: (Double, Double),
                     near: Double, range: Double) -> (near: Double, far: Double) {
            var best = 0.0, bestNear = 0.0
            var d = 1.0
            let nearSteps = near / cell, maxSteps = range / cell
            while d <= maxSteps {
                let x = gx + dir.0 * d, y = gy + dir.1 * d
                if x < 0 || y < 0 || x > Double(h.width - 1) || y > Double(h.height - 1) { break }
                best = Swift.max(best, (ground(x, y) - z0) / (d * cell))
                if d <= nearSteps { bestNear = best }
                d = Swift.max(d + 1, d * 1.15)
            }
            return (bestNear, best)
        }

        sky.values.withUnsafeMutableBufferPointer { skyBuf in
          sun.values.withUnsafeMutableBufferPointer { sunBuf in
            horizon.withUnsafeMutableBufferPointer { horizonBuf in
                let skyOut = SharedBuffer(skyBuf), sunOut = SharedBuffer(sunBuf)
                // Empty unless horizons were asked for, and an empty buffer has no base address.
                let horizonOut = horizons ? SharedBuffer(horizonBuf) : nil
                Parallel.forRanges(h.height) { rows in
                    for y in rows {
                        for x in 0..<h.width {
                            let i = y * h.width + x
                            let gx = Double(x), gy = Double(y)
                            // Lift the eye a little so a cell doesn't occlude itself on slopes.
                            let z0 = ground(gx, gy) + 2
                            var visible = 0.0
                            for (k, dir) in directions.enumerated() {
                                let t = march(gx, gy, z0, dir, near: skyRange,
                                              range: horizons ? sunRange : skyRange)
                                visible += 1 / (1 + t.near * t.near) // cos² of the horizon angle
                                if let out = horizonOut {
                                    out[i * sectors + k] = UInt8((atan(t.far) / (.pi / 2) * 255).rounded())
                                }
                            }
                            skyOut[i] = Float(visible / Double(sectors))
                            let t = march(gx, gy, z0, sunDir, near: sunRange, range: sunRange).far
                            // Soft edge over about a degree: the sun's disc plus some blur.
                            let margin = sunElevation - atan(t)
                            let v = Swift.min(1, Swift.max(0, (margin + 0.01) / 0.02))
                            sunOut[i] = Float(v * v * (3 - 2 * v))
                        }
                    }
                }
            }
          }
        }
        return TerrainLighting(sky: sky, sun: sun, horizon: horizon)
    }
}
