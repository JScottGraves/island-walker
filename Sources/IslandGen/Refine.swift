import Foundation
import IslandCore

/// Final refinement to ~10 m (plan §3.8): upsamples the eroded heightfield and adds the
/// detail the coarse grid can't hold, then fills pits so every land cell still drains to
/// the sea.
///
/// A 78 m grid blurs steep ground into soft slopes: the real Oahu DEM has three times as
/// much land steeper than 40° at 10 m as a bicubic upsampling of its own 78 m mean. The
/// missing relief is gullies and spurs about a hundred metres across, fluting the valley
/// walls downhill, so the residual is rough noise stretched along the coarse downslope
/// direction. Its amplitude rises steeply with the coarse slope and is full on ground of
/// 11° and steeper; flat plains, channel beds and the shore stay smooth. The defaults are
/// fitted so that refining Oahu's 78 m mean reproduces its 10 m slope histogram, with
/// flutes six times longer than wide so they read as gullies rather than grain. Erosion
/// at 10 m would cost minutes per step, so none is run; the channels keep the shape the
/// coarse erosion gave them.
public enum Refine {
    public struct Params: Sendable {
        /// Fine samples per coarse sample along each axis (8 takes 1024² at 78 m to 8192² at 10 m).
        public var factor: Int = 8
        /// Residual amplitude (RMS, metres) on coarse ground of `detailSlope` and steeper.
        public var detail: Double = 13
        /// Coarse slope (rise over run) at which the residual reaches full amplitude; below
        /// it the amplitude grows as (slope / detailSlope)^detailExponent.
        public var detailSlope: Double = 0.2
        public var detailExponent: Double = 1.5
        /// Longest wavelength of the flutes across the slope and along it, in metres;
        /// each runs down to a few fine cells.
        public var acrossWavelength: Double = 100
        public var alongWavelength: Double = 600
        /// Amplitude ratio between successive flute octaves: higher keeps more relief, and
        /// so more slope, in the shortest gullies.
        public var roughness: Double = 0.65
        /// Drainage area (m²) over which the residual fades out into a channel bed.
        public var channelStart: Double = 1e5
        public var channelFull: Double = 1e7
        public var seed: UInt64 = 0
        public init() {}
    }

    /// Orientations the flute noise is built in; each cell blends the ones nearest its
    /// downslope direction, so no coordinate frame is bent across the island.
    static let orientations = 6

    public static func refine(_ h: Grid, drainageArea area: Grid, seaLevel: Float, params p: Params = Params()) -> Grid {
        precondition(p.factor >= 1)
        precondition(area.width == h.width && area.height == h.height)
        let f = p.factor
        let fineCell = h.cellSize / Double(f)
        var fine = Grid(width: h.width * f, height: h.height * f, cellSize: fineCell)

        // Coarse gradient and log drainage area, sampled bilinearly below.
        var gradX = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        var gradY = gradX, logArea = gradX
        for y in 0..<h.height {
            for x in 0..<h.width {
                let xl = Swift.max(x - 1, 0), xr = Swift.min(x + 1, h.width - 1)
                let yd = Swift.max(y - 1, 0), yu = Swift.min(y + 1, h.height - 1)
                gradX[x, y] = Float(Double(h[xr, y] - h[xl, y]) / (Double(xr - xl) * h.cellSize))
                gradY[x, y] = Float(Double(h[x, yu] - h[x, yd]) / (Double(yu - yd) * h.cellSize))
                logArea[x, y] = Float(log(Swift.max(Double(area[x, y]), 1)))
            }
        }

        let octaves = Swift.max(1, Int(log2(p.acrossWavelength / (2 * fineCell)).rounded(.up)))
        let seed = Hash.mix(p.seed ^ 0x5245_4649_4E45)
        let rms = fbmRMS(gain: p.roughness, octaves: octaves)
        let k = orientations
        let axes = (0..<k).map { i -> (Double, Double) in
            let a = Double(i) * Double.pi / Double(k)
            return (cos(a), sin(a))
        }
        let lo = log(p.channelStart), hi = log(p.channelFull)
        let invF = 1 / Double(f)
        let width = fine.width, rowCount = fine.height
        fine.values.withUnsafeMutableBufferPointer { buffer in
            let out = SharedBuffer(buffer)
            Parallel.forRanges(rowCount) { rows in
                for y in rows {
                    let cy = (Double(y) + 0.5) * invF - 0.5
                    for x in 0..<width {
                        let cx = (Double(x) + 0.5) * invF - 0.5
                        // The coastline follows the bilinear surface: bicubic overshoot lifted
                        // the shallow shelf into a chain of islets and cut square notches
                        // into the coast. Bicubic still shapes the land and the sea floor.
                        let isLand = h.sample(x: cx, y: cy) > seaLevel
                        let base = h.sampleCubic(x: cx, y: cy)
                        let above = Swift.max(0, Double(base - seaLevel))
                        var value = isLand ? base : Swift.min(base, seaLevel - 0.01)
                        if isLand {
                            let gx = Double(gradX.sample(x: cx, y: cy)), gy = Double(gradY.sample(x: cx, y: cy))
                            let s = (gx * gx + gy * gy).squareRoot()
                            let a = Double(logArea.sample(x: cx, y: cy))
                            let t = Swift.min(Swift.max((a - lo) / (hi - lo), 0), 1)
                            let channel = t * t * (3 - 2 * t)
                            let shore = Swift.min(1, above / 5)
                            let amplitude = p.detail * pow(Swift.min(s / p.detailSlope, 1), p.detailExponent) * (1 - channel) * shore
                            var n = 0.0
                            if amplitude > 0.01 {
                                // Blend the orientations nearest the downslope direction (axes are
                                // undirected), keeping the variance of one noise.
                                let inv = s > 1e-6 ? 1 / s : 0
                                let dx = gx * inv, dy = gy * inv
                                let px = (Double(x) + 0.5) * fineCell, py = (Double(y) + 0.5) * fineCell
                                var sum = 0.0, weights = 0.0
                                for i in 0..<k {
                                    let (ax, ay) = axes[i]
                                    let c = ax * dx + ay * dy
                                    var wt = s > 1e-6 ? Swift.max(0, 2 * c * c - 1) : 1
                                    wt *= wt
                                    guard wt > 1e-3 else { continue }
                                    let along = (px * ax + py * ay) / p.alongWavelength
                                    let across = (py * ax - px * ay) / p.acrossWavelength
                                    sum += wt * Noise.fbm(along, across, seed: seed &+ UInt64(i), octaves: octaves, gain: p.roughness)
                                    weights += wt * wt
                                }
                                n = weights > 0 ? sum / weights.squareRoot() : 0
                            }
                            // Keep refined land above the sea where the coarse grid had land.
                            value = Swift.max(seaLevel + 0.01, base + Float(amplitude * n / rms))
                        }
                        out[y * width + x] = value
                    }
                }
            }
        }
        PriorityFlood.fill(&fine, seaLevel: seaLevel)
        return fine
    }

    /// RMS of `Noise.fbm` (value noise, RMS about 0.46, summed over independent octaves and
    /// normalised by their weights), so `detail` is an RMS in metres.
    static func fbmRMS(gain: Double, octaves: Int) -> Double {
        var sum = 0.0, squares = 0.0, amp = 1.0
        for _ in 0..<octaves { sum += amp; squares += amp * amp; amp *= gain }
        return 0.458 * squares.squareRoot() / sum
    }
}
