import Foundation
import IslandCore

/// Final refinement to ~10 m (plan §3.8): upsamples the eroded heightfield and adds a
/// detail residual that the coarse grid can't hold, then fills pits so every land cell
/// still drains to the sea.
///
/// The residual is fractal noise whose amplitude follows the coarse terrain: strongest on
/// steep slopes, faded out in channels (where sediment smooths the bed) and towards sea
/// level (beaches and coastal plains). Erosion at 10 m would cost minutes per step, so
/// none is run; the channels keep the shape the coarse erosion gave them.
public enum Refine {
    public struct Params: Sendable {
        /// Fine samples per coarse sample along each axis (8 takes 1024² at 78 m to 8192² at 10 m).
        public var factor: Int = 8
        /// Residual amplitude on steep ground, in metres.
        public var detail: Double = 4
        /// Longest wavelength of the residual, in metres; it runs down to a few fine cells.
        public var wavelength: Double = 400
        /// Drainage area (m²) over which the residual fades out into a channel bed.
        public var channelStart: Double = 1e5
        public var channelFull: Double = 1e7
        public var seed: UInt64 = 0
        public init() {}
    }

    public static func refine(_ h: Grid, drainageArea area: Grid, seaLevel: Float, params p: Params = Params()) -> Grid {
        precondition(p.factor >= 1)
        precondition(area.width == h.width && area.height == h.height)
        let f = p.factor
        let fineCell = h.cellSize / Double(f)
        var fine = Grid(width: h.width * f, height: h.height * f, cellSize: fineCell)

        // Coarse slope (rise over run) and log drainage area, sampled bilinearly below.
        var slope = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        var logArea = Grid(width: h.width, height: h.height, cellSize: h.cellSize)
        for y in 0..<h.height {
            for x in 0..<h.width {
                let xl = Swift.max(x - 1, 0), xr = Swift.min(x + 1, h.width - 1)
                let yd = Swift.max(y - 1, 0), yu = Swift.min(y + 1, h.height - 1)
                let gx = Double(h[xr, y] - h[xl, y]) / (Double(xr - xl) * h.cellSize)
                let gy = Double(h[x, yu] - h[x, yd]) / (Double(yu - yd) * h.cellSize)
                slope[x, y] = Float((gx * gx + gy * gy).squareRoot())
                logArea[x, y] = Float(log(Swift.max(Double(area[x, y]), 1)))
            }
        }

        let octaves = Swift.max(1, Int(log2(p.wavelength / (2 * fineCell)).rounded(.up)))
        let seed = Hash.mix(p.seed ^ 0x5245_4649_4E45)
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
                            let s = Double(slope.sample(x: cx, y: cy))
                            let a = Double(logArea.sample(x: cx, y: cy))
                            let t = Swift.min(Swift.max((a - lo) / (hi - lo), 0), 1)
                            let channel = t * t * (3 - 2 * t)
                            let steepness = 0.3 + 0.7 * Swift.min(1, s / 0.5)
                            let shore = Swift.min(1, above / 5)
                            let amplitude = p.detail * steepness * (1 - channel) * shore
                            let px = (Double(x) + 0.5) * fineCell, py = (Double(y) + 0.5) * fineCell
                            let n = Noise.fbm(px / p.wavelength, py / p.wavelength, seed: seed, octaves: octaves)
                            // Keep refined land above the sea where the coarse grid had land.
                            value = Swift.max(seaLevel + 0.01, base + Float(amplitude * n))
                        }
                        out[y * width + x] = value
                    }
                }
            }
        }
        PriorityFlood.fill(&fine, seaLevel: seaLevel)
        return fine
    }
}
