import Foundation
import IslandCore

/// Final refinement to ~10 m (plan §3.8): upsamples the eroded heightfield and adds the
/// detail the coarse grid can't hold, then fills pits so every land cell still drains to
/// the sea.
///
/// A 78 m grid blurs steep ground into soft slopes: the real Oahu DEM has three times as
/// much land steeper than 40° at 10 m as a bicubic upsampling of its own 78 m mean. The
/// missing relief is gullies and spurs under a hundred metres across, fluting the ground
/// downhill, so the residual is rough noise stretched along the coarse downslope direction
/// and folded so that its zero lines become V-shaped gullies between rounded spurs, which
/// carries the real slopes with two thirds of the height. How much of it a cell gets
/// depends less on its coarse slope than on how dissected the coarse surface is: planar
/// foothills and talus aprons are smooth at 10 m, while ridged ground at the same slope
/// is fluted. So the flutes fade in from 4° to 11° of coarse slope and then scale with the
/// coarse curvature. A gentle layer of long, rounded swales adds a couple of metres on all
/// but the flattest ground. Channel beds and the shore stay smooth. The defaults are fitted
/// so that refining Oahu's 78 m mean reproduces, bin by bin of coarse slope, how far its
/// 10 m DEM departs from the bicubic surface and how much of it is cliff.
/// Erosion at 10 m would cost minutes per step, so none is run; the channels keep the
/// shape the coarse erosion gave them.
public enum Refine {
    public struct Params: Sendable {
        /// Fine samples per coarse sample along each axis (8 takes 1024² at 78 m to 8192² at 10 m).
        public var factor: Int = 8
        /// Flute amplitude (RMS, metres) per unit of `fluteBase` plus the coarse slope (rise
        /// over run), on ground as dissected as is typical for its slope on this island; it
        /// fades in from `fluteStart` to `fluteFull`.
        public var detail: Double = 10
        public var fluteBase: Double = 0.35
        public var fluteStart: Double = 0.07
        public var fluteFull: Double = 0.2
        /// Flutes scale with the coarse curvature relative to the island's mean at the same
        /// slope, up to `dissectionCap` times, so planar aprons stay smooth while dissected
        /// ground at the same slope is fluted.
        public var dissectionCap: Double = 2
        /// Least curvature multiplier that coarse ground of 40° and steeper gets (fading in
        /// from 20°), so even a smooth wall is fluted.
        public var planarWalls: Double = 0.6
        /// Longest wavelength of the flutes across the slope and along it, in metres;
        /// each runs down to a few fine cells.
        public var acrossWavelength: Double = 80
        public var alongWavelength: Double = 480
        /// Amplitude ratio between successive flute octaves: higher keeps more relief, and
        /// so more slope, in the shortest gullies.
        public var roughness: Double = 0.65
        /// Rounded relief (RMS, metres) on ground of `gentleFull` and steeper, fading in from
        /// `gentleStart` (rise over run), so plains stay smooth.
        public var gentle: Double = 2
        public var gentleStart: Double = 0.02
        public var gentleFull: Double = 0.09
        /// Longest wavelength of the rounded relief across and along the slope; it stops a
        /// few fine cells short of the grid, and `gentleRoughness` keeps its short octaves weak.
        public var gentleAcross: Double = 200
        public var gentleAlong: Double = 400
        public var gentleRoughness: Double = 0.5
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
        // How dissected the coarse surface is: the RMS of its Laplacian over 3×3 cells,
        // relative to the island's mean at the same slope. On Oahu, coarse ground of 10–15°
        // in the most curved third has 2.7 times the residual relief of the flattest third
        // and fifty times the cliffs. Relative, because a generated island's eroded surface
        // is about half as curved as Oahu's at every slope.
        var laplacian = gradX, dissected = gradX
        let cell2 = h.cellSize * h.cellSize
        for y in 0..<h.height {
            for x in 0..<h.width {
                let xl = Swift.max(x - 1, 0), xr = Swift.min(x + 1, h.width - 1)
                let yd = Swift.max(y - 1, 0), yu = Swift.min(y + 1, h.height - 1)
                laplacian[x, y] = Float(Double(h[xl, y] + h[xr, y] + h[x, yd] + h[x, yu] - 4 * h[x, y]) / cell2)
            }
        }
        let binWidth = 0.05, bins = 30
        var binSums = [Double](repeating: 0, count: bins), binCounts = binSums
        for y in 0..<h.height {
            for x in 0..<h.width {
                var squares = 0.0, count = 0.0
                for yy in Swift.max(y - 1, 0)...Swift.min(y + 1, h.height - 1) {
                    for xx in Swift.max(x - 1, 0)...Swift.min(x + 1, h.width - 1) {
                        let l = Double(laplacian[xx, yy])
                        squares += l * l
                        count += 1
                    }
                }
                let k = (squares / count).squareRoot()
                dissected[x, y] = Float(k)
                guard h[x, y] > seaLevel else { continue }
                let s = Double(gradX[x, y] * gradX[x, y] + gradY[x, y] * gradY[x, y]).squareRoot()
                let b = Swift.min(Int(s / binWidth), bins - 1)
                binSums[b] += k
                binCounts[b] += 1
            }
        }
        // Mean curvature per slope bin; sparse bins take the next gentler bin's mean.
        var typical = [Double](repeating: 0, count: bins)
        let overall = binSums.reduce(0, +) / Swift.max(binCounts.reduce(0, +), 1)
        for b in 0..<bins {
            typical[b] = binCounts[b] >= 16 ? binSums[b] / binCounts[b] : (b > 0 ? typical[b - 1] : overall)
        }
        for y in 0..<h.height {
            for x in 0..<h.width {
                let s = Double(gradX[x, y] * gradX[x, y] + gradY[x, y] * gradY[x, y]).squareRoot()
                let t = Swift.min(Swift.max(s / binWidth - 0.5, 0), Double(bins - 1))
                let b = Swift.min(Int(t), bins - 2), w = t - Double(b)
                let mean = typical[b] * (1 - w) + typical[b + 1] * w
                dissected[x, y] = Float(mean > 0 ? Swift.min(Double(dissected[x, y]) / mean, p.dissectionCap) : 1)
            }
        }

        let octaves = Swift.max(1, Int(log2(p.acrossWavelength / (2 * fineCell)).rounded(.up)))
        let gentleOctaves = Swift.max(1, Int(log2(p.gentleAcross / (4 * fineCell)).rounded(.up)))
        let seed = Hash.mix(p.seed ^ 0x5245_4649_4E45)
        let gentleSeed = Hash.mix(p.seed ^ 0x4745_4E54_4C45)
        let rms = fbmRMS(gain: p.roughness, octaves: octaves)
        let fold = folded(gain: p.roughness, octaves: octaves)
        let gentleRMS = fbmRMS(gain: p.gentleRoughness, octaves: gentleOctaves)
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
                            let keep = (1 - channel) * shore
                            let dissection = Swift.max(Double(dissected.sample(x: cx, y: cy)), p.planarWalls * smoothstep(0.36, 0.84, s))
                            let flutes = p.detail * (p.fluteBase + Swift.min(s, 1.5)) * smoothstep(p.fluteStart, p.fluteFull, s) * dissection * keep
                            let rounded = p.gentle * smoothstep(p.gentleStart, p.gentleFull, s) * keep
                            var relief = 0.0
                            if flutes > 0.01 || rounded > 0.01 {
                                // Blend the orientations nearest the downslope direction (axes are
                                // undirected), keeping the variance of one noise.
                                let inv = s > 1e-6 ? 1 / s : 0
                                let dx = gx * inv, dy = gy * inv
                                let px = (Double(x) + 0.5) * fineCell, py = (Double(y) + 0.5) * fineCell
                                var fluteSum = 0.0, roundSum = 0.0, weights = 0.0
                                for i in 0..<k {
                                    let (ax, ay) = axes[i]
                                    let c = ax * dx + ay * dy
                                    var wt = s > 1e-6 ? Swift.max(0, 2 * c * c - 1) : 1
                                    wt *= wt
                                    guard wt > 1e-3 else { continue }
                                    let along = px * ax + py * ay, across = py * ax - px * ay
                                    if flutes > 0.01 {
                                        fluteSum += wt * Noise.fbm(along / p.alongWavelength, across / p.acrossWavelength,
                                                                   seed: seed &+ UInt64(i), octaves: octaves, gain: p.roughness)
                                    }
                                    if rounded > 0.01 {
                                        roundSum += wt * Noise.fbm(along / p.gentleAlong, across / p.gentleAcross,
                                                                   seed: gentleSeed &+ UInt64(i), octaves: gentleOctaves, gain: p.gentleRoughness)
                                    }
                                    weights += wt * wt
                                }
                                let norm = weights > 0 ? 1 / weights.squareRoot() : 0
                                // Folding the flute noise creases its zero lines into V-shaped gullies
                                // between rounded spurs: the same slopes for less height.
                                let crease = (fold.mean - abs(fluteSum * norm / rms)) / fold.rms
                                relief = flutes * crease + rounded * roundSum * norm / gentleRMS
                            }
                            // Keep refined land above the sea where the coarse grid had land.
                            value = Swift.max(seaLevel + 0.01, base + Float(relief))
                        }
                        out[y * width + x] = value
                    }
                }
            }
        }
        PriorityFlood.fill(&fine, seaLevel: seaLevel)
        return fine
    }

    static func smoothstep(_ lo: Double, _ hi: Double, _ x: Double) -> Double {
        let t = Swift.min(Swift.max((x - lo) / (hi - lo), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Mean and RMS deviation of |n| for `Noise.fbm` scaled to unit RMS, sampled once.
    static func folded(gain: Double, octaves: Int) -> (mean: Double, rms: Double) {
        let rms = fbmRMS(gain: gain, octaves: octaves)
        var sum = 0.0, squares = 0.0
        let n = 64
        for j in 0..<n {
            for i in 0..<n {
                let v = abs(Noise.fbm(Double(i) * 0.737 + 0.31, Double(j) * 0.737 + 0.17, seed: 0x464F_4C44, octaves: octaves, gain: gain)) / rms
                sum += v
                squares += v * v
            }
        }
        let mean = sum / Double(n * n)
        return (mean, (squares / Double(n * n) - mean * mean).squareRoot())
    }

    /// RMS of `Noise.fbm` (value noise, RMS about 0.46, summed over independent octaves and
    /// normalised by their weights), so `detail` is an RMS in metres.
    static func fbmRMS(gain: Double, octaves: Int) -> Double {
        var sum = 0.0, squares = 0.0, amp = 1.0
        for _ in 0..<octaves { sum += amp; squares += amp * amp; amp *= gain }
        return 0.458 * squares.squareRoot() / sum
    }
}
