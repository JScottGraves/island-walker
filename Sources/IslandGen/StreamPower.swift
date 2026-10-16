import Dispatch
import Foundation
import IslandCore

/// Landscape evolution by the stream power law, solved implicitly
/// (Braun & Willett 2013, "A very efficient O(n), implicit and parallel method to solve
/// the stream power equation governing fluvial incision and landscape evolution").
///
///     dh/dt = U − K · A^m · S^n + G · Qs / Q + Kd · ∇²h
///
/// with A the rain-weighted upstream drainage area and S the slope to the receiver.
/// n > 1 makes steep reaches erode disproportionately fast, so a step in a river profile
/// (a waterfall at a landslide scarp) stays sharp and retreats upstream instead of
/// smoothing out: the knickpoint mechanism behind Hawaiian amphitheater-headed valleys
/// (Lamb et al. 2007; research/hawaiian-geology.md §4). For n ≠ 1 the implicit update is
/// solved per cell by Newton's method (Braun & Willett 2013, eq. 23).
/// The G term is deposition (Yuan et al. 2019, "A new efficient method to solve the
/// stream power law model taking into account sediment deposition"): sediment settles at
/// G · p · Qs / Q (p local rain, Qs sediment flux, Q discharge), so small, loaded streams
/// drop much of their load and big rivers carry theirs to sea. This builds alluvial plains
/// and fans below cliffs instead of grading them to sea level.
/// This is the reference CPU implementation; a Metal port comes after it is validated.
public enum StreamPower {
    public struct Params: Sendable {
        /// Fluvial erodibility, defined at n = 1, m = 0.5 (units 1 / yr with rain-weighted
        /// area). Other exponents are rescaled so a channel with `referenceDischarge` and
        /// `referenceSlope` erodes at the same rate whatever m and n are.
        public var erodibility: Double = 5e-6
        public var areaExponent: Double = 0.5
        /// Slope exponent n. 1 is linear; 1.5–2 keeps knickpoints sharp.
        public var slopeExponent: Double = 1
        /// When > 0, incision is K·A^0.5·S·(1 + (S / knickpointSlope)^(n−1)) instead of a
        /// pure power law: gentle reaches (coastal plains, valley floors) erode as at n = 1,
        /// and only reaches steeper than this slope get the extra nonlinear boost that keeps
        /// waterfalls sharp. referenceDischarge/referenceSlope and areaExponent are unused then.
        public var knickpointSlope: Double = 0
        /// Rain-weighted drainage area (m³/yr) and slope at which m and n leave the rate unchanged.
        public var referenceDischarge: Double = 1e7
        public var referenceSlope: Double = 0.1
        /// Hillslope diffusivity in m² / yr (soil creep).
        public var diffusivity: Double = 0.02
        /// Dimensionless deposition coefficient G (settling velocity over rain rate).
        /// 0 is detachment-limited (no deposition). 1 raised the islands by ~65 m on
        /// average and buried the foot of the pali under a 2–3 km ramp; 0.3 keeps the cliff
        /// and still builds fans and a coastal plain.
        public var deposition: Double = 0.3
        /// Steepest slope rock walls hold before they fail (rise over run); steeper ground
        /// collapses back to it each step. Retreating headwalls then widen valley heads into
        /// bowls instead of narrow slots. 0 disables it.
        public var criticalSlope: Double = 0
        /// Uplift (positive) or subsidence (negative) in m / yr, applied to land and sea floor.
        public var uplift: Double = 0
        /// Time step in years.
        public var timeStep: Double = 2000
        public var seaLevel: Float = 0
        /// Slope-weighted random choice among downslope neighbours (see step 1 below).
        public var randomRouting = true
        /// Seeds the random routing.
        public var seed: UInt64 = 0
        public init() {}
    }

    /// Per-step outputs useful for rendering and metrics.
    public struct Flow {
        /// Index of each cell's downstream receiver; equal to the cell itself for outlets.
        public var receivers: [Int32]
        /// Rain-weighted drainage area in m² (rain in m/yr times area, i.e. discharge in m³/yr
        /// when rainfall is supplied; plain area when it is not).
        public var area: Grid
    }

    /// Advances the landscape by one time step.
    /// - Parameters:
    ///   - rain: optional rainfall map (m/yr) weighting drainage area; nil means uniform 1.
    ///   - hardness: optional rock hardness; erodibility is divided by it.
    @discardableResult
    public static func step(_ h: inout Grid, rain: Grid? = nil, hardness: Grid? = nil, params: Params = Params(),
                            stepIndex: Int = 0, timer: PhaseTimer? = nil) -> Flow {
        let n = h.count, w = h.width, rows = h.height
        let sea = params.seaLevel
        let dt = params.timeStep
        let cell = h.cellSize
        func phase<T>(_ name: String, _ body: () -> T) -> T { timer?.measure(name, body) ?? body() }

        if params.uplift != 0 {
            let du = Float(params.uplift * dt)
            for i in 0..<n where !h.isBoundary(i) { h.values[i] += du }
        }

        phase("fill") { PriorityFlood.fill(&h, seaLevel: sea) }

        // 1. Receivers. Plain steepest-descent D8 carves channels along the grid's 0°, 45°
        //    and 90° lines. With `randomRouting`, each step picks a downslope neighbour at
        //    random, weighted by slope; over many steps flow spreads like multiple-flow
        //    routing and channels follow the terrain instead of the grid.
        //    The choice is a pure hash of (seed, step, cell), so results stay deterministic.
        //    Every cell is independent, so rows run in parallel.
        var receivers = [Int32](repeating: 0, count: n)
        var receiverDistance = [Double](repeating: 1, count: n)
        let stepSeed = Hash.mix(params.seed ^ Hash.mix(UInt64(stepIndex)))
        phase("receivers") {
            let hv = h.values
            receivers.withUnsafeMutableBufferPointer { recBuf in
                receiverDistance.withUnsafeMutableBufferPointer { distBuf in
                    let rec = SharedBuffer(recBuf), dist = SharedBuffer(distBuf)
                    Parallel.forRanges(rows) { rowRange in
                        var slopes = [Double](repeating: 0, count: 8)
                        for y in rowRange {
                            for x in 0..<w {
                                let i = y * w + x
                                rec[i] = Int32(i)
                                if x == 0 || y == 0 || x == w - 1 || y == rows - 1 || hv[i] <= sea { continue }
                                var best = 0.0, bestK = -1, total = 0.0
                                for k in 0..<8 {
                                    let j = (y + D8.dy[k]) * w + (x + D8.dx[k])
                                    let s = Double(hv[i] - hv[j]) / (D8.distance[k] * cell)
                                    slopes[k] = Swift.max(0, s)
                                    total += slopes[k]
                                    if s > best { best = s; bestK = k }
                                }
                                guard bestK >= 0 else { continue }
                                var pick = bestK
                                if params.randomRouting && total > 0 {
                                    var u = Double(Hash.mix(stepSeed ^ UInt64(i)) >> 11) * 0x1.0p-53 * total
                                    for k in 0..<8 where slopes[k] > 0 {
                                        pick = k
                                        u -= slopes[k]
                                        if u < 0 { break }
                                    }
                                }
                                rec[i] = Int32((y + D8.dy[pick]) * w + (x + D8.dx[pick]))
                                dist[i] = D8.distance[pick] * cell
                            }
                        }
                    }
                }
            }
        }

        // 2. Donor lists (counting sort) and the upstream-ordered stack. Each outlet's
        //    basin is one contiguous run of the stack, and basins never share cells, so
        //    the passes below run basin groups in parallel with serial-identical results.
        var stack = [Int32]()
        var basinStarts = [Int]()
        phase("stack") {
            var donorCount = [Int32](repeating: 0, count: n + 1)
            for i in 0..<n where Int(receivers[i]) != i { donorCount[Int(receivers[i]) + 1] += 1 }
            for i in 0..<n { donorCount[i + 1] += donorCount[i] }
            var donors = [Int32](repeating: 0, count: n)
            var fill = donorCount
            for i in 0..<n where Int(receivers[i]) != i {
                let r = Int(receivers[i])
                donors[Int(fill[r])] = Int32(i)
                fill[r] += 1
            }
            stack.reserveCapacity(n)
            var work = [Int32]()
            for i in 0..<n where Int(receivers[i]) == i {
                basinStarts.append(stack.count)
                work.append(Int32(i))
                while let c = work.popLast() {
                    stack.append(c)
                    let ci = Int(c)
                    for k in Int(donorCount[ci])..<Int(donorCount[ci + 1]) { work.append(donors[k]) }
                }
            }
        }
        // Group consecutive basins into roughly equal runs of the stack.
        var grouped = [Range<Int>]()
        let target = Swift.max(1, n / Parallel.chunkCount)
        var groupStart = 0
        for b in basinStarts.indices {
            let end = b + 1 < basinStarts.count ? basinStarts[b + 1] : n
            if end - groupStart >= target || end == n {
                grouped.append(groupStart..<end)
                groupStart = end
            }
        }
        let groups = grouped
        let order = stack, flowTo = receivers, distances = receiverDistance
        func forBasinGroups(_ body: (Range<Int>) -> Void) {
            DispatchQueue.concurrentPerform(iterations: groups.count) { body(groups[$0]) }
        }

        // 3. Drainage area, accumulated from the top of each basin down.
        var area = Grid(width: w, height: rows, cellSize: cell)
        let cellArea = Float(h.cellArea)
        let rainValues = rain?.values
        phase("area") {
            area.values.withUnsafeMutableBufferPointer { areaBuf in
                let a = SharedBuffer(areaBuf)
                forBasinGroups { range in
                    for s in range { let i = Int(order[s]); a[i] = cellArea * (rainValues?[i] ?? 1) }
                    for s in range.reversed() {
                        let i = Int(order[s]), r = Int(flowTo[i])
                        if r != i { a[r] += a[i] }
                    }
                }
            }
        }

        // 4. Implicit fluvial incision, walking downstream-to-upstream so every
        //    receiver already holds its new height.
        let beforeIncision = params.deposition > 0 ? h.values : []
        let hardnessValues = hardness?.values
        let areaValues = area.values
        let slopeN = params.slopeExponent
        let hybrid = params.knickpointSlope > 0 && slopeN != 1
        let linear = slopeN == 1 && params.areaExponent == 0.5
        let areaScale = pow(params.referenceDischarge, 0.5)
        let slopeScale = pow(params.referenceSlope, 1 - slopeN)
        phase("incision") {
            h.values.withUnsafeMutableBufferPointer { hBuf in
                let hv = SharedBuffer(hBuf)
                forBasinGroups { range in
                    for s in range {
                        let i = Int(order[s]), r = Int(flowTo[i])
                        if r == i { continue }
                        var k = params.erodibility
                        if let hardnessValues { k /= Double(hardnessValues[i]) }
                        // Rivers cut down to sea level at the coast, not to the depth of the sea floor.
                        let base = Swift.max(hv[r], sea)
                        if linear {
                            let f = Float(k * dt * pow(Double(areaValues[i]), params.areaExponent) / distances[i])
                            hv[i] = (hv[i] + f * base) / (1 + f)
                        } else {
                            let h0 = Double(hv[i]), b = Double(base)
                            guard h0 > b else { continue }
                            // g(x) = x − h0 + f1 · (x − b) + f · (x − b)^n, with f1 = 0 for the
                            // pure power law.
                            let f1: Double, f: Double
                            if hybrid {
                                f1 = k * dt * pow(Double(areaValues[i]), 0.5) / distances[i]
                                f = f1 / pow(params.knickpointSlope * distances[i], slopeN - 1)
                            } else {
                                let a = Double(areaValues[i]) / params.referenceDischarge
                                f1 = 0
                                f = k * dt * areaScale * pow(Swift.max(a, 0), params.areaExponent)
                                    * slopeScale / pow(distances[i], slopeN)
                            }
                            // Solve g(x) = 0 for x in (b, h0]; g is increasing and convex there,
                            // so Newton from h0 converges from above.
                            var x = h0
                            for _ in 0..<20 {
                                let d = x - b
                                let g = x - h0 + f1 * d + f * pow(d, slopeN)
                                let dg = 1 + f1 + slopeN * f * pow(d, slopeN - 1)
                                let next = Swift.max(b + 1e-9, x - g / dg)
                                if abs(next - x) < 1e-4 { x = next; break }
                                x = next
                            }
                            hv[i] = Float(x)
                        }
                    }
                }
            }
        }

        // 4b. Deposition, walking upstream-to-downstream and carrying the eroded volume
        //     (in metres of height over one cell) along with the flow. Over one step the
        //     rate G · p · Qs / Q settles a share G · (own discharge / total discharge) of
        //     the arriving load. A cell never rises above its lowest donor, so deposits
        //     fill channels and plains without damming them.
        if params.deposition > 0 {
            phase("deposition") {
                var load = [Float](repeating: 0, count: n)
                var lowestDonor = [Float](repeating: .infinity, count: n)
                let g = Float(params.deposition)
                h.values.withUnsafeMutableBufferPointer { hBuf in
                    load.withUnsafeMutableBufferPointer { loadBuf in
                        lowestDonor.withUnsafeMutableBufferPointer { lowBuf in
                            let hv = SharedBuffer(hBuf), ld = SharedBuffer(loadBuf), low = SharedBuffer(lowBuf)
                            forBasinGroups { range in
                                for s in range.reversed() {
                                    let i = Int(order[s]), r = Int(flowTo[i])
                                    let incoming = ld[i]
                                    var carried = incoming + Swift.max(0, beforeIncision[i] - hv[i])
                                    if r != i && incoming > 0 && hv[i] > sea && low[i].isFinite {
                                        let share = Swift.min(1, g * cellArea * (rainValues?[i] ?? 1) / Swift.max(areaValues[i], 1e-6))
                                        let room = Swift.max(0, low[i] - hv[i])
                                        let deposit = Swift.min(incoming * share, room)
                                        hv[i] += deposit
                                        carried -= deposit
                                    }
                                    if r != i {
                                        ld[r] += carried
                                        low[r] = Swift.min(low[r], hv[i])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4c. Threshold hillslopes: land standing more than criticalSlope above any
        //     neighbour fails down to that angle (shallow landslides on over-steepened
        //     walls). Reads the old heights only, so rows run in parallel.
        if params.criticalSlope > 0 {
            let sc = Float(params.criticalSlope * cell)
            phase("landslides") {
                let old = h.values
                h.values.withUnsafeMutableBufferPointer { hBuf in
                    let hv = SharedBuffer(hBuf)
                    Parallel.forRanges(rows - 2) { rowRange in
                        for y in (rowRange.lowerBound + 1)..<(rowRange.upperBound + 1) {
                            for x in 1..<(w - 1) {
                                let i = y * w + x
                                if old[i] <= sea { continue }
                                var cap = old[i]
                                for k in 0..<8 {
                                    let j = (y + D8.dy[k]) * w + (x + D8.dx[k])
                                    cap = Swift.min(cap, Swift.max(old[j], sea) + sc * Float(D8.distance[k]))
                                }
                                hv[i] = cap
                            }
                        }
                    }
                }
            }
        }

        // 5. Hillslope diffusion (explicit; stable while Kd·dt/dx² < 0.25). Reads the old
        //    heights only, so rows run in parallel.
        if params.diffusivity > 0 {
            let c = Float(params.diffusivity * dt / (cell * cell))
            precondition(c < 0.25, "diffusion step unstable; lower timeStep or diffusivity")
            phase("diffusion") {
                let old = h.values
                h.values.withUnsafeMutableBufferPointer { hBuf in
                    let hv = SharedBuffer(hBuf)
                    Parallel.forRanges(rows - 2) { rowRange in
                        for y in (rowRange.lowerBound + 1)..<(rowRange.upperBound + 1) {
                            for x in 1..<(w - 1) {
                                let i = y * w + x
                                if old[i] <= sea { continue }
                                let lap = old[i - 1] + old[i + 1] + old[i - w] + old[i + w] - 4 * old[i]
                                hv[i] = old[i] + c * lap
                            }
                        }
                    }
                }
            }
        }

        return Flow(receivers: receivers, area: area)
    }
}
