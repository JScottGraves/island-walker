import Foundation
import IslandCore

/// Landscape evolution by the stream power law, solved implicitly
/// (Braun & Willett 2013, "A very efficient O(n), implicit and parallel method to solve
/// the stream power equation governing fluvial incision and landscape evolution").
///
///     dh/dt = U − K · A^m · S^n + G · Qs / Q + Kd · ∇²h
///
/// with n = 1, A the rain-weighted upstream drainage area, S the slope to the receiver.
/// The G term is deposition (Yuan et al. 2019, "A new efficient method to solve the
/// stream power law model taking into account sediment deposition"): sediment settles at
/// G · p · Qs / Q (p local rain, Qs sediment flux, Q discharge), so small, loaded streams
/// drop much of their load and big rivers carry theirs to sea. This builds alluvial plains
/// and fans below cliffs instead of grading them to sea level.
/// This is the reference CPU implementation; a Metal port comes after it is validated.
public enum StreamPower {
    public struct Params: Sendable {
        /// Fluvial erodibility. Units with m = 0.5: m^(1-2m) / yr = 1 / yr.
        public var erodibility: Double = 5e-6
        public var areaExponent: Double = 0.5
        /// Hillslope diffusivity in m² / yr (soil creep).
        public var diffusivity: Double = 0.02
        /// Dimensionless deposition coefficient G (settling velocity over rain rate).
        /// 0 is detachment-limited (no deposition); Yuan et al. use values around 1.
        public var deposition: Double = 1
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
                            stepIndex: Int = 0) -> Flow {
        let n = h.count, w = h.width
        let sea = params.seaLevel
        let dt = params.timeStep

        if params.uplift != 0 {
            let du = Float(params.uplift * dt)
            for i in 0..<n where !h.isBoundary(i) { h.values[i] += du }
        }

        PriorityFlood.fill(&h, seaLevel: sea)

        // 1. Receivers. Plain steepest-descent D8 carves channels along the grid's 0°, 45°
        //    and 90° lines. With `randomRouting`, each step picks a downslope neighbour at
        //    random, weighted by slope; over many steps flow spreads like multiple-flow
        //    routing and channels follow the terrain instead of the grid.
        //    The choice is a pure hash of (seed, step, cell), so results stay deterministic.
        var receivers = [Int32](repeating: 0, count: n)
        var receiverDistance = [Double](repeating: 1, count: n)
        let stepSeed = Hash.mix(params.seed ^ Hash.mix(UInt64(stepIndex)))
        var slopes = [Double](repeating: 0, count: 8)
        for i in 0..<n {
            receivers[i] = Int32(i)
            if h.isBoundary(i) || h.values[i] <= sea { continue }
            let x = i % w, y = i / w
            var best = 0.0, bestK = -1, total = 0.0
            for k in 0..<8 {
                let j = (y + D8.dy[k]) * w + (x + D8.dx[k])
                let s = Double(h.values[i] - h.values[j]) / (D8.distance[k] * h.cellSize)
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
            receivers[i] = Int32((y + D8.dy[pick]) * w + (x + D8.dx[pick]))
            receiverDistance[i] = D8.distance[pick] * h.cellSize
        }

        // 2. Donor lists (counting sort) and the upstream-ordered stack.
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

        var stack = [Int32]()
        stack.reserveCapacity(n)
        var work = [Int32]()
        for i in 0..<n where Int(receivers[i]) == i {
            work.append(Int32(i))
            while let c = work.popLast() {
                stack.append(c)
                let ci = Int(c)
                for k in Int(donorCount[ci])..<Int(donorCount[ci + 1]) { work.append(donors[k]) }
            }
        }

        // 3. Drainage area, accumulated from the top of each basin down.
        var area = Grid(width: w, height: h.height, cellSize: h.cellSize)
        let cellArea = Float(h.cellArea)
        for i in 0..<n { area.values[i] = cellArea * (rain?.values[i] ?? 1) }
        for s in stride(from: n - 1, through: 0, by: -1) {
            let i = Int(stack[s]), r = Int(receivers[i])
            if r != i { area.values[r] += area.values[i] }
        }

        // 4. Implicit fluvial incision, walking downstream-to-upstream so every
        //    receiver already holds its new height.
        let beforeIncision = params.deposition > 0 ? h.values : []
        for s in 0..<n {
            let i = Int(stack[s]), r = Int(receivers[i])
            if r == i { continue }
            var k = params.erodibility
            if let hardness { k /= Double(hardness.values[i]) }
            let f = Float(k * dt * pow(Double(area.values[i]), params.areaExponent) / receiverDistance[i])
            // Rivers cut down to sea level at the coast, not to the depth of the sea floor.
            let base = Swift.max(h.values[r], sea)
            h.values[i] = (h.values[i] + f * base) / (1 + f)
        }

        // 4b. Deposition, walking upstream-to-downstream and carrying the eroded volume
        //     (in metres of height over one cell) along with the flow. Over one step the
        //     rate G · p · Qs / Q settles a share G · (own discharge / total discharge) of
        //     the arriving load. A cell never rises above its lowest donor, so deposits
        //     fill channels and plains without damming them.
        if params.deposition > 0 {
            var load = [Float](repeating: 0, count: n)
            var lowestDonor = [Float](repeating: .infinity, count: n)
            let g = Float(params.deposition)
            for s in stride(from: n - 1, through: 0, by: -1) {
                let i = Int(stack[s]), r = Int(receivers[i])
                let incoming = load[i]
                var carried = incoming + Swift.max(0, beforeIncision[i] - h.values[i])
                if r != i && incoming > 0 && h.values[i] > sea && lowestDonor[i].isFinite {
                    let share = Swift.min(1, g * cellArea * (rain?.values[i] ?? 1) / Swift.max(area.values[i], 1e-6))
                    let room = Swift.max(0, lowestDonor[i] - h.values[i])
                    let deposit = Swift.min(incoming * share, room)
                    h.values[i] += deposit
                    carried -= deposit
                }
                if r != i {
                    load[r] += carried
                    lowestDonor[r] = Swift.min(lowestDonor[r], h.values[i])
                }
            }
        }

        // 5. Hillslope diffusion (explicit; stable while Kd·dt/dx² < 0.25).
        if params.diffusivity > 0 {
            let c = Float(params.diffusivity * dt / (h.cellSize * h.cellSize))
            precondition(c < 0.25, "diffusion step unstable; lower timeStep or diffusivity")
            let old = h.values
            for y in 1..<(h.height - 1) {
                for x in 1..<(w - 1) {
                    let i = y * w + x
                    if old[i] <= sea { continue }
                    let lap = old[i - 1] + old[i + 1] + old[i - w] + old[i + w] - 4 * old[i]
                    h.values[i] = old[i] + c * lap
                }
            }
        }

        return Flow(receivers: receivers, area: area)
    }
}
