import Foundation

/// Swell refraction and shoaling over the bathymetry, for the surf (research/waves.md, item 1).
///
/// A swell system's crests are the level sets of its arrival time τ, which solves the eikonal
/// equation |∇τ| = 1/c with c the linear phase speed over the local depth (ω² = g k tanh kh).
/// It is solved by fast marching (Sethian 1996) from the open-sea edges the system comes in
/// through, as a plane wave; land stops it, so the water behind an island is reached only around
/// its ends (late, and with the rays spread wide). Where fronts from two sides meet the earlier
/// one wins (τ is continuous, with a kink).
///
/// The height then follows the energy flux along the rays (no dissipation until breaking):
/// H² · Cg · b stays constant, b the rays' spacing, whose logarithm grows along a ray by the
/// divergence of the unit propagation direction (d ln b / ds = ∇·k̂). Both are carried in arrival
/// order with the same upwind stencil as τ. Where the shoaled height reaches `breakerIndex` × the
/// depth the waves break and are held to it, losing the energy above it (so a reef flat or bar
/// shelters the water behind it).
///
/// Sand a little way inland of the waterline gets arrival times and heights carried on from
/// the water at the shallowest speed (for the swash), without feeding back into the water.
public enum WaveRefraction {
    /// Significant height to depth in a saturated surf zone (individual breakers at ~0.78).
    public static let breakerIndex = 0.6
    /// Depth (m) under which the cap stops shrinking: the coarse bathymetry's last cells straddle
    /// the waterline, and the bores carry on up into the swash.
    public static let capDepth = 0.5
    /// Depth (m) the phase speed stops slowing at: past breaking, bores run at about √(g(h + H)),
    /// ~4.4 m/s (crests ~35 m apart at 8 s).
    public static let minDepth = 2.0
    /// Rays converge to at most this gain in height (a caustic's peak, smoothed).
    public static let maxFocus = 2.5
    static let g = 9.81

    /// One swell system: its peak period (s), the direction it travels towards (radians, x east,
    /// y north) and its significant height offshore (m).
    public struct System: Sendable {
        public var period: Double
        public var towards: Double
        public var height: Double
        public init(period: Double, towards: Double, height: Double) {
            self.period = period
            self.towards = towards
            self.height = height
        }
    }

    /// Per cell: the crests' arrival time (s from an arbitrary origin; infinite where the system
    /// never arrives) and the local significant height (m).
    public struct Field: Sendable {
        public var arrival: Grid
        public var height: Grid
    }

    /// Linear dispersion: the wavenumber (rad/m) of waves of `period` over `depth` metres.
    public static func wavenumber(depth: Double, period: Double) -> Double {
        let w = 2 * Double.pi / period
        let h = Swift.max(depth, 1e-3)
        let k0 = w * w / g
        // Eckart's approximation, then Newton on g k tanh(kh) − ω².
        var k = k0 / Swift.max(tanh(k0 * h), 1e-6).squareRoot()
        for _ in 0..<4 {
            let t = tanh(k * h)
            let f = g * k * t - w * w
            let df = g * t + g * k * h * (1 - t * t)
            k -= f / df
        }
        return k
    }

    public static func phaseSpeed(depth: Double, period: Double) -> Double {
        2 * Double.pi / period / wavenumber(depth: depth, period: period)
    }

    public static func groupSpeed(depth: Double, period: Double) -> Double {
        let k = wavenumber(depth: depth, period: period)
        let kh = k * Swift.max(depth, 1e-3)
        let n = kh > 20 ? 0.5 : 0.5 * (1 + 2 * kh / sinh(2 * kh))
        return n * 2 * Double.pi / period / k
    }

    /// The system's field over `height` (a heightfield in metres, cells in metres). `band`:
    /// how far inland (m) the sand gets arrival times; `shore`, ShoreDistance.signed of `height`
    /// if already at hand.
    public static func bake(height: Grid, seaLevel: Float, system: System, band: Double = 160, shore: Grid? = nil) -> Field {
        let w = height.width, ht = height.height, n = w * ht, cell = height.cellSize
        // Raw buffers: the passes below run over millions of cells, several of them in parallel.
        func buffer<T>(_ value: T) -> UnsafeMutableBufferPointer<T> {
            let b = UnsafeMutableBufferPointer<T>.allocate(capacity: n)
            b.initialize(repeating: value)
            return b
        }
        let depth = buffer(0.0), slowness = buffer(0.0), cg = buffer(0.0), tau = buffer(Double.infinity)
        let kx = buffer(0.0), ky = buffer(0.0), div = buffer(0.0), scratch = buffer(0.0)
        let flux = buffer(0.0), spread = buffer(0.0), hs = buffer(0.0)
        let water = buffer(false), passable = buffer(false), known = buffer(false), seed = buffer(false)
        defer {
            for b in [depth, slowness, cg, tau, kx, ky, div, scratch, flux, spread, hs] { b.deallocate() }
            for b in [water, passable, known, seed] { b.deallocate() }
        }
        let shore = shore ?? ShoreDistance.signed(height: height, seaLevel: seaLevel)
        let period = system.period
        // Depths, what the waves can reach (the water, and the sand within `band` of it), and the
        // phase and group speeds per cell (the sand's at the shallowest).
        Parallel.forRanges(n) { range in
            for k in range {
                depth[k] = Double(seaLevel - height.values[k])
                water[k] = depth[k] > 0
                passable[k] = water[k] || Double(shore.values[k]) > -band
                guard passable[k] else { continue }
                let d = water[k] ? Swift.max(depth[k], minDepth) : minDepth
                let kn = wavenumber(depth: d, period: period)
                let kh = kn * d
                let nRatio = kh > 20 ? 0.5 : 0.5 * (1 + 2 * kh / sinh(2 * kh))
                let c = 2 * Double.pi / period / kn
                slowness[k] = 1 / c
                cg[k] = nRatio * c
            }
        }

        // Seeds: the open-sea edges the waves come in through, where the water is deep (a plane
        // wave there; over shallow edges the system arrives from inside instead).
        let dir = SIMD2(cos(system.towards), sin(system.towards))
        var edges: [Int] = []
        func inflow(_ inward: SIMD2<Double>) -> Bool { dir.x * inward.x + dir.y * inward.y > 1e-6 }
        if inflow(SIMD2(1, 0)) { for j in 0..<ht { edges.append(j * w) } }
        if inflow(SIMD2(-1, 0)) { for j in 0..<ht { edges.append(j * w + w - 1) } }
        if inflow(SIMD2(0, 1)) { for i in 0..<w { edges.append(i) } }
        if inflow(SIMD2(0, -1)) { for i in 0..<w { edges.append((ht - 1) * w + i) } }
        let edgeDeepest = edges.filter { water[$0] }.map { depth[$0] }.max() ?? 0
        let l0 = g * period * period / (2 * Double.pi)
        let seedDepth = Swift.min(0.5 * l0, 0.8 * edgeDeepest)
        let cRef = phaseSpeed(depth: Swift.max(edgeDeepest, minDepth), period: period)
        var heap = MinHeap()
        for k in edges where water[k] && depth[k] >= seedDepth && !seed[k] {
            let x = (Double(k % w) + 0.5) * cell, y = (Double(k / w) + 0.5) * cell
            tau[k] = (x * dir.x + y * dir.y) / cRef
            seed[k] = true
            heap.push(tau[k], k)
        }

        // Fast marching. Land never updates water.
        @inline(__always) func usable(_ from: Int, for target: Int) -> Bool {
            known[from] && (water[from] || !water[target])
        }
        @inline(__always) func solve(_ k: Int) -> Double {
            let i = k % w, j = k / w
            var a = Double.infinity, b = Double.infinity
            if i > 0, usable(k - 1, for: k) { a = tau[k - 1] }
            if i < w - 1, usable(k + 1, for: k) { a = Swift.min(a, tau[k + 1]) }
            if j > 0, usable(k - w, for: k) { b = tau[k - w] }
            if j < ht - 1, usable(k + w, for: k) { b = Swift.min(b, tau[k + w]) }
            let sh = slowness[k] * cell
            if !a.isFinite || !b.isFinite || abs(a - b) >= sh { return Swift.min(a, b) + sh }
            return 0.5 * (a + b + (2 * sh * sh - (a - b) * (a - b)).squareRoot())
        }
        var order: [Int] = []
        order.reserveCapacity(n)
        while let (t, k) = heap.pop() {
            if known[k] || t > tau[k] { continue }
            known[k] = true
            order.append(k)
            let i = k % w, j = k / w
            @inline(__always) func visit(_ nb: Int) {
                guard passable[nb], !known[nb], water[k] || !water[nb], !seed[nb] else { return }
                let c = solve(nb)
                if c < tau[nb] {
                    tau[nb] = c
                    heap.push(c, nb)
                }
            }
            if i > 0 { visit(k - 1) }
            if i < w - 1 { visit(k + 1) }
            if j > 0 { visit(k - w) }
            if j < ht - 1 { visit(k + w) }
        }

        // The rays' unit direction (from τ's gradient) and its divergence, over the water.
        @inline(__always) func reachedWater(_ k: Int) -> Bool { water[k] && known[k] }
        @inline(__always) func derivative(_ f: UnsafeMutableBufferPointer<Double>, _ k: Int, _ step: Int,
                                          _ hasLo: Bool, _ hasHi: Bool) -> Double {
            let lo = hasLo && reachedWater(k - step), hi = hasHi && reachedWater(k + step)
            if lo && hi { return (f[k + step] - f[k - step]) / (2 * cell) }
            if hi { return (f[k + step] - f[k]) / cell }
            if lo { return (f[k] - f[k - step]) / cell }
            return 0
        }
        Parallel.forRanges(ht) { rows in
            for j in rows {
                for i in 0..<w {
                    let k = j * w + i
                    guard reachedWater(k) else { continue }
                    let gx = derivative(tau, k, 1, i > 0, i < w - 1), gy = derivative(tau, k, w, j > 0, j < ht - 1)
                    let l = (gx * gx + gy * gy).squareRoot()
                    if l > 0 { kx[k] = gx / l; ky[k] = gy / l }
                }
            }
        }
        Parallel.forRanges(ht) { rows in
            for j in rows {
                for i in 0..<w {
                    let k = j * w + i
                    guard reachedWater(k) else { continue }
                    div[k] = derivative(kx, k, 1, i > 0, i < w - 1) + derivative(ky, k, w, j > 0, j < ht - 1)
                }
            }
        }
        // Smoothed over the water (caustics and the kinks where fronts meet are single cells).
        for _ in 0..<2 {
            _ = scratch.update(fromContentsOf: div)
            Parallel.forRanges(ht) { rows in
                for j in rows {
                    for i in 0..<w {
                        let k = j * w + i
                        guard reachedWater(k) else { continue }
                        var sum = 0.0, count = 0.0
                        for y in Swift.max(j - 1, 0)...Swift.min(j + 1, ht - 1) {
                            for x in Swift.max(i - 1, 0)...Swift.min(i + 1, w - 1) where reachedWater(y * w + x) {
                                sum += scratch[y * w + x]
                                count += 1
                            }
                        }
                        div[k] = sum / count
                    }
                }
            }
        }

        // Energy flux (F = ln H²·Cg·b) and ray spacing (A = ln b) along the rays, in arrival order.
        let h0 = system.height, minSpread = -2 * log(maxFocus)
        for k in order {
            if seed[k] {
                hs[k] = h0
                flux[k] = log(h0 * h0 * cg[k])
                continue
            }
            let i = k % w, j = k / w
            // Each axis's upwind neighbour (the earlier one), weighted by its time difference.
            var xu = -1, yu = -1
            if i > 0, usable(k - 1, for: k), tau[k - 1] < tau[k] { xu = k - 1 }
            if i < w - 1, usable(k + 1, for: k), tau[k + 1] < tau[k], xu < 0 || tau[k + 1] < tau[xu] { xu = k + 1 }
            if j > 0, usable(k - w, for: k), tau[k - w] < tau[k] { yu = k - w }
            if j < ht - 1, usable(k + w, for: k), tau[k + w] < tau[k], yu < 0 || tau[k + w] < tau[yu] { yu = k + w }
            let a = xu >= 0 ? tau[k] - tau[xu] : 0, b = yu >= 0 ? tau[k] - tau[yu] : 0
            guard a + b > 0 else { continue }
            @inline(__always) func mix(_ f: UnsafeMutableBufferPointer<Double>) -> Double {
                ((xu >= 0 ? a * f[xu] : 0) + (yu >= 0 ? b * f[yu] : 0)) / (a + b)
            }
            if !water[k] {
                hs[k] = mix(hs)
                continue
            }
            let f = mix(flux)
            // d ln b / ds = ∇·k̂: A(a + b) = a·A_x + b·A_y + h²·div / c.
            let s = Swift.max(mix(spread) + cell * cell * div[k] * slowness[k] / (a + b), minSpread)
            var hk = (exp(f - s) / cg[k]).squareRoot()
            let cap = breakerIndex * Swift.max(depth[k], capDepth)
            if hk > cap { hk = cap }
            hs[k] = hk
            spread[k] = s
            flux[k] = Swift.min(f, log(hk * hk * cg[k]) + s)
        }

        var arrival = Grid(width: w, height: ht, cellSize: cell, fill: .infinity)
        var heights = Grid(width: w, height: ht, cellSize: cell)
        let origin = order.first.map { tau[$0] } ?? 0
        for k in 0..<n where known[k] {
            arrival.values[k] = Float(tau[k] - origin)
            heights.values[k] = Float(hs[k])
        }
        return Field(arrival: arrival, height: heights)
    }
}

/// A binary min-heap of (key, index), with stale entries left in (the caller skips them).
struct MinHeap {
    private var keys: [Double] = []
    private var items: [Int32] = []

    mutating func push(_ key: Double, _ item: Int) {
        keys.append(key)
        items.append(Int32(item))
        var c = keys.count - 1
        while c > 0 {
            let p = (c - 1) / 2
            if keys[p] <= key { break }
            keys[c] = keys[p]
            items[c] = items[p]
            c = p
        }
        keys[c] = key
        items[c] = Int32(item)
    }

    mutating func pop() -> (Double, Int)? {
        guard let top = keys.first else { return nil }
        let item = Int(items[0])
        let key = keys.removeLast(), last = items.removeLast()
        let count = keys.count
        if count > 0 {
            var p = 0
            while true {
                let l = 2 * p + 1
                if l >= count { break }
                let m = l + 1 < count && keys[l + 1] < keys[l] ? l + 1 : l
                if keys[m] >= key { break }
                keys[p] = keys[m]
                items[p] = items[m]
                p = m
            }
            keys[p] = key
            items[p] = last
        }
        return (top, item)
    }
}

extension WaveRefraction {
    /// The surf's wave texture over `height`: per texel the trade swell's and the north swell's
    /// arrival in periods (τ/T, continuous, so the shader's fract has no seam), their local
    /// significant heights (m, smoothed over ~100 m) as two halfs packed into one float (low bits
    /// the trade swell's), and the water's depth (m, smoothed as much). Water neither system
    /// reaches (enclosed ponds) and the sand beyond the band carry their neighbours' phase and no
    /// height. The two systems bake concurrently.
    public static func surfTexels(height: Grid, seaLevel: Float, systems: (System, System), smoothing: Int = 2) -> [SIMD4<Float>] {
        let list = [systems.0, systems.1]
        let w = height.width, ht = height.height, n = w * ht
        var channels = [(phase: [Float], height: [Float])?](repeating: nil, count: 2)
        let lock = NSLock()
        let shore = ShoreDistance.signed(height: height, seaLevel: seaLevel)
        DispatchQueue.concurrentPerform(iterations: 2) { s in
            let f = bake(height: height, seaLevel: seaLevel, system: list[s], shore: shore)
            var phase = f.arrival
            let period = Float(list[s].period)
            var valid = [Bool](repeating: false, count: n)
            for k in 0..<n where phase.values[k].isFinite {
                phase.values[k] /= period
                valid[k] = true
            }
            // Unreached cells take their reached neighbours' mean phase, a ring at a time.
            for _ in 0..<6 {
                var next = valid
                for j in 0..<ht {
                    for i in 0..<w where !valid[j * w + i] {
                        var sum: Float = 0, count: Float = 0
                        if i > 0, valid[j * w + i - 1] { sum += phase.values[j * w + i - 1]; count += 1 }
                        if i < w - 1, valid[j * w + i + 1] { sum += phase.values[j * w + i + 1]; count += 1 }
                        if j > 0, valid[j * w + i - w] { sum += phase.values[j * w + i - w]; count += 1 }
                        if j < ht - 1, valid[j * w + i + w] { sum += phase.values[j * w + i + w]; count += 1 }
                        if count > 0 { phase.values[j * w + i] = sum / count; next[j * w + i] = true }
                    }
                }
                valid = next
            }
            // Smoothed over ~±1 cell among the reached cells: first-order marching leaves kinks
            // at the cells' scale (the crests would zigzag), which this rounds off; linear runs stay.
            for _ in 0..<smoothing {
                let src = phase.values
                for j in 1..<(ht - 1) {
                    for i in 1..<(w - 1) where valid[j * w + i] {
                        var sum: Float = 0, all = true
                        for y in (j - 1)...(j + 1) {
                            for x in (i - 1)...(i + 1) {
                                all = all && valid[y * w + x]
                                sum += src[y * w + x]
                            }
                        }
                        // A cell missing neighbours on one side would be pulled off its ramp.
                        if all { phase.values[j * w + i] = sum / 9 }
                    }
                }
            }
            for k in 0..<n where !valid[k] { phase.values[k] = 0 }
            var hs = f.height
            for k in 0..<n where !hs.values[k].isFinite { hs.values[k] = 0 }
            hs = ShoreDistance.smoothed(hs, passes: 1)
            lock.lock(); channels[s] = (phase.values, hs.values); lock.unlock()
        }
        var out = [SIMD4<Float>](repeating: .zero, count: n)
        guard let trade = channels[0], let swell = channels[1] else { return out }
        var depth = height
        for k in 0..<n { depth.values[k] = Swift.max(seaLevel - height.values[k], 0) }
        depth = ShoreDistance.smoothed(depth, passes: 1)
        for k in 0..<n {
            let heights = UInt32(Float16(trade.height[k]).bitPattern) | UInt32(Float16(swell.height[k]).bitPattern) << 16
            out[k] = SIMD4(trade.phase[k], swell.phase[k], Float(bitPattern: heights), depth.values[k])
        }
        return out
    }
}
