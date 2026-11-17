import Foundation

/// How much of the open sea's waves each patch of water gets (research/waves.md, item 2): the
/// FFT sea is one global spectrum, so without this the leeward coast, Kāneʻohe Bay behind its
/// reef and the island's swell shadow are as rough as the open windward sea.
///
/// Each of the spectrum's three systems is localised on its own:
/// - **Wind sea** (the local trade chop): fetch-limited by the water upwind (land or a reef flat
///   upwind stops it; it regrows from there, Hs ∝ U√F), and the wind itself slackens in the lee
///   of high land. Its spectrum is re-evaluated at the local fetch and wind (JONSWAP, as
///   OceanSpectrum.windSea), so a short fetch keeps short ripples and loses the long chop.
/// - **Trade swell** and **north swell**: their refracted, shoaled and reef-broken height from
///   WaveRefraction, over what open-sea waves would have at that depth (deep-water height times
///   the shoaling coefficient, held to the same depth cap): 1 on an open coast, small behind a
///   reef or in the island's shadow, 0 in water neither reaches.
///
/// Then every FFT cascade gets the amplitude factor of the mix of systems it carries
/// (√ of its variance here over its open-sea variance), and the unresolved capillary roughness
/// (Cox–Munk, ∝ wind) the wind's. That is how Crest and Atlas localise a global FFT.
public enum SeaShelter {
    public struct Settings: Sendable {
        /// Most the wind slackens right behind high land (0…1).
        public var lee: Double = 0.6
        /// Land this high (m) or more makes the full lee.
        public var leeHeight: Double = 800
        /// The lee recovers over this many times the land's height downwind.
        public var leeLength: Double = 12
        /// Water shallower than this (m) breaks the wind sea: its fetch starts again past it.
        public var reefDepth: Double = 1
        /// Cell size (m) the wind's fetch is marched on.
        public var windCell: Double = 150
        public init() {}
    }

    /// Per cell of `height`: each cascade's amplitude factor (0…1) and the capillary roughness'
    /// variance factor. `swellHeights`: the trade swell's (with the wind sea) and the north
    /// swell's local significant heights from the refraction bake, on its grid (`swellGrid`
    /// cells, maybe finer than `height`), with the depths it used; nil leaves the swells open.
    public static func factors(height: Grid, seaLevel: Float, spectrum: OceanSpectrum,
                               swells: (trade: Grid, north: Grid, depth: Grid)?,
                               tradeHeight: Double, settings: Settings = Settings()) -> (cascades: [SIMD4<Float>], calm: Grid) {
        let w = height.width, h = height.height, n = w * h
        let wind = SeaShelter.wind(height: height, seaLevel: seaLevel, towards: spectrum.windSea.direction,
                                   maxFetch: spectrum.windFetch, settings: settings)
        // Open-sea variance of each system in each cascade.
        let systems = [spectrum.windSea, spectrum.tradeSwell, spectrum.swell]
        var open = [[Double]](repeating: [Double](repeating: 0, count: OceanSpectrum.cascadeCount), count: 3)
        for (s, system) in systems.enumerated() {
            for c in 0..<OceanSpectrum.cascadeCount { open[s][c] = spectrum.bandVariance(system, cascade: c) }
        }
        let table = WindTable(spectrum: spectrum)
        // Swell exposures, at this grid's cells.
        var trade = Grid(width: w, height: h, cellSize: height.cellSize, fill: 1)
        var north = trade
        if let swells {
            let periods = (2 * Double.pi / spectrum.tradeSwell.peakOmega, 2 * Double.pi / spectrum.swell.peakOmega)
            let sw = swells.trade.width, sh = swells.trade.height
            var et = Grid(width: sw, height: sh, cellSize: swells.trade.cellSize, fill: 1)
            var en = Grid(width: sw, height: sh, cellSize: swells.trade.cellSize, fill: 1)
            let northOpen = spectrum.swell.significantHeight
            et.values.withUnsafeMutableBufferPointer { tb in
                en.values.withUnsafeMutableBufferPointer { nb in
                    let t = SharedBuffer(tb), nn = SharedBuffer(nb)
                    Parallel.forRanges(sh) { rows in
                        for j in rows {
                            for i in 0..<sw {
                                let k = j * sw + i
                                let d = Double(swells.depth.values[k])
                                t[k] = exposure(Double(swells.trade.values[k]), open: tradeHeight, depth: d, period: periods.0)
                                nn[k] = exposure(Double(swells.north.values[k]), open: northOpen, depth: d, period: periods.1)
                            }
                        }
                    }
                }
            }
            trade = resample(et, to: height)
            north = resample(en, to: height)
        }
        var out = [SIMD4<Float>](repeating: SIMD4(repeating: 1), count: n)
        var calm = Grid(width: w, height: h, cellSize: height.cellSize, fill: 1)
        let u0 = spectrum.windSea.windSpeed
        let cm0 = OceanSpectrum.coxMunk(windSpeed: u0)
        let tradeE = trade, northE = north
        out.withUnsafeMutableBufferPointer { ob in
            calm.values.withUnsafeMutableBufferPointer { cb in
                let o = SharedBuffer(ob), cv = SharedBuffer(cb)
                Parallel.forRanges(h) { rows in
                    for j in rows {
                        for i in 0..<w {
                            let k = j * w + i
                            let ws = table.ratios(fetch: Double(wind.fetch.values[k]), wind: Double(wind.speed.values[k]))
                            let et = Double(tradeE.values[k]), en = Double(northE.values[k])
                            var f = SIMD4<Float>(repeating: 1)
                            for c in 0..<OceanSpectrum.cascadeCount {
                                let total = open[0][c] + open[1][c] + open[2][c]
                                guard total > 0 else { continue }
                                let here = open[0][c] * ws[c] + open[1][c] * et * et + open[2][c] * en * en
                                f[c] = Float((here / total).squareRoot())
                            }
                            o[k] = f
                            let cm = OceanSpectrum.coxMunk(windSpeed: u0 * Double(wind.speed.values[k]))
                            cv[k] = Float((cm.upwind + cm.crosswind) / (cm0.upwind + cm0.crosswind))
                        }
                    }
                }
            }
        }
        // Rounded over ~±1 cell: the swell bake's and the fetch's cell edges.
        for c in 0..<4 {
            var g = Grid(width: w, height: h, cellSize: height.cellSize)
            for k in 0..<n { g.values[k] = out[k][c] }
            g = ShoreDistance.smoothed(g, passes: 2)
            for k in 0..<n { out[k][c] = g.values[k] }
        }
        calm = ShoreDistance.smoothed(calm, passes: 2)
        return (out, calm)
    }

    /// A swell's local height `local` over the open sea's at `depth`: the deep-water height
    /// `open` shoaled (Ks = √(Cg∞/Cg)) and held to the refraction's depth cap. 1 where the water
    /// is too shallow to tell (the cap rules both), 0 where the swell never arrives.
    static func exposure(_ local: Double, open: Double, depth: Double, period: Double) -> Float {
        let cap = WaveRefraction.breakerIndex * Swift.max(depth, WaveRefraction.capDepth)
        if depth <= 0.3 || open <= 0 { return 1 }
        let deep = 9.81 * period / (4 * Double.pi)
        let ks = (deep / WaveRefraction.groupSpeed(depth: Swift.max(depth, WaveRefraction.minDepth), period: period)).squareRoot()
        let reference = Swift.min(open * ks, cap)
        guard reference > 1e-3 else { return 1 }
        return Float(Swift.min(Swift.max(local / reference, 0), 1))
    }

    /// The wind sea's fetch (m of water upwind) and the wind's speed factor (0…1, the lee of
    /// high land) per cell of `height`, marched on a ~`settings.windCell` grid over a fan of
    /// directions ±24° around the wind (gusts and the trades' wander soften the shadows' edges).
    /// Off the grid is open sea: the full fetch.
    public static func wind(height: Grid, seaLevel: Float, towards: Double, maxFetch: Double,
                            settings: Settings = Settings()) -> (fetch: Grid, speed: Grid) {
        let factor = Swift.max(1, Int((settings.windCell / height.cellSize).rounded()))
        let cw = (height.width + factor - 1) / factor, ch = (height.height + factor - 1) / factor
        let cell = height.cellSize * Double(factor)
        // Coarse cells: whether any part stops the wind sea, and the highest land in them.
        var barrier = [Bool](repeating: false, count: cw * ch)
        var land = [Float](repeating: 0, count: cw * ch)
        for j in 0..<height.height {
            for i in 0..<height.width {
                let k = (j / factor) * cw + i / factor
                let z = height[i, j] - seaLevel
                if Double(-z) < settings.reefDepth { barrier[k] = true }
                land[k] = Swift.max(land[k], z)
            }
        }
        let fan: [(Double, Double)] = [-24.0, -12, 0, 12, 24].map { a in
            (a * Double.pi / 180, exp(-0.5 * (a / 15) * (a / 15)))
        }
        let weightSum = fan.reduce(0) { $0 + $1.1 }
        let leeReach = settings.leeLength * 1500
        let maxSteps = Int((Swift.max(maxFetch, leeReach) / cell).rounded(.up))
        var fetch = Grid(width: cw, height: ch, cellSize: cell)
        var speed = Grid(width: cw, height: ch, cellSize: cell, fill: 1)
        let barriers = barrier, heights = land
        fetch.values.withUnsafeMutableBufferPointer { fb in
        speed.values.withUnsafeMutableBufferPointer { sb in
        let fv = SharedBuffer(fb), sv = SharedBuffer(sb)
        Parallel.forRanges(ch) { rows in
            for j in rows {
                for i in 0..<cw {
                    let k = j * cw + i
                    if barriers[k] && heights[k] > 0 { continue }   // land: no sea (fetch 0, wind as is)
                    var f = 0.0, s = 0.0
                    for (offset, weight) in fan {
                        // Upwind: against the direction the wind blows towards.
                        let a = towards + Double.pi + offset
                        let dx = cos(a), dy = sin(a)
                        var reach = maxFetch, shadow = 0.0, found = false
                        for step in 1...maxSteps {
                            let d = Double(step) * cell
                            let x = (Double(i) + 0.5) + dx * Double(step), y = (Double(j) + 0.5) + dy * Double(step)
                            let ci = Int(x.rounded(.down)), cj = Int(y.rounded(.down))
                            if ci < 0 || cj < 0 || ci >= cw || cj >= ch { break }
                            let q = cj * cw + ci
                            if barriers[q] && !found {
                                reach = Swift.min(d - 0.5 * cell, maxFetch)
                                found = true
                            }
                            let z = Double(heights[q])
                            if z > 0 {
                                let full = Swift.min(z / settings.leeHeight, 1)
                                shadow = Swift.max(shadow, full * exp(-d / (settings.leeLength * Swift.max(z, 1))))
                            }
                            if found && d > leeReach { break }
                        }
                        f += weight * reach
                        s += weight * (1 - settings.lee * shadow)
                    }
                    fv[k] = Float(f / weightSum)
                    sv[k] = Float(s / weightSum)
                }
            }
        }
        }
        }
        // Back to the height's cells, bilinear (coarse cell centres).
        var outFetch = Grid(width: height.width, height: height.height, cellSize: height.cellSize)
        var outSpeed = Grid(width: height.width, height: height.height, cellSize: height.cellSize, fill: 1)
        for j in 0..<height.height {
            for i in 0..<height.width {
                let x = (Double(i) + 0.5) / Double(factor) - 0.5, y = (Double(j) + 0.5) / Double(factor) - 0.5
                outFetch[i, j] = fetch.sample(x: x, y: y)
                outSpeed[i, j] = speed.sample(x: x, y: y)
            }
        }
        return (outFetch, outSpeed)
    }

    /// `g` on `target`'s cells (same extent): box mean where `g` is a whole multiple finer,
    /// bilinear otherwise.
    static func resample(_ g: Grid, to target: Grid) -> Grid {
        if g.width == target.width && g.height == target.height { return g }
        var out = Grid(width: target.width, height: target.height, cellSize: target.cellSize)
        if g.width % target.width == 0 && g.height % target.height == 0 && g.width / target.width == g.height / target.height {
            let r = g.width / target.width
            for j in 0..<target.height {
                for i in 0..<target.width {
                    var sum: Float = 0
                    for y in (j * r)..<(j * r + r) { for x in (i * r)..<(i * r + r) { sum += g[x, y] } }
                    out[i, j] = sum / Float(r * r)
                }
            }
            return out
        }
        let sx = Double(g.width) / Double(target.width), sy = Double(g.height) / Double(target.height)
        for j in 0..<target.height {
            for i in 0..<target.width {
                out[i, j] = g.sample(x: (Double(i) + 0.5) * sx - 0.5, y: (Double(j) + 0.5) * sy - 0.5)
            }
        }
        return out
    }

    /// The wind sea's variance in each cascade at a fetch and wind speed factor, over its
    /// open-sea variance: tabulated (log fetch × wind) and read bilinearly.
    struct WindTable {
        static let fetches = 16, winds = 9
        let minFetch = 100.0, maxFetch: Double
        let minWind = 0.2
        var values: [SIMD4<Double>]

        init(spectrum: OceanSpectrum) {
            maxFetch = spectrum.windFetch
            let u0 = spectrum.windSea.windSpeed
            var open = SIMD4<Double>(repeating: 0)
            for c in 0..<4 { open[c] = spectrum.bandVariance(spectrum.windSea, cascade: c) }
            var v = [SIMD4<Double>](repeating: SIMD4(repeating: 1), count: WindTable.fetches * WindTable.winds)
            for a in 0..<WindTable.winds {
                let u = minWind + (1 - minWind) * Double(a) / Double(WindTable.winds - 1)
                for b in 0..<WindTable.fetches {
                    let f = minFetch * pow(maxFetch / minFetch, Double(b) / Double(WindTable.fetches - 1))
                    let sea = OceanSpectrum.System.windSea(speed: u0 * u, fetch: f, towards: spectrum.windSea.direction)
                    var r = SIMD4<Double>(repeating: 1)
                    for c in 0..<4 where open[c] > 0 {
                        r[c] = Swift.min(spectrum.bandVariance(sea, cascade: c) / open[c], 1)
                    }
                    v[a * WindTable.fetches + b] = r
                }
            }
            values = v
        }

        func ratios(fetch: Double, wind: Double) -> SIMD4<Double> {
            if fetch <= 1 { return .zero }
            let fb = (log(Swift.max(fetch, minFetch) / minFetch) / log(maxFetch / minFetch)) * Double(WindTable.fetches - 1)
            let wa = (Swift.min(Swift.max(wind, minWind), 1) - minWind) / (1 - minWind) * Double(WindTable.winds - 1)
            let b0 = Swift.min(Int(fb), WindTable.fetches - 2), a0 = Swift.min(Int(wa), WindTable.winds - 2)
            let tb = Swift.min(fb - Double(b0), 1), ta = wa - Double(a0)
            func at(_ a: Int, _ b: Int) -> SIMD4<Double> { values[a * WindTable.fetches + b] }
            let lo = at(a0, b0) * (1 - tb) + at(a0, b0 + 1) * tb
            let hi = at(a0 + 1, b0) * (1 - tb) + at(a0 + 1, b0 + 1) * tb
            var r = lo * (1 - ta) + hi * ta
            // Under the table's shortest fetch the chop fades out with it.
            if fetch < minFetch { r *= fetch / minFetch }
            return r
        }
    }
}

extension OceanSpectrum {
    /// The local fetch (m) the wind sea was grown over, from its peak (inverting Horvath eq. 30).
    public var windFetch: Double {
        let g = OceanSpectrum.gravity, u = windSea.windSpeed
        return pow(22 / windSea.peakOmega, 3) * g * g / u
    }

    /// Height variance (m²) `system` puts in cascade `c`'s band of wavenumbers (deep water:
    /// ω = √(gk)), by quadrature of S(ω) over it; the spreading integrates to one.
    public func bandVariance(_ system: System, cascade c: Int) -> Double {
        let band = band(cascade: c)
        let g = OceanSpectrum.gravity
        let w0 = Swift.max((g * band.lowerBound).squareRoot(), 0.02), w1 = (g * band.upperBound).squareRoot()
        guard w1 > w0 else { return 0 }
        // Log-spaced: the bands span decades in ω.
        let steps = 600
        let r = log(w1 / w0)
        var sum = 0.0
        for i in 0..<steps {
            let w = w0 * exp(r * (Double(i) + 0.5) / Double(steps))
            sum += system.density(omega: w) * w * r / Double(steps)
        }
        return sum
    }
}
