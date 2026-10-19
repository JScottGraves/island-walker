import Foundation

/// The sea state the ocean surface is synthesised from (research/ocean.md): a trade-wind
/// sea and one swell, each a JONSWAP spectrum with directional spreading (Horvath 2015,
/// "Empirical directional wave spectra for computer graphics"), split over four FFT
/// cascades of non-commensurate patch sizes so no repeat lines up.
///
/// Conventions: x east, y north; directions are where the waves travel towards (radians
/// from +x, counter-clockwise), like `TerrainRenderer.windDirection`.
public struct OceanSpectrum: Sendable {
    public static let gravity = 9.81
    public static let cascadeCount = 4
    /// FFT size per cascade.
    public static let resolution = 256
    /// Every mode's frequency is a whole multiple of 2π / this (s), so the sea loops
    /// exactly and the phase ω·t stays small enough for single precision.
    public static let repeatPeriod = 1000.0

    /// One wave system: a JONSWAP spectrum S(ω) times a cos^2s(θ/2) spreading.
    public struct System: Sendable {
        public var alpha: Double
        public var peakOmega: Double
        public var gamma: Double
        /// Direction the waves travel towards.
        public var direction: Double
        /// Fixed spreading exponent, or nil for Hasselmann's frequency-dependent one.
        public var spread: Double?
        /// Wind speed (m/s) for wind seas (Hasselmann spreading), 0 for swell.
        public var windSpeed: Double
        /// 4√m0 when the system was scaled to a height (swell).
        public var significantHeight: Double

        /// S(ω) (m²·s), Horvath 2015 eq. 28 with Hasselmann's σ.
        public func density(omega w: Double) -> Double {
            guard w > 0 else { return 0 }
            let g = OceanSpectrum.gravity
            let sigma = w <= peakOmega ? 0.07 : 0.09
            let r = exp(-(w - peakOmega) * (w - peakOmega) / (2 * sigma * sigma * peakOmega * peakOmega))
            let ratio = peakOmega / w
            return alpha * g * g / pow(w, 5) * exp(-1.25 * ratio * ratio * ratio * ratio) * pow(gamma, r)
        }

        /// Spreading exponent at ω: Hasselmann et al. 1980 (Horvath eq. 43) for a wind sea.
        public func spreadExponent(omega w: Double) -> Double {
            if let spread { return spread }
            let g = OceanSpectrum.gravity
            let x = w / peakOmega
            if x <= 1 { return 6.97 * pow(x, 4.06) }
            let mu = -2.33 - 1.45 * (windSpeed * peakOmega / g - 1.17)
            return 9.77 * pow(x, mu)
        }

        /// 4√m0, from the spectrum itself.
        public var measuredSignificantHeight: Double {
            var m0 = 0.0
            let dw = 0.0005
            var w = 0.02
            while w < 30 {
                m0 += density(omega: w) * dw
                w += dw
            }
            return 4 * m0.squareRoot()
        }

        /// A fetch-limited trade-wind sea (Horvath eqs. 29–30).
        public static func windSea(speed u: Double, fetch f: Double, towards direction: Double) -> System {
            let g = OceanSpectrum.gravity
            let alpha = 0.076 * pow(u * u / (f * g), 0.22)
            let peak = 22 * pow(g * g / (u * f), 1.0 / 3)
            return System(alpha: alpha, peakOmega: peak, gamma: 3.3, direction: direction, spread: nil,
                          windSpeed: u, significantHeight: 0)
        }

        /// A swell of significant height `hs` (m) and peak period `period` (s), its α set so
        /// that 4√m0 = hs; narrow spreading.
        public static func swell(height hs: Double, period: Double, towards direction: Double, spread: Double = 60) -> System {
            var s = System(alpha: 1, peakOmega: 2 * Double.pi / period, gamma: 5, direction: direction, spread: spread,
                           windSpeed: 0, significantHeight: hs)
            var m0 = 0.0
            let dw = 0.0005
            var w = 0.02
            while w < 6 {
                m0 += s.density(omega: w) * dw
                w += dw
            }
            s.alpha = hs * hs / 16 / m0
            return s
        }
    }

    public var windSea: System
    public var swell: System
    /// Patch side (m) per cascade: non-commensurate, coarsest first.
    public var patchSizes: [Double]
    /// Horizontal displacement scale (Tessendorf's λ): sharper crests, flatter troughs.
    public var choppiness: Double

    /// Defaults: ~9 m/s trades blowing towards `windTowards` (the renderer's and the
    /// climate's NE trade: towards 5π/4) over 80 km (Hs ≈ 1.8 m, about fully developed for 9 m/s), and a 1.2 m, 13 s swell from the NNW.
    public init(windSpeed: Double = 9, fetch: Double = 80_000, windTowards: Double = 5 * Double.pi / 4,
                swellHeight: Double = 1.2, swellPeriod: Double = 13,
                swellTowards: Double = -Double.pi / 2 + 22.5 * Double.pi / 180,
                patchSizes: [Double] = [1103, 251, 57.3, 13.1], choppiness: Double = 0.85) {
        windSea = .windSea(speed: windSpeed, fetch: fetch, towards: windTowards)
        swell = .swell(height: swellHeight, period: swellPeriod, towards: swellTowards)
        self.patchSizes = patchSizes
        self.choppiness = choppiness
    }

    /// Normalised cos^2s(θ/2) spreading over θ ∈ (−π, π] (Longuet-Higgins et al.).
    public static func spreading(theta: Double, s: Double) -> Double {
        let s = Swift.max(s, 1e-3)
        let logQ = (2 * s - 1) * log(2) - log(Double.pi) + 2 * lgamma(s + 1) - lgamma(2 * s + 1)
        let c = abs(cos(theta / 2))
        return c <= 0 ? 0 : exp(logQ + 2 * s * log(c))
    }

    /// Kitaigorodskii's depth factor φ(ω_h) of the TMA spectrum, ω_h = ω√(h/g).
    public static func tma(omegaH w: Double) -> Double {
        if w <= 1 { return 0.5 * w * w }
        if w < 2 { return 1 - 0.5 * (2 - w) * (2 - w) }
        return 1
    }

    /// Amplitude factor √φ for a deep-water wavenumber k at `depth` metres (ω_h = √(k h)).
    public static func depthAttenuation(k: Double, depth: Double) -> Double {
        tma(omegaH: (k * Swift.max(depth, 0)).squareRoot()).squareRoot()
    }

    /// Wavenumbers cascade `i` carries: from where the previous one stops to 6 fundamentals
    /// of the next one (each band well resolved by its own patch).
    public func band(cascade i: Int) -> Range<Double> {
        func cut(_ c: Int) -> Double { 2 * Double.pi / patchSizes[c + 1] * 6 }
        let lo = i == 0 ? 0 : cut(i - 1)
        let hi = i == OceanSpectrum.cascadeCount - 1
            ? Double.pi * Double(OceanSpectrum.resolution) / patchSizes[i]
            : cut(i)
        return lo..<hi
    }

    /// Directional height density E(kx, ky) (m⁴) of both systems, deep-water dispersion:
    /// E(k, θ) = S(ω) D(θ) (dω/dk) / k.
    public func density(kx: Double, ky: Double) -> Double {
        let k = (kx * kx + ky * ky).squareRoot()
        guard k > 0 else { return 0 }
        let g = OceanSpectrum.gravity
        let w = (g * k).squareRoot()
        let jacobian = g / (2 * w) / k
        let theta = atan2(ky, kx)
        var e = 0.0
        for s in [windSea, swell] {
            var d = theta - s.direction
            d = atan2(sin(d), cos(d))
            e += s.density(omega: w) * OceanSpectrum.spreading(theta: d, s: s.spreadExponent(omega: w))
        }
        return e * jacobian
    }

    /// One cascade's initial amplitudes h0 on its size² grid of wavenumbers
    /// k = (n − size/2, m − size/2)·2π/L, and their quantised frequencies.
    public struct Modes: Sendable {
        /// (h0(k).re, h0(k).im, h0(−k).re, h0(−k).im), row m, column n.
        public var amplitudes: [SIMD4<Float>]
        public var omegas: [Float]
        /// Σ 2|A|²: the height variance these modes add, in expectation.
        public var expectedVariance: Double
        /// Expected slope variance along x, along y, and their covariance.
        public var slopeXX: Double, slopeYY: Double, slopeXY: Double
        /// Mean wavenumber weighted by height variance, for depth attenuation.
        public var meanK: Double
    }

    public func modes(cascade c: Int, seed: UInt64, size: Int = OceanSpectrum.resolution) -> Modes {
        let L = patchSizes[c]
        let dk = 2 * Double.pi / L
        let band = band(cascade: c)
        var rng = SplitMix64(seed: seed &+ UInt64(c) &* 0x9E37_79B9_7F4A_7C15)
        var h0 = [SIMD2<Double>](repeating: .zero, count: size * size)
        var omegas = [Float](repeating: 0, count: size * size)
        var variance = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0, kSum = 0.0
        let base = 2 * Double.pi / OceanSpectrum.repeatPeriod
        for m in 0..<size {
            for n in 0..<size {
                let kx = Double(n - size / 2) * dk, ky = Double(m - size / 2) * dk
                let k = (kx * kx + ky * ky).squareRoot()
                // Gaussian draws for every cell, so the field doesn't change with the band.
                let g1 = rng.gaussian(), g2 = rng.gaussian()
                omegas[m * size + n] = Float(((OceanSpectrum.gravity * k).squareRoot() / base).rounded(.down) * base)
                guard band.contains(k), k > 0 else { continue }
                // A mode 2Re[h0 e^{i(k·x − ωt)}] has variance 2|A|²: set it to E Δk².
                let a2 = density(kx: kx, ky: ky) * dk * dk / 2
                let a = a2.squareRoot()
                h0[m * size + n] = SIMD2(g1, g2) * (a / 2.0.squareRoot())
                variance += 2 * a2
                sxx += 2 * a2 * kx * kx
                syy += 2 * a2 * ky * ky
                sxy += 2 * a2 * kx * ky
                kSum += 2 * a2 * k
            }
        }
        var amplitudes = [SIMD4<Float>](repeating: .zero, count: size * size)
        for m in 0..<size {
            for n in 0..<size {
                let a = h0[m * size + n], b = h0[((size - m) % size) * size + (size - n) % size]
                amplitudes[m * size + n] = SIMD4(Float(a.x), Float(a.y), Float(b.x), Float(b.y))
            }
        }
        return Modes(amplitudes: amplitudes, omegas: omegas, expectedVariance: variance,
                     slopeXX: sxx, slopeYY: syy, slopeXY: sxy, meanK: variance > 0 ? kSum / variance : 0)
    }

    /// Cox & Munk 1954 clean-surface slope variances up- and crosswind at wind speed U
    /// (m/s at 12.5 m): σu² = 0.00316 U, σc² = 0.003 + 0.00192 U.
    public static func coxMunk(windSpeed u: Double) -> (upwind: Double, crosswind: Double) {
        (0.00316 * u, 0.003 + 0.00192 * u)
    }

    /// Slope variance the cascades resolve, up- and crosswind (expected, all cascades).
    public var resolvedSlopeVariance: (upwind: Double, crosswind: Double) {
        let wx = cos(windSea.direction), wy = sin(windSea.direction)
        var up = 0.0, cross = 0.0
        for c in 0..<OceanSpectrum.cascadeCount {
            let m = modes(cascade: c, seed: 1, size: 64)
            up += m.slopeXX * wx * wx + 2 * m.slopeXY * wx * wy + m.slopeYY * wy * wy
            cross += m.slopeXX * wy * wy - 2 * m.slopeXY * wx * wy + m.slopeYY * wx * wx
        }
        return (up, cross)
    }

    /// What Cox & Munk measured beyond what the cascades resolve: capillary waves shorter
    /// than the finest cascade's texel, which only roughen the glint (at least a little).
    public var unresolvedSlopeVariance: (upwind: Double, crosswind: Double) {
        let total = OceanSpectrum.coxMunk(windSpeed: windSea.windSpeed)
        let resolved = resolvedSlopeVariance
        return (Swift.max(total.upwind - resolved.upwind, 1e-4), Swift.max(total.crosswind - resolved.crosswind, 1e-4))
    }
}

extension SplitMix64 {
    /// A uniform draw in (0, 1), then a standard normal one (Box–Muller).
    mutating func unitOpen() -> Double { (Double(next() >> 11) + 0.5) / Double(1 << 53) }
    mutating func gaussian() -> Double { (-2 * log(unitOpen())).squareRoot() * cos(2 * Double.pi * unitOpen()) }
}

/// CDLOD node selection (as TerrainQuadtree) for the sea: a flat square `reach` metres
/// either side of `centre`, out past the far plane so the water meets the horizon. Each
/// level's grid stays put in the world (no swimming), and each ring's far edge morphs onto
/// the next level's grid.
public struct OceanQuadtree: Sendable {
    public typealias Node = TerrainQuadtree.Node
    public let quads: Int
    public let cellMetres: Double
    public let lodCount: Int
    public let ranges: [Double]
    public let morphStart: [Double]
    public let morphEnd: [Double]
    /// South-west corner and side of the root node.
    public let origin: (x: Double, y: Double)
    public let side: Double
    /// Highest crest above (and deepest trough below) the sea level, for node bounds.
    public var amplitude: Double = 4

    public init(quads: Int = 32, cellMetres: Double, centre: (x: Double, y: Double), reach: Double, rangeFactor: Double = 2.5) {
        self.quads = quads
        self.cellMetres = cellMetres
        let leaf = Double(quads) * cellMetres
        var count = 1
        while leaf * Double(1 << (count - 1)) < 2 * reach { count += 1 }
        lodCount = count
        side = leaf * Double(1 << (count - 1))
        origin = (centre.x - side / 2, centre.y - side / 2)
        var r = (0..<count).map { rangeFactor * leaf * Double(1 << $0) }
        r[count - 1] = .greatestFiniteMagnitude
        ranges = r
        morphEnd = r
        morphStart = (0..<count).map { k in
            let previous = k == 0 ? 0 : r[k - 1]
            return k == count - 1 ? .greatestFiniteMagnitude : previous + (r[k] - previous) * 0.7
        }
    }

    /// Nodes to draw for a camera at `eye` (heights relative to the sea level).
    public func select(eye: (x: Double, y: Double, z: Double),
                       visible: (_ min: SIMD3<Double>, _ max: SIMD3<Double>) -> Bool) -> [Node] {
        var out: [Node] = []
        let e = SIMD3(eye.x, eye.y, eye.z)
        func box(_ x: Double, _ y: Double, _ size: Double) -> (SIMD3<Double>, SIMD3<Double>) {
            (SIMD3(x, y, -amplitude), SIMD3(x + size, y + size, amplitude))
        }
        func reaches(_ range: Double, _ b: (SIMD3<Double>, SIMD3<Double>)) -> Bool {
            if range == .greatestFiniteMagnitude { return true }
            let d = simdClamp(e, b.0, b.1) - e
            return (d * d).sum() <= range * range
        }
        func visit(_ x: Double, _ y: Double, _ lod: Int) -> Bool {
            let size = Double(quads) * cellMetres * Double(1 << lod)
            let b = box(x, y, size)
            if !reaches(ranges[lod], b) { return false }
            if !visible(b.0, b.1) { return true }
            if lod == 0 || !reaches(ranges[lod - 1], b) {
                out.append(Node(x: x, y: y, size: size, lod: lod, half: false))
                return true
            }
            let h = size / 2
            for (cx, cy) in [(x, y), (x + h, y), (x, y + h), (x + h, y + h)] where !visit(cx, cy, lod - 1) {
                let cb = box(cx, cy, h)
                if visible(cb.0, cb.1) { out.append(Node(x: cx, y: cy, size: h, lod: lod, half: true)) }
            }
            return true
        }
        _ = visit(origin.x, origin.y, lodCount - 1)
        return out
    }
}
