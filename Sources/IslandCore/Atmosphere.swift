import Foundation

/// Physically based clear sky after Hillaire 2020, "A Scalable and Production Ready Sky
/// and Atmosphere Rendering Technique" (research/global-illumination.md §5): Rayleigh,
/// Mie and ozone over a spherical Earth, with multiple scattering from a small LUT. Runs
/// on the CPU: the two sun-independent LUTs are built once, and the sky-view LUT (sky
/// radiance in every direction from just above sea level) is rebuilt when the sun moves.
///
/// Units: the sun's irradiance above the atmosphere is 1 in every channel. `SunPath`
/// rescales everything so the reference sun matches `Look.sun`.
public enum Atmosphere {
    static let groundRadius = 6_360_000.0
    static let topRadius = 6_460_000.0
    /// Height of the eye the sky-view LUT is computed for (a walker on the island).
    public static let viewHeight = 300.0

    static let rayleighScattering = RGB(5.802e-6, 13.558e-6, 33.1e-6)
    static let rayleighHeight = 8000.0
    /// Mie aerosol: 6× Hillaire's (very clean) default, an aerosol optical depth of
    /// ~0.03 like clear trade-wind air with some sea salt. His default left sky light on
    /// open ground under 10% of the sun's; 15× (AOD 0.08) gave the right sky light but a
    /// white aureole tens of degrees around the sun, more like a hazy or cirrus day.
    static let mieScattering = 6 * 3.996e-6
    static let mieExtinction = 6 * 4.440e-6
    static let mieHeight = 1200.0
    /// Sea-salt aerosol scatters less sharply forward than continental haze.
    static let mieG = 0.75
    static let ozoneAbsorption = RGB(0.650e-6, 1.881e-6, 0.085e-6)
    /// Ocean around the island, for light the sea bounces back into the sky.
    static let groundAlbedo = 0.1

    struct Medium {
        var rayleigh: RGB, mie: Double, extinction: RGB
    }

    static func medium(height h: Double) -> Medium {
        let r = exp(-Swift.max(0, h) / rayleighHeight)
        let m = exp(-Swift.max(0, h) / mieHeight)
        let o = Swift.max(0, 1 - abs(h - 25_000) / 15_000)
        let rayleigh = rayleighScattering * r
        let extinction = rayleigh + RGB(1, 1, 1) * (mieExtinction * m) + ozoneAbsorption * o
        return Medium(rayleigh: rayleigh, mie: mieScattering * m, extinction: extinction)
    }

    static func exp3(_ c: RGB) -> RGB { RGB(exp(-c.r), exp(-c.g), exp(-c.b)) }

    /// Distance along a ray from radius r with zenith cosine mu to the sphere of radius R,
    /// or nil if it misses (outward-facing hit only).
    static func distanceToSphere(r: Double, mu: Double, radius: Double) -> Double? {
        let d = r * r * (mu * mu - 1) + radius * radius
        if d < 0 { return nil }
        let t = -r * mu + d.squareRoot()
        return t >= 0 ? t : nil
    }

    /// Whether a ray from radius r with zenith cosine mu hits the ground.
    static func hitsGround(r: Double, mu: Double) -> Double? {
        guard mu < 0 else { return nil }
        let d = r * r * (mu * mu - 1) + groundRadius * groundRadius
        if d < 0 { return nil }
        let t = -r * mu - d.squareRoot()
        return t >= 0 ? t : nil
    }

    // MARK: Transmittance LUT (height × zenith cosine), to the top of the atmosphere.

    static let transmittanceSize = (mu: 128, h: 48)

    static let transmittanceLUT: [RGB] = {
        var lut = [RGB](repeating: RGB(0, 0, 0), count: transmittanceSize.mu * transmittanceSize.h)
        for j in 0..<transmittanceSize.h {
            let h = heightFor(v: (Double(j) + 0.5) / Double(transmittanceSize.h))
            for i in 0..<transmittanceSize.mu {
                let mu = -1 + 2 * (Double(i) + 0.5) / Double(transmittanceSize.mu)
                lut[j * transmittanceSize.mu + i] = computeTransmittance(height: h, mu: mu)
            }
        }
        return lut
    }()

    static func heightFor(v: Double) -> Double { v * v * (topRadius - groundRadius) }

    static func computeTransmittance(height h: Double, mu: Double) -> RGB {
        let r = groundRadius + h
        if hitsGround(r: r, mu: mu) != nil { return RGB(0, 0, 0) }
        guard let length = distanceToSphere(r: r, mu: mu, radius: topRadius) else { return RGB(1, 1, 1) }
        let steps = 40
        var depth = RGB(0, 0, 0)
        for k in 0..<steps {
            let t = (Double(k) + 0.5) / Double(steps) * length
            let rr = (r * r + t * t + 2 * r * mu * t).squareRoot()
            depth = depth + medium(height: rr - groundRadius).extinction * (length / Double(steps))
        }
        return exp3(depth)
    }

    /// Transmittance from a point at height h towards a direction with zenith cosine mu.
    static func transmittance(height h: Double, mu: Double) -> RGB {
        let r = groundRadius + Swift.max(0, h)
        // The ground cuts rays off sharply; the LUT would blur that over a texel.
        if hitsGround(r: r, mu: mu) != nil { return RGB(0, 0, 0) }
        let v = (Swift.max(0, h) / (topRadius - groundRadius)).squareRoot()
        return sample(transmittanceLUT, transmittanceSize.mu, transmittanceSize.h,
                      u: (mu + 1) / 2, v: v)
    }

    static func sample(_ lut: [RGB], _ w: Int, _ h: Int, u: Double, v: Double) -> RGB {
        let x = Swift.min(Swift.max(u * Double(w) - 0.5, 0), Double(w - 1))
        let y = Swift.min(Swift.max(v * Double(h) - 0.5, 0), Double(h - 1))
        let x0 = Int(x), y0 = Int(y)
        let x1 = Swift.min(x0 + 1, w - 1), y1 = Swift.min(y0 + 1, h - 1)
        let fx = x - Double(x0), fy = y - Double(y0)
        let a = lut[y0 * w + x0].mix(lut[y0 * w + x1], fx)
        let b = lut[y1 * w + x0].mix(lut[y1 * w + x1], fx)
        return a.mix(b, fy)
    }

    // MARK: Multiple scattering LUT (height × sun zenith cosine), Hillaire §5.5.

    static let multiScatteringSize = 32

    static let multiScatteringLUT: [RGB] = {
        let n = multiScatteringSize
        var lut = [RGB](repeating: RGB(0, 0, 0), count: n * n)
        // Directions spread evenly over the sphere (Fibonacci lattice).
        let directions = (0..<64).map { k -> (Double, Double, Double) in
            let z = 1 - 2 * (Double(k) + 0.5) / 64
            let a = Double(k) * 2.399963229728653
            let s = (1 - z * z).squareRoot()
            return (s * cos(a), s * sin(a), z)
        }
        lut.withUnsafeMutableBufferPointer { buf in
            let out = SharedBuffer(buf)
            Parallel.forRanges(n) { rows in
                for j in rows {
                    let h = heightFor(v: (Double(j) + 0.5) / Double(n))
                    for i in 0..<n {
                        let muS = -1 + 2 * (Double(i) + 0.5) / Double(n)
                        let sun = ((1 - muS * muS).squareRoot(), 0.0, muS)
                        var second = RGB(0, 0, 0), transfer = RGB(0, 0, 0)
                        for d in directions {
                            let (l, f) = integrateMultipleScattering(height: h, direction: d, sun: sun)
                            second = second + l
                            transfer = transfer + f
                        }
                        second = second * (1 / 64.0)
                        transfer = transfer * (1 / 64.0)
                        out[j * n + i] = RGB(second.r / (1 - transfer.r), second.g / (1 - transfer.g),
                                             second.b / (1 - transfer.b))
                    }
                }
            }
        }
        return lut
    }()

    /// Second-order light reaching a point from one direction with an isotropic phase,
    /// and the fraction of light that direction scatters back (Hillaire's L₂ and f_ms).
    static func integrateMultipleScattering(height h: Double, direction d: (Double, Double, Double),
                                            sun: (Double, Double, Double)) -> (RGB, RGB) {
        let r = groundRadius + h
        let mu = d.2
        let ground = hitsGround(r: r, mu: mu)
        guard let length = ground ?? distanceToSphere(r: r, mu: mu, radius: topRadius) else {
            return (RGB(0, 0, 0), RGB(0, 0, 0))
        }
        let steps = 20
        let dt = length / Double(steps)
        let isotropic = 1 / (4 * Double.pi)
        var throughput = RGB(1, 1, 1)
        var light = RGB(0, 0, 0), transfer = RGB(0, 0, 0)
        for k in 0..<steps {
            let t = (Double(k) + 0.5) * dt
            let p = (d.0 * t, d.1 * t, r + d.2 * t)
            let pr = (p.0 * p.0 + p.1 * p.1 + p.2 * p.2).squareRoot()
            let m = medium(height: pr - groundRadius)
            let muSun = (p.0 * sun.0 + p.1 * sun.1 + p.2 * sun.2) / pr
            let scattering = m.rayleigh + RGB(m.mie, m.mie, m.mie)
            let sunT = transmittance(height: pr - groundRadius, mu: muSun)
            let stepT = exp3(m.extinction * dt)
            // Energy-conserving integration over the step (Hillaire §5.3).
            let integral = RGB((1 - stepT.r) / m.extinction.r, (1 - stepT.g) / m.extinction.g,
                               (1 - stepT.b) / m.extinction.b)
            light = light + throughput * scattering * sunT * integral * isotropic
            transfer = transfer + throughput * scattering * integral
            throughput = throughput * stepT
        }
        if ground != nil {
            // Sunlight the ocean reflects back up (Lambertian).
            let p = (d.0 * length, d.1 * length, r + d.2 * length)
            let pr = (p.0 * p.0 + p.1 * p.1 + p.2 * p.2).squareRoot()
            let muSun = (p.0 * sun.0 + p.1 * sun.1 + p.2 * sun.2) / pr
            let sunT = transmittance(height: 0, mu: muSun)
            light = light + throughput * sunT * (Swift.max(0, muSun) * groundAlbedo / Double.pi)
        }
        return (light, transfer)
    }

    static func multipleScattering(height h: Double, muSun: Double) -> RGB {
        let v = (Swift.max(0, h) / (topRadius - groundRadius)).squareRoot()
        return sample(multiScatteringLUT, multiScatteringSize, multiScatteringSize, u: (muSun + 1) / 2, v: v)
    }

    // MARK: Sky radiance

    static func rayleighPhase(_ c: Double) -> Double { 3 / (16 * Double.pi) * (1 + c * c) }

    static func miePhase(_ c: Double) -> Double {
        let g = mieG
        return 3 / (8 * Double.pi) * (1 - g * g) * (1 + c * c) / ((2 + g * g) * pow(1 + g * g - 2 * g * c, 1.5))
    }

    /// Radiance of the sky seen from `viewHeight` along `view` (unit, z up), sun excluded.
    public static func skyRadiance(view v: (x: Double, y: Double, z: Double), sun s: (x: Double, y: Double, z: Double)) -> RGB {
        let r = groundRadius + viewHeight
        let ground = hitsGround(r: r, mu: v.z)
        guard let length = ground ?? distanceToSphere(r: r, mu: v.z, radius: topRadius) else { return RGB(0, 0, 0) }
        let cosTheta = v.x * s.x + v.y * s.y + v.z * s.z
        let pR = rayleighPhase(cosTheta), pM = miePhase(cosTheta)
        let steps = 32
        var throughput = RGB(1, 1, 1), light = RGB(0, 0, 0)
        var tPrevious = 0.0
        for k in 0..<steps {
            // Quadratic spacing: short steps near the eye, where the air is densest.
            let f = (Double(k) + 1) / Double(steps)
            let t1 = f * f * length
            let dt = t1 - tPrevious
            let t = tPrevious + 0.5 * dt
            tPrevious = t1
            let p = (v.x * t, v.y * t, r + v.z * t)
            let pr = (p.0 * p.0 + p.1 * p.1 + p.2 * p.2).squareRoot()
            let h = pr - groundRadius
            let m = medium(height: h)
            let muSun = (p.0 * s.x + p.1 * s.y + p.2 * s.z) / pr
            let sunT = transmittance(height: h, mu: muSun)
            let multi = multipleScattering(height: h, muSun: muSun)
            let scattering = m.rayleigh + RGB(m.mie, m.mie, m.mie)
            let source = (m.rayleigh * pR + RGB(m.mie, m.mie, m.mie) * pM) * sunT + scattering * multi
            let stepT = exp3(m.extinction * dt)
            let integral = RGB((1 - stepT.r) / m.extinction.r, (1 - stepT.g) / m.extinction.g,
                               (1 - stepT.b) / m.extinction.b)
            light = light + throughput * source * integral
            throughput = throughput * stepT
        }
        return light
    }

    /// Sunlight reaching `viewHeight` (irradiance on a surface facing the sun).
    public static func sunIrradiance(sinElevation: Double) -> RGB {
        transmittance(height: viewHeight, mu: sinElevation)
    }

    // MARK: Sky-view LUT

    /// Azimuth × elevation texels. Elevation is squashed towards the horizon, where the
    /// sky changes fastest: v = 0.5 + 0.5 · sign(e) · sqrt(|e| / 90°).
    public static let skyViewSize = (width: 128, height: 64)

    public static func skyViewDirection(u: Double, v: Double) -> (x: Double, y: Double, z: Double) {
        let azimuth = u * 2 * Double.pi
        let w = 2 * v - 1
        let elevation = (w < 0 ? -1.0 : 1.0) * w * w * Double.pi / 2
        return (cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))
    }

    /// Sky radiance per texel, row-major, rows from straight down to straight up.
    public static func skyView(sun: (x: Double, y: Double, z: Double)) -> [RGB] {
        let (w, h) = skyViewSize
        var lut = [RGB](repeating: RGB(0, 0, 0), count: w * h)
        lut.withUnsafeMutableBufferPointer { buf in
            let out = SharedBuffer(buf)
            Parallel.forRanges(h) { rows in
                for j in rows {
                    for i in 0..<w {
                        let d = skyViewDirection(u: (Double(i) + 0.5) / Double(w), v: (Double(j) + 0.5) / Double(h))
                        out[j * w + i] = skyRadiance(view: d, sun: sun)
                    }
                }
            }
        }
        return lut
    }

    /// Irradiance from the sky above the horizon, as 9 spherical-harmonic coefficients
    /// (Ramamoorthi & Hanrahan 2001): E(n) = Σ cᵢ Yᵢ(n). The sea and terrain below the
    /// horizon are left out; the renderer adds bounce light from the ground itself.
    public static func irradianceSH(skyView lut: [RGB]) -> [RGB] {
        let (w, h) = skyViewSize
        var sh = [RGB](repeating: RGB(0, 0, 0), count: 9)
        let dAzimuth = 2 * Double.pi / Double(w)
        for j in (h / 2)..<h {
            let v = (Double(j) + 0.5) / Double(h)
            let wv = 2 * v - 1
            // de/dv for e = w² · π/2 with w = 2v − 1.
            let dElevation = 2 * Double.pi * wv / Double(h)
            for i in 0..<w {
                let d = skyViewDirection(u: (Double(i) + 0.5) / Double(w), v: v)
                let solidAngle = (1 - d.z * d.z).squareRoot() * dElevation * dAzimuth
                let y = basis(d)
                let l = lut[j * w + i] * solidAngle
                for k in 0..<9 { sh[k] = sh[k] + l * y[k] }
            }
        }
        // Convolve with the clamped cosine.
        let a = [Double.pi, 2 * Double.pi / 3, 2 * Double.pi / 3, 2 * Double.pi / 3,
                 Double.pi / 4, Double.pi / 4, Double.pi / 4, Double.pi / 4, Double.pi / 4]
        return (0..<9).map { sh[$0] * a[$0] }
    }

    /// Real SH basis, bands 0–2, in the order the shader uses.
    public static func basis(_ d: (x: Double, y: Double, z: Double)) -> [Double] {
        [0.282095,
         0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
         1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3 * d.z * d.z - 1),
         1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y)]
    }

    public static func evaluateSH(_ sh: [RGB], _ n: (x: Double, y: Double, z: Double)) -> RGB {
        let y = basis(n)
        var e = RGB(0, 0, 0)
        for k in 0..<9 { e = e + sh[k] * y[k] }
        return e
    }
}
