import Foundation

/// The light a sun position gives: direction, sun radiance after the atmosphere, sky
/// irradiance, haze tint and exposure, all pre-exposed like `Look`'s constants.
public struct SunLight: Sendable, Equatable {
    /// Towards the sun (x east, y north, z up), unit length.
    public var direction: (x: Double, y: Double, z: Double)
    /// Sun radiance, reddened and dimmed by the air mass at this elevation.
    public var sun: RGB
    /// Sky irradiance on open horizontal ground.
    public var sky: RGB
    /// Multiplies the haze (horizon) and zenith sky colours.
    public var hazeTint: RGB
    public var zenithTint: RGB
    /// Scale applied before tone mapping: partial eye adaptation, so dawn and dusk read
    /// darker than noon without going black.
    public var exposure: Double

    public static func == (a: SunLight, b: SunLight) -> Bool {
        a.direction == b.direction && a.sun == b.sun && a.sky == b.sky
            && a.hazeTint == b.hazeTint && a.zenithTint == b.zenithTint && a.exposure == b.exposure
    }

    /// The fixed sun every render used before the sun moved; matches `Look`'s constants
    /// exactly, so renders at this sun compare one-to-one with older ones.
    public static var reference: SunLight { SunPath.light(direction: Look.sunDirection) }
}

/// Where the sun is over the island at a given hour and day, and what light that gives
/// (research/global-illumination.md §3, §5). A stand-in for the Hillaire sky model: an
/// analytic air-mass transmittance for the sun and a simple elevation curve for the sky,
/// both normalised so `Look.sunDirection` gives back `Look.sun` and `Look.sky`.
public enum SunPath {
    /// Oʻahu's latitude.
    public static let latitude = 21.4
    /// Mid-October, when this was written: the sun crosses the south at ~60° at noon.
    public static let defaultDay = 288

    /// Direction towards the sun at local solar time `hour` (0–24) on `day` (1–365).
    public static func direction(hour: Double, day: Int = defaultDay) -> (x: Double, y: Double, z: Double) {
        let phi = latitude * .pi / 180
        let declination = 23.44 * .pi / 180 * sin(2 * .pi * Double(284 + day) / 365)
        let h = (hour - 12) * 15 * .pi / 180
        return (-cos(declination) * sin(h),
                cos(phi) * sin(declination) - sin(phi) * cos(declination) * cos(h),
                sin(phi) * sin(declination) + cos(phi) * cos(declination) * cos(h))
    }

    /// Optical depth at the zenith per channel (~650, 550, 450 nm): Rayleigh plus a humid
    /// maritime aerosol load.
    static let opticalDepth = RGB(0.13, 0.20, 0.36)

    /// Relative air mass (Kasten & Young 1989), finite down to the horizon.
    static func airMass(sinElevation s: Double) -> Double {
        let e = asin(Swift.max(-1, Swift.min(1, s))) * 180 / .pi
        let clamped = Swift.max(e, 0)
        return 1 / (sin(clamped * .pi / 180) + 0.50572 * pow(clamped + 6.07995, -1.6364))
    }

    static func transmittance(sinElevation s: Double) -> RGB {
        let m = airMass(sinElevation: s)
        return RGB(exp(-opticalDepth.r * m), exp(-opticalDepth.g * m), exp(-opticalDepth.b * m))
    }

    /// Sky brightness relative to the reference sun: falls with the sun, still lit in
    /// civil twilight, dark by 6° below the horizon.
    static func skyLevel(sinElevation s: Double) -> Double {
        (Swift.max(0, s + 0.1) / (Look.sunDirection.z + 0.1)).squareRoot()
    }

    static func luminance(_ c: RGB) -> Double { 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b }

    public static func light(direction d: (x: Double, y: Double, z: Double)) -> SunLight {
        let s = d.z
        let ref = Look.sunDirection.z
        let t = transmittance(sinElevation: s), t0 = transmittance(sinElevation: ref)
        // The disc sinks below the horizon over about half a degree.
        let disc = Look.smoothstep(-0.006, 0.006, s)
        let sun = Look.sun * RGB(t.r / t0.r, t.g / t0.g, t.b / t0.b) * disc
        let level = skyLevel(sinElevation: s)
        let sky = Look.sky * level
        // Low sun: warm haze along the horizon, a deeper zenith.
        let low = 1 - Look.smoothstep(0.05, 0.4, s)
        let hazeTint = RGB(1, 1, 1).mix(RGB(1.25, 0.95, 0.75), low) * level
        let zenithTint = RGB(1, 1, 1).mix(RGB(0.8, 0.85, 1.0), low) * level
        // Horizontal illuminance relative to the reference; adapt to the square root of
        // the change, capped so night doesn't get amplified into grey.
        let e = luminance(sun) * Swift.max(0, s) + luminance(sky)
        let e0 = luminance(Look.sun) * ref + luminance(Look.sky)
        let exposure = Swift.min(6, Swift.max(0.8, pow(e0 / Swift.max(e, 1e-4), 0.5)))
        return SunLight(direction: d, sun: sun, sky: sky, hazeTint: hazeTint, zenithTint: zenithTint,
                        exposure: exposure)
    }

    public static func light(hour: Double, day: Int = defaultDay) -> SunLight {
        light(direction: direction(hour: hour, day: day))
    }
}
