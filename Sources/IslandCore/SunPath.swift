import Foundation

/// The light a sun position gives, from the `Atmosphere` model, pre-exposed like
/// `Look`'s constants: the reference sun's sunlight has the luminance of `Look.sun`.
public struct SunLight: Sendable {
    /// Towards the sun (x east, y north, z up), unit length.
    public var direction: (x: Double, y: Double, z: Double)
    /// Sunlight on a surface facing the sun, after the atmosphere.
    public var sun: RGB
    /// Sky irradiance on open horizontal ground (for exposure and the CPU views).
    public var sky: RGB
    /// Sky irradiance for any normal, as spherical harmonics (`Atmosphere.irradianceSH`).
    public var skySH: [RGB]
    /// Sky radiance in every direction (`Atmosphere.skyViewSize`, row-major).
    public var skyView: [RGB]
    /// Scale applied before tone mapping: partial eye adaptation, so dawn and dusk read
    /// darker than noon without going black.
    public var exposure: Double

    /// The fixed sun older renders used, for before/after comparisons.
    public static var reference: SunLight { SunPath.light(direction: Look.sunDirection) }
}

/// Where the sun is over the island at a given hour and day, and what light that gives
/// (research/global-illumination.md §3, §5), with sunlight and sky from `Atmosphere`.
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

    /// Scale from the atmosphere's units (sun irradiance 1 above the atmosphere) to the
    /// renderer's pre-exposed irradiance over π.
    static let scale: Double = {
        let ref = Atmosphere.sunIrradiance(sinElevation: Look.sunDirection.z)
        return luminance(Look.sun) / luminance(ref)
    }()

    /// Exposure at the reference sun: an 18% grey card on open ground comes out near
    /// sRGB 125, a little brighter than a light meter's mid-grey, like a photo of a bright day.
    static let exposureKey = 1.25
    /// Brightens the sky as seen directly (and as haze) but not the light it casts. The
    /// single- plus multiple-scattering model gives a zenith about half as bright, relative
    /// to sunlit ground, as the CIE clear-sky luminance model (~4 kcd/m² at a 45° sun),
    /// and noon renders read as dusk without it.
    static let skyViewGain = 1.4

    static func luminance(_ c: RGB) -> Double { 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b }

    /// Horizontal illuminance the eye adapts to; the reference sun's is the fixed point.
    static func adaptationLevel(sun: RGB, sky: RGB, sinElevation s: Double) -> Double {
        luminance(sun) * Swift.max(0, s) + luminance(sky)
    }

    static let referenceLevel: Double = {
        let s = Look.sunDirection
        let sky = Atmosphere.evaluateSH(Atmosphere.irradianceSH(skyView: Atmosphere.skyView(sun: s)), (0, 0, 1)) * scale
        return adaptationLevel(sun: Atmosphere.sunIrradiance(sinElevation: s.z) * scale, sky: sky, sinElevation: s.z)
    }()

    public static func light(direction d: (x: Double, y: Double, z: Double)) -> SunLight {
        // The disc sinks below the horizon over about half a degree.
        let disc = Look.smoothstep(-0.006, 0.006, d.z)
        let sunScale: Double = scale * disc
        let t: RGB = Atmosphere.sunIrradiance(sinElevation: Swift.max(d.z, 0.004))
        let sun = RGB(t.r * sunScale, t.g * sunScale, t.b * sunScale)
        // The shader lights terrain with albedo · E rather than albedo · E / π (the π is in
        // `scale`), so radiance seen directly takes the π back.
        let radiance = Atmosphere.skyView(sun: d).map { $0 * (scale * Double.pi) }
        let skySH = Atmosphere.irradianceSH(skyView: radiance).map { $0 * (1 / Double.pi) }
        let skyView = radiance.map { $0 * skyViewGain }
        let sky = Atmosphere.evaluateSH(skySH, (0, 0, 1))
        // Adapt to the square root of the change, capped so night isn't amplified to grey.
        let level = adaptationLevel(sun: sun, sky: sky, sinElevation: d.z)
        let exposure = exposureKey * Swift.min(8, Swift.max(0.8, pow(referenceLevel / Swift.max(level, 1e-5), 0.5)))
        return SunLight(direction: d, sun: sun, sky: sky, skySH: skySH, skyView: skyView, exposure: exposure)
    }

    public static func light(hour: Double, day: Int = defaultDay) -> SunLight {
        light(direction: direction(hour: hour, day: day))
    }
}
