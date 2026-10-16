import Foundation

/// Linear RGB triple.
public struct RGB: Sendable, Equatable {
    public var r: Double, g: Double, b: Double
    public init(_ r: Double, _ g: Double, _ b: Double) { self.r = r; self.g = g; self.b = b }

    public static func * (a: RGB, s: Double) -> RGB { RGB(a.r * s, a.g * s, a.b * s) }
    public static func * (a: RGB, b: RGB) -> RGB { RGB(a.r * b.r, a.g * b.g, a.b * b.b) }
    public static func + (a: RGB, b: RGB) -> RGB { RGB(a.r + b.r, a.g + b.g, a.b + b.b) }
    public func mix(_ o: RGB, _ t: Double) -> RGB { RGB(r + (o.r - r) * t, g + (o.g - g) * t, b + (o.b - b) * t) }

    /// From an 8-bit sRGB swatch, via the exact sRGB transfer function.
    public static func srgb(_ r: Int, _ g: Int, _ b: Int) -> RGB {
        RGB(Look.decode(Double(r) / 255), Look.decode(Double(g) / 255), Look.decode(Double(b) / 255))
    }

    /// Metal Shading Language literal.
    public var msl: String { String(format: "float3(%.5f, %.5f, %.5f)", r, g, b) }
}

/// Everything that decides how terrain looks, shared by the Metal shader and the CPU
/// ground views so the two stay alike: material albedos, light, haze and tone mapping
/// (research/global-illumination.md §3).
///
/// Albedos are authored as sRGB swatches and converted to linear once; earlier versions
/// used sRGB-looking numbers as linear values, which made everything 2–4× too bright and
/// greyish. Light is pre-exposed for sunny daylight: an 18% grey card lying flat under
/// this sun comes out near sRGB 110–118 after the tone mapper's toe.
public enum Look {
    // Materials (linear albedo). Real vegetation is dark (~0.05–0.1); lateritic soil is
    // red; fresh basalt ~0.07; coral sand ~0.4.
    public static let grassLow = RGB.srgb(72, 98, 50)
    public static let grassHigh = RGB.srgb(92, 104, 64)
    public static let soil = RGB.srgb(124, 84, 63)
    public static let rock = RGB.srgb(76, 72, 68)
    public static let sand = RGB.srgb(175, 165, 140)
    public static let shallowWater = RGB.srgb(40, 110, 115)
    public static let deepWater = RGB.srgb(10, 30, 60)

    /// Direction towards the sun (x east, y north, z up): mid-morning, high in the sky.
    public static let sunDirection: (x: Double, y: Double, z: Double) = {
        let v = (-0.4, 0.5, 0.6)
        let l = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
        return (v.0 / l, v.1 / l, v.2 / l)
    }()
    /// Pre-exposed sun radiance and sky irradiance on an open horizontal surface
    /// (clear sky, sun:sky about 5:1).
    public static let sun = RGB(1.25, 1.20, 1.10)
    public static let sky = RGB(0.17, 0.22, 0.33)
    /// Sunlight bounced off the surrounding ground onto faces that tilt toward it, as a
    /// fraction of the sun on open ground: a stand-in for one-bounce GI (§6) that keeps
    /// shaded walls from going black.
    public static let bounce = 0.08

    /// Haze (aerial perspective) colour, sky colour at the zenith, and how fast haze builds.
    public static let haze = RGB(0.50, 0.60, 0.74)
    public static let zenith = RGB(0.16, 0.30, 0.62)
    public static let hazeDistance = 45_000.0
    public static let hazeMax = 0.85

    /// Linear radiance of a lit surface before haze and tone mapping.
    /// - Parameters:
    ///   - sunVisibility: 0 in terrain shadow, 1 in sun.
    ///   - skyVisibility: fraction of the open sky the point sees (1 on a flat plain).
    public static func lit(albedo: RGB, normalZ: Double, nDotL: Double, sunVisibility: Double, skyVisibility: Double) -> RGB {
        let direct = sun * (Swift.max(0, nDotL) * sunVisibility)
        let ambient = sky * (skyVisibility * (0.5 + 0.5 * normalZ))
        let bounced = sun * (bounce * sunDirection.z * (0.5 - 0.5 * normalZ))
        return albedo * (direct + ambient + bounced)
    }

    public static func hazeAmount(distance: Double) -> Double {
        hazeMax * (1 - exp(-distance / hazeDistance))
    }

    /// Khronos PBR Neutral tone mapper (2024) without its black-level toe: identity below
    /// 0.76, so albedos read true, with a hue-preserving shoulder above. The toe subtracts
    /// up to 0.04, which crushed shaded cliff faces (linear ~0.005–0.01) to black.
    public static func toneMap(_ c: RGB) -> RGB {
        let start = 0.8 - 0.04, desaturation = 0.15
        var color = c
        let peak = Swift.max(color.r, color.g, color.b)
        if peak < start { return color }
        let d = 1 - start
        let newPeak = 1 - d * d / (peak + d - start)
        color = color * (newPeak / peak)
        let g = 1 - 1 / (desaturation * (peak - newPeak) + 1)
        return color.mix(RGB(newPeak, newPeak, newPeak), g)
    }

    /// Exact sRGB transfer functions.
    public static func encode(_ v: Double) -> Double {
        let x = Swift.min(1, Swift.max(0, v))
        return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055
    }

    public static func decode(_ v: Double) -> Double {
        v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
    }

    /// The same constants and tone mapper as Metal source, prepended to the terrain shader.
    public static var metalPrelude: String {
        let s = sunDirection
        return """
        constant float3 kGrassLow = \(grassLow.msl);
        constant float3 kGrassHigh = \(grassHigh.msl);
        constant float3 kSoil = \(soil.msl);
        constant float3 kRock = \(rock.msl);
        constant float3 kSand = \(sand.msl);
        constant float3 kShallowWater = \(shallowWater.msl);
        constant float3 kDeepWater = \(deepWater.msl);
        constant float3 kSun = \(sun.msl);
        constant float3 kSky = \(sky.msl);
        constant float3 kHaze = \(haze.msl);
        constant float3 kSunDirection = \(RGB(s.x, s.y, s.z).msl);
        constant float kHazeDistance = \(hazeDistance);
        constant float kHazeMax = \(hazeMax);
        constant float kBounce = \(bounce);

        // Khronos PBR Neutral without the black-level toe (see Look.toneMap).
        float3 pbrNeutral(float3 color) {
            const float start = 0.8 - 0.04;
            const float desaturation = 0.15;
            float peak = max(color.r, max(color.g, color.b));
            if (peak < start) return color;
            const float d = 1.0 - start;
            float newPeak = 1.0 - d * d / (peak + d - start);
            color *= newPeak / peak;
            float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
            return mix(color, float3(newPeak), g);
        }

        """
    }
}
