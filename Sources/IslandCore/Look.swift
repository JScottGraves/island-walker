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
    // Materials (linear albedo). Real vegetation is dark (~0.05–0.1) and, in wet Hawaiian
    // forest, blue-green rather than olive; lateritic soil is red; fresh basalt ~0.05;
    // coral sand ~0.4.
    public static let grassLow = RGB.srgb(50, 92, 40)
    public static let grassHigh = RGB.srgb(58, 90, 50)
    /// Ferns and moss clinging to steep pali walls: darker and cooler than open grass.
    public static let cliffGreen = RGB.srgb(40, 76, 38)
    public static let soil = RGB.srgb(150, 70, 42)
    public static let rock = RGB.srgb(62, 60, 58)
    public static let sand = RGB.srgb(175, 165, 140)
    /// Water over a sandy bottom a few metres down, and open ocean.
    public static let shallowWater = RGB.srgb(35, 150, 145)
    public static let deepWater = RGB.srgb(6, 80, 98)

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

    /// Land albedo from world position (metres), height above the sea, slope (1 − normal.z)
    /// and the island's top. Steep pali are mostly green, with red soil streaks and black
    /// basalt breaking through in patches (`exposure`) on the steeper spurs and flute walls.
    public static func albedo(x: Double, y: Double, above: Double, slope: Double, maxHeight: Double) -> RGB {
        let grass = grassLow.mix(grassHigh, Swift.min(1, Swift.max(0, above / Swift.max(maxHeight, 1))))
        let bare = exposure(x: x, y: y)
        var a = grass.mix(cliffGreen, smoothstep(0.2, 0.45, slope))
        a = a.mix(soil, smoothstep(0.15, 0.35, slope) * (0.12 + 0.65 * bare))
        a = a.mix(rock, smoothstep(0.4, 0.7, slope) * (0.2 + 0.6 * (1 - bare)))
        // Beach: a narrow band just above the waterline on gentle slopes only.
        return sand.mix(a, Swift.max(smoothstep(1, 3, above), smoothstep(0.1, 0.25, slope)))
    }

    /// 0…1 patches from a few hundred metres to over a kilometre across where bare ground
    /// shows through vegetation: two scales of warped sines, so the Metal shader computes
    /// the identical pattern.
    public static func exposure(x: Double, y: Double) -> Double {
        func blobs(_ x: Double, _ y: Double) -> Double {
            let a = sin(x / 170 + 1.7 * sin(y / 230))
            let b = sin(y / 190 + 1.3 * sin(x / 260))
            return smoothstep(-0.2, 0.6, a * b)
        }
        return Swift.max(blobs(x, y) * (0.4 + 0.6 * blobs(y / 3.1 + 900, x / 3.1 + 400)),
                         0.6 * pow(blobs(x / 2.7 + 500, y / 2.7 + 300), 2))
    }

    /// Water colour by depth below the sea surface (metres).
    public static func waterAlbedo(depth: Double) -> RGB {
        shallowWater.mix(deepWater, smoothstep(0, 40, depth))
    }

    /// Sky radiance along a view ray: haze at the horizon, deepening to the zenith colour.
    public static func skyColor(sinElevation e: Double) -> RGB {
        haze.mix(zenith, 1 - exp(-3 * Swift.max(0, e)))
    }

    /// Fraction of the sky a water surface reflects (Schlick, F0 = 0.02), given the cosine
    /// between the view ray and the mean water surface and the distance to it. Nearby,
    /// waves tilt the surface toward the viewer (an effective cosine of at least ~0.35), so
    /// bays keep their own colour; far out the waves blur into a flat grazing mirror, which
    /// is what lets the distant sea meet the sky without a dark seam.
    public static func waterReflectance(cosView: Double, distance: Double) -> Double {
        let c = Swift.min(1, Swift.max(0, cosView))
        let roughness = 0.35 * (1 - smoothstep(6000, 35_000, distance))
        let m = 1 - (c * c + roughness * roughness).squareRoot()
        return 0.02 + 0.9 * m * m * m * m * m
    }

    static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }

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
    /// Sun and sky light come from `SunLight` through the shader's uniforms instead.
    public static var metalPrelude: String {
        """
        constant float3 kGrassLow = \(grassLow.msl);
        constant float3 kGrassHigh = \(grassHigh.msl);
        constant float3 kCliffGreen = \(cliffGreen.msl);
        constant float3 kSoil = \(soil.msl);
        constant float3 kRock = \(rock.msl);
        constant float3 kSand = \(sand.msl);
        constant float3 kShallowWater = \(shallowWater.msl);
        constant float3 kDeepWater = \(deepWater.msl);
        constant float3 kHaze = \(haze.msl);
        constant float3 kZenith = \(zenith.msl);
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

        // Mirrors Look.albedo, Look.skyColor and Look.waterReflectance.
        float exposureBlobs(float2 p) {
            float a = sin(p.x / 170.0 + 1.7 * sin(p.y / 230.0));
            float b = sin(p.y / 190.0 + 1.3 * sin(p.x / 260.0));
            return smoothstep(-0.2, 0.6, a * b);
        }

        float exposure(float2 p) {
            return max(exposureBlobs(p) * (0.4 + 0.6 * exposureBlobs(p.yx / 3.1 + float2(900.0, 400.0))),
                       0.6 * pow(exposureBlobs(p / 2.7 + float2(500.0, 300.0)), 2.0));
        }

        float3 terrainAlbedo(float2 p, float above, float slope, float maxHeight) {
            float3 grass = mix(kGrassLow, kGrassHigh, saturate(above / max(maxHeight, 1.0)));
            float bare = exposure(p);
            float3 a = mix(grass, kCliffGreen, smoothstep(0.2, 0.45, slope));
            a = mix(a, kSoil, smoothstep(0.15, 0.35, slope) * (0.12 + 0.65 * bare));
            a = mix(a, kRock, smoothstep(0.4, 0.7, slope) * (0.2 + 0.6 * (1.0 - bare)));
            return mix(kSand, a, max(smoothstep(1.0, 3.0, above), smoothstep(0.1, 0.25, slope)));
        }

        float3 waterAlbedo(float depth) {
            return mix(kShallowWater, kDeepWater, smoothstep(0.0, 40.0, depth));
        }

        // haze and zenith arrive pre-tinted for the sun's height (SunLight).
        float3 skyColor(float sinElevation, float3 haze, float3 zenith) {
            return mix(haze, zenith, 1.0 - exp(-3.0 * max(0.0, sinElevation)));
        }

        float waterReflectance(float cosView, float dist) {
            float c = saturate(cosView);
            float roughness = 0.35 * (1.0 - smoothstep(6000.0, 35000.0, dist));
            float m = 1.0 - sqrt(c * c + roughness * roughness);
            return 0.02 + 0.9 * m * m * m * m * m;
        }

        """
    }
}
