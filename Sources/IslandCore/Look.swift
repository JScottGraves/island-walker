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

/// A linear RGB colour per cell over the whole square domain, row 0 at the south edge.
public struct ColourMap: Sendable {
    public let width: Int, height: Int
    /// Domain edge length in metres.
    public let domain: Double
    /// Three floats per cell, row-major.
    public var rgb: [Float]

    public init(width: Int, height: Int, domain: Double, rgb: [Float]) {
        precondition(rgb.count == width * height * 3, "colour map size")
        self.width = width; self.height = height; self.domain = domain; self.rgb = rgb
    }

    /// Bilinear colour at a world position in metres, cell k centred at (k + ½) × cell.
    public func sample(x: Double, y: Double) -> RGB {
        let fx = Swift.min(Swift.max(x / domain * Double(width) - 0.5, 0), Double(width - 1))
        let fy = Swift.min(Swift.max(y / domain * Double(height) - 0.5, 0), Double(height - 1))
        let x0 = Int(fx), y0 = Int(fy)
        let x1 = Swift.min(x0 + 1, width - 1), y1 = Swift.min(y0 + 1, height - 1)
        let tx = fx - Double(x0), ty = fy - Double(y0)
        func at(_ x: Int, _ y: Int) -> RGB {
            let o = 3 * (y * width + x)
            return RGB(Double(rgb[o]), Double(rgb[o + 1]), Double(rgb[o + 2]))
        }
        return at(x0, y0).mix(at(x1, y0), tx).mix(at(x0, y1).mix(at(x1, y1), tx), ty)
    }
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
    /// Canopy colour of each native plant community seen from above, by its bundle channel
    /// (islandgen's Vegetation; research/vegetation.md §2). Stands of ʻōhiʻa and uluhe are
    /// dark blue-green, koa paler and greyer, dry forest grey-green over red soil, strand
    /// and bog sedge straw-coloured.
    public static let communities: [(channel: String, albedo: RGB)] = [
        ("biome-strand", RGB.srgb(118, 120, 82)),
        ("biome-rocky-shore", RGB.srgb(72, 72, 62)),
        ("biome-marsh", RGB.srgb(72, 100, 52)),
        ("biome-dry-forest", RGB.srgb(98, 94, 62)),
        ("biome-mesic-forest", RGB.srgb(60, 84, 46)),
        ("biome-wet-forest", RGB.srgb(36, 70, 38)),
        ("biome-riparian", RGB.srgb(30, 68, 40)),
        ("biome-cloud-shrubland", RGB.srgb(58, 80, 46)),
        ("biome-bog", RGB.srgb(96, 94, 64)),
        ("biome-cliff", RGB.srgb(40, 76, 38)),
        ("biome-montane", RGB.srgb(36, 62, 40)),
    ]

    /// Canopy colour per cell from the community weight maps (all the same size, summing
    /// to 1 on land). Cells with no community (the sea) take the strand colour, so the
    /// coastline doesn't darken where the map is interpolated.
    public static func canopyMap(weights: [String: Grid], domain: Double) -> ColourMap? {
        guard let first = weights.values.first else { return nil }
        let w = first.width, h = first.height
        let fallback = communities[0].albedo
        var rgb = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            var c = RGB(0, 0, 0), total = 0.0
            for (channel, albedo) in communities {
                guard let g = weights[channel], g.width == w, g.height == h else { continue }
                let k = Double(g.values[i])
                c = c + albedo * k
                total += k
            }
            c = total > 1e-3 ? c * (1 / total) : fallback
            rgb[3 * i] = Float(c.r); rgb[3 * i + 1] = Float(c.g); rgb[3 * i + 2] = Float(c.b)
        }
        return ColourMap(width: w, height: h, domain: domain, rgb: rgb)
    }

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
    /// `canopy`, from the plant-community maps, replaces the plain height-graded green.
    public static func albedo(x: Double, y: Double, above: Double, slope: Double, maxHeight: Double, canopy: RGB? = nil) -> RGB {
        let grass = canopy ?? grassLow.mix(grassHigh, Swift.min(1, Swift.max(0, above / Swift.max(maxHeight, 1))))
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

        // canopy: the plant-community colour (rgb) and whether there is one (a).
        float3 terrainAlbedo(float2 p, float above, float slope, float maxHeight, float4 canopy) {
            float3 grass = mix(mix(kGrassLow, kGrassHigh, saturate(above / max(maxHeight, 1.0))), canopy.rgb, canopy.a);
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
