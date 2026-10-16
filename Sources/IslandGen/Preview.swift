import Foundation
import IslandCore
#if canImport(ImageIO)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

/// Quick-look images of generated maps for fast visual iteration.
public enum Preview {
    /// RGB8 image, row-major, top row first.
    public struct Image {
        public var width: Int
        public var height: Int
        public var rgb: [UInt8]
    }

    /// Shaded relief with hypsometric tint: ocean blues, lowland greens, upland browns.
    /// North (+y) is at the top of the image.
    public static func hillshade(_ h: Grid, seaLevel: Float = 0, sunAzimuth: Double = 315, sunAltitude: Double = 40) -> Image {
        let w = h.width, n = h.height
        let az = sunAzimuth * Double.pi / 180, alt = sunAltitude * Double.pi / 180
        let lx = sin(az) * cos(alt), ly = cos(az) * cos(alt), lz = sin(alt)
        let (_, top) = h.range
        var out = [UInt8](repeating: 0, count: w * n * 3)
        for y in 0..<n {
            for x in 0..<w {
                let x0 = Swift.max(x - 1, 0), x1 = Swift.min(x + 1, w - 1)
                let y0 = Swift.max(y - 1, 0), y1 = Swift.min(y + 1, n - 1)
                let dzdx = Double(h[x1, y] - h[x0, y]) / (Double(x1 - x0) * h.cellSize)
                let dzdy = Double(h[x, y1] - h[x, y0]) / (Double(y1 - y0) * h.cellSize)
                let len = (dzdx * dzdx + dzdy * dzdy + 1).squareRoot()
                let shade = Swift.max(0, (-dzdx * lx - dzdy * ly + lz) / len)

                let z = h[x, y]
                var c: (Double, Double, Double)
                if z <= seaLevel {
                    let t = Double(Swift.min(1, (seaLevel - z) / 1500))
                    c = (0.35 - 0.25 * t, 0.65 - 0.4 * t, 0.80 - 0.3 * t)
                } else {
                    let t = Double((z - seaLevel) / Swift.max(top - seaLevel, 1))
                    let low = (0.36, 0.55, 0.30), mid = (0.62, 0.58, 0.40), high = (0.55, 0.45, 0.38)
                    c = t < 0.5 ? mix(low, mid, t * 2) : mix(mid, high, (t - 0.5) * 2)
                    let s = 0.35 + 0.85 * shade
                    c = (c.0 * s, c.1 * s, c.2 * s)
                }
                let row = n - 1 - y
                let o = (row * w + x) * 3
                out[o] = byte(c.0)
                out[o + 1] = byte(c.1)
                out[o + 2] = byte(c.2)
            }
        }
        return Image(width: w, height: n, rgb: out)
    }

    /// Log-scaled drainage area: rivers show as bright lines.
    public static func flow(_ area: Grid) -> Image {
        let w = area.width, n = area.height
        let lo = log(area.cellArea), hi = log(Double(Swift.max(area.range.max, 1)))
        var out = [UInt8](repeating: 0, count: w * n * 3)
        for y in 0..<n {
            for x in 0..<w {
                let t = Swift.min(1, Swift.max(0, (log(Double(Swift.max(area[x, y], 1))) - lo) / Swift.max(hi - lo, 1e-6)))
                let v = pow(t, 2.2)
                let o = ((n - 1 - y) * w + x) * 3
                out[o] = byte(v * 0.6)
                out[o + 1] = byte(v * 0.8)
                out[o + 2] = byte(v)
            }
        }
        return Image(width: w, height: n, rgb: out)
    }

    /// Any scalar map as greyscale between its min and max.
    public static func gray(_ g: Grid) -> Image {
        let (lo, hi) = g.range
        let w = g.width, n = g.height
        var out = [UInt8](repeating: 0, count: w * n * 3)
        for y in 0..<n {
            for x in 0..<w {
                let v = Double((g[x, y] - lo) / Swift.max(hi - lo, 1e-6))
                let o = ((n - 1 - y) * w + x) * 3
                out[o] = byte(v); out[o + 1] = byte(v); out[o + 2] = byte(v)
            }
        }
        return Image(width: w, height: n, rgb: out)
    }

    /// Writes a PNG where ImageIO is available, otherwise a binary PPM next to the path.
    public static func write(_ image: Image, to url: URL) throws {
        #if canImport(ImageIO)
        let provider = CGDataProvider(data: Data(image.rgb) as CFData)!
        let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 24,
                         bytesPerRow: image.width * 3, space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
        #else
        var data = Data("P6\n\(image.width) \(image.height)\n255\n".utf8)
        data.append(contentsOf: image.rgb)
        try data.write(to: url.deletingPathExtension().appendingPathExtension("ppm"))
        #endif
    }

    /// Raw little-endian Float32 heights, row-major, for external tools.
    public static func writeRaw(_ g: Grid, to url: URL) throws {
        let data = g.values.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url)
    }

    private static func mix(_ a: (Double, Double, Double), _ b: (Double, Double, Double), _ t: Double) -> (Double, Double, Double) {
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
    }

    private static func byte(_ v: Double) -> UInt8 {
        UInt8(Swift.max(0, Swift.min(255, (v * 255).rounded())))
    }
}
