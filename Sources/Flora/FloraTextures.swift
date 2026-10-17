import Foundation
#if canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

/// The leaf, frond and bark textures (Assets/, cut by tools/flora_textures.py), decoded
/// to 8-bit sRGB RGBA with their mip chains.
public struct FloraTextures: Sendable {
    /// One texture: `size`² texels at mip 0, then each mip half the size, RGBA8 sRGB.
    public struct Image: Sendable {
        public var size: Int
        public var mips: [[UInt8]]
    }

    public var cards: [Image]
    public var bark: [Image]

    /// Alpha-test threshold the shader uses; mips keep the share of texels above it.
    public static let alphaCutoff: Float = 0.5

    public enum LoadError: Error { case missing(String), decode(String) }

    /// Loads every texture in `FloraMaterial.cardTextures` and `.barkTextures`.
    public static func load() throws -> FloraTextures {
        let cards = try FloraMaterial.cardTextures.map { try image(named: $0, card: true) }
        let bark = try FloraMaterial.barkTextures.map { try image(named: $0, card: false) }
        return FloraTextures(cards: cards, bark: bark)
    }

    static func image(named name: String, card: Bool) throws -> Image {
        guard let url = Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "Assets")
                ?? Bundle.module.url(forResource: name, withExtension: "png") else { throw LoadError.missing(name) }
        let (size, rgba) = try decode(url: url)
        return Image(size: size, mips: mipChain(rgba, size: size, preserveCoverage: card))
    }

    #if canImport(ImageIO)
    static func decode(url: URL) throws -> (Int, [UInt8]) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw LoadError.decode(url.lastPathComponent) }
        let w = image.width, h = image.height
        guard w == h else { throw LoadError.decode("\(url.lastPathComponent) is not square") }
        // Unpremultiplied, so the sRGB colour of half-transparent texels survives as authored.
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let ok: Bool = pixels.withUnsafeMutableBytes { buf in
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw LoadError.decode(url.lastPathComponent) }
        // CoreGraphics only draws premultiplied: undo it.
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let a = Int(pixels[i + 3])
            if a > 0 && a < 255 {
                for c in 0..<3 { pixels[i + c] = UInt8(Swift.min(255, (Int(pixels[i + c]) * 255 + a / 2) / a)) }
            }
        }
        return (w, pixels)
    }
    #else
    static func decode(url: URL) throws -> (Int, [UInt8]) { throw LoadError.decode("no image decoder") }
    #endif

    /// Box-filtered mips (colour averaged in linear light, weighted by alpha). With
    /// `preserveCoverage`, each mip's alpha is scaled so the share of texels passing the
    /// alpha test stays that of mip 0; otherwise distant crowns thin out. `linearData`
    /// averages the channels as they are (normals, not colours).
    public static func mipChain(_ base: [UInt8], size: Int, preserveCoverage: Bool, linearData: Bool = false) -> [[UInt8]] {
        let decode: (Float) -> Float = linearData ? { $0 } : srgbDecode
        let encode: (Float) -> Float = linearData ? { $0 } : srgbEncode
        var mips = [base]
        var level = base.map { Float($0) / 255 }
        var n = size
        func coverage(_ a: [Float], scale: Float) -> Float {
            var count = 0
            for i in stride(from: 3, to: a.count, by: 4) where a[i] * scale >= alphaCutoff { count += 1 }
            return Float(count) / Float(a.count / 4)
        }
        let target = coverage(level, scale: 1)
        while n > 1 {
            let m = n / 2
            var next = [Float](repeating: 0, count: m * m * 4)
            for y in 0..<m {
                for x in 0..<m {
                    var rgb: (Float, Float, Float) = (0, 0, 0)
                    var alpha: Float = 0, weight: Float = 0
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let o = ((2 * y + dy) * n + 2 * x + dx) * 4
                        let a = level[o + 3]
                        let w = preserveCoverage ? a + 1e-3 : 1
                        rgb.0 += decode(level[o]) * w
                        rgb.1 += decode(level[o + 1]) * w
                        rgb.2 += decode(level[o + 2]) * w
                        alpha += a
                        weight += w
                    }
                    let o = (y * m + x) * 4
                    next[o] = encode(rgb.0 / weight)
                    next[o + 1] = encode(rgb.1 / weight)
                    next[o + 2] = encode(rgb.2 / weight)
                    next[o + 3] = alpha / 4
                }
            }
            if preserveCoverage && target > 0 {
                // Bisect the alpha scale that restores mip 0's coverage.
                var lo: Float = 0.5, hi: Float = 4
                for _ in 0..<16 {
                    let mid = (lo + hi) / 2
                    if coverage(next, scale: mid) < target { lo = mid } else { hi = mid }
                }
                let s = (lo + hi) / 2
                for i in stride(from: 3, to: next.count, by: 4) { next[i] = Swift.min(1, next[i] * s) }
            }
            mips.append(next.map { UInt8(Swift.max(0, Swift.min(255, ($0 * 255).rounded()))) })
            level = next
            n = m
        }
        return mips
    }

    static func srgbDecode(_ v: Float) -> Float { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
    static func srgbEncode(_ v: Float) -> Float {
        let x = Swift.min(1, Swift.max(0, v))
        return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055
    }
}
