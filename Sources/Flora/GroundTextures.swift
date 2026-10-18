import Foundation
import IslandCore

/// Tileable ground textures, one per native plant community and one of bare rock for
/// steep slopes (Assets/ground, cut by tools/ground_textures.py from CC0 ambientCG
/// materials; docs/flora.md §9). The terrain shader blends them by the community weights
/// near the camera.
public enum GroundTextures {
    /// Layer names: the bundle's biome channels in Look.communities order without their
    /// "biome-" prefix, then "rock".
    public static let layers: [String] = Look.communities.map { String($0.channel.dropFirst("biome-".count)) } + ["rock"]
    public static var rockLayer: Int { layers.count - 1 }
    /// Side of every layer, texels.
    public static let size = 512

    /// Each layer's colour (sRGB) and normal (OpenGL x and y, linear) with its height in
    /// blue, with box-filtered mips.
    public static func load() throws -> (albedo: [FloraTextures.Image], normal: [FloraTextures.Image]) {
        func image(_ name: String, normal: Bool) throws -> FloraTextures.Image {
            guard let url = Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "Assets/ground")
            else { throw FloraTextures.LoadError.missing(name) }
            let (n, rgba) = try FloraTextures.decode(url: url)
            guard n == size else { throw FloraTextures.LoadError.decode("\(name) is \(n)², not \(size)²") }
            return FloraTextures.Image(size: n, mips: FloraTextures.mipChain(rgba, size: n, preserveCoverage: false,
                                                                            linearData: normal))
        }
        return (try layers.map { try image("ground-\($0)", normal: false) },
                try layers.map { try image("ground-\($0)-normal", normal: true) })
    }
}
