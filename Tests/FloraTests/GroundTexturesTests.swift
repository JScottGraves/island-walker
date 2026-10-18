import XCTest
import IslandCore
@testable import Flora

final class GroundTexturesTests: XCTestCase {
    /// One layer per plant community, in the bundle's biome channel order, then bare rock.
    func testLayersFollowTheCommunities() {
        XCTAssertEqual(GroundTextures.layers.count, Look.communities.count + 1)
        for (layer, community) in zip(GroundTextures.layers, Look.communities) {
            XCTAssertEqual("biome-" + layer, community.channel)
        }
        XCTAssertEqual(GroundTextures.layers.last, "rock")
        XCTAssertEqual(GroundTextures.rockLayer, Look.communities.count)
    }

    /// Every layer loads as a 512² colour and normal map (height in blue) with full mips,
    /// and its colour is ground, not paint: mean linear albedo within real soils' range.
    func testTexturesLoad() throws {
        let t = try GroundTextures.load()
        XCTAssertEqual(t.albedo.count, GroundTextures.layers.count)
        XCTAssertEqual(t.normal.count, GroundTextures.layers.count)
        for (name, image) in zip(GroundTextures.layers, t.albedo) {
            XCTAssertEqual(image.size, GroundTextures.size, name)
            XCTAssertEqual(image.mips.count, 10, name)
            let mean = image.mips.last!
            let luminance = 0.2126 * srgb(mean[0]) + 0.7152 * srgb(mean[1]) + 0.0722 * srgb(mean[2])
            XCTAssert(luminance > 0.02 && luminance < 0.45, "\(name): \(luminance)")
            // No black holes: every texel keeps its colour.
            let darkest = stride(from: 0, to: image.mips[0].count, by: 4).map { Int(image.mips[0][$0]) + Int(image.mips[0][$0 + 1]) }.min()!
            XCTAssertGreaterThan(darkest, 0, name)
        }
        for (name, image) in zip(GroundTextures.layers, t.normal) {
            XCTAssertEqual(image.size, GroundTextures.size, name)
            let base = image.mips[0]
            // Normals mostly point out of the surface: x and y average near zero.
            for c in 0..<2 {
                let mean = stride(from: c, to: base.count, by: 4).reduce(0) { $0 + Int(base[$1]) } / (base.count / 4)
                XCTAssertEqual(Double(mean), 127.5, accuracy: 20, "\(name) channel \(c)")
            }
            // Heights span most of the range, so the height blend has something to work with.
            let heights = stride(from: 2, to: base.count, by: 4).map { base[$0] }
            XCTAssertGreaterThan(Int(heights.max()!) - Int(heights.min()!), 200, name)
        }
    }

    func srgb(_ v: UInt8) -> Double {
        let c = Double(v) / 255
        return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }
}
