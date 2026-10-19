import XCTest
import IslandCore
@testable import Flora

final class GroundCoverTests: XCTestCase {
    /// One entry per plant community, in the bundle's channel order; cliffs carry none.
    func testCommunitiesMapToSpecies() {
        XCTAssertEqual(GroundCover.communities.count, Look.communities.count)
        for (entry, community) in zip(GroundCover.communities, Look.communities) {
            XCTAssertEqual(entry.channel, community.channel)
        }
        func entry(_ channel: String) -> GroundCover.Community { GroundCover.communities.first { $0.channel == channel }! }
        XCTAssertEqual(entry("biome-strand").species, .akiaki)
        XCTAssertEqual(entry("biome-rocky-shore").species, .akiaki)
        XCTAssertEqual(entry("biome-dry-forest").species, .kawelu)
        XCTAssertEqual(entry("biome-marsh").species, .makaloa)
        XCTAssertEqual(entry("biome-mesic-forest").species, .uki)
        XCTAssertEqual(entry("biome-wet-forest").species, .uki)
        XCTAssertEqual(entry("biome-cliff").density, 0)
        XCTAssertLessThan(entry("biome-rocky-shore").density, entry("biome-strand").density)
    }

    /// Densities fit the generation grid (one candidate per grid cell), and bunch species
    /// leave most ground bare: dry lowlands stay mostly red soil.
    func testDensitiesFitTheCandidateGrid() {
        let gridDensity = Double(GroundCover.tileCandidates) / (GroundCover.tileSize * GroundCover.tileSize)
        for c in GroundCover.communities {
            XCTAssertLessThanOrEqual(c.density, gridDensity, c.channel)
            XCTAssertGreaterThanOrEqual(c.cover, 0)
            XCTAssertLessThanOrEqual(c.cover, 0.8, c.channel)
        }
        let dry = GroundCover.communities.first { $0.channel == "biome-dry-forest" }!
        XCTAssertLessThan(dry.cover, 0.3)
    }

    /// Species look like themselves: makaloa tall and upright, ʻakiʻaki short, ʻuki broad,
    /// kawelu arching; colours are plausible linear albedos, darker at the root.
    func testSpeciesShapes() {
        let a = GroundCover.Species.akiaki.shape, k = GroundCover.Species.kawelu.shape
        let m = GroundCover.Species.makaloa.shape, u = GroundCover.Species.uki.shape
        XCTAssertLessThan(a.height.upperBound, 0.4)
        XCTAssertGreaterThanOrEqual(m.height.upperBound, 1.2)
        XCTAssertLessThan(m.bend, 0.2)
        XCTAssertGreaterThan(k.bend, 0.7)
        XCTAssertGreaterThan(u.width, 2 * a.width)
        for s in GroundCover.Species.allCases {
            let shape = s.shape
            for c in [shape.tip, shape.root] {
                XCTAssert(c.r > 0.01 && c.g > 0.01 && c.g < 0.45, "\(s)")
            }
            XCTAssertLessThan(shape.root.g, shape.tip.g, "\(s)")
            XCTAssertLessThan(shape.maxSlope, 1.43 + 1e-9, "\(s): nothing over 55°")
        }
    }

    /// Flags for the app and snapshots.
    func testSettingsFromArguments() {
        let off = GroundCover.Settings(arguments: ["app", "--cover", "off"])
        XCTAssertFalse(off.enabled)
        let s = GroundCover.Settings(arguments: ["app", "--cover-range", "70", "--cover-density", "0.5"])
        XCTAssertTrue(s.enabled)
        XCTAssertEqual(s.range, 70)
        XCTAssertEqual(s.density, 0.5)
        let d = GroundCover.Settings(arguments: ["app"])
        XCTAssertEqual(d.range, 50)
        XCTAssertEqual(d.density, 1)
    }
}
