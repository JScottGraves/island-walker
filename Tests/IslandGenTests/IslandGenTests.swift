import XCTest
@testable import IslandCore
@testable import IslandGen

final class IslandGenTests: XCTestCase {
    /// After filling, every land cell must have a strictly lower D8 neighbour or be an outlet.
    func testPriorityFloodLeavesNoPits() {
        var g = Grid(width: 32, height: 32, cellSize: 10, fill: 5)
        var rng = SplitMix64(seed: 3)
        for i in 0..<g.count where !g.isBoundary(i) { g.values[i] = Float(rng.uniform(1, 100)) }
        PriorityFlood.fill(&g, seaLevel: 0)
        for y in 1..<31 {
            for x in 1..<31 {
                let h = g[x, y]
                let lower = (0..<8).contains { k in g[x + D8.dx[k], y + D8.dy[k]] < h }
                XCTAssertTrue(lower, "pit at \(x),\(y)")
            }
        }
    }

    func testIslandIsDeterministic() {
        var p = Island.Params()
        p.seed = 11
        p.size = 96
        p.steps = 20
        let a = Island.generate(p).height
        let b = Island.generate(p).height
        XCTAssertEqual(a.values, b.values)
    }

    func testIslandHasLandAndSeaAndErodes() {
        var p = Island.Params()
        p.seed = 5
        p.size = 128
        p.steps = 0
        let before = Island.generate(p).height
        p.steps = 60
        let after = Island.generate(p).height

        let land = after.values.filter { $0 > 0 }.count
        XCTAssertGreaterThan(land, after.count / 20, "too little land")
        XCTAssertLessThan(land, after.count / 2, "island fills the domain")
        XCTAssertLessThan(after.range.max, before.range.max, "erosion should lower the summit")
    }

    func testDrainageAreaIsConserved() {
        var h = VolcanicBase.generate(seed: 9, size: 64, domainSize: 80_000).height
        let flow = StreamPower.step(&h, params: StreamPower.Params())
        // With uniform rain, total area at outlets equals the domain area.
        var outletArea = 0.0
        for i in 0..<h.count where Int(flow.receivers[i]) == i { outletArea += Double(flow.area.values[i]) }
        XCTAssertEqual(outletArea, Double(h.count) * h.cellArea, accuracy: Double(h.count) * h.cellArea * 1e-4)
    }

    func testWavesCutTheWindwardCoastFaster() {
        // A flat 100 m plateau with sea along its east (upwind of the default trade wind,
        // which blows towards the southwest) and west edges. Only wave attack is on.
        var h = Grid(width: 32, height: 32, cellSize: 100, fill: 100)
        for y in 0..<32 { for x in 0..<32 where x < 4 || x >= 28 { h[x, y] = -20 } }
        var p = StreamPower.Params()
        p.erodibility = 0
        p.diffusivity = 0
        p.deposition = 0
        p.waterfallRetreat = 0
        _ = StreamPower.step(&h, params: p)
        let windward = h[27, 16], leeward = h[4, 16], inland = h[16, 16]
        XCTAssertEqual(inland, 100, accuracy: 0.1, "waves only reach the shoreline (pit filling tilts the flat slightly)")
        XCTAssertLessThan(windward, leeward, "the windward cliff retreats faster")
        XCTAssertLessThan(leeward, 100, "swell still reaches the leeward coast")
    }

    func testPNGChecksumsMatchKnownValues() {
        XCTAssertEqual(PNG.crc32(Array("IEND".utf8)), 0xAE42_6082)
        XCTAssertEqual(PNG.adler32(Array("Wikipedia".utf8)), 0x11E6_0398)
    }

    func testGroundViewDrawsSkyAndTerrain() {
        let base = VolcanicBase.generate(seed: 2, size: 128, domainSize: 80_000).height
        var summit = 0
        for i in 0..<base.count where base.values[i] > base.values[summit] { summit = i }
        let x = (Double(summit % 128) + 0.5) * base.cellSize, y = (Double(summit / 128) + 0.5) * base.cellSize
        let cam = GroundView.Camera(x: x, y: y, z: Double(base.values[summit]) + 1.7, yaw: 0, pitch: -0.1)
        let img = GroundView.render(base, camera: cam, width: 64, height: 36)
        XCTAssertEqual(img.rgb.count, 64 * 36 * 3)
        XCTAssertNotEqual(img.rgb[0..<3], img.rgb[(img.rgb.count - 3)...], "top row (sky) and bottom row (ground) should differ")
    }

    func testRealismOfAConeIsland() {
        // A cone 1000 m high and 10 km across at 50 m cells: no pits, all land reaches the
        // coast, slopes near atan(1000/5000) ≈ 11°, and hypsometry falls from 1 to ~0.
        var g = Grid(width: 256, height: 256, cellSize: 50, fill: -1)
        for y in 0..<256 {
            for x in 0..<256 {
                let r = (pow(Double(x - 128), 2) + pow(Double(y - 128), 2)).squareRoot() * 50
                if r < 5000 { g[x, y] = Float(1000 * (1 - r / 5000)) }
            }
        }
        let m = Realism.measure(g)
        XCTAssertEqual(m.pits, 0)
        XCTAssertEqual(m.hypsometry[0], 1, accuracy: 1e-9)
        XCTAssertLessThan(m.hypsometry[10], 0.01)
        XCTAssertGreaterThan(m.slopeHistogram[2], 0.8, "most of a 11° cone falls in the 10–15° band")
        XCTAssertEqual(m.cliffFraction, 0)
        // Flow on a cone runs straight outward and never gathers into a 1 km² channel.
        XCTAssertEqual(m.drainageDensity, 0)
    }

    /// A north–south ridge under an easterly wind: the cloud sits on the windward (east)
    /// flank and crest, and the lee flank and the open sea upwind stay clear.
    func testCloudsSitOnWindwardSlopes() {
        var h = Grid(width: 160, height: 160, cellSize: 200, fill: -100)
        for y in 0..<160 {
            for x in 0..<160 {
                let d = Double(x - 80) * 200
                h[x, y] = Float(900 * exp(-d * d / (2 * 3000 * 3000)) - 100)
            }
        }
        var p = Clouds.Params()
        p.windDirection = Double.pi // blowing towards -x, so east is windward
        let c = Clouds.coverage(height: h, seaLevel: 0, params: p)
        func at(_ metresEast: Double) -> Float {
            let x = (16_000 + metresEast) / c.cellSize - 0.5
            return c.sample(x: x, y: Double(c.height) / 2)
        }
        XCTAssertGreaterThan(at(2_000), 0.3, "windward flank should be cloudy")
        XCTAssertGreaterThan(at(2_000), 3 * at(-4_000), "lee flank should be much clearer")
        XCTAssertLessThan(at(14_000), 0.01, "open sea upwind should be clear")

        let flat = Clouds.coverage(height: Grid(width: 64, height: 64, cellSize: 200, fill: -50), seaLevel: 0)
        XCTAssertEqual(flat.range.max, 0)
    }
}
