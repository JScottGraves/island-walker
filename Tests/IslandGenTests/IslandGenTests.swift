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

    func testFlexureTiltsTheCoastalFringe() {
        // Sea west of x = 8, a 10 m plain east of it, 100 m cells.
        var h = Grid(width: 32, height: 8, cellSize: 100, fill: 10)
        for y in 0..<8 { for x in 0..<8 { h[x, y] = -5 } }
        Island.flex(&h, seaLevel: 0, lift: 20, distance: 1000)
        XCTAssertEqual(h[4, 4], -5, "the sea is not lifted")
        XCTAssertEqual(h[8, 4], 12, accuracy: 0.01, "one cell inland gets a tenth of the lift")
        XCTAssertEqual(h[30, 4], 30, accuracy: 0.01, "full lift beyond the flexure distance")
    }

    func testRefineAddsDetailWithoutPits() {
        var p = Island.Params()
        p.size = 64
        p.steps = 20
        let island = Island.generate(p)
        var rp = Refine.Params()
        rp.factor = 4
        let fine = Refine.refine(island.height, drainageArea: island.drainageArea, seaLevel: 0, params: rp)
        XCTAssertEqual(fine.width, 256)
        XCTAssertEqual(fine.cellSize, island.height.cellSize / 4, accuracy: 1e-9)
        XCTAssertEqual(Realism.measure(fine).pits, 0, "every refined land cell drains to the sea")
        // Same island: land area within a few percent, summit within the detail amplitude.
        let coarseLand = Double(island.height.values.filter { $0 > 0 }.count) * island.height.cellArea
        let fineLand = Double(fine.values.filter { $0 > 0 }.count) * fine.cellArea
        XCTAssertEqual(fineLand / coarseLand, 1, accuracy: 0.05)
        XCTAssertEqual(fine.range.max, island.height.range.max, accuracy: Float(rp.detail) + 20)
        XCTAssertEqual(Refine.refine(island.height, drainageArea: island.drainageArea, seaLevel: 0, params: rp), fine,
                       "refinement is deterministic")
    }

    func testPondBuriesOnlyTheDeepestShareOfTheSaddle() {
        // A saddle surface at 200 m over eroded ground whose height rises 0..99 m by column.
        var h = Grid(width: 100, height: 4, cellSize: 100)
        var hardness = Grid(width: 100, height: 4, cellSize: 100, fill: 1)
        var surface = Grid(width: 100, height: 4, cellSize: 100, fill: 200)
        for y in 0..<4 { for x in 0..<100 { h[x, y] = Float(x) + 1 } }
        surface[0, 0] = -.infinity // outside the saddle
        var p = Island.Params()
        p.pondShare = 0.5
        Island.pond(&h, hardness: &hardness, surface: surface, p: p)
        XCTAssertEqual(h[0, 0], 1, "cells outside the saddle are untouched")
        XCTAssertEqual(h[90, 1], 91, "high ground stands above the lava")
        XCTAssertGreaterThan(h[10, 1], 11, "low ground is buried")
        XCTAssertEqual(hardness[10, 1], p.pondHardness, "ponded lava is harder")
        let buried = (0..<100).filter { h[$0, 2] > Float($0) + 1 }.count
        XCTAssertEqual(Double(buried), 50, accuracy: 2)
    }

    /// Milestone 1 acceptance: the same seed bakes a byte-identical bundle on two runs.
    func testSameSeedBakesAnIdenticalBundle() throws {
        func bake(_ url: URL) throws {
            var p = Island.Params()
            p.size = 64
            p.steps = 20
            let island = Island.generate(p)
            var rp = Refine.Params()
            rp.factor = 2
            rp.seed = p.seed
            let fine = Refine.refine(island.height, drainageArea: island.drainageArea, seaLevel: 0, params: rp)
            try IslandBundle.write([("height", fine), ("rain", island.rain), ("drainage", island.drainageArea)],
                                   seaLevel: 0, to: url, tileSize: 32)
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("determinism-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a"), b = root.appendingPathComponent("b")
        try bake(a)
        try bake(b)
        let files = FileManager.default.enumerator(atPath: a.path)!.compactMap { $0 as? String }.sorted()
        let other = FileManager.default.enumerator(atPath: b.path)!.compactMap { $0 as? String }.sorted()
        XCTAssertEqual(files, other)
        XCTAssertGreaterThan(files.count, 10)
        for f in files {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: a.appendingPathComponent(f).path, isDirectory: &isDir), !isDir.boolValue else { continue }
            XCTAssertEqual(try Data(contentsOf: a.appendingPathComponent(f)), try Data(contentsOf: b.appendingPathComponent(f)), f)
        }
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
