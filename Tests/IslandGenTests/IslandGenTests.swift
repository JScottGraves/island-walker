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
}
