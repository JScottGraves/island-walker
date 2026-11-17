import XCTest
@testable import IslandCore
@testable import IslandGen

final class StructureTests: XCTestCase {
    /// A 12 km round island: a gentle coastal apron rising to a 900 m peak, with some
    /// rolling bumps so the grading has something to cut.
    func island(size: Int, domain: Double = 12_000) -> Grid {
        var g = Grid(width: size, height: size, cellSize: domain / Double(size))
        let c = domain / 2
        for j in 0..<size {
            for i in 0..<size {
                let x = (Double(i) + 0.5) * g.cellSize, y = (Double(j) + 0.5) * g.cellSize
                let r = ((x - c) * (x - c) + (y - c) * (y - c)).squareRoot()
                var h = 900 * exp(-pow(r / 1800, 2)) + 40 * (1 - r / 4800) - 2
                h += 4 * sin(x / 300) * cos(y / 410)
                if r > 4800 { h = -20 }
                g[i, j] = Float(h)
            }
        }
        return g
    }

    func context() -> SiteContext {
        let coarse = island(size: 160)
        let fine = island(size: 640)
        return SiteContext(fine: fine, coarse: coarse, seaLevel: 0, drainage: nil, clouds: nil, communities: [],
                           windDirection: 5 * Double.pi / 4, seed: 1)
    }

    func testExactlyOneAirstripAndDeterministic() {
        var p = Airstrip.Params()
        p.spacing = 300
        p.refineCount = 8
        let a = SitePlanner.plan(context(), kinds: [Airstrip(params: p)], log: { _ in })
        let b = SitePlanner.plan(context(), kinds: [Airstrip(params: p)], log: { _ in })
        XCTAssertEqual(a.structures.count, 1)
        XCTAssertEqual(a.structures, b.structures)
        XCTAssertEqual(a.height.values, b.height.values)
        let s = a.structures[0]
        XCTAssertEqual(s.kind, "airstrip")
        XCTAssertLessThanOrEqual(abs(s.slope), p.maxGrade + 1e-9)
        XCTAssertNotNil(s.properties["designators"])
    }

    func testGradedGroundFollowsTheGradeLine() {
        var p = Airstrip.Params()
        p.spacing = 300
        p.refineCount = 8
        let ctx = context()
        let (structures, graded) = SitePlanner.plan(ctx, kinds: [Airstrip(params: p)], log: { _ in })
        let s = structures[0]
        let map = StructureMap(structures)
        var checked = 0
        for j in 0..<graded.height {
            for i in 0..<graded.width {
                let x = (Double(i) + 0.5) * graded.cellSize, y = (Double(j) + 0.5) * graded.cellSize
                let sample = map.sample(x: x, y: y)
                guard let grade = sample.gradeHeight else { continue }
                XCTAssertEqual(Double(graded[i, j]), grade, accuracy: 1e-3)
                XCTAssertGreaterThan(grade, 2.9, "strip below the sea rule")
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 100)
        // The runway centre is paved with nothing growing; the strip grows grass only.
        let centre = map.sample(x: s.x, y: s.y)
        XCTAssertEqual(centre.surface, .paved)
        XCTAssertEqual(centre.plants, .none)
        let shoulder = s.world(u: 0, v: 40)
        XCTAssertEqual(map.sample(x: shoulder.x, y: shoulder.y).plants, .grassOnly)
        XCTAssertEqual(map.sample(x: 10, y: 10), .open)
    }

    func testRunwayRunsCloseToTheTradeWind() {
        var p = Airstrip.Params()
        p.spacing = 300
        p.refineCount = 8
        let s = SitePlanner.plan(context(), kinds: [Airstrip(params: p)], log: { _ in }).structures[0]
        // Trades from 045°: the runway should lie within 30° of the 045°–225° line.
        var delta = abs(s.heading - 45).truncatingRemainder(dividingBy: 180)
        delta = min(delta, 180 - delta)
        XCTAssertLessThan(delta, 30)
    }

    func testManifestRoundTrips() throws {
        let s = Structure(id: "airstrip-1", kind: "airstrip", x: 100, y: 200, heading: 50, elevation: 12, slope: 0.01,
                          zones: [StructureZone(name: "runway", halfLength: 750, halfWidth: 15, surface: .paved,
                                                plants: .none, graded: true)],
                          embankment: 20, properties: ["designators": "05/23"])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("structures-\(UUID()).json")
        try StructureManifest(structures: [s]).write(to: url)
        XCTAssertEqual(try StructureManifest.read(from: url).structures, [s])
        // local and world invert each other.
        let p = s.world(u: 123, v: -45)
        let back = s.local(x: p.x, y: p.y)
        XCTAssertEqual(back.u, 123, accuracy: 1e-9)
        XCTAssertEqual(back.v, -45, accuracy: 1e-9)
    }

    func testSoftMinIsContinuous() {
        var last = Grading.softMin(0, 5, knee: 1.5)
        for k in 1...1000 {
            let a = Double(k) * 0.01
            let v = Grading.softMin(a, 5, knee: 1.5)
            XCTAssertLessThan(abs(v - last), 0.011)
            XCTAssertLessThanOrEqual(v, 5 + 1e-12)
            last = v
        }
    }
}
