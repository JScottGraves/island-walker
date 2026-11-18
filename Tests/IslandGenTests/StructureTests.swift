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
        XCTAssertEqual(a.structures.filter { $0.kind == "airstrip" }.count, 1)
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
        // (Clear of the taxiway, which crosses the strip at the middle or a quarter from an end.)
        let shoulder = s.world(u: 200, v: 30)
        XCTAssertEqual(map.sample(x: shoulder.x, y: shoulder.y).plants, .grassOnly)
        XCTAssertEqual(map.sample(x: 10, y: 10), .open)
    }

    /// A concrete apron beside the strip, joined to the runway by a taxiway across it, both on
    /// the runway's grade line: the runway, the taxiway and the apron meet without a step.
    func testApronBesideTheStripOnTheGradeLine() {
        var p = Airstrip.Params()
        p.spacing = 300
        p.refineCount = 8
        let (structures, graded) = SitePlanner.plan(context(), kinds: [Airstrip(params: p)], log: { _ in })
        let runway = structures.first { $0.kind == "airstrip" }!
        guard let apron = structures.first(where: { $0.kind == "apron" }),
              let taxiway = structures.first(where: { $0.kind == "taxiway" }) else { return XCTFail("no apron") }
        XCTAssertEqual(apron.id, "apron-1")
        XCTAssertEqual(apron.zones.first?.surface, .concrete)
        // Just past the strip, on one side.
        let a = runway.local(x: apron.x, y: apron.y)
        XCTAssertEqual(abs(a.v) - apron.zones[0].halfWidth, runway.gradedHalfWidth, accuracy: 1e-6)
        let map = StructureMap(structures)
        // Walking out along the taxiway from the runway's centreline to the apron's far side:
        // runway asphalt, then concrete all the way, on one grade line.
        let side: Double = a.v > 0 ? 1 : -1
        var surfaces: [StructureSurface] = []
        var v = 0.0
        while v < abs(a.v) + apron.zones[0].halfWidth - 1 {
            let (x, y) = runway.world(u: a.u, v: side * v)
            let s = map.sample(x: x, y: y)
            if surfaces.last != s.surface { surfaces.append(s.surface) }
            XCTAssertEqual(s.gradeHeight ?? -999, runway.gradeHeight(u: a.u), accuracy: 1e-6, "grade at v \(v)")
            v += 0.5
        }
        XCTAssertEqual(surfaces, [.paved, .concrete])
        XCTAssertEqual(taxiway.properties["edges"], "sides")
        // The graded heights follow the line on the apron, and the strip is still exact beside it.
        for (u, v) in [(a.u, a.v), (a.u + 50, a.v), (a.u + 100, side * 30)] {
            let (x, y) = runway.world(u: u, v: v)
            let i = Int(x / graded.cellSize), j = Int(y / graded.cellSize)
            let cx = (Double(i) + 0.5) * graded.cellSize, cy = (Double(j) + 0.5) * graded.cellSize
            guard let grade = map.sample(x: cx, y: cy).gradeHeight else { continue }
            XCTAssertEqual(Double(graded[i, j]), grade, accuracy: 1e-3)
        }
    }

    /// A hangar at the back of the apron, its door towards the runway, its floor concrete on
    /// the apron's grade line, and every bit of it on the apron.
    func testHangarAtTheBackOfTheApronFacingTheRunway() throws {
        var p = Airstrip.Params()
        p.spacing = 300
        p.refineCount = 8
        let (structures, _) = SitePlanner.plan(context(), kinds: [Airstrip(params: p)], log: { _ in })
        let runway = structures.first { $0.kind == "airstrip" }!
        let apron = try XCTUnwrap(structures.first { $0.kind == "apron" })
        let shed = try XCTUnwrap(structures.first { $0.kind == "hangar" })
        XCTAssertEqual(shed.id, "hangar-1")
        let hangar = try XCTUnwrap(Hangar(shed))
        // Every corner on the apron.
        for (u, v) in [(-1.0, -1.0), (-1, 1), (1, -1), (1, 1)] {
            let c = shed.world(u: u * hangar.span / 2, v: v * hangar.depth / 2)
            let q = apron.local(x: c.x, y: c.y)
            XCTAssertLessThan(abs(q.u), apron.zones[0].halfLength)
            XCTAssertLessThan(abs(q.v), apron.zones[0].halfWidth)
        }
        // The door's side is the one nearer the runway's centreline.
        let door = shed.world(u: 0, v: hangar.doorSide * hangar.depth / 2)
        let back = shed.world(u: 0, v: -hangar.doorSide * hangar.depth / 2)
        XCTAssertLessThan(abs(runway.local(x: door.x, y: door.y).v), abs(runway.local(x: back.x, y: back.y).v))
        // Its floor: concrete, nothing growing, on the runway's grade line.
        let map = StructureMap(structures)
        let floor = map.sample(x: shed.x, y: shed.y)
        XCTAssertEqual(floor.surface, .concrete)
        XCTAssertEqual(floor.plants, .none)
        XCTAssertEqual(floor.structure.map { structures[$0].kind }, "hangar")
        let along = runway.local(x: shed.x, y: shed.y).u
        XCTAssertEqual(floor.gradeHeight ?? -999, runway.gradeHeight(u: along), accuracy: 1e-6)
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

    /// A shallow cut's 1:3 slope is narrower than a 10 m cell; graded naively, a diagonal strip
    /// edge aliases into a sawtooth of cells at grade next to cells at natural height. The
    /// slope must ease out over a few cells instead, so the edge reads as a smooth bank.
    func testShallowCutEdgeHasNoStairSteps() {
        var g = Grid(width: 400, height: 400, cellSize: 10)
        for k in 0..<g.count { g.values[k] = 20 }
        var s = Structure(id: "airstrip-1", kind: "airstrip", x: 2000, y: 2000, heading: 33, elevation: 18.5, slope: 0,
                          zones: [StructureZone(name: "strip", halfLength: 810, halfWidth: 45, surface: .mownGrass,
                                                plants: .grassOnly, graded: true)])
        Grading.apply(&s, to: &g, seaLevel: 0)
        var worst = 0.0
        for j in 1..<(g.height - 1) {
            for i in 1..<(g.width - 1) {
                let lap = 4 * Double(g[i, j]) - Double(g[i - 1, j]) - Double(g[i + 1, j]) - Double(g[i, j - 1]) - Double(g[i, j + 1])
                worst = max(worst, abs(lap))
            }
        }
        // A 1.5 m bank spread over ≥ 2.5 cells bends by ~0.6 m per cell along an edge and about
        // twice that at the strip's corners, where two banks meet; a one-cell sawtooth bends by ~3.4 m.
        XCTAssertLessThan(worst, 1.5)
        XCTAssertEqual(Double(g[200, 200]), 18.5, accuracy: 1e-4)
        XCTAssertEqual(Double(g[5, 5]), 20, accuracy: 1e-4)
        // ...and the bank still meets the natural ground close by: a 1.5 m cut is back to
        // natural within a few tens of metres, not easing on for a hundred.
        XCTAssertLessThanOrEqual(s.embankment, 50)
        var shallow = s
        shallow.elevation = 19.6
        shallow.embankment = 0
        for k in 0..<g.count { g.values[k] = 20 }
        Grading.apply(&shallow, to: &g, seaLevel: 0)
        XCTAssertLessThanOrEqual(shallow.embankment, 50)
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
