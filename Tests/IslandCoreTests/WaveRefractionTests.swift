import XCTest
@testable import IslandCore

final class WaveRefractionTests: XCTestCase {
    func testLinearDispersion() {
        let w = 2 * Double.pi / 8
        // Deep water: k = ω²/g.
        XCTAssertEqual(WaveRefraction.wavenumber(depth: 1000, period: 8), w * w / 9.81, accuracy: 1e-6)
        // Shallow water: k ≈ ω/√(gh).
        XCTAssertEqual(WaveRefraction.wavenumber(depth: 0.5, period: 8), w / (9.81 * 0.5).squareRoot(), accuracy: 0.03 * w / 2.2)
        // Group speed: half the phase speed in deep water, all of it in shallow water.
        XCTAssertEqual(WaveRefraction.groupSpeed(depth: 1000, period: 8), 9.81 / (2 * w), accuracy: 1e-3)
        XCTAssertEqual(WaveRefraction.groupSpeed(depth: 0.2, period: 13), (9.81 * 0.2).squareRoot(), accuracy: 0.02)
    }

    /// Open water 100 m deep everywhere: the arrival times are the plane wave's, the heights
    /// the offshore height.
    func testPlaneWaveOverFlatSea() {
        let g = Grid(width: 120, height: 120, cellSize: 50, fill: -100)
        let towards = 200 * Double.pi / 180
        let system = WaveRefraction.System(period: 8, towards: towards, height: 2)
        let f = WaveRefraction.bake(height: g, seaLevel: 0, system: system)
        let c = WaveRefraction.phaseSpeed(depth: 100, period: 8)
        let dir = SIMD2(cos(towards), sin(towards))
        func expected(_ i: Int, _ j: Int) -> Double { (Double(i) * dir.x + Double(j) * dir.y) * 50 / c }
        let base = Double(f.arrival[60, 60]) - expected(60, 60)
        for (i, j) in [(10, 10), (100, 20), (30, 90), (110, 110)] {
            XCTAssertEqual(Double(f.arrival[i, j]) - base, expected(i, j), accuracy: 0.02 * abs(expected(i, j) - expected(60, 60)) + 0.5)
            XCTAssertEqual(Double(f.height[i, j]), 2, accuracy: 0.05)
        }
    }

    /// A straight beach along x = 0, the floor sloping down offshore (east): waves coming in at
    /// 45° turn towards the shore as they slow, keeping sin θ / c (Snell).
    func testSnellOnAPlaneBeach() {
        var g = Grid(width: 200, height: 200, cellSize: 20)
        for j in 0..<200 { for i in 0..<200 { g[i, j] = -Float(i) * 0.25 + 2 } }     // 1:80, land at i < 8
        let towards = Double.pi + Double.pi / 4      // towards the south-west: 45° to the beach's normal
        let f = WaveRefraction.bake(height: g, seaLevel: 0, system: .init(period: 10, towards: towards, height: 0.3))
        func angle(_ i: Int, _ j: Int) -> Double {
            let gx = Double(f.arrival[i + 1, j] - f.arrival[i - 1, j]), gy = Double(f.arrival[i, j + 1] - f.arrival[i, j - 1])
            return atan2(abs(gy), abs(gx))       // from the beach's normal
        }
        let deep = (i: 190, d: 190 * 0.25 - 2.0), shallow = (i: 30, d: 30 * 0.25 - 2.0)
        let snellDeep = sin(angle(deep.i, 100)) / WaveRefraction.phaseSpeed(depth: deep.d, period: 10)
        let snellShallow = sin(angle(shallow.i, 100)) / WaveRefraction.phaseSpeed(depth: shallow.d, period: 10)
        XCTAssertEqual(snellShallow, snellDeep, accuracy: 0.06 * snellDeep)
        XCTAssertLessThan(angle(shallow.i, 100), angle(deep.i, 100) - 0.2)
    }

    /// Head-on over a gentle slope: no refraction, so the height follows shoaling
    /// (H ∝ Cg^-1/2) until the depth limits it.
    func testShoalingGrowsWavesUntilTheyBreak() {
        var g = Grid(width: 300, height: 40, cellSize: 20)
        for j in 0..<40 { for i in 0..<300 { g[i, j] = -Float(i) * 0.2 + 1 } }    // 1:100, deep at the east edge
        let f = WaveRefraction.bake(height: g, seaLevel: 0, system: .init(period: 10, towards: Double.pi, height: 0.5))
        let cg0 = WaveRefraction.groupSpeed(depth: 59, period: 10)
        // At 3 m (unbroken: 0.5 m waves break near 1 m): Green's-law growth over the offshore height.
        let i6 = 20, h6 = Double(i6) * 0.2 - 1
        let shoaled = 0.5 * (cg0 / WaveRefraction.groupSpeed(depth: h6, period: 10)).squareRoot()
        XCTAssertGreaterThan(shoaled, 0.58)
        XCTAssertEqual(Double(f.height[i6, 20]), shoaled, accuracy: 0.04 * shoaled)
        // In the last metre the depth caps them.
        for i in 6...10 {
            let h = Double(i) * 0.2 - 1
            XCTAssertLessThanOrEqual(Double(f.height[i, 20]), WaveRefraction.breakerIndex * max(h, WaveRefraction.capDepth) + 1e-3)
        }
    }

    /// An island in deep water: the sea behind it sees much smaller waves (reached only
    /// around its ends), the sea in front of it the full height.
    func testIslandCastsAShadow() {
        var g = Grid(width: 200, height: 200, cellSize: 40, fill: -200)
        for j in 90..<110 { for i in 90..<110 { g[i, j] = 30 } }
        // Waves travelling west (from the east).
        let f = WaveRefraction.bake(height: g, seaLevel: 0, system: .init(period: 8, towards: Double.pi, height: 2))
        XCTAssertEqual(Double(f.height[130, 100]), 2, accuracy: 0.15)
        XCTAssertLessThan(Double(f.height[70, 100]), 1.0)
        XCTAssertTrue(f.arrival[70, 100].isFinite)
        // Arrives later behind the island than the plane wave would.
        XCTAssertGreaterThan(Double(f.arrival[70, 100] - f.arrival[70, 20]), 1)
    }

    /// Sand just inland of the waterline gets arrival times carried on from the water (for the
    /// swash), later than the water beside it; land far inland none.
    func testBeachGetsArrivalTimes() {
        var g = Grid(width: 100, height: 20, cellSize: 20)
        for j in 0..<20 { for i in 0..<100 { g[i, j] = -Float(i) * 0.5 + 10 } }    // shoreline near i = 20
        let f = WaveRefraction.bake(height: g, seaLevel: 0, system: .init(period: 8, towards: Double.pi, height: 1))
        XCTAssertTrue(f.arrival[18, 10].isFinite)
        XCTAssertGreaterThan(f.arrival[18, 10], f.arrival[22, 10])
        XCTAssertGreaterThan(Double(f.height[18, 10]), 0)
        XCTAssertFalse(f.arrival[2, 10].isFinite)
    }
}
