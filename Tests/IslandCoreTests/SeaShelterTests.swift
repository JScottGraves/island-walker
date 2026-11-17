import XCTest
@testable import IslandCore

final class SeaShelterTests: XCTestCase {
    let ocean = OceanSpectrum()

    func testWindFetchInvertsTheSpectrum() {
        XCTAssertEqual(ocean.windFetch, 40_000, accuracy: 1)
    }

    /// The cascades' bands share out each system's whole variance.
    func testBandVariancesSumToTheSystem() {
        for system in ocean.systems {
            let total = (0..<OceanSpectrum.cascadeCount).map { ocean.bandVariance(system, cascade: $0) }.reduce(0, +)
            let hs = 4 * total.squareRoot()
            XCTAssertEqual(hs, system.measuredSignificantHeight, accuracy: 0.03 * system.measuredSignificantHeight)
        }
    }

    /// A high square island in a 100 m deep sea, the wind blowing towards −x (from the east):
    /// the water upwind is open, the water just downwind sheltered (short fetch, slack wind), and
    /// it opens up again far downwind.
    func testLeeOfAnIslandIsCalm() {
        var g = Grid(width: 200, height: 100, cellSize: 100, fill: -100)
        for j in 40..<60 { for i in 100..<120 { g[i, j] = 900 } }
        let wind = SeaShelter.wind(height: g, seaLevel: 0, towards: Double.pi, maxFetch: ocean.windFetch)
        XCTAssertEqual(Double(wind.fetch[150, 50]), ocean.windFetch, accuracy: 1)
        XCTAssertEqual(Double(wind.speed[150, 50]), 1, accuracy: 1e-6)
        XCTAssertLessThan(Double(wind.fetch[95, 50]), 1_000)
        XCTAssertLessThan(Double(wind.speed[95, 50]), 0.7)
        XCTAssertGreaterThan(Double(wind.fetch[10, 50]), 8_000)
        XCTAssertGreaterThan(wind.speed[10, 50], wind.speed[95, 50])

        let f = SeaShelter.factors(height: g, seaLevel: 0, spectrum: OceanSpectrum(windTowards: Double.pi),
                                   swells: nil, tradeHeight: 2)
        let open = f.cascades[50 * 200 + 150], lee = f.cascades[50 * 200 + 95]
        for c in 0..<4 { XCTAssertEqual(open[c], 1, accuracy: 0.02, "cascade \(c)") }
        // The chop's cascades (1–3) drop in the lee; the swells' cascade 0 keeps its swell.
        XCTAssertLessThan(lee[1], 0.6)
        XCTAssertLessThan(lee[2], 0.8)
        XCTAssertLessThan(f.calm[95, 50], 0.85)
        XCTAssertEqual(f.calm[150, 50], 1, accuracy: 0.01)
    }

    /// Behind a reef flat 0.5 m under water the wind sea starts again.
    func testReefFlatResetsTheFetch() {
        var g = Grid(width: 200, height: 40, cellSize: 50, fill: -20)
        for j in 0..<40 { for i in 120..<124 { g[i, j] = -0.5 } }
        let wind = SeaShelter.wind(height: g, seaLevel: 0, towards: Double.pi, maxFetch: 40_000)
        XCTAssertEqual(Double(wind.fetch[80, 20]), 40 * 50 + 25, accuracy: 400)
        XCTAssertEqual(Double(wind.speed[80, 20]), 1, accuracy: 1e-6)
    }

    func testSwellExposure() {
        // Open sea at its own height: 1; a quarter of the height behind a reef: about a quarter.
        XCTAssertEqual(SeaShelter.exposure(2, open: 2, depth: 500, period: 8), 1, accuracy: 0.01)
        XCTAssertEqual(SeaShelter.exposure(0.5, open: 2, depth: 500, period: 8), 0.25, accuracy: 0.01)
        // Where the depth caps both, the cap isn't counted twice.
        XCTAssertEqual(SeaShelter.exposure(0.6, open: 2, depth: 1, period: 8), 1, accuracy: 0.01)
        // Never reached: none.
        XCTAssertEqual(SeaShelter.exposure(0, open: 2, depth: 10, period: 8), 0, accuracy: 1e-6)
    }
}
