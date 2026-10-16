import XCTest
@testable import IslandCore

final class IslandCoreTests: XCTestCase {
    func testSplitMixIsDeterministic() {
        var a = SplitMix64(seed: 42), b = SplitMix64(seed: 42)
        for _ in 0..<1000 { XCTAssertEqual(a.next(), b.next()) }
    }

    func testNoiseIsDeterministicAndBounded() {
        for i in 0..<500 {
            let x = Double(i) * 0.37, y = Double(i) * -1.13
            let v = Noise.fbm(x, y, seed: 7)
            XCTAssertEqual(v, Noise.fbm(x, y, seed: 7))
            XCTAssertLessThanOrEqual(abs(v), 1.0)
        }
    }

    func testBilinearSampleHitsGridPoints() {
        var g = Grid(width: 4, height: 4, cellSize: 10)
        for i in 0..<g.count { g.values[i] = Float(i) }
        XCTAssertEqual(g.sample(x: 2, y: 1), g[2, 1])
        XCTAssertEqual(g.sample(x: 1.5, y: 0), 1.5, accuracy: 1e-6)
        XCTAssertEqual(g.sample(x: -5, y: 99), g[0, 3])
    }
}
