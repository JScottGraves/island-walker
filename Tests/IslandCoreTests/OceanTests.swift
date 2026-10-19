import XCTest
@testable import IslandCore

final class OceanTests: XCTestCase {
    let ocean = OceanSpectrum()

    /// Spectral moment m0 (variance of the surface height) of one system, by quadrature.
    func variance(_ system: OceanSpectrum.System) -> Double {
        var m0 = 0.0
        let dw = 0.0005
        var w = 0.05
        while w < 30 {
            m0 += system.density(omega: w) * dw
            w += dw
        }
        return m0
    }

    func testTradeWindSeaIsModerate() {
        // ~9 m/s trades over a long fetch: a sea of 1.2–2.4 m with a 5.5–8.5 s peak.
        let sea = ocean.windSea
        let tp = 2 * Double.pi / sea.peakOmega
        XCTAssertGreaterThan(tp, 5.5)
        XCTAssertLessThan(tp, 8.5)
        let hs = 4 * variance(sea).squareRoot()
        XCTAssertGreaterThan(hs, 1.2)
        XCTAssertLessThan(hs, 2.4)
    }

    func testSwellHasItsSignificantHeight() {
        let hs = 4 * variance(ocean.swell).squareRoot()
        XCTAssertEqual(hs, ocean.swell.significantHeight, accuracy: 0.01 * ocean.swell.significantHeight)
        XCTAssertEqual(2 * Double.pi / ocean.swell.peakOmega, 13, accuracy: 0.01)
    }

    func testSpreadingIsNormalised() {
        for s in [1.0, 4, 30, 200] {
            var total = 0.0
            let n = 20_000
            for i in 0..<n {
                let theta = -Double.pi + (Double(i) + 0.5) * 2 * Double.pi / Double(n)
                total += OceanSpectrum.spreading(theta: theta, s: s) * 2 * Double.pi / Double(n)
            }
            XCTAssertEqual(total, 1, accuracy: 1e-3, "s = \(s)")
        }
    }

    func testCascadeBandsTileTheSpectrum() {
        for i in 0..<(OceanSpectrum.cascadeCount - 1) {
            XCTAssertEqual(ocean.band(cascade: i).upperBound, ocean.band(cascade: i + 1).lowerBound)
        }
        XCTAssertEqual(ocean.band(cascade: 0).lowerBound, 0)
        // Each band starts well above its patch's fundamental and ends below its Nyquist.
        for i in 0..<OceanSpectrum.cascadeCount {
            let fundamental = 2 * Double.pi / ocean.patchSizes[i]
            let nyquist = Double.pi * Double(OceanSpectrum.resolution) / ocean.patchSizes[i]
            XCTAssertLessThanOrEqual(ocean.band(cascade: i).upperBound, nyquist * 1.0001)
            if i > 0 { XCTAssertGreaterThan(ocean.band(cascade: i).lowerBound, 4 * fundamental) }
        }
    }

    func testCascadesCarryTheSeaVariance() {
        // The four cascades' modes together hold the spectrum's height variance.
        var total = 0.0
        for c in 0..<OceanSpectrum.cascadeCount {
            let modes = ocean.modes(cascade: c, seed: 7, size: 64)
            total += modes.expectedVariance
        }
        let expected = variance(ocean.windSea) + variance(ocean.swell)
        XCTAssertEqual(total, expected, accuracy: 0.1 * expected)
    }

    func testModesAreDeterministicAndPaired() {
        let a = ocean.modes(cascade: 1, seed: 3, size: 32), b = ocean.modes(cascade: 1, seed: 3, size: 32)
        XCTAssertEqual(a.amplitudes, b.amplitudes)
        // Entry k carries h0(−k) too: (n, m) pairs with (N − n, N − m).
        let n = 32, i = 5 * n + 9, j = (n - 5) * n + (n - 9)
        XCTAssertEqual(a.amplitudes[i].z, a.amplitudes[j].x)
        XCTAssertEqual(a.amplitudes[i].w, a.amplitudes[j].y)
        // Frequencies are whole multiples of the loop's base frequency, so the sea repeats.
        let base = 2 * Double.pi / OceanSpectrum.repeatPeriod
        for w in a.omegas.prefix(200) {
            let k = Double(w) / base
            XCTAssertEqual(k, k.rounded(), accuracy: 1e-2)
        }
    }

    func testCoxMunkRemainderFillsTheSlopeBudget() {
        let r = ocean.unresolvedSlopeVariance
        XCTAssertGreaterThan(r.upwind, 0)
        XCTAssertGreaterThan(r.crosswind, 0)
        let total = OceanSpectrum.coxMunk(windSpeed: ocean.windSea.windSpeed)
        // Their sum, 0.003 + 0.00508 U, is Cox & Munk's total 0.003 + 0.00512 U within its error.
        XCTAssertEqual(total.upwind + total.crosswind, 0.003 + 0.00512 * ocean.windSea.windSpeed, accuracy: 1e-3)
        let resolved = ocean.resolvedSlopeVariance
        XCTAssertEqual(resolved.upwind + r.upwind, max(total.upwind, resolved.upwind + 1e-4), accuracy: 1e-6)
    }

    func testDepthAttenuation() {
        XCTAssertEqual(OceanSpectrum.tma(omegaH: 0), 0)
        XCTAssertEqual(OceanSpectrum.tma(omegaH: 2), 1)
        XCTAssertEqual(OceanSpectrum.tma(omegaH: 3), 1)
        XCTAssertEqual(OceanSpectrum.tma(omegaH: 1 - 1e-9), OceanSpectrum.tma(omegaH: 1 + 1e-9), accuracy: 1e-6)
        var last = -1.0
        for i in 0...40 {
            let v = OceanSpectrum.tma(omegaH: Double(i) * 0.05)
            XCTAssertGreaterThanOrEqual(v, last)
            last = v
        }
        // Long waves shrink over a 2 m reef flat; short chop barely does.
        XCTAssertLessThan(OceanSpectrum.depthAttenuation(k: 0.08, depth: 2), 0.5)
        XCTAssertGreaterThan(OceanSpectrum.depthAttenuation(k: 5, depth: 2), 0.99)
    }

    func testFlatWaterFresnel() {
        XCTAssertEqual(Look.waterReflectance(cosView: 1), 0.02, accuracy: 0.002)
        XCTAssertGreaterThan(Look.waterReflectance(cosView: 0.01), 0.9)
        // No distance term: the same angle reflects the same at any range.
        XCTAssertLessThan(Look.waterReflectance(cosView: 0.5), Look.waterReflectance(cosView: 0.2))
    }

    func testOceanNodesCoverTheSeaOnce() {
        let tree = OceanQuadtree(quads: 32, cellMetres: 0.5, centre: (40_000, 40_000), reach: 200_000, rangeFactor: 2.5)
        let eye = (x: 41_234.0, y: 38_765.0, z: 12.0)
        let nodes = tree.select(eye: eye) { _, _ in true }
        // Every point within 30 km lies in exactly one node.
        for i in 0..<400 {
            let a = Double(i) * 2.399963, r = 30_000 * (Double(i) + 0.5) / 400
            let p = (x: eye.x + r * cos(a), y: eye.y + r * sin(a))
            let inside = nodes.filter { p.x >= $0.x && p.x < $0.x + $0.size && p.y >= $0.y && p.y < $0.y + $0.size }
            XCTAssertEqual(inside.count, 1, "point \(p)")
        }
        // The finest level stands under the camera.
        let under = nodes.first { eye.x >= $0.x && eye.x < $0.x + $0.size && eye.y >= $0.y && eye.y < $0.y + $0.size }
        XCTAssertEqual(under?.lod, 0)
        XCTAssertGreaterThan(tree.lodCount, 12)
        XCTAssertLessThanOrEqual(tree.lodCount, 16)
    }
}
