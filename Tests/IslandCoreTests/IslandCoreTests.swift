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

    func testGreyCardUnderHighSunLooksMidGrey() {
        // An 18% grey card facing a high sun on open ground should come out near sRGB 118.
        let s = Look.sunDirection
        let lit = Look.lit(albedo: RGB(0.18, 0.18, 0.18), normalZ: 1, nDotL: s.z, sunVisibility: 1, skyVisibility: 1)
        let srgb = Look.encode(Look.toneMap(lit).g) * 255
        XCTAssert((100...140).contains(srgb), "grey card came out at sRGB \(srgb)")
    }

    func testToneMapIsMonotoneAndBounded() {
        var previous = -1.0
        for k in 0...200 {
            let v = Double(k) / 20
            let out = Look.toneMap(RGB(v, v, v)).r
            XCTAssertGreaterThanOrEqual(out, previous)
            XCTAssertLessThanOrEqual(out, 1)
            previous = out
        }
    }

    func testTerrainLightingShadesAValleyFloor() {
        // A trench: its floor sees less sky than the open plateau beside it.
        var g = Grid(width: 64, height: 64, cellSize: 50)
        for y in 0..<64 { for x in 0..<64 { g[x, y] = abs(x - 32) < 3 ? 10 : 400 } }
        let light = TerrainLighting.bake(height: g, seaLevel: 0)
        XCTAssertLessThan(light.sky[32, 32], light.sky[10, 32] - 0.2)
        XCTAssertGreaterThan(light.sky[10, 32], 0.9)
    }

    func testDistantSeaMeetsTheHorizonSky() {
        // Seen at a grazing angle, water is mostly reflected sky, so it can't form a dark
        // band against the sky just above the horizon.
        let water = Look.lit(albedo: Look.deepWater, normalZ: 1, nDotL: Look.sunDirection.z, sunVisibility: 1, skyVisibility: 1)
        let cosView = 0.01
        let sea = water.mix(Look.skyColor(sinElevation: cosView), Look.waterReflectance(cosView: cosView))
        let sky = Look.skyColor(sinElevation: 0)
        XCTAssertGreaterThan(sea.b, 0.8 * sky.b)
        XCTAssertLessThan(Look.waterReflectance(cosView: 1), 0.03)
    }

    func testSteepPaliStaysGreen() {
        // A 55° wall is mostly vegetation where no bare patch shows: green dominates red.
        var green = (x: 0.0, y: 0.0), bare = (x: 0.0, y: 0.0)
        for k in 0..<2000 {
            let p = (x: Double(k % 50) * 37, y: Double(k / 50) * 41)
            if Look.exposure(x: p.x, y: p.y) < Look.exposure(x: green.x, y: green.y) { green = p }
            if Look.exposure(x: p.x, y: p.y) > Look.exposure(x: bare.x, y: bare.y) { bare = p }
        }
        let slope = 1 - cos(55 * Double.pi / 180)
        let wall = Look.albedo(x: green.x, y: green.y, above: 400, slope: slope, maxHeight: 1000)
        XCTAssertGreaterThan(wall.g, wall.r)
        // …and red soil shows through in the bare patches.
        let patch = Look.albedo(x: bare.x, y: bare.y, above: 400, slope: slope, maxHeight: 1000)
        XCTAssertGreaterThan(patch.r, patch.g)
    }
}
