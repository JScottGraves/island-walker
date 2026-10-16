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

    func testReferenceSunReproducesLookConstants() {
        let l = SunLight.reference
        XCTAssertEqual(l.sun.r, Look.sun.r, accuracy: 1e-9)
        XCTAssertEqual(l.sun.b, Look.sun.b, accuracy: 1e-9)
        XCTAssertEqual(l.sky.g, Look.sky.g, accuracy: 1e-9)
        XCTAssertEqual(l.exposure, 1, accuracy: 1e-9)
        XCTAssertEqual(l.hazeTint.r, 1, accuracy: 1e-9)
    }

    func testSunPathCrossesTheSouthAtNoonInOctober() {
        let noon = SunPath.direction(hour: 12)
        let elevation = asin(noon.z) * 180 / .pi
        XCTAssertEqual(elevation, 59.5, accuracy: 1.5)
        XCTAssertEqual(noon.x, 0, accuracy: 1e-9)
        XCTAssertLessThan(noon.y, 0)
        XCTAssertGreaterThan(SunPath.direction(hour: 8).x, 0.5)   // morning sun in the east
        XCTAssertLessThan(SunPath.direction(hour: 3).z, 0)         // night
    }

    func testLowSunIsRedderAndDimmer() {
        let noon = SunPath.light(hour: 12), dusk = SunPath.light(hour: 17.6)
        XCTAssertLessThan(dusk.sun.b / dusk.sun.r, noon.sun.b / noon.sun.r * 0.8)
        XCTAssertLessThan(dusk.sun.g, noon.sun.g)
        XCTAssertGreaterThan(dusk.exposure, noon.exposure)
        XCTAssertEqual(SunPath.light(hour: 0).sun.g, 0)
    }

    func testHorizonMapSeesAWall() {
        // A wall to the east of the plain: the east sector's horizon rises, the west stays flat.
        var g = Grid(width: 64, height: 64, cellSize: 50)
        for y in 0..<64 { for x in 0..<64 { g[x, y] = x > 40 ? 2000 : 10 } }
        let light = TerrainLighting.bake(height: g, seaLevel: 0, horizons: true)
        let n = TerrainLighting.sectors
        let i = (32 * 64 + 20) * n
        XCTAssertEqual(light.horizon.count, g.count * n)
        // Sectors 0 and n-1 straddle due east; n/2 - 1 and n/2 straddle due west.
        XCTAssertGreaterThan(light.horizon[i], 100)
        XCTAssertGreaterThan(light.horizon[i + n - 1], 100)
        XCTAssertLessThan(light.horizon[i + n / 2], 5)
        // Asking for horizons leaves sky visibility unchanged.
        XCTAssertEqual(TerrainLighting.bake(height: g, seaLevel: 0).sky.values, light.sky.values)
    }

    func testDistantSeaMeetsTheHorizonSky() {
        // Seen at a grazing angle, water is mostly reflected sky, so it can't form a dark
        // band against the sky just above the horizon.
        let water = Look.lit(albedo: Look.deepWater, normalZ: 1, nDotL: Look.sunDirection.z, sunVisibility: 1, skyVisibility: 1)
        let cosView = 0.01
        let sea = water.mix(Look.skyColor(sinElevation: cosView), Look.waterReflectance(cosView: cosView, distance: 40_000))
        let sky = Look.skyColor(sinElevation: 0)
        XCTAssertGreaterThan(sea.b, 0.8 * sky.b)
        XCTAssertLessThan(Look.waterReflectance(cosView: 1, distance: 500), 0.03)
        // A bay a couple of kilometres off, seen from a ridge, keeps mostly its own colour.
        XCTAssertLessThan(Look.waterReflectance(cosView: 0.1, distance: 2000), 0.3)
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
