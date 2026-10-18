import XCTest
@testable import IslandCore

final class ShadowCascadeTests: XCTestCase {
    let sun = ShadowCascades.normalized(SIMD3(0.4, -0.3, 0.6))
    let splits: [Double] = [25, 80, 220]

    func fit(eye: SIMD3<Double>, forward: SIMD3<Double>) -> [ShadowCascades.Cascade] {
        ShadowCascades.fit(eye: eye, forward: forward, sun: sun, splits: splits, tanHalfFovY: tan(Double.pi / 6),
                           aspect: 16.0 / 9.0, resolutions: [2048, 2048, 1024], casterReach: 300)
    }

    /// Every point of the view frustum nearer than a cascade's split (by distance from the
    /// camera, as the shaders pick cascades) lands inside its map.
    func testCascadesCoverTheFrustumUpToTheirSplit() {
        for forward in [ShadowCascades.normalized(SIMD3(0.3, 1, -0.07)), ShadowCascades.normalized(SIMD3(-1, 0.2, -1.5)),
                        ShadowCascades.normalized(SIMD3(0.1, -1, 0.4))] {
            let eye = SIMD3(41_234.6, 22_617.3, 612.8)
            let cascades = fit(eye: eye, forward: forward)
            XCTAssertEqual(cascades.count, 3)
            let s = ShadowCascades.normalized(ShadowCascades.cross(forward, SIMD3(0, 0, 1)))
            let u = ShadowCascades.cross(s, forward)
            let tanY = tan(Double.pi / 6), tanX = tanY * 16 / 9
            var rng = SplitMix64(seed: 3)
            for (i, c) in cascades.enumerated() {
                for k in 0..<4000 {
                    // Corners and edges of the frustum as well as its inside.
                    var a = Double.random(in: -1...1, using: &rng), b = Double.random(in: -1...1, using: &rng)
                    if k % 4 == 0 { a = a < 0 ? -1 : 1; b = b < 0 ? -1 : 1 }
                    let direction = ShadowCascades.normalized(forward + s * (a * tanX) + u * (b * tanY))
                    let distance = k % 3 == 0 ? splits[i] : Double.random(in: 0..<1, using: &rng) * splits[i]
                    let p = direction * distance
                    let q = c.project(p)
                    XCTAssert(q.x >= 0 && q.x <= 1 && q.y >= 0 && q.y <= 1, "cascade \(i) misses \(p): \(q)")
                    XCTAssert(q.z >= 0 && q.z <= 1, "cascade \(i) depth \(q.z)")
                    // A tree top up to 40 m above the point still falls inside the depth range.
                    XCTAssert(c.project(p + sun * (40 / sun.z)).z >= 0)
                }
            }
        }
    }

    /// The maps hold only what they must: a 16:9 frustum with a 60° vertical field of view
    /// fits in a sphere of radius 0.78 × the split.
    func testCascadesAreTight() {
        let c = fit(eye: SIMD3(0, 0, 0), forward: SIMD3(0, 1, 0))
        for i in 0..<3 { XCTAssertLessThan(c[i].radius, 0.78 * splits[i]) }
    }

    /// The maps stay the same size and on the same world texel grid as the camera moves, so
    /// shadow edges don't crawl while walking.
    func testCascadesSnapToAFixedTexelGrid() {
        let forward = ShadowCascades.normalized(SIMD3(1, 0.2, -0.05))
        let a = fit(eye: SIMD3(40_000, 30_000, 500), forward: forward)
        let b = fit(eye: SIMD3(40_003.37, 29_998.91, 500.4), forward: ShadowCascades.normalized(SIMD3(0.9, 0.35, -0.08)))
        let world = SIMD3(40_020.0, 30_011.0, 498.0)
        for i in 0..<3 {
            XCTAssertEqual(a[i].radius, b[i].radius)
            let side = i == 2 ? 1024.0 : 2048.0
            XCTAssertEqual(a[i].texel, 2 * a[i].radius / side, accuracy: 1e-12)
            // A fixed world point's position in texels differs by a whole number of texels.
            let qa = a[i].project(world - SIMD3(40_000, 30_000, 500))
            let qb = b[i].project(world - SIMD3(40_003.37, 29_998.91, 500.4))
            for d in [(qa.x - qb.x) * side, (qa.y - qb.y) * side] {
                XCTAssertEqual(d, d.rounded(), accuracy: 1e-6, "cascade \(i) shifted by \(d) texels")
            }
        }
    }

    func testDepthGrowsAwayFromTheSun() {
        let c = fit(eye: SIMD3(0, 0, 0), forward: SIMD3(0, 1, 0))[1]
        let p = SIMD3(0.0, 40, 0)
        XCTAssertLessThan(c.project(p + sun * 5).z, c.project(p).z)
        // Depth is linear in metres along the sun: range gives the scale.
        XCTAssertEqual((c.project(p).z - c.project(p + sun * 5).z) * c.depthRange, 5, accuracy: 1e-9)
    }
}
