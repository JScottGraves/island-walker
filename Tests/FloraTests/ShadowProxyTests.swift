import XCTest
import simd
@testable import Flora

final class ShadowProxyTests: XCTestCase {
    /// Every edge is shared by exactly two triangles that run along it in opposite
    /// directions: closed, and wound the same way throughout.
    func assertClosed(_ positions: [SIMD3<Float>], _ indices: [UInt32], _ label: String) {
        XCTAssertEqual(indices.count % 3, 0, label)
        XCTAssertTrue(indices.allSatisfy { Int($0) < positions.count }, label)
        var directed: [UInt64: Int] = [:]
        for t in stride(from: 0, to: indices.count, by: 3) {
            for k in 0..<3 {
                let a = UInt64(indices[t + k]), b = UInt64(indices[t + (k + 1) % 3])
                directed[a << 32 | b, default: 0] += 1
            }
        }
        for (edge, n) in directed {
            let reverse = (edge & 0xFFFF_FFFF) << 32 | edge >> 32
            XCTAssertEqual(n, 1, "\(label): edge used twice in one direction")
            XCTAssertEqual(directed[reverse], 1, "\(label): open or inconsistently wound edge")
        }
    }

    /// Signed volume: positive when the faces wind counter-clockwise seen from outside.
    func volume(_ positions: [SIMD3<Float>], _ indices: [UInt32]) -> Float {
        var v: Float = 0
        for t in stride(from: 0, to: indices.count, by: 3) {
            let a = positions[Int(indices[t])], b = positions[Int(indices[t + 1])], c = positions[Int(indices[t + 2])]
            v += simd_dot(a, simd_cross(b, c)) / 6
        }
        return v
    }

    func testProxiesAreClosedAndWithinBudget() {
        for plant in Plant.allCases {
            for variant in 0..<Flora.variants {
                let p = Flora.shadowProxy(plant, variant: variant)
                let label = "\(plant) v\(variant)"
                assertClosed(p.trunk.positions, p.trunk.indices, label + " trunk")
                assertClosed(p.crown.positions, p.crown.indices, label + " crown")
                XCTAssertGreaterThan(volume(p.trunk.positions, p.trunk.indices), 0, label + " trunk winds outward")
                XCTAssertGreaterThan(volume(p.crown.positions, p.crown.indices), 0, label + " crown winds outward")
                XCTAssertLessThanOrEqual(p.triangleCount, 80, label)
                XCTAssertGreaterThanOrEqual(p.trunk.indices.count / 3, 10, label + " trunk")
            }
        }
    }

    /// The crown sits inside the leaves it stands for (shrunk, so outer leaves stay sunlit)
    /// and the trunk reaches below the ground like the drawn trunks do.
    func testProxiesFitInsideTheirMeshes() {
        for plant in Plant.allCases {
            for variant in 0..<Flora.variants {
                let p = Flora.shadowProxy(plant, variant: variant)
                let mesh = Flora.mesh(plant, variant: variant, lod: 0)
                let (lo, hi) = mesh.bounds
                let label = "\(plant) v\(variant)"
                let crown = p.crown.positions
                let clo = crown.reduce(SIMD3<Float>(repeating: .greatestFiniteMagnitude)) { simd_min($0, $1) }
                let chi = crown.reduce(SIMD3<Float>(repeating: -.greatestFiniteMagnitude)) { simd_max($0, $1) }
                XCTAssertTrue(all(clo .>= lo - 0.01) && all(chi .<= hi + 0.01), label + " crown inside the mesh bounds")
                // Big enough to cast a shadow: at least a quarter of the leaf spread.
                XCTAssertGreaterThan(chi.x - clo.x, 0.25 * (hi.x - lo.x), label)
                XCTAssertLessThan(p.trunk.positions.map(\.z).min()!, 0, label + " trunk below ground")
            }
        }
    }

    /// Palms and tree ferns carry their fronds as one flat disc.
    func testFrondCrownsAreFlat() {
        for plant in [Plant.loulu, .hapuu] {
            let c = Flora.shadowProxy(plant, variant: 0).crown.positions
            let lo = c.reduce(SIMD3<Float>(repeating: .greatestFiniteMagnitude)) { simd_min($0, $1) }
            let hi = c.reduce(SIMD3<Float>(repeating: -.greatestFiniteMagnitude)) { simd_max($0, $1) }
            XCTAssertLessThan(hi.z - lo.z, 0.5 * Swift.min(hi.x - lo.x, hi.y - lo.y), "\(plant)")
        }
    }

    func testProxiesAreDeterministic() {
        let a = Flora.shadowProxy(.ohia, variant: 1), b = Flora.shadowProxy(.ohia, variant: 1)
        XCTAssertEqual(a.crown.positions, b.crown.positions)
        XCTAssertEqual(a.trunk.indices, b.trunk.indices)
    }
}
