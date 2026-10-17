import XCTest
import simd
@testable import Flora

final class FloraTests: XCTestCase {
    func testMeshesAreDeterministic() {
        for plant in Plant.allCases {
            let a = Flora.mesh(plant, variant: 2, lod: 0), b = Flora.mesh(plant, variant: 2, lod: 0)
            XCTAssertEqual(a.vertices, b.vertices, "\(plant)")
            XCTAssertEqual(a.indices, b.indices, "\(plant)")
        }
    }

    func testVariantsDiffer() {
        for plant in Plant.allCases {
            XCTAssertNotEqual(Flora.mesh(plant, variant: 0, lod: 0).vertices, Flora.mesh(plant, variant: 1, lod: 0).vertices, "\(plant)")
        }
    }

    /// LOD0 trees within the design's 8–20k triangles (shrubs and ferns lighter), LOD1 at
    /// most about a fifth of LOD0.
    func testTriangleBudgets() {
        for plant in Plant.allCases {
            for v in 0..<Flora.variants {
                let lod0 = Flora.mesh(plant, variant: v, lod: 0).triangleCount
                let lod1 = Flora.mesh(plant, variant: v, lod: 1).triangleCount
                XCTAssertGreaterThan(lod0, 150, "\(plant) v\(v)")
                XCTAssertLessThanOrEqual(lod0, 20_000, "\(plant) v\(v)")
                XCTAssertLessThanOrEqual(Double(lod1), 0.22 * Double(lod0), "\(plant) v\(v): \(lod1) vs \(lod0)")
                XCTAssertGreaterThan(lod1, 20, "\(plant) v\(v)")
            }
        }
    }

    /// Meshes stand on the origin at about their reference height, with valid indices,
    /// unit normals and AO and wind weights in 0…1.
    func testMeshesAreWellFormed() {
        for plant in Plant.allCases {
            for lod in 0..<Flora.lods {
                let m = Flora.mesh(plant, variant: 0, lod: lod)
                let (lo, hi) = m.bounds
                let h = Float(plant.referenceHeight)
                XCTAssertLessThan(lo.z, 0.05, "\(plant) reaches the ground")
                XCTAssertEqual(hi.z, h, accuracy: 0.35 * h, "\(plant) LOD\(lod) height")
                XCTAssertTrue(m.indices.allSatisfy { Int($0) < m.vertices.count })
                for v in m.vertices {
                    XCTAssertEqual(simd_length(v.normal), 1, accuracy: 1e-3)
                    XCTAssertTrue((0...1).contains(v.ao) && (0...1).contains(v.sway) && (0...1).contains(v.flutter))
                    XCTAssertTrue(Int(v.material) < FloraMaterial.table.count)
                }
            }
        }
    }

    func testCommunitySharesSumToOne() {
        XCTAssertEqual(Communities.table.count, 11)
        for e in Communities.table {
            if e.canopy > 0 { XCTAssertEqual(e.canopyShares.reduce(0) { $0 + $1.share }, 1, accuracy: 1e-9, e.channel) }
            if e.shrubs > 0 { XCTAssertEqual(e.shrubShares.reduce(0) { $0 + $1.share }, 1, accuracy: 1e-9, e.channel) }
            XCTAssertLessThanOrEqual(e.canopy, 1e4 / (Communities.canopyCell * Communities.canopyCell), e.channel)
            XCTAssertLessThanOrEqual(e.shrubs, 1e4 / (Communities.shrubCell * Communities.shrubCell), e.channel)
        }
    }

    func testMaterialsCoverTheirTextures() {
        for m in FloraMaterial.table {
            let names = m.card ? FloraMaterial.cardTextures : FloraMaterial.barkTextures
            XCTAssertTrue(names.indices.contains(m.layer), m.name)
        }
        XCTAssertEqual(FloraMaterial.table.count, FloraMaterial.ID.allCases.count)
    }

    func testTexturesLoadWithCoveragePreservingMips() throws {
        let t = try FloraTextures.load()
        XCTAssertEqual(t.cards.count, FloraMaterial.cardTextures.count)
        XCTAssertEqual(t.bark.count, FloraMaterial.barkTextures.count)
        for image in t.cards + t.bark {
            XCTAssertLessThanOrEqual(image.size, 512)
            XCTAssertEqual(image.mips.count, Int(log2(Double(image.size))) + 1)
        }
        func coverage(_ m: [UInt8]) -> Double {
            var n = 0
            for i in stride(from: 3, to: m.count, by: 4) where m[i] >= 128 { n += 1 }
            return Double(n) / Double(m.count / 4)
        }
        for image in t.cards {
            let c0 = coverage(image.mips[0])
            XCTAssertGreaterThan(c0, 0.1)
            // 512 → 32: still within a few percent of mip 0.
            XCTAssertEqual(coverage(image.mips[4]), c0, accuracy: 0.05)
        }
    }
}
