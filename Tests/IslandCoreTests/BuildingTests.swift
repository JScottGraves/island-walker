import XCTest
@testable import IslandCore

final class BuildingTests: XCTestCase {
    let hangar = Hangar(span: 28, depth: 36, eaves: 4.8, rise: 6, doorSide: -1, doorWidth: 25.6, doorHeight: 5, seed: 7)

    /// Where a segment first meets the mesh's triangles (Möller–Trumbore), as a fraction of its
    /// length, or nil.
    func hit(_ mesh: BuildingMesh, from a: SIMD3<Float>, to b: SIMD3<Float>) -> Float? {
        let d = b - a
        var best: Float?
        for k in stride(from: 0, to: mesh.indices.count, by: 3) {
            let p0 = mesh.vertices[Int(mesh.indices[k])].position
            let p1 = mesh.vertices[Int(mesh.indices[k + 1])].position
            let p2 = mesh.vertices[Int(mesh.indices[k + 2])].position
            let e1 = p1 - p0, e2 = p2 - p0
            let h = cross(d, e2)
            let det = dot(e1, h)
            if abs(det) < 1e-9 { continue }
            let s = a - p0
            let u = dot(s, h) / det
            if u < 0 || u > 1 { continue }
            let q = cross(s, e1)
            let v = dot(d, q) / det
            if v < 0 || u + v > 1 { continue }
            let t = dot(e2, q) / det
            if t >= 0 && t <= 1 && t < (best ?? .infinity) { best = t }
        }
        return best
    }

    func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float { (a * b).sum() }

    func testArchSpringsFromTheEavesAndRisesToTheCrown() {
        XCTAssertEqual(hangar.roofHeight(across: 0), 10.8, accuracy: 1e-9)
        XCTAssertEqual(hangar.roofHeight(across: 14), 4.8, accuracy: 1e-9)
        XCTAssertEqual(hangar.roofHeight(across: -14), 4.8, accuracy: 1e-9)
        XCTAssertGreaterThan(hangar.roofHeight(across: 7), 4.8)
    }

    func testMeshStaysOnItsFootprintWithUnitNormals() {
        let mesh = hangar.mesh()
        XCTAssertGreaterThan(mesh.triangleCount, 1000)
        XCTAssertEqual(mesh.indices.count % 3, 0)
        for v in mesh.vertices {
            XCTAssertLessThanOrEqual(abs(v.position.x), 14 + Float(Hangar.eaveOverhang) + 0.01)
            XCTAssertLessThanOrEqual(abs(v.position.y), 18 + Float(Hangar.gableOverhang) + 0.01)
            XCTAssertGreaterThanOrEqual(v.position.z, -Float(Hangar.footing) - 0.01)
            XCTAssertLessThanOrEqual(v.position.z, 10.85)
            XCTAssertEqual(dot(v.normal, v.normal), 1, accuracy: 1e-4)
        }
        XCTAssertTrue(mesh.vertices.contains { $0.material == .roofIron })
        XCTAssertTrue(mesh.vertices.contains { $0.material == .block })
    }

    /// The door faces −v: walking in from there at chest height, some of the opening is
    /// clear; the back and the side walls (between the windows) are solid, and the roof is
    /// closed overhead in most places but not all (the missing sheets).
    func testDoorIsOpenAndTheWallsAreClosed() {
        let mesh = hangar.mesh()
        let clear = stride(from: -12.0, through: 12.0, by: 0.5).filter { s in
            hit(mesh, from: SIMD3(Float(s), -25, 1.5), to: SIMD3(Float(s), 0, 1.5)) == nil
        }
        XCTAssertGreaterThan(clear.count, 10, "door opening")
        XCTAssertNotNil(hit(mesh, from: SIMD3(0, 25, 1.5), to: SIMD3(0, 0, 1.5)), "back wall")
        XCTAssertNotNil(hit(mesh, from: SIMD3(25, 0, 1.0), to: SIMD3(0, 0, 1.0)), "side wall")
        XCTAssertNotNil(hit(mesh, from: SIMD3(-25, 0, 1.0), to: SIMD3(0, 0, 1.0)), "side wall")
        var open = 0, total = 0
        for s in stride(from: -13.5, through: 13.5, by: 0.6) {
            for t in stride(from: -17.0, through: 17.0, by: 0.6) {
                total += 1
                if hit(mesh, from: SIMD3(Float(s), Float(t), 2), to: SIMD3(Float(s), Float(t), 20)) == nil { open += 1 }
            }
        }
        XCTAssertGreaterThan(open, 0, "no sheets missing")
        XCTAssertLessThan(Double(open) / Double(total), 0.15)
    }

    func testWallsBlockAllButTheDoor() {
        let walls = hangar.walls()
        func blocked(_ u: Double, _ v: Double) -> Bool { walls.contains { $0.distance(u: u, v: v) < $0.halfThickness + 0.3 } }
        // Every side wall and the back wall.
        XCTAssertTrue(blocked(13.9, 0))
        XCTAssertTrue(blocked(-13.9, 5))
        XCTAssertTrue(blocked(3, 17.9))
        // The middle of the hangar and somewhere in the door opening are free.
        XCTAssertFalse(blocked(0, 0))
        XCTAssertTrue(stride(from: -12.0, through: 12.0, by: 0.5).contains { !blocked($0, -17.5) })
    }

    func testReadsAStructuresProperties() {
        let s = Structure(id: "hangar-1", kind: "hangar", x: 100, y: 200, heading: 45, elevation: 12, slope: 0,
                          zones: [], properties: ["span": "30", "depth": "40", "door": "1", "doorWidth": "27"])
        let h = Hangar(s)
        XCTAssertEqual(h?.span, 30)
        XCTAssertEqual(h?.depth, 40)
        XCTAssertEqual(h?.doorSide, 1)
        XCTAssertEqual(h?.doorWidth, 27)
        XCTAssertEqual(h?.eaves, 4.8)
        XCTAssertNil(Hangar(Structure(id: "apron-1", kind: "apron", x: 0, y: 0, heading: 0, elevation: 0, slope: 0, zones: [])))
        XCTAssertEqual(Hangar(s)?.mesh().vertices, Hangar(s)?.mesh().vertices)
    }
}
