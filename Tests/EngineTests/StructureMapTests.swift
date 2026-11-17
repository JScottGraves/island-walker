import Metal
import XCTest
import IslandCore
@testable import Engine

/// The renderer's structure lookup (StructureLookup + the MSL structureAt) must say what
/// IslandCore.StructureMap says, everywhere: the bins only speed it up.
final class StructureMapTests: XCTestCase {
    func airstrip(id: String, x: Double, y: Double, heading: Double, elevation: Double, slope: Double,
                  embankment: Double) -> Structure {
        Structure(id: id, kind: "airstrip", x: x, y: y, heading: heading, elevation: elevation, slope: slope,
                  zones: [
                    StructureZone(name: "runway", halfLength: 750, halfWidth: 15, surface: .paved, plants: .none, graded: true),
                    StructureZone(name: "strip", halfLength: 810, halfWidth: 45, surface: .mownGrass, plants: .grassOnly, graded: true),
                  ],
                  embankment: embankment, properties: ["designators": "06/24"])
    }

    var structures: [Structure] {
        [airstrip(id: "airstrip-1", x: 33285, y: 18405, heading: 61, elevation: 22.2, slope: 0.0141, embankment: 36),
         // A second one overlapping the first one's surroundings, to check the first-match order.
         airstrip(id: "airstrip-2", x: 34100, y: 18900, heading: 128, elevation: 30, slope: -0.01, embankment: 20)]
    }

    /// Distance of (u, v) from the nearest edge of any zone of any structure (to skip points
    /// a float rounding away from an edge).
    func nearEdge(_ x: Double, _ y: Double) -> Bool {
        for s in structures {
            let (u, v) = s.local(x: x, y: y)
            var boxes = s.zones.map { ($0.halfLength, $0.halfWidth) }
            boxes.append((s.gradedHalfLength + s.embankment, s.gradedHalfWidth + s.embankment))
            for (hl, hw) in boxes {
                if abs(abs(u) - hl) < 0.05 && abs(v) <= hw + 0.05 { return true }
                if abs(abs(v) - hw) < 0.05 && abs(u) <= hl + 0.05 { return true }
            }
        }
        return false
    }

    func randomPoints(_ count: Int) -> [SIMD2<Double>] {
        var rng = SplitMix64(seed: 7)
        func unit() -> Double { Double(rng.next() >> 11) / Double(1 << 53) }
        return (0..<count).compactMap { _ in
            let x = 32_000 + 3_500 * unit(), y = 17_000 + 3_200 * unit()
            return nearEdge(x, y) ? nil : SIMD2(x, y)
        }
    }

    func testSwiftPortAgreesWithIslandCore() {
        let lookup = StructureLookup(structures: structures)
        let map = StructureMap(structures)
        var inside = 0, embankment = 0
        for p in randomPoints(200_000) {
            let want = map.sample(x: p.x, y: p.y)
            let got = lookup.sample(x: Float(p.x), y: Float(p.y))
            XCTAssertEqual(got.surface, want.surface.code, "surface at \(p)")
            XCTAssertEqual(got.plants, want.plants.code, "plants at \(p)")
            XCTAssertEqual(got.structure, want.structure ?? -1, "structure at \(p)")
            XCTAssertEqual(got.graded, want.gradeHeight != nil, "graded at \(p)")
            if let h = want.gradeHeight {
                XCTAssertEqual(Double(got.gradeHeight), h, accuracy: 0.01)
                inside += 1
            } else if want.structure != nil {
                embankment += 1
            }
        }
        XCTAssertGreaterThan(inside, 1000)
        XCTAssertGreaterThan(embankment, 200)
    }

    func testGPUAgreesWithSwiftPort() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let probe = """
        kernel void structureProbe(const device uchar *sm [[buffer(20)]], const device float2 *points [[buffer(1)]],
                                   device float4 *out [[buffer(2)]], device float *clear [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
            StructureSample s = structureAt(sm, points[i]);
            out[i] = float4(float(s.surface) + 10.0 * float(s.plants) + 100.0 * float(s.graded ? 1 : 0),
                            s.gradeHeight, float(s.structure), s.edge);
            clear[i] = structureClearance(sm, points[i]);
        }
        """
        let source = terrainShaderSource + floraShaderSource + groundCoverShaderSource + oceanShaderSource + probe
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "structureProbe")!)
        let lookup = StructureLookup(structures: structures)
        let bytes = lookup.bytes()
        let sm = bytes.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let points = randomPoints(50_000).map { SIMD2<Float>(Float($0.x), Float($0.y)) }
        let pb = points.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let out = device.makeBuffer(length: points.count * 16, options: .storageModeShared)!
        let clear = device.makeBuffer(length: points.count * 4, options: .storageModeShared)!
        let cmd = device.makeCommandQueue()!.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(sm, offset: 0, index: 20)
        enc.setBuffer(pb, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBuffer(clear, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: points.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let o = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: points.count)
        let c = clear.contents().bindMemory(to: Float.self, capacity: points.count)
        var found = 0
        for (k, p) in points.enumerated() {
            let s = lookup.sample(x: p.x, y: p.y)
            let code = Float(s.surface) + 10 * Float(s.plants) + 100 * (s.graded ? 1 : 0)
            XCTAssertEqual(o[k].x, code, "codes at \(p)")
            XCTAssertEqual(Int(o[k].z), s.structure, "structure at \(p)")
            if s.structure >= 0 {
                found += 1
                XCTAssertEqual(o[k].y, s.gradeHeight, accuracy: 1e-3)
                XCTAssertEqual(o[k].w, s.edge, accuracy: 1e-2)
            }
            XCTAssertEqual(c[k], lookup.clearance(x: p.x, y: p.y), accuracy: 1e-2)
        }
        XCTAssertGreaterThan(found, 1000)
    }

    func testNoStructuresIsOpenEverywhere() {
        let lookup = StructureLookup(structures: [])
        XCTAssertEqual(lookup.bytes().count, StructureLookup.headerBytes)
        let s = lookup.sample(x: 100, y: 100)
        XCTAssertEqual(s.structure, -1)
        XCTAssertEqual(s.plants, PlantRule.untouched.code)
        XCTAssertGreaterThan(lookup.clearance(x: 100, y: 100), 1e5)
    }

    func testClearanceIsDistanceToTheRestrictedGround() {
        let s = structures[0]
        let lookup = StructureLookup(structures: [s])
        // 10 m beyond the embankment's edge, straight out to the side of the runway's centre.
        let out = s.world(u: 0, v: s.gradedHalfWidth + s.embankment + 10)
        XCTAssertEqual(lookup.clearance(x: Float(out.x), y: Float(out.y)), 10, accuracy: 0.05)
        // On the runway: inside, negative.
        XCTAssertLessThan(lookup.clearance(x: Float(s.x), y: Float(s.y)), 0)
        // Far away: no restriction known.
        XCTAssertGreaterThan(lookup.clearance(x: Float(s.x + 5000), y: Float(s.y)), 30)
    }

    /// The canopy cover is cut over the footprint where nothing tall may grow.
    func testFootprintCoverage() {
        let s = structures[0]
        let lookup = StructureLookup(structures: [s])
        // A 78 m cell centred on the runway is fully restricted; one 2 km away is free.
        XCTAssertEqual(lookup.restrictedFraction(x0: Float(s.x) - 39, y0: Float(s.y) - 39, size: 78), 1, accuracy: 1e-6)
        XCTAssertEqual(lookup.restrictedFraction(x0: Float(s.x) + 2000, y0: Float(s.y), size: 78), 0, accuracy: 1e-6)
    }
}
