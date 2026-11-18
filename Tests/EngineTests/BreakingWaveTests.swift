import Metal
import XCTest
import IslandCore
@testable import Engine

/// Breaking waves (research/waves.md, item 4; OceanShaders: surfProgress, surfPlunge, surfShape,
/// surfCrestProfile): a wave's break progress grows shoreward from its break line, its type
/// follows the Iribarren number, and its face stands up (vertical by p ≈ 0.3), throws its crest
/// forward (plunging) and collapses into a bore.
final class BreakingWaveTests: XCTestCase {
    static let probeSource = """
    // x: progress, y: plunge, z: shape (gain, face, throw) packed by the caller per probe kind.
    kernel void breakProbe(constant OceanParams &o [[buffer(0)]], device float4 *out [[buffer(1)]],
                           constant float4 *probes [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float4 q = probes[i];
        // q: hs, h, beta, period → progress, plunge, break depth.
        float hb = surfBreakDepth(o, q.x, 1.0, 1.0);
        float plunge = surfPlunge(o, hb, q.z, q.w);
        float p = surfProgress(hb, q.y, q.z, plunge, 0.0);
        out[i] = float4(p, plunge, hb, 0.0);
    }

    // Profile of one crest at progress probe.x and type probe.y, ξ from −6 to 6 m: the surface
    // point's forward position x' = ξ + d (x), height (y), and the tangent's run 1 + dd/dξ (z).
    kernel void profileProbe(constant OceanParams &o [[buffer(0)]], device float4 *out [[buffer(1)]],
                             constant float4 *probes [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float4 q = probes[0];
        float xi = -6.0 + float(i) * 0.002;
        float3 s = surfShape(q.x, q.y);
        float a = 0.8 * s.x, wf = 1.6 * s.y;
        float4 c = surfCrestProfile(a, xi, 5.0, wf, s.z);
        out[i] = float4(xi + c.z, c.x, 1.0 + c.w, s.x);
    }
    """

    func run(_ kernel: String, probes: [SIMD4<Float>], count: Int) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let source = terrainShaderSource + floraShaderSource + groundCoverShaderSource + oceanShaderSource
            + BreakingWaveTests.probeSource
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: kernel)!)
        let paramsSize = MemoryLayout<OceanRenderer.Params>.stride
        let params = device.makeBuffer(length: paramsSize, options: .storageModeShared)!
        memset(params.contents(), 0, paramsSize)
        let waves = MemoryLayout<OceanRenderer.Params>.offset(of: \.waves)!
        params.contents().storeBytes(of: SIMD4<Float>(8, 13, Float(WaveRefraction.breakerIndex), 7),
                                     toByteOffset: waves, as: SIMD4<Float>.self)
        let breaking = MemoryLayout<OceanRenderer.Params>.offset(of: \.breaking)!
        params.contents().storeBytes(of: SIMD4<Float>(1, 1, 0.8, 512), toByteOffset: breaking, as: SIMD4<Float>.self)
        let input = device.makeBuffer(bytes: probes, length: probes.count * 16, options: .storageModeShared)!
        let out = device.makeBuffer(length: count * 16, options: .storageModeShared)!
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(params, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBuffer(input, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    func profile(p: Float, plunge: Float) throws -> [SIMD4<Float>] {
        try run("profileProbe", probes: [SIMD4(p, plunge, 0, 0)], count: 6001)
    }

    func testProgressGrowsShorewardFromTheBreakLine() throws {
        // A 1 m sea on a 1:30 slope, 8 s: from 4 m deep in to the waterline.
        let depths = stride(from: Float(4), through: 0, by: -0.25).map { $0 }
        let s = try run("breakProbe", probes: depths.map { SIMD4(1, $0, 1.0 / 30, 8) }, count: depths.count)
        for k in 1..<s.count { XCTAssertGreaterThan(s[k].x, s[k - 1].x, "progress must grow as the water shallows") }
        // Zero where the water is the break depth.
        let hb = s[0].z
        let at = try run("breakProbe", probes: [SIMD4(1, hb, 1.0 / 30, 8)], count: 1)
        XCTAssertEqual(at[0].x, 0, accuracy: 1e-3)
        XCTAssertLessThan(s[0].x, 0, "4 m deep a 1 m sea hasn't broken")
        XCTAssertGreaterThan(s.last!.x, 1, "at the waterline the wave has collapsed into a bore")
    }

    func testBreakerTypeFollowsIribarren() throws {
        // A steep shore break (1:12, 13 s, 1.25 m: ξ ≈ 1) plunges; a gentle sand flat (1:50,
        // 8 s, 1.1 m: ξ ≈ 0.2) spills.
        let s = try run("breakProbe", probes: [SIMD4(1.25, 2, 1.0 / 12, 13), SIMD4(1.1, 2, 1.0 / 50, 8)], count: 2)
        XCTAssertGreaterThan(s[0].y, 0.9)
        XCTAssertLessThan(s[1].y, 0.1)
    }

    func testShoalingWaveIsNotThrown() throws {
        for q in try profile(p: -0.8, plunge: 1) {
            XCTAssertEqual(q.z, 1, accuracy: 1e-4, "no forward displacement before breaking")
        }
    }

    func testFaceStandsVerticalByAThird() throws {
        // The steepest the tangent's run gets is about zero: a vertical face, not yet folded.
        let s = try profile(p: 0.3, plunge: 1)
        let run = s.map { $0.z }.min()!
        XCTAssertEqual(run, 0, accuracy: 0.08)
    }

    func testPlungingCrestOverhangs() throws {
        let s = try profile(p: 0.65, plunge: 1)
        XCTAssertLessThan(s.map { $0.z }.min()!, -0.5, "the face folds under the thrown crest")
        // The crest's top is ahead of the face below it.
        let top = s.max { $0.y < $1.y }!
        let below = s.filter { $0.y < 0.5 * top.y && $0.y > 0.3 * top.y }
        XCTAssertTrue(below.contains { $0.x < top.x - 0.3 }, "a curl: water under the lip lies behind its tip")
    }

    func testSpillingCrestOnlyCrumbles() throws {
        for p: Float in [0.3, 0.5, 0.7, 0.9] {
            let run = try profile(p: p, plunge: 0).map { $0.z }.min()!
            XCTAssertGreaterThan(run, -0.4, "a spilling wave at p \(p) barely overhangs")
        }
    }

    func testCollapsesIntoABore() throws {
        let peak = try profile(p: 0.35, plunge: 1)[0].w
        let bore = try profile(p: 1.3, plunge: 1)
        XCTAssertLessThan(bore[0].w, 0.6 * peak)
        for q in bore { XCTAssertEqual(q.z, 1, accuracy: 0.05, "the bore's crest is no longer thrown") }
    }

    /// The lip's axis: the dominant swell near the camera and its mean heading in the surf zone.
    func testLipPicksTheDominantSwellAndItsHeading() {
        let n = 64, cell: Float = 40
        var texels = [SIMD4<Float>](repeating: SIMD4(0, 0, 0, 100), count: n * n)
        func pack(_ a: Float, _ b: Float) -> Float {
            let lo = UInt32(Float16(a).bitPattern), hi = UInt32(Float16(b).bitPattern)
            return Float(bitPattern: lo | hi << 16)
        }
        for j in 0..<n {
            for i in 0..<n {
                // A beach along x, the shore at y = 1600 m: the trade swell (system 0) comes in
                // heading north-north-east at 1.2 m, the north swell (1) at 0.3 m heading west.
                let x = (Float(i) + 0.5) * cell, y = (Float(j) + 0.5) * cell
                let depth = max((1600 - y) / 40, 0)
                let a0 = (x * 0.26 + y * 0.97) / 60, a1 = -x / 120
                texels[j * n + i] = SIMD4(a0, a1, pack(depth > 0 ? 1.2 : 0, depth > 0 ? 0.3 : 0), depth)
            }
        }
        let field = SurfLip.Field(texels: texels, width: n, cell: cell)
        let axis = field.dominant(x: 1280, y: 1450, radius: 320)
        XCTAssertEqual(axis?.system, 0)
        if let h = axis?.heading {
            XCTAssertEqual(Double(h.x), 0.26, accuracy: 0.03)
            XCTAssertEqual(Double(h.y), 0.97, accuracy: 0.03)
        }
        // Far inland, no surf: no lip.
        XCTAssertNil(field.dominant(x: 1280, y: 2400, radius: 320))
    }

    /// The lip's rows: one every row period on a fixed clock; a jump or reset starts over with
    /// the last rows a lifetime back.
    func testLipEmissionScheduleKeepsAFixedClock() {
        var clock = SurfLip.EmissionClock()
        let first = clock.step(time: 100.0)
        XCTAssertTrue(first.reset)
        XCTAssertEqual(first.count, SurfLip.rows)
        let next = clock.step(time: 100.0 + 2.5 * SurfLip.rowPeriod)
        XCTAssertFalse(next.reset)
        XCTAssertEqual(next.count, 2)
        XCTAssertEqual(next.first, first.first + Int64(SurfLip.rows))
        let same = clock.step(time: 100.0 + 2.5 * SurfLip.rowPeriod)
        XCTAssertEqual(same.count, 0)
        let back = clock.step(time: 50)
        XCTAssertTrue(back.reset)
    }

    /// The bed's slope along each swell's heading, baked once (SurfLip.bedSlopes) from the
    /// depth the surf breaks against: on a plane beach it is the beach's slope, the way the waves run.
    func testBedSlopeIsBakedAlongEachHeading() {
        let n = 64, cell: Float = 40
        var texels = [SIMD4<Float>](repeating: .zero, count: n * n)
        for j in 0..<n {
            for i in 0..<n {
                let y = (Float(j) + 0.5) * cell, x = (Float(i) + 0.5) * cell
                // Shallowing northward 1:40; system 0 runs north, system 1 east (along the beach).
                texels[j * n + i] = SIMD4(y / 50, x / 50, 0, max((2400 - y) / 40, 0))
            }
        }
        var smoothed = Grid(width: 32, height: 32, cellSize: 80)
        for k in 0..<smoothed.values.count { smoothed.values[k] = 100 }
        let slopes = SurfLip.bedSlopes(texels: texels, width: n, smoothedDepth: smoothed, domain: Float(n) * cell)
        let mid = slopes[55 * n + 30]   // 4.5 m deep
        XCTAssertEqual(Double(mid.x), 1.0 / 40, accuracy: 2e-3)
        XCTAssertEqual(Double(mid.y), 0, accuracy: 2e-3)
    }
}
