import Metal
import XCTest
import IslandCore
@testable import Engine

/// The surf's clocks (OceanShaders: surfBreakers, surfCrest, swashPhase) probed on the GPU along
/// a train of waves: each wave's foam, height and swash run on unbroken from its own front to
/// the next one's, with no seam between (a seam half a wave behind each front cut the
/// whitewater into square-ended slabs).
final class SurfShaderTests: XCTestCase {
    static let probeSource = """
    kernel void surfProbe(constant OceanParams &o [[buffer(0)]], device float4 *out [[buffer(1)]],
                          constant float4 &probe [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float arrival = probe.x + float(i) * probe.y;
        float2 xy = probe.zw;
        float front, n, age, wave;
        // A wave train breaking on and off from wave to wave (Hs/h about the set's threshold).
        float foam = surfBreakers(o, arrival, 60.0, 1.0, 2.0, 0.03, o.waves.x, xy, 0.0, 0.01, 0.0, 1.0, front, age, wave);
        float z = surfCrest(o, arrival, 60.0, 1.0, 2.0, 0.03, o.waves.x, xy, 0.01, o.time.x, 0.0, 1.0).z;
        float phase = swashPhase(arrival, o.waves.x, xy, o.time.x, n);
        out[i] = float4(foam, front, z, phase);
    }
    """

    /// Each probe: foam, roller, crest height and swash phase, every 1e-4 period over 8 periods.
    func probe(at xy: SIMD2<Float>) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let source = terrainShaderSource + floraShaderSource + groundCoverShaderSource + oceanShaderSource
            + SurfShaderTests.probeSource
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "surfProbe")!)
        let paramsSize = MemoryLayout<OceanRenderer.Params>.stride
        let params = device.makeBuffer(length: paramsSize, options: .storageModeShared)!
        memset(params.contents(), 0, paramsSize)
        let time = MemoryLayout<OceanRenderer.Params>.offset(of: \.time)!
        let waves = MemoryLayout<OceanRenderer.Params>.offset(of: \.waves)!
        params.contents().storeBytes(of: SIMD4<Float>(1000, 1000, 1, 0), toByteOffset: time, as: SIMD4<Float>.self)
        params.contents().storeBytes(of: SIMD4<Float>(8, 13, Float(WaveRefraction.breakerIndex), 7),
                                     toByteOffset: waves, as: SIMD4<Float>.self)
        let count = 80_000
        let out = device.makeBuffer(length: count * 16, options: .storageModeShared)!
        var probe = SIMD4<Float>(40, 1e-4, xy.x, xy.y)
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(params, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&probe, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    let spots: [SIMD2<Float>] = [SIMD2(66480, 31367), SIMD2(1234.5, 8765.4), SIMD2(70011, 25037)]

    func testWhitewaterJumpsOnlyAtEachFront() throws {
        for xy in spots {
            let s = try probe(at: xy)
            // A jump in the foam: either the front arriving (the roller, from nearly nothing to
            // its full cover) or a seam. Over 8 periods there are at most 8 fronts.
            var jumps = 0
            for i in 1..<s.count where abs(s[i].x - s[i - 1].x) > 0.15 {
                jumps += 1
                // At a front, the foam on its brighter side is all roller.
                let bright = s[i].x > s[i - 1].x ? s[i] : s[i - 1]
                XCTAssertGreaterThanOrEqual(bright.y, 0.9 * bright.x, "seam in the foam at \(xy), probe \(i)")
            }
            XCTAssertLessThanOrEqual(jumps, 9, "at \(xy)")
        }
    }

    func testCrestHeightIsContinuous() throws {
        for xy in spots {
            let s = try probe(at: xy)
            // The steepest face moves a few millimetres per probe (6 mm of travel): a breaking crest
            // ~1 m high (0.55 of its ~2.5 m break depth) over a face ~2 m long (surfShape's 0.7 of
            // 0.05 × 60 m) is a slope of ~0.5. A seam would jump by centimetres.
            let worst = (1..<s.count).map { abs(s[$0].z - s[$0 - 1].z) }.max()!
            XCTAssertLessThan(worst, 0.004, "a step in the surf's crests at \(xy)")
        }
    }

    func testSwashPhaseOnlyWraps() throws {
        for xy in spots {
            let s = try probe(at: xy)
            for i in 1..<s.count {
                let d = abs(s[i].w - s[i - 1].w)
                XCTAssertTrue(d < 0.001 || d > 0.9, "swash phase jumps by \(d) at \(xy), probe \(i)")
                if !(d < 0.001 || d > 0.9) { break }
            }
        }
    }
}
