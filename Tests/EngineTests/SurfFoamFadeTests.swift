import Metal
import XCTest
import IslandCore
@testable import Engine

/// How long the surf's trailing foam lasts (OceanShaders: surfBreakers, surfFoam), on the
/// foam's own clock (seconds since its wave's front passed). The dense whitewater right behind
/// the roller stays; the trail thins to lace within ~5 s and is gone well before the next wave,
/// so between waves the surf zone is mostly clear water, not a standing web of rafts.
final class SurfFoamFadeTests: XCTestCase {
    static let probeSource = """
    kernel void trailProbe(constant OceanParams &o [[buffer(0)]], device float4 *out [[buffer(1)]],
                           constant float4 &probe [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float arrival = probe.x + float(i) * probe.y;
        float front, age, wave;
        float foam = surfBreakers(o, arrival, probe.z, 1.0, 2.0, 0.03, o.waves.x, float2(66480.0, 31367.0), 0.0, 0.01, 0.0,
                                  1.0, front, age, wave);
        out[i] = float4(foam, front, age, wave);
    }

    kernel void fieldProbe(constant OceanParams &o [[buffer(0)]], device float4 *out [[buffer(1)]],
                           constant float4 &probe [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float2 xy = probe.zw + float2(float(i % 512u), float(i / 512u)) * 0.23;
        float roller;
        // Waves that all break (Hs well over the breaker depth), 60 m apart.
        float4 setting = float4(2.0, 0.0, 1.5, 1.0);
        float4 crests = float4(probe.x, 0.0, 60.0, 60.0);
        float foam = surfFoam(o, setting, crests, float2(0.0, 1.0), float2(0.03), xy, 1.5, 0.05, 0.0, 0.0, roller);
        float front, age, wave;
        surfBreakers(o, probe.x, 60.0, 2.0, 1.5, 0.03, o.waves.x, xy, groundNoise(xy / 2.5 + 4.1) - 0.5, 0.05, 0.0, 1.0,
                     front, age, wave);
        out[i] = float4(foam, roller, age, wave);
    }
    """

    func run(_ kernel: String, period: Float, probe: SIMD4<Float>, count: Int) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let source = terrainShaderSource + floraShaderSource + groundCoverShaderSource + oceanShaderSource
            + SurfFoamFadeTests.probeSource
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: kernel)!)
        let paramsSize = MemoryLayout<OceanRenderer.Params>.stride
        let params = device.makeBuffer(length: paramsSize, options: .storageModeShared)!
        memset(params.contents(), 0, paramsSize)
        let time = MemoryLayout<OceanRenderer.Params>.offset(of: \.time)!
        let waves = MemoryLayout<OceanRenderer.Params>.offset(of: \.waves)!
        params.contents().storeBytes(of: SIMD4<Float>(1000, 1000, 1, 0), toByteOffset: time, as: SIMD4<Float>.self)
        params.contents().storeBytes(of: SIMD4<Float>(period, 13, Float(WaveRefraction.breakerIndex), 7),
                                     toByteOffset: waves, as: SIMD4<Float>.self)
        let out = device.makeBuffer(length: count * 16, options: .storageModeShared)!
        var p = probe
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(params, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&p, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let ptr = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// The trail's cover against the time since its front passed, for the trade swell's period
    /// and a long north swell's (whose crests are further apart, so a trail fading over metres
    /// lasts longer in seconds).
    func testTrailThinsWithinSecondsAndIsGoneBeforeTheNextWave() throws {
        for (period, spacing) in [(Float(8), Float(60)), (Float(13), Float(110))] {
            let s = try run("trailProbe", period: period, probe: SIMD4(40, 1e-4, spacing, 0), count: 80_000)
            let dense = s.filter { $0.z > 0.3 && $0.z < 1.2 }.map(\.x).max() ?? 0
            XCTAssertGreaterThan(dense, 0.6, "the whitewater right behind the roller (period \(period))")
            let lace = s.filter { $0.z >= 4.5 && $0.z < 6.0 }.map(\.x).max() ?? 1
            XCTAssertLessThan(lace, 0.2, "trail 4.5–6 s behind its front (period \(period))")
            let late = s.filter { $0.z >= 6.0 }.map(\.x).max() ?? 1
            XCTAssertLessThan(late, 0.04, "trail over 6 s behind its front (period \(period))")
        }
    }

    /// Over a 118 m patch of surf the foam an age behind its front: dense right behind the
    /// roller, and once old, mostly clear water with a few streaks (the rafts' web drawn at the
    /// trail's thin cover made a uniform lattice over the whole surf zone).
    func testOldFoamLeavesMostlyClearWater() throws {
        let period: Float = 8
        var young: [Float] = [], old: [Float] = []
        for target in stride(from: Float(0.5), through: 7.5, by: 0.5) {
            let arrival = 1000 / period - target / period
            let s = try run("fieldProbe", period: period, probe: SIMD4(arrival, 0, 20_311, 7_120), count: 512 * 512)
            for v in s {
                if v.z > 0.3 && v.z < 1.0 { young.append(v.x) }
                if v.z >= 5.0 && v.z < 7.8 { old.append(v.x) }
            }
        }
        XCTAssertGreaterThan(young.count, 10_000)
        XCTAssertGreaterThan(old.count, 10_000)
        let youngMean = young.reduce(0, +) / Float(young.count)
        let oldMean = old.reduce(0, +) / Float(old.count)
        let oldCovered = Float(old.filter { $0 > 0.3 }.count) / Float(old.count)
        XCTAssertGreaterThan(youngMean, 0.5, "dense whitewater right behind the roller")
        XCTAssertLessThan(oldMean, 0.04, "foam 5 s and more behind its front")
        XCTAssertLessThan(oldCovered, 0.08, "share of old foam drawn over 0.3")
    }
}
