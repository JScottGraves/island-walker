import Metal
import XCTest
import IslandCore
@testable import Engine

/// The whitecaps' foam (OceanShaders: oceanFoam, OceanRenderer.FoamClock): each cascade's foam
/// accumulates where its crests fold and fades over seconds after, at the same rate whatever
/// the frame rate, starting over after a jump in time; on the open trade-wind sea a small share
/// of it is white at any moment.
final class OceanFoamTests: XCTestCase {
    struct GPU {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let library: MTLLibrary
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
        }
    }

    /// The mean whiteness (×1000, summed) of the foam over a 256 m square of open sea, as the
    /// water shades it close up (oceanFoamCover; every cascade at full strength, in deep water).
    static let probeSource = """
    kernel void oceanFoamCoverProbe(constant OceanParams &o [[buffer(0)]],
                                    texture2d_array<float> foam [[texture(0)]],
                                    device atomic_uint *sum [[buffer(2)]],
                                    uint2 g [[thread_position_in_grid]]) {
        float2 xy = float2(g) * 0.25 + 0.125 + float2(3000.0, 5000.0);
        float white = oceanFoamCover(o, foam, xy, 0.05, 0.05, float4(1.0), 100.0);
        atomic_fetch_add_explicit(sum, uint(white * 1000.0 + 0.5), memory_order_relaxed);
    }
    """

    func gpu() throws -> GPU {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let source = terrainShaderSource + floraShaderSource + groundCoverShaderSource + oceanShaderSource
            + OceanFoamTests.probeSource
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        return GPU(device: device, queue: device.makeCommandQueue()!, library: library)
    }

    /// A 256² array texture of `slices`, shared so the test can fill and read it.
    func maps(_ device: MTLDevice, format: MTLPixelFormat, slices: Int) -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = format
        d.width = OceanSpectrum.resolution
        d.height = OceanSpectrum.resolution
        d.arrayLength = slices
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        return device.makeTexture(descriptor: d)!
    }

    /// Fills every slice of an rgba16Float map with one value.
    func fill(_ t: MTLTexture, _ v: SIMD4<Float>) {
        let n = t.width
        let texels = [SIMD4<Float16>](repeating: SIMD4<Float16>(v), count: n * n)
        for s in 0..<t.arrayLength {
            texels.withUnsafeBytes { raw in
                t.replace(region: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0, slice: s,
                          withBytes: raw.baseAddress!, bytesPerRow: n * 8, bytesPerImage: n * n * 8)
            }
        }
    }

    /// Foam values of one slice.
    func read(_ t: MTLTexture, slice: Int) -> [Float] {
        let n = t.width
        var values = [Float](repeating: 0, count: n * n)
        values.withUnsafeMutableBytes { raw in
            t.getBytes(raw.baseAddress!, bytesPerRow: n * 4, bytesPerImage: n * n * 4,
                       from: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0, slice: slice)
        }
        return values
    }

    /// Params as OceanRenderer's for the default sea (only what the waves and the foam read).
    func params(_ spectrum: OceanSpectrum) -> OceanRenderer.Params {
        var p = OceanRenderer.Params(
            invPatch: SIMD4(spectrum.patchSizes.map { Float(1 / $0) }), meanK: .zero, slopeXX: .zero, slopeYY: .zero,
            slopeXY: .zero, remainder: .zero, origin: (.zero, .zero), morphStart: (.zero, .zero, .zero, .zero),
            morphEnd: (.zero, .zero, .zero, .zero), grid: SIMD4(0.5, 1, 1, Float(spectrum.choppiness)), time: .zero,
            absorption: .zero, scattering: .zero, sss: .zero, surf: .zero, surfFrom: .zero, waves: .zero, breaking: .zero)
        for c in 0..<OceanSpectrum.cascadeCount {
            let m = spectrum.modes(cascade: c, seed: 0x0A11_0CEA)
            p.slopeXX[c] = Float(m.slopeXX)
            p.slopeYY[c] = Float(m.slopeYY)
            p.slopeXY[c] = Float(m.slopeXY)
        }
        return p
    }

    /// One foam step over every cascade with foam: `step` as OceanRenderer.FoamClock hands it.
    func foamStep(_ g: GPU, _ pipeline: MTLComputePipelineState, params: inout OceanRenderer.Params,
                  step: OceanRenderer.FoamStep, derivatives: MTLTexture, displacement: MTLTexture,
                  from previous: MTLTexture, to foam: MTLTexture) {
        var s = step.uniform
        let cmd = g.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&params, length: MemoryLayout<OceanRenderer.Params>.stride, index: 0)
        enc.setBytes(&s, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
        enc.setTextures([derivatives, displacement, previous, foam], range: 0..<4)
        let n = OceanSpectrum.resolution
        enc.dispatchThreads(MTLSize(width: n, height: n, depth: OceanRenderer.foamCascades),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// A fold held for half a second, then open water: the foam outlasts the fold by seconds,
    /// fading as it goes, and is gone a little later.
    func testFoamFadesOverSecondsAfterTheFold() throws {
        let g = try gpu()
        let pipeline = try g.pipeline("oceanFoam")
        var p = params(OceanSpectrum(fetch: 40_000, windTowards: 5 * .pi / 4))
        let derivatives = maps(g.device, format: .rgba16Float, slices: 4)
        let displacement = maps(g.device, format: .rgba16Float, slices: 4)
        var foam = [maps(g.device, format: .r32Float, slices: 3), maps(g.device, format: .r32Float, slices: 3)]
        var clock = OceanRenderer.FoamClock()
        // Folded hard: λ∂Dx/∂x = λ∂Dy/∂y = −0.8 (J = 0.04).
        fill(derivatives, SIMD4(0, 0, -0.8, -0.8))
        fill(displacement, .zero)
        var t = 100.0
        var history: [Float] = []
        for frame in 0..<(60 * 25) {
            if frame == 30 { fill(derivatives, .zero) }
            let step = clock.step(time: t, updated: 0b111)
            foamStep(g, pipeline, params: &p, step: step, derivatives: derivatives, displacement: displacement,
                     from: foam[0], to: foam[1])
            foam.swapAt(0, 1)
            if frame % 60 == 59 { history.append(read(foam[0], slice: 1)[1000]) }
            t += 1.0 / 60
        }
        // history[k]: k + 1 s in, k + 0.5 s after the fold.
        XCTAssertGreaterThan(history[0], 0.6, "fresh foam dense")
        XCTAssertGreaterThan(history[4], 0.25, "still there 4.5 s on: \(history)")
        XCTAssertLessThan(history[4], history[1], "fading: \(history)")
        XCTAssertLessThan(history[24], 0.05, "gone after 25 s: \(history)")
    }

    /// The same fold run at 30 and at 120 frames a second leaves the same foam.
    func testFadeIsIndependentOfFrameRate() throws {
        let g = try gpu()
        let pipeline = try g.pipeline("oceanFoam")
        var p = params(OceanSpectrum(fetch: 40_000, windTowards: 5 * .pi / 4))
        let derivatives = maps(g.device, format: .rgba16Float, slices: 4)
        let displacement = maps(g.device, format: .rgba16Float, slices: 4)
        fill(displacement, .zero)
        var results: [Float] = []
        for fps in [30.0, 120.0] {
            var foam = [maps(g.device, format: .r32Float, slices: 3), maps(g.device, format: .r32Float, slices: 3)]
            var clock = OceanRenderer.FoamClock()
            fill(derivatives, SIMD4(0, 0, -0.8, -0.8))
            var t = 50.0
            for frame in 0..<Int(6 * fps) {
                if frame == Int(0.5 * fps) { fill(derivatives, .zero) }
                let step = clock.step(time: t, updated: 0b111)
                foamStep(g, pipeline, params: &p, step: step, derivatives: derivatives, displacement: displacement,
                         from: foam[0], to: foam[1])
                foam.swapAt(0, 1)
                t += 1 / fps
            }
            results.append(read(foam[0], slice: 2)[777])
        }
        XCTAssertEqual(results[0], results[1], accuracy: 0.03 * max(results[0], 0.05), "\(results)")
    }

    /// A jump in time (a new view at another hour, the clock moved on by hours) starts the foam
    /// over from the folds of that moment; a pause (no time passing) keeps it as it is.
    func testFoamClockResetsOnJumpsAndHoldsOnPauses() {
        var clock = OceanRenderer.FoamClock()
        XCTAssertTrue(clock.step(time: 10, updated: 0b111).reset, "first frame")
        let normal = clock.step(time: 10 + 1.0 / 60, updated: 0b111)
        XCTAssertFalse(normal.reset)
        XCTAssertEqual(normal.dt, 1.0 / 60, accuracy: 1e-6)
        let paused = clock.step(time: 10 + 1.0 / 60, updated: 0b111)
        XCTAssertFalse(paused.reset)
        XCTAssertEqual(paused.dt, 0)
        // A long hitch is clamped, not taken whole.
        let hitch = clock.step(time: 10.5, updated: 0b111)
        XCTAssertFalse(hitch.reset)
        XCTAssertLessThanOrEqual(hitch.dt, 0.1)
        XCTAssertTrue(clock.step(time: 3600, updated: 0b111).reset, "an hour on")
        XCTAssertTrue(clock.step(time: 3599, updated: 0b111).reset, "back in time")
        // A cascade updated every other frame decays by the time since its own last update.
        var turns = OceanRenderer.FoamClock()
        _ = turns.step(time: 0, updated: 0b111)
        _ = turns.step(time: 1.0 / 60, updated: 0b110)
        let both = turns.step(time: 2.0 / 60, updated: 0b111)
        XCTAssertEqual(both.cascadeDt(0), 2.0 / 60, accuracy: 1e-6)
        XCTAssertEqual(both.cascadeDt(1), 1.0 / 60, accuracy: 1e-6)
    }

    /// The default trade-wind sea run through the FFT for half a minute: a small share of the
    /// open sea is white (whitecaps cover ~0.5–1% of a sea in a 7–10 m/s trade wind).
    func testOpenSeaCoverage() throws {
        let g = try gpu()
        let rows = try g.pipeline("oceanRows"), columns = try g.pipeline("oceanColumns")
        let foamPipeline = try g.pipeline("oceanFoam"), cover = try g.pipeline("oceanFoamCoverProbe")
        let spectrum = OceanSpectrum(fetch: 40_000, windTowards: 5 * .pi / 4)
        var p = params(spectrum)
        let n = OceanSpectrum.resolution, cascades = OceanSpectrum.cascadeCount
        var h0 = [SIMD4<Float>](), w = [Float]()
        for c in 0..<cascades {
            let m = spectrum.modes(cascade: c, seed: 0x0A11_0CEA)
            h0 += m.amplitudes
            w += m.omegas
        }
        let a = g.device.makeBuffer(bytes: h0, length: h0.count * 16, options: .storageModeShared)!
        let o = g.device.makeBuffer(bytes: w, length: w.count * 4, options: .storageModeShared)!
        let work = g.device.makeBuffer(length: cascades * 4 * n * n * 8, options: .storageModePrivate)!
        let displacement = maps(g.device, format: .rgba16Float, slices: 4)
        let derivatives = maps(g.device, format: .rgba16Float, slices: 4)
        let moments = maps(g.device, format: .rgba16Float, slices: 4)
        var foam = [maps(g.device, format: .r32Float, slices: 3), maps(g.device, format: .r32Float, slices: 3)]
        let sums = g.device.makeBuffer(length: 16, options: .storageModeShared)!
        var clock = OceanRenderer.FoamClock()
        var t = 200.0
        var samples: [Double] = []
        for frame in 0..<(30 * 40) {
            p.time = SIMD4(Float(t.truncatingRemainder(dividingBy: OceanSpectrum.repeatPeriod)), 0, 0, 0)
            var updated = SIMD4<UInt32>(0, 1, 2, 3)
            var step = clock.step(time: t, updated: 0b111).uniform
            let cmd = g.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rows)
            enc.setBytes(&p, length: MemoryLayout<OceanRenderer.Params>.stride, index: 0)
            enc.setBuffer(a, offset: 0, index: 1)
            enc.setBuffer(o, offset: 0, index: 2)
            enc.setBuffer(work, offset: 0, index: 3)
            enc.setBytes(&updated, length: 16, index: 4)
            let group = MTLSize(width: 128, height: 1, depth: 1)
            enc.dispatchThreadgroups(MTLSize(width: n, height: cascades, depth: 1), threadsPerThreadgroup: group)
            enc.setComputePipelineState(columns)
            enc.setBuffer(work, offset: 0, index: 1)
            enc.setTextures([displacement, derivatives, moments], range: 0..<3)
            enc.dispatchThreadgroups(MTLSize(width: n, height: cascades, depth: 1), threadsPerThreadgroup: group)
            enc.setComputePipelineState(foamPipeline)
            enc.setBytes(&step, length: 16, index: 1)
            enc.setTextures([derivatives, displacement, foam[0], foam[1]], range: 0..<4)
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: OceanRenderer.foamCascades),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            if frame >= 30 * 20 && frame % 30 == 0 {
                // The mean whiteness over a 256 m square of open sea (oceanFoamCover, as the
                // water shades it close up).
                memset(sums.contents(), 0, 16)
                enc.setComputePipelineState(cover)
                enc.setTexture(foam[1], index: 0)
                enc.setBuffer(sums, offset: 0, index: 2)
                enc.dispatchThreads(MTLSize(width: 1024, height: 1024, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            }
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            if frame >= 30 * 20 && frame % 30 == 0 {
                let s = sums.contents().load(as: UInt32.self)
                samples.append(Double(s) / 1000.0 / Double(1024 * 1024))
            }
            foam.swapAt(0, 1)
            t += 1.0 / 30
        }
        let mean = samples.reduce(0, +) / Double(samples.count)
        print("cascade σ of 1 − J:", (0..<3).map { p.grid.w * (p.slopeXX[$0] + p.slopeYY[$0]).squareRoot() })
        print(String(format: "open-sea whitecap cover %.3f%% (samples %@)", 100 * mean,
                     samples.map { String(format: "%.2f", 100 * $0) }.joined(separator: " ")))
        XCTAssertGreaterThan(mean, 0.004, "whitecaps cover ≥ 0.4% of the open sea")
        XCTAssertLessThan(mean, 0.015, "whitecaps cover ≤ 1.5% of the open sea")
    }
}
