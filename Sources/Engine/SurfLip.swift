#if canImport(MetalKit)
import Metal
import simd
import IslandCore

/// The lips of plunging breakers (research/waves.md, item 4; SurfLipShaders), after Thürey et
/// al., "Real-time Breaking Waves for Shallow Water Simulations" (2007), on the surf's own clock.
///
/// Near the camera a wave line runs across the dominant swell's heading (`slots` points 1 m
/// apart). Every `rowPeriod` each point traces its ray in along the heading for the crest that
/// is throwing its lip (the surf's break progress 0.3…0.85, plunging) and launches a particle
/// from the crest's thrown-forward tip, faster than the wave runs. The last `rows` rows are kept
/// in a ring; the particles fly ballistically (their position is their launch's, worked out at
/// any moment), and are drawn joined row to row and point to point as a thin sheet of water. A
/// particle that meets the water dies, throwing spray (billboards) and leaving foam on a small
/// world-fixed grid that the whitewater reads (lipFoamAt), so the bore starts where the lip lands.
/// Everything runs on the GPU in fixed-size buffers; with no plunging surf near the camera
/// nothing is dispatched or drawn.
final class SurfLip {
    /// LipParams in SurfLipShaders.
    struct Params {
        var axis = SIMD4<Float>(0, 1, -1, 0)
        var frame = SIMD4<Float>(0, 0, Float(SurfLip.rowPeriod), 0)
        var march = SIMD4<Float>(0, 0, 0, 0)
        var slots = SIMD4<Int32>(0, Int32(SurfLip.slots), Int32(SurfLip.rows), 0)
        var rows = SIMD4<Int32>(0, 0, 0, 0)
        var emit = SIMD4<Float>(0, 0, 0, 1)
    }

    /// Points along the wave line (1 m apart) and rows of particles kept (one every row period).
    static let slots = 512
    static let rows = 32
    static let rowPeriod = 1.0 / 30
    /// A particle's longest flight (s), within the ring's span.
    static let lifetime: Float = 1.0
    static let sprayCapacity = 16384
    static let sprayLifetime: Float = 1.4
    /// The impact foam grid: cells (m) and side, wrapped in the world (±200 m around the camera).
    static let foamCell: Float = 0.8
    static let foamSide = 512
    /// Seconds the impact foam fades over (e-folding).
    static let foamFade = 2.5
    /// The trace's reach along the heading either side of the camera (m).
    static let reach: Float = 280

    /// The surf's wave texels on the CPU (WaveRefraction.surfTexels), for picking the wave line's
    /// system and heading near the camera.
    struct Field {
        let texels: [SIMD4<Float>]
        let width: Int
        let cell: Float

        /// The swell bringing the most breaking energy (Σ Hs²) into the surf (0.3–8 m deep)
        /// within `radius` metres, and its mean heading there; nil without surf worth a lip.
        func dominant(x: Double, y: Double, radius: Double) -> (system: Int, heading: SIMD2<Float>)? {
            let r = Int((radius / Double(cell)).rounded(.up))
            let ci = Int(x / Double(cell)), cj = Int(y / Double(cell))
            var weight: [Float] = [0, 0]
            var sum: [SIMD2<Float>] = [.zero, .zero]
            func heights(_ t: SIMD4<Float>) -> (Float, Float) {
                let bits = t.z.bitPattern
                return (Float(Float16(bitPattern: UInt16(bits & 0xFFFF))), Float(Float16(bitPattern: UInt16(bits >> 16))))
            }
            for j in Swift.max(cj - r, 1)...Swift.min(cj + r, width - 2) {
                for i in Swift.max(ci - r, 1)...Swift.min(ci + r, width - 2) {
                    let dx = Double(i - ci), dy = Double(j - cj)
                    if (dx * dx + dy * dy) * Double(cell * cell) > radius * radius { continue }
                    let t = texels[j * width + i]
                    guard t.w > 0.3, t.w < 8 else { continue }
                    let (h0, h1) = heights(t)
                    for s in 0..<2 {
                        let hs = s == 0 ? h0 : h1
                        guard hs > 0.2 else { continue }
                        let e = texels[j * width + i + 1][s] - texels[j * width + i - 1][s]
                        let n = texels[(j + 1) * width + i][s] - texels[(j - 1) * width + i][s]
                        let g = SIMD2<Float>(e, n)
                        let length = simd_length(g)
                        guard length.isFinite, length > 1e-6 else { continue }
                        weight[s] += hs * hs
                        sum[s] += hs * hs * g / length
                    }
                }
            }
            let best = weight[0] >= weight[1] ? 0 : 1
            // A few texels of surf, coming in from about one way.
            guard weight[best] > 0.15, simd_length(sum[best]) > 0.3 * weight[best] else { return nil }
            return (best, simd_normalize(sum[best]))
        }
    }

    /// The rows' clock: one every row period on a fixed grid of moments, so rows emitted in
    /// different frames line up whatever the frame rate. A jump back or by more than the ring
    /// starts over with the last `rows` moments.
    struct EmissionClock {
        var last: Int64?
        struct Step {
            var first: Int64
            var count: Int
            var reset: Bool
        }

        mutating func step(time: Double) -> Step {
            let now = Int64((time / SurfLip.rowPeriod).rounded(.down))
            if let last, now >= last, now - last <= Int64(SurfLip.rows) {
                self.last = now
                return Step(first: last + 1, count: Int(now - last), reset: false)
            }
            last = now
            return Step(first: now - Int64(SurfLip.rows) + 1, count: SurfLip.rows, reset: true)
        }
    }

    let field: Field
    let enabled: Bool
    /// Spray drawn (tuning: spray=0 leaves it out, to look at the sheets alone).
    var sprayEnabled = true
    let particles: MTLBuffer
    let spray: MTLBuffer
    let sprayHead: MTLBuffer
    /// The impact foam (uint, 1/4096), read by oceanFragment at buffer 9.
    let foam: MTLBuffer
    let emitPipeline: MTLComputePipelineState
    let collidePipeline: MTLComputePipelineState
    let fadePipeline: MTLComputePipelineState
    let lipPipeline: MTLRenderPipelineState
    let sprayPipeline: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    var params = Params()
    var clock = EmissionClock()
    var axis: (system: Int, heading: SIMD2<Float>)?
    var picked: SIMD2<Double>?
    /// Whether this frame has a lip to work out and draw.
    private(set) var active = false
    var needsClear = true
    var lastTime: Double?
    var newest: Int64 = 0

    init(device: MTLDevice, library: MTLLibrary, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat, field: Field,
         enabled: Bool) throws {
        self.field = field
        self.enabled = enabled
        let particleStride = 48, sprayStride = 32
        guard let p = device.makeBuffer(length: SurfLip.slots * SurfLip.rows * particleStride, options: .storageModeShared),
              let s = device.makeBuffer(length: SurfLip.sprayCapacity * sprayStride, options: .storageModePrivate),
              let h = device.makeBuffer(length: 16, options: .storageModePrivate),
              let f = device.makeBuffer(length: SurfLip.foamSide * SurfLip.foamSide * 4, options: .storageModePrivate)
        else { throw RendererError.setup("surf lip buffers") }
        particles = p
        spray = s
        sprayHead = h
        foam = f
        func kernel(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw RendererError.setup("surf lip \(name)") }
            return try device.makeComputePipelineState(function: fn)
        }
        emitPipeline = try kernel("lipEmit")
        collidePipeline = try kernel("lipCollide")
        fadePipeline = try kernel("lipFoamFade")
        func blended(_ vertex: String, _ fragment: String) throws -> MTLRenderPipelineState {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = library.makeFunction(name: vertex)
            desc.fragmentFunction = library.makeFunction(name: fragment)
            desc.depthAttachmentPixelFormat = depthFormat
            let c = desc.colorAttachments[0]!
            c.pixelFormat = colorFormat
            c.isBlendingEnabled = true
            c.rgbBlendOperation = .add
            c.alphaBlendOperation = .add
            c.sourceRGBBlendFactor = .sourceAlpha
            c.destinationRGBBlendFactor = .oneMinusSourceAlpha
            c.sourceAlphaBlendFactor = .one
            c.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            return try device.makeRenderPipelineState(descriptor: desc)
        }
        lipPipeline = try blended("lipVertex", "lipFragment")
        sprayPipeline = try blended("sprayVertex", "sprayFragment")
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .lessEqual
        depth.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: depth) else { throw RendererError.setup("surf lip depth") }
        depthState = ds
    }

    /// For tuning (ISLAND_LIP_DEBUG): launched and flying particles, and the spread of their
    /// launch progress and heights, read back from the last finished frame.
    func debugSummary() -> String {
        let q = particles.contents().bindMemory(to: SIMD4<Float>.self, capacity: SurfLip.slots * SurfLip.rows * 3)
        var launched = 0, flying = 0
        var zs: [Float] = [], ps: [Float] = []
        for k in 0..<(SurfLip.slots * SurfLip.rows) {
            let start = q[3 * k], velocity = q[3 * k + 1], tag = q[3 * k + 2]
            guard start.w.isFinite else { continue }
            launched += 1
            if velocity.w > 1e29 { flying += 1 }
            zs.append(start.z)
            ps.append(tag.w)
        }
        // Quads the sheet would draw (as lipVertex: four flying corners of one wave).
        var quads = 0, sameRow = 0, sameSlot = 0
        let S = SurfLip.slots, R = SurfLip.rows
        func at(_ j: Int, _ age: Int) -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
            let slot = Int(params.slots.x) + j
            let column = ((slot % S) + S) % S
            let row = ((Int(params.rows.x) - age) % R + R) % R
            let k = row * S + column
            return (q[3 * k], q[3 * k + 1], q[3 * k + 2])
        }
        func live(_ p: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>), _ slot: Int, _ wave: Float) -> Bool {
            p.0.w.isFinite && Int(p.2.x.bitPattern) == slot && p.2.y == wave && params.frame.x >= p.0.w
                && (p.1.w > 1e29 || params.frame.x - p.1.w < 0.05)
        }
        for k in 0..<(R - 1) {
            for j in 0..<(S - 1) {
                let a = at(j, k), b = at(j + 1, k), c = at(j, k + 1)
                let slot = Int(params.slots.x) + j
                guard live(a, slot, a.2.y) else { continue }
                let lb = live(b, slot + 1, a.2.y), lc = live(c, slot, a.2.y)
                if lb { sameRow += 1 }
                if lc { sameSlot += 1 }
                if lb && lc && live(at(j + 1, k + 1), slot + 1, a.2.y) { quads += 1 }
            }
        }
        let axisText = axis.map { String(format: "system %d heading (%.2f, %.2f)", $0.system, $0.heading.x, $0.heading.y) } ?? "none"
        return String(format: "lip: %@, active %d, launched %d, flying %d, z %.2f…%.2f, p %.2f…%.2f; pairs along %d, in time %d, quads %d (now %.3f)",
                      axisText, active ? 1 : 0, launched, flying, zs.min() ?? 0, zs.max() ?? 0, ps.min() ?? 0, ps.max() ?? 0,
                      sameRow, sameSlot, quads, params.frame.x)
    }

    /// Forgets the lips (a new view or a jump in time).
    func reset() {
        clock = EmissionClock()
        lastTime = nil
        needsClear = true
    }

    /// Picks the wave line near `eye`, emits this frame's rows, flies the particles into the
    /// water (spray, foam) and fades the foam. `now` and `previous` are the sea's wrapped clock
    /// (OceanParams.time), `time` the same unwrapped.
    func update(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms, ocean: inout OceanRenderer.Params,
                terrainParams: TerrainRenderer.TerrainParams, shore: MTLTexture, waves: MTLTexture, eye: SIMD3<Double>,
                time: Double) {
        guard enabled else {
            active = false
            return
        }
        // The wave line's system and heading, picked again when the camera has moved on.
        let here = SIMD2(eye.x, eye.y)
        if picked == nil || simd_distance(picked!, here) > 40 {
            picked = here
            let pick = field.dominant(x: eye.x, y: eye.y, radius: Double(SurfLip.reach))
            if let pick {
                let turned = axis.map { $0.system != pick.system || simd_dot($0.heading, pick.heading) < cos(12 * Float.pi / 180) } ?? true
                if turned {
                    axis = pick
                    needsClear = true
                    clock = EmissionClock()
                }
            } else {
                axis = nil
            }
        }
        // Not near enough to the shore to see one: nothing to do (the summit costs nothing).
        guard let axis, eye.z - Double(uniforms.seaLevel) < 400 else {
            active = false
            return
        }
        active = true
        let wrap = OceanSpectrum.repeatPeriod
        var step = clock.step(time: time)
        let firstTime = (Double(step.first) * SurfLip.rowPeriod).truncatingRemainder(dividingBy: wrap)
        // Across the sea's wrap: start over.
        if firstTime + Double(step.count) * SurfLip.rowPeriod > wrap || needsClear {
            if !step.reset {
                clock = EmissionClock()
                step = clock.step(time: time)
            }
        }
        if step.reset { needsClear = true }
        if needsClear, let blit = cmd.makeBlitCommandEncoder() {
            blit.label = "surf lip clear"
            blit.fill(buffer: particles, range: 0..<particles.length, value: 0xFF)
            blit.fill(buffer: spray, range: 0..<spray.length, value: 0xFF)
            blit.fill(buffer: sprayHead, range: 0..<sprayHead.length, value: 0)
            blit.fill(buffer: foam, range: 0..<foam.length, value: 0)
            blit.endEncoding()
            needsClear = false
        }
        if step.count > 0 { newest = step.first + Int64(step.count) - 1 }
        let h = axis.heading, across = SIMD2<Float>(-h.y, h.x)
        let e = SIMD2<Float>(Float(eye.x), Float(eye.y))
        let dt = lastTime.map { Swift.min(Swift.max(time - $0, 0), 0.1) } ?? 0
        lastTime = time
        let ring = Int64(SurfLip.rows)
        params.axis = SIMD4(h.x, h.y, across.x, across.y)
        params.frame = SIMD4(ocean.time.x, ocean.time.y, Float(SurfLip.rowPeriod), uniforms.seaLevel)
        params.march = SIMD4(e.x, e.y, simd_dot(e, h) - SurfLip.reach, 2 * SurfLip.reach)
        params.slots = SIMD4(Int32((simd_dot(e, across) / 1).rounded()) - Int32(SurfLip.slots / 2), Int32(SurfLip.slots),
                             Int32(SurfLip.rows), Int32(axis.system))
        params.rows = SIMD4(Int32(((newest % ring) + ring) % ring), Int32(step.count),
                            Int32(((step.first % ring) + ring) % ring), step.reset ? 1 : 0)
        params.emit = SIMD4(Float((Double(step.first) * SurfLip.rowPeriod).truncatingRemainder(dividingBy: wrap)),
                            SurfLip.lifetime, SurfLip.sprayLifetime, Float(exp(-dt / SurfLip.foamFade)))
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "surf lip"
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
        enc.setBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.setBytes(&ocean, length: MemoryLayout<OceanRenderer.Params>.stride, index: 2)
        var t = terrainParams
        enc.setBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 3)
        enc.setBuffer(particles, offset: 0, index: 4)
        enc.setBuffer(spray, offset: 0, index: 5)
        enc.setBuffer(sprayHead, offset: 0, index: 6)
        enc.setBuffer(foam, offset: 0, index: 7)
        enc.setTextures([shore, waves], range: 0..<2)
        if step.count > 0 {
            enc.setComputePipelineState(emitPipeline)
            enc.dispatchThreads(MTLSize(width: SurfLip.slots, height: step.count, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        enc.setComputePipelineState(collidePipeline)
        enc.dispatchThreads(MTLSize(width: SurfLip.slots * SurfLip.rows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if dt > 0 {
            enc.setComputePipelineState(fadePipeline)
            enc.dispatchThreads(MTLSize(width: SurfLip.foamSide * SurfLip.foamSide, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// The sheet, then the spray, blended over the water in the shading pass (after the sea, with
    /// its parameters and sky environment bound).
    func draw(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms) {
        guard active else { return }
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)
        enc.setRenderPipelineState(lipPipeline)
        enc.setVertexBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.setVertexBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        enc.setVertexBuffer(particles, offset: 0, index: 3)
        enc.setVertexBuffer(spray, offset: 0, index: 4)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6 * (SurfLip.slots - 1) * (SurfLip.rows - 1))
        guard sprayEnabled else { return }
        enc.setRenderPipelineState(sprayPipeline)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6 * SurfLip.sprayCapacity)
    }
}
#endif
