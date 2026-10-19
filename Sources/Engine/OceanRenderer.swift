#if canImport(MetalKit)
import Metal
import simd
import IslandCore

/// The sea surface (research/ocean.md, step O2). Tessendorf FFT waves in four cascades of
/// 256² (OceanSpectrum: a JONSWAP trade-wind sea and a swell), evolved and transformed on the
/// GPU into displacement, derivative and slope-moment maps with mips (only the cascades some
/// pixel used lately, the two long ones in turn). A CDLOD
/// quadtree of grid nodes centred on the camera carries them, each level's grid fixed in
/// the world and morphing into the next, out past the far plane. The mesh is displaced in
/// the vertex shader, depth-tested against the terrain, and shaded with Fresnel reflection
/// of a sky environment, Cox–Munk glint over the slope variance no normal resolves, and the
/// sea floor (shaded here from the streamed heights, along the refracted view) seen through a
/// Beer–Lambert water column with single scattering.
///
/// Frame order (TerrainRenderer.encode): `update` before the depth prepass; `drawPrepass`
/// last in the depth prepass (writing the water's own motion, which motionVectors then leaves);
/// `drawShading` last in the shading pass, where the prepass left the water nearest.
final class OceanRenderer {
    /// OceanParams in OceanShaders.
    struct Params {
        var invPatch: SIMD4<Float>
        var meanK: SIMD4<Float>
        var slopeXX: SIMD4<Float>
        var slopeYY: SIMD4<Float>
        var slopeXY: SIMD4<Float>
        var remainder: SIMD4<Float>
        var origin: (SIMD4<Float>, SIMD4<Float>)
        var morphStart: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var morphEnd: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var grid: SIMD4<Float>
        var time: SIMD4<Float>
        var absorption: SIMD4<Float>
        var scattering: SIMD4<Float>
        var sss: SIMD4<Float>
        var surf: SIMD4<Float>
        var surfFrom: SIMD4<Float>
    }

    let spectrum: OceanSpectrum
    let tree: OceanQuadtree
    /// Sea nodes the terrain hides (from the overview heights) are not drawn.
    let occlusion: TerrainOcclusion
    let seaLevel: Float
    var params: Params
    let rowsPipeline: MTLComputePipelineState
    let columnsPipeline: MTLComputePipelineState
    let environmentPipeline: MTLComputePipelineState
    /// Depth prepass, writing the surface for the sun rays (with ray tracing) and the water's
    /// motion straight into the motion texture (with temporal antialiasing).
    let prepassPipeline: MTLRenderPipelineState
    let shadePipeline: MTLRenderPipelineState
    let writesMotion: Bool
    let amplitudes: MTLBuffer
    let omegas: MTLBuffer
    let work: MTLBuffer
    /// This frame's and last frame's displacement (for motion vectors), then the
    /// derivatives and slope moments.
    let displacement: [MTLTexture]
    let derivatives: MTLTexture
    let moments: MTLTexture
    /// The sky the sea reflects, as an upper-hemisphere octahedral map with mips.
    let environment: MTLTexture
    /// Over the domain, from the overview, for the surf: signed distance to the shoreline (m,
    /// positive at sea), and how exposed the water is to the swell and to the wind sea.
    let shore: MTLTexture
    var current = 0
    var previousValid = false
    var frameIndex = 0
    /// Which cascades the water's pixels used (a bit each), one buffer per frame in flight:
    /// cleared at the start of a frame, read back two frames later.
    let usageBuffers: [MTLBuffer]
    /// Cascades updated last frame.
    private(set) var updatedCascades = 0
    /// No wave moved this frame: last frame's displacement is this frame's.
    var motionless = false
    var environmentKey: EnvironmentKey?
    let nodeIndices: (full: MTLBuffer, half: MTLBuffer)
    let nodeBuffers: [MTLBuffer]
    var nodeBufferIndex = 0
    var nodeCounts = (full: 0, half: 0)
    static let maxNodes = 2048
    static let nodeQuads = 32
    /// Nodes drawn last frame, for the overlay.
    private(set) var drawnNodes = 0

    init(device: MTLDevice, library: MTLLibrary, constants: MTLFunctionConstantValues, colorFormat: MTLPixelFormat,
         depthFormat: MTLPixelFormat, rayTraced: Bool, motion: Bool, seaLevel: Float, domain: Double, overview: Grid,
         windTowards: Double,
         queue: MTLCommandQueue) throws {
        self.seaLevel = seaLevel
        occlusion = TerrainOcclusion(height: overview)
        let spectrum = OceanSpectrum(windTowards: windTowards)
        self.spectrum = spectrum
        let distance = ShoreDistance.signed(height: overview, seaLevel: seaLevel)
        let swellTowards = spectrum.swell.direction
        let swellExposure = ShoreDistance.exposure(height: overview, seaLevel: seaLevel,
                                                   from: SIMD2(-cos(swellTowards), -sin(swellTowards)))
        let seaExposure = ShoreDistance.exposure(height: overview, seaLevel: seaLevel,
                                                 from: SIMD2(-cos(windTowards), -sin(windTowards)))
        var texels = [Float](repeating: 0, count: distance.values.count * 4)
        for k in 0..<distance.values.count {
            texels[4 * k] = distance.values[k]
            texels[4 * k + 1] = swellExposure.values[k]
            texels[4 * k + 2] = seaExposure.values[k]
        }
        let shoreDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: distance.width,
                                                                 height: distance.height, mipmapped: false)
        shoreDesc.usage = [.shaderRead]
        shoreDesc.storageMode = .shared
        guard let shoreTexture = device.makeTexture(descriptor: shoreDesc) else { throw RendererError.setup("shore distance") }
        texels.withUnsafeBytes { raw in
            shoreTexture.replace(region: MTLRegionMake2D(0, 0, distance.width, distance.height), mipmapLevel: 0,
                                 withBytes: raw.baseAddress!, bytesPerRow: distance.width * 16)
        }
        shore = shoreTexture
        tree = OceanQuadtree(quads: OceanRenderer.nodeQuads, cellMetres: 0.5, centre: (domain / 2, domain / 2),
                             reach: domain / 2 + 160_000, rangeFactor: 2)
        func kernel(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw RendererError.setup("ocean \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        rowsPipeline = try kernel("oceanRows")
        columnsPipeline = try kernel("oceanColumns")
        environmentPipeline = try kernel("oceanSkyEnvironment")
        writesMotion = motion
        let desc = MTLRenderPipelineDescriptor()
        desc.depthAttachmentPixelFormat = depthFormat
        desc.vertexFunction = library.makeFunction(name: motion ? "oceanMotionVertex" : "oceanVertex")
        desc.colorAttachments[0].pixelFormat = rayTraced ? .rgba32Uint : .invalid
        switch (rayTraced, motion) {
        case (true, true): desc.fragmentFunction = library.makeFunction(name: "oceanSurfaceMotion")
        case (false, true): desc.fragmentFunction = library.makeFunction(name: "oceanMotion")
        case (true, false): desc.fragmentFunction = library.makeFunction(name: "oceanSurface")
        case (false, false): desc.fragmentFunction = nil
        }
        prepassPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.vertexFunction = library.makeFunction(name: "oceanVertex")
        desc.fragmentFunction = try library.makeFunction(name: "oceanFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        shadePipeline = try device.makeRenderPipelineState(descriptor: desc)

        // Initial amplitudes and frequencies of every cascade, once.
        let n = OceanSpectrum.resolution, cascades = OceanSpectrum.cascadeCount
        var h0 = [SIMD4<Float>](), w = [Float]()
        var heightVariance = 0.0
        var meanK = SIMD4<Float>(), sxx = SIMD4<Float>(), syy = SIMD4<Float>(), sxy = SIMD4<Float>()
        for c in 0..<cascades {
            let modes = spectrum.modes(cascade: c, seed: 0x0A11_0CEA)
            h0 += modes.amplitudes
            w += modes.omegas
            heightVariance += modes.expectedVariance
            meanK[c] = Float(modes.meanK)
            sxx[c] = Float(modes.slopeXX)
            syy[c] = Float(modes.slopeYY)
            sxy[c] = Float(modes.slopeXY)
        }
        guard let a = device.makeBuffer(bytes: h0, length: h0.count * 16, options: .storageModeShared),
              let o = device.makeBuffer(bytes: w, length: w.count * 4, options: .storageModeShared),
              let work = device.makeBuffer(length: cascades * 4 * n * n * 8, options: .storageModePrivate)
        else { throw RendererError.setup("ocean buffers") }
        amplitudes = a
        omegas = o
        self.work = work
        func maps() throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = .rgba16Float
            d.width = n
            d.height = n
            d.arrayLength = cascades
            d.mipmapLevelCount = Int(log2(Double(n))) + 1
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else { throw RendererError.setup("ocean maps") }
            return t
        }
        displacement = [try maps(), try maps()]
        derivatives = try maps()
        moments = try maps()
        let envDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 128, height: 128, mipmapped: true)
        envDesc.usage = [.shaderRead, .shaderWrite]
        envDesc.storageMode = .private
        guard let env = device.makeTexture(descriptor: envDesc) else { throw RendererError.setup("ocean environment") }
        environment = env

        func grid(_ quads: Int) -> MTLBuffer? {
            let row = quads + 1
            var idx = [UInt16]()
            idx.reserveCapacity(quads * quads * 6)
            for j in 0..<quads {
                for i in 0..<quads {
                    let a = UInt16(j * row + i), b = a + 1, c = a + UInt16(row), d = c + 1
                    idx.append(contentsOf: [a, b, d, a, d, c])
                }
            }
            return device.makeBuffer(bytes: idx, length: idx.count * 2, options: .storageModeShared)
        }
        guard let full = grid(OceanRenderer.nodeQuads), let half = grid(OceanRenderer.nodeQuads / 2) else {
            throw RendererError.setup("ocean grids")
        }
        nodeIndices = (full, half)
        usageBuffers = try (0..<3).map { _ in
            guard let b = device.makeBuffer(length: 16, options: .storageModeShared) else { throw RendererError.setup("ocean usage") }
            b.contents().storeBytes(of: UInt32(0xF), as: UInt32.self)
            return b
        }
        nodeBuffers = try (0..<3).map { _ in
            guard let b = device.makeBuffer(length: OceanRenderer.maxNodes * MemoryLayout<TerrainRenderer.NodeInstance>.stride,
                                            options: .storageModeShared) else { throw RendererError.setup("ocean nodes") }
            return b
        }

        func floats(_ a: [Double]) -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
            let v = (0..<16).map { $0 < a.count ? Float(Swift.min(a[$0], 1e30)) : 1e30 }
            return (SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                    SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15]))
        }
        guard tree.lodCount <= 16 else { throw RendererError.setup("ocean too large for the LOD tables") }
        let remainder = spectrum.unresolvedSlopeVariance
        let hs = 4 * heightVariance.squareRoot()
        params = Params(
            invPatch: SIMD4(spectrum.patchSizes.map { Float(1 / $0) }),
            meanK: meanK, slopeXX: sxx, slopeYY: syy, slopeXY: sxy,
            remainder: SIMD4(Float(remainder.upwind), Float(remainder.crosswind),
                             Float(cos(windTowards)), Float(sin(windTowards))),
            origin: (.zero, .zero),
            morphStart: floats(tree.morphStart), morphEnd: floats(tree.morphEnd),
            grid: SIMD4(Float(tree.cellMetres), Float(tree.lodCount), Float(hs), Float(spectrum.choppiness)),
            time: .zero,
            // Clear oceanic water (Jerlov I): pure-water absorption (Pope & Fry 1997) with a
            // little more in the blue; Rayleigh-like molecular scattering; few particles.
            absorption: SIMD4(0.45, 0.064, 0.017, 0),
            scattering: SIMD4(0.0008, 0.0019, 0.0045, 0.02),
            sss: SIMD4(0.05, 0.35, 0.3, 0.06),
            surf: SIMD4(Float(spectrum.windSea.measuredSignificantHeight), Float(spectrum.swell.measuredSignificantHeight),
                        Float(2 * Double.pi / spectrum.swell.peakOmega), 0),
            // Waves come from the opposite of where they travel.
            surfFrom: SIMD4(Float(-cos(spectrum.swell.direction)), Float(-sin(spectrum.swell.direction)),
                            Float(-cos(windTowards)), Float(-sin(windTowards))))
    }

    /// Picks this frame's sea nodes and runs the waves forward to `time` (seconds), then
    /// rebuilds the sky environment if the sun, the clouds or the camera moved enough.
    func update(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms, eye: SIMD3<Double>,
                viewProjection m: simd_float4x4, time: Double, skyView: MTLTexture, cloudMap: MTLTexture) {
        // Nodes, culled against the plain projection in camera-relative space.
        func row(_ i: Int) -> SIMD4<Double> {
            SIMD4(Double(m.columns.0[i]), Double(m.columns.1[i]), Double(m.columns.2[i]), Double(m.columns.3[i]))
        }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        let planes: [SIMD4<Double>] = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
        let rel = SIMD3<Double>(eye.x, eye.y, eye.z - Double(seaLevel))
        // A named, fully typed test: as a trailing closure CI's compiler can't type-check it in time.
        func visible(_ lo: SIMD3<Double>, _ hi: SIMD3<Double>) -> Bool {
            let a: SIMD3<Double> = lo - rel, b: SIMD3<Double> = hi - rel
            for p in planes {
                let x: Double = p.x > 0 ? b.x : a.x
                let y: Double = p.y > 0 ? b.y : a.y
                let z: Double = p.z > 0 ? b.z : a.z
                let side: Double = p.x * x + p.y * y + p.z * z + p.w
                if side < 0 { return false }
            }
            return true
        }
        // Then the nodes the terrain hides from the eye: every ray to nine points over the node
        // (at crest height) blocked.
        let crest: Double = Double(seaLevel) + tree.amplitude
        let world = SIMD3<Double>(eye.x, eye.y, eye.z)
        let selected = tree.select(eye: (rel.x, rel.y, rel.z), visible: visible).filter { node in
            var points: [SIMD3<Double>] = []
            for j in 0...2 {
                for i in 0...2 {
                    points.append(SIMD3<Double>(node.x + node.size * Double(i) / 2, node.y + node.size * Double(j) / 2, crest))
                }
            }
            return !occlusion.hides(eye: world, points: points)
        }
        nodeBufferIndex = (nodeBufferIndex + 1) % nodeBuffers.count
        let out = nodeBuffers[nodeBufferIndex].contents().bindMemory(to: TerrainRenderer.NodeInstance.self,
                                                                       capacity: OceanRenderer.maxNodes)
        let full = selected.filter { !$0.half }, half = selected.filter { $0.half }
        var count = 0
        for node in full + half where count < OceanRenderer.maxNodes {
            out[count] = TerrainRenderer.NodeInstance(origin: SIMD2(Float(node.x), Float(node.y)), lod: Int32(node.lod),
                                                      quads: Int32(node.half ? OceanRenderer.nodeQuads / 2 : OceanRenderer.nodeQuads))
            count += 1
        }
        let fullCount = Swift.min(full.count, count)
        nodeCounts = (fullCount, count - fullCount)
        drawnNodes = count

        // Each cascade's patch origin: whole patches, near the camera, so map coordinates
        // stay small.
        var origins = [Float]()
        for size in spectrum.patchSizes {
            origins += [Float((eye.x / size).rounded() * size), Float((eye.y / size).rounded() * size)]
        }
        params.origin = (SIMD4(origins[0], origins[1], origins[2], origins[3]), SIMD4(origins[4], origins[5], origins[6], origins[7]))
        params.time.x = Float(time.truncatingRemainder(dividingBy: OceanSpectrum.repeatPeriod))

        let n = OceanSpectrum.resolution
        // Only the cascades some pixel used two frames ago (the frame whose usage has surely
        // come back: from a summit or a ridge the short waves have faded into roughness
        // everywhere), and of those the two long-wave ones take turns, each every other frame
        // (their waves move little in a frame). The ones left out keep their last maps. After a
        // reset all four run.
        frameIndex += 1
        let usage: UInt32 = usageBuffers[(frameIndex + 1) % usageBuffers.count].contents().load(as: UInt32.self)
        var list: [UInt32] = []
        for c in 0..<OceanSpectrum.cascadeCount {
            let used: Bool = usage & (UInt32(1) << UInt32(c)) != 0
            let turn: Bool = c > 1 || c == frameIndex % 2
            if !previousValid || (used && turn) { list.append(UInt32(c)) }
        }
        updatedCascades = list.count
        // A frame that moves no wave keeps this frame's displacement as the last one too (no
        // motion of the water's own).
        motionless = list.isEmpty
        if !list.isEmpty && previousValid { current = 1 - current }
        // The sky environment only when what it shows has changed: the sun by ~0.1°, the cloud
        // field drifted by ~10 m, or the camera moved ~50 m under the cloud base.
        let key = EnvironmentKey(sun: SIMD3(uniforms.sunDirection), clock: Double(uniforms.cloudLayer.z),
                                 camera: SIMD3(uniforms.cameraPosition))
        let rebuild = environmentKey.map { !$0.matches(key) } ?? true
        if rebuild { environmentKey = key }
        let usageBuffer = usageBuffers[frameIndex % usageBuffers.count]
        if !list.isEmpty || rebuild, let enc = cmd.makeComputeCommandEncoder() {
            enc.label = "ocean waves"
            if !list.isEmpty {
                var updated = SIMD4<UInt32>(0, 0, 0, 0)
                for (k, c) in list.enumerated() { updated[k] = c }
                let group = MTLSize(width: 128, height: 1, depth: 1)
                let lines = MTLSize(width: n, height: list.count, depth: 1)
                enc.setComputePipelineState(rowsPipeline)
                enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
                enc.setBuffer(amplitudes, offset: 0, index: 1)
                enc.setBuffer(omegas, offset: 0, index: 2)
                enc.setBuffer(work, offset: 0, index: 3)
                enc.setBytes(&updated, length: 16, index: 4)
                enc.dispatchThreadgroups(lines, threadsPerThreadgroup: group)
                enc.setComputePipelineState(columnsPipeline)
                enc.setBuffer(work, offset: 0, index: 1)
                enc.setTextures([displacement[current], derivatives, moments], range: 0..<3)
                enc.dispatchThreadgroups(lines, threadsPerThreadgroup: group)
            }
            if rebuild {
                enc.setComputePipelineState(environmentPipeline)
                enc.setBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 0)
                enc.setTextures([skyView, cloudMap, environment], range: 0..<3)
                enc.dispatchThreads(MTLSize(width: environment.width, height: environment.height, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            }
            enc.endEncoding()
        }
        guard let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.label = "ocean mips"
        // The resting cascades' displacement carries over into this frame's map; this frame's
        // usage starts empty.
        if !list.isEmpty && previousValid {
            for c in 0..<OceanSpectrum.cascadeCount where !list.contains(UInt32(c)) {
                blit.copy(from: displacement[1 - current], sourceSlice: c, sourceLevel: 0, to: displacement[current],
                          destinationSlice: c, destinationLevel: 0, sliceCount: 1, levelCount: 1)
            }
        }
        blit.fill(buffer: usageBuffer, range: 0..<4, value: 0)
        if !list.isEmpty { for t in [displacement[current], derivatives, moments] { blit.generateMipmaps(for: t) } }
        if rebuild { blit.generateMipmaps(for: environment) }
        blit.endEncoding()
        previousValid = true
    }

    /// What the sky environment was last built for.
    struct EnvironmentKey {
        var sun: SIMD3<Float>
        var clock: Double
        var camera: SIMD3<Float>

        func matches(_ o: EnvironmentKey) -> Bool {
            let turned: Float = simd_dot(simd_normalize(sun), simd_normalize(o.sun))
            let moved: Float = simd_length(camera - o.camera)
            // cos 0.1° ≈ 1 − 1.5e-6; trade cumulus drift ~7.5 m/s, so 1.3 s ≈ 10 m.
            return turned > 1 - 1.5e-6 && abs(clock - o.clock) < 1.3 && moved < 50
        }
    }

    /// Forgets last frame's waves (a new view: no motion to reproject).
    func resetHistory() {
        previousValid = false
        environmentKey = nil
    }

    private func bind(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms) {
        enc.setVertexBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.setVertexBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setFragmentBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setVertexTexture(displacement[current], index: 4)
    }

    private func drawNodes(_ enc: MTLRenderCommandEncoder) {
        let stride = MemoryLayout<TerrainRenderer.NodeInstance>.stride, quads = OceanRenderer.nodeQuads
        let nodes = nodeBuffers[nodeBufferIndex]
        if nodeCounts.full > 0 {
            enc.setVertexBuffer(nodes, offset: 0, index: 4)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: quads * quads * 6, indexType: .uint16,
                                      indexBuffer: nodeIndices.full, indexBufferOffset: 0, instanceCount: nodeCounts.full)
        }
        if nodeCounts.half > 0 {
            enc.setVertexBuffer(nodes, offset: nodeCounts.full * stride, index: 4)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: (quads / 2) * (quads / 2) * 6, indexType: .uint16,
                                      indexBuffer: nodeIndices.half, indexBufferOffset: 0, instanceCount: nodeCounts.half)
        }
    }

    /// Into the depth prepass (the terrain's heights already bound for the water depth).
    func drawPrepass(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms, motion: MTLTexture?) {
        enc.setRenderPipelineState(prepassPipeline)
        bind(enc, uniforms: &uniforms)
        if writesMotion, let motion {
            enc.setFragmentTexture(motion, index: 10)
            enc.setFragmentBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
            enc.setVertexTexture(displacement[previousValid && !motionless ? 1 - current : current], index: 5)
        }
        drawNodes(enc)
    }

    /// The water's colour (and its floor's), drawn in the shading pass where the prepass left it
    /// nearest; the terrain's heights must be bound (TerrainRenderer.bindTerrain).
    func drawShading(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms,
                     terrainParams: TerrainRenderer.TerrainParams, pages: MTLBuffer, tiles: MTLTexture) {
        enc.setRenderPipelineState(shadePipeline)
        bind(enc, uniforms: &uniforms)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.setFragmentTextures([derivatives, moments, environment, tiles], range: 6..<10)
        enc.setVertexTexture(shore, index: 6)
        enc.setFragmentBuffer(usageBuffers[frameIndex % usageBuffers.count], offset: 0, index: 8)
        var t = terrainParams
        enc.setFragmentBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 3)
        enc.setFragmentBuffer(pages, offset: 0, index: 5)
        drawNodes(enc)
    }

    /// The sky pass shades the sea beyond the far plane from these.
    func bindSky(_ enc: MTLRenderCommandEncoder) {
        enc.setFragmentBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setFragmentTexture(environment, index: 8)
    }
}
#endif

