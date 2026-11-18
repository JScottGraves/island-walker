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
        var waves: SIMD4<Float>
        var breaking: SIMD4<Float>
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
    /// Whitecap foam (oceanFoam) of the cascades with whitecaps, in their patches' space (fixed
    /// in the world, so a breaking crest moves on and leaves its foam to fade), this frame's
    /// and last frame's in turn, with mips (their mean is the foam's cover far out).
    let foam: [MTLTexture]
    var foamCurrent = 0
    var foamClock = FoamClock()
    let foamPipeline: MTLComputePipelineState
    /// The sky the sea reflects, as an upper-hemisphere octahedral map with mips.
    let environment: MTLTexture
    /// Over the domain, from the overview, for the surf: signed distance to the shoreline (m,
    /// positive at sea), the capillary roughness's factor where the wind is in the lee
    /// (SeaShelter), one channel unused, and the water's depth smoothed.
    let shore: MTLTexture
    /// Over the domain, for the surf, from the next finer level: the trade swell's and the north
    /// swell's arrival (in their periods), local significant heights (m, two halfs in one float)
    /// and the water's depth, refracted and shoaled (WaveRefraction.surfTexels).
    let waves: MTLTexture
    /// Over the domain, from the overview: each cascade's amplitude factor where the sea is
    /// sheltered (SeaShelter: fetch, the wind's lee, reefs, the swells' shadows), 1 on open sea.
    let shelter: MTLTexture
    /// The lips of plunging breakers near the camera, their spray and impact foam (SurfLip).
    let lip: SurfLip
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
         bathymetry: Grid? = nil,
         windTowards: Double,
         queue: MTLCommandQueue) throws {
        self.seaLevel = seaLevel
        occlusion = TerrainOcclusion(height: overview)
        let tuning = OceanRenderer.tuning()
        let spectrum = OceanSpectrum(fetch: tuning["fetch"] ?? 40_000, windTowards: windTowards,
                                     tradeSwellHeight: tuning["trade"] ?? 2, tradeSwellPeriod: tuning["tradeT"] ?? 8,
                                     swellHeight: tuning["swell"] ?? 1.5, choppiness: tuning["chop"] ?? 1.2)
        self.spectrum = spectrum
        // Smoothed over ~150 m, so the surf's fronts (which follow it) curve without corners.
        let distance = ShoreDistance.smoothed(ShoreDistance.signed(height: overview, seaLevel: seaLevel), passes: 3)
        // Water depth smoothed over ~150 m, for where waves start to break (a line along the reef
        // or bar, not the floor's contours).
        var depth = overview
        for k in 0..<depth.values.count { depth.values[k] = Swift.max(seaLevel - depth.values[k], 0) }
        depth = ShoreDistance.smoothed(depth, passes: 2)
        var texels = [Float](repeating: 0, count: distance.values.count * 4)
        for k in 0..<distance.values.count {
            texels[4 * k] = distance.values[k]
            texels[4 * k + 3] = depth.values[k]
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
        // Refraction and shoaling of the trade swell (with the wind sea riding on it) and the
        // north swell over the bathymetry (WaveRefraction): their crests' arrival and heights.
        let bakeStart = Date()
        let tradeHeight = (pow(spectrum.windSea.measuredSignificantHeight, 2)
                           + pow(spectrum.tradeSwell.measuredSignificantHeight, 2)).squareRoot()
        let tradePeriod = 2 * Double.pi / spectrum.tradeSwell.peakOmega, swellPeriod = 2 * Double.pi / spectrum.swell.peakOmega
        let floor = bathymetry ?? overview
        let waveTexels = WaveRefraction.surfTexels(
            height: floor, seaLevel: seaLevel,
            systems: (.init(period: tradePeriod, towards: spectrum.tradeSwell.direction, height: tradeHeight),
                      .init(period: swellPeriod, towards: spectrum.swell.direction,
                            height: spectrum.swell.measuredSignificantHeight)))
        print(String(format: "surf refraction: %.2f s for %d² cells", Date().timeIntervalSince(bakeStart), floor.width))
        if let dir = ProcessInfo.processInfo.environment["ISLAND_WAVE_DUMP"] {
            // Raw float32 fields for the debug images: the heights, then the wave texels.
            let url = URL(fileURLWithPath: dir)
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try? floor.values.withUnsafeBytes { Data($0) }.write(to: url.appendingPathComponent("height.f32"))
            try? waveTexels.withUnsafeBytes { Data($0) }.write(to: url.appendingPathComponent("waves.f32"))
            try? distance.values.withUnsafeBytes { Data($0) }.write(to: url.appendingPathComponent("shore.f32"))
        }
        let wavesDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: floor.width,
                                                                 height: floor.height, mipmapped: false)
        wavesDesc.usage = [.shaderRead]
        wavesDesc.storageMode = .shared
        guard let wavesTexture = device.makeTexture(descriptor: wavesDesc) else { throw RendererError.setup("surf waves") }
        waveTexels.withUnsafeBytes { raw in
            wavesTexture.replace(region: MTLRegionMake2D(0, 0, floor.width, floor.height), mipmapLevel: 0,
                                 withBytes: raw.baseAddress!, bytesPerRow: floor.width * 16)
        }
        waves = wavesTexture
        // The lips (SurfLip) pick their wave line from the same texels.
        let lipField = SurfLip.Field(texels: waveTexels, width: floor.width, cell: Float(floor.cellSize))
        // How much of the open sea's waves each patch of water gets (SeaShelter): per cascade
        // here, the capillary roughness's factor in the shore texture's y.
        let shelterStart = Date()
        var shelterTexels = [SIMD4<Float16>](repeating: SIMD4(repeating: 1), count: distance.values.count)
        if (tuning["shelter"] ?? 1) > 0 {
            var settings = SeaShelter.Settings()
            if let v = tuning["lee"] { settings.lee = v }
            if let v = tuning["leeL"] { settings.leeLength = v }
            if let v = tuning["reef"] { settings.reefDepth = v }
            var tradeH = Grid(width: floor.width, height: floor.height, cellSize: floor.cellSize)
            var northH = tradeH, swellDepth = tradeH
            for k in 0..<waveTexels.count {
                let bits = waveTexels[k].z.bitPattern
                tradeH.values[k] = Float(Float16(bitPattern: UInt16(bits & 0xFFFF)))
                northH.values[k] = Float(Float16(bitPattern: UInt16(bits >> 16)))
                swellDepth.values[k] = waveTexels[k].w
            }
            let shelter = SeaShelter.factors(height: overview, seaLevel: seaLevel, spectrum: spectrum,
                                             swells: (tradeH, northH, swellDepth), tradeHeight: tradeHeight,
                                             settings: settings)
            for k in 0..<shelterTexels.count {
                shelterTexels[k] = SIMD4<Float16>(shelter.cascades[k])
                texels[4 * k + 1] = shelter.calm.values[k]
            }
            texels.withUnsafeBytes { raw in
                shoreTexture.replace(region: MTLRegionMake2D(0, 0, distance.width, distance.height), mipmapLevel: 0,
                                     withBytes: raw.baseAddress!, bytesPerRow: distance.width * 16)
            }
            if let dir = ProcessInfo.processInfo.environment["ISLAND_WAVE_DUMP"] {
                let url = URL(fileURLWithPath: dir)
                try? shelter.cascades.withUnsafeBytes { Data($0) }.write(to: url.appendingPathComponent("shelter.f32"))
                try? shelter.calm.values.withUnsafeBytes { Data($0) }.write(to: url.appendingPathComponent("calm.f32"))
            }
        } else {
            for k in 0..<distance.values.count { texels[4 * k + 1] = 1 }
            texels.withUnsafeBytes { raw in
                shoreTexture.replace(region: MTLRegionMake2D(0, 0, distance.width, distance.height), mipmapLevel: 0,
                                     withBytes: raw.baseAddress!, bytesPerRow: distance.width * 16)
            }
        }
        let shelterDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: distance.width,
                                                                   height: distance.height, mipmapped: false)
        shelterDesc.usage = [.shaderRead]
        shelterDesc.storageMode = .shared
        guard let shelterTexture = device.makeTexture(descriptor: shelterDesc) else { throw RendererError.setup("sea shelter") }
        shelterTexels.withUnsafeBytes { raw in
            shelterTexture.replace(region: MTLRegionMake2D(0, 0, distance.width, distance.height), mipmapLevel: 0,
                                   withBytes: raw.baseAddress!, bytesPerRow: distance.width * 8)
        }
        shelter = shelterTexture
        print(String(format: "sea shelter: %.2f s", Date().timeIntervalSince(shelterStart)))
        tree = OceanQuadtree(quads: OceanRenderer.nodeQuads, cellMetres: 0.5, centre: (domain / 2, domain / 2),
                             reach: domain / 2 + 160_000, rangeFactor: tuning["range"] ?? 2)
        func kernel(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw RendererError.setup("ocean \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        rowsPipeline = try kernel("oceanRows")
        columnsPipeline = try kernel("oceanColumns")
        environmentPipeline = try kernel("oceanSkyEnvironment")
        foamPipeline = try kernel("oceanFoam")
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
        lip = try SurfLip(device: device, library: library, colorFormat: colorFormat, depthFormat: depthFormat, field: lipField,
                          enabled: (tuning["lip"] ?? 1) > 0)
        lip.sprayEnabled = (tuning["spray"] ?? 1) > 0

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
        func foamMaps() throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = .r32Float
            d.width = n
            d.height = n
            d.arrayLength = OceanRenderer.foamCascades
            d.mipmapLevelCount = Int(log2(Double(n))) + 1
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else { throw RendererError.setup("ocean foam") }
            return t
        }
        foam = [try foamMaps(), try foamMaps()]
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
            time: SIMD4(0, 0, Float(tuning["surfwave"] ?? 1.5), 0),
            // Clear oceanic water (Jerlov I): pure-water absorption (Pope & Fry 1997) with a
            // little more in the blue; Rayleigh-like molecular scattering; few particles.
            absorption: SIMD4(0.45, 0.064, 0.017, 0),
            scattering: SIMD4(0.0008, 0.0019, 0.0045, 0.02),
            sss: SIMD4(0.05, 0.35, 0.3, 0.06),
            // x, y and z unused by the surf since it is refracted (`waves` instead).
            surf: SIMD4(Float((pow(spectrum.windSea.measuredSignificantHeight, 2)
                               + pow(spectrum.tradeSwell.measuredSignificantHeight, 2)).squareRoot()),
                        Float(spectrum.swell.measuredSignificantHeight),
                        Float(2 * Double.pi / spectrum.swell.peakOmega), Float(tuning["shallow"] ?? 0.6)),
            // Waves come from the opposite of where they travel (unused since the surf is refracted).
            surfFrom: SIMD4(Float(-cos(spectrum.swell.direction)), Float(-sin(spectrum.swell.direction)),
                            Float(-cos(windTowards)), Float(-sin(windTowards))),
            waves: SIMD4(Float(tradePeriod), Float(swellPeriod), Float(WaveRefraction.breakerIndex), 7),
            // The bathymetry (10 m grid, the surf's bake 40 m) smooths reef edges and bars into
            // gentler slopes than the real bed's: their Iribarren numbers ×1.5.
            breaking: SIMD4(Float(tuning["plunge"] ?? 1.5), Float(tuning["lip"] ?? 1), SurfLip.foamCell, Float(SurfLip.foamSide)))
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
        let now = Float(time.truncatingRemainder(dividingBy: OceanSpectrum.repeatPeriod))
        // Last frame's time, for the surf's waves' motion (none after a reset).
        params.time.y = previousValid ? params.time.x : now
        params.time.x = now

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
        var foamUpdated = false
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
                // The whitecaps' foam: the updated cascades' folds added, faded by the time since.
                var mask: UInt32 = 0
                for c in list where c < UInt32(OceanRenderer.foamCascades) { mask |= 1 << c }
                if mask != 0 {
                    var step = foamClock.step(time: time, updated: mask).uniform
                    foamCurrent = 1 - foamCurrent
                    enc.setComputePipelineState(foamPipeline)
                    enc.setBytes(&step, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
                    enc.setTextures([derivatives, displacement[current], foam[1 - foamCurrent], foam[foamCurrent]], range: 0..<4)
                    enc.dispatchThreads(MTLSize(width: n, height: n, depth: OceanRenderer.foamCascades),
                                        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                    foamUpdated = true
                }
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
        if foamUpdated { blit.generateMipmaps(for: foam[foamCurrent]) }
        if rebuild { blit.generateMipmaps(for: environment) }
        blit.endEncoding()
        previousValid = true
    }

    /// Cascades with whitecap foam (the shortest one's caps are lost in the others').
    static let foamCascades = 3

    /// One step of the whitecaps' foam: for each cascade, the seconds it fades by this frame
    /// (since its own last update; −1 for a cascade not updated, whose foam is kept), and
    /// whether the foam starts over from this moment's folds.
    struct FoamStep {
        var cascadeDts: SIMD4<Float>
        var reset: Bool
        /// Seconds since the last frame, clamped (0 after a reset).
        var dt: Double

        func cascadeDt(_ c: Int) -> Double { Double(cascadeDts[c]) }
        /// oceanFoam's `step`.
        var uniform: SIMD4<Float> { SIMD4(cascadeDts.x, cascadeDts.y, cascadeDts.z, reset ? 1 : 0) }
    }

    /// The foam's time step from the sea's clock (seconds, unwrapped): each cascade fades by the
    /// time since its last update, clamped over a hitch; a jump of more than a couple of seconds
    /// either way (a new hour, a snapshot of another moment) starts the foam over.
    struct FoamClock {
        var last: Double?
        var cascadeLast = [Double](repeating: 0, count: OceanRenderer.foamCascades)

        mutating func step(time: Double, updated: UInt32) -> FoamStep {
            var dts = SIMD4<Float>(repeating: -1)
            let reset: Bool
            var frame = 0.0
            if let last, time >= last, time - last < 2 {
                reset = false
                frame = Swift.min(time - last, 0.1)
            } else {
                reset = true
                for c in 0..<cascadeLast.count { cascadeLast[c] = time }
            }
            for c in 0..<cascadeLast.count where updated & (UInt32(1) << UInt32(c)) != 0 {
                let dt: Double = Swift.min(Swift.max(time - cascadeLast[c], 0), 0.1)
                dts[c] = Float(dt)
                cascadeLast[c] = time
            }
            last = time
            return FoamStep(cascadeDts: dts, reset: reset, dt: frame)
        }
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

    /// Sea-state overrides for tuning, from the environment: ISLAND_OCEAN="trade=2,tradeT=8,
    /// swell=1.5,fetch=40000,chop=1.2,shallow=0.6,surfwave=1.5,shelter=1,lee=0.6,leeL=12,reef=1,
    /// plunge=1,lip=1,range=2"
    /// (trade swell height and period, north swell height, local wind fetch (m), choppiness, the
    /// depth-limited Hs/h, the surf waves' height scale; shelter=0 turns SeaShelter off, lee and
    /// leeL its wind lee's strength and length (× the land's height), reef the depth (m) that
    /// breaks the wind sea; plunge scales the breakers' Iribarren number, lip=0 leaves out their
    /// lips (SurfLip), range is the sea mesh's ring radius in nodes). Unset keys keep the defaults.
    static func tuning() -> [String: Double] {
        var out: [String: Double] = [:]
        for pair in (ProcessInfo.processInfo.environment["ISLAND_OCEAN"] ?? "").split(separator: ",") {
            let kv = pair.split(separator: "=")
            if kv.count == 2, let v = Double(kv[1]) { out[String(kv[0])] = v }
        }
        if !out.isEmpty { print("ocean tuning: \(out)") }
        return out
    }

    /// Forgets last frame's waves (a new view: no motion to reproject).
    func resetHistory() {
        previousValid = false
        environmentKey = nil
        lip.reset()
    }

    /// The breakers' lips for this frame (after `update`, which sets the sea's clock).
    func updateLip(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms,
                   terrainParams: TerrainRenderer.TerrainParams, eye: SIMD3<Double>, time: Double) {
        lip.update(cmd, uniforms: &uniforms, ocean: &params, terrainParams: terrainParams, shore: shore, waves: waves,
                   eye: eye, time: time)
    }

    /// The lips' sheets and spray, blended over the water (after `drawShading`, in the same pass).
    func drawLip(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms) {
        lip.draw(enc, uniforms: &uniforms)
    }

    private func bind(_ enc: MTLRenderCommandEncoder, uniforms: inout TerrainRenderer.Uniforms) {
        enc.setVertexBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 1)
        enc.setVertexBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setFragmentBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setVertexTexture(displacement[current], index: 4)
        // The surf's waves lift the water near the shore, in every pass that draws it.
        enc.setVertexTexture(shore, index: 6)
        enc.setVertexTexture(waves, index: 7)
        enc.setVertexTexture(shelter, index: 8)
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
        enc.setFragmentTexture(foam[foamCurrent], index: 12)
        enc.setFragmentBuffer(usageBuffers[frameIndex % usageBuffers.count], offset: 0, index: 8)
        enc.setFragmentBuffer(lip.foam, offset: 0, index: 9)
        var t = terrainParams
        enc.setFragmentBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 3)
        enc.setFragmentBuffer(pages, offset: 0, index: 5)
        drawNodes(enc)
    }

    /// The sky pass shades the sea beyond the far plane from these.
    func bindSky(_ enc: MTLRenderCommandEncoder) {
        enc.setFragmentBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
        enc.setFragmentTexture(environment, index: 8)
        // The terrain's swash (swashSand) reads the shore too.
        enc.setFragmentTexture(shore, index: 9)
        enc.setFragmentTexture(waves, index: 10)
    }
}
#endif

