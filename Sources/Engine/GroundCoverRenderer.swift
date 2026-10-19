import Flora

/// The ground cover's flags (--cover off, --cover-range M, --cover-density X), for the app.
public typealias GroundCoverSettings = GroundCover.Settings

#if canImport(MetalKit)
import Metal
import simd
import Flora
import IslandCore

/// Native grasses and sedges as blades near the walker (docs/flora.md §12, GroundCover):
/// generated once per 8 m tile as it enters the ring around the camera and kept in a slot
/// of a toroidal cache; every frame a cull pass picks the blades in view, thins them with
/// distance and puts them on the ground the terrain draws, into two LODs' indirect draws.
/// Drawn in the depth prepass (writing the surface the sun rays trace from, as a tree
/// pixel) and shaded with equal depth after the trees, like them.
final class GroundCoverRenderer {
    let settings: GroundCover.Settings
    let device: MTLDevice
    let clearPipeline: MTLComputePipelineState
    let generatePipeline: MTLComputePipelineState
    let cullPipeline: MTLComputePipelineState
    let argumentsPipeline: MTLComputePipelineState
    let prepassPipeline: MTLRenderPipelineState
    let shadePipeline: MTLRenderPipelineState
    let shadeDepthState: MTLDepthStencilState

    /// Tiles per side of the cache: enough for the ring and a tile of margin.
    let slotsPerSide: Int
    var slotCount: Int { slotsPerSide * slotsPerSide }
    let slotCapacity = GroundCover.tileCandidates
    /// Blades drawn per LOD at most.
    static let drawCapacity = 1 << 17
    /// Where each slot's tile is (CPU's copy) and the GPU's, one buffer per frame in flight.
    var slotTiles: [SIMD2<Int32>]
    let slotTileBuffers: [MTLBuffer]
    let generateBuffers: [MTLBuffer]
    var frameIndex = 0
    let slotCounts: MTLBuffer
    let seeds: MTLBuffer
    let drawCounts: MTLBuffer
    let blades: MTLBuffer
    let arguments: MTLBuffer
    /// Blades drawn last frame per LOD (stats).
    var lastDrawn: (Int, Int) {
        let c = drawCounts.contents().bindMemory(to: UInt32.self, capacity: 2)
        return (Int(c[0]), Int(c[1]))
    }

    init(device: MTLDevice, library: MTLLibrary, constants: MTLFunctionConstantValues, colorFormat: MTLPixelFormat,
         depthFormat: MTLPixelFormat, rayTraced: Bool, settings: GroundCover.Settings) throws {
        self.device = device
        self.settings = settings
        func kernel(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw RendererError.setup("ground cover \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        clearPipeline = try kernel("coverClear")
        generatePipeline = try kernel("coverGenerate")
        cullPipeline = try kernel("coverCull")
        argumentsPipeline = try kernel("coverArguments")
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "coverVertex")
        desc.depthAttachmentPixelFormat = depthFormat
        desc.inputPrimitiveTopology = .triangle
        if rayTraced {
            desc.fragmentFunction = library.makeFunction(name: "coverSurface")
            desc.colorAttachments[0].pixelFormat = .rgba32Uint
        } else {
            desc.fragmentFunction = library.makeFunction(name: "coverDepth")
            desc.colorAttachments[0].pixelFormat = .invalid
        }
        prepassPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.fragmentFunction = try library.makeFunction(name: "coverFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        shadePipeline = try device.makeRenderPipelineState(descriptor: desc)
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .equal
        depthDesc.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("cover depth state") }
        shadeDepthState = ds

        let tile = Float(GroundCover.tileSize)
        slotsPerSide = Int((2 * (settings.range + tile) / tile).rounded(.up)) + 1
        let slots = slotsPerSide * slotsPerSide
        slotTiles = Array(repeating: SIMD2(Int32.min, Int32.min), count: slots)
        func buffer(_ length: Int, _ mode: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: Swift.max(length, 16), options: mode) else { throw RendererError.setup("cover buffers") }
            return b
        }
        slotTileBuffers = try (0..<3).map { _ in try buffer(slots * 16) }
        generateBuffers = try (0..<3).map { _ in try buffer(slots * 4) }
        slotCounts = try buffer(slots * 4)
        memset(slotCounts.contents(), 0, slotCounts.length)
        seeds = try buffer(slots * slotCapacity * 32, .storageModePrivate)
        drawCounts = try buffer(8)
        blades = try buffer(2 * GroundCoverRenderer.drawCapacity * 48, .storageModePrivate)
        arguments = try buffer(2 * 16)
        memset(arguments.contents(), 0, arguments.length)
    }

    /// The terrain shader's view of the cover (coverRange): full-density distance, end, on.
    var terrainRange: SIMD4<Float> { SIMD4(0.5 * settings.range, settings.range, 1, 0) }

    /// Generates the tiles that entered the ring and culls every kept blade into this
    /// frame's draws. `viewProjection` is the unjittered one; `hzb` last frame's pyramid
    /// (FloraRenderer.hzbBinding), if any.
    func update(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms, eye: SIMD3<Double>,
                viewProjection m: simd_float4x4, renderHeight: Int, groundHeight: Double,
                terrainParams: TerrainRenderer.TerrainParams, detail: TerrainDetail, pages: MTLBuffer, tiles: MTLTexture,
                weights: [MTLTexture], canopyCover: MTLTexture, flora: FloraRenderer?, forward: SIMD3<Float>) {
        frameIndex = (frameIndex + 1) % 3
        // The tiles the ring needs, and which slots must be regenerated for them.
        let tile = GroundCover.tileSize
        let reach = Double(settings.range) + tile
        let cx = Int((eye.x / tile).rounded(.down)), cy = Int((eye.y / tile).rounded(.down))
        let r = Int((reach / tile).rounded(.up))
        var generate: [UInt32] = []
        let n = slotsPerSide
        for ty in (cy - r)...(cy + r) {
            for tx in (cx - r)...(cx + r) {
                let dx = (Double(tx) + 0.5) * tile - eye.x, dy = (Double(ty) + 0.5) * tile - eye.y
                if dx * dx + dy * dy > (reach + tile) * (reach + tile) { continue }
                let slot = ((tx % n + n) % n) * n + ((ty % n + n) % n)
                let want = SIMD2(Int32(tx), Int32(ty))
                if slotTiles[slot] != want {
                    slotTiles[slot] = want
                    generate.append(UInt32(slot))
                }
            }
        }
        let tileBuffer = slotTileBuffers[frameIndex], generateBuffer = generateBuffers[frameIndex]
        let out = tileBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: slotCount)
        for (k, t) in slotTiles.enumerated() { out[k] = SIMD4(t.x, t.y, 0, 0) }
        generate.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(generateBuffer.contents(), base, raw.count) }
        }

        func row(_ i: Int) -> SIMD4<Float> { SIMD4(m.columns.0[i], m.columns.1[i], m.columns.2[i], m.columns.3[i]) }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        var params: [SIMD4<Float>] = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map { p in
            let l = simd_length(SIMD3(p.x, p.y, p.z))
            return l > 0 ? p / l : p
        }
        params.append(SIMD4(0.5 * settings.range, settings.range, settings.density, Float(eye.z - groundHeight)))
        params.append(SIMD4(Float(n), Float(slotCapacity), Float(GroundCoverRenderer.drawCapacity), Float(generate.count)))
        params.append(SIMD4(12, Float(renderHeight), 0, 0))

        guard let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.label = "cover clear"
        blit.fill(buffer: drawCounts, range: 0..<8, value: 0)
        blit.endEncoding()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "ground cover"
        var t = terrainParams
        var d = detail.params
        enc.setBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 0)
        enc.setBuffer(generateBuffer, offset: 0, index: 1)
        enc.setBuffer(slotCounts, offset: 0, index: 2)
        enc.setBytes(&params, length: params.count * 16, index: 3)
        enc.setBuffer(tileBuffer, offset: 0, index: 4)
        enc.setBuffer(seeds, offset: 0, index: 5)
        enc.setBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 6)
        enc.setBuffer(pages, offset: 0, index: 7)
        enc.setTexture(tiles, index: 0)
        if !generate.isEmpty {
            enc.setComputePipelineState(clearPipeline)
            enc.dispatchThreads(MTLSize(width: generate.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            // The tree scatter's settings, to keep stems clear (none without trees).
            var floraParams = flora?.params(eye: eye, viewProjection: matrix_identity_float4x4) ?? [SIMD4<Float>](repeating: .zero, count: 80)
            var haveFlora: UInt32 = flora == nil ? 0 : 1
            enc.setComputePipelineState(generatePipeline)
            enc.setBytes(&floraParams, length: floraParams.count * 16, index: 8)
            enc.setBytes(&haveFlora, length: 4, index: 9)
            enc.setTextures(weights + [canopyCover], range: 1..<5)
            let biomes = flora?.biomes ?? weights
            enc.setTextures(biomes, range: 5..<8)
            let g = GroundCover.tileGrid
            enc.dispatchThreads(MTLSize(width: g, height: g, depth: generate.count),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        var hzbInfo = flora?.hzbInfo(eye: eye, forward: forward) ?? [SIMD4<Float>](repeating: .zero, count: 6)
        enc.setComputePipelineState(cullPipeline)
        enc.setBytes(&d, length: MemoryLayout<TerrainDetail.Params>.stride, index: 10)
        enc.setBuffer(drawCounts, offset: 0, index: 11)
        enc.setBuffer(blades, offset: 0, index: 12)
        enc.setBytes(&hzbInfo, length: hzbInfo.count * 16, index: 13)
        enc.setTexture(detail.texture, index: 8)
        enc.setTexture(flora?.hzb ?? weights[0], index: 9)
        enc.dispatchThreads(MTLSize(width: slotCapacity, height: slotCount, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.setComputePipelineState(argumentsPipeline)
        enc.setBuffer(arguments, offset: 0, index: 14)
        enc.dispatchThreads(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1))
        enc.endEncoding()
    }

    func draw(_ enc: MTLRenderCommandEncoder) {
        enc.setVertexBuffer(blades, offset: 0, index: 2)
        for lod in 0..<2 {
            var segments: UInt32 = lod == 0 ? 5 : 3
            enc.setVertexBytes(&segments, length: 4, index: 3)
            enc.drawPrimitives(type: .triangleStrip, indirectBuffer: arguments, indirectBufferOffset: lod * 16)
        }
    }

    /// In the depth prepass, after the trees (same depth state and uniforms).
    func drawPrepass(_ enc: MTLRenderCommandEncoder) {
        enc.setRenderPipelineState(prepassPipeline)
        draw(enc)
    }

    /// In the shading pass, after the trees: their fragment textures stay bound.
    func drawShading(_ enc: MTLRenderCommandEncoder, skyVisibility: MTLTexture, horizons: MTLTexture) {
        enc.setRenderPipelineState(shadePipeline)
        enc.setDepthStencilState(shadeDepthState)
        enc.setFragmentTexture(skyVisibility, index: 8)
        enc.setFragmentTexture(horizons, index: 9)
        draw(enc)
    }
}
#endif
