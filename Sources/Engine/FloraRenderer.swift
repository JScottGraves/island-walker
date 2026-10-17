#if canImport(MetalKit)
import Metal
import simd
import Flora
import IslandCore

/// What the tree scatter reads from the bundle: the plant-community weights (in
/// `Communities.table` order, all the same size, summing to 1 on land) and a 0…1 site
/// factor (wetter and more sheltered is taller).
public struct FloraMaps: Sendable {
    public var weights: [Grid]
    public var site: Grid?

    public init(weights: [Grid], site: Grid?) {
        self.weights = weights
        self.site = site
    }

    /// Site factor from the bake's moisture availability (mm/yr) and crest exposure (0…1).
    public static func site(mai: Grid?, exposure: Grid?) -> Grid? {
        guard let mai else { return nil }
        var g = mai
        for i in 0..<g.count {
            let wet = Swift.min(1, Swift.max(0, (Double(mai.values[i]) + 500) / 3500))
            var e = 0.0
            if let exposure, exposure.width == mai.width, exposure.height == mai.height { e = Double(exposure.values[i]) }
            g.values[i] = Float(wet * (1 - 0.6 * e))
        }
        return g
    }
}

/// The native trees (docs/flora.md §4–5): species meshes from the Flora library in one
/// vertex and index buffer, a GPU scatter that fills one instance range and one indirect
/// draw per (plant, variant, LOD) bucket each frame, and the tree passes: alpha-tested in
/// the depth prepass (writing the surface the sun rays trace from) and in the shading pass.
final class FloraRenderer {
    /// One tree, laid out as the Metal `FloraInstance` (48 bytes).
    struct Instance {
        var x: Float, y: Float, z: Float
        var scale: Float
        var cosYaw: Float, sinYaw: Float
        var leanX: Float, leanY: Float
        var fade: Float
        var tint: Float
        var phase: Float
        var height: Float
    }

    struct Mesh {
        var indexCount: UInt32
        var indexStart: UInt32
        var baseVertex: Int32
        var pad: UInt32 = 0
    }

    /// A tree placed by hand (floragen's specimen sheets) instead of by the scatter.
    struct Specimen {
        /// nil: the scale figure.
        var plant: Plant?
        var variant: Int
        var lod: Int
        var position: SIMD3<Float>
        var yaw: Float
        var height: Float
    }

    /// Buckets per variant: the mesh LODs, then the impostor.
    static let lodBuckets = Flora.lods + 1
    /// One bucket per (plant, variant, LOD), plus floragen's scale figure.
    static let plantBuckets = Plant.allCases.count * Flora.variants * lodBuckets
    static let buckets = plantBuckets + 1
    /// Instances per bucket.
    static let capacity = 4096
    /// Side of an impostor's baked view, texels.
    static let impostorSize = 256

    /// LOD0 out to `lod0Distance`, LOD1 to `lod1Distance` (both pushed out for tall plants),
    /// then impostors to the layer's range, cross-fading over `fadeWidth`.
    var lod0Distance: Float = 30
    var lod1Distance: Float = 80
    var fadeWidth: Float = 8
    var canopyRange: Float = 600
    var shrubRange: Float = 120
    /// Ground and tree pixels this close to the camera take shadows from the trees' proxies
    /// (fading out over the last 50 m); beyond it the canopy stand-in shades the ground.
    var shadowRange: Float = 200
    /// Trees this much past the shadow range still cast into it (the structure is rebuilt
    /// only every `shadowRebuild` metres of camera travel, and shadows fall at an angle).
    var shadowMargin: Float = 60
    var shadowRebuild: Double = 10

    let device: MTLDevice
    let scatterPipeline: MTLComputePipelineState
    let argumentsPipeline: MTLComputePipelineState
    let prepassPipeline: MTLRenderPipelineState
    let shadePipeline: MTLRenderPipelineState
    /// Shading pass: depth equal to the prepass's, no writes.
    let shadeDepthState: MTLDepthStencilState
    let impostorPrepassPipeline: MTLRenderPipelineState
    let impostorShadePipeline: MTLRenderPipelineState
    /// Each variant's side view: albedo and coverage, and normal, AO and leaf flag.
    let impostorAlbedo: MTLTexture
    let impostorNormal: MTLTexture
    /// Per variant: half-width and z range of its LOD0 mesh (FloraImpostor).
    let impostorExtents: MTLBuffer
    let vertices: MTLBuffer
    let indices: MTLBuffer
    let meshes: MTLBuffer
    let counts: MTLBuffer
    let instances: MTLBuffer
    let arguments: MTLBuffer
    let cards: MTLTexture
    let bark: MTLTexture
    /// Community weights packed four to a texture, the site factor in the last one's alpha.
    let biomes: [MTLTexture]
    let seed: UInt32
    let seaLevel: Float
    let domainSize: Float
    /// Fixed instances (specimen sheets): the scatter is skipped.
    var specimens: [Specimen]?
    /// Triangles per bucket, for the stats.
    let triangles: [Int]

    /// Tree shadows (with ray tracing and scattered trees): two proxy structures per (plant,
    /// variant), trunk then crown, and the instance structure over every tree in the shadow
    /// range (its trunk and crown as instances 2k and 2k + 1), rebuilt on the GPU as the
    /// camera moves.
    struct Shadows {
        let proxies: [MTLAccelerationStructure]
        let proxyIDs: MTLBuffer
        let scatter: MTLComputePipelineState
        let clamp: MTLComputePipelineState
        let descriptors: MTLBuffer
        let scattered: MTLBuffer
        let count: MTLBuffer
        let descriptor: MTLIndirectInstanceAccelerationStructureDescriptor
        let accel: MTLAccelerationStructure
        let scratch: MTLBuffer
    }
    let shadows: Shadows?
    /// Camera position the shadow structure was last built around, and whether every tile
    /// it stands on had loaded by then (otherwise it is rebuilt next frame).
    var shadowCentre: SIMD3<Double>?
    var shadowSettled = false
    /// Instances in the shadow structure: two per tree.
    static let shadowCapacity = 32768

    init(device: MTLDevice, library: MTLLibrary, constants: MTLFunctionConstantValues, colorFormat: MTLPixelFormat,
         depthFormat: MTLPixelFormat, rayTraced: Bool, treeShadows: Bool, maps: FloraMaps?, seed: UInt32,
         seaLevel: Float, domainSize: Float) throws {
        self.device = device
        self.seed = seed
        self.seaLevel = seaLevel
        self.domainSize = domainSize
        guard let scatter = library.makeFunction(name: "floraScatter"),
              let args = library.makeFunction(name: "floraArguments") else { throw RendererError.setup("flora kernels") }
        scatterPipeline = try device.makeComputePipelineState(function: scatter)
        argumentsPipeline = try device.makeComputePipelineState(function: args)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "floraVertex")
        desc.depthAttachmentPixelFormat = depthFormat
        if rayTraced {
            desc.fragmentFunction = library.makeFunction(name: "floraSurface")
            desc.colorAttachments[0].pixelFormat = .rgba32Uint
        } else {
            desc.fragmentFunction = library.makeFunction(name: "floraDepth")
            desc.colorAttachments[0].pixelFormat = .invalid
        }
        prepassPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.fragmentFunction = try library.makeFunction(name: "floraFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        shadePipeline = try device.makeRenderPipelineState(descriptor: desc)
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .equal
        depthDesc.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("flora depth state") }
        shadeDepthState = ds
        desc.vertexFunction = library.makeFunction(name: "floraImpostorVertex")
        desc.fragmentFunction = library.makeFunction(name: rayTraced ? "floraImpostorSurface" : "floraImpostorDepth")
        desc.colorAttachments[0].pixelFormat = rayTraced ? .rgba32Uint : .invalid
        impostorPrepassPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.fragmentFunction = try library.makeFunction(name: "floraImpostorFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        impostorShadePipeline = try device.makeRenderPipelineState(descriptor: desc)

        // Every plant's meshes in one buffer pair, bucket order (plant, variant, LOD), each
        // variant's impostor quad after its meshes.
        var allVertices: [FloraVertex] = []
        var allIndices: [UInt32] = []
        var table: [Mesh] = []
        var tris: [Int] = []
        var extents: [SIMD4<Float>] = []
        var lod0: [Mesh] = []
        let quad: [FloraVertex] = [(0, 0), (1, 0), (1, 1), (0, 1)].map { c in
            FloraVertex(x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, u: c.0, v: c.1, ao: 1, sway: 0, flutter: 0, material: 0)
        }
        for variants in Flora.library() {
            for lods in variants {
                for (lod, mesh) in lods.enumerated() {
                    let m = Mesh(indexCount: UInt32(mesh.indices.count), indexStart: UInt32(allIndices.count),
                                 baseVertex: Int32(allVertices.count))
                    table.append(m)
                    if lod == 0 {
                        lod0.append(m)
                        var r: Float = 0.1, top: Float = 0.1
                        for v in mesh.vertices {
                            r = Swift.max(r, abs(v.x), abs(v.y))
                            top = Swift.max(top, v.z)
                        }
                        extents.append(SIMD4(r, 0, top, 0))
                    }
                    tris.append(mesh.triangleCount)
                    allVertices += mesh.vertices
                    allIndices += mesh.indices
                }
                table.append(Mesh(indexCount: 6, indexStart: UInt32(allIndices.count), baseVertex: Int32(allVertices.count)))
                tris.append(2)
                allVertices += quad
                allIndices += [0, 1, 2, 0, 2, 3]
            }
        }
        let figure = Flora.figure()
        table.append(Mesh(indexCount: UInt32(figure.indices.count), indexStart: UInt32(allIndices.count),
                          baseVertex: Int32(allVertices.count)))
        tris.append(figure.triangleCount)
        allVertices += figure.vertices
        allIndices += figure.indices
        triangles = tris
        guard let vb = device.makeBuffer(bytes: allVertices, length: allVertices.count * MemoryLayout<FloraVertex>.stride, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: allIndices, length: allIndices.count * 4, options: .storageModeShared),
              let mb = device.makeBuffer(bytes: table, length: table.count * MemoryLayout<Mesh>.stride, options: .storageModeShared),
              let cb = device.makeBuffer(length: FloraRenderer.buckets * 4, options: .storageModeShared),
              let instanceBuffer = device.makeBuffer(length: FloraRenderer.buckets * FloraRenderer.capacity * MemoryLayout<Instance>.stride,
                                                     options: .storageModeShared),
              let ab = device.makeBuffer(length: FloraRenderer.buckets * 20, options: .storageModeShared)
        else { throw RendererError.setup("flora buffers") }
        vertices = vb; indices = ib; meshes = mb; counts = cb; instances = instanceBuffer; arguments = ab
        memset(ab.contents(), 0, ab.length)
        guard let eb = device.makeBuffer(bytes: extents, length: extents.count * 16, options: .storageModeShared)
        else { throw RendererError.setup("impostor extents") }
        impostorExtents = eb

        let textures = try FloraTextures.load()
        cards = try FloraRenderer.textureArray(device: device, images: textures.cards)
        bark = try FloraRenderer.textureArray(device: device, images: textures.bark)
        biomes = try FloraRenderer.biomeTextures(device: device, maps: maps)
        (impostorAlbedo, impostorNormal) = try FloraRenderer.bakeImpostors(
            device: device, library: library, vertices: vb, indices: ib, meshes: lod0, extents: extents, cards: cards, bark: bark)
        shadows = treeShadows ? try FloraRenderer.makeShadows(device: device, library: library) : nil
    }

    static func makeShadows(device: MTLDevice, library: MTLLibrary) throws -> Shadows {
        guard MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride == 72 else {
            throw RendererError.setup("instance descriptor layout")
        }
        guard let scatter = library.makeFunction(name: "floraShadows"),
              let clamp = library.makeFunction(name: "floraShadowCount") else { throw RendererError.setup("tree shadow kernels") }
        // Every variant's proxy: positions and indices in one buffer pair, one structure each.
        var positions: [Float] = [], indices: [UInt32] = []
        var parts: [[(vertex: Int, index: Int, triangles: Int)]] = []
        for plant in Plant.allCases {
            for v in 0..<Flora.variants {
                let proxy = Flora.shadowProxy(plant, variant: v)
                parts.append([proxy.trunk, proxy.crown].map { part in
                    let entry = (vertex: positions.count * 4, index: indices.count * 4, triangles: part.indices.count / 3)
                    for p in part.positions { positions += [p.x, p.y, p.z] }
                    indices += part.indices
                    return entry
                })
            }
        }
        guard let queue = device.makeCommandQueue(),
              let vb = device.makeBuffer(bytes: positions, length: positions.count * 4, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(), let enc = cmd.makeAccelerationStructureCommandEncoder()
        else { throw RendererError.setup("tree shadow proxies") }
        var proxies: [MTLAccelerationStructure] = []
        var scratches: [MTLBuffer] = []
        for part in parts.joined() {
            let desc = MTLPrimitiveAccelerationStructureDescriptor()
            desc.geometryDescriptors = [part].map { part in
                let g = MTLAccelerationStructureTriangleGeometryDescriptor()
                g.vertexBuffer = vb
                g.vertexBufferOffset = part.vertex
                g.vertexStride = 12
                g.vertexFormat = .float3
                g.indexBuffer = ib
                g.indexBufferOffset = part.index
                g.indexType = .uint32
                g.triangleCount = part.triangles
                g.opaque = true
                return g
            }
            let sizes = device.accelerationStructureSizes(descriptor: desc)
            guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)
            else { throw RendererError.setup("tree shadow proxy") }
            enc.build(accelerationStructure: accel, descriptor: desc, scratchBuffer: scratch, scratchBufferOffset: 0)
            proxies.append(accel)
            scratches.append(scratch)
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let ids = proxies.map { $0.gpuResourceID._impl }

        let capacity = FloraRenderer.shadowCapacity
        let descriptor = MTLIndirectInstanceAccelerationStructureDescriptor()
        guard let idBuffer = device.makeBuffer(bytes: ids, length: ids.count * 8, options: .storageModeShared),
              let descriptors = device.makeBuffer(length: capacity * 72, options: .storageModePrivate),
              let scattered = device.makeBuffer(length: 4, options: .storageModePrivate),
              let count = device.makeBuffer(length: 4, options: .storageModeShared)
        else { throw RendererError.setup("tree shadow buffers") }
        memset(count.contents(), 0, 4)
        descriptor.instanceDescriptorType = .indirect
        descriptor.instanceDescriptorBuffer = descriptors
        descriptor.instanceDescriptorStride = 72
        descriptor.maxInstanceCount = capacity
        descriptor.instanceCountBuffer = count
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)
        else { throw RendererError.setup("tree shadow structure") }
        return Shadows(proxies: proxies, proxyIDs: idBuffer,
                       scatter: try device.makeComputePipelineState(function: scatter),
                       clamp: try device.makeComputePipelineState(function: clamp),
                       descriptors: descriptors, scattered: scattered, count: count, descriptor: descriptor,
                       accel: accel, scratch: scratch)
    }

    /// Draws every variant's LOD0 mesh from the side into the impostor arrays, then gives
    /// them mip chains (coverage-preserving for the albedo's alpha).
    static func bakeImpostors(device: MTLDevice, library: MTLLibrary, vertices: MTLBuffer, indices: MTLBuffer,
                              meshes: [Mesh], extents: [SIMD4<Float>], cards: MTLTexture,
                              bark: MTLTexture) throws -> (MTLTexture, MTLTexture) {
        let n = impostorSize
        func array(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = format
            d.width = n
            d.height = n
            d.arrayLength = meshes.count
            d.mipmapLevelCount = Int(log2(Double(n))) + 1
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .shared
            guard let t = device.makeTexture(descriptor: d) else { throw RendererError.setup("impostor textures") }
            return t
        }
        let albedo = try array(.rgba8Unorm_srgb), normal = try array(.rgba8Unorm)
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: n, height: n, mipmapped: false)
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .private
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "floraBakeVertex")
        desc.fragmentFunction = library.makeFunction(name: "floraBakeFragment")
        desc.colorAttachments[0].pixelFormat = .rgba8Unorm_srgb
        desc.colorAttachments[1].pixelFormat = .rgba8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float
        let depthStateDesc = MTLDepthStencilDescriptor()
        depthStateDesc.depthCompareFunction = .less
        depthStateDesc.isDepthWriteEnabled = true
        guard let depth = device.makeTexture(descriptor: depthDesc),
              let depthState = device.makeDepthStencilState(descriptor: depthStateDesc),
              let queue = device.makeCommandQueue(), let cmd = queue.makeCommandBuffer()
        else { throw RendererError.setup("impostor bake") }
        let pipeline = try device.makeRenderPipelineState(descriptor: desc)
        for (layer, mesh) in meshes.enumerated() {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = albedo
            pass.colorAttachments[0].slice = layer
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[1].texture = normal
            pass.colorAttachments[1].slice = layer
            pass.colorAttachments[1].loadAction = .clear
            pass.colorAttachments[1].clearColor = MTLClearColor(red: 0.5, green: 0.5, blue: 1, alpha: 0)
            pass.colorAttachments[1].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.storeAction = .dontCare
            guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { throw RendererError.setup("impostor bake pass") }
            enc.setRenderPipelineState(pipeline)
            enc.setDepthStencilState(depthState)
            enc.setCullMode(.none)
            var e = extents[layer]
            enc.setVertexBuffer(vertices, offset: Int(mesh.baseVertex) * MemoryLayout<FloraVertex>.stride, index: 2)
            enc.setVertexBytes(&e, length: 16, index: 4)
            enc.setFragmentTexture(cards, index: 6)
            enc.setFragmentTexture(bark, index: 7)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: Int(mesh.indexCount), indexType: .uint32,
                                      indexBuffer: indices, indexBufferOffset: Int(mesh.indexStart) * 4)
            enc.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        for layer in 0..<meshes.count {
            for (texture, coverage) in [(albedo, true), (normal, false)] {
                var base = [UInt8](repeating: 0, count: n * n * 4)
                texture.getBytes(&base, bytesPerRow: n * 4, bytesPerImage: n * n * 4,
                                 from: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0, slice: layer)
                let mips = FloraTextures.mipChain(base, size: n, preserveCoverage: coverage, linearData: !coverage)
                for (level, texels) in mips.enumerated() where level > 0 {
                    let m = Swift.max(1, n >> level)
                    texture.replace(region: MTLRegionMake2D(0, 0, m, m), mipmapLevel: level, slice: layer, withBytes: texels,
                                    bytesPerRow: m * 4, bytesPerImage: m * m * 4)
                }
            }
        }
        return (albedo, normal)
    }

    static func textureArray(device: MTLDevice, images: [FloraTextures.Image]) throws -> MTLTexture {
        guard let size = images.first?.size else { throw RendererError.setup("flora textures") }
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba8Unorm_srgb
        desc.width = size
        desc.height = size
        desc.arrayLength = images.count
        desc.mipmapLevelCount = images[0].mips.count
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let t = device.makeTexture(descriptor: desc) else { throw RendererError.setup("flora texture array") }
        for (slice, image) in images.enumerated() {
            guard image.size == size else { throw RendererError.setup("flora textures differ in size") }
            for (level, texels) in image.mips.enumerated() {
                let n = Swift.max(1, size >> level)
                t.replace(region: MTLRegionMake2D(0, 0, n, n), mipmapLevel: level, slice: slice, withBytes: texels,
                          bytesPerRow: n * 4, bytesPerImage: n * n * 4)
            }
        }
        return t
    }

    static func biomeTextures(device: MTLDevice, maps: FloraMaps?) throws -> [MTLTexture] {
        let w = maps?.weights.first?.width ?? 1, h = maps?.weights.first?.height ?? 1
        return try (0..<3).map { k in
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            guard let t = device.makeTexture(descriptor: d) else { throw RendererError.setup("biome textures") }
            var texels = [Float](repeating: 0, count: w * h * 4)
            if let maps {
                for c in 0..<4 {
                    let channel = 4 * k + c
                    if channel < maps.weights.count {
                        let g = maps.weights[channel]
                        guard g.width == w, g.height == h else { continue }
                        for i in 0..<(w * h) { texels[4 * i + c] = g.values[i] }
                    } else if channel == 11 {
                        // Site factor: from the bake where it has one, middling elsewhere.
                        let site = maps.site.flatMap { $0.width == w && $0.height == h ? $0 : nil }
                        for i in 0..<(w * h) { texels[4 * i + c] = site?.values[i] ?? 0.5 }
                    }
                }
            }
            texels.withUnsafeBytes { raw in
                t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: w * 16)
            }
            return t
        }
    }

    /// The scatter's settings for this frame (FloraParams in the shader).
    func params(eye: SIMD3<Double>, viewProjection m: simd_float4x4) -> [SIMD4<Float>] {
        func row(_ i: Int) -> SIMD4<Float> { SIMD4(m.columns.0[i], m.columns.1[i], m.columns.2[i], m.columns.3[i]) }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        var out: [SIMD4<Float>] = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map { p in
            let l = simd_length(SIMD3(p.x, p.y, p.z))
            return l > 0 ? p / l : p
        }
        let canopyCell = Float(Communities.canopyCell), shrubCell = Float(Communities.shrubCell)
        out.append(SIMD4(canopyCell, shrubCell, canopyRange, shrubRange))
        out.append(SIMD4(lod0Distance, fadeWidth, seaLevel, domainSize))
        out.append(SIMD4(lod1Distance, 0, 0, 0))
        let canopyCells = Int((2 * canopyRange / canopyCell).rounded(.up)) + 2
        let shrubCells = Int((2 * shrubRange / shrubCell).rounded(.up)) + 2
        func origin(_ cell: Float, _ cells: Int) -> SIMD2<Float> {
            SIMD2((Float(eye.x) / cell).rounded(.down) - Float(cells / 2), (Float(eye.y) / cell).rounded(.down) - Float(cells / 2))
        }
        let oc = origin(canopyCell, canopyCells), os = origin(shrubCell, shrubCells)
        out.append(SIMD4(oc.x, oc.y, os.x, os.y))
        out.append(SIMD4(Float(canopyCells), Float(shrubCells), Float(FloraRenderer.capacity), Float(seed % 65536)))
        for e in Communities.table { out.append(SIMD4(Float(e.canopy), Float(e.shrubs), 0, 0)) }
        func cumulative(_ shares: [(plant: Plant, share: Double)]) -> [SIMD4<Float>] {
            var c = [Float](repeating: 0, count: 8)
            for plant in Plant.allCases {
                c[plant.rawValue] = Float(shares.first { $0.plant == plant }?.share ?? 0)
            }
            var sum: Float = 0
            for i in 0..<8 {
                sum += c[i]
                c[i] = sum
            }
            // A layer with no stems never picks; one with stems always does.
            if sum > 0 { c[7] = 2 }
            return [SIMD4(c[0], c[1], c[2], c[3]), SIMD4(c[4], c[5], c[6], c[7])]
        }
        for e in Communities.table { out += cumulative(e.canopyShares) }
        for e in Communities.table { out += cumulative(e.shrubShares) }
        for plant in Plant.allCases {
            // Canopy trees on slopes to ~40°, shrubs and ferns to ~55°.
            let steepest: Float = plant == .ohia || plant == .koa || plant == .loulu ? 0.84 : 1.43
            out.append(SIMD4(Float(plant.heightRange.lowerBound), Float(plant.heightRange.upperBound),
                             Float(plant.referenceHeight), steepest))
        }
        return out
    }

    /// Builds this frame's instances: clears the bucket counts, scatters both layers and
    /// writes the indirect draw arguments. Specimen sheets write theirs once instead.
    func scatter(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms, eye: SIMD3<Double>,
                 viewProjection: simd_float4x4, terrainParams: TerrainRenderer.TerrainParams, detail: TerrainDetail,
                 pages: MTLBuffer, tiles: MTLTexture) {
        if specimens != nil { return }
        guard let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.label = "flora clear"
        blit.fill(buffer: counts, range: 0..<counts.length, value: 0)
        blit.endEncoding()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "flora scatter"
        var p = params(eye: eye, viewProjection: viewProjection)
        var t = terrainParams
        var d = detail.params
        enc.setComputePipelineState(scatterPipeline)
        enc.setBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 0)
        enc.setBytes(&p, length: p.count * 16, index: 1)
        enc.setBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 2)
        enc.setBytes(&d, length: MemoryLayout<TerrainDetail.Params>.stride, index: 3)
        enc.setBuffer(pages, offset: 0, index: 4)
        enc.setBuffer(counts, offset: 0, index: 5)
        enc.setBuffer(instances, offset: 0, index: 6)
        enc.setTexture(tiles, index: 0)
        enc.setTexture(detail.texture, index: 1)
        enc.setTextures(biomes, range: 2..<5)
        let group = MTLSize(width: 8, height: 8, depth: 1)
        for layer in 0..<2 {
            var l = UInt32(layer)
            enc.setBytes(&l, length: 4, index: 7)
            let side = Int(layer == 0 ? p[10].x : p[10].y)
            enc.dispatchThreads(MTLSize(width: side, height: side, depth: 1), threadsPerThreadgroup: group)
        }
        enc.setComputePipelineState(argumentsPipeline)
        enc.setBuffer(meshes, offset: 0, index: 8)
        enc.setBuffer(arguments, offset: 0, index: 9)
        enc.dispatchThreads(MTLSize(width: FloraRenderer.buckets, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Rebuilds the tree shadow structure around `eye` when the camera has moved
    /// `shadowRebuild` metres since the last build, or the last build stood on tiles still
    /// loading. `settled`: every tile the view needs has loaded.
    func updateShadows(_ cmd: MTLCommandBuffer, uniforms: inout TerrainRenderer.Uniforms, eye: SIMD3<Double>,
                       terrainParams: TerrainRenderer.TerrainParams, detail: TerrainDetail, pages: MTLBuffer,
                       tiles: MTLTexture, settled: Bool) {
        guard let shadows, specimens == nil else { return }
        if let c = shadowCentre, shadowSettled, simd_distance(c, eye) < shadowRebuild { return }
        shadowCentre = eye
        shadowSettled = settled
        guard let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.label = "tree shadows clear"
        blit.fill(buffer: shadows.scattered, range: 0..<4, value: 0)
        blit.endEncoding()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "tree shadows scatter"
        var p = params(eye: eye, viewProjection: matrix_identity_float4x4)
        var t = terrainParams
        var d = detail.params
        let reach = shadowRange + shadowMargin
        let shrubReach = Swift.min(shrubRange - 10, reach)
        let canopyCell = Float(Communities.canopyCell), shrubCell = Float(Communities.shrubCell)
        let canopyCells = Int((2 * reach / canopyCell).rounded(.up)) + 2
        let shrubCells = Int((2 * shrubReach / shrubCell).rounded(.up)) + 2
        func origin(_ cell: Float, _ cells: Int) -> SIMD2<Float> {
            SIMD2((Float(eye.x) / cell).rounded(.down) - Float(cells / 2), (Float(eye.y) / cell).rounded(.down) - Float(cells / 2))
        }
        let oc = origin(canopyCell, canopyCells), os = origin(shrubCell, shrubCells)
        var sp: [SIMD4<Float>] = [SIMD4(oc.x, oc.y, os.x, os.y),
                                  SIMD4(Float(canopyCells), Float(shrubCells), reach, shrubReach),
                                  SIMD4(Float(FloraRenderer.shadowCapacity), 0, 0, 0)]
        enc.setComputePipelineState(shadows.scatter)
        enc.setBytes(&uniforms, length: MemoryLayout<TerrainRenderer.Uniforms>.stride, index: 0)
        enc.setBytes(&p, length: p.count * 16, index: 1)
        enc.setBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 2)
        enc.setBytes(&d, length: MemoryLayout<TerrainDetail.Params>.stride, index: 3)
        enc.setBuffer(pages, offset: 0, index: 4)
        enc.setBuffer(shadows.scattered, offset: 0, index: 5)
        enc.setBuffer(shadows.descriptors, offset: 0, index: 6)
        enc.setBytes(&sp, length: sp.count * 16, index: 8)
        enc.setBuffer(shadows.proxyIDs, offset: 0, index: 9)
        enc.setTexture(tiles, index: 0)
        enc.setTexture(detail.texture, index: 1)
        enc.setTextures(biomes, range: 2..<5)
        let group = MTLSize(width: 8, height: 8, depth: 1)
        for layer in 0..<2 {
            var l = UInt32(layer)
            enc.setBytes(&l, length: 4, index: 7)
            let side = layer == 0 ? canopyCells : shrubCells
            enc.dispatchThreads(MTLSize(width: side, height: side, depth: 1), threadsPerThreadgroup: group)
        }
        enc.setComputePipelineState(shadows.clamp)
        enc.setBuffer(shadows.count, offset: 0, index: 10)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        guard let build = cmd.makeAccelerationStructureCommandEncoder() else { return }
        build.label = "tree shadows build"
        build.useResources(shadows.proxies, usage: .read)
        build.build(accelerationStructure: shadows.accel, descriptor: shadows.descriptor,
                    scratchBuffer: shadows.scratch, scratchBufferOffset: 0)
        build.endEncoding()
    }

    /// Trees in the shadow structure at its last build. Only meaningful once that build has
    /// completed.
    var shadowInstances: Int {
        shadows.map { Int($0.count.contents().load(as: UInt32.self)) / 2 } ?? 0
    }

    /// Places `specimens` and writes their draw arguments; the scatter stays off.
    func place(_ list: [Specimen]) {
        specimens = list
        let table = meshes.contents().bindMemory(to: Mesh.self, capacity: FloraRenderer.buckets)
        let out = instances.contents().bindMemory(to: Instance.self, capacity: FloraRenderer.buckets * FloraRenderer.capacity)
        let args = arguments.contents().bindMemory(to: UInt32.self, capacity: FloraRenderer.buckets * 5)
        let count = counts.contents().bindMemory(to: UInt32.self, capacity: FloraRenderer.buckets)
        for b in 0..<FloraRenderer.buckets { count[b] = 0 }
        for s in list {
            let bucket = s.plant.map { ($0.rawValue * Flora.variants + s.variant) * FloraRenderer.lodBuckets + s.lod }
                ?? FloraRenderer.plantBuckets
            let k = Int(count[bucket])
            guard k < FloraRenderer.capacity else { continue }
            let reference = Float(s.plant?.referenceHeight ?? 1.7)
            out[bucket * FloraRenderer.capacity + k] = Instance(
                x: s.position.x, y: s.position.y, z: s.position.z, scale: s.height / reference,
                cosYaw: cos(s.yaw), sinYaw: sin(s.yaw), leanX: 0, leanY: 0, fade: 0, tint: 1, phase: Float(k), height: s.height)
            count[bucket] += 1
        }
        for b in 0..<FloraRenderer.buckets {
            let m = table[b]
            args[b * 5] = m.indexCount
            args[b * 5 + 1] = count[b]
            args[b * 5 + 2] = m.indexStart
            args[b * 5 + 3] = UInt32(bitPattern: m.baseVertex)
            args[b * 5 + 4] = UInt32(b * FloraRenderer.capacity)
        }
    }

    /// The mesh buckets (both LODs and the figure), then the impostors with their pipeline.
    func draw(_ enc: MTLRenderCommandEncoder, impostors: MTLRenderPipelineState) {
        enc.setVertexBuffer(vertices, offset: 0, index: 2)
        enc.setVertexBuffer(instances, offset: 0, index: 3)
        enc.setFragmentTexture(cards, index: 6)
        // Nearest LOD first, so the depth test rejects more of the farther trees' cards.
        let order = (0..<Flora.lods).flatMap { lod in
            stride(from: lod, to: FloraRenderer.plantBuckets, by: FloraRenderer.lodBuckets)
        } + [FloraRenderer.plantBuckets]
        for b in order {
            enc.drawIndexedPrimitives(type: .triangle, indexType: .uint32, indexBuffer: indices, indexBufferOffset: 0,
                                      indirectBuffer: arguments, indirectBufferOffset: b * 20)
        }
        enc.setRenderPipelineState(impostors)
        var capacity = UInt32(FloraRenderer.capacity)
        enc.setVertexBuffer(impostorExtents, offset: 0, index: 4)
        enc.setVertexBytes(&capacity, length: 4, index: 5)
        enc.setFragmentTexture(impostorAlbedo, index: 10)
        enc.setFragmentTexture(impostorNormal, index: 11)
        for b in stride(from: Flora.lods, to: FloraRenderer.plantBuckets, by: FloraRenderer.lodBuckets) {
            enc.drawIndexedPrimitives(type: .triangle, indexType: .uint32, indexBuffer: indices, indexBufferOffset: 0,
                                      indirectBuffer: arguments, indirectBufferOffset: b * 20)
        }
    }

    /// Trees in the depth prepass, after the terrain (same depth state and uniforms).
    func drawPrepass(_ enc: MTLRenderCommandEncoder) {
        enc.setRenderPipelineState(prepassPipeline)
        draw(enc, impostors: impostorPrepassPipeline)
    }

    /// Trees in the shading pass, after the terrain: the terrain's fragment textures stay
    /// bound (sky view, clouds, cloud shadows, traced sun); these add the plant textures
    /// and the ground's sky visibility and horizon map.
    func drawShading(_ enc: MTLRenderCommandEncoder, skyVisibility: MTLTexture, horizons: MTLTexture) {
        enc.setRenderPipelineState(shadePipeline)
        enc.setDepthStencilState(shadeDepthState)
        enc.setFragmentTexture(bark, index: 7)
        enc.setFragmentTexture(skyVisibility, index: 8)
        enc.setFragmentTexture(horizons, index: 9)
        draw(enc, impostors: impostorShadePipeline)
    }

    /// Instances drawn last frame (both LODs, cross-fades counted twice) and their triangles.
    /// Only meaningful once that frame has completed.
    func lastCounts() -> (instances: Int, triangles: Int) {
        let args = arguments.contents().bindMemory(to: UInt32.self, capacity: FloraRenderer.buckets * 5)
        var n = 0, tris = 0
        for b in 0..<FloraRenderer.buckets {
            let k = Int(args[b * 5 + 1])
            n += k
            tris += k * triangles[b]
        }
        return (n, tris)
    }
}
#endif
