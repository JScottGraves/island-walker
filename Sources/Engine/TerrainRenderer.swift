#if canImport(MetalKit)
import Metal
import MetalKit
import simd
import IslandCore
import Flora

/// The terrain renderer (plan §4.2): heights stream in tile by tile from the island's mip
/// pyramid, and a CDLOD quadtree draws them with morphing between levels, 10 m grid
/// spacing nearest the camera. Sun and sky lighting, ray-traced sun shadows and the cloud
/// layer are drawn over it.
public final class TerrainRenderer {
    struct NodeInstance {
        var origin: SIMD2<Float>
        var lod: Int32
        var quads: Int32
    }

    struct TerrainParams {
        var cellMetres: Float, heightOffset: Float, heightScale: Float, domainSize: Float
        var levels: Int32, tileSize: Int32, lodCount: Int32, detailLods: Int32
        var levelSize: (SIMD4<Int32>, SIMD4<Int32>)
        var levelTiles: (SIMD4<Int32>, SIMD4<Int32>)
        var pageOffset: (SIMD4<Int32>, SIMD4<Int32>)
        var morphStart: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var morphEnd: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    }

    struct Uniforms {
        var viewProjection: simd_float4x4
        var inverseViewProjection: simd_float4x4
        var cameraPosition: SIMD3<Float>
        var seaLevel: Float
        var sunDirection: SIMD3<Float>
        var maxHeight: Float
        var sunRadiance: SIMD3<Float>
        var exposure: Float
        var skySH: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                    SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var cloudLayer: SIMD4<Float>
        var cloudWind: SIMD4<Float>
        var previousViewProjection: simd_float4x4
        var previousCamera: SIMD4<Float>
        var frame: SIMD4<Float>
    }

    /// How terrain casts sun shadows.
    public enum Shadows: String, Sendable {
        /// Rays traced against the terrain mesh in the fragment shader (Metal ray tracing;
        /// hardware-accelerated on M3 and later).
        case rayTraced
        /// The baked 16-sector horizon map: cheaper, coarser, works on any GPU.
        case horizonMap
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    let skyPipeline: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let depthPipeline: MTLRenderPipelineState
    let prepassDepthState: MTLDepthStencilState
    let skyDepthState: MTLDepthStencilState
    let depthFormat: MTLPixelFormat
    let cloudMarch: MTLComputePipelineState
    let cloudResolve: MTLComputePipelineState
    let cloudShadowPipeline: MTLComputePipelineState
    /// Ray-traced sun visibility per pixel, from the prepass surface (nil without ray tracing).
    let sunShadowPipeline: MTLComputePipelineState?
    /// Sunlight through the cloud layer over the island, from the cloud base up (r16Float).
    let cloudShadows: MTLTexture
    /// Tiling cloud noise baked at start: Perlin–Worley shapes (128³) and Worley edge
    /// detail (64³), both mipmapped.
    let cloudShapeNoise: MTLTexture
    let cloudDetailNoise: MTLTexture
    /// Per-size render targets: terrain depth (read by the cloud march), this frame's
    /// reduced-resolution clouds, and the two cloud histories the resolve ping-pongs between.
    struct Targets {
        let width: Int, height: Int
        let depth: MTLTexture
        /// With ray tracing: each pixel's surface position and normal, and its sun visibility.
        let surface: MTLTexture?, sun: MTLTexture?
        let clouds: MTLTexture, cloudDistance: MTLTexture
        let history: [MTLTexture], historyDistance: [MTLTexture]
    }
    var targets: Targets?
    var historyIndex = 0
    var historyValid = false
    var frameNumber = 0
    var previousViewProjection = matrix_identity_float4x4
    var previousCamera = SIMD3<Float>(0, 0, 0)
    let terrain: TerrainTiles
    let heights: TerrainHeights
    let streamer: TerrainStreamer
    let quadtree: TerrainQuadtree
    let terrainParams: TerrainParams
    let overview: Grid
    /// Index buffers of a node's full grid and of a quarter node's half-density grid.
    let nodeIndices: (full: MTLBuffer, half: MTLBuffer)
    /// Selected nodes, one buffer per frame in flight.
    let nodeBuffers: [MTLBuffer]
    var nodeBufferIndex = 0
    static let maxNodes = 4096
    let skyVisibility: MTLTexture
    let horizons: MTLTexture
    /// The overview heights the shadow mesh is built from, for `sunShadow` to tell crowns
    /// standing above that mesh from trunks and low leaves below it.
    let shadowMeshHeights: MTLTexture
    let terrainAccel: MTLAccelerationStructure?
    /// Generated ground below the data's 10 m grid, near the camera.
    let detail: TerrainDetail
    /// Plant-community ground colour (rgb) with alpha 1, or one texel of alpha 0 without one.
    let canopy: MTLTexture
    /// Fraction of the ground the tree crowns cover (r32Float), for shading the floor under
    /// the drawn trees until they cast shadows (zero without trees).
    let canopyCover: MTLTexture
    let seaLevel: Float
    let maxHeight: Float
    /// Native trees and shrubs (nil when drawn without them).
    let flora: FloraRenderer?
    /// What the last frame drew and how much is still loading, for the debug overlay.
    public private(set) var stats = Stats()
    public struct Stats: Sendable {
        public var nodes = 0
        public var residentTiles = 0
        public var loadingTiles = 0
        /// GPU time of the last completed frame drawn with `draw(in:)`, or of a settled frame
        /// in `snapshot`, in milliseconds.
        public var gpuMilliseconds = 0.0
        /// `snapshot` only: depth prepass, clouds (shadow map, march, resolve), shading.
        public var passMilliseconds: [Double] = []
        /// `snapshot` only: tree instances drawn (cross-fading ones twice) and their triangles.
        public var floraInstances = 0
        public var floraTriangles = 0
        /// `snapshot` only: trees casting traced shadows.
        public var shadowTrees = 0
    }
    /// Which shadow path this renderer uses (ray tracing unless unsupported or declined).
    public let shadows: Shadows
    /// The sun and sky to light the next frame with; set it to move the sun.
    public var light = SunLight.reference {
        didSet { uploadSkyView() }
    }
    /// The sky-view LUT, one texture per frame in flight so an upload never overwrites
    /// one the GPU is still reading.
    let skyViews: [MTLTexture]
    var skyViewIndex = 0

    /// Baked orographic cloud coverage (islandgen's clouds-trades.r32) over the domain.
    let cloudMap: MTLTexture
    let domainSize: Float
    /// Local solar time (hours) for the clouds' daily cycle; keep it with `light`'s sun.
    public var hour = 9.5
    /// Seconds of wind so far: how far the trade wind has carried the cloud texture.
    public var clock: Double = 0
    /// Direction the trade wind blows towards (radians, x east, y north) and its speed.
    public var windDirection = 5 * Double.pi / 4
    public var windSpeed = 7.5
    /// Trade-wind inversion: the flat cap on cloud tops (normal regime ~2,000 m).
    public var inversionHeight = 2000.0

    /// Number of quads per side of a CDLOD node's grid.
    public static let nodeQuads = 64

    /// - Parameters:
    ///   - terrain: the heightfield, streamed tile by tile from its mip pyramid.
    ///   - clouds: islandgen's standing cloud coverage over the same domain.
    ///   - detailMaps: the bake's rain, drainage and hardness, to shape the fine detail.
    ///   - fineDetail: false draws only the data's own grid (for comparisons).
    ///   - flora: the plant-community maps the native trees are scattered from; nil draws
    ///     no trees. `floraSpecimens` places them by hand instead (floragen).
    public init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat,
                terrain: TerrainTiles, shadows requested: Shadows = .rayTraced, clouds: Grid? = nil,
                detailMaps: DetailMaps? = nil, fineDetail: Bool = true, flora floraMaps: FloraMaps? = nil,
                floraSpecimens: Bool = false) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw RendererError.setup("command queue") }
        self.queue = queue
        self.terrain = terrain
        self.seaLevel = terrain.seaLevel
        self.depthFormat = depthFormat
        heights = TerrainHeights(tiles: terrain)

        let shadows: Shadows = requested == .rayTraced && device.supportsRaytracingFromRender ? .rayTraced : .horizonMap
        self.shadows = shadows
        let options = MTLCompileOptions()
        options.preserveInvariance = true  // the depth prepass and shading pass must agree
        let library = try device.makeLibrary(source: terrainShaderSource + floraShaderSource, options: options)
        let constants = MTLFunctionConstantValues()
        var rayTraced = shadows == .rayTraced
        constants.setConstantValue(&rayTraced, type: .bool, index: 0)
        // Trees cast traced shadows when they are scattered (not floragen's hand-placed sheets).
        var treeShadows = rayTraced && floraMaps != nil
        constants.setConstantValue(&treeShadows, type: .bool, index: 1)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "terrainVertex")
        desc.fragmentFunction = try library.makeFunction(name: "terrainFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = depthFormat
        pipeline = try device.makeRenderPipelineState(descriptor: desc)
        // Depth only: the terrain is shaded (with its cloud march) once per pixel, not once
        // per overlapping triangle, and the sky only where the terrain leaves it uncovered.
        // With ray tracing it also records each pixel's surface for the sun rays.
        if shadows == .rayTraced {
            desc.fragmentFunction = library.makeFunction(name: "terrainSurface")
            desc.colorAttachments[0].pixelFormat = .rgba32Uint
            let sun = try library.makeFunction(name: "sunShadow", constantValues: constants)
            sunShadowPipeline = try device.makeComputePipelineState(function: sun)
        } else {
            desc.fragmentFunction = nil
            desc.colorAttachments[0].pixelFormat = .invalid  // a pass of its own, with no colour target
            sunShadowPipeline = nil
        }
        depthPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.vertexFunction = library.makeFunction(name: "skyVertex")
        desc.fragmentFunction = library.makeFunction(name: "skyFragment")
        skyPipeline = try device.makeRenderPipelineState(descriptor: desc)
        guard let march = library.makeFunction(name: "cloudMarch"),
              let resolve = library.makeFunction(name: "cloudResolve"),
              let shadowMap = library.makeFunction(name: "cloudShadowMap") else { throw RendererError.setup("cloud kernels") }
        cloudMarch = try device.makeComputePipelineState(function: march)
        cloudResolve = try device.makeComputePipelineState(function: resolve)
        cloudShadowPipeline = try device.makeComputePipelineState(function: shadowMap)
        // ~160 m texels over an 80 km island: finer than the cloud shapes.
        let shadowDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 512, height: 512, mipmapped: false)
        shadowDesc.usage = [.shaderRead, .shaderWrite]
        shadowDesc.storageMode = .private
        guard let shadowTexture = device.makeTexture(descriptor: shadowDesc) else { throw RendererError.setup("cloud shadow map") }
        cloudShadows = shadowTexture
        guard let shapeBake = library.makeFunction(name: "cloudShapeNoise"),
              let detailBake = library.makeFunction(name: "cloudDetailNoise") else { throw RendererError.setup("cloud noise kernels") }
        cloudShapeNoise = try TerrainRenderer.bakeNoise(device: device, queue: queue, side: 128,
                                                        kernel: try device.makeComputePipelineState(function: shapeBake))
        cloudDetailNoise = try TerrainRenderer.bakeNoise(device: device, queue: queue, side: 64,
                                                         kernel: try device.makeComputePipelineState(function: detailBake))

        let prepassDesc = MTLDepthStencilDescriptor()
        prepassDesc.depthCompareFunction = .less
        prepassDesc.isDepthWriteEnabled = true
        guard let pds = device.makeDepthStencilState(descriptor: prepassDesc) else { throw RendererError.setup("prepass depth state") }
        prepassDepthState = pds
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("depth state") }
        depthState = ds
        let skyDepthDesc = MTLDepthStencilDescriptor()
        skyDepthDesc.depthCompareFunction = .lessEqual
        skyDepthDesc.isDepthWriteEnabled = false
        guard let sds = device.makeDepthStencilState(descriptor: skyDepthDesc) else { throw RendererError.setup("sky depth state") }
        skyDepthState = sds

        // Streaming and LOD. The overview level (at most 1024² samples) stays in memory for
        // node bounds, the lighting bake and the shadow mesh.
        let streamer = try TerrainStreamer(device: device, source: terrain)
        self.streamer = streamer
        let overview = try terrain.grid(level: terrain.overviewLevel())
        self.overview = overview
        maxHeight = overview.range.max
        domainSize = Float(terrain.domainMetres)
        // The finest CDLOD levels draw generated detail, each half the spacing of the next.
        let detailLods = fineDetail ? TerrainDetail.levelCount(cellMetres: terrain.cellMetres) : 0
        detail = try TerrainDetail(device: device, library: library, levels: detailLods, cellMetres: terrain.cellMetres,
                                   seaLevel: terrain.seaLevel, seed: TerrainDetail.seed(of: overview), maps: detailMaps)
        canopy = try TerrainRenderer.canopyTexture(device: device, map: detailMaps?.canopy)
        canopyCover = try TerrainRenderer.canopyCoverTexture(device: device, maps: floraMaps)
        flora = floraMaps != nil || floraSpecimens
            ? try FloraRenderer(device: device, library: library, constants: constants, colorFormat: colorFormat,
                                depthFormat: depthFormat, rayTraced: shadows == .rayTraced, treeShadows: treeShadows,
                                maps: floraMaps,
                                seed: TerrainDetail.seed(of: overview), seaLevel: terrain.seaLevel,
                                domainSize: Float(terrain.domainMetres))
            : nil
        // Rings two node sizes wide: three drew 40% more triangles, mostly a few pixels across
        // far away (shading 3 ms slower at 1440p), with no relief visible in the difference.
        let tree = TerrainQuadtree(quads: TerrainRenderer.nodeQuads, cellMetres: terrain.cellMetres / Double(1 << detailLods),
                                   domain: terrain.domainMetres, overview: overview, rangeFactor: 2)
        quadtree = tree
        guard terrain.levels <= 8, tree.lodCount <= 16 else { throw RendererError.setup("terrain too large for the LOD tables") }
        func ints(_ f: (Int) -> Int) -> (SIMD4<Int32>, SIMD4<Int32>) {
            let v = (0..<8).map { $0 < terrain.levels ? Int32(f($0)) : 0 }
            return (SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]))
        }
        func floats(_ a: [Double]) -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
            let v = (0..<16).map { $0 < a.count ? Float(Swift.min(a[$0], 1e30)) : 1e30 }
            return (SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                    SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15]))
        }
        let pageOffsets = streamer.pageOffsets
        terrainParams = TerrainParams(
            cellMetres: Float(terrain.cellMetres), heightOffset: terrain.offset, heightScale: terrain.scale,
            domainSize: Float(terrain.domainMetres), levels: Int32(terrain.levels), tileSize: Int32(terrain.tileSize),
            lodCount: Int32(tree.lodCount), detailLods: Int32(detailLods),
            levelSize: ints { terrain.levelSize($0) }, levelTiles: ints { terrain.tilesPerSide($0) },
            pageOffset: ints { pageOffsets[$0] },
            morphStart: floats(tree.morphStart), morphEnd: floats(tree.morphEnd))
        func grid(_ quads: Int) -> MTLBuffer? {
            let n = quads + 1
            var idx = [UInt16]()
            idx.reserveCapacity(quads * quads * 6)
            for j in 0..<quads {
                for i in 0..<quads {
                    let a = UInt16(j * n + i), b = a + 1, c = a + UInt16(n), d = c + 1
                    idx.append(contentsOf: [a, b, d, a, d, c])
                }
            }
            return device.makeBuffer(bytes: idx, length: idx.count * 2, options: .storageModeShared)
        }
        guard let full = grid(TerrainRenderer.nodeQuads), let half = grid(TerrainRenderer.nodeQuads / 2) else {
            throw RendererError.setup("node grids")
        }
        nodeIndices = (full, half)
        nodeBuffers = try (0..<3).map { _ in
            guard let b = device.makeBuffer(length: TerrainRenderer.maxNodes * MemoryLayout<NodeInstance>.stride,
                                            options: .storageModeShared) else { throw RendererError.setup("node buffer") }
            return b
        }

        // Sky visibility and the horizon map, baked from the overview. With ray tracing the
        // horizon map decides which pixels need rays at all.
        let lighting = TerrainLighting.bake(height: overview, seaLevel: terrain.seaLevel, horizons: true)
        let skyVisDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: overview.width,
                                                                  height: overview.height, mipmapped: false)
        skyVisDesc.usage = [.shaderRead]
        skyVisDesc.storageMode = .shared
        guard let skyVis = device.makeTexture(descriptor: skyVisDesc) else { throw RendererError.setup("sky visibility") }
        lighting.sky.values.withUnsafeBytes { raw in
            skyVis.replace(region: MTLRegionMake2D(0, 0, overview.width, overview.height), mipmapLevel: 0,
                           withBytes: raw.baseAddress!, bytesPerRow: overview.width * 4)
        }
        skyVisibility = skyVis
        guard let meshHeights = device.makeTexture(descriptor: skyVisDesc) else { throw RendererError.setup("shadow mesh heights") }
        overview.values.withUnsafeBytes { raw in
            meshHeights.replace(region: MTLRegionMake2D(0, 0, overview.width, overview.height), mipmapLevel: 0,
                                withBytes: raw.baseAddress!, bytesPerRow: overview.width * 4)
        }
        shadowMeshHeights = meshHeights
        // 16 sectors as 4 RGBA slices.
        let horizonSide = lighting.horizon.isEmpty ? 1 : overview.width
        let horizonDesc = MTLTextureDescriptor()
        horizonDesc.textureType = .type2DArray
        horizonDesc.pixelFormat = .rgba8Unorm
        horizonDesc.width = horizonSide
        horizonDesc.height = horizonSide
        horizonDesc.arrayLength = 4
        horizonDesc.usage = [.shaderRead]
        horizonDesc.storageMode = .shared
        guard let horizonTexture = device.makeTexture(descriptor: horizonDesc) else { throw RendererError.setup("horizon map") }
        let sectors = TerrainLighting.sectors
        for slice in 0..<4 {
            var bytes = [UInt8](repeating: 0, count: horizonSide * horizonSide * 4)
            if !lighting.horizon.isEmpty {
                for i in 0..<(horizonSide * horizonSide) {
                    for c in 0..<4 { bytes[i * 4 + c] = lighting.horizon[i * sectors + slice * 4 + c] }
                }
            }
            horizonTexture.replace(region: MTLRegionMake2D(0, 0, horizonSide, horizonSide), mipmapLevel: 0, slice: slice,
                                   withBytes: bytes, bytesPerRow: horizonSide * 4, bytesPerImage: bytes.count)
        }
        horizons = horizonTexture

        let skyDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: Atmosphere.skyViewSize.width,
            height: Atmosphere.skyViewSize.height, mipmapped: false)
        skyDesc.usage = [.shaderRead]
        skyDesc.storageMode = .shared
        let coverage = clouds ?? Grid(width: 1, height: 1, cellSize: 1)
        let cloudDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: coverage.width, height: coverage.height, mipmapped: false)
        cloudDesc.usage = [.shaderRead]
        cloudDesc.storageMode = .shared
        guard let cloudTexture = device.makeTexture(descriptor: cloudDesc) else { throw RendererError.setup("cloud map") }
        coverage.values.withUnsafeBytes { raw in
            cloudTexture.replace(region: MTLRegionMake2D(0, 0, coverage.width, coverage.height), mipmapLevel: 0,
                                 withBytes: raw.baseAddress!, bytesPerRow: coverage.width * 4)
        }
        cloudMap = cloudTexture
        skyViews = try (0..<3).map { _ in
            guard let t = device.makeTexture(descriptor: skyDesc) else { throw RendererError.setup("sky-view texture") }
            return t
        }
        // Sun shadows are traced against a mesh of the overview: coarser than the drawn
        // terrain, but it holds the whole island (the shader lets rays out from below it).
        terrainAccel = shadows == .rayTraced
            ? try TerrainRenderer.buildShadowMesh(device: device, queue: queue, overview: overview)
            : nil
        uploadSkyView()
    }

    /// A cube of tiling noise written by `kernel`, with its mip chain.
    static func bakeNoise(device: MTLDevice, queue: MTLCommandQueue, side: Int,
                          kernel: MTLComputePipelineState) throws -> MTLTexture {
        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rgba8Unorm
        desc.width = side
        desc.height = side
        desc.depth = side
        desc.mipmapLevelCount = Int(log2(Double(side))) + 1
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let texture = device.makeTexture(descriptor: desc), let cmd = queue.makeCommandBuffer(),
              let compute = cmd.makeComputeCommandEncoder() else { throw RendererError.setup("cloud noise") }
        compute.setComputePipelineState(kernel)
        compute.setTexture(texture, index: 0)
        compute.dispatchThreads(MTLSize(width: side, height: side, depth: side),
                                threadsPerThreadgroup: MTLSize(width: 4, height: 4, depth: 4))
        compute.endEncoding()
        guard let blit = cmd.makeBlitCommandEncoder() else { throw RendererError.setup("cloud noise mips") }
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return texture
    }

    /// Convenience for an in-memory heightfield (a height.r32 or a quick bake).
    public convenience init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat,
                            height: Grid, seaLevel: Float, shadows: Shadows = .rayTraced, clouds: Grid? = nil) throws {
        try self.init(device: device, colorFormat: colorFormat, depthFormat: depthFormat,
                      terrain: GridTerrain(height: height, seaLevel: seaLevel), shadows: shadows, clouds: clouds)
    }

    /// Places trees by hand (floragen's specimen sheets) instead of scattering them; needs
    /// a renderer made with `floraSpecimens`. Heights are metres; LOD 0 or 1; a nil plant is
    /// a 1.7 m scale figure.
    public func placeSpecimens(_ list: [(plant: Plant?, variant: Int, lod: Int, x: Double, y: Double, yaw: Double, height: Double)]) {
        flora?.place(list.map { s in
            FloraRenderer.Specimen(plant: s.plant, variant: s.variant, lod: s.lod,
                                   position: SIMD3(Float(s.x), Float(s.y), Float(groundHeight(x: s.x, y: s.y))),
                                   yaw: Float(s.yaw), height: Float(s.height))
        })
    }

    /// Ground height (metres) under a world position, as drawn up close: for the walker.
    public func groundHeight(x: Double, y: Double) -> Double {
        detail.height(x: x, y: y) ?? heights.height(x: x, y: y)
    }

    /// Generates the fine ground around (x, y) now, so `groundHeight` there returns the
    /// detailed surface before the first frame is drawn (snapshots, spawning).
    public func prepareGround(x: Double, y: Double) {
        let eye = SIMD3(x, y, heights.height(x: x, y: y))
        streamer.request(detail.neededTiles(eye: eye, streamer: streamer), synchronously: true)
        guard let cmd = queue.makeCommandBuffer() else { return }
        detail.update(cmd, eye: eye, streamer: streamer, terrainParams: terrainParams, pages: streamer.update())
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Vertices at the overview's cell centres, quads split along (i, j)–(i+1, j+1): the
    /// surface `Grid.meshHeight` and the shader's `shadowMeshHeight` reproduce.
    static func buildShadowMesh(device: MTLDevice, queue: MTLCommandQueue, overview g: Grid) throws -> MTLAccelerationStructure {
        let n = g.width
        var positions = [Float]()
        positions.reserveCapacity(n * n * 3)
        for j in 0..<n {
            for i in 0..<n {
                positions.append(contentsOf: [Float((Double(i) + 0.5) * g.cellSize), Float((Double(j) + 0.5) * g.cellSize), g[i, j]])
            }
        }
        var idx = [UInt32]()
        idx.reserveCapacity((n - 1) * (n - 1) * 6)
        for j in 0..<(n - 1) {
            for i in 0..<(n - 1) {
                let a = UInt32(j * n + i), b = a + 1, c = a + UInt32(n), d = c + 1
                idx.append(contentsOf: [a, b, d, a, d, c])
            }
        }
        guard let vb = device.makeBuffer(bytes: positions, length: positions.count * 4, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared)
        else { throw RendererError.setup("shadow mesh") }
        return try buildAccelerationStructure(device: device, queue: queue, vertices: vb, stride: 12,
                                              indices: ib, triangles: idx.count / 3)
    }

    /// Cloud base (lifting condensation level): ~400 m at night, ~800 m early afternoon
    /// (Hartley & Chen; research/clouds.md §2).
    var cloudBase: Double { 600 - 200 * cos(2 * Double.pi * (hour - 14) / 24) }

    /// Windward orographic cloud is thickest around dawn and thinnest mid-afternoon.
    var orographicGain: Double { 0.85 + 0.15 * cos(2 * Double.pi * (hour - 7) / 24) }

    func uploadSkyView() {
        skyViewIndex = (skyViewIndex + 1) % skyViews.count
        let (w, h) = Atmosphere.skyViewSize
        var texels = [Float]()
        texels.reserveCapacity(w * h * 4)
        for c in light.skyView { texels += [Float(c.r), Float(c.g), Float(c.b), 1] }
        skyViews[skyViewIndex].replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                                       withBytes: texels, bytesPerRow: w * 16)
    }

    /// One bottom-level acceleration structure over the whole terrain mesh, built once and
    /// compacted. Shares the render vertex buffer: positions are its first 12 bytes.
    static func buildAccelerationStructure(device: MTLDevice, queue: MTLCommandQueue, vertices: MTLBuffer,
                                           stride: Int, indices: MTLBuffer, triangles: Int) throws -> MTLAccelerationStructure {
        let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometry.vertexBuffer = vertices
        geometry.vertexStride = stride
        geometry.vertexFormat = .float3
        geometry.indexBuffer = indices
        geometry.indexType = .uint32
        geometry.triangleCount = triangles
        geometry.opaque = true
        let desc = MTLPrimitiveAccelerationStructureDescriptor()
        desc.geometryDescriptors = [geometry]

        let sizes = device.accelerationStructureSizes(descriptor: desc)
        guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
              let compactedSize = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(), let enc = cmd.makeAccelerationStructureCommandEncoder()
        else { throw RendererError.setup("acceleration structure") }
        enc.build(accelerationStructure: accel, descriptor: desc, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.writeCompactedSize(accelerationStructure: accel, buffer: compactedSize, offset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let size = Int(compactedSize.contents().load(as: UInt32.self))
        guard let compact = device.makeAccelerationStructure(size: size),
              let cmd2 = queue.makeCommandBuffer(), let enc2 = cmd2.makeAccelerationStructureCommandEncoder()
        else { throw RendererError.setup("compacted acceleration structure") }
        enc2.copyAndCompact(sourceAccelerationStructure: accel, destinationAccelerationStructure: compact)
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return compact
    }

    /// Clear colour behind the sky pass (which covers every pixel): the horizon haze, tone
    /// mapped like the terrain (linear, since the target applies the sRGB encoding on write).
    public static var skyClearColor: MTLClearColor {
        // Only shows if the sky pass is skipped; the reference sun's haze is close enough.
        let c = Look.toneMap(Look.haze)
        return MTLClearColor(red: c.r, green: c.g, blue: c.b, alpha: 1)
    }

    /// Renders one frame into the view's drawable.
    public func draw(in view: MTKView, eye: SIMD3<Double>, forward: SIMD3<Float>) {
        guard let drawable = view.currentDrawable, let cmd = queue.makeCommandBuffer() else { return }
        encode([cmd], color: drawable.texture, eye: eye, forward: forward)
        cmd.present(drawable)
        cmd.addCompletedHandler { [weak self] done in
            let ms = (done.gpuEndTime - done.gpuStartTime) * 1000
            DispatchQueue.main.async { self?.stats.gpuMilliseconds = ms }
        }
        cmd.commit()
    }

    /// Number of frames a snapshot renders at one camera so the cloud history converges.
    public static let snapshotFrames = 24

    /// Renders a view offscreen and returns it as 8-bit RGB, top row first: `snapshotFrames`
    /// frames at the same camera, so the clouds look as they do in the running app.
    /// Lets tools and CI inspect what the app draws without a window or screen capture.
    public func snapshot(width: Int, height: Int, eye: SIMD3<Double>, forward: SIMD3<Float>,
                         colorFormat: MTLPixelFormat = .bgra8Unorm_srgb) throws -> [UInt8] {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: colorFormat, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget]
        colorDesc.storageMode = .shared
        guard let color = device.makeTexture(descriptor: colorDesc), let cmd = queue.makeCommandBuffer(),
              let timed = queue.makeCommandBuffer()
        else { throw RendererError.setup("snapshot targets") }
        // A fresh history, then enough frames at this camera for it to settle.
        historyValid = false
        for k in 0..<TerrainRenderer.snapshotFrames {
            // Load every tile the view needs before the first frame: snapshots don't stream.
            encode([cmd], color: color, eye: eye, forward: forward, synchronousStreaming: k == 0)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        // Timing, on an idle GPU: settled frames back to back as the app draws them, then
        // one frame split into its passes, each run alone.
        let timedFrames = 8
        for _ in 0..<timedFrames { encode([timed], color: color, eye: eye, forward: forward) }
        timed.commit()
        timed.waitUntilCompleted()
        stats.gpuMilliseconds = (timed.gpuEndTime - timed.gpuStartTime) * 1000 / Double(timedFrames)
        if let counts = flora?.lastCounts() {
            stats.floraInstances = counts.instances
            stats.floraTriangles = counts.triangles
            stats.shadowTrees = flora?.shadowInstances ?? 0
        }
        let passes = (0..<3).compactMap { _ in queue.makeCommandBuffer() }
        if passes.count == 3 {
            encode(passes, color: color, eye: eye, forward: forward)
            stats.passMilliseconds = passes.map { pass in
                pass.commit()
                pass.waitUntilCompleted()
                return (pass.gpuEndTime - pass.gpuStartTime) * 1000
            }
        }

        var bgra = [UInt8](repeating: 0, count: width * height * 4)
        bgra.withUnsafeMutableBytes { buf in
            color.getBytes(buf.baseAddress!, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) {
            rgb[p * 3] = bgra[p * 4 + 2]
            rgb[p * 3 + 1] = bgra[p * 4 + 1]
            rgb[p * 3 + 2] = bgra[p * 4]
        }
        return rgb
    }

    func makeTargets(width: Int, height: Int) -> Targets? {
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: depthFormat, width: width, height: height, mipmapped: false)
        depthDesc.usage = [.renderTarget, .shaderRead]
        depthDesc.storageMode = .private
        // Clouds at no more than 360 rows (half resolution at 720p, a quarter at 1440p),
        // so their cost stays near that of a 720p frame; the history smooths the rest.
        let scale = Swift.max(2, (height + 359) / 360)
        let cloudWidth = (width + scale - 1) / scale, cloudHeight = (height + scale - 1) / scale
        func reduced(_ format: MTLPixelFormat) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: cloudWidth, height: cloudHeight, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        var surface: MTLTexture?, sun: MTLTexture?
        if sunShadowPipeline != nil {
            let surfaceDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Uint, width: width, height: height, mipmapped: false)
            surfaceDesc.usage = [.renderTarget, .shaderRead]
            surfaceDesc.storageMode = .private
            // Terrain visibility (r), the trees' cover (gb) and what they let through (a).
            let sunDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
            sunDesc.usage = [.shaderRead, .shaderWrite]  // read_write in sunShadow's second pass
            sunDesc.storageMode = .private
            surface = device.makeTexture(descriptor: surfaceDesc)
            sun = device.makeTexture(descriptor: sunDesc)
            if surface == nil || sun == nil { return nil }
        }
        guard let depth = device.makeTexture(descriptor: depthDesc),
              let clouds = reduced(.rgba16Float), let cloudDistance = reduced(.rg32Float),
              let h0 = reduced(.rgba16Float), let h1 = reduced(.rgba16Float),
              let d0 = reduced(.rg32Float), let d1 = reduced(.rg32Float) else { return nil }
        return Targets(width: width, height: height, depth: depth, surface: surface, sun: sun,
                       clouds: clouds, cloudDistance: cloudDistance, history: [h0, h1], historyDistance: [d0, d1])
    }

    struct FrameTerrain {
        var pages: MTLBuffer
        var nodes: MTLBuffer
        var full: Int
        var half: Int
    }

    /// Picks this frame's nodes, asks for their tiles and uploads what has loaded.
    func prepareTerrain(eye: SIMD3<Double>, viewProjection m: simd_float4x4, synchronously: Bool) -> FrameTerrain {
        // Frustum planes in camera-relative space, from the rows of the view-projection.
        func row(_ i: Int) -> SIMD4<Double> {
            SIMD4(Double(m.columns.0[i]), Double(m.columns.1[i]), Double(m.columns.2[i]), Double(m.columns.3[i]))
        }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        let planes: [SIMD4<Double>] = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
        // A box is outside when its corner furthest along a plane's normal is behind it.
        func visible(_ lo: SIMD3<Double>, _ hi: SIMD3<Double>) -> Bool {
            let a: SIMD3<Double> = lo - eye, b: SIMD3<Double> = hi - eye
            for p in planes {
                let x: Double = p.x > 0 ? b.x : a.x
                let y: Double = p.y > 0 ? b.y : a.y
                let z: Double = p.z > 0 ? b.z : a.z
                let distance: Double = p.x * x + p.y * y + p.z * z + p.w
                if distance < 0 { return false }
            }
            return true
        }
        let selected = quadtree.select(eye: (eye.x, eye.y, eye.z), visible: visible)
        var keys: [TerrainStreamer.Key] = []
        let detailLods = detail.levels
        for node in selected {
            let level = Swift.min(Swift.max(node.lod - detailLods, 0), terrain.levels - 1)
            keys += streamer.keys(level: level, x: node.x, y: node.y, size: node.size)
            if node.lod >= detailLods && level + 1 < terrain.levels {
                keys += streamer.keys(level: level + 1, x: node.x, y: node.y, size: node.size)
            }
        }
        keys += detail.neededTiles(eye: eye, streamer: streamer)
        streamer.request(keys, synchronously: synchronously)
        let pages = streamer.update()

        nodeBufferIndex = (nodeBufferIndex + 1) % nodeBuffers.count
        let buffer = nodeBuffers[nodeBufferIndex]
        let out = buffer.contents().bindMemory(to: NodeInstance.self, capacity: TerrainRenderer.maxNodes)
        let full = selected.filter { !$0.half }, half = selected.filter { $0.half }
        var count = 0
        for node in full + half where count < TerrainRenderer.maxNodes {
            out[count] = NodeInstance(origin: SIMD2(Float(node.x), Float(node.y)), lod: Int32(node.lod),
                                      quads: Int32(node.half ? TerrainRenderer.nodeQuads / 2 : TerrainRenderer.nodeQuads))
            count += 1
        }
        stats.nodes = count
        stats.residentTiles = streamer.residentCount
        stats.loadingTiles = streamer.backlog
        let fullCount = Swift.min(full.count, count)
        return FrameTerrain(pages: pages, nodes: buffer, full: fullCount, half: count - fullCount)
    }

    func bindTerrain(_ enc: MTLRenderCommandEncoder, frame: FrameTerrain) {
        var params = terrainParams
        enc.setVertexBytes(&params, length: MemoryLayout<TerrainParams>.stride, index: 3)
        enc.setVertexBuffer(frame.pages, offset: 0, index: 5)
        enc.setVertexTexture(streamer.tiles, index: 0)
        enc.setVertexTexture(skyVisibility, index: 1)
        enc.setVertexTexture(horizons, index: 2)
        var detailParams = detail.params
        enc.setVertexBytes(&detailParams, length: MemoryLayout<TerrainDetail.Params>.stride, index: 6)
        enc.setVertexTexture(detail.texture, index: 3)
    }

    func drawTerrain(_ enc: MTLRenderCommandEncoder, frame: FrameTerrain) {
        let stride = MemoryLayout<NodeInstance>.stride
        let quads = TerrainRenderer.nodeQuads
        if frame.full > 0 {
            enc.setVertexBuffer(frame.nodes, offset: 0, index: 4)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: quads * quads * 6, indexType: .uint16,
                                      indexBuffer: nodeIndices.full, indexBufferOffset: 0, instanceCount: frame.full)
        }
        if frame.half > 0 {
            enc.setVertexBuffer(frame.nodes, offset: frame.full * stride, index: 4)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: (quads / 2) * (quads / 2) * 6, indexType: .uint16,
                                      indexBuffer: nodeIndices.half, indexBufferOffset: 0, instanceCount: frame.half)
        }
    }

    /// Encodes one frame. With three command buffers, the depth prepass, the cloud passes and
    /// the shading pass go one into each, so each can be timed on its own.
    private func encode(_ cmds: [MTLCommandBuffer], color: MTLTexture, eye: SIMD3<Double>, forward: SIMD3<Float>,
                        synchronousStreaming: Bool = false) {
        let cmd = cmds[0]
        let cloudCmd = cmds[Swift.min(1, cmds.count - 1)], shadeCmd = cmds[Swift.min(2, cmds.count - 1)]
        let width = color.width, height = color.height
        if targets?.width != width || targets?.height != height {
            targets = makeTargets(width: width, height: height)
            historyValid = false
        }
        guard let targets else { return }
        let aspect = Float(width) / Float(Swift.max(height, 1))
        let proj = Matrix.perspective(fovY: 60 * .pi / 180, aspect: aspect, near: 0.3, far: 150_000)
        let viewProjection = proj * Matrix.lookDirection(forward)
        func f3(_ c: RGB) -> SIMD3<Float> { SIMD3(Float(c.r), Float(c.g), Float(c.b)) }
        func f4(_ k: Int) -> SIMD4<Float> { let c = light.skySH[k]; return SIMD4(Float(c.r), Float(c.g), Float(c.b), 0) }
        let d = light.direction
        var u = Uniforms(
            viewProjection: viewProjection,
            inverseViewProjection: viewProjection.inverse,
            cameraPosition: SIMD3<Float>(Float(eye.x), Float(eye.y), Float(eye.z)),
            seaLevel: seaLevel,
            sunDirection: SIMD3<Float>(Float(d.x), Float(d.y), Float(d.z)),
            maxHeight: maxHeight,
            sunRadiance: f3(light.sun),
            exposure: Float(light.exposure),
            skySH: (f4(0), f4(1), f4(2), f4(3), f4(4), f4(5), f4(6), f4(7), f4(8)),
            cloudLayer: SIMD4(Float(cloudBase), Float(inversionHeight),
                              // Wrapped to keep the noise coordinates small enough for Float.
                              Float(clock.truncatingRemainder(dividingBy: 86_400 * 4)), domainSize),
            cloudWind: SIMD4(Float(cos(windDirection) * windSpeed), Float(sin(windDirection) * windSpeed),
                             Float(orographicGain), 0),
            previousViewProjection: previousViewProjection,
            previousCamera: SIMD4(previousCamera, 0),
            frame: SIMD4(Float(width), Float(height), Float(frameNumber % 4096), historyValid ? 1 : 0))
        let skyView = skyViews[skyViewIndex]
        let frameTerrain = prepareTerrain(eye: eye, viewProjection: viewProjection, synchronously: synchronousStreaming)
        // 0. Fine detail for the ground the camera has moved over, then this frame's trees on it.
        detail.update(cmd, eye: eye, streamer: streamer, terrainParams: terrainParams, pages: frameTerrain.pages)
        flora?.scatter(cmd, uniforms: &u, eye: eye, viewProjection: viewProjection, terrainParams: terrainParams,
                       detail: detail, pages: frameTerrain.pages, tiles: streamer.tiles)
        flora?.updateShadows(cmd, uniforms: &u, eye: eye, terrainParams: terrainParams, detail: detail,
                             pages: frameTerrain.pages, tiles: streamer.tiles, settled: streamer.backlog == 0)

        // 1. Terrain depth.
        let depthPass = MTLRenderPassDescriptor()
        depthPass.depthAttachment.texture = targets.depth
        depthPass.depthAttachment.loadAction = .clear
        depthPass.depthAttachment.storeAction = .store
        depthPass.depthAttachment.clearDepth = 1
        if let surface = targets.surface {
            depthPass.colorAttachments[0].texture = surface
            depthPass.colorAttachments[0].loadAction = .dontCare  // read only where depth < 1
            depthPass.colorAttachments[0].storeAction = .store
        }
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: depthPass) else { return }
        enc.label = "depth"
        enc.setRenderPipelineState(depthPipeline)
        enc.setDepthStencilState(prepassDepthState)
        enc.setCullMode(.none)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        bindTerrain(enc, frame: frameTerrain)
        drawTerrain(enc, frame: frameTerrain)
        flora?.drawPrepass(enc)
        enc.endEncoding()

        // 2. The cloud shadow map, then clouds at reduced resolution, marched up to that depth
        // and blended into the history.
        let previous = historyIndex, next = 1 - historyIndex
        guard let compute = cloudCmd.makeComputeCommandEncoder() else { return }
        compute.label = "clouds"
        let group = MTLSize(width: 8, height: 8, depth: 1)
        compute.setComputePipelineState(cloudShadowPipeline)
        compute.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        compute.setTextures([cloudMap, cloudShadows, cloudShapeNoise, cloudDetailNoise], range: 0..<4)
        compute.dispatchThreads(MTLSize(width: cloudShadows.width, height: cloudShadows.height, depth: 1),
                                threadsPerThreadgroup: group)
        let grid = MTLSize(width: targets.clouds.width, height: targets.clouds.height, depth: 1)
        compute.setComputePipelineState(cloudMarch)
        compute.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        compute.setTextures([targets.depth, skyView, cloudMap, targets.clouds, targets.cloudDistance,
                             cloudShapeNoise, cloudDetailNoise], range: 0..<7)
        compute.dispatchThreads(grid, threadsPerThreadgroup: group)
        compute.setComputePipelineState(cloudResolve)
        compute.setTextures([targets.clouds, targets.cloudDistance, targets.history[previous], targets.historyDistance[previous],
                             targets.history[next], targets.historyDistance[next]], range: 0..<6)
        compute.dispatchThreads(grid, threadsPerThreadgroup: group)
        compute.endEncoding()

        // 3. Sun rays from every terrain pixel (after the cloud shadow map, which skips them
        // under thick cloud).
        if let sunShadowPipeline, let terrainAccel, let surface = targets.surface, let sun = targets.sun,
           let rays = shadeCmd.makeComputeCommandEncoder() {
            rays.label = "sun rays"
            rays.setComputePipelineState(sunShadowPipeline)
            rays.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            rays.setAccelerationStructure(terrainAccel, bufferIndex: 1)
            rays.setTextures([surface, targets.depth, horizons, cloudShadows, sun, shadowMeshHeights], range: 0..<6)
            var treeRange: Float = 0
            if let trees = flora?.shadows {
                rays.setAccelerationStructure(trees.accel, bufferIndex: 3)
                rays.useResources(trees.proxies, usage: .read)
                treeRange = flora?.shadowRange ?? 0
            }
            rays.setBytes(&treeRange, length: 4, index: 4)
            // The trees at one pixel per 8×8 block, then per 4×4 where those disagree; the
            // terrain at one per 2×2 block, then the rest where those disagree.
            var pass: UInt32 = 0
            if flora?.shadows != nil {
                for (p, step) in [(UInt32(3), 8), (2, 4)] {
                    pass = p
                    rays.setBytes(&pass, length: 4, index: 2)
                    rays.dispatchThreads(MTLSize(width: (width + step - 1) / step, height: (height + step - 1) / step, depth: 1),
                                         threadsPerThreadgroup: group)
                }
            }
            pass = 0
            rays.setBytes(&pass, length: 4, index: 2)
            rays.dispatchThreads(MTLSize(width: (width + 1) / 2, height: (height + 1) / 2, depth: 1), threadsPerThreadgroup: group)
            pass = 1
            rays.setBytes(&pass, length: 4, index: 2)
            rays.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
            rays.endEncoding()
        }

        // 4. Sky where nothing covers it, then the terrain's nearest surface, both with clouds.
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = TerrainRenderer.skyClearColor
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .dontCare
        guard let shade = shadeCmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        shade.label = "shade"
        shade.setRenderPipelineState(skyPipeline)
        shade.setDepthStencilState(skyDepthState)
        shade.setCullMode(.none)
        shade.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        shade.setFragmentTextures([skyView, cloudMap, targets.history[next], targets.historyDistance[next], cloudShadows],
                                  range: 0..<5)
        shade.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        shade.setRenderPipelineState(pipeline)
        shade.setDepthStencilState(depthState)
        shade.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        bindTerrain(shade, frame: frameTerrain)
        if let sun = targets.sun { shade.setFragmentTexture(sun, index: 5) }
        shade.setFragmentTexture(canopy, index: 6)
        shade.setFragmentTexture(canopyCover, index: 7)
        var floraRange = SIMD2<Float>(flora?.canopyRange ?? 0, flora?.shadows != nil ? flora?.shadowRange ?? 0 : 0)
        shade.setFragmentBytes(&floraRange, length: 8, index: 2)
        drawTerrain(shade, frame: frameTerrain)
        flora?.drawShading(shade, skyVisibility: skyVisibility, horizons: horizons)
        shade.endEncoding()

        historyIndex = next
        historyValid = true
        frameNumber += 1
        previousViewProjection = viewProjection
        previousCamera = u.cameraPosition
    }
}

extension TerrainRenderer {
    /// Crown cover per cell from the community weights and each community's canopy
    /// stems per hectare (Communities.table), taking a crown as ~25 m² and crowns as
    /// randomly placed: cover = 1 − exp(−stems × area).
    static func canopyCoverTexture(device: MTLDevice, maps: FloraMaps?) throws -> MTLTexture {
        let first = maps?.weights.first
        let w = first?.width ?? 1, h = first?.height ?? 1
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { throw RendererError.setup("canopy cover") }
        var cover = [Float](repeating: 0, count: w * h)
        if let maps {
            let crownArea = 25.0
            for (k, entry) in Communities.table.enumerated() where k < maps.weights.count {
                let g = maps.weights[k]
                guard g.width == w, g.height == h, entry.canopy > 0 else { continue }
                let c = Float(1 - exp(-entry.canopy * crownArea / 10_000))
                for i in 0..<(w * h) { cover[i] += g.values[i] * c }
            }
        }
        cover.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: w * 4)
        }
        return tex
    }

    static func canopyTexture(device: MTLDevice, map: ColourMap?) throws -> MTLTexture {
        let w = map?.width ?? 1, h = map?.height ?? 1
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { throw RendererError.setup("canopy map") }
        var texels = [Float](repeating: 0, count: w * h * 4)
        if let map {
            for i in 0..<(w * h) {
                texels[4 * i] = map.rgb[3 * i]
                texels[4 * i + 1] = map.rgb[3 * i + 1]
                texels[4 * i + 2] = map.rgb[3 * i + 2]
                texels[4 * i + 3] = 1
            }
        }
        texels.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: w * 16)
        }
        return tex
    }
}

public enum RendererError: Error {
    case setup(String)
}
#endif
