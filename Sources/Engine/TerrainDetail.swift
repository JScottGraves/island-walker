import Foundation
import IslandCore
#if canImport(Metal)
import Metal
#endif

/// Baked maps that steer the fine detail, each a square grid over the whole domain at its
/// own resolution. Any can be missing (an imported DEM has no drainage or hardness).
public struct DetailMaps: Sendable {
    /// Annual rainfall (m).
    public var rain: Grid?
    /// Upstream drainage area (m²).
    public var drainage: Grid?
    /// Rock hardness (erodibility divisor; ~0.4 soft to ~1.25 dike-hardened).
    public var hardness: Grid?
    /// Ground colour from the native plant communities (`Look.canopyMap`).
    public var canopy: ColourMap?
    /// The communities' weight maps in Look.communities order (all the same size), for the
    /// ground textures; empty when the bundle has none.
    public var communities: [Grid]

    public init(rain: Grid? = nil, drainage: Grid? = nil, hardness: Grid? = nil, canopy: ColourMap? = nil,
                communities: [Grid] = []) {
        self.rain = rain
        self.drainage = drainage
        self.hardness = hardness
        self.canopy = canopy
        self.communities = communities
    }
}

#if canImport(Metal)
/// Fine terrain detail near the camera (plan §4.1): below the baked 10 m grid, a compute
/// pass generates the surface down to ~30 cm into clipmap levels centred on the camera,
/// each half the spacing of the one above it. The detail is noise shaped by the baked
/// maps (DetailShaders), a pure function of position and the island, so it comes out the
/// same every time. A level is regenerated only in the rows and columns that scroll in as
/// the camera moves, which at walking pace is a few hundred samples a frame.
final class TerrainDetail {
    struct Level {
        var origin: SIMD2<Int32>
        var spacing: Float
        var valid: Int32
    }

    struct Params {
        var levels: Int32
        var size: Int32
        var seed: UInt32
        var seaLevel: Float
        var level: (Level, Level, Level, Level, Level, Level, Level, Level)
    }

    struct Update {
        var level: Int32
        var oldValid: Int32
        var oldOrigin: SIMD2<Int32>
    }

    /// Samples held per side of each level. A CDLOD level's vertices reach ~283 of its
    /// sample spacings from the camera (ring radius plus a node diagonal); 384 leaves room
    /// for the camera to wander off-centre while tiles load.
    static let size = 768
    /// Origins move only once the camera has moved this many samples, to batch updates.
    static let hysteresis: Int32 = 4

    let levels: Int
    let cellMetres: Double
    let texture: MTLTexture
    let maps: [MTLTexture]
    let pipeline: MTLComputePipelineState
    private(set) var params: Params

    /// Samples of each level the CPU can read now: those of its last completed update,
    /// less what an update still in flight overwrites. (lo, hi) half-open, in samples.
    var readable: [(lo: SIMD2<Int32>, hi: SIMD2<Int32>)]
    var generation: [Int]
    let lock = NSLock()

    /// Detail levels to put under data spacing `cellMetres`: down to about 30 cm, and only
    /// over grids fine enough (≤ 20 m) for 12 m features to sit under them.
    static func levelCount(cellMetres: Double) -> Int {
        guard cellMetres <= 20 else { return 0 }
        return Swift.max(0, Swift.min(6, Int((log2(cellMetres / 0.3)).rounded(.down))))
    }

    init(device: MTLDevice, library: MTLLibrary, levels: Int, cellMetres: Double, seaLevel: Float,
         seed: UInt32, maps: DetailMaps?) throws {
        precondition(levels <= 8)
        self.levels = levels
        self.cellMetres = cellMetres
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .r32Float
        desc.width = levels > 0 ? Self.size : 1
        desc.height = desc.width
        desc.arrayLength = Swift.max(levels, 1)
        desc.usage = [.shaderRead, .shaderWrite]
        // Shared so the walker can stand on what is drawn.
        desc.storageMode = .shared
        guard let t = device.makeTexture(descriptor: desc) else { throw RendererError.setup("detail clipmap") }
        texture = t
        guard let fn = library.makeFunction(name: "detailUpdate") else { throw RendererError.setup("detail kernel") }
        pipeline = try device.makeComputePipelineState(function: fn)

        func upload(_ grid: Grid?, fallback: Float, transform: (Float) -> Float = { $0 }) throws -> MTLTexture {
            let g = grid ?? Grid(width: 1, height: 1, cellSize: 1, fill: fallback)
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: g.width, height: g.height, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            guard let tex = device.makeTexture(descriptor: d) else { throw RendererError.setup("detail maps") }
            let values = g.values.map(transform)
            values.withUnsafeBytes { raw in
                tex.replace(region: MTLRegionMake2D(0, 0, g.width, g.height), mipmapLevel: 0,
                            withBytes: raw.baseAddress!, bytesPerRow: g.width * 4)
            }
            return tex
        }
        // Without a drainage map, hollows in the 10 m surface still smooth valley floors.
        self.maps = [
            try upload(maps?.rain, fallback: 2),
            try upload(maps?.drainage, fallback: 0, transform: { log10(Swift.max($0, 1)) }),
            try upload(maps?.hardness, fallback: 0.8),
        ]

        let empty = Level(origin: .zero, spacing: 1, valid: 0)
        var p = Params(levels: Int32(levels), size: Int32(Self.size), seed: seed, seaLevel: seaLevel,
                       level: (empty, empty, empty, empty, empty, empty, empty, empty))
        withUnsafeMutableBytes(of: &p.level) { raw in
            let l = raw.bindMemory(to: Level.self)
            for k in 0..<levels { l[k].spacing = Float(cellMetres * pow(2, Double(k - levels))) }
        }
        params = p
        readable = Array(repeating: (.zero, .zero), count: Swift.max(levels, 1))
        generation = Array(repeating: 0, count: Swift.max(levels, 1))
    }

    func level(_ k: Int) -> Level {
        withUnsafeBytes(of: params.level) { $0.bindMemory(to: Level.self)[k] }
    }

    func setLevel(_ k: Int, _ value: Level) {
        withUnsafeMutableBytes(of: &params.level) { $0.bindMemory(to: Level.self)[k] = value }
    }

    func spacing(_ k: Int) -> Double { cellMetres * pow(2, Double(k - levels)) }

    /// Where level k's origin should be for a camera at `eye`.
    func wantedOrigin(_ k: Int, eye: SIMD3<Double>) -> SIMD2<Int32> {
        let s = spacing(k)
        let half = Int32(Self.size / 2)
        return SIMD2(Int32((eye.x / s).rounded(.down)) - half, Int32((eye.y / s).rounded(.down)) - half)
    }

    /// Level-0 data tiles a level needs to fill the region at `origin` (with the margin
    /// the Catmull-Rom base reads).
    func tileKeys(_ k: Int, origin: SIMD2<Int32>, streamer: TerrainStreamer) -> [TerrainStreamer.Key] {
        let s = spacing(k)
        return streamer.keys(level: 0, x: Double(origin.x) * s, y: Double(origin.y) * s, size: Double(Self.size - 1) * s)
    }

    /// Data tiles every level needs for the camera at `eye`, to request with the frame's.
    func neededTiles(eye: SIMD3<Double>, streamer: TerrainStreamer) -> [TerrainStreamer.Key] {
        // The coarsest level covers the others.
        guard levels > 0 else { return [] }
        return tileKeys(levels - 1, origin: wantedOrigin(levels - 1, eye: eye), streamer: streamer)
    }

    /// Moves the levels the camera has left behind and encodes their new samples into
    /// `cmd`. A level waits (keeping its old region, which the shaders fall back from) until
    /// the data tiles under its new region are resident.
    func update(_ cmd: MTLCommandBuffer, eye: SIMD3<Double>, streamer: TerrainStreamer,
                terrainParams: TerrainRenderer.TerrainParams, pages: MTLBuffer) {
        guard levels > 0 else { return }
        var updates: [Update] = []
        for k in 0..<levels {
            let current = level(k)
            let wanted = wantedOrigin(k, eye: eye)
            let d = wanted &- current.origin
            if current.valid != 0 && abs(d.x) < Self.hysteresis && abs(d.y) < Self.hysteresis { continue }
            guard tileKeys(k, origin: wanted, streamer: streamer).allSatisfy({ streamer.slotOf[$0] != nil }) else { continue }
            updates.append(Update(level: Int32(k), oldValid: current.valid, oldOrigin: current.origin))
            setLevel(k, Level(origin: wanted, spacing: current.spacing, valid: 1))
        }
        guard !updates.isEmpty, let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "detail"
        enc.setComputePipelineState(pipeline)
        var t = terrainParams
        var p = params
        enc.setBytes(&t, length: MemoryLayout<TerrainRenderer.TerrainParams>.stride, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        enc.setBuffer(pages, offset: 0, index: 3)
        enc.setTexture(streamer.tiles, index: 0)
        enc.setTexture(texture, index: 1)
        enc.setTextures(maps, range: 2..<5)
        let n = Self.size
        var stamps: [(Int, Int, SIMD2<Int32>)] = []
        lock.lock()
        for var u in updates {
            enc.setBytes(&u, length: MemoryLayout<Update>.stride, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            let k = Int(u.level)
            let origin = level(k).origin
            // Until this completes, only samples both regions share are certain.
            let r = readable[k]
            let lo = pointwiseMax(r.lo, origin), hi = pointwiseMin(r.hi, origin &+ Int32(n))
            readable[k] = (lo, hi)
            generation[k] += 1
            stamps.append((k, generation[k], origin))
        }
        lock.unlock()
        enc.endEncoding()
        cmd.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            for (k, g, origin) in stamps where self.generation[k] == g {
                self.readable[k] = (origin, origin &+ Int32(n))
            }
            self.lock.unlock()
        }
    }

    /// Height of the finest generated surface at (x, y), bilinear like the vertex shader,
    /// or nil where no level holds it yet.
    func height(x: Double, y: Double) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        let n = Self.size
        for k in 0..<levels {
            let s = spacing(k)
            let sx = x / s, sy = y / s
            let i = SIMD2(Int32(sx.rounded(.down)), Int32(sy.rounded(.down)))
            let r = readable[k]
            guard all(i .>= r.lo) && all(i &+ 1 .< r.hi) else { continue }
            func sample(_ dx: Int32, _ dy: Int32) -> Float {
                let jx = ((Int(i.x + dx) % n) + n) % n, jy = ((Int(i.y + dy) % n) + n) % n
                var v: Float = 0
                texture.getBytes(&v, bytesPerRow: 4, bytesPerImage: 4, from: MTLRegionMake2D(jx, jy, 1, 1),
                                 mipmapLevel: 0, slice: k)
                return v
            }
            let fx = Float(sx - Double(i.x)), fy = Float(sy - Double(i.y))
            let h00 = sample(0, 0), h10 = sample(1, 0), h01 = sample(0, 1), h11 = sample(1, 1)
            let bottom = h00 + (h10 - h00) * fx, top = h01 + (h11 - h01) * fx
            return Double(bottom + (top - bottom) * fy)
        }
        return nil
    }

    /// A seed for the detail noise that belongs to the island: a hash of its overview
    /// heights, so the same bundle always gets the same rocks.
    static func seed(of overview: Grid) -> UInt32 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in overview.values {
            h ^= UInt64(v.bitPattern)
            h = h &* 0x100_0000_01b3
        }
        return UInt32(truncatingIfNeeded: Hash.mix(h))
    }
}
#endif
