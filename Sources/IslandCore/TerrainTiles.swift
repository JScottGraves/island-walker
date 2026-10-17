import Foundation

/// Heights for the streaming renderer and the walker: square tiles of quantised samples in
/// a mip pyramid, laid out like the island bundle (value = offset + scale × sample, row 0
/// south, level 0 full resolution, each level above a 2 × 2 mean of the one below).
public protocol TerrainTiles: AnyObject, Sendable {
    /// Samples per side at level 0.
    var size: Int { get }
    /// Level-0 sample spacing in metres; sample k sits at the centre of cell k.
    var cellMetres: Double { get }
    var levels: Int { get }
    var tileSize: Int { get }
    var seaLevel: Float { get }
    var offset: Float { get }
    var scale: Float { get }
    /// One tile, `tileSize`² samples row-major; edge tiles repeat the last row and column.
    func rawTile(level: Int, tx: Int, ty: Int) throws -> [UInt16]
}

extension TerrainTiles {
    public var domainMetres: Double { Double(size) * cellMetres }

    public func levelSize(_ level: Int) -> Int { IslandBundle.levelSize(size, level: level) }

    public func tilesPerSide(_ level: Int) -> Int { (levelSize(level) + tileSize - 1) / tileSize }

    /// A whole level assembled into one grid of heights.
    public func grid(level: Int) throws -> Grid {
        let side = levelSize(level), t = tileSize, tiles = tilesPerSide(level)
        var g = Grid(width: side, height: side, cellSize: cellMetres * Double(1 << level))
        for ty in 0..<tiles {
            for tx in 0..<tiles {
                let raw = try rawTile(level: level, tx: tx, ty: ty)
                for y in 0..<t where ty * t + y < side {
                    for x in 0..<t where tx * t + x < side {
                        g[tx * t + x, ty * t + y] = offset + scale * Float(raw[y * t + x])
                    }
                }
            }
        }
        return g
    }

    /// The finest level at most `maxSide` samples across: small enough to keep whole for
    /// lighting bakes, viewpoints and culling bounds.
    public func overviewLevel(maxSide: Int = 1024) -> Int {
        var level = 0
        while level < levels - 1 && levelSize(level) > maxSide { level += 1 }
        return level
    }
}

/// The height channel of an island bundle, read tile by tile from disk.
public final class BundleTerrain: TerrainTiles {
    public let bundle: IslandBundle
    let channel: IslandBundle.Channel
    public var size: Int { channel.size }
    public var cellMetres: Double { channel.cellMetres }
    public var levels: Int { channel.levels }
    public var tileSize: Int { bundle.manifest.tileSize }
    public var seaLevel: Float { Float(bundle.manifest.seaLevel) }
    public var offset: Float { Float(channel.offset) }
    public var scale: Float { Float(channel.scale) }

    public init(bundle: IslandBundle) throws {
        self.bundle = bundle
        channel = try bundle.channel("height")
    }

    public func rawTile(level: Int, tx: Int, ty: Int) throws -> [UInt16] {
        try bundle.rawTile("height", level: level, tx: tx, ty: ty)
    }
}

/// An in-memory heightfield (a height.r32 or a quick bake) served as bundle-style tiles.
public final class GridTerrain: TerrainTiles {
    let pyramid: [Grid]
    public let tileSize: Int
    public let seaLevel: Float
    public let offset: Float
    public let scale: Float
    public var size: Int { pyramid[0].width }
    public var cellMetres: Double { pyramid[0].cellSize }
    public var levels: Int { pyramid.count }

    public init(height: Grid, seaLevel: Float, tileSize: Int = 256) {
        precondition(height.width == height.height, "terrain grids are square")
        var levels = [height]
        while levels[levels.count - 1].width > tileSize { levels.append(IslandBundle.downsample(levels[levels.count - 1])) }
        pyramid = levels
        self.tileSize = tileSize
        self.seaLevel = seaLevel
        let (lo, hi) = height.range
        offset = lo
        scale = hi > lo ? (hi - lo) / 65535 : 1
    }

    public func rawTile(level: Int, tx: Int, ty: Int) throws -> [UInt16] {
        let g = pyramid[level], t = tileSize
        var out = [UInt16](repeating: 0, count: t * t)
        for y in 0..<t {
            let gy = Swift.min(ty * t + y, g.height - 1)
            for x in 0..<t {
                let gx = Swift.min(tx * t + x, g.width - 1)
                let q = ((g[gx, gy] - offset) / scale).rounded()
                out[y * t + x] = UInt16(Swift.min(Swift.max(q, 0), 65535))
            }
        }
        return out
    }
}

/// CPU height queries against level 0 (the walker's ground), with a small cache of decoded
/// tiles. Matches what the renderer draws up close.
public final class TerrainHeights: @unchecked Sendable {
    public let tiles: TerrainTiles
    var cache: [Int: (values: [Float], used: Int)] = [:]
    var clock = 0
    let capacity: Int
    let lock = NSLock()

    public init(tiles: TerrainTiles, capacity: Int = 64) {
        self.tiles = tiles
        self.capacity = capacity
    }

    func sample(_ x: Int, _ y: Int) -> Float {
        let side = tiles.levelSize(0), t = tiles.tileSize
        let cx = Swift.min(Swift.max(x, 0), side - 1), cy = Swift.min(Swift.max(y, 0), side - 1)
        let tx = cx / t, ty = cy / t
        let key = ty * tiles.tilesPerSide(0) + tx
        clock += 1
        if cache[key] == nil {
            if cache.count >= capacity, let oldest = cache.min(by: { $0.value.used < $1.value.used })?.key {
                cache[oldest] = nil
            }
            let raw = (try? tiles.rawTile(level: 0, tx: tx, ty: ty)) ?? [UInt16](repeating: 0, count: t * t)
            let offset = tiles.offset, scale = tiles.scale
            cache[key] = (raw.map { offset + scale * Float($0) }, clock)
        }
        cache[key]!.used = clock
        return cache[key]!.values[(cy - ty * t) * t + (cx - tx * t)]
    }

    /// Height in metres at world position (x, y), bilinear between level-0 samples.
    public func height(x: Double, y: Double) -> Double {
        lock.lock()
        defer { lock.unlock() }
        let sx = x / tiles.cellMetres - 0.5, sy = y / tiles.cellMetres - 0.5
        let x0 = Int(sx.rounded(.down)), y0 = Int(sy.rounded(.down))
        let fx = Float(sx - Double(x0)), fy = Float(sy - Double(y0))
        let h00 = sample(x0, y0), h10 = sample(x0 + 1, y0), h01 = sample(x0, y0 + 1), h11 = sample(x0 + 1, y0 + 1)
        let bottom = h00 + (h10 - h00) * fx, top = h01 + (h11 - h01) * fx
        return Double(bottom + (top - bottom) * fy)
    }
}
