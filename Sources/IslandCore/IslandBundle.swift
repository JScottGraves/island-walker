import Foundation

/// The island bundle (plan §2): the baked maps of one island on disk, split into square
/// tiles with a mip pyramid, so the runtime can stream just the tiles near the walker.
///
/// Layout of `<name>.bundle/`:
/// - `manifest.json`: format version, tile size, and per channel its size, cell size,
///   mip level count and quantisation.
/// - `<channel>/<level>/<ty>_<tx>.lz4`: one tile, `tileSize`² little-endian UInt16 samples
///   (row 0 south, like `Grid`), LZ4-compressed. Edge tiles are padded by repeating the
///   last row and column.
///
/// Samples are quantised as `offset + scale × value`, scale chosen from the channel's
/// range; for heights from −2 km to +2 km that is a 6 cm step. Level 0 is full
/// resolution and each level above halves it (2 × 2 mean), down to one tile.
public struct IslandBundle: Sendable {
    public static let formatVersion = 1
    public static let manifestName = "manifest.json"

    public struct Channel: Codable, Sendable, Equatable {
        public var name: String
        public var size: Int
        public var cellMetres: Double
        public var levels: Int
        /// value = offset + scale × sample
        public var offset: Double
        public var scale: Double
    }

    public struct Manifest: Codable, Sendable, Equatable {
        public var format: Int
        public var tileSize: Int
        public var seaLevel: Double
        public var channels: [Channel]
    }

    public let url: URL
    public let manifest: Manifest

    public enum Failure: Error, CustomStringConvertible {
        case unsupportedFormat(Int)
        case missingChannel(String)
        case badTile(String)
        case notSquare(String)
        public var description: String {
            switch self {
            case .unsupportedFormat(let v): return "unsupported bundle format \(v)"
            case .missingChannel(let n): return "bundle has no channel \"\(n)\""
            case .badTile(let p): return "corrupt or truncated tile \(p)"
            case .notSquare(let n): return "channel \"\(n)\" is not a square grid"
            }
        }
    }

    public init(url: URL) throws {
        self.url = url
        let data = try Data(contentsOf: url.appendingPathComponent(Self.manifestName))
        manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.format == Self.formatVersion else { throw Failure.unsupportedFormat(manifest.format) }
    }

    public func channel(_ name: String) throws -> Channel {
        guard let c = manifest.channels.first(where: { $0.name == name }) else { throw Failure.missingChannel(name) }
        return c
    }

    /// Side length in samples of `channel` at mip `level`.
    public static func levelSize(_ size: Int, level: Int) -> Int { Swift.max(1, size >> level) }

    /// Tiles per side of `channel` at mip `level`.
    public func tilesPerSide(_ channel: Channel, level: Int) -> Int {
        (Self.levelSize(channel.size, level: level) + manifest.tileSize - 1) / manifest.tileSize
    }

    static func tilePath(_ channel: String, level: Int, tx: Int, ty: Int) -> String {
        "\(channel)/\(level)/\(ty)_\(tx).lz4"
    }

    /// One tile's values, `tileSize`² row-major, row 0 south.
    public func tile(_ name: String, level: Int, tx: Int, ty: Int) throws -> [Float] {
        let c = try channel(name)
        let path = Self.tilePath(name, level: level, tx: tx, ty: ty)
        let packed = try Data(contentsOf: url.appendingPathComponent(path))
        let raw: Data
        do { raw = try (packed as NSData).decompressed(using: .lz4) as Data } catch { throw Failure.badTile(path) }
        let n = manifest.tileSize * manifest.tileSize
        guard raw.count == n * 2 else { throw Failure.badTile(path) }
        let offset = Float(c.offset), scale = Float(c.scale)
        return raw.withUnsafeBytes { bytes in
            (0..<n).map { i in offset + scale * Float(UInt16(littleEndian: bytes.load(fromByteOffset: i * 2, as: UInt16.self))) }
        }
    }

    /// A whole mip level of a channel assembled into one grid.
    public func grid(_ name: String, level: Int = 0) throws -> Grid {
        let c = try channel(name)
        let side = Self.levelSize(c.size, level: level)
        let t = manifest.tileSize
        var g = Grid(width: side, height: side, cellSize: c.cellMetres * Double(1 << level))
        let tiles = tilesPerSide(c, level: level)
        for ty in 0..<tiles {
            for tx in 0..<tiles {
                let values = try tile(name, level: level, tx: tx, ty: ty)
                for y in 0..<t where ty * t + y < side {
                    for x in 0..<t where tx * t + x < side {
                        g[tx * t + x, ty * t + y] = values[y * t + x]
                    }
                }
            }
        }
        return g
    }

    // MARK: Writing

    /// Writes `channels` (each a square grid; sizes may differ) as a new bundle at `url`,
    /// replacing any bundle already there.
    @discardableResult
    public static func write(_ channels: [(name: String, grid: Grid)], seaLevel: Double, to url: URL,
                             tileSize: Int = 256) throws -> IslandBundle {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        var entries: [Channel] = []
        for (name, grid) in channels {
            guard grid.width == grid.height else { throw Failure.notSquare(name) }
            let (lo, hi) = grid.range
            let span = Double(hi - lo)
            let scale = span > 0 ? span / 65535 : 1
            var entry = Channel(name: name, size: grid.width, cellMetres: grid.cellSize, levels: 0,
                                offset: Double(lo), scale: scale)
            var level = grid
            var index = 0
            while true {
                try writeLevel(level, channel: entry, level: index, tileSize: tileSize, to: url)
                index += 1
                if level.width <= tileSize { break }
                level = downsample(level)
            }
            entry.levels = index
            entries.append(entry)
        }
        let manifest = Manifest(format: formatVersion, tileSize: tileSize, seaLevel: seaLevel, channels: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: url.appendingPathComponent(manifestName))
        return try IslandBundle(url: url)
    }

    static func writeLevel(_ g: Grid, channel: Channel, level: Int, tileSize t: Int, to url: URL) throws {
        let tiles = (g.width + t - 1) / t
        try FileManager.default.createDirectory(at: url.appendingPathComponent("\(channel.name)/\(level)"),
                                                withIntermediateDirectories: true)
        let offset = channel.offset, inverse = 1 / channel.scale
        var failure: Error?
        let lock = NSLock()
        Parallel.forRanges(tiles * tiles, chunks: tiles * tiles) { range in
            for k in range {
                let tx = k % tiles, ty = k / tiles
                var raw = Data(count: t * t * 2)
                raw.withUnsafeMutableBytes { bytes in
                    for y in 0..<t {
                        let gy = Swift.min(ty * t + y, g.height - 1)
                        for x in 0..<t {
                            let gx = Swift.min(tx * t + x, g.width - 1)
                            let q = ((Double(g[gx, gy]) - offset) * inverse).rounded()
                            let sample = UInt16(Swift.min(Swift.max(q, 0), 65535))
                            bytes.storeBytes(of: sample.littleEndian, toByteOffset: (y * t + x) * 2, as: UInt16.self)
                        }
                    }
                }
                do {
                    let packed = try (raw as NSData).compressed(using: .lz4) as Data
                    try packed.write(to: url.appendingPathComponent(tilePath(channel.name, level: level, tx: tx, ty: ty)))
                } catch {
                    lock.lock(); failure = failure ?? error; lock.unlock()
                }
            }
        }
        if let failure { throw failure }
    }

    /// Halves a grid by 2 × 2 means (an odd last row or column is averaged with itself).
    public static func downsample(_ g: Grid) -> Grid {
        let w = (g.width + 1) / 2, h = (g.height + 1) / 2
        var out = Grid(width: w, height: h, cellSize: g.cellSize * 2)
        for y in 0..<h {
            let y0 = 2 * y, y1 = Swift.min(2 * y + 1, g.height - 1)
            for x in 0..<w {
                let x0 = 2 * x, x1 = Swift.min(2 * x + 1, g.width - 1)
                out[x, y] = 0.25 * (g[x0, y0] + g[x1, y0] + g[x0, y1] + g[x1, y1])
            }
        }
        return out
    }
}
