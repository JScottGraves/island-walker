#if canImport(Metal)
import Foundation
import Metal
import IslandCore

/// Keeps the height tiles the camera needs resident on the GPU: a texture array of tile
/// slots, filled from disk on a background queue, with a page table per mip level that
/// the vertex shader reads (−1 for a tile that isn't resident, so it falls back to the
/// next level up). The coarsest level is loaded up front and never evicted, so there is
/// always something to draw.
final class TerrainStreamer {
    struct Key: Hashable { var level: Int, tx: Int, ty: Int }

    let source: TerrainTiles
    let tiles: MTLTexture
    let capacity: Int
    /// Where each level's entries start in the page table.
    let pageOffsets: [Int]
    var pageTable: [Int32]
    /// One page-table buffer per frame in flight.
    let pageBuffers: [MTLBuffer]
    var bufferIndex = 0

    var slotOf: [Key: Int] = [:]
    var keyOf: [Key?]
    var lastUsed: [Int]
    var pinned: [Bool]
    var frame = 0

    let queue = DispatchQueue(label: "terrain-tiles", qos: .userInitiated, attributes: .concurrent)
    let lock = NSLock()
    var pending = Set<Key>()
    var loaded: [(Key, [UInt16])] = []
    /// Tiles waiting for a free slot or a background load: how much is still streaming.
    var backlog: Int { lock.lock(); defer { lock.unlock() }; return pending.count + loaded.count }
    var residentCount: Int { slotOf.count }

    init(device: MTLDevice, source: TerrainTiles, capacity: Int = 512) throws {
        self.source = source
        self.capacity = capacity
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .r16Uint
        desc.width = source.tileSize
        desc.height = source.tileSize
        desc.arrayLength = capacity
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let t = device.makeTexture(descriptor: desc) else { throw RendererError.setup("terrain tile cache") }
        tiles = t
        var offsets: [Int] = []
        var total = 0
        for level in 0..<source.levels {
            offsets.append(total)
            total += source.tilesPerSide(level) * source.tilesPerSide(level)
        }
        pageOffsets = offsets
        pageTable = [Int32](repeating: -1, count: total)
        pageBuffers = try (0..<3).map { _ in
            guard let b = device.makeBuffer(length: Swift.max(total, 1) * 4, options: .storageModeShared) else {
                throw RendererError.setup("terrain page table")
            }
            return b
        }
        keyOf = [Key?](repeating: nil, count: capacity)
        lastUsed = [Int](repeating: -1, count: capacity)
        pinned = [Bool](repeating: false, count: capacity)

        // The coarsest level, always resident.
        let top = source.levels - 1
        for ty in 0..<source.tilesPerSide(top) {
            for tx in 0..<source.tilesPerSide(top) {
                let key = Key(level: top, tx: tx, ty: ty)
                let raw = try source.rawTile(level: top, tx: tx, ty: ty)
                if let slot = install(key, raw) { pinned[slot] = true }
            }
        }
    }

    /// Tiles of `level` covering the square at (x, y) of side `size` metres, with a sample
    /// of margin for bilinear filtering and normals.
    func keys(level: Int, x: Double, y: Double, size: Double) -> [Key] {
        let cell = source.cellMetres * Double(1 << level)
        let t = Double(source.tileSize), n = source.tilesPerSide(level)
        let tx0 = Swift.max(0, Int(((x / cell) - 2) / t)), tx1 = Swift.min(n - 1, Int(((x + size) / cell + 2) / t))
        let ty0 = Swift.max(0, Int(((y / cell) - 2) / t)), ty1 = Swift.min(n - 1, Int(((y + size) / cell + 2) / t))
        guard tx0 <= tx1, ty0 <= ty1 else { return [] }
        var out: [Key] = []
        for ty in ty0...ty1 { for tx in tx0...tx1 { out.append(Key(level: level, tx: tx, ty: ty)) } }
        return out
    }

    /// Marks tiles as needed this frame and starts loading the missing ones.
    func request(_ keys: [Key], synchronously: Bool = false) {
        var missing: [Key] = []
        for key in keys {
            if let slot = slotOf[key] { lastUsed[slot] = frame } else { missing.append(key) }
        }
        if synchronously {
            for key in Set(missing) {
                if let raw = try? source.rawTile(level: key.level, tx: key.tx, ty: key.ty), let slot = install(key, raw) {
                    lastUsed[slot] = frame
                }
            }
            return
        }
        lock.lock()
        let fresh = missing.filter { pending.insert($0).inserted }
        lock.unlock()
        for key in fresh {
            queue.async { [source, weak self] in
                let raw = try? source.rawTile(level: key.level, tx: key.tx, ty: key.ty)
                guard let self else { return }
                self.lock.lock()
                if let raw { self.loaded.append((key, raw)) } else { self.pending.remove(key) }
                self.lock.unlock()
            }
        }
    }

    /// Uploads finished tiles and returns the page table for this frame.
    func update(maxUploads: Int = 64) -> MTLBuffer {
        lock.lock()
        let batch = Array(loaded.prefix(maxUploads))
        loaded.removeFirst(batch.count)
        for (key, _) in batch { pending.remove(key) }
        lock.unlock()
        for (key, raw) in batch where slotOf[key] == nil {
            if let slot = install(key, raw) { lastUsed[slot] = frame }
        }
        frame += 1
        bufferIndex = (bufferIndex + 1) % pageBuffers.count
        let buffer = pageBuffers[bufferIndex]
        pageTable.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return buffer
    }

    /// Puts a tile in a free slot, or in the least recently used one that no frame in
    /// flight can still be reading.
    @discardableResult
    func install(_ key: Key, _ raw: [UInt16]) -> Int? {
        var slot = keyOf.firstIndex(where: { $0 == nil })
        if slot == nil {
            var best = -1, oldest = Int.max
            for s in 0..<capacity where !pinned[s] && lastUsed[s] < frame - 3 && lastUsed[s] < oldest {
                best = s
                oldest = lastUsed[s]
            }
            guard best >= 0 else { return nil }
            slot = best
        }
        guard let s = slot else { return nil }
        if let old = keyOf[s] {
            slotOf[old] = nil
            pageTable[entry(old)] = -1
        }
        let t = source.tileSize
        raw.withUnsafeBytes { bytes in
            tiles.replace(region: MTLRegionMake2D(0, 0, t, t), mipmapLevel: 0, slice: s,
                          withBytes: bytes.baseAddress!, bytesPerRow: t * 2, bytesPerImage: t * t * 2)
        }
        keyOf[s] = key
        slotOf[key] = s
        lastUsed[s] = frame
        pageTable[entry(key)] = Int32(s)
        return s
    }

    func entry(_ key: Key) -> Int {
        pageOffsets[key.level] + key.ty * source.tilesPerSide(key.level) + key.tx
    }
}
#endif
