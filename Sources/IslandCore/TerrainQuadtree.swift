import Foundation

/// CDLOD node selection (Strugar 2009) over the terrain: a quadtree whose level-k nodes
/// carry a grid of `quads`² quads at a spacing of `cellMetres × 2^k`, chosen by distance
/// rings around the camera, with each ring's far edge morphing into the next ring's grid
/// so no vertex pops and no crack opens between levels.
public struct TerrainQuadtree: Sendable {
    /// One node to draw: a full grid of `quads`² quads, or (`half`) a quarter of a node
    /// drawn at its parent's level with `quads/2`² quads.
    public struct Node: Sendable, Equatable {
        public var x: Double, y: Double
        public var size: Double
        public var lod: Int
        public var half: Bool
    }

    public let quads: Int
    public let cellMetres: Double
    /// Quadtree depth: level `lodCount - 1` is the root, covering the whole domain.
    public let lodCount: Int
    /// Radius of each level's ring (metres); the root's is unbounded.
    public let ranges: [Double]
    public let morphStart: [Double]
    public let morphEnd: [Double]
    let domain: Double
    /// Lowest and highest ground per node, padded: from a min/max pyramid of the overview.
    let bounds: [[(lo: Float, hi: Float)]]
    let boundsCell: Double

    /// - Parameters:
    ///   - overview: the whole island at a coarse level, for node height bounds.
    ///   - rangeFactor: ring radius in node sizes (level k reaches `rangeFactor × node size`).
    public init(quads: Int = 64, cellMetres: Double, domain: Double, overview: Grid, rangeFactor: Double = 3) {
        self.quads = quads
        self.cellMetres = cellMetres
        self.domain = domain
        let leaf = Double(quads) * cellMetres
        var count = 1
        while leaf * Double(1 << (count - 1)) < domain { count += 1 }
        lodCount = count
        var r = (0..<count).map { rangeFactor * leaf * Double(1 << $0) }
        r[count - 1] = .greatestFiniteMagnitude
        ranges = r
        morphEnd = r
        morphStart = (0..<count).map { k in
            let previous = k == 0 ? 0 : r[k - 1]
            return k == count - 1 ? .greatestFiniteMagnitude : previous + (r[k] - previous) * 0.7
        }

        // Min/max pyramid of the overview. Fine detail can poke above the coarse mean by a
        // few tens of metres at most: pad the bounds by that.
        let pad: Float = 40
        var level = overview.values.map { (lo: $0 - pad, hi: $0 + pad) }
        var side = overview.width
        var pyramid = [level]
        while side > 1 {
            let next = (side + 1) / 2
            var reduced = [(lo: Float, hi: Float)](repeating: (0, 0), count: next * next)
            for y in 0..<next {
                for x in 0..<next {
                    var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = Swift.min(2 * x + dx, side - 1), sy = Swift.min(2 * y + dy, side - 1)
                        let v = level[sy * side + sx]
                        lo = Swift.min(lo, v.lo)
                        hi = Swift.max(hi, v.hi)
                    }
                    reduced[y * next + x] = (lo, hi)
                }
            }
            level = reduced
            side = next
            pyramid.append(level)
        }
        bounds = pyramid
        boundsCell = overview.cellSize
    }

    /// Ground height bounds under the square at (x, y) of side `size`.
    func heightBounds(x: Double, y: Double, size: Double) -> (lo: Float, hi: Float) {
        // The pyramid level whose cells are at least this big, then the cells it overlaps.
        var p = 0
        while p < bounds.count - 1 && boundsCell * Double(1 << p) < size { p += 1 }
        let cell = boundsCell * Double(1 << p)
        let side = Int(Double(bounds[p].count).squareRoot().rounded())
        let x0 = Swift.max(0, Swift.min(side - 1, Int(x / cell))), x1 = Swift.max(0, Swift.min(side - 1, Int((x + size) / cell)))
        let y0 = Swift.max(0, Swift.min(side - 1, Int(y / cell))), y1 = Swift.max(0, Swift.min(side - 1, Int((y + size) / cell)))
        var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
        for j in y0...y1 {
            for i in x0...x1 {
                let b = bounds[p][j * side + i]
                lo = Swift.min(lo, b.lo)
                hi = Swift.max(hi, b.hi)
            }
        }
        return (lo, hi)
    }

    /// Nodes to draw for a camera at `eye`. `visible` tests a world-space box
    /// (min, max corners) against the view frustum.
    public func select(eye: (x: Double, y: Double, z: Double),
                       visible: @escaping (_ min: SIMD3<Double>, _ max: SIMD3<Double>) -> Bool) -> [Node] {
        var out: [Node] = []
        func box(_ x: Double, _ y: Double, _ size: Double) -> (SIMD3<Double>, SIMD3<Double>) {
            let b = heightBounds(x: x, y: y, size: size)
            return (SIMD3(x, y, Double(b.lo)), SIMD3(x + size, y + size, Double(b.hi)))
        }
        func reaches(_ range: Double, _ b: (SIMD3<Double>, SIMD3<Double>)) -> Bool {
            if range == .greatestFiniteMagnitude { return true }
            let e = SIMD3(eye.x, eye.y, eye.z)
            let nearest = simdClamp(e, b.0, b.1)
            let d = nearest - e
            return (d * d).sum() <= range * range
        }
        // Returns false when the node is beyond its level's ring (the parent draws it).
        func visit(_ x: Double, _ y: Double, _ lod: Int) -> Bool {
            let size = Double(quads) * cellMetres * Double(1 << lod)
            if x >= domain || y >= domain { return true } // nothing there to draw
            let b = box(x, y, size)
            if !reaches(ranges[lod], b) { return false }
            if !visible(b.0, b.1) { return true }
            if lod == 0 || !reaches(ranges[lod - 1], b) {
                out.append(Node(x: x, y: y, size: size, lod: lod, half: false))
                return true
            }
            let h = size / 2
            for (cx, cy) in [(x, y), (x + h, y), (x, y + h), (x + h, y + h)] where !visit(cx, cy, lod - 1) {
                if cx < domain && cy < domain {
                    let cb = box(cx, cy, h)
                    if visible(cb.0, cb.1) { out.append(Node(x: cx, y: cy, size: h, lod: lod, half: true)) }
                }
            }
            return true
        }
        _ = visit(0, 0, lodCount - 1)
        return out
    }
}

@inline(__always)
func simdClamp(_ v: SIMD3<Double>, _ lo: SIMD3<Double>, _ hi: SIMD3<Double>) -> SIMD3<Double> {
    SIMD3(Swift.min(Swift.max(v.x, lo.x), hi.x), Swift.min(Swift.max(v.y, lo.y), hi.y), Swift.min(Swift.max(v.z, lo.z), hi.z))
}
