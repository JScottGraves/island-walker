import Foundation

/// Signed distance to the shoreline over a heightfield, for the surf: positive out to sea,
/// negative inland, in metres. Exact Euclidean distances between cell centres (Felzenszwalb &
/// Huttenlocher 2012, separable), less half a cell so the shoreline itself, between a land and
/// a sea cell, sits at zero.
public enum ShoreDistance {
    public static func signed(height: Grid, seaLevel: Float) -> Grid {
        let w = height.width, h = height.height
        var sea = [Bool](repeating: false, count: w * h)
        for k in 0..<(w * h) { sea[k] = height.values[k] < seaLevel }
        let toLand = distances(w, h, sources: sea.map { !$0 })
        let toSea = distances(w, h, sources: sea)
        var out = Grid(width: w, height: h, cellSize: height.cellSize)
        let half = 0.5 * height.cellSize
        for k in 0..<(w * h) {
            let d = sea[k] ? toLand[k] : toSea[k]
            let metres = d.isFinite ? d.squareRoot() * height.cellSize - half : 1e6
            out.values[k] = Float(sea[k] ? metres : -metres)
        }
        return out
    }

    /// `grid` blurred by `passes` 3×3 box filters (edges clamped): about a Gaussian of
    /// √(2·passes/3) cells, which keeps linear ramps and rounds the corners of a distance field.
    public static func smoothed(_ grid: Grid, passes: Int) -> Grid {
        var g = grid
        let w = g.width, h = g.height
        for _ in 0..<passes {
            for horizontal in [true, false] {
                let src = g.values
                for j in 0..<h {
                    for i in 0..<w {
                        var sum: Float = 0
                        for d in -1...1 {
                            let x = horizontal ? Swift.min(Swift.max(i + d, 0), w - 1) : i
                            let y = horizontal ? j : Swift.min(Swift.max(j + d, 0), h - 1)
                            sum += src[y * w + x]
                        }
                        g.values[j * w + i] = sum / 3
                    }
                }
            }
        }
        return g
    }

    /// How much of a wave system coming `from` that direction (a unit vector) reaches each
    /// cell, 0…1: the water upstream at 120, 350 and 900 m. Land there stops the waves; a reef
    /// or bar shallower than ~3 m has broken them already. Off the grid counts as open sea.
    public static func exposure(height: Grid, seaLevel: Float, from: SIMD2<Double>) -> Grid {
        let w = height.width, h = height.height, cell = height.cellSize
        var out = Grid(width: w, height: h, cellSize: cell)
        let reach: [Double] = [120, 350, 900]
        for j in 0..<h {
            for i in 0..<w {
                var e = 1.0
                for r in reach {
                    let x = (Double(i) + 0.5) * cell + from.x * r, y = (Double(j) + 0.5) * cell + from.y * r
                    let ci = Int((x / cell).rounded(.down)), cj = Int((y / cell).rounded(.down))
                    if ci < 0 || cj < 0 || ci >= w || cj >= h { break }
                    let depth = Double(seaLevel - height[ci, cj])
                    e = Swift.min(e, Swift.max(0, Swift.min(1, (depth - 0.3) / 3)))
                }
                out[i, j] = Float(e)
            }
        }
        return out
    }

    /// Squared distance (in cells) from every cell to the nearest source cell.
    static func distances(_ w: Int, _ h: Int, sources: [Bool]) -> [Double] {
        var f = sources.map { $0 ? 0.0 : Double.infinity }
        var line = [Double](repeating: 0, count: Swift.max(w, h))
        for x in 0..<w {
            for y in 0..<h { line[y] = f[y * w + x] }
            let d = transform(Array(line[0..<h]))
            for y in 0..<h { f[y * w + x] = d[y] }
        }
        for y in 0..<h {
            let d = transform(Array(f[(y * w)..<(y * w + w)]))
            for x in 0..<w { f[y * w + x] = d[x] }
        }
        return f
    }

    /// One dimension of the squared-distance transform: the lower envelope of parabolas.
    static func transform(_ f: [Double]) -> [Double] {
        let n = f.count
        var d = [Double](repeating: .infinity, count: n)
        var v = [Int](repeating: 0, count: n)
        var z = [Double](repeating: 0, count: n + 1)
        var k = -1
        for q in 0..<n where f[q].isFinite {
            if k < 0 {
                k = 0; v[0] = q; z[0] = -.infinity; z[1] = .infinity
                continue
            }
            var s = 0.0
            while true {
                let p = v[k]
                s = ((f[q] + Double(q * q)) - (f[p] + Double(p * p))) / Double(2 * q - 2 * p)
                if s <= z[k] && k > 0 { k -= 1 } else { break }
            }
            if s <= z[k] {
                // k == 0 and the new parabola wins everywhere.
                v[0] = q; z[0] = -.infinity; z[1] = .infinity
            } else {
                k += 1; v[k] = q; z[k] = s; z[k + 1] = .infinity
            }
        }
        if k < 0 { return d }
        var j = 0
        for q in 0..<n {
            while z[j + 1] < Double(q) { j += 1 }
            let p = v[j]
            d[q] = Double((q - p) * (q - p)) + f[p]
        }
        return d
    }
}
