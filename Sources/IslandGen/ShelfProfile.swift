import Foundation
import IslandCore

/// How the sea floor falls away from the shore: depth percentiles at set distances
/// offshore, sea area in depth bands and how far out the isobaths lie, overall and on the
/// windward and leeward sides. Computed from the heightfield alone, so a generated island
/// and a real topobathy DEM are measured the same way.
public struct ShelfProfile: Sendable {
    public static let distances: [Double] = [100, 250, 500, 1000, 2000, 3000, 5000, 8000]
    public static let bands: [(Double, Double)] = [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 200)]
    public static let isobaths: [Double] = [5, 20, 50, 100]
    /// Sea counted in the depth bands lies within this distance of the shore (km²), so a
    /// neighbouring bank at the edge of a real DEM's tile doesn't count.
    public static let bandReach: Double = 10_000

    public struct Side: Sendable {
        /// 10th, 50th and 90th percentile depth (m, positive down) at each of `distances`.
        public var depth: [[Double]]
        /// Sea area (km²) in each of `bands`, within `bandReach` of the shore.
        public var bandArea: [Double]
        /// Median distance offshore (m) of sea cells within 10% of each isobath's depth.
        public var isobath: [Double]
    }

    public var all: Side
    public var windward: Side
    public var leeward: Side

    /// Exact Euclidean distance (cells) from every cell to the nearest cell where `isTarget`
    /// holds (Felzenszwalb & Huttenlocher 2012).
    public static func distanceTransform(width w: Int, height rows: Int, isTarget: (Int) -> Bool) -> [Float] {
        let big = Float(1e20)
        var f = [Float](repeating: 0, count: w * rows)
        for i in 0..<f.count { f[i] = isTarget(i) ? 0 : big }
        func pass(_ get: (Int) -> Float, _ set: (Int, Float) -> Void, _ n: Int) {
            var v = [Int](repeating: 0, count: n), z = [Float](repeating: 0, count: n + 1)
            var k = 0
            v[0] = 0; z[0] = -big; z[1] = big
            let fq = (0..<n).map(get)
            if n > 1 {
                for q in 1..<n {
                    var s: Float
                    repeat {
                        let p = v[k]
                        s = ((fq[q] + Float(q * q)) - (fq[p] + Float(p * p))) / Float(2 * q - 2 * p)
                        if s <= z[k] { k -= 1 } else { break }
                    } while k >= 0
                    k += 1
                    v[k] = q; z[k] = s; z[k + 1] = big
                }
            }
            k = 0
            for q in 0..<n {
                while z[k + 1] < Float(q) { k += 1 }
                let d = Float(q - v[k])
                set(q, d * d + fq[v[k]])
            }
        }
        f.withUnsafeMutableBufferPointer { buf in
            let b = SharedBuffer(buf)
            Parallel.forRanges(rows) { range in
                for y in range { pass({ b[y * w + $0] }, { b[y * w + $0] = $1 }, w) }
            }
            Parallel.forRanges(w) { range in
                for x in range { pass({ b[$0 * w + x] }, { b[$0 * w + x] = $1 }, rows) }
            }
        }
        return f.map { $0.squareRoot() }
    }

    /// - Parameter windDirection: direction the wind blows towards (radians); a sea cell is
    ///   windward when it lies upwind of the island's land centroid.
    public static func measure(_ h: Grid, seaLevel: Float = 0, windDirection: Double = 5 * Double.pi / 4) -> ShelfProfile {
        let w = h.width, rows = h.height, cell = h.cellSize
        let d = distanceTransform(width: w, height: rows) { h.values[$0] > seaLevel }
        var cx = 0.0, cy = 0.0, landCount = 0.0
        for i in 0..<h.count where h.values[i] > seaLevel {
            cx += Double(i % w); cy += Double(i / w); landCount += 1
        }
        cx /= Swift.max(landCount, 1); cy /= Swift.max(landCount, 1)
        let up = (-cos(windDirection), -sin(windDirection))

        // Per side: depth samples near each distance, band cell counts, isobath distances.
        var near = [[[Float]]](repeating: [[Float]](repeating: [], count: distances.count), count: 3)
        var bandCount = [[Double]](repeating: [Double](repeating: 0, count: bands.count), count: 3)
        var iso = [[[Float]]](repeating: [[Float]](repeating: [], count: isobaths.count), count: 3)
        for i in 0..<h.count where h.values[i] <= seaLevel {
            let metres = Double(d[i]) * cell
            guard metres <= 1.2 * distances.last! else { continue }
            let depth = Double(seaLevel - h.values[i])
            let side = (Double(i % w) - cx) * up.0 + (Double(i / w) - cy) * up.1 > 0 ? 1 : 2
            for s in [0, side] {
                for (k, target) in distances.enumerated() where abs(metres - target) <= Swift.max(0.1 * target, cell) {
                    near[s][k].append(Float(depth))
                }
                if metres <= bandReach {
                    for (k, b) in bands.enumerated() where depth >= b.0 && depth < b.1 { bandCount[s][k] += 1 }
                }
                for (k, z) in isobaths.enumerated() where abs(depth - z) <= 0.1 * z { iso[s][k].append(Float(metres)) }
            }
        }
        func quantile(_ v: [Float], _ q: Double) -> Double {
            guard !v.isEmpty else { return .nan }
            let s = v.sorted()
            return Double(s[Swift.min(s.count - 1, Int(q * Double(s.count)))])
        }
        let km2 = cell * cell / 1e6
        let sides = (0..<3).map { s in
            Side(depth: near[s].map { [quantile($0, 0.1), quantile($0, 0.5), quantile($0, 0.9)] },
                 bandArea: bandCount[s].map { $0 * km2 },
                 isobath: iso[s].map { quantile($0, 0.5) })
        }
        return ShelfProfile(all: sides[0], windward: sides[1], leeward: sides[2])
    }

    /// A plain-text table of the profile, one block per side.
    public func table(_ title: String) -> String {
        func f(_ x: Double) -> String { x.isNaN ? "    -" : String(format: "%5.0f", x) }
        func f1(_ x: Double) -> String { String(format: "%6.1f", x) }
        var out = "== \(title)\n"
        for (name, s) in [("all", all), ("windward", windward), ("leeward", leeward)] {
            out += "  [\(name)] depth p10/p50/p90 (m) at offshore distance:\n"
            for (k, dist) in Self.distances.enumerated() {
                out += String(format: "    %5.0f m:", dist) + " \(f(s.depth[k][0])) \(f(s.depth[k][1])) \(f(s.depth[k][2]))\n"
            }
            out += "    area km² (≤10 km out) " + zip(Self.bands, s.bandArea).map { "\(Int($0.0))–\(Int($0.1)):\(f1($1))" }.joined(separator: " ") + "\n"
            out += "    isobath median distance (m) " + zip(Self.isobaths, s.isobath).map { "\(Int($0)) m:\(f($1))" }.joined(separator: " ") + "\n"
        }
        return out
    }
}
