import Foundation
import IslandCore

/// Terrain statistics that can be compared between a generated island and a real DEM
/// (plan §5, acceptance criteria): hypsometry, slope distribution, drainage density,
/// Hack's law and pits. Everything is computed from the heightfield alone, with plain
/// (not rain-weighted) drainage area, so a real DEM is scored the same way.
public struct Realism: Sendable {
    /// Fraction of land area at or above each tenth of the relief (11 values, 1 → 0).
    public var hypsometry: [Double]
    /// Mean normalised elevation of land (the hypsometric integral).
    public var hypsometricIntegral: Double
    /// Fraction of land in each 5° slope band, 0–5° … 85–90° (18 values).
    public var slopeHistogram: [Double]
    /// Fraction of land steeper than 40°, the cliff tail.
    public var cliffFraction: Double
    /// Channel length per land area (km / km²), channels being cells draining ≥ `channelArea`.
    public var drainageDensity: Double
    /// Exponent h in L ∝ A^h (main stream length vs basin area), fitted over 1–100 km².
    public var hackExponent: Double
    /// Land cells with no lower neighbour that are not next to the sea (should be 0).
    public var pits: Int
    public var landAreaKm2: Double
    public var reliefMetres: Double

    public static let channelArea = 1e6 // m²

    public static func measure(_ h: Grid, seaLevel: Float = 0) -> Realism {
        let w = h.width, rows = h.height, n = h.count
        let cell = h.cellSize
        let sea = seaLevel
        var land = [Int]()
        land.reserveCapacity(n)
        var top: Float = sea
        for i in 0..<n where h.values[i] > sea {
            land.append(i)
            top = Swift.max(top, h.values[i])
        }
        let relief = Double(top - sea)
        let landCount = Swift.max(land.count, 1)

        // Hypsometry.
        var hyps = [Double](repeating: 0, count: 11)
        var integral = 0.0
        for i in land {
            let e = relief > 0 ? Double(h.values[i] - sea) / relief : 0
            integral += e
            for k in 0...10 where e >= Double(k) / 10 { hyps[k] += 1 }
        }
        hyps = hyps.map { $0 / Double(landCount) }
        integral /= Double(landCount)

        // Slopes (central differences) and steepest-descent receivers.
        var histogram = [Double](repeating: 0, count: 18)
        var cliffs = 0
        var receiver = [Int](0..<n)
        var dist = [Double](repeating: 0, count: n)
        var pits = 0
        for i in land {
            let x = i % w, y = i / w
            let xl = Swift.max(x - 1, 0), xr = Swift.min(x + 1, w - 1)
            let yd = Swift.max(y - 1, 0), yu = Swift.min(y + 1, rows - 1)
            let gx = Double(h[xr, y] - h[xl, y]) / (Double(xr - xl) * cell)
            let gy = Double(h[x, yu] - h[x, yd]) / (Double(yu - yd) * cell)
            let degrees = atan((gx * gx + gy * gy).squareRoot()) * 180 / .pi
            histogram[Swift.min(17, Int(degrees / 5))] += 1
            if degrees > 40 { cliffs += 1 }

            var best = 0.0, bestK = -1
            var touchesSea = false
            for k in 0..<8 {
                let nx = x + D8.dx[k], ny = y + D8.dy[k]
                guard h.contains(nx, ny) else { touchesSea = true; continue }
                let j = ny * w + nx
                if h.values[j] <= sea { touchesSea = true }
                let drop = Double(h.values[i] - h.values[j]) / (D8.distance[k] * cell)
                if drop > best { best = drop; bestK = k }
            }
            if bestK >= 0 {
                receiver[i] = (y + D8.dy[bestK]) * w + (x + D8.dx[bestK])
                dist[i] = D8.distance[bestK] * cell
            } else if !touchesSea {
                pits += 1
            }
        }
        histogram = histogram.map { $0 / Double(landCount) }

        // Drainage area and main stream length, accumulated from the highest cell down:
        // a receiver is always lower, so every donor is finished before its receiver.
        let order = land.sorted { h.values[$0] > h.values[$1] }
        var area = [Double](repeating: 0, count: n)
        var length = [Double](repeating: 0, count: n)
        for i in land { area[i] = h.cellArea }
        for i in order {
            let r = receiver[i]
            guard r != i, h.values[r] > sea else { continue }
            area[r] += area[i]
            length[r] = Swift.max(length[r], length[i] + dist[i])
        }

        var channelCells = 0.0
        var sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, m = 0.0
        for i in land where area[i] >= channelArea {
            channelCells += 1
            if area[i] <= 1e8, length[i] > 0 {
                let lx = log(area[i]), ly = log(length[i])
                sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; m += 1
            }
        }
        let landKm2 = Double(land.count) * h.cellArea / 1e6
        let density = landKm2 > 0 ? channelCells * cell / 1000 / landKm2 : 0
        let denominator = m * sxx - sx * sx
        let hack = m > 2 && denominator > 0 ? (m * sxy - sx * sy) / denominator : 0

        return Realism(hypsometry: hyps, hypsometricIntegral: integral, slopeHistogram: histogram,
                       cliffFraction: Double(cliffs) / Double(landCount), drainageDensity: density,
                       hackExponent: hack, pits: pits, landAreaKm2: landKm2, reliefMetres: relief)
    }

    /// JSON-ready dictionary, rounded for reading.
    public var json: [String: Any] {
        func r(_ v: Double, _ places: Double = 1000) -> Double { (v * places).rounded() / places }
        return [
            "hypsometry": hypsometry.map { r($0) },
            "hypsometricIntegral": r(hypsometricIntegral),
            "slopeHistogram5deg": slopeHistogram.map { r($0, 10000) },
            "cliffFraction": r(cliffFraction, 10000),
            "drainageDensityKmPerKm2": r(drainageDensity, 100),
            "hackExponent": r(hackExponent),
            "pits": pits,
            "landAreaKm2": r(landAreaKm2, 10),
            "reliefMetres": r(reliefMetres, 10),
        ]
    }
}
