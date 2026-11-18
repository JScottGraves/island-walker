import Foundation
import IslandCore

extension Island {
    /// Oahu's median sea depth (m) against distance offshore (m), measured on the CUDEM
    /// topobathy bundle (`islandgen --shelf`): a few metres of reef flat and sand in the first
    /// few hundred metres, a shelf to about 100 m at 3–4 km, then the island's flank.
    public static let oahuShelf: [(distance: Double, depth: Double)] = [
        (0, 0), (100, 3), (250, 4.5), (500, 8), (1000, 15), (2000, 49), (3000, 94), (5000, 360), (8000, 568),
    ]

    /// Re-imposes a sea floor measured from the eroded shoreline. Wave-cut retreat leaves
    /// the ground it took from the coast just below sea level, and nothing in the erosion
    /// model works on the sea floor, so without this the islands end in a platform a few
    /// decimetres deep and kilometres wide. On real Hawaiian islands that platform is
    /// abraded and drowned (sea-level cycles, subsidence) into a reef and shelf that deepen
    /// steadily from the shore.
    ///
    /// Within `blendStart` of the shore the sea floor follows `oahuShelf`, its width and
    /// depth varied along the coast by low-frequency noise. From `blendStart` to `blendEnd`
    /// it fades back to the deeper of that profile and the original flank, and beyond it the
    /// flank is kept. Land is not touched and sea stays below sea level.
    public static func shelve(_ h: inout Grid, seaLevel: Float, seed: UInt64, windDirection: Double = 5 * Double.pi / 4,
                              blendStart: Double = 4000, blendEnd: Double = 10000) {
        let w = h.width, cell = h.cellSize
        let values = h.values
        let d = ShelfProfile.distanceTransform(width: w, height: h.height) { values[$0] > seaLevel }
        var cx = 0.0, cy = 0.0, land = 0.0
        for i in 0..<h.count where values[i] > seaLevel { cx += Double(i % w); cy += Double(i / w); land += 1 }
        cx = (cx / Swift.max(land, 1) + 0.5) * cell
        cy = (cy / Swift.max(land, 1) + 0.5) * cell
        let up = (-cos(windDirection), -sin(windDirection))
        let noiseSeed = Hash.mix(seed ^ 0x5348_454C_46)
        let table = oahuShelf
        func profile(_ x: Double) -> Double {
            guard x < table.last!.distance else {
                let a = table[table.count - 2], b = table[table.count - 1]
                return b.depth + (x - b.distance) * (b.depth - a.depth) / (b.distance - a.distance)
            }
            var k = 1
            while table[k].distance < x { k += 1 }
            let a = table[k - 1], b = table[k]
            return a.depth + (x - a.distance) / (b.distance - a.distance) * (b.depth - a.depth)
        }
        // Log shelf-width factor per cell: wider and narrower along the coast, and wider on
        // the windward side, where Oahu's reefs are. Centred on its median over the shelf so
        // that each island's median profile is Oahu's whatever its noise.
        func logStretch(_ px: Double, _ py: Double) -> Double {
            let rx = px - cx, ry = py - cy
            let windward = (rx * up.0 + ry * up.1) / Swift.max(1, (rx * rx + ry * ry).squareRoot())
            return 0.8 * Noise.fbm(px / 8000, py / 8000, seed: noiseSeed, octaves: 3) + 0.2 * windward
        }
        func offshore(_ i: Int) -> Double { Swift.max(0, (Double(d[i]) - 0.5) * cell) }
        func position(_ i: Int) -> (Double, Double) { ((Double(i % w) + 0.5) * cell, (Double(i / w) + 0.5) * cell) }
        var shelfStretch: [Double] = []
        for i in 0..<h.count where values[i] <= seaLevel && offshore(i) < blendStart {
            let (px, py) = position(i)
            shelfStretch.append(logStretch(px, py))
        }
        shelfStretch.sort()
        let centre = shelfStretch.isEmpty ? 0 : shelfStretch[shelfStretch.count / 2]

        for i in 0..<h.count where values[i] <= seaLevel {
            let offshore = offshore(i)
            guard offshore < blendEnd else { continue }
            let (px, py) = position(i)
            let stretch = exp(logStretch(px, py) - centre)
            // Patch reefs and channels on the shelf.
            let relief = 1 + 0.2 * Noise.fbm(px / 1500, py / 1500, seed: noiseSeed &+ 1, octaves: 3)
                + 0.6 * Noise.fbm(px / 400, py / 400, seed: noiseSeed &+ 2, octaves: 3)
            let shelf = profile(offshore / stretch) * relief
            let original = Double(seaLevel - values[i])
            let t = Swift.min(1, Swift.max(0, (offshore - blendStart) / (blendEnd - blendStart)))
            let keep = t * t * (3 - 2 * t)
            let depth = (1 - keep) * shelf + keep * Swift.max(shelf, original)
            h.values[i] = seaLevel - Float(Swift.max(0.1, depth))
        }
    }
}
