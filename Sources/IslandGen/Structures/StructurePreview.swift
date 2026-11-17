import Foundation
import IslandCore

extension Preview {
    /// `base` (a map of the whole domain, north up) with each structure's zones drawn on
    /// it: paved red, grass strips yellow, at least `minPixels` wide so they show on a
    /// whole-island map.
    public static func structures(_ structures: [Structure], over base: Image, domain: Double, minPixels: Double = 3) -> Image {
        var out = base
        let scale = Double(base.width) / domain
        for s in structures {
            for zone in s.zones.reversed() {
                let colour: (UInt8, UInt8, UInt8)
                switch zone.surface {
                case .paved: colour = (230, 30, 40)
                case .mownGrass: colour = (250, 220, 40)
                default: colour = (250, 250, 250)
                }
                let hw = Swift.max(zone.halfWidth, minPixels / 2 / scale)
                let hl = Swift.max(zone.halfLength, minPixels / 2 / scale)
                let reach = (hl * hl + hw * hw).squareRoot()
                let x0 = Swift.max(0, Int((s.x - reach) * scale)), x1 = Swift.min(base.width - 1, Int((s.x + reach) * scale) + 1)
                let y0 = Swift.max(0, Int((s.y - reach) * scale)), y1 = Swift.min(base.height - 1, Int((s.y + reach) * scale) + 1)
                guard x0 <= x1, y0 <= y1 else { continue }
                for py in y0...y1 {
                    for px in x0...x1 {
                        let (u, v) = s.local(x: (Double(px) + 0.5) / scale, y: (Double(py) + 0.5) / scale)
                        guard abs(u) <= hl, abs(v) <= hw else { continue }
                        let o = ((base.height - 1 - py) * base.width + px) * 3
                        out.rgb[o] = colour.0; out.rgb[o + 1] = colour.1; out.rgb[o + 2] = colour.2
                    }
                }
            }
        }
        return out
    }

    /// A close-up hillshade of `height` around a structure, `size` metres square at
    /// `metresPerPixel`, with the zone outlines traced on it: for checking the grading.
    public static func structureCloseUp(_ s: Structure, height: Grid, seaLevel: Float, size: Double = 3000,
                                        metresPerPixel: Double = 5) -> Image {
        let n = Int(size / metresPerPixel)
        var crop = Grid(width: n, height: n, cellSize: metresPerPixel)
        let x0 = s.x - size / 2, y0 = s.y - size / 2
        for j in 0..<n {
            for i in 0..<n {
                let wx = x0 + (Double(i) + 0.5) * metresPerPixel, wy = y0 + (Double(j) + 0.5) * metresPerPixel
                crop[i, j] = height.sampleCubic(x: wx / height.cellSize - 0.5, y: wy / height.cellSize - 0.5)
            }
        }
        var image = hillshade(crop, seaLevel: seaLevel, sunAzimuth: 315, sunAltitude: 35)
        // Zone outlines, one pixel wide.
        for j in 0..<n {
            for i in 0..<n {
                let wx = x0 + (Double(i) + 0.5) * metresPerPixel, wy = y0 + (Double(j) + 0.5) * metresPerPixel
                let (u, v) = s.local(x: wx, y: wy)
                for zone in s.zones {
                    let eu = abs(abs(u) - zone.halfLength), ev = abs(abs(v) - zone.halfWidth)
                    let onEdge = (eu < metresPerPixel / 2 && abs(v) <= zone.halfWidth)
                        || (ev < metresPerPixel / 2 && abs(u) <= zone.halfLength)
                    if onEdge {
                        let o = ((n - 1 - j) * n + i) * 3
                        let paved = zone.surface == .paved
                        image.rgb[o] = paved ? 230 : 250; image.rgb[o + 1] = paved ? 30 : 220; image.rgb[o + 2] = paved ? 40 : 40
                    }
                }
            }
        }
        return image
    }
}
