import Foundation
import IslandCore

/// Giant flank collapse, like Oahu's Nuʻuanu landslide off the Koʻolau.
///
/// Mature Hawaiian shields usually lose a sector of the flank that faces open ocean. The
/// curved headscarp left behind is what erosion turns into a pali: waterfalls on the big
/// windward streams retreat inland from it and carve amphitheater-headed valleys, and the
/// divides between them sharpen into knife-edge ridges. So the collapse runs before
/// erosion, not after it (research/hawaiian-geology.md, sections 4 and 5).
public enum FlankCollapse {
    public struct Sector: Sendable {
        /// Summit the sector is measured from, in metres.
        public var x: Double
        public var y: Double
        /// Direction the sector faces, in radians (atan2(dy, dx) convention).
        public var direction: Double
        /// Half of the sector's angular width, in radians.
        public var halfAngle: Double
        /// Distance from the summit to the headscarp, in metres.
        public var scarpDistance: Double
        /// Horizontal width of the scarp face, in metres. Smaller is steeper.
        public var scarpWidth: Double = 1200
        /// Height of the debris surface at the scarp foot, and its seaward slope. A small
        /// positive foot leaves a narrow coastal plain under the cliff, like Kāneʻohe.
        public var footHeight: Double = 40
        public var footSlope: Double = 0.02
    }

    /// Chooses which shield collapses and in which direction: the shield whose outward side
    /// faces most into the wind, collapsing outward and upwind. That puts the pali on the
    /// wet side, where rain drives the valleys back into it, as on the Koʻolau.
    public static func choose(shields: [ShieldVolcano], windDirection: Double, seed: UInt64) -> Sector? {
        guard !shields.isEmpty else { return nil }
        var rng = SplitMix64(seed: seed ^ 0x434F_4C4C_4150)
        let cx = shields.map(\.x).reduce(0, +) / Double(shields.count)
        let cy = shields.map(\.y).reduce(0, +) / Double(shields.count)
        let upwind = (-cos(windDirection), -sin(windDirection))

        var best = 0, bestScore = -Double.infinity
        var bestDir = (upwind.0, upwind.1)
        for (k, s) in shields.enumerated() {
            var ox = s.x - cx, oy = s.y - cy
            let len = (ox * ox + oy * oy).squareRoot()
            if len < 1 { ox = upwind.0; oy = upwind.1 } else { ox /= len; oy /= len }
            // Blend outward and upwind, then score how open and how windward it is.
            var dx = ox + upwind.0, dy = oy + upwind.1
            let dl = Swift.max((dx * dx + dy * dy).squareRoot(), 1e-9)
            dx /= dl; dy /= dl
            let score = dx * upwind.0 + dy * upwind.1 + 0.5 * (dx * ox + dy * oy) + 0.2 * rng.unit()
            if score > bestScore { bestScore = score; best = k; bestDir = (dx, dy) }
        }
        let s = shields[best]
        return Sector(x: s.x, y: s.y,
                      direction: atan2(bestDir.1, bestDir.0),
                      halfAngle: rng.uniform(0.75, 1.0),
                      scarpDistance: s.radius * rng.uniform(0.10, 0.14))
    }

    /// Cuts the sector out of the heightfield and softens the rock on the scarp face
    /// (fresh, shattered breccia erodes fast).
    public static func apply(_ sector: Sector, height: inout Grid, hardness: inout Grid, seed: UInt64) {
        let noiseSeed = Hash.mix(seed ^ 0x5343_4152_50)
        for y in 0..<height.height {
            for x in 0..<height.width {
                let px = (Double(x) + 0.5) * height.cellSize, py = (Double(y) + 0.5) * height.cellSize
                let rx = px - sector.x, ry = py - sector.y
                let r = (rx * rx + ry * ry).squareRoot()
                guard r > 1 else { continue }
                var dphi = atan2(ry, rx) - sector.direction
                dphi = atan2(sin(dphi), cos(dphi))
                let inSector = 1 - smoothstep(sector.halfAngle - 0.2, sector.halfAngle + 0.2, abs(dphi))
                guard inSector > 0 else { continue }

                // Curved, irregular headscarp: its distance wanders with angle.
                let wobble = 1 + 0.18 * Noise.fbm(dphi * 3 + 10, 0.5, seed: noiseSeed, octaves: 3)
                let r0 = sector.scarpDistance * wobble
                let t = smoothstep(r0, r0 + sector.scarpWidth, r)
                guard t > 0 else { continue }

                let debris = sector.footHeight - sector.footSlope * (r - r0 - sector.scarpWidth)
                    + 15 * Noise.fbm(px / 1500, py / 1500, seed: noiseSeed &+ 1, octaves: 4)
                let i = height.index(x, y)
                let h = Double(height.values[i])
                let cut = Swift.min(h, debris)
                height.values[i] = Float(h + (cut - h) * t * inSector)
                // Shattered rock on and just below the scarp face is weak.
                let face = t * (1 - t) * 4 * inSector
                hardness.values[i] = Float(Double(hardness.values[i]) * (1 - 0.3 * face))
            }
        }
    }

    private static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }
}
