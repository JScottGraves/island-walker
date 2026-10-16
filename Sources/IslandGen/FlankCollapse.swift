import Foundation
import IslandCore

/// Giant flank collapse, like Oahu's Nuʻuanu landslide off the Koʻolau.
///
/// Mature Hawaiian shields usually lose a sector of the flank that faces open ocean. The
/// curved headscarp left behind is what erosion turns into a pali: waterfalls on the big
/// windward streams retreat inland from it and carve amphitheater-headed valleys, and the
/// divides between them sharpen into knife-edge ridges. So the collapse runs before
/// erosion, not after it (research/hawaiian-geology.md, sections 4 and 5).
///
/// Geometry: the flank is cut along a curved headscarp that passes just windward of the
/// summit and bends seaward at both ends (a parabola opening away from the summit). The
/// summit and the whole leeward half keep their height, and the scarp ends run out to sea
/// on their own, so there are no radial seams.
public enum FlankCollapse {
    public struct Scarp: Sendable {
        /// The shield's actual summit (highest ground near its centre), in metres.
        public var x: Double
        public var y: Double
        /// Direction the collapse faces, in radians (atan2(dy, dx) convention).
        public var direction: Double
        /// Distance from the summit to the scarp crest along `direction`, in metres.
        public var offset: Double = 1500
        /// How sharply the scarp curves seaward away from its midpoint; the scarp moves
        /// out by perp² / (2 · curvatureRadius).
        public var curvatureRadius: Double = 9000
        /// Horizontal width of the scarp face, in metres. Smaller is steeper.
        public var faceWidth: Double = 900
        /// Height of the debris surface at the scarp foot and its seaward slope: a narrow
        /// coastal plain under the cliff, like Kāneʻohe, then down below sea level.
        public var footHeight: Double = 50
        public var footSlope: Double = 0.02
    }

    /// Chooses which shield collapses and which way. The collapse faces across the shield's
    /// main rift (a flank, not down a rift arm), on the side facing the wind, so the pali is
    /// on the wet side where rain drives valleys back into it, as on the Koʻolau.
    public static func choose(shields: [ShieldVolcano], height: Grid, windDirection: Double, seed: UInt64) -> Scarp? {
        guard !shields.isEmpty else { return nil }
        var rng = SplitMix64(seed: seed ^ 0x434F_4C4C_4150)
        let cx = shields.map(\.x).reduce(0, +) / Double(shields.count)
        let cy = shields.map(\.y).reduce(0, +) / Double(shields.count)
        let upwind = (-cos(windDirection), -sin(windDirection))

        var best = 0, bestScore = -Double.infinity, bestDir = 0.0
        for (k, s) in shields.enumerated() {
            // Main rift axis: the longest rift.
            var axis = s.riftAngles.first ?? 0, longest = 0.0
            for (a, len) in zip(s.riftAngles, s.riftLengths) where len > longest { longest = len; axis = a }
            var nx = -sin(axis), ny = cos(axis)
            if nx * upwind.0 + ny * upwind.1 < 0 { nx = -nx; ny = -ny }
            // Prefer flanks that face open ocean (away from the other shields) and the wind.
            var ox = s.x - cx, oy = s.y - cy
            let ol = (ox * ox + oy * oy).squareRoot()
            if ol > 1 { ox /= ol; oy /= ol } else { ox = nx; oy = ny }
            let score = (nx * upwind.0 + ny * upwind.1) + 0.7 * (nx * ox + ny * oy) + 0.2 * rng.unit()
            if score > bestScore { bestScore = score; best = k; bestDir = atan2(ny, nx) }
        }

        // Anchor on the real summit: domain warping moves peaks off the skeleton centre.
        let s = shields[best]
        let reach = Int(6000 / height.cellSize)
        let gx = Int(s.x / height.cellSize), gy = Int(s.y / height.cellSize)
        var top = height.index(Swift.min(Swift.max(gx, 0), height.width - 1), Swift.min(Swift.max(gy, 0), height.height - 1))
        for y in Swift.max(0, gy - reach)...Swift.min(height.height - 1, gy + reach) {
            for x in Swift.max(0, gx - reach)...Swift.min(height.width - 1, gx + reach) {
                let i = height.index(x, y)
                if height.values[i] > height.values[top] { top = i }
            }
        }
        var scarp = Scarp(x: (Double(top % height.width) + 0.5) * height.cellSize,
                          y: (Double(top / height.width) + 0.5) * height.cellSize,
                          direction: bestDir)
        scarp.offset = rng.uniform(800, 2200)
        scarp.curvatureRadius = rng.uniform(7000, 12000)
        return scarp
    }

    /// Cuts the flank away beyond the scarp and weakens the rock on the scarp face
    /// (fresh, shattered breccia erodes fast).
    public static func apply(_ scarp: Scarp, height: inout Grid, hardness: inout Grid, seed: UInt64) {
        let noiseSeed = Hash.mix(seed ^ 0x5343_4152_50)
        let dx = cos(scarp.direction), dy = sin(scarp.direction)
        for y in 0..<height.height {
            for x in 0..<height.width {
                let px = (Double(x) + 0.5) * height.cellSize, py = (Double(y) + 0.5) * height.cellSize
                let rx = px - scarp.x, ry = py - scarp.y
                let along = rx * dx + ry * dy
                guard along > 0 else { continue }
                let perp = -rx * dy + ry * dx
                // Curved, irregular scarp line.
                let wobble = 500 * Noise.fbm(perp / 4000 + 10, 0.5, seed: noiseSeed, octaves: 4)
                let crest = scarp.offset + perp * perp / (2 * scarp.curvatureRadius) + wobble
                let t = smoothstep(crest, crest + scarp.faceWidth, along)
                guard t > 0 else { continue }

                let beyond = along - crest - scarp.faceWidth
                let debris = scarp.footHeight - scarp.footSlope * Swift.max(0, beyond)
                    + 12 * Noise.fbm(px / 1500, py / 1500, seed: noiseSeed &+ 1, octaves: 4)
                let i = height.index(x, y)
                let h = Double(height.values[i])
                let cut = Swift.min(h, debris)
                height.values[i] = Float(h + (cut - h) * t)
                // Shattered rock on and just below the scarp face is weak.
                let face = 4 * t * (1 - t)
                hardness.values[i] = Float(Double(hardness.values[i]) * (1 - 0.3 * face))
            }
        }
    }

    private static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }
}
