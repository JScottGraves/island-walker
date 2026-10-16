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
/// Geometry: the flank is cut along a headscarp that follows the shield's real crest line
/// a short way to windward (domain warping bends rifts, so a straight or circular line
/// would run through the crest in places). The scarp bends seaward at both ends, and the
/// cut fades out where the crest drops away, so the cliff loses height toward its ends
/// instead of ending in a straight seam. A few notches set back into the scarp seed the
/// amphitheater heads that erosion then drives inland. Below the cliff the debris forms a
/// gently sloping coastal plain that steepens offshore into the old flank's bathymetry.
public enum FlankCollapse {
    public struct Scarp: Sendable {
        /// The shield's actual summit (highest ground near its centre), in metres.
        public var x: Double
        public var y: Double
        /// Summit height in metres above sea level.
        public var summitHeight: Double = 1000
        /// Direction the collapse faces, in radians (atan2(dy, dx) convention).
        public var direction: Double
        /// Distance from the crest line to the top of the scarp, in metres.
        public var offset: Double = 800
        /// How sharply the scarp curves seaward away from its midpoint; the scarp moves
        /// out by perp² / (2 · curvatureRadius).
        public var curvatureRadius: Double = 9000
        /// Half the length of the scarp along the crest; the cut fades out beyond this.
        public var halfLength: Double = 9000
        /// Horizontal width of the scarp face, in metres. Smaller is steeper.
        public var faceWidth: Double = 900
        /// Debris surface: `footHeight` at the scarp foot, falling at `plainSlope` across a
        /// plain `plainWidth` wide (Kāneʻohe), then at `shelfSlope` out to sea.
        public var footHeight: Double = 70
        public var plainSlope: Double = 0.012
        public var plainWidth: Double = 3500
        public var shelfSlope: Double = 0.06
        /// Amphitheater seeds: positions along the scarp (perpendicular offset from the
        /// summit line, metres), half-widths and set-backs into the flank.
        public var notches: [(perp: Double, width: Double, depth: Double)] = []
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
        scarp.summitHeight = Double(height.values[top])
        scarp.offset = rng.uniform(500, 1100)
        scarp.curvatureRadius = rng.uniform(7000, 12000)
        scarp.halfLength = rng.uniform(7000, 11000)
        scarp.footHeight = rng.uniform(55, 85)
        scarp.plainWidth = rng.uniform(2500, 4500)
        let count = 4 + Int(rng.unit() * 5)
        for _ in 0..<count {
            scarp.notches.append((perp: rng.uniform(-0.8, 0.8) * scarp.halfLength,
                                  width: rng.uniform(300, 700),
                                  depth: scarp.offset * rng.uniform(0.6, 1.0) + 300))
        }
        return scarp
    }

    /// Cuts the flank away beyond the scarp and weakens the rock on the scarp face
    /// (fresh, shattered breccia erodes fast) and along the notch axes.
    public static func apply(_ scarp: Scarp, height: inout Grid, hardness: inout Grid, seed: UInt64) {
        let noiseSeed = Hash.mix(seed ^ 0x5343_4152_50)
        let dx = cos(scarp.direction), dy = sin(scarp.direction)
        let cell = height.cellSize

        // Crest line: for each strip across the scarp, the highest point along the collapse
        // direction near the summit line, smoothed so the scarp doesn't copy every knob.
        let reach = scarp.halfLength * 1.5
        let bins = Int(2 * reach / cell) + 1
        var crestAlong = [Double](repeating: 0, count: bins)
        var crestHeight = [Double](repeating: 0, count: bins)
        let original = height
        for b in 0..<bins {
            let perp = -reach + Double(b) * cell
            var best = -Double.infinity, bestAlong = 0.0
            var along = -3000.0
            while along <= 6000 {
                let px = scarp.x + along * dx - perp * dy, py = scarp.y + along * dy + perp * dx
                let h = Double(original.sample(x: px / cell - 0.5, y: py / cell - 0.5))
                if h > best { best = h; bestAlong = along }
                along += cell / 2
            }
            crestAlong[b] = bestAlong
            crestHeight[b] = best
        }
        let radius = Int(1200 / cell)
        for _ in 0..<2 {
            crestAlong = boxSmooth(crestAlong, radius: radius)
            crestHeight = boxSmooth(crestHeight, radius: radius)
        }
        func lookup(_ table: [Double], _ perp: Double) -> Double {
            let f = Swift.min(Swift.max((perp + reach) / cell, 0), Double(bins - 1))
            let i = Swift.min(Int(f), bins - 2), t = f - Double(i)
            return table[i] * (1 - t) + table[i + 1] * t
        }

        for y in 0..<height.height {
            for x in 0..<height.width {
                let px = (Double(x) + 0.5) * cell, py = (Double(y) + 0.5) * cell
                let rx = px - scarp.x, ry = py - scarp.y
                let along = rx * dx + ry * dy
                let perp = -rx * dy + ry * dx
                guard abs(perp) < reach else { continue }
                // Fade the cut where the crest falls away (past the rift's end) and beyond
                // the scarp's ends, so the cliff dies out instead of ending in a seam.
                let strength = smoothstep(0.3, 0.55, lookup(crestHeight, perp) / scarp.summitHeight)
                    * (1 - smoothstep(0.75 * scarp.halfLength, scarp.halfLength, abs(perp)))
                guard strength > 0 else { continue }

                var notch = 0.0
                for n in scarp.notches {
                    let u = (perp - n.perp) / n.width
                    notch = Swift.max(notch, exp(-u * u))
                }
                let wobble = 400 * Noise.fbm(perp / 4000 + 10, 0.5, seed: noiseSeed, octaves: 3)
                    + 150 * Noise.fbm(perp / 1200 + 30, 0.5, seed: noiseSeed &+ 2, octaves: 3)
                var crest = lookup(crestAlong, perp) + scarp.offset + perp * perp / (2 * scarp.curvatureRadius) + wobble
                for n in scarp.notches {
                    let u = (perp - n.perp) / n.width
                    crest -= n.depth * exp(-u * u)
                }
                let i = height.index(x, y)

                // Weak rock along the notch axes, from behind the notch head out to the
                // plain, so streams concentrate there and cut back into the cliff.
                if notch > 0.05 && along > crest - 2000 && along < crest + scarp.faceWidth + scarp.plainWidth {
                    hardness.values[i] = Float(Double(hardness.values[i]) * (1 - 0.4 * notch * strength))
                }

                let t = smoothstep(crest, crest + scarp.faceWidth, along) * strength
                guard t > 0 else { continue }
                let beyond = Swift.max(0, along - crest - scarp.faceWidth)
                let debris = scarp.footHeight - scarp.plainSlope * Swift.min(beyond, scarp.plainWidth)
                    - scarp.shelfSlope * Swift.max(0, beyond - scarp.plainWidth)
                    + 15 * Noise.fbm(px / 1500, py / 1500, seed: noiseSeed &+ 1, octaves: 4)
                let h = Double(height.values[i])
                let cut = Swift.min(h, debris)
                height.values[i] = Float(h + (cut - h) * t)
                // Shattered rock on and just below the scarp face is weak.
                let face = 4 * t * (1 - t)
                hardness.values[i] = Float(Double(hardness.values[i]) * (1 - 0.3 * face))
            }
        }
    }

    private static func boxSmooth(_ v: [Double], radius: Int) -> [Double] {
        guard radius > 0 else { return v }
        var out = v
        for i in 0..<v.count {
            let lo = Swift.max(0, i - radius), hi = Swift.min(v.count - 1, i + radius)
            var sum = 0.0
            for j in lo...hi { sum += v[j] }
            out[i] = sum / Double(hi - lo + 1)
        }
        return out
    }

    private static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }
}
