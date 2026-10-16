import Foundation
import IslandCore

/// One shield volcano: a broad, low-angle cone stretched along its rift zones.
public struct ShieldVolcano: Sendable {
    /// Summit position in metres from the domain's lower-left corner.
    public var x: Double
    public var y: Double
    /// Summit height above the sea floor reference, in metres.
    public var summitHeight: Double
    /// Distance from the rift skeleton at which the flank meets the sea floor, in metres.
    public var radius: Double
    /// Rift zone directions in radians.
    public var riftAngles: [Double]
    /// Rift zone lengths in metres, one per angle.
    public var riftLengths: [Double]

    public init(x: Double, y: Double, summitHeight: Double, radius: Double, riftAngles: [Double], riftLengths: [Double]) {
        precondition(riftAngles.count == riftLengths.count)
        self.x = x
        self.y = y
        self.summitHeight = summitHeight
        self.radius = radius
        self.riftAngles = riftAngles
        self.riftLengths = riftLengths
    }

    /// Distance from (px, py) to the volcano's skeleton (summit point plus rift segments).
    func skeletonDistance(_ px: Double, _ py: Double) -> Double {
        var best = ((px - x) * (px - x) + (py - y) * (py - y)).squareRoot()
        for (a, len) in zip(riftAngles, riftLengths) {
            let dx = cos(a), dy = sin(a)
            let t = Swift.min(Swift.max((px - x) * dx + (py - y) * dy, 0), len)
            let qx = x + dx * t, qy = y + dy * t
            best = Swift.min(best, ((px - qx) * (px - qx) + (py - qy) * (py - qy)).squareRoot())
        }
        return best
    }

    /// Distance to the rift segments alone (not the summit). Dikes cluster here.
    func riftDistance(_ px: Double, _ py: Double) -> Double {
        var best = Double.infinity
        for (a, len) in zip(riftAngles, riftLengths) {
            let dx = cos(a), dy = sin(a)
            let t = Swift.min(Swift.max((px - x) * dx + (py - y) * dy, 0), len)
            let qx = x + dx * t, qy = y + dy * t
            best = Swift.min(best, ((px - qx) * (px - qx) + (py - qy) * (py - qy)).squareRoot())
        }
        return best
    }
}

/// Builds the pre-erosion island: sea floor, overlapping shield volcanoes and a rock
/// hardness map. Erosion later carves this into ridges and valleys.
public enum VolcanicBase {
    /// Picks 2 or 3 shields placed like the Hawaiian pattern: offset along a hotspot track
    /// through the middle of the domain, each with 2 or 3 rift zones.
    public static func randomShields(seed: UInt64, domainSize: Double) -> [ShieldVolcano] {
        var rng = SplitMix64(seed: seed ^ 0x5348_4945_4C44)
        let count = rng.unit() < 0.7 ? 2 : 3
        let track = rng.uniform(0, 2 * Double.pi)
        let spacing = domainSize * rng.uniform(0.22, 0.30)
        let cx = domainSize / 2, cy = domainSize / 2
        var shields: [ShieldVolcano] = []
        for k in 0..<count {
            let offset = (Double(k) - Double(count - 1) / 2) * spacing
            let jitter = domainSize * 0.04
            let x = cx + cos(track) * offset + rng.uniform(-jitter, jitter)
            let y = cy + sin(track) * offset + rng.uniform(-jitter, jitter)
            let riftCount = rng.unit() < 0.5 ? 2 : 3
            let base = rng.uniform(0, 2 * Double.pi)
            var angles: [Double] = []
            var lengths: [Double] = []
            for r in 0..<riftCount {
                angles.append(base + Double(r) * 2 * Double.pi / Double(riftCount) + rng.uniform(-0.5, 0.5))
                lengths.append(domainSize * rng.uniform(0.10, 0.24))
            }
            shields.append(ShieldVolcano(
                x: x, y: y,
                summitHeight: rng.uniform(2600, 3400),
                radius: domainSize * rng.uniform(0.32, 0.40),
                riftAngles: angles, riftLengths: lengths))
        }
        return shields
    }

    public struct Output {
        public var height: Grid
        /// Relative rock resistance to erosion; 1 is average, higher is harder.
        public var hardness: Grid
    }

    /// - Parameters:
    ///   - size: grid samples per side
    ///   - domainSize: domain edge length in metres
    ///   - seaFloor: depth of the surrounding ocean floor in metres (negative)
    public static func generate(seed: UInt64, size: Int, domainSize: Double, seaFloor: Double = -1800,
                                shields: [ShieldVolcano]? = nil) -> Output {
        let cell = domainSize / Double(size)
        let volcanoes = shields ?? randomShields(seed: seed, domainSize: domainSize)
        var height = Grid(width: size, height: size, cellSize: cell)
        var hardness = Grid(width: size, height: size, cellSize: cell, fill: 1)
        let noiseSeed = Hash.mix(seed ^ 0x4E4F_4953_45)
        let hardSeed = Hash.mix(seed ^ 0x4841_5244)

        for y in 0..<size {
            for x in 0..<size {
                let px = (Double(x) + 0.5) * cell, py = (Double(y) + 0.5) * cell
                // Domain warp so flanks are lobed rather than perfectly conical.
                let wx = px + 2500 * Noise.fbm(px / 9000, py / 9000, seed: noiseSeed, octaves: 4)
                let wy = py + 2500 * Noise.fbm(px / 9000 + 31.7, py / 9000 - 17.3, seed: noiseSeed, octaves: 4)

                var h = seaFloor
                var riftNear = Double.infinity
                for v in volcanoes {
                    let d = v.skeletonDistance(wx, wy) / v.radius
                    // Shield profile: convex near the summit, long gentle apron at the base.
                    let profile = exp(-3.0 * pow(d, 1.4))
                    h = Swift.max(h, seaFloor + v.summitHeight * profile)
                    riftNear = Swift.min(riftNear, v.riftDistance(wx, wy))
                }
                // Lava-flow scale roughness (tens of metres), stronger on land.
                h += 35 * Noise.fbm(px / 1800, py / 1800, seed: noiseSeed &+ 7, octaves: 5)

                height[x, y] = Float(h)
                // Dike swarms along rift zones resist erosion; ash and weathered flows are soft.
                // Kept modest and narrow: a hard, wide crest stops valley heads from cutting
                // back into the ridge, which leaves flat-topped ridges instead of sharp ones.
                let dike = exp(-pow(riftNear / 600, 2))
                let variation = 0.25 * Noise.fbm(px / 4000, py / 4000, seed: hardSeed, octaves: 4)
                hardness[x, y] = Float(Swift.max(0.4, 0.9 + 0.35 * dike + variation))
            }
        }
        return Output(height: height, hardness: hardness)
    }
}
