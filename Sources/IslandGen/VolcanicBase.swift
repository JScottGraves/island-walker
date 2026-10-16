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

    /// Distances to the summit point and to each rift segment, with how far along its rift
    /// the nearest point of each segment lies (0 at the summit, 1 at the rift's tip).
    func skeletonParts(_ px: Double, _ py: Double) -> [(distance: Double, along: Double)] {
        var parts = [(((px - x) * (px - x) + (py - y) * (py - y)).squareRoot(), 0.0)]
        for (a, len) in zip(riftAngles, riftLengths) {
            let dx = cos(a), dy = sin(a)
            let t = Swift.min(Swift.max((px - x) * dx + (py - y) * dy, 0), len)
            let qx = x + dx * t, qy = y + dy * t
            parts.append((((px - qx) * (px - qx) + (py - qy) * (py - qy)).squareRoot(), len > 0 ? t / len : 0))
        }
        return parts
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
        public var shields: [ShieldVolcano]
    }

    /// - Parameters:
    ///   - size: grid samples per side
    ///   - domainSize: domain edge length in metres
    ///   - seaFloor: depth of the surrounding ocean floor in metres (negative)
    ///   - landExponent: 0 keeps the classic exp(−3·d^1.4) shield, whose land is only the
    ///     steep summit cap; scored against real Oahu it has half the flat ground and four
    ///     times the cliffs. Above 0, land follows top·(1 − r/r_shore)^landExponent: steep near
    ///     the summit and flattening to the shore, so overlapping shields meet in low saddles
    ///     and coastal plains, like Oahu (2 gives a hypsometric integral near Oahu's 0.18).
    ///     The coastline sits where the classic profile put it, and below sea level the flank
    ///     drops at `submarineSlope` to the sea floor.
    ///   - coastSlope: share of a straight cone mixed into the concave land profile. 0 leaves
    ///     the land flattening to nothing at the shore, a ring of near-flat coastal plain;
    ///     above 0 the land still rises from the sea at a few degrees, as Oahu's does, and the
    ///     flat ground is left in the saddles between shields.
    ///   - riftDescent: how far each rift crest drops from the summit to its tip, as a share of
    ///     the summit height. 0 keeps the crest level along the whole rift, which leaves long
    ///     broad summits; real rift ridges fall away from the caldera.
    public static func generate(seed: UInt64, size: Int, domainSize: Double, seaFloor: Double = -1800,
                                shields: [ShieldVolcano]? = nil, landExponent: Double = 2,
                                submarineSlope: Double = 0.15, coastSlope: Double = 0,
                                riftDescent: Double = 0) -> Output {
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
                    if landExponent > 0 {
                        // Same coastline as the classic profile, a concave land surface above it.
                        let top = v.summitHeight + seaFloor
                        let shore = v.radius * pow(log(v.summitHeight / -seaFloor) / 3, 1 / 1.4)
                        for (r, along) in v.skeletonParts(wx, wy) {
                            let crest = top * (1 - riftDescent * along)
                            let u = r / shore
                            let shield = u < 1
                                ? crest * ((1 - coastSlope) * pow(1 - u, landExponent) + coastSlope * (1 - u))
                                : Swift.max(seaFloor, submarineDepth(r - shore, slope: submarineSlope))
                            h = Swift.max(h, shield)
                        }
                    } else {
                        let d = v.skeletonDistance(wx, wy) / v.radius
                        // Shield profile: convex near the summit, long gentle apron at the base.
                        let profile = exp(-3.0 * pow(d, 1.4))
                        h = Swift.max(h, seaFloor + v.summitHeight * profile)
                    }
                    riftNear = Swift.min(riftNear, v.riftDistance(wx, wy))
                }
                // Lava-flow scale roughness (tens of metres). With the concave profile the coastal
                // plain sits only metres above the sea, so the roughness fades out near sea level
                // there instead of fraying the coast into lagoons and islets.
                let roughness = landExponent > 0 ? Swift.min(1, abs(h) / 80) : 1
                h += 35 * roughness * Noise.fbm(px / 1800, py / 1800, seed: noiseSeed &+ 7, octaves: 5)

                height[x, y] = Float(h)
                // Dike swarms along rift zones resist erosion; ash and weathered flows are soft.
                // Kept modest and narrow: a hard, wide crest stops valley heads from cutting
                // back into the ridge, which leaves flat-topped ridges instead of sharp ones.
                let dike = exp(-pow(riftNear / 600, 2))
                let variation = 0.25 * Noise.fbm(px / 4000, py / 4000, seed: hardSeed, octaves: 4)
                hardness[x, y] = Float(Swift.max(0.4, 0.9 + 0.35 * dike + variation))
            }
        }
        return Output(height: height, hardness: hardness, shields: volcanoes)
    }

    /// Depth below sea level `offshore` metres out from the shoreline: a shelf that steepens
    /// smoothly over its first 2 km to `slope`, so the gentle coastal plain doesn't meet a
    /// steep sea floor in a hard break.
    static func submarineDepth(_ offshore: Double, slope: Double, shelf: Double = 2000) -> Double {
        offshore < shelf ? -slope * offshore * offshore / (2 * shelf) : -slope * (offshore - shelf / 2)
    }
}
