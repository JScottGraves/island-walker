import Foundation
import IslandCore

/// CPU-rendered first-person pictures of a heightfield, for checking what terrain looks
/// like on foot without a GPU (CI, Linux). Column ray-marching ("voxel space"): each
/// screen column marches one ray outward and draws terrain front to back.
/// Materials, light, haze and tone mapping come from `Look`, shared with the Metal
/// terrain shader, so the two read alike.
public enum GroundView {
    public struct Camera: Sendable {
        /// Eye position in metres; z is height above sea level.
        public var x: Double
        public var y: Double
        public var z: Double
        /// Heading in radians, 0 = north (+y), increasing clockwise.
        public var yaw: Double
        /// Radians above the horizon.
        public var pitch: Double = 0
        public var horizontalFOV: Double = 75 * Double.pi / 180

        public init(x: Double, y: Double, z: Double, yaw: Double, pitch: Double = 0) {
            self.x = x
            self.y = y
            self.z = z
            self.yaw = yaw
            self.pitch = pitch
        }
    }

    /// - Parameter lighting: baked sky and sun visibility for `h`; baked here when nil.
    ///   Pass one in when rendering several views of the same island.
    public static func render(_ h: Grid, camera c: Camera, width: Int = 1280, height: Int = 720,
                              seaLevel: Float = 0, maxDistance: Double = 90_000,
                              lighting: TerrainLighting? = nil) -> Preview.Image {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let focal = Double(width) / 2 / tan(c.horizontalFOV / 2)
        let horizon = Double(height) / 2 + tan(c.pitch) * focal
        let light = lighting ?? TerrainLighting.bake(height: h, seaLevel: seaLevel)
        let maxHeight = Swift.max(1, Double(h.range.max) - Double(seaLevel))
        let sea = Double(seaLevel)
        // Keep the eye above the surface as rendered (bicubic + detail), which can sit higher
        // than the raw cell height the camera was placed from.
        var c = c
        let ex = c.x / h.cellSize - 0.5, ey = c.y / h.cellSize - 0.5
        let under = Double(h.sampleCubic(x: ex, y: ey))
        let rendered = Swift.max(under + detail(c.x, c.y, ground: under, sea: sea, distance: 0), sea)
        c.z = Swift.max(c.z, rendered + 1.7)

        func put(_ col: Int, _ row: Int, _ color: RGB) {
            let c = Look.toneMap(color)
            let o = (row * width + col) * 3
            rgb[o] = byte(Look.encode(c.r))
            rgb[o + 1] = byte(Look.encode(c.g))
            rgb[o + 2] = byte(Look.encode(c.b))
        }

        for col in 0..<width {
            let offset = atan((Double(col) + 0.5 - Double(width) / 2) / focal)
            let angle = c.yaw + offset
            let dx = sin(angle), dy = cos(angle)
            let depthScale = cos(offset) // distance along the ray → depth along the view axis
            var top = height              // rows [top, height) are already drawn
            var d = 0.5
            while d < maxDistance && top > 0 {
                let px = c.x + dx * d, py = c.y + dy * d
                let gx = px / h.cellSize - 0.5, gy = py / h.cellSize - 0.5
                if gx < 0 || gy < 0 || gx > Double(h.width - 1) || gy > Double(h.height - 1) { break }
                let base = Double(h.sampleCubic(x: gx, y: gy))
                let ground = base + detail(px, py, ground: base, sea: sea, distance: d)
                let surface = Swift.max(ground, sea)
                let depth = d * depthScale
                let screenY = horizon - (surface - c.z) * focal / depth
                let row = Swift.max(0, Int(screenY.rounded(.up)))
                if row < top {
                    let color = shade(h, gx: gx, gy: gy, px: px, py: py, ground: ground, sea: sea, distance: d, light: light, maxHeight: maxHeight)
                    for r in row..<top { put(col, r, color) }
                    top = row
                }
                d += Swift.max(0.25, d * 0.004)
            }
            // Sky above whatever terrain was drawn.
            for r in 0..<top {
                let t = Swift.min(1, Swift.max(0, (horizon - Double(r)) / Double(height)))
                put(col, r, Look.haze.mix(Look.zenith, t))
            }
        }
        return Preview.Image(width: width, height: height, rgb: rgb)
    }

    private static func shade(_ h: Grid, gx: Double, gy: Double, px: Double, py: Double, ground: Double, sea: Double,
                              distance: Double, light: TerrainLighting, maxHeight: Double) -> RGB {
        var albedo: RGB
        var n = (0.0, 0.0, 1.0)
        let depthBelow = sea - ground
        if depthBelow > 0 {
            albedo = Look.shallowWater.mix(Look.deepWater, Swift.min(1, depthBelow / 400))
        } else {
            let e = 1.0
            var hx = Double(h.sampleCubic(x: gx + e, y: gy) - h.sampleCubic(x: gx - e, y: gy)) / (2 * e * h.cellSize)
            var hy = Double(h.sampleCubic(x: gx, y: gy + e) - h.sampleCubic(x: gx, y: gy - e)) / (2 * e * h.cellSize)
            // Detail-noise gradient, faded out with distance like the displacement itself.
            let de = 2.0
            let fade = detailFade(distance)
            if fade > 0 {
                hx += (detailNoise(px + de, py) - detailNoise(px - de, py)) / (2 * de) * fade
                hy += (detailNoise(px, py + de) - detailNoise(px, py - de)) / (2 * de) * fade
            }
            n = normalize((-hx, -hy, 1))
            let slope = 1 - n.2
            let above = ground - sea
            let grass = Look.grassLow.mix(Look.grassHigh, Swift.min(1, Swift.max(0, above / maxHeight)))
            albedo = grass.mix(Look.soil, smoothstep(0.15, 0.30, slope))
            albedo = albedo.mix(Look.rock, smoothstep(0.35, 0.55, slope))
            // Beach: a narrow band just above the waterline on gentle slopes only.
            albedo = Look.sand.mix(albedo, Swift.max(smoothstep(1, 3, above), smoothstep(0.1, 0.25, slope)))
        }
        let s = Look.sunDirection
        let color = Look.lit(albedo: albedo, normalZ: n.2, nDotL: n.0 * s.x + n.1 * s.y + n.2 * s.z,
                             sunVisibility: Double(light.sun.sample(x: gx, y: gy)),
                             skyVisibility: Double(light.sky.sample(x: gx, y: gy)))
        return color.mix(Look.haze, Look.hazeAmount(distance: distance))
    }

    /// Metre-scale relief the 10–80 m base grid can't hold: hummocks, lava-flow lumps.
    /// Stand-in for the runtime detail amplification in the plan.
    static func detailNoise(_ x: Double, _ y: Double) -> Double {
        6.0 * Noise.fbm(x / 90, y / 90, seed: 0xD37A_11, octaves: 4)
            + 1.5 * Noise.fbm(x / 15, y / 15, seed: 0xD37A_12, octaves: 3)
            + 0.3 * Noise.fbm(x / 3, y / 3, seed: 0xD37A_13, octaves: 2)
    }

    private static func detailFade(_ distance: Double) -> Double {
        1 - smoothstep(1500, 4000, distance)
    }

    private static func detail(_ x: Double, _ y: Double, ground: Double, sea: Double, distance: Double) -> Double {
        let fade = detailFade(distance)
        guard fade > 0, ground > sea + 1 else { return 0 }
        return detailNoise(x, y) * fade * smoothstep(sea + 1, sea + 6, ground)
    }

    private static func normalize(_ v: (Double, Double, Double)) -> (Double, Double, Double) {
        let l = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
        return (v.0 / l, v.1 / l, v.2 / l)
    }

    private static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }

    private static func byte(_ v: Double) -> UInt8 {
        UInt8(Swift.max(0, Swift.min(255, (v * 255).rounded())))
    }
}

/// Picks informative viewpoints automatically: from the summit down the biggest valley to
/// the sea, from a ridge looking across into a valley, and from the coast looking inland.
public enum Viewpoints {
    public struct Named {
        public var name: String
        public var camera: GroundView.Camera
    }

    /// With `scarp`, the pali view looks at that flank-collapse cliff from its plain;
    /// without it, at the biggest cliff anywhere.
    public static func automatic(height h: Grid, drainageArea area: Grid, seaLevel: Float = 0,
                                 scarp: FlankCollapse.Scarp? = nil) -> [Named] {
        var summit = 0
        for i in 0..<h.count where h.values[i] > h.values[summit] { summit = i }
        let maxH = h.values[summit] - seaLevel

        // Valley: the wettest mid-elevation cell. Coast: the biggest river just above sea level.
        var valley = -1, valleyArea: Float = 0
        var coast = -1, coastArea: Float = 0
        for i in 0..<h.count {
            let z = h.values[i] - seaLevel
            if z > 0.25 * maxH && z < 0.5 * maxH && area.values[i] > valleyArea { valley = i; valleyArea = area.values[i] }
            if z > 3 && z < 30 && area.values[i] > coastArea { coast = i; coastArea = area.values[i] }
        }

        func position(_ i: Int) -> (Double, Double) {
            ((Double(i % h.width) + 0.5) * h.cellSize, (Double(i / h.width) + 0.5) * h.cellSize)
        }
        /// Looks from `i` toward `t`. With `brink`, first walks from `i` toward `t` to the
        /// edge where the ground starts falling away, so a broad summit or ridge top doesn't
        /// fill the frame with foreground.
        func camera(from i: Int, to t: Int, brink: Bool = false, pitchOffset: Double = 0) -> GroundView.Camera {
            var (x, y) = position(i)
            let (tx, ty) = position(t)
            let total = Swift.max(1, ((tx - x) * (tx - x) + (ty - y) * (ty - y)).squareRoot())
            let ux = (tx - x) / total, uy = (ty - y) / total
            if brink {
                let start = Double(h.values[i])
                var d = 0.0
                while d < total * 0.6 {
                    let ahead = Double(h.sampleCubic(x: (x + ux * (d + h.cellSize)) / h.cellSize - 0.5,
                                                     y: (y + uy * (d + h.cellSize)) / h.cellSize - 0.5))
                    if start - ahead > 40 { break }
                    d += h.cellSize / 2
                }
                x += ux * d
                y += uy * d
            }
            let z = Double(h.sampleCubic(x: x / h.cellSize - 0.5, y: y / h.cellSize - 0.5)) + 1.7
            let dist = Swift.max(1, ((tx - x) * (tx - x) + (ty - y) * (ty - y)).squareRoot())
            let pitch = atan((Double(Swift.max(h.values[t], seaLevel)) - z) / dist) + pitchOffset
            return GroundView.Camera(x: x, y: y, z: z, yaw: atan2(ux, uy), pitch: pitch)
        }

        /// The lowland spot with the most relief 2–3.5 km away: standing on a coastal plain
        /// looking up at a cliff (a pali), far enough back to see its brink. The first
        /// kilometre toward the cliff must stay low, so the camera is on open plain rather
        /// than in a valley or at the foot of a hummock. Searches a coarse lattice in 24
        /// directions.
        func cliff() -> (Int, Int)? {
            let stride = 4, directions = 24
            var bestFoot = -1, bestTop = -1, bestRelief: Float = 150
            for cy in Swift.stride(from: 0, to: h.height, by: stride) {
                for cx in Swift.stride(from: 0, to: h.width, by: stride) {
                    let i = cy * h.width + cx
                    let z = h.values[i] - seaLevel
                    guard z > 5 && z < 150 else { continue }
                    if let scarp {
                        // Only in front of the scarp's middle, on its plain.
                        let (px, py) = position(i)
                        let rx = px - scarp.x, ry = py - scarp.y
                        let along = rx * cos(scarp.direction) + ry * sin(scarp.direction)
                        let perp = -rx * sin(scarp.direction) + ry * cos(scarp.direction)
                        guard along > 0 && abs(perp) < 0.5 * scarp.halfLength else { continue }
                    }
                    for k in 0..<directions {
                        let a = Double(k) * 2 * Double.pi / Double(directions)
                        if let scarp {
                            // Facing inland, within 40° of straight at the cliff.
                            var off = abs(a - (scarp.direction + Double.pi)).truncatingRemainder(dividingBy: 2 * Double.pi)
                            if off > Double.pi { off = 2 * Double.pi - off }
                            if off > 40 * Double.pi / 180 { continue }
                        }
                        var clear = true
                        var near = h.cellSize
                        while near <= 1000 && clear {
                            let nx = cx + Int((cos(a) * near / h.cellSize).rounded())
                            let ny = cy + Int((sin(a) * near / h.cellSize).rounded())
                            clear = h.contains(nx, ny) && h.values[ny * h.width + nx] - h.values[i] < 60
                            near += h.cellSize
                        }
                        guard clear else { continue }
                        var d = 2000.0
                        while d <= 3500 {
                            let tx = cx + Int((cos(a) * d / h.cellSize).rounded())
                            let ty = cy + Int((sin(a) * d / h.cellSize).rounded())
                            if h.contains(tx, ty) {
                                let j = ty * h.width + tx
                                let relief = h.values[j] - h.values[i]
                                if relief > bestRelief { bestRelief = relief; bestFoot = i; bestTop = j }
                            }
                            d += h.cellSize
                        }
                    }
                }
            }
            return bestFoot >= 0 ? (bestFoot, bestTop) : nil
        }

        var views: [Named] = []
        if coast >= 0 {
            views.append(Named(name: "summit", camera: camera(from: summit, to: coast, brink: true, pitchOffset: 0.05)))
        }
        if valley >= 0 {
            // Highest cell 1.5–4 km from the valley floor: a ridge overlooking it.
            let (vx, vy) = position(valley)
            var ridge = -1
            let r = Int(4000 / h.cellSize) + 1
            let cx = valley % h.width, cy = valley / h.width
            for y in Swift.max(0, cy - r)...Swift.min(h.height - 1, cy + r) {
                for x in Swift.max(0, cx - r)...Swift.min(h.width - 1, cx + r) {
                    let i = y * h.width + x
                    let (px, py) = position(i)
                    let d = ((px - vx) * (px - vx) + (py - vy) * (py - vy)).squareRoot()
                    if d >= 1500 && d <= 4000 && (ridge < 0 || h.values[i] > h.values[ridge]) { ridge = i }
                }
            }
            if ridge >= 0 { views.append(Named(name: "ridge", camera: camera(from: ridge, to: valley, brink: true, pitchOffset: 0.05))) }
        }
        if coast >= 0 {
            views.append(Named(name: "coast", camera: camera(from: coast, to: summit, pitchOffset: 0.02)))
        }
        if let (foot, top) = cliff() {
            views.append(Named(name: "pali", camera: camera(from: foot, to: top, pitchOffset: 0.03)))
        }
        return views
    }
}
