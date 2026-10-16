import Foundation
import IslandCore

/// CPU-rendered first-person pictures of a heightfield, for checking what terrain looks
/// like on foot without a GPU (CI, Linux). Column ray-marching ("voxel space"): each
/// screen column marches one ray outward and draws terrain front to back.
/// Colours and fog follow the Metal terrain shader so the two read alike.
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

    public static func render(_ h: Grid, camera c: Camera, width: Int = 1280, height: Int = 720,
                              seaLevel: Float = 0, maxDistance: Double = 90_000) -> Preview.Image {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let focal = Double(width) / 2 / tan(c.horizontalFOV / 2)
        let horizon = Double(height) / 2 + tan(c.pitch) * focal
        let sun = normalize((-0.4, 0.5, 0.6))
        let fogColor = (0.62, 0.72, 0.85)
        let sea = Double(seaLevel)

        func put(_ col: Int, _ row: Int, _ color: (Double, Double, Double)) {
            var r = color.0, g = color.1, b = color.2
            r = tonemap(r); g = tonemap(g); b = tonemap(b)
            let o = (row * width + col) * 3
            rgb[o] = byte(pow(r, 1 / 2.2))
            rgb[o + 1] = byte(pow(g, 1 / 2.2))
            rgb[o + 2] = byte(pow(b, 1 / 2.2))
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
                let ground = Double(h.sample(x: gx, y: gy)) + detail(px, py, ground: Double(h.sample(x: gx, y: gy)), sea: sea, distance: d)
                let surface = Swift.max(ground, sea)
                let depth = d * depthScale
                let screenY = horizon - (surface - c.z) * focal / depth
                let row = Swift.max(0, Int(screenY.rounded(.up)))
                if row < top {
                    let color = shade(h, gx: gx, gy: gy, px: px, py: py, ground: ground, sea: sea, distance: d, sun: sun, fog: fogColor)
                    for r in row..<top { put(col, r, color) }
                    top = row
                }
                d += Swift.max(0.25, d * 0.004)
            }
            // Sky above whatever terrain was drawn.
            for r in 0..<top {
                let t = Swift.min(1, Swift.max(0, (horizon - Double(r)) / Double(height)))
                put(col, r, (fogColor.0 * (1 - t) + 0.25 * t, fogColor.1 * (1 - t) + 0.42 * t, fogColor.2 * (1 - t) + 0.80 * t))
            }
        }
        return Preview.Image(width: width, height: height, rgb: rgb)
    }

    private static func shade(_ h: Grid, gx: Double, gy: Double, px: Double, py: Double, ground: Double, sea: Double, distance: Double,
                              sun: (Double, Double, Double), fog: (Double, Double, Double)) -> (Double, Double, Double) {
        var albedo: (Double, Double, Double)
        var n = (0.0, 0.0, 1.0)
        let depthBelow = sea - ground
        if depthBelow > 0 {
            let t = Swift.min(1, depthBelow / 400)
            albedo = (0.20 + (0.02 - 0.20) * t, 0.55 + (0.10 - 0.55) * t, 0.60 + (0.25 - 0.60) * t)
        } else {
            let e = 1.0
            var hx = Double(h.sample(x: gx + e, y: gy) - h.sample(x: gx - e, y: gy)) / (2 * e * h.cellSize)
            var hy = Double(h.sample(x: gx, y: gy + e) - h.sample(x: gx, y: gy - e)) / (2 * e * h.cellSize)
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
            let grass = (0.20, 0.36, 0.14), soil = (0.45, 0.25, 0.15), rock = (0.32, 0.27, 0.24)
            albedo = mix(grass, soil, smoothstep(0.15, 0.30, slope))
            albedo = mix(albedo, rock, smoothstep(0.35, 0.55, slope))
            albedo = mix((0.76, 0.70, 0.55), albedo, smoothstep(1.5, 6, above)) // beach
        }
        let diffuse = Swift.max(0, n.0 * sun.0 + n.1 * sun.1 + n.2 * sun.2)
        let ambient = 0.5 * (0.5 + 0.5 * n.2)
        let light = (2.2 * diffuse + 0.45 * ambient, 2.09 * diffuse + 0.6 * ambient, 1.87 * diffuse + 0.85 * ambient)
        var color = (albedo.0 * light.0, albedo.1 * light.1, albedo.2 * light.2)
        let f = 0.85 * (1 - exp(-distance / 60_000))
        color = mix(color, fog, f)
        return color
    }

    /// Metre-scale relief the 10–80 m base grid can't hold: hummocks, lava-flow lumps.
    /// Stand-in for the runtime detail amplification in the plan.
    static func detailNoise(_ x: Double, _ y: Double) -> Double {
        3.0 * Noise.fbm(x / 60, y / 60, seed: 0xD37A_11, octaves: 4) + 0.6 * Noise.fbm(x / 8, y / 8, seed: 0xD37A_12, octaves: 3)
    }

    private static func detailFade(_ distance: Double) -> Double {
        1 - smoothstep(1500, 4000, distance)
    }

    private static func detail(_ x: Double, _ y: Double, ground: Double, sea: Double, distance: Double) -> Double {
        let fade = detailFade(distance)
        guard fade > 0, ground > sea + 1 else { return 0 }
        return detailNoise(x, y) * fade * smoothstep(sea + 1, sea + 6, ground)
    }

    private static func tonemap(_ v: Double) -> Double {
        let white = 2.5
        return v * (1 + v / (white * white)) / (1 + v)
    }

    private static func normalize(_ v: (Double, Double, Double)) -> (Double, Double, Double) {
        let l = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
        return (v.0 / l, v.1 / l, v.2 / l)
    }

    private static func mix(_ a: (Double, Double, Double), _ b: (Double, Double, Double), _ t: Double) -> (Double, Double, Double) {
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
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

    public static func automatic(height h: Grid, drainageArea area: Grid, seaLevel: Float = 0) -> [Named] {
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
        func camera(from i: Int, to t: Int, pitchOffset: Double = 0) -> GroundView.Camera {
            let (x, y) = position(i), (tx, ty) = position(t)
            let z = Double(h.values[i]) + 1.7
            let dist = Swift.max(1, ((tx - x) * (tx - x) + (ty - y) * (ty - y)).squareRoot())
            let pitch = atan((Double(Swift.max(h.values[t], seaLevel)) - z) / dist) * 0.5 + pitchOffset
            return GroundView.Camera(x: x, y: y, z: z, yaw: atan2(tx - x, ty - y), pitch: pitch)
        }

        var views: [Named] = []
        if coast >= 0 {
            views.append(Named(name: "summit", camera: camera(from: summit, to: coast)))
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
            if ridge >= 0 { views.append(Named(name: "ridge", camera: camera(from: ridge, to: valley))) }
        }
        if coast >= 0 {
            views.append(Named(name: "coast", camera: camera(from: coast, to: summit, pitchOffset: 0.02)))
        }
        return views
    }
}
