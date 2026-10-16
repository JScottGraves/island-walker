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
            r = r / (1 + r); g = g / (1 + g); b = b / (1 + b)
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
                let ground = Double(h.sample(x: gx, y: gy))
                let surface = Swift.max(ground, sea)
                let depth = d * depthScale
                let screenY = horizon - (surface - c.z) * focal / depth
                let row = Swift.max(0, Int(screenY.rounded(.up)))
                if row < top {
                    let color = shade(h, gx: gx, gy: gy, ground: ground, sea: sea, distance: d, sun: sun, fog: fogColor)
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

    private static func shade(_ h: Grid, gx: Double, gy: Double, ground: Double, sea: Double, distance: Double,
                              sun: (Double, Double, Double), fog: (Double, Double, Double)) -> (Double, Double, Double) {
        var albedo: (Double, Double, Double)
        var n = (0.0, 0.0, 1.0)
        let depthBelow = sea - ground
        if depthBelow > 0 {
            let t = Swift.min(1, depthBelow / 400)
            albedo = (0.20 + (0.02 - 0.20) * t, 0.55 + (0.10 - 0.55) * t, 0.60 + (0.25 - 0.60) * t)
        } else {
            let e = 1.0
            let hx = Double(h.sample(x: gx + e, y: gy) - h.sample(x: gx - e, y: gy)) / (2 * e * h.cellSize)
            let hy = Double(h.sample(x: gx, y: gy + e) - h.sample(x: gx, y: gy - e)) / (2 * e * h.cellSize)
            n = normalize((-hx, -hy, 1))
            let slope = 1 - n.2
            let above = ground - sea
            if above < 4 {
                albedo = (0.76, 0.70, 0.55) // beach
            } else {
                let grass = (0.20, 0.36, 0.14), soil = (0.45, 0.25, 0.15), rock = (0.32, 0.27, 0.24)
                albedo = mix(grass, soil, smoothstep(0.15, 0.30, slope))
                albedo = mix(albedo, rock, smoothstep(0.35, 0.55, slope))
            }
        }
        let diffuse = Swift.max(0, n.0 * sun.0 + n.1 * sun.1 + n.2 * sun.2)
        let ambient = 0.5 * (0.5 + 0.5 * n.2)
        let light = (2.2 * diffuse + 0.45 * ambient, 2.09 * diffuse + 0.6 * ambient, 1.87 * diffuse + 0.85 * ambient)
        var color = (albedo.0 * light.0, albedo.1 * light.1, albedo.2 * light.2)
        let f = 1 - exp(-distance / 30_000)
        color = mix(color, fog, f)
        return color
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

/// Picks informative viewpoints automatically: the summit, a river valley and the coast.
public enum Viewpoints {
    public struct Named {
        public var name: String
        public var camera: GroundView.Camera
    }

    public static func automatic(height h: Grid, drainageArea area: Grid, seaLevel: Float = 0) -> [Named] {
        var summit = 0
        for i in 0..<h.count where h.values[i] > h.values[summit] { summit = i }
        let maxH = h.values[summit]
        let (sx, sy) = position(h, summit)

        // Valley: the wettest mid-elevation cell, looking up toward the summit.
        var valley = -1, valleyArea: Float = 0
        // Coast: the biggest river just above sea level, looking inland.
        var coast = -1, coastArea: Float = 0
        for i in 0..<h.count {
            let z = h.values[i] - seaLevel
            if z > 0.25 * maxH && z < 0.5 * maxH && area.values[i] > valleyArea { valley = i; valleyArea = area.values[i] }
            if z > 3 && z < 30 && area.values[i] > coastArea { coast = i; coastArea = area.values[i] }
        }

        func eye(_ i: Int, lookingAt tx: Double, _ ty: Double, pitch: Double) -> GroundView.Camera {
            let (x, y) = position(h, i)
            return GroundView.Camera(x: x, y: y, z: Double(h.values[i]) + 1.7, yaw: atan2(tx - x, ty - y), pitch: pitch)
        }

        // From the summit look toward the far side of the domain centre.
        let cx = Double(h.width) * h.cellSize / 2, cy = Double(h.height) * h.cellSize / 2
        var views = [Named(name: "summit", camera: eye(summit, lookingAt: 2 * cx - sx, 2 * cy - sy, pitch: -0.12))]
        if valley >= 0 { views.append(Named(name: "valley", camera: eye(valley, lookingAt: sx, sy, pitch: 0.06))) }
        if coast >= 0 { views.append(Named(name: "coast", camera: eye(coast, lookingAt: sx, sy, pitch: 0.03))) }
        return views
    }

    private static func position(_ h: Grid, _ i: Int) -> (Double, Double) {
        ((Double(i % h.width) + 0.5) * h.cellSize, (Double(i / h.width) + 0.5) * h.cellSize)
    }
}
