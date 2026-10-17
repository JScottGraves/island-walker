import simd

/// One vertex of a plant mesh, laid out as the Metal `FloraVertex` (48 bytes, no padding).
/// Positions are metres at the plant's reference height, z up, base of the trunk at the
/// origin.
public struct FloraVertex: Sendable, Equatable {
    public var x: Float, y: Float, z: Float
    /// Shading normal. On foliage cards it is blended towards the crown's ellipsoid normal,
    /// so a crown shades as one soft volume rather than a heap of flat cards.
    public var nx: Float, ny: Float, nz: Float
    public var u: Float, v: Float
    /// Ambient occlusion baked from depth inside the crown (1 open, ~0.3 deep inside).
    public var ao: Float
    /// Wind weights: how far the branch carrying this vertex sways (0 at the trunk base,
    /// 1 at the crown's outer tips), and how much a leaf flutters on its own.
    public var sway: Float, flutter: Float
    /// Index into `FloraMaterial.table`.
    public var material: Float

    public init(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float, u: Float, v: Float,
                ao: Float, sway: Float, flutter: Float, material: Float) {
        self.x = x; self.y = y; self.z = z
        self.nx = nx; self.ny = ny; self.nz = nz
        self.u = u; self.v = v
        self.ao = ao; self.sway = sway; self.flutter = flutter; self.material = material
    }

    public var position: SIMD3<Float> { SIMD3(x, y, z) }
    public var normal: SIMD3<Float> { SIMD3(nx, ny, nz) }
}

/// One level of detail of one plant variant.
public struct PlantMesh: Sendable {
    public var vertices: [FloraVertex] = []
    public var indices: [UInt32] = []

    public var triangleCount: Int { indices.count / 3 }

    /// Axis-aligned bounds of the vertex positions.
    public var bounds: (min: SIMD3<Float>, max: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for v in vertices {
            lo = simd_min(lo, v.position)
            hi = simd_max(hi, v.position)
        }
        return (lo, hi)
    }
}

/// The crown as an ellipsoid: foliage normals lean towards its surface normal and AO
/// darkens towards its centre.
struct Crown {
    var centre: SIMD3<Float>
    var radii: SIMD3<Float>

    /// 0 at the centre, 1 on the surface.
    func depth(_ p: SIMD3<Float>) -> Float {
        simd_length((p - centre) / radii)
    }

    func normal(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let q = (p - centre) / (radii * radii)
        let l = simd_length(q)
        return l > 1e-6 ? q / l : SIMD3(0, 0, 1)
    }
}

/// A frame along a branch or trunk: position, the axis's direction and a perpendicular
/// reference, and the tube radius there.
struct Ring {
    var position: SIMD3<Float>
    var direction: SIMD3<Float>
    var side: SIMD3<Float>
    var radius: Float
}

/// Builds plant meshes from tubes, cards and ribbons. Wind weights and AO are filled in
/// by `finish`.
struct MeshBuilder {
    var mesh = PlantMesh()
    let height: Float
    var crown: Crown

    init(height: Float, crown: Crown) {
        self.height = height
        self.crown = crown
    }

    mutating func vertex(_ p: SIMD3<Float>, _ n: SIMD3<Float>, _ uv: SIMD2<Float>, material: FloraMaterial.ID,
                         flutter: Float = 0, sway: Float = -1) -> UInt32 {
        let index = UInt32(mesh.vertices.count)
        mesh.vertices.append(FloraVertex(x: p.x, y: p.y, z: p.z, nx: n.x, ny: n.y, nz: n.z, u: uv.x, v: uv.y,
                                         ao: 1, sway: sway, flutter: flutter, material: Float(material.rawValue)))
        return index
    }

    mutating func triangle(_ a: UInt32, _ b: UInt32, _ c: UInt32) {
        mesh.indices.append(contentsOf: [a, b, c])
    }

    /// A generalised cylinder through `rings` with `sides` faces around, its texture wrapped
    /// once around and repeating every `repeatLength` metres along it. Open at both ends
    /// (the base sits below the ground, the tip inside a fork or a leaf cluster).
    mutating func tube(_ rings: [Ring], sides: Int, material: FloraMaterial.ID, repeatLength: Float) {
        guard rings.count >= 2 else { return }
        let base = UInt32(mesh.vertices.count)
        var along: Float = 0
        for (k, ring) in rings.enumerated() {
            if k > 0 { along += simd_distance(ring.position, rings[k - 1].position) }
            let side = simd_normalize(ring.side - ring.direction * simd_dot(ring.side, ring.direction))
            let other = simd_cross(ring.direction, side)
            for s in 0...sides {
                let a = Float(s) / Float(sides) * 2 * .pi
                let n: SIMD3<Float> = side * cos(a) + other * sin(a)
                _ = vertex(ring.position + n * ring.radius, n, SIMD2(Float(s) / Float(sides), along / repeatLength),
                           material: material)
            }
        }
        let row = UInt32(sides + 1)
        for k in 0..<UInt32(rings.count - 1) {
            for s in 0..<UInt32(sides) {
                let a = base + k * row + s, b = a + 1, c = a + row, d = c + 1
                triangle(a, b, d)
                triangle(a, d, c)
            }
        }
    }

    /// A foliage card centred at `centre`, spanning ±`right` and ±`up`. Its normal leans
    /// `bend` of the way towards the crown's surface normal (soft, volumetric crowns).
    mutating func card(centre: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>, material: FloraMaterial.ID,
                       bend: Float = 0.85, sway: Float = -1) {
        let flat = simd_normalize(simd_cross(right, up))
        let corners: [(Float, Float)] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        let base = UInt32(mesh.vertices.count)
        for (cx, cy) in corners {
            let p: SIMD3<Float> = centre + right * cx + up * cy
            var n: SIMD3<Float> = crown.normal(p)
            // Facing the crown's outside either way: cards are two-sided.
            let f: SIMD3<Float> = simd_dot(flat, n) < 0 ? -flat : flat
            n = simd_normalize(f * (1 - bend) + n * bend)
            _ = vertex(p, n, SIMD2((cx + 1) / 2, (cy + 1) / 2), material: material, flutter: 1, sway: sway)
        }
        triangle(base, base + 1, base + 2)
        triangle(base, base + 2, base + 3)
    }

    /// Fills in what the parts left open: sway (by height and distance from the trunk where
    /// not set) and AO from depth inside the crown (and near the ground).
    mutating func finish(aoFloor: Float = 0.45) {
        let h = Swift.max(height, 0.1)
        let reach = Swift.max(crown.radii.x, crown.radii.y, 0.1)
        for i in mesh.vertices.indices {
            var v = mesh.vertices[i]
            let p = v.position
            if v.sway < 0 {
                let up = Swift.max(p.z / h, 0)
                let out = simd_length(SIMD2(p.x, p.y)) / reach
                v.sway = Swift.min(1, up * up * 0.6 + out * out * 0.6)
            }
            let depth = crown.depth(p)
            var ao = aoFloor + (1 - aoFloor) * smoothstep(0.15, 1.05, depth)
            // Below the crown, the canopy overhead hides much of the sky.
            let below = crown.centre.z - crown.radii.z
            if p.z < below {
                let under = 1 - smoothstep(0.6, 1.2, simd_length(SIMD2(p.x, p.y) / SIMD2(crown.radii.x, crown.radii.y)))
                ao *= 1 - 0.35 * under * (1 - 0.5 * smoothstep(0, below, p.z))
            }
            // Ground contact: the lowest metre of trunk sees less sky.
            ao *= 0.65 + 0.35 * smoothstep(-0.2, Swift.min(1.5, 0.3 * h), p.z)
            v.ao = ao
            mesh.vertices[i] = v
        }
    }
}

func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = Swift.min(1, Swift.max(0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)
}

/// Any vector perpendicular to `d` (unit).
func perpendicular(_ d: SIMD3<Float>) -> SIMD3<Float> {
    let a: SIMD3<Float> = abs(d.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
    return simd_normalize(simd_cross(d, a))
}

/// Rotates `v` about the unit axis `axis` by `angle` radians (Rodrigues).
func rotate(_ v: SIMD3<Float>, about axis: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
    let c = cos(angle), s = sin(angle)
    let k: SIMD3<Float> = simd_cross(axis, v)
    let d: Float = simd_dot(axis, v) * (1 - c)
    return v * c + k * s + axis * d
}
