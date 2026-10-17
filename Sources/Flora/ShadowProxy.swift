import simd

/// The opaque stand-in a plant variant casts sun shadows with (docs/flora.md §8): a closed
/// low-poly trunk prism and a crown of a few overlapping lobes, about 60–80 triangles in
/// all. The crown is shrunk to ~70% of the leaves' spread, so the outer, sunlit leaves
/// stay outside their own proxy and shadow edges break up. Positions are metres at the
/// plant's reference height, like `PlantMesh`; faces wind counter-clockwise seen from
/// outside.
public struct ShadowProxy: Sendable {
    public struct Part: Sendable {
        public var positions: [SIMD3<Float>] = []
        public var indices: [UInt32] = []
    }

    /// Bark: the renderer lets no light through it.
    public var trunk = Part()
    /// Foliage: the renderer lets some light through, in patches.
    public var crown = Part()

    public var triangleCount: Int { (trunk.indices.count + crown.indices.count) / 3 }
}

extension Flora {
    /// Share of the leaves' spread a crown lobe covers.
    static let proxyShrink: Float = 0.7

    /// The shadow proxy of one variant, fitted to its LOD0 mesh.
    public static func shadowProxy(_ plant: Plant, variant: Int) -> ShadowProxy {
        let mesh = mesh(plant, variant: variant, lod: 0)
        var leaves: [SIMD3<Float>] = [], bark: [SIMD3<Float>] = []
        for v in mesh.vertices {
            if FloraMaterial.table[Int(v.material)].card { leaves.append(v.position) } else { bark.append(v.position) }
        }
        var proxy = ShadowProxy()
        let flat = plant == .loulu || plant == .hapuu
        // Lobes as (centre, radii), in the order they are fitted.
        var lobes: [(SIMD3<Float>, SIMD3<Float>)] = []
        if flat {
            // One disc for the whole frond crown.
            var (c, r) = fitEllipsoid(leaves)
            r.z = Swift.min(r.z, 0.4 * Swift.min(r.x, r.y))
            lobes.append((c, r))
            bicone(&proxy.crown, centre: c, radii: r, sides: 10)
        } else {
            for cluster in kMeans(leaves, k: 3) where cluster.count >= 4 {
                let (c, r) = fitEllipsoid(cluster)
                lobes.append((c, r))
                icosahedron(&proxy.crown, centre: c, radii: r)
            }
        }

        // Trunk: from below the ground (drawn trunks reach 0.6 m down) to the lowest lobe's
        // centre, following the trunk's lean. Its radius is the base's.
        let top = lobes.map { $0.0.z }.min() ?? Float(plant.referenceHeight) * 0.5
        func slice(_ z0: Float, _ z1: Float) -> [SIMD3<Float>] {
            let s = bark.filter { $0.z >= z0 && $0.z <= z1 }
            return s.isEmpty ? [SIMD3(0, 0, (z0 + z1) / 2)] : s
        }
        let base = slice(0, Swift.max(0.4, 0.05 * top))
        let baseCentre = centroid(base)
        let radius = Swift.max(0.03, base.map { simd_length(SIMD2($0.x - baseCentre.x, $0.y - baseCentre.y)) }
            .reduce(0, +) / Float(base.count))
        let topCentre = centroid(slice(top - Swift.max(0.5, 0.1 * top), top))
        prism(&proxy.trunk, from: SIMD3(baseCentre.x, baseCentre.y, -0.6), to: SIMD3(topCentre.x, topCentre.y, top),
              radius: radius, sides: 5)
        return proxy
    }

    static func centroid(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
        points.reduce(SIMD3<Float>(repeating: 0), +) / Float(Swift.max(points.count, 1))
    }

    /// Axis-aligned lobe over `points`: their centroid, and per axis the 90th percentile
    /// of their offsets from it, shrunk by `proxyShrink`.
    static func fitEllipsoid(_ points: [SIMD3<Float>]) -> (SIMD3<Float>, SIMD3<Float>) {
        let c = centroid(points)
        var r = SIMD3<Float>(repeating: 0)
        for axis in 0..<3 {
            let offsets = points.map { abs($0[axis] - c[axis]) }.sorted()
            let p90 = offsets.isEmpty ? 0 : offsets[Swift.min(offsets.count - 1, offsets.count * 9 / 10)]
            r[axis] = Swift.max(0.1, p90 * proxyShrink)
        }
        return (c, r)
    }

    /// k-means, seeded deterministically with the point farthest from those chosen so far.
    static func kMeans(_ points: [SIMD3<Float>], k: Int) -> [[SIMD3<Float>]] {
        guard !points.isEmpty else { return [] }
        var centres = [centroid(points)]
        while centres.count < k {
            let far = points.max { a, b in
                centres.map { simd_distance_squared($0, a) }.min()! < centres.map { simd_distance_squared($0, b) }.min()!
            }!
            centres.append(far)
        }
        // The overall centroid only seeded the farthest-point picks.
        if k > 1 { centres.removeFirst(); centres.append(points[points.count / 2]) }
        var clusters: [[SIMD3<Float>]] = []
        for _ in 0..<12 {
            clusters = Array(repeating: [], count: centres.count)
            for p in points {
                var best = 0
                for j in 1..<centres.count where simd_distance_squared(p, centres[j]) < simd_distance_squared(p, centres[best]) {
                    best = j
                }
                clusters[best].append(p)
            }
            for j in centres.indices where !clusters[j].isEmpty { centres[j] = centroid(clusters[j]) }
        }
        return clusters
    }

    /// Appends a convex piece: `faces` index `points`; each face is turned to wind
    /// counter-clockwise seen from outside (away from the piece's centre).
    static func appendConvex(_ part: inout ShadowProxy.Part, _ points: [SIMD3<Float>], _ faces: [(Int, Int, Int)]) {
        let base = UInt32(part.positions.count)
        let centre = centroid(points)
        part.positions += points
        for (a, b, c) in faces {
            let n: SIMD3<Float> = simd_cross(points[b] - points[a], points[c] - points[a])
            let outward = simd_dot(n, points[a] - centre) >= 0
            part.indices += outward ? [base + UInt32(a), base + UInt32(b), base + UInt32(c)]
                                    : [base + UInt32(a), base + UInt32(c), base + UInt32(b)]
        }
    }

    /// A 20-triangle icosahedron stretched to an ellipsoid.
    static func icosahedron(_ part: inout ShadowProxy.Part, centre: SIMD3<Float>, radii: SIMD3<Float>) {
        let t: Float = (1 + Float(5).squareRoot()) / 2
        let raw: [SIMD3<Float>] = [
            SIMD3(-1, t, 0), SIMD3(1, t, 0), SIMD3(-1, -t, 0), SIMD3(1, -t, 0),
            SIMD3(0, -1, t), SIMD3(0, 1, t), SIMD3(0, -1, -t), SIMD3(0, 1, -t),
            SIMD3(t, 0, -1), SIMD3(t, 0, 1), SIMD3(-t, 0, -1), SIMD3(-t, 0, 1),
        ]
        let points = raw.map { centre + simd_normalize($0) * radii }
        let faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                     (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                     (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                     (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        appendConvex(&part, points, faces)
    }

    /// A double cone: a `sides`-gon rim with an apex above and below (2 × sides triangles).
    static func bicone(_ part: inout ShadowProxy.Part, centre: SIMD3<Float>, radii: SIMD3<Float>, sides: Int) {
        var points = [centre + SIMD3(0, 0, radii.z), centre - SIMD3(0, 0, radii.z)]
        for s in 0..<sides {
            let a = Float(s) / Float(sides) * 2 * .pi
            points.append(centre + SIMD3(cos(a) * radii.x, sin(a) * radii.y, 0))
        }
        var faces: [(Int, Int, Int)] = []
        for s in 0..<sides {
            let a = 2 + s, b = 2 + (s + 1) % sides
            faces += [(0, a, b), (1, b, a)]
        }
        appendConvex(&part, points, faces)
    }

    /// A `sides`-sided prism from `from` to `to` with horizontal caps (4 × sides − 4 triangles).
    static func prism(_ part: inout ShadowProxy.Part, from: SIMD3<Float>, to: SIMD3<Float>, radius: Float, sides: Int) {
        var points: [SIMD3<Float>] = []
        for end in [from, to] {
            for s in 0..<sides {
                let a = Float(s) / Float(sides) * 2 * .pi
                points.append(end + SIMD3(cos(a), sin(a), 0) * radius)
            }
        }
        var faces: [(Int, Int, Int)] = []
        for s in 0..<sides {
            let a = s, b = (s + 1) % sides
            faces += [(a, b, sides + b), (a, sides + b, sides + a)]
        }
        for s in 1..<(sides - 1) {
            faces += [(0, s + 1, s), (sides, sides + s, sides + s + 1)]
        }
        appendConvex(&part, points, faces)
    }
}
