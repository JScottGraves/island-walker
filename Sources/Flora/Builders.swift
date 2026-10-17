import IslandCore
import simd

extension SplitMix64 {
    mutating func float(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * Float(unit()) }
    mutating func int(_ r: ClosedRange<Int>) -> Int { r.lowerBound + Int(next() % UInt64(r.count)) }
    /// A random unit vector.
    mutating func direction() -> SIMD3<Float> {
        let z = float(-1, 1), a = float(0, 2 * .pi)
        let s = (1 - z * z).squareRoot()
        return SIMD3(s * cos(a), s * sin(a), z)
    }
}

/// Branching trees and shrubs (ʻōhiʻa, koa, ʻaʻaliʻi): a sympodial skeleton of crooked
/// segments whose tips reach for the crown's envelope, with leaf-cluster cards at the tips.
/// LOD1 draws the same skeleton down to `lod1Order` and covers each of those branches'
/// leaf tufts with a few large cards, so the silhouette matches LOD0.
struct BranchingTree {
    var height: Float
    /// Stems rising from the base, and how far they lean apart (radians).
    var stems: ClosedRange<Int>
    var stemLean: Float
    /// Trunk radius at the base, metres.
    var trunkRadius: Float
    /// Share of the height at which each stem first forks.
    var forkHeight: ClosedRange<Float>
    var children: ClosedRange<Int>
    /// Angle between a child and its parent's direction (radians).
    var branchAngle: ClosedRange<Float>
    /// A child reaches this share of the way from its start to the crown's envelope.
    var reach: ClosedRange<Float>
    /// Branching orders below the trunk.
    var orders: Int
    var segments: Int
    /// Random bend per segment (radians) and pull towards the sky per segment.
    var crooked: Float
    var upward: Float
    /// Crown ellipsoid: centre height and radii as shares of the height.
    var crownCentre: Float
    var crownRadii: SIMD3<Float>
    /// Leaf-cluster card half-size (metres) and cards per tip.
    var card: Float
    var cardsPerTip: ClosedRange<Int>
    /// Cards per ring along terminal branches and along the order before them.
    var alongTerminal = 2
    var alongOuter = 1
    var leaves: FloraMaterial.ID
    var flowers: FloraMaterial.ID?
    var flowerChance: Float = 0
    var bark: FloraMaterial.ID
    /// Faces around a branch, by order (trunk first), at LOD0 and LOD1.
    var sides: [Int]
    var sides1: [Int]
    var lod1Order = 2
    /// Large LOD1 cards: half-size multiple, and LOD0 cards per LOD1 card.
    var lod1Card: Float = 2.4
    var lod1Ratio = 6
    var barkRepeat: Float = 1.5

    struct Branch {
        var rings: [Ring]
        var order: Int
        var parent: Int
        var terminal = true
    }

    struct Card {
        var centre: SIMD3<Float>
        var right: SIMD3<Float>
        var up: SIMD3<Float>
        var material: FloraMaterial.ID
        /// The branch at `lod1Order` (or shallower terminal one) this tuft belongs to.
        var cluster: Int
    }

    func build(seed: UInt64, lod: Int) -> PlantMesh {
        var rng = SplitMix64(seed: seed)
        let crown = Crown(centre: SIMD3(0, 0, crownCentre * height), radii: crownRadii * height)
        var branches: [Branch] = []
        var cards: [Card] = []

        // Distance from p along d to the crown's envelope (or a little if already outside).
        func toEnvelope(_ p: SIMD3<Float>, _ d: SIMD3<Float>) -> Float {
            let o = (p - crown.centre) / crown.radii, v = d / crown.radii
            let a: Float = simd_dot(v, v), b: Float = 2 * simd_dot(o, v), c: Float = simd_dot(o, o) - 1
            let disc = b * b - 4 * a * c
            if disc <= 0 { return 0.15 * height }
            let t = (-b + disc.squareRoot()) / (2 * a)
            return Swift.max(t, 0.05 * height)
        }

        // A leaf-cluster card near `p`, facing roughly out of the crown.
        func addCard(at p: SIMD3<Float>, along d: SIMD3<Float>, size: Float, cluster: Int, rng: inout SplitMix64) {
            let facing = simd_normalize(crown.normal(p) * 0.8 + rng.direction())
            let right = rotate(perpendicular(facing), about: facing, by: rng.float(0, 2 * .pi))
            let up = simd_cross(facing, right)
            let s = card * size * rng.float(0.8, 1.2)
            let centre = p + rng.direction() * (card * 0.6)
            cards.append(Card(centre: centre, right: right * s, up: up * s, material: leaves, cluster: cluster))
        }

        // One branch: a crooked path from `start` heading along `dir`.
        func grow(start: SIMD3<Float>, dir d0: SIMD3<Float>, length: Float, radius r0: Float, tipRadius r1: Float,
                  order: Int, parent: Int, cluster c0: Int, rng: inout SplitMix64) {
            let n = Swift.max(2, order == 0 ? segments + 2 : segments - Swift.min(order, 2))
            var rings: [Ring] = []
            var p = start, d = d0
            var side = perpendicular(d)
            for k in 0...n {
                let t = Float(k) / Float(n)
                rings.append(Ring(position: p, direction: d, side: side, radius: r0 + (r1 - r0) * t))
                if k == n { break }
                p += d * (length / Float(n))
                let bend: SIMD3<Float> = rng.direction() * crooked + SIMD3(0, 0, upward)
                d = simd_normalize(d + bend)
                side = simd_normalize(side - d * simd_dot(side, d))
            }
            let index = branches.count
            branches.append(Branch(rings: rings, order: order, parent: parent))
            let cluster = order == lod1Order ? index : c0
            // Leaves along the outer branches too, so the crown reads as foliage, not twigs.
            if order >= orders - 1 && order > 0 {
                let leafCluster = order <= lod1Order ? index : cluster
                for ring in rings.dropFirst() {
                    for _ in 0..<(order == orders ? alongTerminal : alongOuter) {
                        addCard(at: ring.position, along: ring.direction, size: 0.9, cluster: leafCluster, rng: &rng)
                    }
                }
            }
            if order < orders {
                branches[index].terminal = false
                let count = rng.int(children)
                let spin = rng.float(0, 2 * .pi)
                for c in 0..<count {
                    // Sympodial: the axis ends in a fork; children spread around it.
                    let around = spin + Float(c) * 2.39996 + rng.float(-0.4, 0.4)
                    let axis = rotate(side, about: d, by: around)
                    let angle = rng.float(branchAngle.lowerBound, branchAngle.upperBound)
                    var cd = simd_normalize(rotate(d, about: axis, by: angle) + SIMD3(0, 0, upward * 2))
                    if order == 0 && cd.z < 0.1 { cd = simd_normalize(cd + SIMD3(0, 0, 0.3)) }
                    let len = Swift.max(0.08 * height / Float(order + 1),
                                        toEnvelope(p, cd) * rng.float(reach.lowerBound, reach.upperBound))
                    let cr = r1 * rng.float(0.65, 0.85)
                    grow(start: p, dir: cd, length: len, radius: cr, tipRadius: cr * 0.55,
                         order: order + 1, parent: index, cluster: cluster, rng: &rng)
                }
            } else {
                // Leaf tuft at the tip.
                let tuftCluster = order <= lod1Order ? index : cluster
                for _ in 0..<rng.int(cardsPerTip) {
                    addCard(at: p + d * (card * 0.4), along: d, size: 1, cluster: tuftCluster, rng: &rng)
                }
                if let flowers, rng.float(0, 1) < flowerChance {
                    let facing = simd_normalize(crown.normal(p) + SIMD3(0, 0, 0.6))
                    let right = rotate(perpendicular(facing), about: facing, by: rng.float(0, 2 * .pi))
                    let s = card * 0.45
                    cards.append(Card(centre: p + facing * (card * 0.5), right: right * s,
                                      up: simd_cross(facing, right) * s, material: flowers, cluster: tuftCluster))
                }
            }
        }

        let stemCount = rng.int(stems)
        for s in 0..<stemCount {
            let lean = stemCount == 1 ? stemLean * 0.3 : stemLean
            let azimuth = Float(s) / Float(stemCount) * 2 * .pi + rng.float(-0.5, 0.5)
            let tilt = rng.float(0.3, 1.0) * lean
            let dir = simd_normalize(SIMD3(sin(tilt) * cos(azimuth), sin(tilt) * sin(azimuth), cos(tilt)))
            let base: SIMD3<Float> = SIMD3(cos(azimuth), sin(azimuth), 0) * (stemCount == 1 ? 0 : trunkRadius * 0.6)
                + SIMD3(0, 0, -0.6)
            let radius = trunkRadius * (stemCount == 1 ? 1 : 0.75) * rng.float(0.85, 1.1)
            let length = height * rng.float(forkHeight.lowerBound, forkHeight.upperBound) + 0.6
            grow(start: base, dir: dir, length: length, radius: radius * 1.15, tipRadius: radius * 0.8,
                 order: 0, parent: -1, cluster: -1, rng: &rng)
        }

        var b = MeshBuilder(height: height, crown: crown)
        for (index, branch) in branches.enumerated() {
            if lod > 0 && branch.order > lod1Order { continue }
            let table = lod > 0 ? sides1 : sides
            let faces = table[Swift.min(branch.order, table.count - 1)]
            var rings = branch.rings
            if lod > 0 && rings.count > 3 {
                rings = stride(from: 0, to: rings.count, by: 2).map { rings[$0] }
                if let last = branch.rings.last, rings.last?.position != last.position { rings.append(last) }
            }
            // Start each child inside its parent, so forks don't show a gap.
            if branch.parent >= 0, var first = rings.first {
                first.position -= first.direction * first.radius
                rings[0] = first
            }
            _ = index
            b.tube(rings, sides: faces, material: bark, repeatLength: barkRepeat)
        }
        if lod == 0 {
            for c in cards { b.card(centre: c.centre, right: c.right, up: c.up, material: c.material) }
        } else {
            // A few large cards over each LOD1 branch's tufts.
            var byCluster: [Int: [Card]] = [:]
            for c in cards { byCluster[c.cluster, default: []].append(c) }
            for key in byCluster.keys.sorted() {
                guard let group = byCluster[key] else { continue }
                let count = Swift.max(1, (group.count + lod1Ratio / 2) / lod1Ratio)
                for k in 0..<count {
                    let c = group[(k * group.count) / count]
                    let flower = c.material != leaves
                    b.card(centre: c.centre, right: c.right * (flower ? 1.2 : lod1Card),
                           up: c.up * (flower ? 1.2 : lod1Card), material: c.material)
                }
            }
        }
        b.finish()
        return b.mesh
    }
}

/// Loulu: a slightly curved, smooth grey trunk under a crown of stiff, pleated fan leaves
/// on long petioles, sometimes with a short skirt of dead leaves.
struct Palm {
    var height: Float
    var trunkRadius: Float = 0.14
    var leaves: ClosedRange<Int> = 20...28
    var leafRadius: ClosedRange<Float> = 0.8...0.95
    var petiole: Float = 1.1
    var deadChance: Float = 0.6

    func build(seed: UInt64, lod: Int) -> PlantMesh {
        var rng = SplitMix64(seed: seed)
        // The height is to the top of the crown, which reaches ~1.5 m above the trunk.
        let crownTop = height
        let crown = Crown(centre: SIMD3(0, 0, crownTop - 1.4), radii: SIMD3(2.2, 2.2, 1.6))
        var b = MeshBuilder(height: height, crown: crown)

        // Trunk: a gentle curve, swollen at the base.
        let lean = rng.float(0.02, 0.09), azimuth = rng.float(0, 2 * .pi)
        let bendDir = SIMD3<Float>(cos(azimuth), sin(azimuth), 0)
        let trunkTop = crownTop - 2.1
        let steps = lod == 0 ? 12 : 4
        var rings: [Ring] = []
        var top = SIMD3<Float>(0, 0, 0), topDir = SIMD3<Float>(0, 0, 1)
        for k in 0...steps {
            let t = Float(k) / Float(steps)
            let z = -0.6 + (trunkTop + 0.6) * t
            let offset = bendDir * (lean * z * t * 1.5)
            let dir = simd_normalize(SIMD3(0, 0, 1) + bendDir * (lean * 3 * t))
            let swell: Float = 1 + 0.6 * (1 - smoothstep(0, 0.12, t))
            let r = trunkRadius * swell * (1 - 0.15 * t)
            rings.append(Ring(position: SIMD3(offset.x, offset.y, z), direction: dir, side: SIMD3(1, 0, 0), radius: r))
            top = SIMD3(offset.x, offset.y, z)
            topDir = dir
        }
        b.tube(rings, sides: lod == 0 ? 9 : 5, material: .louluTrunk, repeatLength: 0.9)
        // The crown shaft: leaf bases wrapped round the top.
        let shaft = [Ring(position: top - topDir * 0.2, direction: topDir, side: SIMD3(1, 0, 0), radius: trunkRadius * 1.15),
                     Ring(position: top + topDir * 0.5, direction: topDir, side: SIMD3(1, 0, 0), radius: trunkRadius * 1.35),
                     Ring(position: top + topDir * 0.75, direction: topDir, side: SIMD3(1, 0, 0), radius: trunkRadius * 0.4)]
        b.tube(shaft, sides: lod == 0 ? 9 : 5, material: .louluTrunk, repeatLength: 0.9)

        let count = lod == 0 ? rng.int(leaves) : 11
        let hub = top + topDir * 0.55
        let dead = rng.float(0, 1) < deadChance ? rng.int(3...6) : 0
        for k in 0..<(count + (lod == 0 ? dead : Swift.min(dead, 2))) {
            let isDead = k >= count
            // Spread: young leaves near upright, old ones out and down; dead ones hang.
            let age = isDead ? 1 : (Float(k) + rng.float(0, 1)) / Float(count)
            let fromVertical = isDead ? rng.float(2.5, 2.9) : 0.2 + 1.85 * age + rng.float(-0.15, 0.15)
            let az = Float(k) * 2.39996 + rng.float(-0.3, 0.3)
            let out = SIMD3<Float>(cos(az), sin(az), 0)
            let d = simd_normalize(SIMD3(0, 0, cos(fromVertical)) + out * sin(fromVertical))
            let petioleLength = (isDead ? 0.6 : petiole) * rng.float(0.8, 1.15)
            let bladeStart = hub + d * petioleLength
            // Petiole: a thin stalk.
            let side = perpendicular(d)
            b.tube([Ring(position: hub, direction: d, side: side, radius: 0.035),
                    Ring(position: bladeStart, direction: d, side: side, radius: 0.022)],
                   sides: lod == 0 ? 4 : 3, material: .louluTrunk, repeatLength: 1)
            fan(&b, at: bladeStart, direction: d, radius: rng.float(leafRadius.lowerBound, leafRadius.upperBound),
                roll: rng.float(-0.35, 0.35), dead: isDead, lod: lod, rng: &rng)
        }
        b.finish(aoFloor: 0.45)
        return b.mesh
    }

    /// One costapalmate fan: a pleated disc sector opening from the hastula along `direction`,
    /// mapped with the card layout `tools/flora_textures.py` draws (FAN_* constants there).
    func fan(_ b: inout MeshBuilder, at origin: SIMD3<Float>, direction: SIMD3<Float>, radius: Float, roll: Float,
             dead: Bool, lod: Int, rng: inout SplitMix64) {
        let halfAngle: Float = 100 * .pi / 180
        let segments = lod == 0 ? 20 : 6
        let rings = lod == 0 ? 4 : 2
        // The blade's plane holds `direction`; its normal leans up (or out, for hanging leaves).
        var across = simd_normalize(simd_cross(direction, SIMD3(0, 0, 1)))
        if !across.x.isFinite { across = SIMD3(1, 0, 0) }
        across = rotate(across, about: direction, by: roll)
        let normal = simd_normalize(simd_cross(across, direction))
        let pleat: Float = lod == 0 ? 0.035 : 0
        let droop: Float = dead ? 0.1 : 0.22
        let material: FloraMaterial.ID = dead ? .louluDead : .louluFan
        let base = UInt32(b.mesh.vertices.count)
        for j in 0...rings {
            let rho = Float(j) / Float(rings)
            for i in 0...segments {
                let theta = -halfAngle + 2 * halfAngle * Float(i) / Float(segments)
                // Pleats alternate up and down; the outer blade droops away from the hastula.
                let fold: Float = (i % 2 == 0 ? 1 : -1) * pleat * rho
                let sag: Float = droop * rho * rho * radius
                let planar: SIMD3<Float> = direction * (cos(theta) * rho * radius) + across * (sin(theta) * rho * radius)
                let p: SIMD3<Float> = origin + planar + normal * (fold - sag * (0.4 + 0.6 * abs(sin(theta))))
                let u: Float = 0.5 + 0.49 * rho * sin(theta)
                let v: Float = 0.16 + 0.83 * rho * cos(theta)
                let n = simd_normalize(normal + direction * (2 * droop * rho) + b.crown.normal(p) * 0.3)
                _ = b.vertex(p, n, SIMD2(u, v), material: material, flutter: 0.5 * rho)
            }
        }
        let row = UInt32(segments + 1)
        for j in 0..<UInt32(rings) {
            for i in 0..<UInt32(segments) {
                let a = base + j * row + i, c = a + row
                b.triangle(a, a + 1, c + 1)
                b.triangle(a, c + 1, c)
            }
        }
    }
}

/// Hāpuʻu: a thick fibrous trunk topped with golden pulu, a rosette of arching bipinnate
/// fronds (ribbon strips with the frond texture), and fiddleheads uncurling at the centre.
struct TreeFern {
    var height: Float
    var trunkRadius: Float = 0.22
    var fronds: ClosedRange<Int> = 9...14
    var frondLength: ClosedRange<Float> = 2.6...3.4

    func build(seed: UInt64, lod: Int) -> PlantMesh {
        var rng = SplitMix64(seed: seed)
        var b = MeshBuilder(height: height, crown: Crown(centre: SIMD3(0, 0, height * 0.75), radii: SIMD3(2.6, 2.6, 1.4)))
        let lean = rng.float(0.0, 0.12), azimuth = rng.float(0, 2 * .pi)
        let leanDir = SIMD3<Float>(cos(azimuth), sin(azimuth), 0)
        let trunkTop = height * rng.float(0.38, 0.55)
        let steps = lod == 0 ? 8 : 3
        var rings: [Ring] = []
        var top = SIMD3<Float>(0, 0, 0)
        for k in 0...steps {
            let t = Float(k) / Float(steps)
            let z = -0.5 + (trunkTop + 0.5) * t
            let p = SIMD3<Float>(0, 0, z) + leanDir * (lean * z * t)
            // Root mat flares the base; knobbly along the trunk.
            let r = trunkRadius * (1 + 0.5 * (1 - smoothstep(0, 0.2, t))) * rng.float(0.92, 1.08)
            rings.append(Ring(position: p, direction: simd_normalize(SIMD3(0, 0, 1) + leanDir * lean), side: SIMD3(1, 0, 0), radius: r))
            top = p
        }
        b.tube(rings, sides: lod == 0 ? 8 : 5, material: .hapuuTrunk, repeatLength: 1.2)
        // Pulu: a golden woolly cap with the frond bases.
        let up = SIMD3<Float>(0, 0, 1)
        b.tube([Ring(position: top - up * 0.15, direction: up, side: SIMD3(1, 0, 0), radius: trunkRadius * 1.02),
                Ring(position: top + up * 0.2, direction: up, side: SIMD3(1, 0, 0), radius: trunkRadius * 1.2),
                Ring(position: top + up * 0.38, direction: up, side: SIMD3(1, 0, 0), radius: trunkRadius * 0.5)],
               sides: lod == 0 ? 8 : 5, material: .pulu, repeatLength: 0.6)

        let hub = top + up * 0.25
        let count = lod == 0 ? rng.int(fronds) : 8
        for k in 0..<count {
            let az = Float(k) * 2.39996 + rng.float(-0.25, 0.25)
            let rise = rng.float(0.3, 0.75)   // radians from vertical at the base
            let length = rng.float(frondLength.lowerBound, frondLength.upperBound) * (lod == 0 ? 1 : 1.05)
            frond(&b, from: hub, azimuth: az, rise: rise, length: length, lod: lod)
        }
        if lod == 0 {
            // Fiddleheads (koʻokoʻolau): croziers coiled in pulu.
            for k in 0..<rng.int(2...4) {
                let az = Float(k) * 2.1 + rng.float(0, 1)
                let out = SIMD3<Float>(cos(az), sin(az), 0)
                var ringsF: [Ring] = []
                for s in 0...10 {
                    let t = Float(s) / 10
                    let a = t * 4.2
                    let r = 0.18 * (1 - 0.7 * t)
                    let p = hub + out * (0.06 + r * (1 - cos(a)) * 0.6) + up * (0.25 * t + r * sin(a) * 0.6)
                    let d = simd_normalize(out * sin(a) + up * cos(a))
                    ringsF.append(Ring(position: p, direction: d, side: perpendicular(d), radius: 0.03 * (1 - 0.5 * t)))
                }
                b.tube(ringsF, sides: 4, material: .pulu, repeatLength: 0.4)
            }
        }
        b.finish(aoFloor: 0.4)
        return b.mesh
    }

    /// An arching frond: a ribbon with a slight V across it carrying the whole frond texture
    /// (its lanceolate outline and bare stipe are in the texture's alpha).
    func frond(_ b: inout MeshBuilder, from hub: SIMD3<Float>, azimuth: Float, rise: Float, length: Float, lod: Int) {
        let steps = lod == 0 ? 10 : 4
        let out = SIMD3<Float>(cos(azimuth), sin(azimuth), 0)
        let across = SIMD3<Float>(-out.y, out.x, 0)
        let base = UInt32(b.mesh.vertices.count)
        let columns = lod == 0 ? 3 : 2
        var p = hub
        for k in 0...steps {
            let t = Float(k) / Float(steps)
            // Arch: starts at `rise` from vertical and bends over past horizontal.
            let angle = rise + t * t * 1.5
            let d = simd_normalize(SIMD3(0, 0, cos(angle)) + out * sin(angle))
            if k > 0 { p += d * (length / Float(steps)) }
            let width: Float = 0.62 * (0.35 + 0.65 * smoothstep(0.0, 0.3, t)) * (1 - 0.5 * smoothstep(0.5, 1.0, t))
            let lift = simd_normalize(simd_cross(across, d))
            for c in 0..<columns {
                let s: Float = columns == 2 ? Float(c) * 2 - 1 : Float(c) - 1
                let pos: SIMD3<Float> = p + across * (s * width) + lift * ((1 - abs(s)) * width * 0.25)
                let n = simd_normalize(lift * 0.8 + across * (-s * 0.3) + b.crown.normal(pos) * 0.4)
                let u: Float = 0.5 + 0.5 * s
                _ = b.vertex(pos, n, SIMD2(u, t), material: .hapuuFrond, flutter: t)
            }
        }
        let row = UInt32(columns)
        for k in 0..<UInt32(steps) {
            for c in 0..<UInt32(columns - 1) {
                let a = base + k * row + c, d = a + row
                b.triangle(a, a + 1, d + 1)
                b.triangle(a, d + 1, d)
            }
        }
    }
}

/// Naupaka kahakai: a dense mound, wider than tall, of short stems ending in leaf rosettes,
/// wrapped in a shell of rosette cards facing out and up.
struct Shrub {
    var height: Float
    var width: Float
    var stems: ClosedRange<Int> = 7...11
    var shellCards = 70
    var innerCards = 20
    var card: Float = 0.32

    func build(seed: UInt64, lod: Int) -> PlantMesh {
        var rng = SplitMix64(seed: seed)
        let radii = SIMD3<Float>(width * 0.5 * rng.float(0.85, 1.1), width * 0.5 * rng.float(0.85, 1.1), height)
        let crown = Crown(centre: SIMD3(0, 0, 0), radii: radii)
        var b = MeshBuilder(height: height, crown: crown)
        if lod == 0 {
            for _ in 0..<rng.int(stems) {
                let az = rng.float(0, 2 * .pi)
                let tilt = rng.float(0.5, 1.2)
                let d = simd_normalize(SIMD3(sin(tilt) * cos(az), sin(tilt) * sin(az), cos(tilt)))
                let start = SIMD3<Float>(0, 0, -0.2)
                let end = start + d * (0.75 * simd_length(radii * d))
                b.tube([Ring(position: start, direction: d, side: perpendicular(d), radius: 0.04),
                        Ring(position: end, direction: d, side: perpendicular(d), radius: 0.015)],
                       sides: 4, material: .naupakaStem, repeatLength: 1)
            }
        }
        // Rosette cards on the dome (Fibonacci spiral over the upper hemisphere) and inside it.
        let shell = lod == 0 ? shellCards : shellCards / 5
        let scale: Float = lod == 0 ? 1 : 2.1
        let golden: Float = 2.39996
        for k in 0..<shell {
            let z = 1 - (Float(k) + 0.5) / Float(shell) * 0.95
            let r = (1 - z * z).squareRoot()
            let a = Float(k) * golden + rng.float(-0.2, 0.2)
            let unit = SIMD3<Float>(r * cos(a), r * sin(a), z)
            let p: SIMD3<Float> = unit * radii * rng.float(0.88, 1.0)
            let facing = simd_normalize(crown.normal(p) + SIMD3(0, 0, 0.5) + rng.direction() * 0.35)
            let right = rotate(perpendicular(facing), about: facing, by: rng.float(0, 2 * .pi))
            let s = card * scale * rng.float(0.85, 1.15)
            b.card(centre: p, right: right * s, up: simd_cross(facing, right) * s, material: .naupakaLeaves, bend: 0.6)
        }
        if lod == 0 {
            for _ in 0..<innerCards {
                var unit = rng.direction()
                unit.z = abs(unit.z)
                let p: SIMD3<Float> = unit * radii * rng.float(0.4, 0.75)
                let facing = simd_normalize(rng.direction() + SIMD3(0, 0, 0.8))
                let right = perpendicular(facing)
                let s = card * rng.float(0.9, 1.2)
                b.card(centre: p, right: right * s, up: simd_cross(facing, right) * s, material: .naupakaLeaves, bend: 0.5)
            }
        }
        b.finish(aoFloor: 0.35)
        return b.mesh
    }
}
