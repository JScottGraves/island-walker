import Foundation

/// Buildings (docs/architecture §structures): structures that stand up off the ground, built
/// from their properties into a mesh the renderer draws and traces sun rays against, and
/// into wall lines the walker can't pass. The ground under one is still its zones (a hangar's
/// floor is a concrete zone like the apron round it).
///
/// The only kind so far is the airfield's hangar: concrete block side and back walls, a
/// barrel roof of corrugated iron, and a wide door opening in the front gable under a
/// concrete lintel, the lunette above it clad in iron sheeting. Abandoned: sheets missing
/// from the roof, sliding door leaves stuck where they were left or gone.

/// What a building's surfaces are made of (BuildingShaders' material codes).
public enum BuildingMaterial: Int, Sendable {
    /// Concrete block laid in running bond, once whitewashed.
    case block = 0
    /// The roof's corrugated iron, corrugations running down the arch.
    case roofIron = 1
    /// Painted iron sheeting with vertical corrugations: door leaves and the lunette.
    case cladding = 2
    /// Cast concrete: the door lintel and the window sills.
    case concrete = 3
    /// Rusted steel sections: window frames, door tracks.
    case steel = 4
}

/// Triangles in a building's own frame: u along its structure's axis, v across it (+v right
/// of +u), z up from its floor (metres). `uv` lays a material's pattern in metres.
public struct BuildingMesh: Sendable {
    public struct Vertex: Sendable, Equatable {
        public var position: SIMD3<Float>
        public var normal: SIMD3<Float>
        public var uv: SIMD2<Float>
        public var material: BuildingMaterial
    }
    public var vertices: [Vertex] = []
    public var indices: [UInt32] = []

    public init() {}

    public var triangleCount: Int { indices.count / 3 }
}

/// A wall line the walker can't cross, in the building's (u, v) frame.
public struct BuildingWall: Sendable, Equatable {
    public var a: SIMD2<Double>
    public var b: SIMD2<Double>
    /// Half the wall's thickness (m).
    public var halfThickness: Double

    /// Distance from (u, v) to the wall's centre line.
    public func distance(u: Double, v: Double) -> Double {
        let p = SIMD2(u, v), d = b - a
        let t = Swift.min(Swift.max(((p - a) * d).sum() / Swift.max((d * d).sum(), 1e-12), 0), 1)
        let q = a + d * t - p
        return (q * q).sum().squareRoot()
    }
}

/// An airfield hangar, read from a "hangar" structure's properties.
public struct Hangar: Sendable, Equatable {
    /// Across the door (along the structure's u axis) and from the door to the back wall (m).
    public var span: Double
    public var depth: Double
    /// Height of the side walls' tops above the floor (the arch springs from there), and the
    /// arch's rise above them (m).
    public var eaves: Double
    public var rise: Double
    /// The side the door faces: +1 towards +v, −1 towards −v.
    public var doorSide: Double
    /// The door opening's width and height (m).
    public var doorWidth: Double
    public var doorHeight: Double
    /// Picks which roof sheets are gone and where the door leaves were left.
    public var seed: UInt64

    public static let wallThickness = 0.25
    /// Depth of the lintel over the door (m).
    public static let lintelDepth = 0.6
    /// The walls carry on this far below the floor, so a floor on a slight grade meets them.
    public static let footing = 0.8
    /// The roof's overhang past the gables and the side walls (m).
    public static let gableOverhang = 0.5
    public static let eaveOverhang = 0.35

    public init(span: Double = 28, depth: Double = 36, eaves: Double = 4.8, rise: Double = 6, doorSide: Double = -1,
                doorWidth: Double = 25.6, doorHeight: Double = 5.0, seed: UInt64 = 1) {
        self.span = span
        self.depth = depth
        self.eaves = eaves
        self.rise = rise
        self.doorSide = doorSide
        self.doorWidth = doorWidth
        self.doorHeight = doorHeight
        self.seed = seed
    }

    /// The hangar a structure describes, or nil when it isn't one.
    public init?(_ s: Structure) {
        guard s.kind == "hangar" else { return nil }
        func value(_ key: String, _ fallback: Double) -> Double { s.properties[key].flatMap(Double.init) ?? fallback }
        self.init(span: value("span", 28), depth: value("depth", 36), eaves: value("eaves", 4.8), rise: value("rise", 6),
                  doorSide: value("door", -1) < 0 ? -1 : 1, doorWidth: value("doorWidth", 25.6),
                  doorHeight: value("doorHeight", 5.0), seed: Hangar.hash(s.id) ^ UInt64(bitPattern: Int64(s.x * 10 + s.y)))
    }

    static func hash(_ text: String) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for b in text.utf8 { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        return h
    }

    /// The circle through both eaves and the crown: its radius and its centre's height.
    var arch: (radius: Double, centre: Double) {
        let h = span / 2
        let r = (h * h + rise * rise) / (2 * rise)
        return (r, eaves + rise - r)
    }

    /// The roof's height above the floor at `s` metres across the span from its middle.
    public func roofHeight(across s: Double) -> Double {
        let (r, c) = arch
        return c + (Swift.max(r * r - s * s, 0)).squareRoot()
    }

    /// The highest point of the roof above the floor.
    public var top: Double { eaves + rise }

    /// Building frame (s across the span, t in from the door, z up) → structure frame (u, v, z).
    func toLocal(_ s: Double, _ t: Double, _ z: Double) -> SIMD3<Float> {
        SIMD3(Float(s), Float(doorSide * (depth / 2 - t)), Float(z))
    }

    func toLocalNormal(_ n: SIMD3<Double>) -> SIMD3<Float> {
        SIMD3(Float(n.x), Float(-doorSide * n.y), Float(n.z))
    }

    /// The walls the walker can't pass, in the structure's (u, v) frame: the side walls, the
    /// back wall and the piers either side of the door, and the door leaves still standing.
    public func walls() -> [BuildingWall] {
        let w = Hangar.wallThickness, hs = span / 2, half = w / 2
        func wall(_ s0: Double, _ t0: Double, _ s1: Double, _ t1: Double, _ thick: Double) -> BuildingWall {
            let a = toLocal(s0, t0, 0), b = toLocal(s1, t1, 0)
            return BuildingWall(a: SIMD2(Double(a.x), Double(a.y)), b: SIMD2(Double(b.x), Double(b.y)), halfThickness: thick)
        }
        var out = [
            wall(-hs + half, 0, -hs + half, depth, half),
            wall(hs - half, 0, hs - half, depth, half),
            wall(-hs, depth - half, hs, depth - half, half),
            wall(-hs, half, -doorWidth / 2, half, half),
            wall(doorWidth / 2, half, hs, half, half),
        ]
        for leaf in doorLeaves() {
            out.append(wall(leaf.s0, leaf.t, leaf.s1, leaf.t, 0.05))
        }
        return out
    }

    /// The sliding door leaves left in the opening: their extent across it and how far in
    /// their track runs.
    func doorLeaves() -> [(s0: Double, s1: Double, t: Double)] {
        var rng = BuildingRandom(seed: seed ^ 0xd00d)
        let n = 4
        let width = doorWidth / Double(n)
        let tracks = [Hangar.wallThickness + 0.15, Hangar.wallThickness + 0.4]
        var leaves: [(Double, Double, Double)] = []
        // Leaves stuck at either end (closed), one slid half across behind its neighbour,
        // one or two gone: the opening left is a third to a half of the door.
        let left = -doorWidth / 2
        leaves.append((left, left + width, tracks[0]))
        let slid = 0.35 + 0.4 * rng.next()
        leaves.append((left + width * slid, left + width * (1 + slid), tracks[1]))
        if rng.next() < 0.6 {
            leaves.append((doorWidth / 2 - width, doorWidth / 2, tracks[0]))
        }
        return leaves.map { (s0: $0.0, s1: $0.1, t: $0.2) }
    }

    /// The roof sheets that are gone, as (first, last) ranges of sheet columns along the
    /// depth and of arch segments.
    func missingSheets(columns: Int, segments: Int) -> [(c0: Int, c1: Int, a0: Int, a1: Int)] {
        var rng = BuildingRandom(seed: seed ^ 0x5eed)
        var holes: [(Int, Int, Int, Int)] = []
        let count = 3 + Int(rng.next() * 3)
        for _ in 0..<count {
            let width = 1 + Int(rng.next() * 2.5)
            let length = 2 + Int(rng.next() * 4)
            let c0 = 2 + Int(rng.next() * Double(columns - 4 - width))
            let a0 = 1 + Int(rng.next() * Double(segments - 2 - length))
            holes.append((c0, c0 + width - 1, a0, a0 + length - 1))
        }
        return holes.map { (c0: $0.0, c1: $0.1, a0: $0.2, a1: $0.3) }
    }

    /// The window openings along each side wall: (from, to) metres in from the door, and the
    /// sill and head heights.
    func windows() -> [(t0: Double, t1: Double, z0: Double, z1: Double)] {
        let bays = Int((depth / 6).rounded())
        let bay = depth / Double(bays)
        return (0..<bays).map { k in
            let mid = (Double(k) + 0.5) * bay
            return (mid - 1.2, mid + 1.2, 2.4, 3.8)
        }
    }

    public func mesh() -> BuildingMesh {
        var b = BuildingMeshBuilder(hangar: self)
        let w = Hangar.wallThickness, hs = span / 2, base = -Hangar.footing
        let holes = windows().map { Opening(a0: $0.t0, a1: $0.t1, z0: $0.z0, z1: $0.z1) }

        // Side walls: outer face at |s| = hs, inner at hs − w, from the door's face to the back.
        for side in [-1.0, 1.0] {
            b.wall(plane: .s(side * hs), outward: side, from: 0, to: depth, base: base, top: eaves, holes: holes,
                   material: .block)
            b.wall(plane: .s(side * (hs - w)), outward: -side, from: w, to: depth - w, base: base, top: eaves,
                   holes: holes, material: .block)
            // Reveals round each window, sills of cast concrete, and the wall's top.
            for o in holes {
                b.reveal(plane: .s(side * hs), depth: -side * w, opening: o, material: .block, sill: .concrete)
            }
            b.quad(b.p(side * hs, 0, eaves), b.p(side * (hs - w), 0, eaves), b.p(side * (hs - w), depth, eaves),
                   b.p(side * hs, depth, eaves), normal: SIMD3(0, 0, 1), uv: { s, t, _ in SIMD2(t, s) },
                   material: .concrete)
            // Steel frames in the windows: a mullion and a transom.
            for o in holes {
                let s = side * (hs - w / 2)
                let mid = (o.a0 + o.a1) / 2, zm = (o.z0 + o.z1) / 2
                b.box(s: (s - 0.03, s + 0.03), t: (mid - 0.03, mid + 0.03), z: (o.z0, o.z1), material: .steel)
                b.box(s: (s - 0.03, s + 0.03), t: (o.a0, o.a1), z: (zm - 0.03, zm + 0.03), material: .steel)
            }
        }

        // Back gable: block up to the arch, with a personnel door off-centre.
        let person = Opening(a0: span * 0.22, a1: span * 0.22 + 1.0, z0: 0, z1: 2.1)
        b.gable(t: depth, outward: 1, from: -hs, to: hs, base: base, top: { roofHeight(across: $0) }, holes: [person],
                material: .block)
        b.gable(t: depth - w, outward: -1, from: -hs + w, to: hs - w, base: base, top: { roofHeight(across: $0) },
                holes: [person], material: .block)
        b.reveal(plane: .t(depth), depth: -w, opening: person, material: .block, sill: nil)

        // Front gable: block piers either side of the opening, a concrete lintel across it,
        // and the lunette above clad in iron sheeting.
        let dh = doorWidth / 2
        for side in [-1.0, 1.0] {
            let (s0, s1) = side < 0 ? (-hs, -dh) : (dh, hs)
            b.gable(t: 0, outward: -1, from: s0, to: s1, base: base, top: { roofHeight(across: $0) }, holes: [],
                    material: .block)
            let (i0, i1) = side < 0 ? (-hs + w, -dh) : (dh, hs - w)
            b.gable(t: w, outward: 1, from: i0, to: i1, base: base, top: { roofHeight(across: $0) }, holes: [],
                    material: .block)
            // The pier's face in the opening.
            let s = side * dh
            b.gable(plane: .s(s), outward: -side, from: 0, to: w, base: base,
                    top: { _ in roofHeight(across: s) }, material: .block)
        }
        let lintelTop = doorHeight + Hangar.lintelDepth
        b.box(s: (-dh, dh), t: (0, w), z: (doorHeight, lintelTop), material: .concrete)
        b.gable(t: w / 2, outward: -1, from: -dh, to: dh, base: lintelTop, top: { roofHeight(across: $0) }, holes: [],
                material: .cladding)

        // Door leaves on their tracks just inside the opening, and the tracks themselves.
        for leaf in doorLeaves() {
            b.box(s: (leaf.s0, leaf.s1), t: (leaf.t - 0.04, leaf.t + 0.04), z: (0.02, doorHeight - 0.05),
                  material: .cladding)
        }
        for t in [w + 0.15, w + 0.4] {
            b.box(s: (-dh, dh), t: (t - 0.06, t + 0.06), z: (doorHeight - 0.05, doorHeight + 0.03), material: .steel)
        }

        // The barrel roof: corrugated sheets ~0.9 m wide laid down the arch, over the gables
        // and the eaves, some of them gone.
        b.roof(missing: missingSheets)
        return b.mesh
    }
}

/// A rectangular hole in a wall: from a0 to a1 along it, from z0 up to z1.
struct Opening {
    var a0, a1, z0, z1: Double
}

/// Deterministic 0…1 numbers from a seed.
struct BuildingRandom {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z ^= z >> 31
        return Double(z >> 11) / Double(1 << 53)
    }
}

/// Assembles a hangar's triangles in its building frame (s, t, z), converted to the
/// structure frame as they're added.
struct BuildingMeshBuilder {
    let hangar: Hangar
    var mesh = BuildingMesh()

    init(hangar: Hangar) { self.hangar = hangar }

    enum Plane {
        /// A wall at constant s (a side wall), running along t.
        case s(Double)
        /// A wall at constant t (a gable), running along s.
        case t(Double)
    }

    func p(_ s: Double, _ t: Double, _ z: Double) -> SIMD3<Double> { SIMD3(s, t, z) }

    /// Adds a quad (corners in order round it); `uv` maps (s, t, z) to the pattern's metres.
    mutating func quad(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>, _ d: SIMD3<Double>,
                       normal: SIMD3<Double>, uv: (Double, Double, Double) -> SIMD2<Double>,
                       material: BuildingMaterial) {
        let base = UInt32(mesh.vertices.count)
        let n = hangar.toLocalNormal(normal)
        for q in [a, b, c, d] {
            let m = uv(q.x, q.y, q.z)
            mesh.vertices.append(.init(position: hangar.toLocal(q.x, q.y, q.z), normal: n,
                                       uv: SIMD2(Float(m.x), Float(m.y)), material: material))
        }
        mesh.indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    }

    /// A flat wall face from `from` to `to` along the plane, `base` to `top` high, with
    /// rectangular holes; `outward` is the side (±1 along the plane's axis) the face looks to.
    mutating func wall(plane: Plane, outward: Double, from a0: Double, to a1: Double, base: Double, top: Double,
                       holes: [Opening], material: BuildingMaterial) {
        gableStrips(plane: plane, outward: outward, from: a0, to: a1, base: base, top: { _ in top }, holes: holes,
                    material: material, steps: 1)
    }

    /// A gable face at depth t: from `from` to `to` across the span, from `base` up to `top(s)`.
    mutating func gable(t: Double, outward: Double, from a0: Double, to a1: Double, base: Double,
                        top: (Double) -> Double, holes: [Opening], material: BuildingMaterial) {
        let steps = Swift.max(1, Int(((a1 - a0) / 1.0).rounded(.up)))
        gableStrips(plane: .t(t), outward: outward, from: a0, to: a1, base: base, top: top, holes: holes,
                    material: material, steps: steps)
    }

    /// A face at constant s across a short run of t (a pier's side in the door opening).
    mutating func gable(plane: Plane, outward: Double, from a0: Double, to a1: Double, base: Double,
                        top: (Double) -> Double, material: BuildingMaterial) {
        gableStrips(plane: plane, outward: outward, from: a0, to: a1, base: base, top: top, holes: [],
                    material: material, steps: 1)
    }

    /// Splits a face into vertical strips at its holes' sides (and every `steps`th of its run
    /// for a curved top), each filled where no hole is.
    mutating func gableStrips(plane: Plane, outward: Double, from a0: Double, to a1: Double, base: Double,
                              top: (Double) -> Double, holes: [Opening], material: BuildingMaterial,
                              steps: Int) {
        var cuts = (0...steps).map { a0 + (a1 - a0) * Double($0) / Double(steps) }
        for h in holes { cuts += [h.a0, h.a1] }
        cuts = Array(Set(cuts.filter { $0 >= a0 - 1e-9 && $0 <= a1 + 1e-9 })).sorted()
        for k in 0..<(cuts.count - 1) {
            let x0 = cuts[k], x1 = cuts[k + 1]
            guard x1 - x0 > 1e-6 else { continue }
            let mid = (x0 + x1) / 2
            // Height intervals of this strip not inside a hole.
            var spans = [(base, Swift.min(top(x0), top(x1)))]
            for h in holes where h.a0 <= mid && mid <= h.a1 {
                spans = spans.flatMap { s -> [(Double, Double)] in
                    var out: [(Double, Double)] = []
                    if h.z0 > s.0 { out.append((s.0, Swift.min(s.1, h.z0))) }
                    if h.z1 < s.1 { out.append((Swift.max(s.0, h.z1), s.1)) }
                    return out.filter { $0.1 - $0.0 > 1e-6 }
                }
            }
            for (j, span) in spans.enumerated() {
                // The strip's top follows the curve where it is the face's top.
                let curved = j == spans.count - 1 && span.1 >= Swift.min(top(x0), top(x1)) - 1e-6
                let z00 = span.0, z10 = span.0
                let z01 = curved ? top(x0) : span.1, z11 = curved ? top(x1) : span.1
                switch plane {
                case .s(let s):
                    quad(p(s, x0, z00), p(s, x1, z10), p(s, x1, z11), p(s, x0, z01), normal: SIMD3(outward, 0, 0),
                         uv: { _, t, z in SIMD2(t, z) }, material: material)
                case .t(let t):
                    quad(p(x0, t, z00), p(x1, t, z10), p(x1, t, z11), p(x0, t, z01), normal: SIMD3(0, outward, 0),
                         uv: { s, _, z in SIMD2(s, z) }, material: material)
                }
            }
        }
    }

    /// The faces lining a hole through a wall `depth` thick (signed, from the plane inwards):
    /// its sides, its head, and its sill (or the wall's material when nil, or none at floor level).
    mutating func reveal(plane: Plane, depth: Double, opening o: Opening, material: BuildingMaterial,
                         sill: BuildingMaterial?) {
        func point(_ a: Double, _ d: Double, _ z: Double) -> SIMD3<Double> {
            switch plane {
            case .s(let s): return p(s + d, a, z)
            case .t(let t): return p(a, t + d, z)
            }
        }
        func along(_ x: Double) -> SIMD3<Double> {
            switch plane {
            case .s: return SIMD3(0, x, 0)
            case .t: return SIMD3(x, 0, 0)
            }
        }
        let uvAlong: (Double, Double, Double) -> SIMD2<Double> = { s, t, z in SIMD2(s + t, z) }
        // Jambs, facing into the hole.
        quad(point(o.a0, 0, o.z0), point(o.a0, depth, o.z0), point(o.a0, depth, o.z1), point(o.a0, 0, o.z1),
             normal: along(1), uv: uvAlong, material: material)
        quad(point(o.a1, 0, o.z0), point(o.a1, depth, o.z0), point(o.a1, depth, o.z1), point(o.a1, 0, o.z1),
             normal: along(-1), uv: uvAlong, material: material)
        // Head, facing down.
        quad(point(o.a0, 0, o.z1), point(o.a1, 0, o.z1), point(o.a1, depth, o.z1), point(o.a0, depth, o.z1),
             normal: SIMD3(0, 0, -1), uv: { s, t, _ in SIMD2(s, t) }, material: material)
        if o.z0 > 0.01, let sill {
            quad(point(o.a0, 0, o.z0), point(o.a1, 0, o.z0), point(o.a1, depth, o.z0), point(o.a0, depth, o.z0),
                 normal: SIMD3(0, 0, 1), uv: { s, t, _ in SIMD2(s, t) }, material: sill)
        }
    }

    /// A box between the given extents, all six faces.
    mutating func box(s: (Double, Double), t: (Double, Double), z: (Double, Double), material: BuildingMaterial) {
        let (s0, s1) = s, (t0, t1) = t, (z0, z1) = z
        quad(p(s0, t0, z0), p(s1, t0, z0), p(s1, t0, z1), p(s0, t0, z1), normal: SIMD3(0, -1, 0),
             uv: { s, _, z in SIMD2(s, z) }, material: material)
        quad(p(s0, t1, z0), p(s1, t1, z0), p(s1, t1, z1), p(s0, t1, z1), normal: SIMD3(0, 1, 0),
             uv: { s, _, z in SIMD2(s, z) }, material: material)
        quad(p(s0, t0, z0), p(s0, t1, z0), p(s0, t1, z1), p(s0, t0, z1), normal: SIMD3(-1, 0, 0),
             uv: { _, t, z in SIMD2(t, z) }, material: material)
        quad(p(s1, t0, z0), p(s1, t1, z0), p(s1, t1, z1), p(s1, t0, z1), normal: SIMD3(1, 0, 0),
             uv: { _, t, z in SIMD2(t, z) }, material: material)
        quad(p(s0, t0, z1), p(s1, t0, z1), p(s1, t1, z1), p(s0, t1, z1), normal: SIMD3(0, 0, 1),
             uv: { s, t, _ in SIMD2(s, t) }, material: material)
        quad(p(s0, t0, z0), p(s1, t0, z0), p(s1, t1, z0), p(s0, t1, z0), normal: SIMD3(0, 0, -1),
             uv: { s, t, _ in SIMD2(s, t) }, material: material)
    }

    /// The barrel roof as a grid of sheets: columns ~0.9 m along the depth, segments down the
    /// arch; `missing` names the ones that are gone. Pattern uv: (metres down the arch from the
    /// crown, metres along the depth).
    mutating func roof(missing: (Int, Int) -> [(c0: Int, c1: Int, a0: Int, a1: Int)]) {
        let h = hangar
        let (r, centre) = h.arch
        let reach = h.span / 2 + Hangar.eaveOverhang
        let angle = asin(Swift.min(reach / r, 1))
        let segments = 28
        let t0 = -Hangar.gableOverhang, t1 = h.depth + Hangar.gableOverhang
        let columns = Int(((t1 - t0) / 0.9).rounded())
        let holes = missing(columns, segments)
        // The roof sits on the arch's top: lift it by half a corrugation so it clears the gables.
        let lift = 0.03
        func point(_ k: Int) -> (s: Double, z: Double, n: SIMD3<Double>, arc: Double) {
            let a = -angle + 2 * angle * Double(k) / Double(segments)
            return (r * sin(a), centre + r * cos(a) + lift, SIMD3(sin(a), 0, cos(a)), r * a)
        }
        for c in 0..<columns {
            let ta = t0 + (t1 - t0) * Double(c) / Double(columns)
            let tb = t0 + (t1 - t0) * Double(c + 1) / Double(columns)
            for k in 0..<segments {
                if holes.contains(where: { c >= $0.c0 && c <= $0.c1 && k >= $0.a0 && k <= $0.a1 }) { continue }
                let e0 = point(k), e1 = point(k + 1)
                let base = UInt32(mesh.vertices.count)
                for (e, t) in [(e0, ta), (e1, ta), (e1, tb), (e0, tb)] {
                    mesh.vertices.append(.init(position: h.toLocal(e.s, t, e.z), normal: h.toLocalNormal(e.n),
                                               uv: SIMD2(Float(e.arc), Float(t)), material: .roofIron))
                }
                mesh.indices += [base, base + 1, base + 2, base, base + 2, base + 3]
            }
        }
    }
}
