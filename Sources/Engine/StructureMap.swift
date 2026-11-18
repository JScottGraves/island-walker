import Foundation
import simd
import IslandCore

/// The renderer's structure map (docs/architecture §structures): one lookup every system
/// asks what lies at a point (the fine detail, the terrain shading, the trees and the
/// ground cover), so a new structure type needs no renderer change, only zones.
///
/// Built once at load from the bundle's structures into one GPU buffer:
///
///     header (32 B) | zones (64 B each) | bins (uint2 offset, count) | zone indices (uint)
///     | buildings (from a 16 B aligned offset the header gives: a count, then 48 B each)
///
/// A zone is an oriented box with what the ground is, what grows on it and whether it is
/// graded. Every structure's zones are listed smallest first (IslandCore.StructureMap.zoneOrder:
/// a structure's own innermost first, and an apron's taxiway ahead of the runway's grass strip
/// it crosses), then one extra zone per structure for its cut and fill slopes (the graded box
/// grown by `embankment`: natural ground, grass only).
/// A uniform grid of `binSize` bins over the structures' surroundings lists the zones that
/// come within `margin` of each bin, in zone order, so the first zone containing a point is
/// the one that applies (as IslandCore.StructureMap.sample has it). With no structures the
/// buffer is the header alone and every lookup returns open country at once.
public struct StructureLookup: Sendable {
    public static let headerBytes = 32

    /// One zone as the GPU reads it (StructureZoneGPU, four float4).
    public struct Zone: Sendable {
        /// Centre x, y (m); unit +u axis x, y.
        public var a: SIMD4<Float>
        /// Half length, half width (m); grade line elevation (m) at the centre and slope along +u.
        public var b: SIMD4<Float>
        /// Surface code, plant code, graded (0/1), structure index.
        public var c: SIMD4<Float>
        /// The structure's graded half length and width (m); its runway designators as
        /// 100 × the one painted at the −u end (read travelling +u) + the one at the +u end
        /// (0 for none), or −1 for paving whose ends join other paving (only its sides crumble,
        /// properties["edges"] = "sides"); its upkeep, 0 abandoned … 1 maintained.
        public var d: SIMD4<Float>
    }

    /// What one point is (StructureSample in MSL).
    public struct Sample: Sendable, Equatable {
        public var surface: Int
        public var plants: Int
        public var graded: Bool
        /// The structure's grade line at this point (clamped to its graded ends), when in one.
        public var gradeHeight: Float
        /// Along and across the structure's axis, metres from its centre.
        public var u: Float, v: Float
        public var zone: Int
        public var structure: Int
        /// Signed distance outside the structure's graded box (negative inside); outside every
        /// graded zone, the distance to the nearest graded zone of any structure.
        public var edge: Float
        /// How kept the structure is: 0 abandoned (cracked, worn, overgrown) … 1 maintained.
        public var upkeep: Float = 1
        /// Runway designators painted at its −u and +u ends, 0 for none.
        public var designators: (Int, Int) = (0, 0)

        public static func == (a: Sample, b: Sample) -> Bool {
            a.surface == b.surface && a.plants == b.plants && a.graded == b.graded && a.gradeHeight == b.gradeHeight
                && a.u == b.u && a.v == b.v && a.zone == b.zone && a.structure == b.structure && a.edge == b.edge
                && a.upkeep == b.upkeep && a.designators == b.designators
        }

        public static let open = Sample(surface: 0, plants: 0, graded: false, gradeHeight: 0, u: 0, v: 0, zone: -1,
                                        structure: -1, edge: 1e9)
    }

    public private(set) var zones: [Zone] = []
    public private(set) var bins: [SIMD2<UInt32>] = []
    public private(set) var indices: [UInt32] = []
    /// The structures that stand up off the ground (Buildings.swift), for the light under
    /// their roofs and the ground cover kept out of them (BuildingGPU in MSL).
    public private(set) var buildings: [Building] = []

    /// One building as the GPU reads it (BuildingGPU, three float4).
    public struct Building: Sendable, Equatable {
        /// Centre x, y (m); unit +u axis x, y.
        public var a: SIMD4<Float>
        /// Half span (along u), half depth (along v), floor height at the centre, eaves height above it (m).
        public var b: SIMD4<Float>
        /// The arch's rise above the eaves, the side its door faces (±1 along v), the door's half width and height (m).
        public var c: SIMD4<Float>
    }
    public private(set) var origin = SIMD2<Float>(0, 0)
    public private(set) var binsX = 0, binsY = 0
    public let binSize: Float
    /// Zones are listed in every bin within this many metres of them, so clearance (the
    /// distance to ground where nothing tall grows) is exact out to it.
    public let margin: Float

    /// - Parameter upkeep: how kept a structure is when its properties don't say ("upkeep"
    ///   0…1, or "condition" abandoned / neglected / maintained): abandoned by default.
    public init(structures: [Structure], binSize: Float = 256, margin: Float = 64, upkeep: Float = 0) {
        self.binSize = binSize
        self.margin = margin
        func zone(_ k: Int, _ hl: Double, _ hw: Double, _ surface: StructureSurface, _ plants: PlantRule,
                  _ graded: Bool) -> Zone {
            let s = structures[k], axis = s.axis
            let designators = Self.designators(s)
            let marks = s.properties["edges"] == "sides" ? -1 : Float(100 * designators.0 + designators.1)
            return Zone(a: SIMD4(Float(s.x), Float(s.y), Float(axis.x), Float(axis.y)),
                        b: SIMD4(Float(hl), Float(hw), Float(s.elevation), Float(s.slope)),
                        c: SIMD4(Float(surface.code), Float(plants.code), graded ? 1 : 0, Float(k)),
                        d: SIMD4(Float(s.gradedHalfLength), Float(s.gradedHalfWidth), marks, Self.upkeep(s) ?? upkeep))
        }
        for (k, i) in StructureMap.zoneOrder(structures) {
            let z = structures[k].zones[i]
            zones.append(zone(k, z.halfLength, z.halfWidth, z.surface, z.plants, z.graded))
        }
        for (k, s) in structures.enumerated() where s.embankment > 0 && s.gradedHalfLength > 0 {
            zones.append(zone(k, s.gradedHalfLength + s.embankment, s.gradedHalfWidth + s.embankment, .natural, .grassOnly, false))
        }
        for s in structures {
            guard let h = Hangar(s) else { continue }
            let axis = s.axis
            buildings.append(Building(a: SIMD4(Float(s.x), Float(s.y), Float(axis.x), Float(axis.y)),
                                      b: SIMD4(Float(h.span / 2), Float(h.depth / 2), Float(s.elevation), Float(h.eaves)),
                                      c: SIMD4(Float(h.rise), Float(h.doorSide), Float(h.doorWidth / 2), Float(h.doorHeight))))
        }
        guard !zones.isEmpty else { return }
        // Each zone's bounding box, grown by the margin.
        let boxes: [(lo: SIMD2<Float>, hi: SIMD2<Float>)] = zones.map { z in
            let ax = abs(z.a.z), ay = abs(z.a.w)
            let ex = ax * z.b.x + ay * z.b.y + margin, ey = ay * z.b.x + ax * z.b.y + margin
            return (SIMD2(z.a.x - ex, z.a.y - ey), SIMD2(z.a.x + ex, z.a.y + ey))
        }
        var lo = boxes[0].lo, hi = boxes[0].hi
        for b in boxes { lo = pointwiseMin(lo, b.lo); hi = pointwiseMax(hi, b.hi) }
        origin = (lo / binSize).rounded(.down) * binSize
        binsX = Int(((hi.x - origin.x) / binSize).rounded(.up))
        binsY = Int(((hi.y - origin.y) / binSize).rounded(.up))
        var lists = [[UInt32]](repeating: [], count: binsX * binsY)
        for (k, b) in boxes.enumerated() {
            let i0 = Swift.max(0, Int(((b.lo.x - origin.x) / binSize).rounded(.down)))
            let i1 = Swift.min(binsX - 1, Int(((b.hi.x - origin.x) / binSize).rounded(.down)))
            let j0 = Swift.max(0, Int(((b.lo.y - origin.y) / binSize).rounded(.down)))
            let j1 = Swift.min(binsY - 1, Int(((b.hi.y - origin.y) / binSize).rounded(.down)))
            guard i0 <= i1, j0 <= j1 else { continue }
            for j in j0...j1 { for i in i0...i1 { lists[j * binsX + i].append(UInt32(k)) } }
        }
        for list in lists {
            bins.append(SIMD2(UInt32(indices.count), UInt32(list.count)))
            indices += list
        }
    }

    /// The runway designators to paint at the −u end (read by a pilot travelling +u, along
    /// the heading) and at the +u end, from properties["designators"] ("06/24"), or (0, 0).
    static func designators(_ s: Structure) -> (Int, Int) {
        guard s.properties["edges"] != "sides", let text = s.properties["designators"] else { return (0, 0) }
        let parts = text.split(separator: "/").compactMap { Int($0) }
        guard parts.count == 2 else { return (0, 0) }
        var n = Int((s.heading / 10).rounded()) % 36
        if n == 0 { n = 36 }
        // The number nearest the heading is the one read travelling along it.
        func turn(_ d: Int) -> Int { let t = abs(d - n) % 36; return Swift.min(t, 36 - t) }
        return turn(parts[0]) <= turn(parts[1]) ? (parts[0], parts[1]) : (parts[1], parts[0])
    }

    /// A structure's own upkeep, from properties["upkeep"] (0…1) or ["condition"].
    static func upkeep(_ s: Structure) -> Float? {
        if let text = s.properties["upkeep"], let v = Float(text) { return Swift.min(Swift.max(v, 0), 1) }
        switch s.properties["condition"] {
        case "abandoned": return 0
        case "neglected": return 0.4
        case "maintained": return 1
        default: return nil
        }
    }

    /// The buffer the shaders read.
    public func bytes() -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(Self.headerBytes + zones.count * 64 + bins.count * 8 + indices.count * 4)
        func put<T>(_ value: T) { withUnsafeBytes(of: value) { out += $0 } }
        let tail = Self.headerBytes + zones.count * 64 + bins.count * 8 + indices.count * 4
        let buildingOffset = buildings.isEmpty ? 0 : (tail + 15) / 16 * 16
        put(UInt32(zones.count)); put(UInt32(binsX)); put(UInt32(binsY)); put(UInt32(buildingOffset))
        put(origin.x); put(origin.y); put(binSize); put(margin)
        for z in zones { put(z.a); put(z.b); put(z.c); put(z.d) }
        for b in bins { put(b) }
        for i in indices { put(i) }
        if !buildings.isEmpty {
            out += [UInt8](repeating: 0, count: buildingOffset - out.count)
            put(UInt32(buildings.count)); put(UInt32(0)); put(UInt32(0)); put(UInt32(0))
            for b in buildings { put(b.a); put(b.b); put(b.c) }
        }
        return out
    }

    /// The zone indices listed for the bin holding (x, y), or none outside the bins.
    func binZones(_ x: Float, _ y: Float) -> ArraySlice<UInt32> {
        guard !zones.isEmpty else { return [] }
        let i = Int(((x - origin.x) / binSize).rounded(.down)), j = Int(((y - origin.y) / binSize).rounded(.down))
        guard i >= 0, j >= 0, i < binsX, j < binsY else { return [] }
        let b = bins[j * binsX + i]
        return indices[Int(b.x)..<Int(b.x + b.y)]
    }

    static func local(_ z: Zone, _ x: Float, _ y: Float) -> (u: Float, v: Float) {
        let dx = x - z.a.x, dy = y - z.a.y
        return (dx * z.a.z + dy * z.a.w, dx * z.a.w - dy * z.a.z)
    }

    /// Signed distance from (u, v) to a centred box of half extents (hl, hw).
    static func boxDistance(_ u: Float, _ v: Float, _ hl: Float, _ hw: Float) -> Float {
        let qx = abs(u) - hl, qy = abs(v) - hw
        let outside = SIMD2(Swift.max(qx, 0), Swift.max(qy, 0))
        return simd_length(outside) + Swift.min(Swift.max(qx, qy), 0)
    }

    /// What lies at (x, y): the Swift twin of the MSL structureAt.
    public func sample(x: Float, y: Float) -> Sample {
        for k in binZones(x, y) {
            let z = zones[Int(k)]
            let (u, v) = Self.local(z, x, y)
            guard abs(u) <= z.b.x && abs(v) <= z.b.y else { continue }
            let graded = z.c.z > 0.5
            var edge = Self.boxDistance(u, v, z.d.x, z.d.y)
            if !graded {
                // Outside the graded ground: how far to the nearest graded zone of any structure,
                // so the fine detail eases in the same way beside a runway's strip and an apron.
                for j in binZones(x, y) where zones[Int(j)].c.z > 0.5 {
                    let g = zones[Int(j)]
                    let (gu, gv) = Self.local(g, x, y)
                    edge = Swift.min(edge, Self.boxDistance(gu, gv, g.b.x, g.b.y))
                }
            }
            let marks = Swift.max(Int(z.d.z), 0)
            return Sample(surface: Int(z.c.x), plants: Int(z.c.y), graded: graded,
                          gradeHeight: z.b.z + z.b.w * Swift.min(Swift.max(u, -z.d.x), z.d.x), u: u, v: v,
                          zone: Int(k), structure: Int(z.c.w), edge: edge,
                          upkeep: z.d.w, designators: (marks / 100, marks % 100))
        }
        return .open
    }

    /// Signed distance (m) to the nearest ground where only grass or nothing may grow;
    /// exact within `margin`, 1e9 where none is that near.
    public func clearance(x: Float, y: Float) -> Float {
        var best: Float = 1e9
        for k in binZones(x, y) {
            let z = zones[Int(k)]
            guard z.c.y > 0.5 else { continue }
            let (u, v) = Self.local(z, x, y)
            best = Swift.min(best, Self.boxDistance(u, v, z.b.x, z.b.y))
        }
        return best
    }

    /// Share of the square (x0, y0)…(x0 + size, y0 + size) where trees may not grow
    /// (16 × 16 samples), to cut the canopy cover maps over the footprints.
    public func restrictedFraction(x0: Float, y0: Float, size: Float) -> Float {
        guard !zones.isEmpty else { return 0 }
        let n = 16
        var hit = 0
        for j in 0..<n {
            for i in 0..<n {
                let s = sample(x: x0 + (Float(i) + 0.5) * size / Float(n), y: y0 + (Float(j) + 0.5) * size / Float(n))
                if s.plants != PlantRule.untouched.code { hit += 1 }
            }
        }
        return Float(hit) / Float(n * n)
    }
}

/// The MSL twin of StructureLookup, spliced into terrainShaderSource ahead of the detail
/// recipe. Every kernel and fragment that asks binds the buffer at `structureBufferIndex`.
let structureShaderSource = """
// ---- Structure map (StructureMap.swift) ----
struct StructureMapHeader {
    uint zoneCount;
    uint binsX;
    uint binsY;
    uint buildings;     // byte offset of the building table, 0 for none
    float2 origin;
    float binSize;
    float margin;
};

struct StructureZoneGPU {
    float4 a;   // centre xy, +u axis xy
    float4 b;   // half length, half width, grade line elevation, slope
    float4 c;   // surface, plants, graded, structure index
    float4 d;   // graded half length, half width, 100 × designator at the −u end + at the +u end (−1: only the sides crumble), upkeep
};

constant int kSurfaceNatural = 0;
constant int kSurfacePaved = 1;
constant int kSurfaceMownGrass = 2;
constant int kSurfaceConcrete = 5;
// Hard paving: the runway's asphalt or concrete slabs.
bool structurePaved(int surface) { return surface == kSurfacePaved || surface == kSurfaceConcrete; }
constant int kPlantsUntouched = 0;
constant int kPlantsGrassOnly = 1;
constant int kPlantsNone = 2;

struct StructureSample {
    int surface;
    int plants;
    bool graded;
    float gradeHeight;  // the structure's grade line here (clamped to its graded ends)
    float2 uv;          // along, across its axis (m from its centre)
    int zone;
    int structure;
    float edge;         // signed distance outside its graded box (m)
    float upkeep;       // 0 abandoned (cracked, worn, overgrown) … 1 maintained
};

const device StructureZoneGPU *structureZones(const device uchar *sm) {
    return (const device StructureZoneGPU *)(sm + 32);
}

// The zone list of the bin holding xy: its first index and count, (0, 0) outside the bins.
uint2 structureBin(const device uchar *sm, float2 xy) {
    const device StructureMapHeader &h = *(const device StructureMapHeader *)sm;
    if (h.zoneCount == 0u) return uint2(0u);
    int2 b = int2(floor((xy - h.origin) / h.binSize));
    if (b.x < 0 || b.y < 0 || b.x >= int(h.binsX) || b.y >= int(h.binsY)) return uint2(0u);
    const device uint2 *bins = (const device uint2 *)(sm + 32 + 64 * h.zoneCount);
    return bins[uint(b.y) * h.binsX + uint(b.x)];
}

const device uint *structureIndices(const device uchar *sm) {
    const device StructureMapHeader &h = *(const device StructureMapHeader *)sm;
    return (const device uint *)(sm + 32 + 64 * h.zoneCount + 8 * h.binsX * h.binsY);
}

float2 structureLocal(StructureZoneGPU z, float2 xy) {
    float2 d = xy - z.a.xy;
    return float2(dot(d, z.a.zw), d.x * z.a.w - d.y * z.a.z);
}

float structureBoxDistance(float2 uv, float2 extent) {
    float2 q = abs(uv) - extent;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

StructureSample structureAt(const device uchar *sm, float2 xy) {
    StructureSample s;
    s.surface = kSurfaceNatural;
    s.plants = kPlantsUntouched;
    s.graded = false;
    s.gradeHeight = 0.0;
    s.uv = float2(0.0);
    s.zone = -1;
    s.structure = -1;
    s.edge = 1e9;
    s.upkeep = 1.0;
    uint2 bin = structureBin(sm, xy);
    if (bin.y == 0u) return s;
    const device StructureZoneGPU *zones = structureZones(sm);
    const device uint *indices = structureIndices(sm);
    for (uint k = 0u; k < bin.y; k++) {
        uint i = indices[bin.x + k];
        StructureZoneGPU z = zones[i];
        float2 uv = structureLocal(z, xy);
        if (abs(uv.x) > z.b.x || abs(uv.y) > z.b.y) continue;
        s.surface = int(z.c.x);
        s.plants = int(z.c.y);
        s.graded = z.c.z > 0.5;
        s.gradeHeight = z.b.z + z.b.w * clamp(uv.x, -z.d.x, z.d.x);
        s.uv = uv;
        s.zone = int(i);
        s.structure = int(z.c.w);
        s.edge = structureBoxDistance(uv, z.d.xy);
        s.upkeep = z.d.w;
        if (!s.graded) {
            // Outside the graded ground: the distance to the nearest graded zone of any structure.
            for (uint j = 0u; j < bin.y; j++) {
                StructureZoneGPU g = zones[indices[bin.x + j]];
                if (g.c.z < 0.5) continue;
                s.edge = min(s.edge, structureBoxDistance(structureLocal(g, xy), g.b.xy));
            }
        }
        return s;
    }
    return s;
}

float structureNoise(float2 q) {
    float2 i = floor(q), f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    float v[4];
    for (int k = 0; k < 4; k++) {
        float2 c = i + float2(float(k & 1), float(k >> 1));
        uint h = uint(int(c.x)) * 0x9e3779b1u ^ uint(int(c.y)) * 0x85ebca77u;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
        v[k] = float(h & 0xffffu) / 65535.0;
    }
    return mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
}

// How much of the ground at xy is a structure's mown grass (0…1): all of its graded
// grass zones, thinning out over a couple of metres across their outer edge with a ragged,
// mower-cut line instead of a ruled one. 0 on pavement and away from structures.
float structureMownShare(StructureSample s, float2 xy) {
    if (s.structure < 0 || structurePaved(s.surface)) return 0.0;
    if (s.surface != kSurfaceMownGrass && s.edge > 6.0) return 0.0;
    float ragged = 0.7 * structureNoise(xy / 1.7 + 3.1) + 0.3 * structureNoise(xy / 0.55 + 8.9) - 0.5;
    // An abandoned strip's edge has grown out into the scrub's and back: a wider, vaguer line.
    float blur = 1.5 + 3.0 * (1.0 - s.upkeep);
    return 1.0 - smoothstep(-blur, blur, s.edge + (2.4 + 3.0 * (1.0 - s.upkeep)) * ragged);
}

uint structureHash(int2 c, uint salt) {
    uint h = uint(c.x) * 0x8da6b343u ^ uint(c.y) * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

float structureUnit(uint h) { return float(h >> 8) * (1.0 / 16777216.0); }

// Distance from q (in cells) to the nearest border of a jittered Voronoi tiling, the
// nearest cell, and a 0…1 hash of the border's cell pair (to keep some borders uncracked).
float voronoiBorder(float2 q, uint salt, thread int2 &cellOut, thread float &pairHash) {
    int2 c0 = int2(floor(q));
    float best = 1e9;
    float2 site = 0.0;
    int2 cell = c0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 c = c0 + int2(i, j);
            uint h = structureHash(c, salt);
            float2 s = float2(c) + 0.15 + 0.7 * float2(structureUnit(h), structureUnit(h * 0x9e3779b9u));
            float d = length_squared(s - q);
            if (d < best) { best = d; site = s; cell = c; }
        }
    }
    float border = 1e9;
    pairHash = 1.0;
    // Sites keep 0.15 of a cell from their cell's edges, so the nearest border lies with one
    // of the eight cells around the nearest site's.
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 c = cell + int2(i, j);
            if (all(c == cell)) continue;
            uint h = structureHash(c, salt);
            float2 s = float2(c) + 0.15 + 0.7 * float2(structureUnit(h), structureUnit(h * 0x9e3779b9u));
            float2 n = s - site;
            float d = dot(0.5 * (s + site) - q, n) * rsqrt(dot(n, n));
            if (d < border) {
                border = d;
                int2 lo = min(c, cell), hi = max(c, cell);
                pairHash = structureUnit(structureHash(lo * 31 + hi, salt + 5u));
            }
        }
    }
    cellOut = cell;
    return border;
}

// The cracks in a structure's paving at uv (its local metres), kept `upkeep` (0 abandoned
// … 1 maintained): distance (m) to the nearest crack's centre line and that crack's half
// width; how broken up the ground is here (alligator cracking, 0…1); and a pothole (0…1)
// where a block of the alligator cracking has broken out.
struct PavementCracks {
    float distance;
    float halfWidth;
    float heavy;
    float pothole;
    float hairline;     // distance (m) to the nearest hairline branch, ≥ 1 where none is drawn
};

PavementCracks pavementCracks(float2 uv, float upkeep, float footprint) {
    PavementCracks k;
    k.distance = 1e9;
    k.halfWidth = 0.0;
    k.heavy = 0.0;
    k.pothole = 0.0;
    k.hairline = 1.0;
    float neglect = 1.0 - upkeep;
    if (neglect <= 0.0) return k;
    k.heavy = neglect * smoothstep(0.45, 0.78, 0.7 * structureNoise(uv / 31.0 + 5.3) + 0.3 * structureNoise(uv / 7.3 + 1.1));
    // Cracks wander: the lines below are drawn through coordinates warped by noise at two
    // scales, so no stretch runs straight for more than a few tens of centimetres.
    float2 warp = 0.35 * (float2(structureNoise(uv / 0.9 + 3.3), structureNoise(uv / 0.9 + 17.1)) - 0.5);
    // The fine wander only where a pixel can show it.
    if (footprint < 0.05) warp += 0.1 * (float2(structureNoise(uv / 0.23 + 8.8), structureNoise(uv / 0.23 + 2.6)) - 0.5);
    uv += warp;
    float open = mix(0.006, 0.022, structureNoise(uv / 1.7 + 2.2));
    // Transverse cracks across the runway every ~14 m, most of them right across, wandering.
    float t = uv.x / 14.0;
    float ti = floor(t + 0.5);
    uint th = structureHash(int2(int(ti), 7), 11u);
    if (structureUnit(th) < 0.35 + 0.6 * neglect) {
        float line = (ti + 0.3 * (structureUnit(th * 0x27d4eb2fu) - 0.5)) * 14.0
                   + 0.9 * (structureNoise(float2(uv.y / 3.1, ti * 7.3)) - 0.5);
        float reach = mix(0.4, 1.0, structureUnit(th * 0x165667b1u)) * 20.0;
        if (abs(uv.y + 6.0 * (structureUnit(th * 0x85ebca6bu) - 0.5)) < reach) {
            float d = abs(uv.x - line);
            if (d < k.distance) { k.distance = d; k.halfWidth = open * 1.4; }
        }
    }
    // Longitudinal: the paving lanes' joints (centreline and ±7.5 m), in runs tens of metres long.
    float lane = abs(uv.y) < 3.75 ? 0.0 : 7.5;
    float joint = lane + 0.25 * (structureNoise(float2(uv.x / 4.3, lane)) - 0.5);
    if (structureNoise(float2(uv.x / 37.0, 3.0 + lane)) < 0.35 + 0.5 * neglect) {
        float d = abs(abs(uv.y) - joint);
        if (d < k.distance) { k.distance = d; k.halfWidth = open * 1.2; }
    }
    // Block cracking everywhere it is old: borders of ~4 m blocks, about half of them open.
    int2 cell;
    float pair;
    float d = voronoiBorder(uv / 4.2, 21u, cell, pair) * 4.2;
    if (pair < 0.3 + 0.5 * neglect + 0.3 * k.heavy && d < k.distance) { k.distance = d; k.halfWidth = open; }
    // Alligator cracking in the worst patches: ~1 m blocks, some broken out into potholes.
    if (k.heavy > 0.05) {
        float a = voronoiBorder(uv / 1.1, 37u, cell, pair) * 1.1;
        if (pair < k.heavy * 1.3 && a < k.distance) { k.distance = a; k.halfWidth = open * 0.8; }
        uint ph = structureHash(cell, 41u);
        if (structureUnit(ph) < 0.07 * k.heavy * k.heavy) k.pothole = smoothstep(0.02, 0.12, a);
    }
    // Each crack opens and closes along its length.
    k.halfWidth *= 0.35 + 1.3 * structureNoise(uv / 0.31 + 4.7);
    // Hairline cracks branching across the blocks, only where a pixel can show a millimetre line.
    if (footprint < 0.012) {
        float hair = voronoiBorder(uv / 0.7 + 13.0, 53u, cell, pair) * 0.7;
        if (pair < 0.25 * neglect + 0.2 * k.heavy) k.hairline = hair;
    }
    return k;
}

// An offset for the zone's structure's ageing patterns, from where it stands, so no two
// runways crack alike.
float2 structureSeed(StructureZoneGPU z) {
    return float2(fmod(z.a.x * 0.37, 613.0), fmod(z.a.y * 0.53, 587.0));
}

// How far the grass has taken the paving's edges back (m), at uv on a runway of half width
// `halfWidth` and half length `halfLength`: broken edges crumbled to a ragged line.
float pavementEdgeLoss(float2 uv, float2 seed, float halfWidth, float halfLength, float upkeep) {
    float neglect = 1.0 - upkeep;
    if (neglect <= 0.0) return 0.0;
    float edge = min(halfWidth - abs(uv.y), halfLength - abs(uv.x));
    float2 q = uv + seed;
    float ragged = neglect * (0.4 + 1.1 * (0.6 * structureNoise(q / 2.3 + 4.4) + 0.4 * structureNoise(q / 0.7 + 9.2)));
    return ragged - edge;   // > 0: this point's paving is gone
}

// Concrete slabs (StructureSurface.concrete): 5 m panels laid square to the zone's box.
// Distance (m) from uv (the zone's own local metres, half extents `extent`) to the nearest
// joint, and the joint's half opening, kept `upkeep` (sealed when kept, opened when not).
constant float kConcretePanel = 5.0;
float concreteJoint(float2 uv, float2 extent, float2 seed, float upkeep, thread float &open) {
    float2 g = (uv + extent) / kConcretePanel;
    float2 c = abs(fract(g) - 0.5);
    open = 0.008 + (1.0 - upkeep) * mix(0.004, 0.02, structureNoise((uv + seed) / 3.7 + 2.1));
    return (0.5 - max(c.x, c.y)) * kConcretePanel;
}

// How far the grass has taken concrete's edges back (m; > 0: this point's slab is gone):
// all round, or only along the sides when the zone's ends meet other paving (d.z < 0).
float concreteEdgeLoss(float2 uv, float2 seed, StructureZoneGPU z, float upkeep) {
    float neglect = 1.0 - upkeep;
    if (neglect <= 0.0) return -1e9;
    float edge = z.d.z < 0.0 ? z.b.x - abs(uv.x) : min(z.b.y - abs(uv.y), z.b.x - abs(uv.x));
    float2 q = uv + seed;
    float ragged = neglect * (0.3 + 0.9 * (0.6 * structureNoise(q / 2.3 + 4.4) + 0.4 * structureNoise(q / 0.7 + 9.2)));
    return ragged - edge;
}

// Signed distance (m) to the nearest ground where only grass or nothing may grow: exact
// within the map's margin (64 m), 1e9 where none is that near.
float structureClearance(const device uchar *sm, float2 xy) {
    uint2 bin = structureBin(sm, xy);
    float best = 1e9;
    if (bin.y == 0u) return best;
    const device StructureZoneGPU *zones = structureZones(sm);
    const device uint *indices = structureIndices(sm);
    for (uint k = 0u; k < bin.y; k++) {
        StructureZoneGPU z = zones[indices[bin.x + k]];
        if (z.c.y < 0.5) continue;
        best = min(best, structureBoxDistance(structureLocal(z, xy), z.b.xy));
    }
    return best;
}

// ---- Buildings (Buildings.swift): what stands on a structure's ground ----
struct BuildingGPU {
    float4 a;   // centre xy, +u axis xy
    float4 b;   // half span (u), half depth (v), floor height, eaves height above the floor
    float4 c;   // arch rise above the eaves, door side (±1 along v), door half width, door height
};

uint structureBuildingCount(const device uchar *sm) {
    const device StructureMapHeader &h = *(const device StructureMapHeader *)sm;
    return h.buildings == 0u ? 0u : *(const device uint *)(sm + h.buildings);
}

BuildingGPU structureBuilding(const device uchar *sm, uint i) {
    const device StructureMapHeader &h = *(const device StructureMapHeader *)sm;
    return ((const device BuildingGPU *)(sm + h.buildings + 16u))[i];
}

// (along, across) metres from a building's centre.
float2 buildingLocal(BuildingGPU g, float2 xy) {
    float2 d = xy - g.a.xy;
    return float2(dot(d, g.a.zw), d.x * g.a.w - d.y * g.a.z);
}

// The barrel roof's height above the floor `s` metres across the span from its middle.
float buildingRoof(BuildingGPU g, float s) {
    float h = g.b.x, rise = g.c.x;
    float r = (h * h + rise * rise) / (2.0 * rise);
    return g.b.w + rise - r + sqrt(max(r * r - s * s, 0.0));
}

// True inside a building's walls (xy only).
bool buildingCovers(const device uchar *sm, float2 xy) {
    uint n = structureBuildingCount(sm);
    for (uint i = 0u; i < n; i++) {
        BuildingGPU g = structureBuilding(sm, i);
        float2 q = buildingLocal(g, xy);
        if (abs(q.x) < g.b.x && abs(q.y) < g.b.y) return true;
    }
    return false;
}

// How much of the sky and of the lit ground around reaches a point (1 in the open): under a
// roof, what the door lets in, falling off with the distance in from it, and a little more
// through the windows and the gaps in the roof.
float buildingShelter(const device uchar *sm, float3 p) {
    uint n = structureBuildingCount(sm);
    for (uint i = 0u; i < n; i++) {
        BuildingGPU g = structureBuilding(sm, i);
        float2 q = buildingLocal(g, p.xy);
        if (abs(q.x) > g.b.x - 0.01 || abs(q.y) > g.b.y - 0.01) continue;
        if (p.z - g.b.z > buildingRoof(g, q.x) - 0.01) continue;
        float inward = max(g.b.y - g.c.y * q.y, 0.0);
        // Beside the door's edges the piers shade a little more.
        float edge = smoothstep(g.c.z - 2.0, g.c.z + 1.0, abs(q.x));
        float door = 1.0 / (1.0 + (inward / 10.0) * (inward / 10.0) + 1.5 * edge * saturate(inward / 3.0));
        return max(door, 0.15);
    }
    return 1.0;
}

"""

#if canImport(Metal)
import Metal

extension StructureLookup {
    /// Where every shader that asks finds the map.
    static let bufferIndex = 20

    func makeBuffer(device: MTLDevice) throws -> MTLBuffer {
        let data = bytes()
        guard let buffer = data.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                                     options: .storageModeShared) }) else {
            throw RendererError.setup("structure map")
        }
        buffer.label = "structure map"
        return buffer
    }
}
#endif
