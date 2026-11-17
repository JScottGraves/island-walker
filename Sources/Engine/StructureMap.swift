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
///
/// A zone is an oriented box with what the ground is, what grows on it and whether it is
/// graded; each structure's zones are listed innermost first, then one extra zone for its
/// cut and fill slopes (the graded box grown by `embankment`: natural ground, grass only).
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
        /// The structure's graded half length and width (m); runway designators painted at
        /// the −u end (read travelling +u) and at the +u end, 0 for none.
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
        /// Signed distance outside the structure's graded box (negative inside).
        public var edge: Float

        public static let open = Sample(surface: 0, plants: 0, graded: false, gradeHeight: 0, u: 0, v: 0, zone: -1,
                                        structure: -1, edge: 1e9)
    }

    public private(set) var zones: [Zone] = []
    public private(set) var bins: [SIMD2<UInt32>] = []
    public private(set) var indices: [UInt32] = []
    public private(set) var origin = SIMD2<Float>(0, 0)
    public private(set) var binsX = 0, binsY = 0
    public let binSize: Float
    /// Zones are listed in every bin within this many metres of them, so clearance (the
    /// distance to ground where nothing tall grows) is exact out to it.
    public let margin: Float

    public init(structures: [Structure], binSize: Float = 256, margin: Float = 64) {
        self.binSize = binSize
        self.margin = margin
        for (k, s) in structures.enumerated() {
            let axis = s.axis
            let ghl = Float(s.gradedHalfLength), ghw = Float(s.gradedHalfWidth)
            let designators = Self.designators(s)
            func zone(_ hl: Double, _ hw: Double, _ surface: StructureSurface, _ plants: PlantRule, _ graded: Bool) -> Zone {
                Zone(a: SIMD4(Float(s.x), Float(s.y), Float(axis.x), Float(axis.y)),
                     b: SIMD4(Float(hl), Float(hw), Float(s.elevation), Float(s.slope)),
                     c: SIMD4(Float(surface.code), Float(plants.code), graded ? 1 : 0, Float(k)),
                     d: SIMD4(ghl, ghw, Float(designators.0), Float(designators.1)))
            }
            for z in s.zones { zones.append(zone(z.halfLength, z.halfWidth, z.surface, z.plants, z.graded)) }
            if s.embankment > 0, s.gradedHalfLength > 0 {
                zones.append(zone(s.gradedHalfLength + s.embankment, s.gradedHalfWidth + s.embankment, .natural, .grassOnly, false))
            }
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
        guard let text = s.properties["designators"] else { return (0, 0) }
        let parts = text.split(separator: "/").compactMap { Int($0) }
        guard parts.count == 2 else { return (0, 0) }
        var n = Int((s.heading / 10).rounded()) % 36
        if n == 0 { n = 36 }
        // The number nearest the heading is the one read travelling along it.
        func turn(_ d: Int) -> Int { let t = abs(d - n) % 36; return Swift.min(t, 36 - t) }
        return turn(parts[0]) <= turn(parts[1]) ? (parts[0], parts[1]) : (parts[1], parts[0])
    }

    /// The buffer the shaders read.
    public func bytes() -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(Self.headerBytes + zones.count * 64 + bins.count * 8 + indices.count * 4)
        func put<T>(_ value: T) { withUnsafeBytes(of: value) { out += $0 } }
        put(UInt32(zones.count)); put(UInt32(binsX)); put(UInt32(binsY)); put(UInt32(0))
        put(origin.x); put(origin.y); put(binSize); put(margin)
        for z in zones { put(z.a); put(z.b); put(z.c); put(z.d) }
        for b in bins { put(b) }
        for i in indices { put(i) }
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
            return Sample(surface: Int(z.c.x), plants: Int(z.c.y), graded: z.c.z > 0.5,
                          gradeHeight: z.b.z + z.b.w * Swift.min(Swift.max(u, -z.d.x), z.d.x), u: u, v: v,
                          zone: Int(k), structure: Int(z.c.w), edge: Self.boxDistance(u, v, z.d.x, z.d.y))
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
    uint pad;
    float2 origin;
    float binSize;
    float margin;
};

struct StructureZoneGPU {
    float4 a;   // centre xy, +u axis xy
    float4 b;   // half length, half width, grade line elevation, slope
    float4 c;   // surface, plants, graded, structure index
    float4 d;   // graded half length, half width, designator at the −u end, at the +u end
};

constant int kSurfaceNatural = 0;
constant int kSurfacePaved = 1;
constant int kSurfaceMownGrass = 2;
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
    if (s.structure < 0 || s.surface == kSurfacePaved) return 0.0;
    if (s.surface != kSurfaceMownGrass && s.edge > 2.5) return 0.0;
    float ragged = 0.7 * structureNoise(xy / 1.7 + 3.1) + 0.3 * structureNoise(xy / 0.55 + 8.9) - 0.5;
    return 1.0 - smoothstep(-1.5, 1.5, s.edge + 2.4 * ragged);
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
