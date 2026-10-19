import Flora

/// Metal for the ground cover (GroundCoverRenderer; docs/flora.md §12). The prelude is
/// spliced into terrainShaderSource ahead of terrainFragment, which tints the ground
/// texture with it; the rest after floraShaderSource, whose scatter helpers it reuses.
let groundCoverPreludeSource = GroundCover.metalTable + groundCoverNoiseSource + groundCoverWindSource

let groundCoverWindSource = """
// The trade wind's gusts over the ground at xy (m) and time `clock` (s): its direction and
// strength (cloudWind) times a 0…1.6 gust field of two octaves travelling downwind, so a
// gust is seen to roll across a meadow. A pure function of position and time, so the
// trees can share it and a vertex's previous position (for motion vectors) is this at the
// previous clock. Returns the push, metres per metre of blade height.
float2 tradeWindGust(float4 cloudWind, float clock, float2 xy) {
    float2 wind = cloudWind.xy;
    float speed = length(wind);
    if (speed < 1e-3) return float2(0.0);
    float2 along = wind / speed;
    float2 across = float2(-along.y, along.x);
    // Gusts travel at the wind speed; the fine octave a little faster, and across the wind.
    float2 q = float2(dot(xy, along) - clock * speed, dot(xy, across));
    float g = 0.65 * coverNoise(q / float2(14.0, 9.0)) + 0.35 * coverNoise(q / float2(4.0, 3.0) + 31.7 - clock * 0.4);
    float gust = 0.25 + 1.35 * smoothstep(0.25, 0.8, g);
    return along * (0.035 * speed / 7.5) * gust;
}

"""

/// Noise, patches and the far tint (the cover kernels share them).
let groundCoverNoiseSource = """
float coverHash(float2 c) {
    uint h = uint(int(c.x)) * 0x8da6b343u ^ uint(int(c.y)) * 0xd8163841u;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
    return float(h & 0xffffu) / 65535.0;
}

float coverNoise(float2 q) {
    float2 i = floor(q), f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(coverHash(i), coverHash(i + float2(1.0, 0.0)), f.x),
               mix(coverHash(i + float2(0.0, 1.0)), coverHash(i + float2(1.0, 1.0)), f.x), f.y);
}

// How much of a community's cover grows at p (0…1): patches a few metres to tens of
// metres across, the patchier the community the more bare ground between.
float coverPatch(float2 p, float patchiness) {
    float n = 0.6 * coverNoise(p / 7.0 + 3.1) + 0.4 * coverNoise(p / 2.3 + 17.9);
    return smoothstep(patchiness - 0.12, patchiness + 0.12, n);
}

// The ground texture's tint where cover thins out with distance: the cover's colour and
// how much of the ground it hides, by the community weights (w0…w2, Look.communities
// order) and the same patches the blades grow in.
float4 coverFarTint(float2 p, float4 w0, float4 w1, float4 w2) {
    float w[11] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z };
    float3 colour = 0.0;
    float amount = 0.0, total = 0.0;
    for (int i = 0; i < 11; i++) {
        float4 c = kCoverCommunity[i];
        float k = w[i] * c.w * coverPatch(p, c.z);
        int s = int(c.x);
        colour += k * mix(kCoverRoot[s], kCoverTip[s], 0.6);
        amount += k;
        total += w[i];
    }
    return amount > 1e-4 ? float4(colour / amount, amount / max(total, 1e-3)) : float4(0.0);
}

"""

/// The cover's kernels and passes, after floraShaderSource.
let groundCoverShaderSource = """
// ---- Ground cover (GroundCoverRenderer) ----

// One generated blade, in its tile's slot: world xy, height factor and bend jitter; its
// facing (unit xy, the way its tip leans), width factor and species + a 0…1 rank (thinning
// keeps the lowest ranks).
struct CoverSeed {
    float4 a;               // x, y, height factor, bend factor
    float4 b;               // facing x, y, width factor, species + rank
};

// A blade drawn this frame.
struct CoverBlade {
    float4 root;            // world position (sunk a little), height (m)
    float4 facing;          // facing x, y, width (m), bend
    float4 ground;          // ground normal xyz, species + per-blade random
};

struct CoverParams {
    float4 frustum[6];      // camera-relative planes (unjittered)
    float4 range;           // full-density distance, end distance (m), density scale, height above ground (m)
    float4 counts;          // slots per side, slot capacity, draw capacity, generate count
    float4 lods;            // LOD0 distance (m), render height (px), unused, unused
};

// Plants the tree scatter puts near p (canopy trees and shrubs), as floraCandidate picks
// them: their stems are kept clear. Only the cheap part of the pick (cell hash, community,
// density, plant), not its slope and sea tests, so it errs on the side of clearing.
bool coverNearStem(constant FloraParams &f, texture2d<float> b0, texture2d<float> b1, texture2d<float> b2, float2 p) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    for (int layer = 0; layer < 2; layer++) {
        bool canopy = layer == 0;
        float cellSize = canopy ? f.cells.x : f.cells.y;
        float clear = canopy ? 0.7 : 0.45;
        int2 c0 = int2(floor(p / cellSize));
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int2 cell = c0 + int2(i, j);
                uint h = floraHash(cell, canopy ? 0u : 1u, uint(f.counts.w));
                float2 xy = (float2(cell) + float2(floraRandom(h, 0), floraRandom(h, 1))) * cellSize;
                if (distance(xy, p) > clear) continue;
                float2 uv = xy / f.lods.w;
                int community = floraPickCommunity(b0.sample(linear, uv, level(0)), b1.sample(linear, uv, level(0)),
                                                   b2.sample(linear, uv, level(0)), floraRandom(h, 2));
                if (community < 0) continue;
                float density = canopy ? f.community[community].x : f.community[community].y;
                if (floraRandom(h, 3) >= density * cellSize * cellSize * 1e-4) continue;
                if (floraPickPlant(canopy ? f.canopyShare : f.shrubShare, community, floraRandom(h, 4)) >= 0) return true;
            }
        }
    }
    return false;
}

// Resets the counts of the slots generated this frame.
kernel void coverClear(const device uint *generate [[buffer(1)]],
                       device atomic_uint *slotCounts [[buffer(2)]],
                       constant CoverParams &cp [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= uint(cp.counts.w)) return;
    atomic_store_explicit(&slotCounts[generate[i]], 0u, memory_order_relaxed);
}

// One candidate per grid cell of each tile entering the ring: a blade, or none. Kept until
// the tile leaves the ring. Density is the community's × slope window × wetness × canopy
// shade × patches; stems of trees and shrubs are kept clear.
kernel void coverGenerate(constant Uniforms &u [[buffer(0)]],
                          const device uint *generate [[buffer(1)]],
                          device atomic_uint *slotCounts [[buffer(2)]],
                          constant CoverParams &cp [[buffer(3)]],
                          const device int4 *slotTiles [[buffer(4)]],
                          device CoverSeed *seeds [[buffer(5)]],
                          constant TerrainParams &t [[buffer(6)]],
                          const device int *pages [[buffer(7)]],
                          constant FloraParams &f [[buffer(8)]],
                          constant uint &haveFlora [[buffer(9)]],
                          texture2d_array<ushort, access::read> tiles [[texture(0)]],
                          texture2d<float> w0 [[texture(1)]],
                          texture2d<float> w1 [[texture(2)]],
                          texture2d<float> w2 [[texture(3)]],
                          texture2d<float> canopyCover [[texture(4)]],
                          texture2d<float> b0 [[texture(5)]],
                          texture2d<float> b1 [[texture(6)]],
                          texture2d<float> b2 [[texture(7)]],
                          uint3 gid [[thread_position_in_grid]]) {
    if (gid.z >= uint(cp.counts.w) || gid.x >= uint(kCoverGrid) || gid.y >= uint(kCoverGrid)) return;
    uint slot = generate[gid.z];
    int2 tile = slotTiles[slot].xy;
    int2 cell = tile * kCoverGrid + int2(gid.xy);
    float step = kCoverTile / float(kCoverGrid);
    uint h = floraHash(cell, 9u, uint(f.counts.w));
    float2 p = (float2(cell) + float2(floraRandom(h, 0), floraRandom(h, 1))) * step;

    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 uv = p / t.domainSize;
    float4 a = w0.sample(linear, uv, level(0)), b = w1.sample(linear, uv, level(0)), c = w2.sample(linear, uv, level(0));
    int community = floraPickCommunity(a, b, c, floraRandom(h, 2));
    if (community < 0) return;
    float4 cc = kCoverCommunity[community];
    int species = int(cc.x);
    float4 sa = kCoverShapeA[species], sb = kCoverShapeB[species];
    float density = cc.y * cp.range.z * step * step;
    // Slope from the data's own 10 m grid: nothing steeper than the species grows on.
    float e = t.cellMetres;
    float2 grad = float2(levelHeight(t, pages, tiles, 0, p + float2(e, 0.0)) - levelHeight(t, pages, tiles, 0, p - float2(e, 0.0)),
                         levelHeight(t, pages, tiles, 0, p + float2(0.0, e)) - levelHeight(t, pages, tiles, 0, p - float2(0.0, e))) / (2.0 * e);
    float slope = length(grad);
    density *= 1.0 - smoothstep(0.7 * sa.x, sa.x, slope);
    // Nothing grows on the beach face the waves run up (the swash; terrainFragment): bare below
    // ~1.5 m above the sea, thinning in up to 2.5 m.
    density *= smoothstep(1.5, 2.5, levelHeight(t, pages, tiles, 0, p) - f.lods.z);
    // Wetter communities nearby help the forest sedge.
    float wet = b.y + b.z + a.z + c.x;  // wet forest, riparian, marsh, bog
    if (species == 3) density *= 0.7 + 0.3 * saturate(2.0 * wet);
    float shade = canopyCover.sample(linear, uv, level(0)).r;
    density *= 1.0 - (1.0 - sb.w) * shade;
    density *= coverPatch(p, cc.z);
    // Clumps: Voronoi cells of the species' spacing, sharing a lean and a height; bunch
    // species keep only the blades near a clump's centre.
    float spacing = sb.y;
    float2 q = p / spacing;
    float2 q0 = floor(q);
    float best = 1e9;
    float2 centre = 0.0, clumpCell = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float2 k = q0 + float2(float(i), float(j));
            float2 o = k + float2(coverHash(k * 1.7 + 0.3), coverHash(k * 2.3 + 5.1));
            float dd = distance(q, o);
            if (dd < best) { best = dd; centre = o * spacing; clumpCell = k; }
        }
    }
    float bunch = sb.z;
    if (bunch > 0.0) {
        float r = bunch * spacing * (0.7 + 0.6 * coverHash(clumpCell + 9.7));
        if (best * spacing > r) return;
    }
    if (floraRandom(h, 3) >= density) return;
    if (haveFlora != 0u && coverNearStem(f, b0, b1, b2, p)) return;

    // Facing: away from the clump's centre (a tussock fans out), with some spread; the
    // clump leans one way together.
    float2 out = p - centre;
    float lean = coverHash(clumpCell + 3.3) * 6.2831853;
    float2 dir = length(out) > 1e-3 ? normalize(out) : float2(cos(lean), sin(lean));
    float spread = (floraRandom(h, 4) - 0.5) * 1.6;
    dir = normalize(mix(dir, float2(cos(lean), sin(lean)), 0.35));
    dir = float2(dir.x * cos(spread) - dir.y * sin(spread), dir.x * sin(spread) + dir.y * cos(spread));
    float clumpHeight = 0.75 + 0.5 * coverHash(clumpCell + 1.9);
    // Bunches are tallest in the middle.
    float centreFactor = bunch > 0.0 ? 1.0 - 0.35 * saturate(best * spacing / max(bunch * spacing, 1e-3)) : 1.0;
    uint index = atomic_fetch_add_explicit(&slotCounts[slot], 1u, memory_order_relaxed);
    if (index >= uint(cp.counts.y)) return;
    CoverSeed s;
    s.a = float4(p, clumpHeight * centreFactor * (0.7 + 0.6 * floraRandom(h, 5)), 0.7 + 0.6 * floraRandom(h, 6));
    s.b = float4(dir, 0.7 + 0.6 * floraRandom(h, 7), float(species) + 0.999 * floraRandom(h, 8));
    seeds[slot * uint(cp.counts.y) + index] = s;
}

// Every frame, every kept blade: within range (full density to range.x, thinning by rank
// and shrinking to range.y), in view and not hidden (last frame's pyramid), then on the
// ground the terrain draws there, into LOD0 or LOD1's list.
kernel void coverCull(constant Uniforms &u [[buffer(0)]],
                      const device uint *slotCounts [[buffer(2)]],
                      constant CoverParams &cp [[buffer(3)]],
                      const device CoverSeed *seeds [[buffer(5)]],
                      constant TerrainParams &t [[buffer(6)]],
                      const device int *pages [[buffer(7)]],
                      constant DetailParams &d [[buffer(10)]],
                      device atomic_uint *drawCounts [[buffer(11)]],
                      device CoverBlade *blades [[buffer(12)]],
                      constant FloraHzb &hzbInfo [[buffer(13)]],
                      texture2d_array<ushort, access::read> tiles [[texture(0)]],
                      texture2d_array<float, access::read> detail [[texture(8)]],
                      texture2d<float> hzb [[texture(9)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint slot = gid.y;
    uint capacity = uint(cp.counts.y);
    if (gid.x >= min(slotCounts[slot], capacity)) return;
    CoverSeed s = seeds[slot * capacity + gid.x];
    float2 rel = s.a.xy - u.cameraPosition.xy;
    float end = cp.range.y;
    if (dot(rel, rel) > end * end) return;
    int species = int(s.b.w);
    float rank = fract(s.b.w);
    float4 sa = kCoverShapeA[species];
    // Fade out when flying: none once the camera stands 25 m above the ground.
    float above = cp.range.w;
    float keep = (1.0 - smoothstep(6.0, 25.0, above));
    float horizontal = length(rel);
    float far = smoothstep(cp.range.x, end * 0.94, horizontal);
    keep *= 1.0 - far;
    if (rank >= keep) return;
    float height = mix(sa.x, sa.y, saturate(s.a.z - 0.5)) * s.a.z * mix(1.0, 0.65, far);
    float rough = levelHeight(t, pages, tiles, min(2, t.levels - 1), s.a.xy);
    float3 centre = float3(s.a.xy, rough + 0.5 * height) - u.cameraPosition;
    float radius = height + 0.5;
    for (int i = 0; i < 6; i++) {
        if (dot(cp.frustum[i].xyz, centre) + cp.frustum[i].w < -radius - 2.0) return;
    }
    float2 gradient;
    float ground = floraGround(t, d, detail, pages, tiles, s.a.xy, length(float3(rel, u.cameraPosition.z - rough)), gradient);
    if (length(gradient) > 1.43) return;
    float3 root = float3(s.a.xy, ground);
    if (floraOccluded(hzbInfo, hzb, root + float3(0.0, 0.0, 0.5 * height), radius)) return;
    float dist = distance(root, u.cameraPosition);
    // Thinner blades stay at least a pixel wide; thinned ones widen to keep the coverage.
    float pixel = dist * 1.1547 / cp.lods.y;
    float width = max(sa.z * s.b.z * mix(1.0, 1.8, far), pixel);
    int lod = dist < cp.lods.x ? 0 : 1;
    uint drawCapacity = uint(cp.counts.z);
    uint index = atomic_fetch_add_explicit(&drawCounts[lod], 1u, memory_order_relaxed);
    if (index >= drawCapacity) return;
    CoverBlade b;
    b.root = float4(root - float3(0.0, 0.0, 0.03), height);
    b.facing = float4(s.b.xy, width, sa.w * s.a.w);
    b.ground = float4(normalize(float3(-gradient, 1.0)), float(species) + 0.999 * fract(rank * 7.31 + s.a.z));
    blades[uint(lod) * drawCapacity + index] = b;
}

// MTLDrawPrimitivesIndirectArguments.
struct CoverDrawArguments {
    uint vertexCount;
    uint instanceCount;
    uint vertexStart;
    uint baseInstance;
};

// Leaves per tuft, near and far.
constant uint kCoverLeaves0 = 4u, kCoverLeaves1 = 2u;

kernel void coverArguments(const device uint *drawCounts [[buffer(11)]],
                           device CoverDrawArguments *args [[buffer(14)]],
                           constant CoverParams &cp [[buffer(3)]],
                           uint lod [[thread_position_in_grid]]) {
    if (lod > 1u) return;
    uint capacity = uint(cp.counts.z);
    // Segments 5 and 3, two vertices each plus the tip; each blade a tuft of leaves.
    uint leaves = lod == 0u ? kCoverLeaves0 : kCoverLeaves1;
    args[lod].vertexCount = lod == 0u ? 12u : 8u;
    args[lod].instanceCount = min(drawCounts[lod], capacity) * leaves;
    args[lod].vertexStart = 0u;
    args[lod].baseInstance = lod * capacity * leaves;
}

struct CoverOut {
    float4 position [[position, invariant]];
    float3 world;
    float3 normal;
    float3 ground;
    float t [[user(along)]];
    float tint;
    float dead [[flat]];
    int species [[flat]];
};

// A blade as a quadratic Bézier from its root, built from the vertex id (a triangle strip,
// two vertices per segment, both at the tip for the last), leaning with the ground's normal
// and pushed by the gusts.
vertex CoverOut coverVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                            constant Uniforms &u [[buffer(1)]],
                            const device CoverBlade *blades [[buffer(2)]],
                            constant uint &segments [[buffer(3)]]) {
    // One instance per leaf of a tuft: the leaves fan out round the blade's facing from
    // roots a few centimetres apart, the outer ones shorter.
    uint leaves = segments == 5u ? kCoverLeaves0 : kCoverLeaves1;
    uint leaf = iid % leaves;
    CoverBlade b = blades[iid / leaves];
    int species = int(b.ground.w);
    float r = fract(b.ground.w * 13.7 + float(leaf) * 0.618);
    float fan = (float(leaf) - 0.5 * float(leaves - 1u)) * 0.7 + (r - 0.5) * 0.5;
    b.facing.xy = float2(b.facing.x * cos(fan) - b.facing.y * sin(fan), b.facing.x * sin(fan) + b.facing.y * cos(fan));
    b.root.xy += b.facing.xy * (0.04 * r);
    float height = b.root.w * (leaf == 0u ? 1.0 : 0.7 + 0.3 * r);
    float3 n = b.ground.xyz;
    float3 up = normalize(float3(0.0, 0.0, 1.0) + n * 0.5);
    float3 facing = normalize(float3(b.facing.xy, 0.0) - up * dot(float3(b.facing.xy, 0.0), up));
    float bend = b.facing.w;
    float3 p0 = b.root.xyz;
    float3 p1 = p0 + up * (height * 0.6) + facing * (height * bend * 0.15);
    float3 p2 = p0 + up * (height * (1.0 - 0.45 * bend)) + facing * (height * bend * 0.75);
    float2 gust = tradeWindGust(u.cloudWind, u.cloudLayer.z, p0.xy) * height;
    p2.xy += gust * 1.0;
    p1.xy += gust * 0.35;
    // Keep the blade's length about the same as it bends over in the wind.
    p2.z -= 0.5 * dot(gust, gust) / max(height, 0.05);

    uint seg = min(vid / 2u, segments);
    float t = float(seg) / float(segments);
    float side = (vid & 1u) == 0u ? -1.0 : 1.0;
    float3 pos = (1.0 - t) * (1.0 - t) * p0 + 2.0 * t * (1.0 - t) * p1 + t * t * p2;
    float3 tangent = normalize(2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1));
    float3 across;
    if (kCoverRound[species] != 0) {
        across = normalize(cross(tangent, normalize(u.cameraPosition - pos)));
    } else {
        across = normalize(cross(tangent, facing));
    }
    float taper = kCoverRound[species] != 0 ? 1.0 - 0.6 * t : pow(1.0 - t, 0.8);
    float halfWidth = 0.5 * b.facing.z * taper;
    pos += across * (side * halfWidth);
    float3 normal = normalize(cross(across, tangent));
    // Rounded: the normal tilts out towards each edge.
    normal = normalize(normal + across * (side * 0.6));

    CoverOut o;
    o.position = u.viewProjection * float4(pos - u.cameraPosition, 1.0);
    o.world = pos;
    o.normal = normal;
    o.ground = n;
    o.t = t;
    o.tint = 0.8 + 0.4 * fract(b.ground.w);
    o.dead = r < 0.16 ? 0.4 + 2.5 * r : 0.0;
    o.species = species;
    return o;
}

// Prepass with ray-traced shadows: the surface, marked as a tree pixel (two-sided, so the
// sun rays treat it like leaves; within a metre of the ground it takes the ground's).
fragment uint4 coverSurface(CoverOut in [[stage_in]], bool front [[front_facing]]) {
    float3 n = normalize(front ? in.normal : -in.normal);
    float3 p = in.world;
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z),
                 (pack_float_to_snorm2x16(octEncode(n)) & ~kImpostorSurface) | kTreeSurface);
}

fragment void coverDepth(CoverOut in [[stage_in]]) {}

// Shading: equal depth against the prepass. Colour from root to tip, the root blended into
// the ground (colour, normal and occlusion), and sun through the blade when looking
// towards a low sun.
[[early_fragment_tests]]
fragment float4 coverFragment(CoverOut in [[stage_in]], bool front [[front_facing]],
                              constant Uniforms &u [[buffer(1)]],
                              texture2d<float> skyView [[texture(0)]],
                              texture2d<float> clouds [[texture(2)]],
                              texture2d<float> cloudDistance [[texture(3)]],
                              texture2d<float> cloudShadows [[texture(4)]],
                              texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                              texture2d<float> skyVisibility [[texture(8)]],
                              texture2d_array<float, access::read> horizons [[texture(9)]]) {
    int s = in.species;
    float t = saturate(in.t);
    float3 albedo = mix(kCoverRoot[s], kCoverTip[s], smoothstep(0.0, 0.85, t)) * in.tint;
    // About one leaf in six dry and straw-coloured, as in any stand of grass or sedge.
    albedo = mix(albedo, float3(0.30, 0.25, 0.13) * in.tint, in.dead);
    float3 blade = normalize(front ? in.normal : -in.normal);
    // Grass lit as a layer more than as single blades: the normal halfway to the ground's,
    // all of the way at the root.
    float3 n = normalize(mix(in.ground, blade, 0.5 * smoothstep(0.0, 0.45, t)));
    float occlusion = mix(0.35, 1.0, smoothstep(0.0, 0.6, t));
    float3 l = normalize(u.sunDirection);
    float nDotL = saturate((dot(n, l) + 0.4) / 1.4);
    float cloudSun = cloudSunlight(u, cloudShadows, in.world);
    float sunVisibility = 0.0;
    if (l.z > -0.01 && cloudSun > 0.01) {
        if (kRayTraced) {
            sunVisibility = tracedSunAt(tracedSun, uint2(in.position.xy));
        } else {
            int side = int(horizons.get_width());
            uint2 cell = uint2(clamp(int2(in.world.xy / u.cloudLayer.w * float(side)), int2(0), int2(side - 1)));
            float2 bounds;
            float margin = horizonMargin(horizons.read(cell, 0) * M_PI_2_F, horizons.read(cell, 1) * M_PI_2_F,
                                         horizons.read(cell, 2) * M_PI_2_F, horizons.read(cell, 3) * M_PI_2_F, l, bounds);
            sunVisibility = smoothstep(0.0, 1.0, saturate((margin + 0.01) / 0.02));
        }
    }
    sunVisibility *= cloudSun;
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float sky = saturate(skyVisibility.sample(linear, in.world.xy / u.cloudLayer.w, level(0)).r);
    float3 color = litSurface(u, albedo, n, nDotL, sunVisibility, sky, occlusion, cloudSun);
    // Sun through the blade looking towards a low sun.
    float back = saturate(dot(normalize(in.world - u.cameraPosition), l));
    float low = 1.0 - smoothstep(0.2, 0.7, l.z);
    color += albedo * u.sunRadiance * (0.6 * sunVisibility * back * back * low * t);
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}

"""
