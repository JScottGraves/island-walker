import Flora

/// Metal source for the native trees (docs/flora.md §4–5, FloraRenderer): the GPU scatter
/// that builds each frame's instances from a world-space hash grid and the biome weights,
/// and the tree passes. Spliced after terrainShaderSource, whose surface heights, lighting
/// tail (litSurface, finishSurface) and sun-shadow helpers it reuses.
let floraShaderSource = FloraMaterial.metalTable + """
// The scatter's settings, laid out as FloraRenderer.Params.
struct FloraParams {
    float4 planes[6];       // view frustum, camera-relative (a point is inside when dot ≥ 0)
    float4 cells;           // canopy cell, shrub cell (m), canopy range, shrub range (m)
    float4 lods;            // LOD0 distance, unused, sea level, domain size (m)
    float4 lods2;           // LOD1 distance (m), fade progress per frame (of 4095), frame number
    float4 origins;         // first canopy cell (xy), first shrub cell (zw)
    float4 counts;          // canopy cells per side, shrub cells per side, bucket capacity, seed
    float4 community[11];   // canopy stems/ha, shrub stems/ha
    float4 canopyShare[22]; // per community, 8 cumulative shares in Plant order
    float4 shrubShare[22];
    float4 plant[7];        // shortest and tallest height (m), reference height (m), steepest slope (tan)
};

// One tree, as FloraRenderer.Instance.
struct FloraInstance {
    packed_float3 position; // trunk base, world metres
    float scale;            // of the reference-height mesh
    float2 rotation;        // cos, sin of the yaw
    float2 lean;            // horizontal shear per metre of height
    float fade;             // dithered cross-fade: > 0 drawn where noise < fade, < 0 where noise ≥ −fade, 0 always
    float tint;             // albedo multiplier, for stand variety
    float phase;            // wind phase
    float height;           // metres
};

struct FloraVertex {
    packed_float3 position;
    packed_float3 normal;
    float2 uv;
    float ao;
    float sway;
    float flutter;
    float material;
};

constant int kFloraPlants = 7;
constant int kFloraVariants = \(Flora.variants);
// Buckets per variant: the two mesh LODs, then the impostor (LOD2).
constant int kFloraLods = \(Flora.lods + 1);

uint floraHash(int2 cell, uint layer, uint seed) {
    uint h = uint(cell.x) * 0x8da6b343u ^ uint(cell.y) * 0xd8163841u ^ (layer + seed * 0x9e3779b9u) * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// The k-th uniform number in [0, 1) of a hash.
float floraRandom(uint h, uint k) {
    uint x = h ^ (k * 0x9e3779b9u);
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x >> 8) * (1.0 / 16777216.0);
}

// Height of the terrain as drawn at a world position, `dist` from the camera: the CDLOD
// level that distance falls in, on that level's triangles (and the next level's, as far
// as it has morphed), so trunks meet the ground the terrain actually draws.
float floraGround(constant TerrainParams &t, constant DetailParams &d, texture2d_array<float, access::read> detail,
                  const device int *pages, texture2d_array<ushort, access::read> tiles, float2 xy, float dist,
                  thread float2 &gradient) {
    int lod = 0;
    while (lod + 1 < t.lodCount && dist > at(t.morphEnd, lod)) lod++;
    float ms = at(t.morphStart, lod), me = at(t.morphEnd, lod);
    float morph = saturate((dist - ms) / max(me - ms, 1.0));
    float h = 0.0;
    gradient = float2(0.0);
    for (int k = 0; k < 2; k++) {
        float w = k == 0 ? 1.0 - morph : morph;
        if (w <= 0.0) continue;
        float spacing = ldexp(t.cellMetres, lod - t.detailLods + k);
        float2 g = floor(xy / spacing);
        float2 f = xy / spacing - g;
        float2 o = g * spacing;
        float m = float(k);
        float a = surfaceHeight(t, d, detail, pages, tiles, lod, m, o);
        float b = surfaceHeight(t, d, detail, pages, tiles, lod, m, o + float2(spacing, 0.0));
        float c = surfaceHeight(t, d, detail, pages, tiles, lod, m, o + float2(0.0, spacing));
        float e = surfaceHeight(t, d, detail, pages, tiles, lod, m, o + float2(spacing, spacing));
        // The node grid's split: triangles (a, b, e) and (a, e, c).
        float2 slope = f.x > f.y ? float2(b - a, e - b) : float2(e - c, c - a);
        h += w * (a + slope.x * f.x + slope.y * f.y);
        gradient += w * slope / spacing;
    }
    return h;
}

// Picks index i with probability weights[i] / sum, for a uniform r in [0, 1).
int floraPickCommunity(float4 w0, float4 w1, float4 w2, float r) {
    float w[11] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z };
    float sum = 0.0;
    for (int i = 0; i < 11; i++) sum += max(w[i], 0.0);
    if (sum < 1e-3) return -1;
    float target = r * sum;
    for (int i = 0; i < 11; i++) {
        target -= max(w[i], 0.0);
        if (target < 0.0) return i;
    }
    return 10;
}

int floraPickPlant(constant float4 *shares, int community, float r) {
    float4 a = shares[community * 2], b = shares[community * 2 + 1];
    float cumulative[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    for (int i = 0; i < kFloraPlants; i++) if (r < cumulative[i]) return i;
    return -1;
}

void floraAppend(device atomic_uint *counts, device FloraInstance *instances, uint capacity, int bucket, FloraInstance inst) {
    uint index = atomic_fetch_add_explicit(&counts[bucket], 1u, memory_order_relaxed);
    if (index < capacity) instances[uint(bucket) * capacity + index] = inst;
}

// The plant (if any) a hash-grid cell grows: layer 0 the canopy, layer 1 the shrubs and
// subcanopy. A tree always stands at the same place on revisits, and nothing is stored
// per tree. Fills everything in `inst` but the fade. Shared by the draw scatter and the
// shadow casters' scatter, so shadows stand exactly where the trees are drawn.
bool floraCandidate(constant Uniforms &u, constant FloraParams &f, constant TerrainParams &t,
                    constant DetailParams &d, const device int *pages, texture2d_array<ushort, access::read> tiles,
                    texture2d_array<float, access::read> detail, texture2d<float> biome0, texture2d<float> biome1,
                    texture2d<float> biome2, bool canopy, int2 cell, float range,
                    thread FloraInstance &inst, thread int &plantOut, thread int &variantOut) {
    float cellSize = canopy ? f.cells.x : f.cells.y;
    uint h = floraHash(cell, canopy ? 0u : 1u, uint(f.counts.w));
    float2 xy = (float2(cell) + float2(floraRandom(h, 0), floraRandom(h, 1))) * cellSize;
    float2 rel = xy - u.cameraPosition.xy;
    if (dot(rel, rel) > range * range) return false;

    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 uv = xy / f.lods.w;
    float4 w0 = biome0.sample(linear, uv, level(0)), w1 = biome1.sample(linear, uv, level(0)),
           w2 = biome2.sample(linear, uv, level(0));
    int community = floraPickCommunity(w0, w1, w2, floraRandom(h, 2));
    if (community < 0) return false;
    float density = canopy ? f.community[community].x : f.community[community].y;
    if (floraRandom(h, 3) >= density * cellSize * cellSize * 1e-4) return false;
    int plant = floraPickPlant(canopy ? f.canopyShare : f.shrubShare, community, floraRandom(h, 4));
    if (plant < 0) return false;
    float4 info = f.plant[plant];

    float2 gradient;
    // On the terrain level drawn at this distance.
    float ground = floraGround(t, d, detail, pages, tiles, xy, length(float3(rel, u.cameraPosition.z)), gradient);
    if (ground < f.lods.z + 0.5 || length(gradient) > info.w) return false;
    // Taller where wetter and sheltered (site, in biome2.w), with some spread.
    float site = w2.w;
    float height = mix(info.x, info.y, 0.2 + 0.8 * site) * (0.75 + 0.4 * floraRandom(h, 5));

    inst.position = packed_float3(xy.x, xy.y, ground);
    inst.scale = height / info.z;
    float yaw = floraRandom(h, 6) * 6.2831853;
    inst.rotation = float2(cos(yaw), sin(yaw));
    float leanAngle = floraRandom(h, 7) * 6.2831853;
    inst.lean = float2(cos(leanAngle), sin(leanAngle)) * (0.04 * floraRandom(h, 8));
    inst.fade = 0.0;
    inst.tint = 0.85 + 0.3 * floraRandom(h, 9);
    inst.phase = floraRandom(h, 10) * 6.2831853;
    inst.height = height;
    plantOut = plant;
    variantOut = int(floraRandom(h, 11) * float(kFloraVariants)) % kFloraVariants;
    return true;
}

// Each plant's LOD state across frames, so a plant crossing a LOD boundary fades between
// its two LODs over a fraction of a second, not over a band of distance: a fixed dither
// pattern, which a distance band leaves on every plant inside it at once, then shows only
// briefly on the few plants crossing. An open-addressed table keyed by the plant's grid
// cell (FloraRenderer.lodStates). Value: current LOD (bits 0–1; 3 = not drawn), the LOD it
// is fading from (2–3), fade progress (4–15, 0…4095) and the frame last seen (16–31).
constant uint kLodTableMask = \(FloraRenderer.lodStates - 1)u;
constant uint kLodProgressMax = 4095u;
// A plant switches LOD only once its distance is this far past the boundary, so standing
// still at a boundary doesn't make it flicker between the two.
constant float kLodHysteresis = 0.04;

uint floraLodKey(int2 cell, bool canopy) {
    uint h = uint(cell.x) * 0x9e3779b1u ^ uint(cell.y) * 0x85ebca77u ^ (canopy ? 0x5bd1e995u : 0u);
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
    return h | 1u;  // 0 marks an empty slot
}

// The slot holding `key`, claimed if absent (taking an empty slot or one unseen for a
// second), or -1 when the probe runs out.
int floraLodSlot(device atomic_uint *keys, device uint *values, uint key, uint frame) {
    for (uint i = 0u; i < 8u; i++) {
        uint slot = (key + i * 0x9e37u) & kLodTableMask;
        uint k = atomic_load_explicit(&keys[slot], memory_order_relaxed);
        if (k == key) return int(slot);
        bool stale = k == 0u || ((frame - (values[slot] >> 16)) & 0xffffu) > 60u;
        if (stale && atomic_compare_exchange_weak_explicit(&keys[slot], &k, key, memory_order_relaxed, memory_order_relaxed)) {
            values[slot] = 0u;  // frame 0: never seen
            return int(slot);
        }
    }
    return -1;
}

// ---- Occlusion culling ----
// Last frame's depth prepass (terrain and trees) as a pyramid of farthest depths
// (FloraRenderer.buildHzb), with the camera it was drawn from. A plant whose nearest point
// lies behind everything in its screen rectangle there is hidden. Plants outside that view
// count as visible, and the pyramid isn't used once the camera has moved or turned much.
struct FloraHzb {
    float4x4 viewProjection;    // camera-relative world → clip, as drawn
    float4 eye;                 // camera position (xyz), 1 if the pyramid may be used (w)
    float4 size;                // level 0 width, height (texels), levels, projection y scale
};

kernel void floraHzbFirst(depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(out.get_width(), out.get_height()))) return;
    uint2 size = uint2(depth.get_width(), depth.get_height());
    uint2 p = gid * 2u;
    float d = 0.0;
    for (uint k = 0u; k < 4u; k++) d = max(d, depth.read(min(p + uint2(k & 1u, k >> 1u), size - 1u)));
    out.write(float4(d), gid);
}

kernel void floraHzbDown(texture2d<float, access::read> in [[texture(0)]],
                         texture2d<float, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(out.get_width(), out.get_height()))) return;
    uint2 size = uint2(in.get_width(), in.get_height());
    uint2 p = gid * 2u;
    float d = 0.0;
    for (uint k = 0u; k < 4u; k++) d = max(d, in.read(min(p + uint2(k & 1u, k >> 1u), size - 1u)).r);
    // An odd-sized level's last row and column cover one more texel of the level above.
    if ((size.x & 1u) != 0u && gid.x == out.get_width() - 1u) {
        for (uint y = 0u; y < 2u; y++) d = max(d, in.read(min(uint2(size.x - 1u, p.y + y), size - 1u)).r);
    }
    if ((size.y & 1u) != 0u && gid.y == out.get_height() - 1u) {
        for (uint x = 0u; x < 2u; x++) d = max(d, in.read(min(uint2(p.x + x, size.y - 1u), size - 1u)).r);
    }
    out.write(float4(d), gid);
}

bool floraOccluded(constant FloraHzb &h, texture2d<float> hzb, float3 centre, float radius) {
    if (h.eye.w == 0.0) return false;
    float3 rel = centre - h.eye.xyz;
    float4 clip = h.viewProjection * float4(rel, 1.0);
    if (clip.w < radius * 2.0 + 1.0) return false;  // around the camera
    float2 ndc = clip.xy / clip.w;
    float r = radius * h.size.w / clip.w;  // in NDC (y scale; x is narrower)
    if (any(abs(ndc) + r > 1.0)) return false;  // not wholly in last frame's view
    float2 lo = float2(ndc.x - r, -ndc.y - r) * 0.5 + 0.5, hi = float2(ndc.x + r, -ndc.y + r) * 0.5 + 0.5;
    float extent = max(hi.x - lo.x, hi.y - lo.y) * max(h.size.x, h.size.y);
    int level = clamp(int(ceil(log2(max(extent, 1.0)))), 0, int(h.size.z) - 1);
    uint2 n = uint2(max(uint(h.size.x) >> uint(level), 1u), max(uint(h.size.y) >> uint(level), 1u));
    uint2 a = min(uint2(lo * float2(n)), n - 1u), b = min(uint2(hi * float2(n)), n - 1u);
    float farthest = max(max(hzb.read(a, level).r, hzb.read(uint2(b.x, a.y), level).r),
                         max(hzb.read(uint2(a.x, b.y), level).r, hzb.read(b, level).r));
    float3 nearPoint = rel - normalize(rel) * radius;
    float4 c = h.viewProjection * float4(nearPoint, 1.0);
    return c.z / c.w > farthest;
}

// One candidate per hash-grid cell around the camera, culled to the view and sorted into
// its (plant, variant, LOD) bucket.
kernel void floraScatter(constant Uniforms &u [[buffer(0)]],
                         constant FloraParams &f [[buffer(1)]],
                         constant TerrainParams &t [[buffer(2)]],
                         constant DetailParams &d [[buffer(3)]],
                         const device int *pages [[buffer(4)]],
                         device atomic_uint *counts [[buffer(5)]],
                         device FloraInstance *instances [[buffer(6)]],
                         constant uint &layer [[buffer(7)]],
                         device atomic_uint *lodKeys [[buffer(10)]],
                         device uint *lodValues [[buffer(11)]],
                         constant FloraHzb &hzbInfo [[buffer(12)]],
                         texture2d<float> hzb [[texture(5)]],
                         texture2d_array<ushort, access::read> tiles [[texture(0)]],
                         texture2d_array<float, access::read> detail [[texture(1)]],
                         texture2d<float> biome0 [[texture(2)]],
                         texture2d<float> biome1 [[texture(3)]],
                         texture2d<float> biome2 [[texture(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    bool canopy = layer == 0;
    int side = int(canopy ? f.counts.x : f.counts.y);
    if (int(gid.x) >= side || int(gid.y) >= side) return;
    float range = canopy ? f.cells.z : f.cells.w;
    int2 cell = int2(canopy ? f.origins.xy : f.origins.zw) + int2(gid);
    FloraInstance inst;
    int plant, variant;
    // Looked at a little past the range, so plants leaving it can fade out.
    float reach = range * (1.0 + 2.0 * kLodHysteresis);
    if (!floraCandidate(u, f, t, d, pages, tiles, detail, biome0, biome1, biome2, canopy, cell, reach,
                        inst, plant, variant)) return;
    float height = inst.height;

    // Frustum test on a sphere round the plant. Plants out of view keep their state
    // untouched: when they come back it is stale, and they take their LOD at once.
    float3 centre = float3(inst.position) + float3(0.0, 0.0, 0.55 * height) - u.cameraPosition;
    float radius = 0.65 * height + 1.0;
    for (int i = 0; i < 6; i++) {
        if (dot(f.planes[i].xyz, centre) + f.planes[i].w < -radius) return;
    }
    // Hidden behind nearer trees or terrain last frame: like out of view.
    if (floraOccluded(hzbInfo, hzb, centre + u.cameraPosition, radius)) return;
    // LOD boundaries by distance, pushed out for big plants (screen-size bias).
    float dist = length(centre);
    float bias = sqrt(clamp(height / 12.0, 0.4, 2.0));
    float edges[3] = { f.lods.x * bias, min(f.lods2.x * bias, range), range };

    uint frame = uint(f.lods2.z) & 0xffffu;
    uint key = floraLodKey(cell, canopy);
    int slot = floraLodSlot(lodKeys, lodValues, key, frame);
    uint state = slot >= 0 ? lodValues[slot] : 0u;
    bool seen = slot >= 0 && ((frame - (state >> 16)) & 0xffffu) <= 2u && (state >> 16) != 0u;
    uint current = state & 3u, from = (state >> 2) & 3u, progress = (state >> 4) & 0xfffu;
    uint target;
    if (seen) {
        // Step from the current LOD only past the boundary by the hysteresis margin.
        target = current;
        while (target < 3u && dist > edges[target] * (1.0 + kLodHysteresis)) target++;
        while (target > 0u && dist < edges[target - 1u] * (1.0 - kLodHysteresis)) target--;
    } else {
        target = dist < edges[0] ? 0u : (dist < edges[1] ? 1u : (dist < edges[2] ? 2u : 3u));
        current = target;
        from = target;
        progress = kLodProgressMax;
    }
    if (target != current) {
        // Mid-fade, a plant turning back fades from where it is.
        from = current;
        current = target;
        progress = 0u;
    } else {
        progress = min(progress + uint(f.lods2.y), kLodProgressMax);
    }
    if (slot >= 0) lodValues[slot] = current | (from << 2) | (progress << 4) | (max(frame, 1u) << 16);

    int bucket0 = (plant * kFloraVariants + variant) * kFloraLods;
    uint capacity = uint(f.counts.z);
    float p = float(progress) / float(kLodProgressMax);
    bool fading = progress < kLodProgressMax && from != current;
    // The fade's complementary dithers: the incoming LOD where noise < p, the outgoing where
    // noise ≥ p, so every pixel shows exactly one of the two.
    if (current < 3u) {
        inst.fade = fading ? max(p, 1e-3) : 0.0;
        floraAppend(counts, instances, capacity, bucket0 + int(current), inst);
    }
    if (fading && from < 3u) {
        inst.fade = -max(p, 1e-3);
        floraAppend(counts, instances, capacity, bucket0 + int(from), inst);
    }
}

// One thread per bucket: this frame's instance count into its indirect draw arguments.
struct FloraMesh {
    uint indexCount;
    uint indexStart;
    int baseVertex;
    uint pad;
};

// MTLDrawIndexedPrimitivesIndirectArguments.
struct FloraDrawArguments {
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int baseVertex;
    uint baseInstance;
};

kernel void floraArguments(constant FloraParams &f [[buffer(1)]],
                           const device uint *counts [[buffer(5)]],
                           constant FloraMesh *meshes [[buffer(8)]],
                           device FloraDrawArguments *args [[buffer(9)]],
                           uint bucket [[thread_position_in_grid]]) {
    if (bucket > uint(kFloraPlants * kFloraVariants * kFloraLods)) return;  // plants, then the scale figure
    uint capacity = uint(f.counts.z);
    FloraMesh m = meshes[bucket];
    args[bucket].indexCount = m.indexCount;
    args[bucket].instanceCount = min(counts[bucket], capacity);
    args[bucket].indexStart = m.indexStart;
    args[bucket].baseVertex = m.baseVertex;
    args[bucket].baseInstance = bucket * capacity;
}

// ---- Tree shadow casters ----
// Every plant near enough to shade what the camera sees (no view culling: trees behind the
// camera and off to the sides cast into the view) goes into each cascade whose map it
// overlaps, in a (cascade, plant, variant, LOD) bucket (FloraRenderer.renderShadows). The
// LOD goes by the tree's distance from the camera, never finer than the cascade's: LOD0
// within kCasterLod0 in the nearest cascade, LOD1 within kCasterLod1 short of the far
// cascade, impostors facing the sun beyond. A far tree casting into a near cascade (at low
// sun, long shadows reach the camera from 100 m away) stands well above what it shades,
// so its penumbra is wide and its mesh's detail would be lost anyway. The casters' cost is
// in their vertices, not their fill: on Oʻahu's ridge at 07:00, LOD0 to 35 m and LOD1 to
// 150 m took 2.4 and 1.6 ms for the near two maps, 20 and 60 m about half that. Same
// placement as floraScatter (the ground at its drawn LOD), so each tree meets its shadow.

// Where the caster scatter looks, laid out as FloraRenderer.renderShadows writes it.
struct FloraCasterParams {
    float4 origins;         // first canopy cell (xy), first shrub cell (zw)
    float4 extent;          // canopy cells per side, shrub cells per side, canopy reach, shrub reach (m)
    float4 redraw;          // per cascade: 1 if its map is drawn this frame
};

constant int kFloraPlantVariants = kFloraPlants * kFloraVariants;
constant float kCasterLod0 = 20.0, kCasterLod1 = 60.0;

kernel void floraCasters(constant Uniforms &u [[buffer(0)]],
                         constant FloraParams &f [[buffer(1)]],
                         constant TerrainParams &t [[buffer(2)]],
                         constant DetailParams &d [[buffer(3)]],
                         const device int *pages [[buffer(4)]],
                         device atomic_uint *counts [[buffer(5)]],
                         device FloraInstance *instances [[buffer(6)]],
                         constant uint &layer [[buffer(7)]],
                         constant FloraCasterParams &s [[buffer(8)]],
                         constant TreeShadows &shadows [[buffer(9)]],
                         texture2d_array<ushort, access::read> tiles [[texture(0)]],
                         texture2d_array<float, access::read> detail [[texture(1)]],
                         texture2d<float> biome0 [[texture(2)]],
                         texture2d<float> biome1 [[texture(3)]],
                         texture2d<float> biome2 [[texture(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    bool canopy = layer == 0;
    int side = int(canopy ? s.extent.x : s.extent.y);
    if (int(gid.x) >= side || int(gid.y) >= side) return;
    int2 cell = int2(canopy ? s.origins.xy : s.origins.zw) + int2(gid);
    FloraInstance inst;
    int plant, variant;
    if (!floraCandidate(u, f, t, d, pages, tiles, detail, biome0, biome1, biome2, canopy, cell,
                        canopy ? s.extent.z : s.extent.w, inst, plant, variant)) return;
    float3 centre = float3(inst.position) + float3(0.0, 0.0, 0.55 * inst.height) - u.cameraPosition;
    float radius = 0.65 * inst.height + 1.0;
    uint capacity = uint(f.counts.z);
    float dist = length(centre.xy);
    int nearest = dist < kCasterLod0 ? 0 : (dist < kCasterLod1 ? 1 : 2);
    int bucket0 = (plant * kFloraVariants + variant) * kFloraLods;
    for (int c = 0; c < kTreeCascades; c++) {
        if (s.redraw[c] == 0.0) continue;
        TreeCascade k = shadows.cascades[c];
        float across = radius / k.info.w;
        float x = dot(k.x.xyz, centre) + k.x.w, y = dot(k.y.xyz, centre) + k.y.w;
        float z = dot(k.z.xyz, centre) + k.z.w;
        float deep = radius / k.info.y;
        if (abs(x) > 1.0 + across || abs(y) > 1.0 + across || z < -deep || z > 1.0 + deep) continue;
        int lod = c == kTreeCascades - 1 ? 2 : max(nearest, c);
        floraAppend(counts, instances, capacity, c * kFloraPlantVariants * kFloraLods + bucket0 + lod, inst);
    }
}

// One thread per caster bucket: its count and mesh into the indirect arguments.
kernel void floraCasterArguments(constant FloraParams &f [[buffer(1)]],
                                 const device uint *counts [[buffer(5)]],
                                 constant FloraMesh *meshes [[buffer(8)]],
                                 device FloraDrawArguments *args [[buffer(9)]],
                                 uint bucket [[thread_position_in_grid]]) {
    uint perCascade = uint(kFloraPlantVariants * kFloraLods);
    if (bucket >= uint(kTreeCascades) * perCascade) return;
    uint capacity = uint(f.counts.z);
    FloraMesh m = meshes[bucket % perCascade];
    args[bucket].indexCount = m.indexCount;
    args[bucket].instanceCount = min(counts[bucket], capacity);
    args[bucket].indexStart = m.indexStart;
    args[bucket].baseVertex = m.baseVertex;
    args[bucket].baseInstance = bucket * capacity;
}

struct FloraOut {
    // Invariant: the prepass and the shading pass must produce identical depths.
    float4 position [[position, invariant]];
    float3 world;
    float3 normal;
    float2 uv;
    float ao;
    float tint;
    float fade [[flat]];
    int material [[flat]];
};

// The same vertex code feeds the prepass and the shading pass. Wind: the whole plant
// sways along the trade wind, more at the tips; leaves flutter on their own.
vertex FloraOut floraVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                            constant Uniforms &u [[buffer(1)]],
                            const device FloraVertex *vertices [[buffer(2)]],
                            const device FloraInstance *instances [[buffer(3)]]) {
    FloraVertex v = vertices[vid];
    FloraInstance inst = instances[iid];
    float3 p = float3(v.position) * inst.scale;
    float2 r = inst.rotation;
    p.xy = float2(r.x * p.x - r.y * p.y, r.y * p.x + r.x * p.y);
    float3 n = float3(v.normal);
    n.xy = float2(r.x * n.x - r.y * n.y, r.y * n.x + r.x * n.y);
    p.xy += inst.lean * max(p.z, 0.0);

    float clock = u.cloudLayer.z;
    float2 wind = u.cloudWind.xy;
    float speed = length(wind);
    float2 along = speed > 1e-3 ? wind / speed : float2(1.0, 0.0);
    float gust = 0.6 + 0.4 * sin(clock * 0.37 + inst.phase + dot(float3(inst.position).xy, float2(0.013, 0.017)));
    float sway = v.sway * inst.height * 0.006 * (speed / 7.5) * (gust + 0.35 * sin(clock * 1.7 + inst.phase));
    p.xy += along * sway;
    p += n * (v.flutter * 0.025 * sin(clock * 7.0 + inst.phase * 3.0 + dot(p, float3(3.1, 2.3, 1.7))));

    FloraOut o;
    float3 world = float3(inst.position) + p;
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.normal = n;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.ao = v.ao;
    o.tint = inst.tint;
    o.fade = inst.fade;
    o.material = int(v.material);
    return o;
}

// Alpha test and LOD cross-fade, identical in both passes (same texture, same mip: the
// derivatives come from the same invariant positions). False: discard.
bool floraVisible(FloraOut in, texture2d_array<float> cards) {
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    if (in.fade > 0.0 && noise >= in.fade) return false;
    if (in.fade < 0.0 && noise < -in.fade) return false;
    int2 layer = kFloraLayer[in.material];
    if (layer.y == 0) return true;
    constexpr sampler s(filter::linear, mip_filter::linear, address::repeat);
    return cards.sample(s, in.uv, uint(layer.x)).a >= \(FloraTextures.alphaCutoff);
}

// Prepass with ray-traced shadows: the surface for sunShadow, marked as a tree pixel.
fragment uint4 floraSurface(FloraOut in [[stage_in]], texture2d_array<float> cards [[texture(6)]]) {
    if (!floraVisible(in, cards)) discard_fragment();
    float3 n = normalize(in.normal);
    float3 p = in.world;
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z), (pack_float_to_snorm2x16(octEncode(n)) & ~kImpostorSurface) | kTreeSurface);
}

// Prepass without ray tracing: depth only.
fragment void floraDepth(FloraOut in [[stage_in]], texture2d_array<float> cards [[texture(6)]]) {
    if (!floraVisible(in, cards)) discard_fragment();
}

// Shading pass: depth `equal` against the prepass, with the test before the shader runs.
// Only the fragment that wrote a pixel's depth matches it (positions are invariant), so
// texels the prepass discarded (alpha test, cross-fade) fail here without testing again,
// and every visible tree pixel is shaded exactly once.
[[early_fragment_tests]]
fragment float4 floraFragment(FloraOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                              texture2d<float> skyView [[texture(0)]],
                              texture2d<float> clouds [[texture(2)]],
                              texture2d<float> cloudDistance [[texture(3)]],
                              texture2d<float> cloudShadows [[texture(4)]],
                              texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                              texture2d_array<float> cards [[texture(6)]],
                              texture2d_array<float> bark [[texture(7)]],
                              texture2d<float> skyVisibility [[texture(8)]],
                              texture2d_array<float, access::read> horizons [[texture(9)]]) {
    int2 layer = kFloraLayer[in.material];
    float4 tint = kFloraTint[in.material];
    bool card = layer.y != 0;
    constexpr sampler s(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(4));
    float3 albedo = (card ? cards.sample(s, in.uv, uint(layer.x)).rgb : bark.sample(s, in.uv, uint(layer.x)).rgb)
        * tint.rgb * in.tint;

    float3 n = normalize(in.normal);
    float3 l = normalize(u.sunDirection);
    float3 toEye = u.cameraPosition - in.world;
    // Cards: wrapped diffuse, so a crown's shadowed side isn't flat black; bark: Lambert.
    float nDotL = card ? saturate((dot(n, l) + 0.6) / 1.6) : saturate(dot(n, l));
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
    float3 color = litSurface(u, albedo, n, nDotL, sunVisibility, sky, in.ao, cloudSun);
    if (card) {
        // Sun through the leaves when looking towards it.
        float back = saturate(dot(normalize(-toEye), l));
        color += albedo * u.sunRadiance * (tint.a * sunVisibility * back * back * in.ao);
    }
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}

// ---- Impostors (LOD2) ----
// Each plant variant drawn once from the side at start-up (floraBake*) into an albedo
// and a normal texture array, then drawn far away as a camera-facing quad.

// Per plant variant: half-width, lowest and highest z of the LOD0 mesh (reference scale);
// per stand patch: half-width, crown cover, height.
struct FloraImpostor {
    float4 extent;
};

constant int kFloraImpostorViews = \(FloraRenderer.impostorViews);

// The two baked views (FloraRenderer.bakeImpostors) nearest the way `look` (world xy, from
// the plant outward) points in a plant turned by `rotation` (cos, sin of its yaw), the
// blend between them, and each view's right-hand axis in the world.
struct FloraImpostorViews {
    int first, second;
    float blend;
    float4 rights;
};

FloraImpostorViews floraImpostorViews(float2 look, float2 rotation, int variant) {
    // Into the plant's frame. View k looks from azimuth φk − π/2, φk = k · 2π / views.
    float2 local = float2(rotation.x * look.x + rotation.y * look.y, -rotation.y * look.x + rotation.x * look.y);
    float steps = (atan2(local.y, local.x) + M_PI_2_F) / (2.0 * M_PI_F) * float(kFloraImpostorViews);
    steps -= floor(steps / float(kFloraImpostorViews)) * float(kFloraImpostorViews);
    int k0 = int(floor(steps)) % kFloraImpostorViews, k1 = (k0 + 1) % kFloraImpostorViews;
    FloraImpostorViews v;
    v.first = variant * kFloraImpostorViews + k0;
    v.second = variant * kFloraImpostorViews + k1;
    v.blend = fract(steps);
    // View k's right is R(φk)(1, 0) in the plant's frame; then turned by its yaw.
    float a0 = float(k0) * 2.0 * M_PI_F / float(kFloraImpostorViews), a1 = float(k1) * 2.0 * M_PI_F / float(kFloraImpostorViews);
    float2 r0 = float2(cos(a0), sin(a0)), r1 = float2(cos(a1), sin(a1));
    v.rights = float4(rotation.x * r0.x - rotation.y * r0.y, rotation.y * r0.x + rotation.x * r0.y,
                      rotation.x * r1.x - rotation.y * r1.y, rotation.y * r1.x + rotation.x * r1.y);
    return v;
}

struct FloraBakeOut {
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float ao;
    int material [[flat]];
};

// Orthographic side view along +y of the plant turned by −extent.w: x right, z up,
// nearer (smaller y) in front.
vertex FloraBakeOut floraBakeVertex(uint vid [[vertex_id]],
                                    const device FloraVertex *vertices [[buffer(2)]],
                                    constant float4 &extent [[buffer(4)]]) {
    FloraVertex v = vertices[vid];
    float c = cos(extent.w), s = sin(extent.w);
    float3 p = float3(v.position);
    p.xy = float2(c * p.x + s * p.y, -s * p.x + c * p.y);
    float3 n = float3(v.normal);
    n.xy = float2(c * n.x + s * n.y, -s * n.x + c * n.y);
    float r = extent.x, z0 = extent.y, z1 = extent.z;
    FloraBakeOut o;
    o.position = float4(p.x / r, (p.z - z0) / (z1 - z0) * 2.0 - 1.0, (p.y + r) / (2.0 * r), 1.0);
    o.normal = n;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.ao = v.ao;
    o.material = int(v.material);
    return o;
}

struct FloraBakeTargets {
    float4 albedo [[color(0)]];  // rgb: albedo (sRGB target), a: coverage
    float4 normal [[color(1)]];  // rg: view-space normal (octahedral, 0…1), b: AO, a: 1 on leaves
};

fragment FloraBakeTargets floraBakeFragment(FloraBakeOut in [[stage_in]],
                                            texture2d_array<float> cards [[texture(6)]],
                                            texture2d_array<float> bark [[texture(7)]]) {
    int2 layer = kFloraLayer[in.material];
    float4 tint = kFloraTint[in.material];
    constexpr sampler s(filter::linear, mip_filter::linear, address::repeat);
    float4 c = layer.y != 0 ? cards.sample(s, in.uv, uint(layer.x)) : float4(bark.sample(s, in.uv, uint(layer.x)).rgb, 1.0);
    if (c.a < \(FloraTextures.alphaCutoff)) discard_fragment();
    float3 n = normalize(in.normal);
    float3 view = float3(n.x, n.z, -n.y);  // right, up, towards the viewer
    FloraBakeTargets o;
    o.albedo = float4(c.rgb * tint.rgb, 1.0);
    o.normal = float4(octEncode(view) * 0.5 + 0.5, in.ao, layer.y != 0 ? 1.0 : 0.0);
    return o;
}

struct FloraImpostorOut {
    float4 position [[position, invariant]];
    float3 world;
    float2 uv;
    float2 right;           // the quad's horizontal axis (world xy)
    float tint;
    float fade [[flat]];
    int layer [[flat]];     // impostor texture layers of the two nearest views
    int layer2 [[flat]];
    float blend [[flat]];   // 0: all layer, 1: all layer2
    float4 rights [[flat]]; // each view's right-hand axis (world xy)
};

// A quad turned about z to face the camera, spanning the variant's side view.
vertex FloraImpostorOut floraImpostorVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                            constant Uniforms &u [[buffer(1)]],
                                            const device FloraVertex *vertices [[buffer(2)]],
                                            const device FloraInstance *instances [[buffer(3)]],
                                            constant FloraImpostor *impostors [[buffer(4)]],
                                            constant uint &capacity [[buffer(5)]]) {
    FloraVertex v = vertices[vid];
    FloraInstance inst = instances[iid];
    int variant = int(iid / capacity) / kFloraLods;  // baseInstance = bucket × capacity
    float4 e = impostors[variant].extent;
    float3 base = float3(inst.position);
    float2 toCamera = u.cameraPosition.xy - base.xy;
    float len = length(toCamera);
    float2 back = len > 1e-3 ? -toCamera / len : float2(0.0, 1.0);  // the way the camera looks
    float2 right = float2(back.y, -back.x);
    FloraImpostorViews views = floraImpostorViews(-back, inst.rotation, variant);
    float side = (v.uv.x - 0.5) * 2.0 * e.x * inst.scale;
    float z = mix(e.y, e.z, v.uv.y) * inst.scale;
    FloraImpostorOut o;
    float3 world = base + float3(right * side, z);
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.right = right;
    o.tint = inst.tint;
    o.fade = inst.fade;
    o.layer = views.first;
    o.layer2 = views.second;
    o.blend = views.blend;
    o.rights = views.rights;
    return o;
}

bool floraImpostorVisible(FloraImpostorOut in, texture2d_array<float> albedo) {
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    if (in.fade > 0.0 && noise >= in.fade) return false;
    if (in.fade < 0.0 && noise < -in.fade) return false;
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    float a = albedo.sample(s, in.uv, uint(in.layer)).a;
    if (in.blend > 0.0) a = mix(a, albedo.sample(s, in.uv, uint(in.layer2)).a, in.blend);
    return a >= \(FloraTextures.alphaCutoff);
}

fragment uint4 floraImpostorSurface(FloraImpostorOut in [[stage_in]], texture2d_array<float> albedo [[texture(10)]]) {
    if (!floraImpostorVisible(in, albedo)) discard_fragment();
    float3 p = in.world;
    float3 n = float3(in.right.y, -in.right.x, 0.0);  // towards the camera
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z),
                 pack_float_to_snorm2x16(octEncode(n)) | kTreeSurface | kImpostorSurface);
}

// A caster for the far shadow cascade: the variant's side view on a quad facing the sun,
// pivoting about the middle of the crown so the crown's shadow lands where the mesh's would.
// Its shadow is the silhouette drawn out by 1 / sin(elevation) along the sun, as a round
// crown's is.
vertex FloraImpostorOut floraImpostorCasterVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                                  constant Uniforms &u [[buffer(1)]],
                                                  const device FloraVertex *vertices [[buffer(2)]],
                                                  const device FloraInstance *instances [[buffer(3)]],
                                                  constant FloraImpostor *impostors [[buffer(4)]],
                                                  constant uint &capacity [[buffer(5)]]) {
    FloraVertex v = vertices[vid];
    FloraInstance inst = instances[iid];
    int variant = (int(iid / capacity) / kFloraLods) % kFloraPlantVariants;  // baseInstance = bucket × capacity
    float4 e = impostors[variant].extent;
    float3 l = normalize(u.sunDirection);
    // The baked view nearest the sun's azimuth (high sun: any will do).
    float2 sunward = length(l.xy) > 1e-3 ? normalize(l.xy) : float2(0.0, -1.0);
    FloraImpostorViews views = floraImpostorViews(sunward, inst.rotation, variant);
    int layer = views.blend < 0.5 ? views.first : views.second;
    float3 right = normalize(cross(l, abs(l.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
    float3 up = cross(right, l);
    float side = (v.uv.x - 0.5) * 2.0 * e.x * inst.scale;
    float mid = 0.5 * (e.y + e.z) * inst.scale;
    float z = mix(e.y, e.z, v.uv.y) * inst.scale;
    FloraImpostorOut o;
    float3 world = float3(inst.position) + float3(0.0, 0.0, mid) + right * side + up * (z - mid);
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.right = right.xy;
    o.tint = inst.tint;
    o.fade = 0.0;
    o.layer = layer;
    o.layer2 = layer;
    o.blend = 0.0;
    o.rights = 0.0;
    return o;
}

fragment void floraImpostorDepth(FloraImpostorOut in [[stage_in]], texture2d_array<float> albedo [[texture(10)]]) {
    if (!floraImpostorVisible(in, albedo)) discard_fragment();
}

// Lit like the meshes, with the baked side view's normals turned to face the camera.
[[early_fragment_tests]]
fragment float4 floraImpostorFragment(FloraImpostorOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                      texture2d<float> skyView [[texture(0)]],
                                      texture2d<float> clouds [[texture(2)]],
                                      texture2d<float> cloudDistance [[texture(3)]],
                                      texture2d<float> cloudShadows [[texture(4)]],
                                      texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                                      texture2d<float> skyVisibility [[texture(8)]],
                                      texture2d_array<float, access::read> horizons [[texture(9)]],
                                      texture2d_array<float> albedoMap [[texture(10)]],
                                      texture2d_array<float> normalMap [[texture(11)]]) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    // Each view's colour and normal, its normal out of that view's frame (right, up,
    // towards its viewer), then blended.
    float3 albedo = albedoMap.sample(s, in.uv, uint(in.layer)).rgb;
    float4 packed = normalMap.sample(s, in.uv, uint(in.layer));
    float3 nv = octDecode(packed.xy * 2.0 - 1.0);
    float3 n = float3(in.rights.xy, 0.0) * nv.x + float3(0.0, 0.0, nv.y) + float3(in.rights.y, -in.rights.x, 0.0) * nv.z;
    if (in.blend > 0.0) {
        albedo = mix(albedo, albedoMap.sample(s, in.uv, uint(in.layer2)).rgb, in.blend);
        float4 packed2 = normalMap.sample(s, in.uv, uint(in.layer2));
        float3 nv2 = octDecode(packed2.xy * 2.0 - 1.0);
        float3 n2 = float3(in.rights.zw, 0.0) * nv2.x + float3(0.0, 0.0, nv2.y) + float3(in.rights.w, -in.rights.z, 0.0) * nv2.z;
        n = mix(n, n2, in.blend);
        packed = mix(packed, packed2, in.blend);
    }
    albedo *= in.tint;
    n = normalize(n);
    bool leaf = packed.a > 0.5;
    float3 l = normalize(u.sunDirection);
    float nDotL = leaf ? saturate((dot(n, l) + 0.6) / 1.6) : saturate(dot(n, l));
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
    float3 color = litSurface(u, albedo, n, nDotL, sunVisibility, sky, packed.b, cloudSun);
    if (leaf) {
        float back = saturate(dot(normalize(in.world - u.cameraPosition), l));
        color += albedo * u.sunRadiance * (0.25 * sunVisibility * back * back * packed.b);
    }
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}

// ---- Stand patches (docs/flora.md §11) ----
// Beyond the single trees, out to a few kilometres, each forested 12 m cell is one
// camera-facing card showing a patch of its community's stand, baked from LOD0 trees in
// the impostors' eight views (FloraRenderer.bakePatches). They give ridgelines and pali
// edges their broken silhouette. Same view selection, LOD fades and occlusion as the trees.

constant float kPatchCell = \(FloraRenderer.patchCell);

// A patch tree for the bake: floraVertex's placement without the wind, the whole patch
// turned by −frame.z about its centre, then the orthographic side view of floraBakeVertex
// over frame.x either side, from 0.15 × frame.y below the ground to frame.y above it.
vertex FloraBakeOut floraPatchBakeVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                         const device FloraVertex *vertices [[buffer(2)]],
                                         const device FloraInstance *instances [[buffer(3)]],
                                         constant float4 &frame [[buffer(4)]]) {
    FloraVertex v = vertices[vid];
    FloraInstance inst = instances[iid];
    float3 p = float3(v.position) * inst.scale;
    float2 r = inst.rotation;
    p.xy = float2(r.x * p.x - r.y * p.y, r.y * p.x + r.x * p.y);
    float3 n = float3(v.normal);
    n.xy = float2(r.x * n.x - r.y * n.y, r.y * n.x + r.x * n.y);
    p += float3(inst.position);
    float c = cos(frame.z), s = sin(frame.z);
    p.xy = float2(c * p.x + s * p.y, -s * p.x + c * p.y);
    n.xy = float2(c * n.x + s * n.y, -s * n.x + c * n.y);
    float hw = frame.x;
    FloraBakeOut o;
    o.position = float4(p.x / hw, (p.z + 0.15 * frame.y) / (1.15 * frame.y) * 2.0 - 1.0, (p.y + hw) / (2.0 * hw), 1.0);
    o.normal = n;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.ao = v.ao;
    o.material = int(v.material);
    return o;
}

// Where the patch scatter looks, laid out as FloraRenderer.scatterPatches writes it.
struct FloraPatchParams {
    float4 grid;            // first cell (xy), cells per side, unused
    float4 range;           // nearest, farthest distance (m), bucket capacity, unused
};

kernel void floraPatches(constant Uniforms &u [[buffer(0)]],
                         constant FloraParams &f [[buffer(1)]],
                         constant TerrainParams &t [[buffer(2)]],
                         constant DetailParams &d [[buffer(3)]],
                         const device int *pages [[buffer(4)]],
                         device atomic_uint *counts [[buffer(5)]],
                         device FloraInstance *instances [[buffer(6)]],
                         constant FloraPatchParams &pp [[buffer(7)]],
                         constant FloraImpostor *patches [[buffer(8)]],
                         device atomic_uint *lodKeys [[buffer(10)]],
                         device uint *lodValues [[buffer(11)]],
                         constant FloraHzb &hzbInfo [[buffer(12)]],
                         texture2d_array<ushort, access::read> tiles [[texture(0)]],
                         texture2d_array<float, access::read> detail [[texture(1)]],
                         texture2d<float> biome0 [[texture(2)]],
                         texture2d<float> biome1 [[texture(3)]],
                         texture2d<float> biome2 [[texture(4)]],
                         texture2d<float> hzb [[texture(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int side = int(pp.grid.z);
    if (int(gid.x) >= side || int(gid.y) >= side) return;
    int2 cell = int2(pp.grid.xy) + int2(gid);
    uint h = floraHash(cell, 7u, uint(f.counts.w));
    float2 xy = (float2(cell) + 0.5 + 0.3 * (float2(floraRandom(h, 0), floraRandom(h, 1)) - 0.5)) * kPatchCell;
    float2 rel = xy - u.cameraPosition.xy;
    float nearest = pp.range.x, farthest = pp.range.y;
    if (dot(rel, rel) > pow(farthest * (1.0 + 2.0 * kLodHysteresis), 2.0)) return;
    if (dot(rel, rel) < pow(nearest * (1.0 - 2.0 * kLodHysteresis), 2.0)) return;
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 uv = xy / f.lods.w;
    int community = floraPickCommunity(biome0.sample(linear, uv, level(0)), biome1.sample(linear, uv, level(0)),
                                       biome2.sample(linear, uv, level(0)), floraRandom(h, 2));
    if (community < 0) return;
    float4 e = patches[community].extent;
    if (e.z <= 0.0) return;  // nothing grows there tall enough
    // Open stands leave cells empty, by the cover of what the patch shows.
    if (floraRandom(h, 3) >= 1.3 * e.y) return;
    float2 gradient;
    float ground = floraGround(t, d, detail, pages, tiles, xy, length(rel), gradient);
    // Walls steeper than 45° are the canopy shading's: a card there stands off the slope.
    if (ground < f.lods.z + 0.5 || length(gradient) > 1.0) return;

    FloraInstance inst;
    float scale = 0.85 + 0.3 * floraRandom(h, 4);
    inst.position = packed_float3(xy.x, xy.y, ground);
    inst.scale = scale;
    float yaw = floraRandom(h, 5) * 6.2831853;
    inst.rotation = float2(cos(yaw), sin(yaw));
    inst.lean = 0.0;
    inst.tint = 0.9 + 0.2 * floraRandom(h, 6);
    inst.phase = 0.0;
    inst.height = e.z * scale;
    float3 centre = float3(xy, ground + 0.5 * inst.height) - u.cameraPosition;
    float radius = length(float2(e.x, 0.5 * e.z)) * scale;
    for (int i = 0; i < 6; i++) {
        if (dot(f.planes[i].xyz, centre) + f.planes[i].w < -radius) return;
    }
    if (floraOccluded(hzbInfo, hzb, centre + u.cameraPosition, radius)) return;

    // In range or not, faded in time as the trees' LODs are (state 0 drawn, 3 not).
    float dist = length(centre);
    uint frame = uint(f.lods2.z) & 0xffffu;
    uint key = floraLodKey(cell, true) * 0x9e3779b1u | 1u;
    int slot = floraLodSlot(lodKeys, lodValues, key, frame);
    uint state = slot >= 0 ? lodValues[slot] : 0u;
    bool seen = slot >= 0 && ((frame - (state >> 16)) & 0xffffu) <= 2u && (state >> 16) != 0u;
    uint current = state & 3u, from = (state >> 2) & 3u, progress = (state >> 4) & 0xfffu;
    uint target;
    if (seen) {
        float slack = current == 0u ? kLodHysteresis : -kLodHysteresis;
        target = dist > nearest * (1.0 - slack) && dist < farthest * (1.0 + slack) ? 0u : 3u;
    } else {
        target = dist > nearest && dist < farthest ? 0u : 3u;
        current = target;
        from = target;
        progress = kLodProgressMax;
    }
    if (target != current) {
        from = current;
        current = target;
        progress = 0u;
    } else {
        progress = min(progress + uint(f.lods2.y), kLodProgressMax);
    }
    if (slot >= 0) lodValues[slot] = current | (from << 2) | (progress << 4) | (max(frame, 1u) << 16);
    float p = float(progress) / float(kLodProgressMax);
    bool fading = progress < kLodProgressMax && from != current;
    uint capacity = uint(pp.range.z);
    if (current == 0u) {
        inst.fade = fading ? max(p, 1e-3) : 0.0;
        floraAppend(counts, instances, capacity, community, inst);
    } else if (fading && from == 0u) {
        inst.fade = -max(p, 1e-3);
        floraAppend(counts, instances, capacity, community, inst);
    }
}

// One thread per community: its patch count into the indirect arguments (the impostor quad).
kernel void floraPatchArguments(const device uint *counts [[buffer(5)]],
                                constant FloraMesh &quad [[buffer(8)]],
                                device FloraDrawArguments *args [[buffer(9)]],
                                constant FloraPatchParams &pp [[buffer(7)]],
                                uint bucket [[thread_position_in_grid]]) {
    if (bucket >= 11u) return;
    uint capacity = uint(pp.range.z);
    args[bucket].indexCount = quad.indexCount;
    args[bucket].instanceCount = min(counts[bucket], capacity);
    args[bucket].indexStart = quad.indexStart;
    args[bucket].baseVertex = quad.baseVertex;
    args[bucket].baseInstance = bucket * capacity;
}

// A patch's card: as floraImpostorVertex, its layers the community's views.
vertex FloraImpostorOut floraPatchVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                         constant Uniforms &u [[buffer(1)]],
                                         const device FloraVertex *vertices [[buffer(2)]],
                                         const device FloraInstance *instances [[buffer(3)]],
                                         constant FloraImpostor *patches [[buffer(4)]],
                                         constant uint &capacity [[buffer(5)]]) {
    FloraVertex v = vertices[vid];
    FloraInstance inst = instances[iid];
    int community = int(iid / capacity);
    float4 e = patches[community].extent;
    float3 base = float3(inst.position);
    float2 toCamera = u.cameraPosition.xy - base.xy;
    float len = length(toCamera);
    float2 back = len > 1e-3 ? -toCamera / len : float2(0.0, 1.0);
    float2 right = float2(back.y, -back.x);
    FloraImpostorViews views = floraImpostorViews(-back, inst.rotation, community);
    float side = (v.uv.x - 0.5) * 2.0 * e.x * inst.scale;
    // From a little below the ground (as baked), so slopes don't show daylight under it.
    float z = mix(-0.15 * e.z, e.z, v.uv.y) * inst.scale;
    FloraImpostorOut o;
    float3 world = base + float3(right * side, z);
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.right = right;
    o.tint = inst.tint;
    o.fade = inst.fade;
    o.layer = views.first;
    o.layer2 = views.second;
    o.blend = views.blend;
    o.rights = views.rights;
    return o;
}

"""