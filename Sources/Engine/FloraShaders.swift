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
    float4 lods;            // LOD0 distance, cross-fade width (m), sea level, domain size (m)
    float4 lods2;           // LOD1 distance (m), unused
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
// shadow scatter, so shadows stand exactly where the trees are drawn.
bool floraCandidate(constant Uniforms &u, constant FloraParams &f, constant TerrainParams &t,
                    constant DetailParams &d, const device int *pages, texture2d_array<ushort, access::read> tiles,
                    texture2d_array<float, access::read> detail, texture2d<float> biome0, texture2d<float> biome1,
                    texture2d<float> biome2, bool canopy, int2 cell, float range, bool finest,
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
    // On the terrain level drawn at this distance, or (`finest`: shadow proxies, which must
    // not move as the camera does between rebuilds) the finest.
    float ground = floraGround(t, d, detail, pages, tiles, xy, finest ? 0.0 : length(float3(rel, u.cameraPosition.z)), gradient);
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
    if (!floraCandidate(u, f, t, d, pages, tiles, detail, biome0, biome1, biome2, canopy, cell, range, false,
                        inst, plant, variant)) return;
    float height = inst.height;

    // Frustum test on a sphere round the plant.
    float3 centre = float3(inst.position) + float3(0.0, 0.0, 0.55 * height) - u.cameraPosition;
    float radius = 0.65 * height + 1.0;
    for (int i = 0; i < 6; i++) {
        if (dot(f.planes[i].xyz, centre) + f.planes[i].w < -radius) return;
    }
    // LODs by distance, the boundaries pushed out for big plants (screen-size bias).
    float dist = length(centre);
    if (dist > range) return;
    float bias = sqrt(clamp(height / 12.0, 0.4, 2.0));
    float fadeWidth = f.lods.y * bias;
    float lod0 = f.lods.x * bias, lod1 = min(f.lods2.x * bias, range - fadeWidth);

    int bucket0 = (plant * kFloraVariants + variant) * kFloraLods;
    uint capacity = uint(f.counts.z);
    // Each LOD fades out over the last fadeWidth before its far edge while the next fades
    // in with the complementary dither; the impostors fade out at the end of the range.
    int lod = dist < lod0 ? 0 : (dist < lod1 ? 1 : 2);
    float edge = lod == 0 ? lod0 : (lod == 1 ? lod1 : range);
    float inside = saturate((edge - dist) / fadeWidth);  // 1 well inside this LOD
    if (lod == 2) {
        inst.fade = inside >= 1.0 ? 0.0 : inside;
        if (inside > 0.0) floraAppend(counts, instances, capacity, bucket0 + 2, inst);
        return;
    }
    inst.fade = inside >= 1.0 ? 0.0 : -(1.0 - inside);
    floraAppend(counts, instances, capacity, bucket0 + lod, inst);
    if (inside < 1.0) {
        inst.fade = 1.0 - inside;
        floraAppend(counts, instances, capacity, bucket0 + lod + 1, inst);
    }
}

// ---- Tree shadows ----
// Every plant within the shadow range (no view culling: trees behind the camera shade what
// it sees) becomes an instance of its variant's shadow proxy in an instance acceleration
// structure, built on the GPU from these descriptors (FloraRenderer.updateShadows).

// MTLIndirectAccelerationStructureInstanceDescriptor (72 bytes).
struct FloraShadowInstance {
    packed_float3 columns[4];   // MTLPackedFloat4x3: the linear part, then the translation
    uint options;
    uint mask;
    uint intersectionFunctionTableOffset;
    uint userID;
    uint2 accelerationStructure; // MTLResourceID of the variant's proxy
};

// Where the shadow scatter looks, laid out as FloraRenderer.ShadowParams.
struct FloraShadowParams {
    float4 origins;         // first canopy cell (xy), first shrub cell (zw)
    float4 extent;          // canopy cells per side, shrub cells per side, canopy range, shrub range (m)
    float4 capacity;        // instances (x)
};

kernel void floraShadows(constant Uniforms &u [[buffer(0)]],
                         constant FloraParams &f [[buffer(1)]],
                         constant TerrainParams &t [[buffer(2)]],
                         constant DetailParams &d [[buffer(3)]],
                         const device int *pages [[buffer(4)]],
                         device atomic_uint *count [[buffer(5)]],
                         device FloraShadowInstance *out [[buffer(6)]],
                         constant uint &layer [[buffer(7)]],
                         constant FloraShadowParams &s [[buffer(8)]],
                         const device uint2 *proxies [[buffer(9)]],
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
                        canopy ? s.extent.z : s.extent.w, true, inst, plant, variant)) return;
    // Trunk and crown are instances of their own (2k and 2k + 1), so each has a tight box:
    // a ray along the ground meets few crown boxes, and few trunk boxes higher up.
    uint index = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    if (2u * index + 1u >= uint(s.capacity.x)) return;
    // floraVertex's placement without the wind: scale, yaw, then lean with height.
    float k = inst.scale;
    float2 r = inst.rotation;
    FloraShadowInstance o;
    o.columns[0] = packed_float3(k * r.x, k * r.y, 0.0);
    o.columns[1] = packed_float3(-k * r.y, k * r.x, 0.0);
    o.columns[2] = packed_float3(k * inst.lean.x, k * inst.lean.y, k);
    o.columns[3] = inst.position;
    o.options = 5u;  // disable triangle culling | opaque
    o.mask = 1u;  // trunks: kTrunkMask
    o.intersectionFunctionTableOffset = 0u;
    o.userID = 0u;
    uint proxy = uint(plant * kFloraVariants + variant);
    o.accelerationStructure = proxies[2u * proxy];
    out[2u * index] = o;
    o.accelerationStructure = proxies[2u * proxy + 1u];
    o.mask = 2u;  // crowns: kCrownMask
    out[2u * index + 1u] = o;
}

// The instance count the build reads, no more than the descriptors written.
kernel void floraShadowCount(const device uint *scattered [[buffer(5)]],
                             device uint *count [[buffer(10)]],
                             constant FloraShadowParams &s [[buffer(8)]]) {
    count[0] = min(2u * scattered[0], uint(s.capacity.x) & ~1u);
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

// Per plant variant: half-width, lowest and highest z of the LOD0 mesh (reference scale).
struct FloraImpostor {
    float4 extent;
};

struct FloraBakeOut {
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float ao;
    int material [[flat]];
};

// Orthographic side view along +y: x right, z up, nearer (smaller y) in front.
vertex FloraBakeOut floraBakeVertex(uint vid [[vertex_id]],
                                    const device FloraVertex *vertices [[buffer(2)]],
                                    constant float4 &extent [[buffer(4)]]) {
    FloraVertex v = vertices[vid];
    float3 p = float3(v.position);
    float r = extent.x, z0 = extent.y, z1 = extent.z;
    FloraBakeOut o;
    o.position = float4(p.x / r, (p.z - z0) / (z1 - z0) * 2.0 - 1.0, (p.y + r) / (2.0 * r), 1.0);
    o.normal = float3(v.normal);
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
    int layer [[flat]];     // impostor texture layer: plant × variants + variant
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
    int layer = int(iid / capacity) / kFloraLods;  // baseInstance = bucket × capacity
    float4 e = impostors[layer].extent;
    float3 base = float3(inst.position);
    float2 toCamera = u.cameraPosition.xy - base.xy;
    float len = length(toCamera);
    float2 back = len > 1e-3 ? -toCamera / len : float2(0.0, 1.0);  // the way the camera looks
    float2 right = float2(back.y, -back.x);
    float side = (v.uv.x - 0.5) * 2.0 * e.x * inst.scale;
    float z = mix(e.y, e.z, v.uv.y) * inst.scale;
    FloraImpostorOut o;
    float3 world = base + float3(right * side, z);
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    // Variants are mirrored at random for variety.
    o.uv = float2(inst.rotation.y < 0.0 ? 1.0 - v.uv.x : v.uv.x, 1.0 - v.uv.y);
    o.right = inst.rotation.y < 0.0 ? -right : right;
    o.tint = inst.tint;
    o.fade = inst.fade;
    o.layer = layer;
    return o;
}

bool floraImpostorVisible(FloraImpostorOut in, texture2d_array<float> albedo) {
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    if (in.fade > 0.0 && noise >= in.fade) return false;
    if (in.fade < 0.0 && noise < -in.fade) return false;
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    return albedo.sample(s, in.uv, uint(in.layer)).a >= \(FloraTextures.alphaCutoff);
}

fragment uint4 floraImpostorSurface(FloraImpostorOut in [[stage_in]], texture2d_array<float> albedo [[texture(10)]]) {
    if (!floraImpostorVisible(in, albedo)) discard_fragment();
    float3 p = in.world;
    float3 n = float3(-in.right.y, in.right.x, 0.0);
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z),
                 pack_float_to_snorm2x16(octEncode(n)) | kTreeSurface | kImpostorSurface);
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
    float3 albedo = albedoMap.sample(s, in.uv, uint(in.layer)).rgb * in.tint;
    float4 packed = normalMap.sample(s, in.uv, uint(in.layer));
    float3 nv = octDecode(packed.xy * 2.0 - 1.0);
    float3 right = float3(in.right, 0.0);
    float3 towards = float3(-in.right.y, in.right.x, 0.0);
    float3 n = normalize(right * nv.x + float3(0.0, 0.0, 1.0) * nv.y + towards * nv.z);
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
"""