import IslandCore

/// Metal source for the first terrain pass, compiled at runtime so `swift build` needs no
/// Xcode-only Metal toolchain step. Moves to .metal files once the app has an Xcode target.
let terrainShaderSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

""" + Look.metalPrelude + """
// True: sun shadows from rays traced against the terrain mesh. False: from the baked
// horizon map, for GPUs that can't trace rays from a fragment shader.
constant bool kRayTraced [[function_constant(0)]];
// True: the native trees cast shadows through their cascaded shadow maps (FloraRenderer).
constant bool kTreeShadows [[function_constant(1)]];

// The streamed heightfield (TerrainStreamer) and the CDLOD layout (TerrainQuadtree).
struct TerrainParams {
    float cellMetres;       // level-0 sample spacing
    float heightOffset;     // height = offset + scale × sample
    float heightScale;
    float domainSize;       // metres across
    int levels;             // data mip levels
    int tileSize;
    int lodCount;
    int detailLods;         // CDLOD levels finer than data level 0 (TerrainDetail)
    int4 levelSize[2];      // samples per side, per data level
    int4 levelTiles[2];     // tiles per side, per data level
    int4 pageOffset[2];     // first page-table entry, per data level
    float4 morphStart[4];   // per LOD: distance where vertices start morphing to the next LOD
    float4 morphEnd[4];
};

struct TerrainNode {
    float2 origin;          // metres
    int lod;
    int quads;              // quads per side in this draw's grid
};

int at(constant int4 *a, int i) { return a[i >> 2][i & 3]; }
float at(constant float4 *a, int i) { return a[i >> 2][i & 3]; }

// A level-`level` sample, falling back to coarser levels while its tile isn't resident.
float rawHeight(constant TerrainParams &t, const device int *pages,
                texture2d_array<ushort, access::read> tiles, int level, int2 s) {
    for (int m = level; m < t.levels; m++) {
        int size = at(t.levelSize, m);
        int2 c = clamp(s, int2(0), int2(size - 1));
        int2 tile = c / t.tileSize;
        int page = pages[at(t.pageOffset, m) + tile.y * at(t.levelTiles, m) + tile.x];
        if (page >= 0) return float(tiles.read(uint2(c - tile * t.tileSize), uint(page)).r);
        s = c / 2;
    }
    return 0.0;
}

// Height in metres at a world position from data level `level`, bilinear between samples
// (sample k sits at the centre of cell k).
float levelHeight(constant TerrainParams &t, const device int *pages,
                  texture2d_array<ushort, access::read> tiles, int level, float2 xy) {
    float2 s = xy / (t.cellMetres * float(1 << level)) - 0.5;
    float2 s0 = floor(s);
    float2 f = s - s0;
    int2 i = int2(s0);
    float h00 = rawHeight(t, pages, tiles, level, i);
    float h10 = rawHeight(t, pages, tiles, level, i + int2(1, 0));
    float h01 = rawHeight(t, pages, tiles, level, i + int2(0, 1));
    float h11 = rawHeight(t, pages, tiles, level, i + int2(1, 1));
    return t.heightOffset + t.heightScale * mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// The surface a LOD-`lod` vertex sits on: its own data level, blended into the next one
// as it morphs, so a fully morphed vertex matches the coarser neighbouring node exactly.
float terrainHeight(constant TerrainParams &t, const device int *pages,
                    texture2d_array<ushort, access::read> tiles, int lod, float morph, float2 xy) {
    int m = min(lod, t.levels - 1);
    float h = levelHeight(t, pages, tiles, m, xy);
    if (morph > 0.0 && m + 1 < t.levels) h = mix(h, levelHeight(t, pages, tiles, m + 1, xy), morph);
    return h;
}
""" + detailShaderSource + """
struct Uniforms {
    float4x4 viewProjection;
    float4x4 inverseViewProjection; // clip space → camera-relative world directions
    float3 cameraPosition;  // metres, world space
    float seaLevel;
    float3 sunDirection;    // towards the sun
    float maxHeight;
    float3 sunRadiance;     // SunLight.sun
    float exposure;
    float4 skySH[9];        // SunLight.skySH: sky irradiance for any normal (rgb)
    float4 cloudLayer;      // cloud base, inversion (m above sea), clock (s), domain size (m)
    float4 cloudWind;       // trade wind velocity xy (m/s), orographic coverage gain, unused
    float4x4 previousViewProjection; // last frame's, for reprojecting the cloud history
    float4 previousCamera;  // last frame's camera position (xyz)
    float4 frame;           // render target width, height (px), frame number, history valid (0/1)
};

// Sky radiance along a direction, from the sky-view LUT (Atmosphere.skyView).
float3 skyRadiance(texture2d<float> skyView, float3 dir) {
    constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
    float azimuth = atan2(dir.y, dir.x);
    float elevation = asin(clamp(dir.z, -1.0, 1.0));
    float w = sign(elevation) * sqrt(abs(elevation) / M_PI_2_F);
    return skyView.sample(s, float2(azimuth / (2.0 * M_PI_F), 0.5 + 0.5 * w)).rgb;
}

// Haze and reflections look along a direction lifted to the horizon at least: the sky
// below it is never seen through air in the renderer.
float3 skyAbove(texture2d<float> skyView, float3 dir) {
    return skyRadiance(skyView, normalize(float3(dir.xy, max(dir.z, 0.002 * length(dir.xy)))));
}

float3 skyIrradiance(constant Uniforms &u, float3 n) {
    float3 e = u.skySH[0].rgb * 0.282095
        + u.skySH[1].rgb * (0.488603 * n.y) + u.skySH[2].rgb * (0.488603 * n.z) + u.skySH[3].rgb * (0.488603 * n.x)
        + u.skySH[4].rgb * (1.092548 * n.x * n.y) + u.skySH[5].rgb * (1.092548 * n.y * n.z)
        + u.skySH[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
        + u.skySH[7].rgb * (1.092548 * n.x * n.z) + u.skySH[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, 0.0);
}
""" + cloudShaderSource + """
struct VOut {
    // Invariant: the depth prepass and the shading pass must produce identical depths.
    float4 position [[position, invariant]];
    float3 world;
    float3 normal;
    float2 light;
    // Horizon elevation (radians) in 16 azimuth sectors, for the baked shadow path.
    float4 horizon0;
    float4 horizon1;
    float4 horizon2;
    float4 horizon3;
};

// One vertex of a CDLOD node's grid. Vertices near the far edge of their LOD ring slide
// onto the next LOD's grid (odd rows and columns collapse onto even ones), and the height
// blends to the next data level with them.
vertex VOut terrainVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                          constant Uniforms &u [[buffer(1)]],
                          constant TerrainParams &t [[buffer(3)]],
                          const device TerrainNode *nodes [[buffer(4)]],
                          const device int *pages [[buffer(5)]],
                          texture2d_array<ushort, access::read> tiles [[texture(0)]],
                          texture2d<float> skyVisibility [[texture(1)]],
                          texture2d_array<float, access::read> horizons [[texture(2)]],
                          constant DetailParams &d [[buffer(6)]],
                          texture2d_array<float, access::read> detail [[texture(3)]]) {
    TerrainNode node = nodes[iid];
    uint row = uint(node.quads + 1);
    float2 g = float2(float(vid % row), float(vid / row));
    float spacing = ldexp(t.cellMetres, node.lod - t.detailLods);
    float2 xy = node.origin + g * spacing;
    float h0 = surfaceHeight(t, d, detail, pages, tiles, node.lod, 0.0, xy);
    float ms = at(t.morphStart, node.lod), me = at(t.morphEnd, node.lod);
    float morph = saturate((distance(float3(xy, h0), u.cameraPosition) - ms) / max(me - ms, 1.0));
    xy -= fmod(g, 2.0) * spacing * morph;
    float h = surfaceHeight(t, d, detail, pages, tiles, node.lod, morph, xy);
    // Normal from central differences on the same surface, one grid step apart.
    float hx = surfaceHeight(t, d, detail, pages, tiles, node.lod, morph, xy + float2(spacing, 0.0))
             - surfaceHeight(t, d, detail, pages, tiles, node.lod, morph, xy - float2(spacing, 0.0));
    float hy = surfaceHeight(t, d, detail, pages, tiles, node.lod, morph, xy + float2(0.0, spacing))
             - surfaceHeight(t, d, detail, pages, tiles, node.lod, morph, xy - float2(0.0, spacing));

    VOut o;
    float3 world = float3(xy, h);
    // Camera-relative: subtract before projecting so precision stays high near the viewer.
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.normal = normalize(float3(-hx, -hy, 2.0 * spacing));
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    o.light = float2(skyVisibility.sample(linear, xy / t.domainSize, level(0)).r, 1.0);
    // The horizon map's nearest cell: horizon angles don't interpolate across a ridge.
    int side = int(horizons.get_width());
    uint2 cell = uint2(clamp(int2(xy / t.domainSize * float(side)), int2(0), int2(side - 1)));
    o.horizon0 = horizons.read(cell, 0) * M_PI_2_F;
    o.horizon1 = horizons.read(cell, 1) * M_PI_2_F;
    o.horizon2 = horizons.read(cell, 2) * M_PI_2_F;
    o.horizon3 = horizons.read(cell, 3) * M_PI_2_F;
    return o;
}

// Tree shadows fade out over the last stretch of their range while the canopy stand-in
// fades in, so neither pops at the edge.
constant float kTreeShadowFade = 40.0;

// The sun's disc is 0.53° across; shadows are softened over about a degree, like the
// CPU bake, which also hides the mesh's facets in shadow edges.
constant float kSunConeRadius = 0.01;

// Fraction of 4 rays towards the sun's disc that escape the terrain. A per-pixel rotation
// (interleaved gradient noise) spreads the 4 samples into a smooth penumbra.
float rayTracedSun(float3 p, float3 n, float3 l, float2 pixel, float distance, float maxHeight,
                   primitive_acceleration_structure terrain) {
    float3 t = normalize(cross(l, abs(l.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
    float3 b = cross(l, t);
    float spin = 6.2831853 * fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
    // Leave the surface along the normal so the ray can't hit its own triangle.
    float3 origin = p + n * 1.0;
    intersector<triangle_data> isect;
    isect.accept_any_intersection(true);
    // The shadow mesh is coarser than the drawn terrain, so a valley floor can sit below
    // it. Rays only stop on faces turned towards them: from below, they pass through.
    isect.set_triangle_front_facing_winding(winding::clockwise);
    isect.set_triangle_cull_mode(triangle_cull_mode::back);
    // Two rays from opposite sides of the sun's disc first: where they agree the point is
    // almost surely fully lit or fully shadowed, and only penumbrae pay for the other two.
    // Far away a penumbra is narrower than a pixel, so two rays are enough there.
    float hits[4];
    const int order[4] = { 1, 3, 0, 2 };
    for (int i = 0; i < 4; i++) {
        int k = order[i];
        float r = kSunConeRadius * sqrt((float(k) + 0.5) / 4.0);
        float a = float(k) * 2.3999632 + spin;
        float3 d = normalize(l + (t * cos(a) + b * sin(a)) * r);
        // Nothing can block a ray once it has climbed above the highest peak.
        float reach = d.z > 1e-4 ? (maxHeight + 1.0 - origin.z) / d.z : 200000.0;
        ray shadowRay(origin, d, 0.5, clamp(reach, 1.0, 200000.0));
        hits[i] = isect.intersect(shadowRay, terrain).type == intersection_type::none ? 1.0 : 0.0;
        if (i == 1 && (hits[0] == hits[1] || distance > 4000.0)) return 0.5 * (hits[0] + hits[1]);
    }
    return 0.25 * (hits[0] + hits[1] + hits[2] + hits[3]);
}

// How far the sun stands above the terrain's horizon (radians), from the 16-sector
// horizon map interpolated at the sun's azimuth. `bounds` gets the margin against the
// higher and the lower of the two sectors either side: where even the higher one leaves
// the sun clear, or even the lower one hides it, the answer can't be in doubt.
float horizonMargin(float4 h0, float4 h1, float4 h2, float4 h3, float3 l, thread float2 &bounds) {
    float h[16] = {
        h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w,
        h2.x, h2.y, h2.z, h2.w, h3.x, h3.y, h3.z, h3.w };
    float azimuth = atan2(l.y, l.x);
    if (azimuth < 0.0) azimuth += 6.2831853;
    float f = azimuth / (6.2831853 / 16.0) - 0.5;
    float f0 = floor(f);
    int a = (int(f0) + 16) % 16;
    int b = (a + 1) % 16;
    float horizonAngle = mix(h[a], h[b], f - f0);
    float elevation = asin(clamp(l.z, -1.0, 1.0));
    bounds = float2(elevation - max(h[a], h[b]), elevation - min(h[a], h[b]));
    return elevation - horizonAngle;
}

// The traced sun (sunShadow): terrain visibility (r) times what the trees let through (g).
float tracedSunAt(texture2d<float, access::read> traced, uint2 pixel) {
    float4 v = traced.read(pixel);
    return v.x * v.y;
}

// Sun visibility from the horizon map, with the same soft edge as the CPU bake.
float horizonSun(VOut in, float3 l) {
    float2 bounds;
    float margin = horizonMargin(in.horizon0, in.horizon1, in.horizon2, in.horizon3, l, bounds);
    return smoothstep(0.0, 1.0, saturate((margin + 0.01) / 0.02));
}

// The lighting tail shared by the terrain and the trees (FloraShaders), so the two can't
// drift apart. Sun (already scaled by its visibility, traced shadows and cloud), sky light
// from the open part of the sky, and one bounce off the surrounding terrain
// (global-illumination.md §6, cheap version): it fills the sky the terrain blocks, plus
// the ground a tilted face looks down at. Neighbours are taken to share this albedo and to
// be half in sun. `occlusion` darkens the sky and bounce light (AO inside a crown; 1 on
// the terrain). Mirrors Look.lit.
float3 litSurface(constant Uniforms &u, float3 albedo, float3 n, float nDotL, float sunVisibility,
                  float skyVisibility, float occlusion, float cloudSun) {
    float3 l = normalize(u.sunDirection);
    float3 direct = u.sunRadiance * (nDotL * sunVisibility);
    float3 ambient = skyIrradiance(u, n) * (skyVisibility * occlusion);
    float3 groundLight = u.sunRadiance * (0.5 * max(l.z, 0.0) * cloudSun) + skyIrradiance(u, float3(0.0, 0.0, 1.0));
    float terrainSeen = (1.0 - skyVisibility) * (0.5 + 0.5 * n.z) + (0.5 - 0.5 * n.z);
    float3 bounced = albedo * groundLight * (terrainSeen * occlusion);
    return albedo * (direct + ambient + bounced);
}

// Aerial perspective (haze takes the colour of the sky behind it, so it brightens towards
// the sun and warms at sunset), the clouds in front, and tone mapping. The render target is
// an _srgb format, which applies the sRGB encoding on write.
float4 finishSurface(constant Uniforms &u, texture2d<float> skyView, texture2d<float> clouds,
                     texture2d<float> cloudDistance, float3 color, float3 world, float2 pixel) {
    float3 toEye = u.cameraPosition - world;
    float dist = length(toEye);
    color = mix(color, skyAbove(skyView, -toEye), kHazeMax * (1.0 - exp(-dist / kHazeDistance)));
    color = compositeClouds(u, clouds, cloudDistance, color, pixel, dist);
    return float4(pbrNeutral(color * u.exposure), 1.0);
}

// ---- Ground textures (GroundTextures; docs/flora.md §9) ----
// Near the camera the ground shows its community's tileable texture (litter, soil, sand,
// moss, rock) instead of the canopy colour seen from above, fading back to it between
// kGroundNear and kGroundFar.
constant float kGroundNear = 150.0, kGroundFar = 300.0;
// Metres per repeat of the two scales the texture is laid at.
constant float kGroundTile = 2.0, kGroundTileWide = 5.3;
constant int kGroundLayers = 11;    // communities; bare rock follows
constant int kGroundRock = 11;

// Value noise in 0…1 on a unit lattice.
float groundNoise(float2 q) {
    float2 i = floor(q), f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    float v[4];
    for (int k = 0; k < 4; k++) {
        float2 c = i + float2(float(k & 1), float(k >> 1));
        uint h = uint(int(c.x)) * 0x8da6b343u ^ uint(int(c.y)) * 0xd8163841u;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
        v[k] = float(h & 0xffffu) / 65535.0;
    }
    return mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
}

struct GroundTexel {
    float4 albedo;          // rgb linear, a height 0…1
    float3 normal;          // tangent space: x along world x, y along world y, z out
};

// One layer at two scales, the wider one turned, switched between by a noise field tens
// of metres across, so neither repeat lines up into a grid.
GroundTexel groundLayer(texture2d_array<float> albedo, texture2d_array<float> normal, int layer, float2 p, float wide) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));
    float2 a = p / kGroundTile;
    float2 b = float2(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y) / kGroundTileWide + 0.37;
    // Normal maps: OpenGL x and y in red and green, height in blue.
    // Only where the switch between them is under way does a pixel sample both.
    float3 pa = 0.0, pb = 0.0, ca = 0.0, cb = 0.0;
    if (wide < 0.99) {
        pa = normal.sample(s, a, uint(layer)).xyz;
        ca = albedo.sample(s, a, uint(layer)).rgb;
    }
    if (wide > 0.01) {
        pb = normal.sample(s, b, uint(layer)).xyz;
        cb = albedo.sample(s, b, uint(layer)).rgb;
    }
    float2 na = pa.xy * 2.0 - 1.0, nb = pb.xy * 2.0 - 1.0;
    // The wider scale's normal turned back into world axes.
    nb = float2(0.8 * nb.x + 0.6 * nb.y, -0.6 * nb.x + 0.8 * nb.y);
    GroundTexel t;
    t.albedo = float4(mix(ca, cb, wide), mix(pa.z, pb.z, wide));
    // OpenGL normals: y up the image, and v runs down it, so world y is −y.
    float2 n = mix(na, nb, wide);
    t.normal = float3(n.x, -n.y, sqrt(saturate(1.0 - dot(n, n))));
    return t;
}

// Blends b over a by `t` (0…1), letting the higher texels of each poke through the other
// over a short height band, so borders break up along stones and litter, not smoothly. A
// layer needs about a quarter share before its highest texels show through.
GroundTexel heightBlend(GroundTexel a, GroundTexel b, float t) {
    float ha = 0.5 * a.albedo.a + (1.0 - t), hb = 0.5 * b.albedo.a + t;
    float top = max(ha, hb) - 0.15;
    float wa = max(ha - top, 0.0), wb = max(hb - top, 0.0);
    float k = wb / max(wa + wb, 1e-4);
    GroundTexel o;
    o.albedo = mix(a.albedo, b.albedo, k);
    o.normal = mix(a.normal, b.normal, k);
    return o;
}

// The textured ground at p: the two communities weighing most there (the weights carry the
// ecotone noise already; a little more breaks the bilinear ramps between 80 m texels into
// patches), slope-weighted rock as in terrainAlbedo, sand at the waterline. Writes the
// shading normal through `n`.
float3 groundTexture(texture2d_array<float> albedo, texture2d_array<float> normal, texture2d<float> w0,
                     texture2d<float> w1, texture2d<float> w2, float2 p, float uvScale, float above, float slope,
                     thread float3 &n) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 uv = p * uvScale;
    float4 a = w0.sample(linear, uv), b = w1.sample(linear, uv), c = w2.sample(linear, uv);
    float w[kGroundLayers] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z };
    int first = 0, second = 1;
    float wf = -1.0, ws = -1.0;
    // One noise value, spread over the communities by golden-ratio steps.
    float jitter = groundNoise(p / 9.0);
    for (int i = 0; i < kGroundLayers; i++) {
        float v = w[i] * (0.75 + 0.5 * fract(jitter + 0.618034 * float(i)));
        if (v > wf) { second = first; ws = wf; first = i; wf = v; }
        else if (v > ws) { second = i; ws = v; }
    }
    float wide = smoothstep(0.35, 0.65, groundNoise(p / 23.0 + 11.0));
    GroundTexel t = groundLayer(albedo, normal, first, p, wide);
    float share = ws / max(wf + ws, 1e-4);
    if (share > 0.02) t = heightBlend(t, groundLayer(albedo, normal, second, p, wide), share);
    // Sand where the far albedo has it, a band just above the waterline on gentle slopes,
    // but only along the shore: dry forest on a low plain stands on its own soil.
    float sand = (1.0 - max(smoothstep(1.0, 3.0, above), smoothstep(0.1, 0.25, slope))) * saturate(2.0 * (a.x + a.y));
    if (sand > 0.01 && first != 0) t = heightBlend(t, groundLayer(albedo, normal, 0, p, wide), sand);
    float rock = smoothstep(0.4, 0.7, slope) * (0.2 + 0.6 * (1.0 - exposure(p)));
    if (rock > 0.01) t = heightBlend(t, groundLayer(albedo, normal, kGroundRock, p, wide), rock);
    // Brightness drifting over a few metres, as wetness and litter do.
    float3 colour = t.albedo.rgb * (0.88 + 0.24 * groundNoise(p / 3.7 + 5.0));
    float3 tx = normalize(float3(1.0, 0.0, 0.0) - n * n.x);
    float3 ty = cross(n, tx);
    // The maps' relief at 60%: at full strength the deepest crevices turn from the sun
    // into black specks.
    n = normalize((tx * t.normal.x + ty * t.normal.y) * 0.6 + n * max(t.normal.z, 0.05));
    return colour;
}

// Mirrors Look.lit and GroundView.shade; constants come from Look.metalPrelude, the sun
// and sky from SunLight through the uniforms.
fragment float4 terrainFragment(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                texture2d<float> skyView [[texture(0)]],
                                texture2d<float> cloudMap [[texture(1)]],
                                texture2d<float> clouds [[texture(2)]],
                                texture2d<float> cloudDistance [[texture(3)]],
                                texture2d<float> cloudShadows [[texture(4)]],
                                texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                                texture2d<float> canopyMap [[texture(6)]],
                                texture2d<float> canopyCover [[texture(7)]],
                                texture2d_array<float> groundAlbedo [[texture(12)]],
                                texture2d_array<float> groundNormal [[texture(13)]],
                                texture2d<float> groundWeights0 [[texture(14)]],
                                texture2d<float> groundWeights1 [[texture(15)]],
                                texture2d<float> groundWeights2 [[texture(16)]],
                                constant float2 &floraRange [[buffer(2)]]) {
    float3 n = normalize(in.normal);
    float slope = 1.0 - n.z;
    float h = in.world.z - u.seaLevel;
    float3 l = normalize(u.sunDirection);

    float3 albedo;
    float3 surface = in.world;
    // Ground under the trees drawn around the camera (out to floraRange.x): the crowns hide
    // half its sky, by the stand's canopy cover. They block its sun too: within their
    // shadow maps' range (floraRange.y, 0 without) the maps do that, and beyond
    // it the cover stands in for them, fading in where those fade out. Further out the
    // terrain colour is the canopy itself, seen from above.
    float underCanopy = 0.0, sunCover = 0.0;
    if (h < 0.0) {
        albedo = waterAlbedo(-h);
        n = float3(0.0, 0.0, 1.0);
        surface.z = u.seaLevel; // shadows fall on the water, not the sea floor
    } else {
        constexpr sampler linear(filter::linear, address::clamp_to_edge);
        float4 canopy = canopyMap.sample(linear, in.world.xy / u.cloudLayer.w);
        // A 1×1 weight map: no communities in the bundle, so no ground textures.
        bool communities = groundWeights0.get_width() > 1;
        // The waterline's sand band only along the shore (strand and rocky shore weights),
        // as groundTexture has it: dry forest on a low plain stands on its own soil.
        float shore = communities ? saturate(2.0 * dot(groundWeights0.sample(linear, in.world.xy / u.cloudLayer.w).xy, float2(1.0))) : 1.0;
        albedo = terrainAlbedo(in.world.xy, max(h, 3.0 * (1.0 - shore)), slope, u.maxHeight, canopy);
        float d = distance(u.cameraPosition, in.world);
        float near = communities ? 1.0 - smoothstep(kGroundNear, kGroundFar, d) : 0.0;
        if (near > 0.0) {
            float3 textured = n;
            float3 ground = groundTexture(groundAlbedo, groundNormal, groundWeights0, groundWeights1, groundWeights2,
                                          in.world.xy, 1.0 / u.cloudLayer.w, h, slope, textured);
            albedo = mix(albedo, ground, near);
            n = normalize(mix(n, textured, near));
        }
        float nearby = 1.0 - smoothstep(floraRange.x - 100.0, floraRange.x, d);
        if (nearby > 0.0) underCanopy = nearby * canopyCover.sample(linear, in.world.xy / u.cloudLayer.w).r;
        sunCover = underCanopy * smoothstep(floraRange.y - kTreeShadowFade, floraRange.y, d);
    }

    float skyVisibility = saturate(in.light.x);
    float nDotL = saturate(dot(n, l));
    // Cloud shadows drifting over the terrain; under thick cloud no ray is worth tracing.
    float cloudSun = cloudSunlight(u, cloudShadows, surface);
    float sunVisibility = 0.0;
    if (nDotL > 0.0 && l.z > -0.01 && cloudSun > 0.01) {
        // Traced per pixel by sunShadow, over the surface this pixel shows.
        if (kRayTraced) {
            sunVisibility = tracedSunAt(tracedSun, uint2(in.position.xy));
        } else {
            sunVisibility = horizonSun(in, l);
        }
    }
    sunVisibility *= cloudSun;
    // Sunflecks get through gaps in the crowns; the sky is half hidden by them.
    sunVisibility *= 1.0 - 0.85 * sunCover;
    float3 color = litSurface(u, albedo, n, nDotL, sunVisibility, skyVisibility, 1.0 - 0.5 * underCanopy, cloudSun);

    float3 toEye = u.cameraPosition - in.world;
    float dist = length(toEye);
    if (h < 0.0) {
        // Flat water mirrors the sky at the view ray's depression angle.
        float cosView = toEye.z / max(dist, 1e-3);
        float3 reflected = float3(-toEye.xy, toEye.z);
        color = mix(color, skyAbove(skyView, reflected), waterReflectance(cosView, dist));
    }
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}

// Depth prepass with ray-traced shadows: the surface each pixel shows (world position,
// and its normal as a packed octahedral pair) for sunShadow. Water shows its surface.
// The normal's lowest bit marks tree pixels (FloraShaders), which stand above the ground
// the horizon map describes.
constant uint kTreeSurface = 1u;
// Bit 16 (the y component's lowest) marks impostor pixels, whose flat quads stand inside
// their own tree's crown.
constant uint kImpostorSurface = 1u << 16;
float2 octEncode(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 o = n.xy;
    if (n.z < 0.0) o = (1.0 - abs(n.yx)) * select(float2(-1.0), float2(1.0), n.xy >= 0.0);
    return o;
}

float3 octDecode(float2 o) {
    float3 n = float3(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * select(float2(-1.0), float2(1.0), n.xy >= 0.0);
    return normalize(n);
}

fragment uint4 terrainSurface(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    float3 p = in.world;
    if (p.z < u.seaLevel) {
        n = float3(0.0, 0.0, 1.0);
        p.z = u.seaLevel;  // shadows fall on the water, not the sea floor
    }
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z), pack_float_to_snorm2x16(octEncode(n)) & ~(kTreeSurface | kImpostorSurface));
}

// Sun visibility (without cloud) for every terrain pixel, traced in a compute pass over the
// prepass surface rather than in the shading pass: there, tiny distant triangles leave
// most of each 2×2 quad as helper lanes that trace rays too.
struct SunSurface {
    float3 p;
    float3 n;
    bool sky;
    bool tree;
    bool impostor;
};

SunSurface sunSurface(texture2d<uint, access::read> surface, depth2d<float, access::read> depth, uint2 g) {
    SunSurface s;
    s.sky = depth.read(g) >= 1.0;
    uint4 v = surface.read(g);
    s.p = float3(as_type<float>(v.x), as_type<float>(v.y), as_type<float>(v.z));
    s.n = octDecode(unpack_snorm2x16_to_float(v.w));
    s.tree = (v.w & kTreeSurface) != 0;
    s.impostor = (v.w & kImpostorSurface) != 0;
    return s;
}

// Height under xy of the overview mesh the rays are traced against: vertices at cell
// centres, each quad split along its (i, j)–(i+1, j+1) diagonal (Grid.meshHeight).
float shadowMeshHeight(texture2d<float, access::read> heights, float2 xy, float domain) {
    int2 size = int2(heights.get_width(), heights.get_height());
    float2 uv = clamp(xy / domain * float2(size) - 0.5, float2(0.0), float2(size - 1));
    int2 c = min(int2(uv), max(size - 2, int2(0)));
    int2 c1 = min(c + 1, size - 1);
    float2 f = uv - float2(c);
    float a = heights.read(uint2(c)).r, b = heights.read(uint2(c1.x, c.y)).r;
    float e = heights.read(uint2(c.x, c1.y)).r, d = heights.read(uint2(c1)).r;
    return f.x >= f.y ? a + f.x * (b - a) + f.y * (d - b) : a + f.y * (e - a) + f.x * (d - e);
}

// ---- Tree shadows: cascaded shadow maps (FloraRenderer.renderShadows) ----

// One cascade, as ShadowCascades.Cascade: clip-space rows for a camera-relative point
// (x, y across the map, Metal clip space; z its depth, 0 nearest the sun), and its sizes.
struct TreeCascade {
    float4 x;
    float4 y;
    float4 z;
    float4 info;            // texel (m), depth range (m), split distance (m), half-width (m)
};

constant int kTreeCascades = 3;

struct TreeShadows {
    TreeCascade cascades[kTreeCascades];
    float4 range;           // shadow range (m), then each cascade's side over the map array's
};

// Half the sun's disc (0.53° across), radians.
constant float kSunRadius = 0.00463;
// Occluders further than this above a receiver are not searched for (metres along the sun).
constant float kBlockerReach = 30.0;
// Light that still gets through a crown's shadow: leaves pass ~10% (green) and the LOD
// meshes' cluster cards close gaps a real crown has.
constant float kCrownLeak = 0.15;
// Penumbrae this many times the sun disc's (1: as wide as the disc makes them).
constant float kPenumbraScale = 1.0;

constant float2 kPoisson[16] = {
    float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725), float2(-0.09418410, -0.92938870),
    float2(0.34495938, 0.29387760), float2(-0.91588581, 0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543, 0.27676845), float2(0.97484398, 0.75648379), float2(0.44323325, -0.97511554),
    float2(0.53742981, -0.47373420), float2(-0.26496911, -0.41893023), float2(0.79197514, 0.19090188),
    float2(-0.24188840, 0.99706507), float2(-0.81409955, 0.91437590), float2(0.19984126, 0.78641367),
    float2(0.14383161, -0.14100790) };

// Sun the trees let through at camera-relative p in cascade c: a blocker search finds how
// far above the receiver the leaves are, which sets the penumbra (PCSS), then a rotated
// Poisson disc of hardware-filtered comparisons. Trunks meeting the ground stay sharp, the
// shadow of a crown 15 m up softens over ~0.2 m. `offset`: the normal offset's direction;
// `share`: the cascade's side over the map array's (it draws into that corner).
float cascadeSun(depth2d_array<float> maps, TreeCascade k, uint c, float share, float3 p, float3 offset, float2x2 spin) {
    float texel = k.info.x, range = k.info.y, width = 2.0 * k.info.w / share;
    p += offset * (1.5 * texel);
    float3 q = float3(dot(k.x.xyz, p) + k.x.w, dot(k.y.xyz, p) + k.y.w, dot(k.z.xyz, p) + k.z.w);
    float2 uv = float2(0.5 + 0.5 * q.x, 0.5 - 0.5 * q.y) * share;
    float depth = q.z - texel / range;  // constant bias: one texel
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    float search = max(kSunRadius * kBlockerReach, 1.5 * texel) / width;
    float blockers = 0.0, found = 0.0;
    for (int i = 0; i < 8; i++) {
        float d = maps.sample(nearest, uv + spin * kPoisson[2 * i] * search, c, level(0));
        if (d < depth) {
            blockers += d;
            found += 1.0;
        }
    }
    if (found == 0.0) return 1.0;
    float gap = (depth - blockers / found) * range;  // metres from the leaves to the receiver
    float radius = max(gap * kSunRadius * kPenumbraScale, 0.75 * texel) / width;
    // Deep in shadow: every tap of the search found leaves over an area wider than the
    // penumbra. Most of a closed forest's floor, which then skips the filter.
    if (found == 8.0 && radius < 0.7 * search) return 0.0;
    constexpr sampler compare(filter::linear, compare_func::less_equal, address::clamp_to_edge);
    float lit = 0.0;
    for (int i = 0; i < 16; i++) lit += maps.sample_compare(compare, uv + spin * kPoisson[i] * radius, c, depth, level(0));
    return lit / 16.0;
}

// Sun the trees let through at world point p, `distance` from the camera: the cascade by
// distance, blending into the next over the last fifth of each, and fading out over the
// last kTreeShadowFade metres of the range, where the canopy stand-in takes over.
float treeSunlight(constant Uniforms &u, constant TreeShadows &t, depth2d_array<float> maps, float3 p, float3 offset,
                   float distance, float2 pixel) {
    float range = t.range.x;
    if (distance >= range) return 1.0;
    float a = 6.2831853 * fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
    float2x2 spin = float2x2(float2(cos(a), sin(a)), float2(-sin(a), cos(a)));
    float3 rel = p - u.cameraPosition;
    uint c = distance < t.cascades[0].info.z ? 0u : (distance < t.cascades[1].info.z ? 1u : 2u);
    float v = cascadeSun(maps, t.cascades[c], c, t.range[c + 1u], rel, offset, spin);
    if (c + 1u < uint(kTreeCascades)) {
        float split = t.cascades[c].info.z;
        float next = saturate((distance - 0.8 * split) / (0.2 * split));
        if (next > 0.0) v = mix(v, cascadeSun(maps, t.cascades[c + 1u], c + 1u, t.range[c + 2u], rel, offset, spin), next);
    }
    v = kCrownLeak + (1.0 - kCrownLeak) * v;
    return mix(v, 1.0, smoothstep(range - kTreeShadowFade, range, distance));
}

// −1 where only rays can tell; otherwise the answer (sky, facing away, under thick cloud,
// or decided by the horizon map). `crown`: a tree pixel standing clear above the shadow mesh.
float sunWithoutRays(constant Uniforms &u, SunSurface s, bool crown, float3 l,
                     texture2d_array<float, access::read> horizons, texture2d<float> cloudShadows) {
    if (s.sky) return 1.0;
    // Leaves are two-sided and pass light through: a tree pixel facing away still counts.
    if ((!s.tree && dot(s.n, l) <= 0.0) || l.z <= -0.01 || cloudSunlight(u, cloudShadows, s.p) <= 0.01) return 0.0;
    // The horizon map comes from the same mesh the rays trace, so where it puts the sun
    // well clear of every nearby horizon, or well below them all, the rays would only
    // agree with it; the slack covers the drawn surface sitting off the coarse mesh.
    int side = int(horizons.get_width());
    uint2 cell = uint2(clamp(int2(s.p.xy / u.cloudLayer.w * float(side)), int2(0), int2(side - 1)));
    float2 bounds;
    horizonMargin(horizons.read(cell, 0) * M_PI_2_F, horizons.read(cell, 1) * M_PI_2_F,
                  horizons.read(cell, 2) * M_PI_2_F, horizons.read(cell, 3) * M_PI_2_F, l, bounds);
    if (bounds.x > 0.05) return 1.0;
    // A crown 15–25 m up can still see a sun the ground's horizon already hides. Below the
    // shadow mesh (a valley floor the coarse mesh bridges over) a ray would start under the
    // ridges and pass out through their backs, so trunks and low leaves there count as ground.
    if (bounds.y < -0.05 && !crown) return 0.0;
    return -1.0;
}

// Two passes over the prepass surface. Pass 0 traces the terrain at one pixel per 2×2
// block (crowns per 4×4). Pass 1 fills in the rest: where the traced pixels around one agree (all lit
// or all shadowed, on the same surface) it takes their answer, crowns among tree pixels
// take their mean whatever the depths (foliage being noisy anyway), and at shadow edges
// and penumbrae it traces its own rays. Each pixel then looks up the trees' shadow maps
// for itself. Writes the terrain's visibility (r) and the sun the trees let through (g);
// shading multiplies them. Impostors take no tree shadows: their flat quads stand inside
// their own tree's crown.
kernel void sunShadow(constant Uniforms &u [[buffer(0)]],
                      primitive_acceleration_structure terrain [[buffer(1)]],
                      constant uint &pass [[buffer(2)]],
                      texture2d<uint, access::read> surface [[texture(0)]],
                      depth2d<float, access::read> depth [[texture(1)]],
                      texture2d_array<float, access::read> horizons [[texture(2)]],
                      texture2d<float> cloudShadows [[texture(3)]],
                      texture2d<float, access::read_write> out [[texture(4)]],
                      texture2d<float, access::read> meshHeights [[texture(5)]],
                      constant TreeShadows &trees [[buffer(3), function_constant(kTreeShadows)]],
                      depth2d_array<float> treeMaps [[texture(6), function_constant(kTreeShadows)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint2 g = pass == 0u ? gid * 2u : gid;
    uint2 size = uint2(out.get_width(), out.get_height());
    if (any(g >= size)) return;
    // Pass 1 skips the pixels pass 0 traced, but not the crowns it left (blue set).
    if (pass == 1u && all((g & 1u) == 0u) && out.read(g).b < 0.5) return;
    float3 l = normalize(u.sunDirection);
    SunSurface s = sunSurface(surface, depth, g);
    float meshHeight = s.sky ? 0.0 : shadowMeshHeight(meshHeights, s.p.xy, u.cloudLayer.w);
    bool crown = s.tree && !s.sky && s.p.z > meshHeight + 1.0;
    float distance = length(u.cameraPosition - s.p);
    float visibility = sunWithoutRays(u, s, crown, l, horizons, cloudShadows);
    // Crowns trace the terrain at one pixel per 4×4 block: at low sun, when the horizon
    // map can't tell, every crown pixel needs rays, and crowns cover much of a forest view.
    // Foliage is noisy anyway; the rest take their 4×4 neighbours' mean in pass 1.
    if (visibility < 0.0 && crown && pass == 0u && any((g & 3u) != 0u)) {
        out.write(float4(0.0, 1.0, 1.0, 0.0), g);
        return;
    }
    if (visibility < 0.0 && crown && pass == 1u) {
        uint2 base = g & ~3u;
        uint2 off = select(uint2(0u), uint2(4u), (g & 3u) != 0u);
        uint2 last = (size - 1u) & ~3u;
        uint2 corners[4] = { min(base, last), min(base + uint2(off.x, 0u), last), min(base + uint2(0u, off.y), last),
                             min(base + off, last) };
        float sum = 0.0;
        bool foliage = true;
        for (int k = 0; k < 4; k++) {
            SunSurface t = sunSurface(surface, depth, corners[k]);
            foliage = foliage && t.tree && !t.sky;
            sum += out.read(corners[k]).r;
        }
        if (foliage) visibility = 0.25 * sum;
    }
    if (visibility < 0.0 && pass == 1u) {
        // The traced pixels of the 2×2 blocks touching this one.
        uint2 base = g & ~1u;
        uint2 off = (g & 1u) * 2u;
        uint2 last = (size - 1u) & ~1u;
        uint2 corners[4] = { min(base, last), min(base + uint2(off.x, 0u), last), min(base + uint2(0u, off.y), last),
                             min(base + off, last) };
        float lo = 1.0, hi = 0.0, sum = 0.0;
        bool same = true, foliage = crown;
        for (int k = 0; k < 4; k++) {
            SunSurface t = sunSurface(surface, depth, corners[k]);
            float4 c = out.read(corners[k]);
            // A crown pass 0 left has no answer yet.
            bool known = c.b < 0.5;
            same = same && known && !t.sky && length(t.p - s.p) < 0.02 * distance + 2.0;
            foliage = foliage && known && t.tree && !t.sky;
            float v = c.r;
            lo = min(lo, v);
            hi = max(hi, v);
            sum += v;
        }
        if (foliage) visibility = 0.25 * sum;
        else if (same && (hi <= 0.0 || lo >= 1.0)) visibility = lo;
    }
    if (visibility < 0.0) {
        // Crown pixels leave towards the sun: a card's normal may face away from it. Anything
        // else starts no lower than the shadow mesh: from under it (a valley floor the coarse
        // mesh bridges over) a ray would pass out through the backs of the ridges around it.
        float3 origin = crown ? s.p : float3(s.p.xy, max(s.p.z, meshHeight));
        visibility = rayTracedSun(origin, crown ? l : s.n, l, float2(g) + 0.5, distance, u.maxHeight, terrain);
    }
    // Leaves offset towards the sun (cards are two-sided), the ground along its normal.
    float leaves = 1.0;
    if (kTreeShadows && visibility > 0.0 && !s.sky && !s.impostor) {
        leaves = treeSunlight(u, trees, treeMaps, s.p, s.tree ? l : s.n, distance, float2(g) + 0.5);
    }
    out.write(float4(visibility, leaves, 0.0, 0.0), g);
}

// Sky: one full-screen triangle at the far plane, drawn after the terrain's depth prepass
// so it shades only where no terrain covers it.
struct SkyOut {
    float4 position [[position]];
    float2 ndc;
};

vertex SkyOut skyVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    SkyOut o;
    o.position = float4(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

fragment float4 skyFragment(SkyOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                            texture2d<float> skyView [[texture(0)]],
                            texture2d<float> clouds [[texture(2)]],
                            texture2d<float> cloudDistance [[texture(3)]]) {
    float4 far = u.inverseViewProjection * float4(in.ndc, 1.0, 1.0);
    float3 dir = normalize(far.xyz / far.w);
    float3 color = skyRadiance(skyView, dir);
    if (dir.z < 0.0) {
        // Open sea beyond the heightfield, shaded like the terrain shader's distant water
        // so the two meet without a seam.
        float height = max(u.cameraPosition.z - u.seaLevel, 1.0);
        float dist = height / max(-dir.z, 1e-5);
        float3 l = normalize(u.sunDirection);
        float3 water = kDeepWater * (u.sunRadiance * max(l.z, 0.0) + skyIrradiance(u, float3(0.0, 0.0, 1.0)));
        float cosView = -dir.z;
        water = mix(water, skyAbove(skyView, float3(dir.xy, -dir.z)), waterReflectance(cosView, dist));
        color = mix(water, skyAbove(skyView, dir), kHazeMax * (1.0 - exp(-dist / kHazeDistance)));
    }
    // The sun's disc (0.53°), far brighter than anything else; the tone mapper clips it.
    float cosSun = dot(dir, normalize(u.sunDirection));
    color += u.sunRadiance * (2000.0 * smoothstep(cos(0.0050), cos(0.0042), cosSun));
    // Below the horizon the view ends on the sea; clouds in between still count.
    color = compositeClouds(u, clouds, cloudDistance, color, in.position.xy, skyViewEnd(u, dir));
    return float4(pbrNeutral(color * u.exposure), 1.0);
}
"""
