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

// The sun's disc is 0.53° across; shadows are softened over about a degree, like the
// CPU bake, which also hides the mesh's facets in shadow edges.
constant float kSunConeRadius = 0.01;

// Fraction of 4 rays towards the sun's disc that escape the terrain. A per-pixel rotation
// (interleaved gradient noise) spreads the 4 samples into a smooth penumbra.
float rayTracedSun(float3 p, float3 n, float3 l, float2 pixel, float distance, primitive_acceleration_structure terrain) {
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
        ray shadowRay(origin, d, 0.5, 200000.0);
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

// Sun visibility from the horizon map, with the same soft edge as the CPU bake.
float horizonSun(VOut in, float3 l) {
    float2 bounds;
    float margin = horizonMargin(in.horizon0, in.horizon1, in.horizon2, in.horizon3, l, bounds);
    return smoothstep(0.0, 1.0, saturate((margin + 0.01) / 0.02));
}

// Mirrors Look.lit and GroundView.shade; constants come from Look.metalPrelude, the sun
// and sky from SunLight through the uniforms.
fragment float4 terrainFragment(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                texture2d<float> skyView [[texture(0)]],
                                texture2d<float> cloudMap [[texture(1)]],
                                texture2d<float> clouds [[texture(2)]],
                                texture2d<float> cloudDistance [[texture(3)]],
                                texture2d<float> cloudShadows [[texture(4)]],
                                texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]]) {
    float3 n = normalize(in.normal);
    float slope = 1.0 - n.z;
    float h = in.world.z - u.seaLevel;
    float3 l = normalize(u.sunDirection);

    float3 albedo;
    float3 surface = in.world;
    if (h < 0.0) {
        albedo = waterAlbedo(-h);
        n = float3(0.0, 0.0, 1.0);
        surface.z = u.seaLevel; // shadows fall on the water, not the sea floor
    } else {
        albedo = terrainAlbedo(in.world.xy, h, slope, u.maxHeight);
    }

    float skyVisibility = saturate(in.light.x);
    float nDotL = saturate(dot(n, l));
    // Cloud shadows drifting over the terrain; under thick cloud no ray is worth tracing.
    float cloudSun = cloudSunlight(u, cloudShadows, surface);
    float sunVisibility = 0.0;
    if (nDotL > 0.0 && l.z > -0.01 && cloudSun > 0.01) {
        // Traced per pixel by sunShadow, over the surface this pixel shows.
        if (kRayTraced) {
            sunVisibility = tracedSun.read(uint2(in.position.xy)).r;
        } else {
            sunVisibility = horizonSun(in, l);
        }
    }
    sunVisibility *= cloudSun;
    float3 direct = u.sunRadiance * (nDotL * sunVisibility);
    // Sky light from the part of the sky the terrain leaves open.
    float3 ambient = skyIrradiance(u, n) * skyVisibility;
    // One bounce off the surrounding terrain (global-illumination.md §6, cheap version):
    // it fills the sky the terrain blocks, plus the ground a tilted face looks down at.
    // Neighbours are taken to share this albedo and to be half in sun.
    float3 groundLight = u.sunRadiance * (0.5 * max(l.z, 0.0) * cloudSun) + skyIrradiance(u, float3(0.0, 0.0, 1.0));
    float terrainSeen = (1.0 - skyVisibility) * (0.5 + 0.5 * n.z) + (0.5 - 0.5 * n.z);
    float3 bounced = albedo * groundLight * terrainSeen;
    float3 color = albedo * (direct + ambient + bounced);

    float3 toEye = u.cameraPosition - in.world;
    float dist = length(toEye);
    if (h < 0.0) {
        // Flat water mirrors the sky at the view ray's depression angle.
        float cosView = toEye.z / max(dist, 1e-3);
        float3 reflected = float3(-toEye.xy, toEye.z);
        color = mix(color, skyAbove(skyView, reflected), waterReflectance(cosView, dist));
    }
    // Aerial perspective: haze takes the colour of the sky behind it, so it brightens
    // towards the sun and warms at sunset.
    color = mix(color, skyAbove(skyView, -toEye), kHazeMax * (1.0 - exp(-dist / kHazeDistance)));
    color = compositeClouds(u, clouds, cloudDistance, color, in.position.xy, dist);

    color = pbrNeutral(color * u.exposure);
    // The render target is an _srgb format, which applies the sRGB encoding on write.
    return float4(color, 1.0);
}

// Depth prepass with ray-traced shadows: the surface each pixel shows (world position,
// and its normal as a packed octahedral pair) for sunShadow. Water shows its surface.
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
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z), pack_float_to_snorm2x16(octEncode(n)));
}

// Sun visibility (without cloud) for every terrain pixel, traced in a compute pass over the
// prepass surface rather than in the shading pass: there, tiny distant triangles leave
// most of each 2×2 quad as helper lanes that trace rays too.
kernel void sunShadow(constant Uniforms &u [[buffer(0)]],
                      primitive_acceleration_structure terrain [[buffer(1)]],
                      texture2d<uint, access::read> surface [[texture(0)]],
                      depth2d<float, access::read> depth [[texture(1)]],
                      texture2d_array<float, access::read> horizons [[texture(2)]],
                      texture2d<float> cloudShadows [[texture(3)]],
                      texture2d<float, access::write> out [[texture(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    if (depth.read(gid) >= 1.0) { out.write(float4(1.0), gid); return; }  // sky
    uint4 s = surface.read(gid);
    float3 p = float3(as_type<float>(s.x), as_type<float>(s.y), as_type<float>(s.z));
    float3 n = octDecode(unpack_snorm2x16_to_float(s.w));
    float3 l = normalize(u.sunDirection);
    float visibility = 0.0;
    if (dot(n, l) > 0.0 && l.z > -0.01 && cloudSunlight(u, cloudShadows, p) > 0.01) {
        // The horizon map comes from the same mesh the rays trace, so where it puts the sun
        // well clear of every nearby horizon, or well below them all, the rays would only
        // agree with it; the slack covers the drawn surface sitting off the coarse mesh.
        int side = int(horizons.get_width());
        uint2 cell = uint2(clamp(int2(p.xy / u.cloudLayer.w * float(side)), int2(0), int2(side - 1)));
        float2 bounds;
        horizonMargin(horizons.read(cell, 0) * M_PI_2_F, horizons.read(cell, 1) * M_PI_2_F,
                      horizons.read(cell, 2) * M_PI_2_F, horizons.read(cell, 3) * M_PI_2_F, l, bounds);
        if (bounds.x > 0.05) visibility = 1.0;
        else if (bounds.y < -0.05) visibility = 0.0;
        else visibility = rayTracedSun(p, n, l, float2(gid) + 0.5, length(u.cameraPosition - p), terrain);
    }
    out.write(float4(visibility), gid);
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
