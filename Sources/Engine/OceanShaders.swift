/// Metal source for the ocean surface (OceanRenderer; research/ocean.md).
///
/// `oceanPreludeSource` goes into the terrain source after `finishSurface`, so the sky pass
/// can shade the sea beyond the far plane the same way; `oceanShaderSource` is appended
/// after everything else.

/// Shared by the sky pass and the ocean: parameters, the sky environment and the shading.
let oceanPreludeSource = """
// ---- Ocean (OceanRenderer, OceanShaders) ----
struct OceanParams {
    float4 invPatch;        // 1 / patch side (m) per cascade
    float4 meanK;           // per cascade: mean wavenumber, for the TMA depth factor
    float4 slopeXX;         // per cascade: expected slope variance along x, y, and covariance
    float4 slopeYY;
    float4 slopeXY;
    float4 remainder;       // Cox–Munk slope variance no cascade resolves: up-, crosswind; wind dir (x, y)
    float4 origin[2];       // per cascade (xy, xy): a whole number of patches near the camera
    float4 morphStart[4];   // per LOD of the sea's quadtree
    float4 morphEnd[4];
    float4 grid;            // finest cell (m), LOD count, significant wave height (m), choppiness
    float4 time;            // seconds into the sea's loop, unused ×3
    float4 absorption;      // water absorption per metre (rgb)
    float4 scattering;      // molecular scattering per metre (rgb); w: particle scattering per metre
    float4 sss;             // subsurface tint (rgb), strength
};

constant float kWaterIndex = 1.333;

// Unpolarised Fresnel reflectance of water for a cosine `c` on the air side (Look.waterReflectance).
float waterFresnel(float c) {
    c = clamp(c, 1e-4, 1.0);
    float st2 = (1.0 - c * c) / (kWaterIndex * kWaterIndex);
    float ct = sqrt(max(1.0 - st2, 0.0));
    float rs = (c - kWaterIndex * ct) / (c + kWaterIndex * ct);
    float rp = (kWaterIndex * c - ct) / (kWaterIndex * c + ct);
    return 0.5 * (rs * rs + rp * rp);
}

// The upper hemisphere on a square (octahedral, turned 45°), for the sky environment.
float2 hemiOctEncode(float3 d) {
    d.z = max(d.z, 0.0);
    float2 p = d.xy / (abs(d.x) + abs(d.y) + d.z);
    return float2(p.x + p.y, p.x - p.y) * 0.5 + 0.5;
}

float3 hemiOctDecode(float2 uv) {
    float2 q = uv * 2.0 - 1.0;
    float2 p = float2(q.x + q.y, q.x - q.y) * 0.5;
    return normalize(float3(p, max(1.0 - abs(p.x) - abs(p.y), 0.0)));
}

// Sky (and the cloud cover in it) along `r`, blurred over a lobe `width` radians across.
float3 oceanEnvironment(texture2d<float> env, float3 r, float width) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    float texel = 1.6 / float(env.get_width());   // radians per texel, roughly
    float lod = log2(max(width / texel, 1.0));
    return env.sample(s, hemiOctEncode(r), level(lod)).rgb;
}

// Cascade c's patch coordinates at plane position xy.
float2 oceanUV(constant OceanParams &o, int c, float2 xy) {
    float2 origin = c < 2 ? (c == 0 ? o.origin[0].xy : o.origin[0].zw) : (c == 2 ? o.origin[1].xy : o.origin[1].zw);
    return (xy - origin) * o.invPatch[c];
}

// The TMA amplitude factor √φ(ω_h) for a deep-water wavenumber k over `depth` metres
// (OceanSpectrum.depthAttenuation).
float oceanDepthFactor(float k, float depth) {
    float w = sqrt(max(k * depth, 0.0));
    float phi = w <= 1.0 ? 0.5 * w * w : (w < 2.0 ? 1.0 - 0.5 * (2.0 - w) * (2.0 - w) : 1.0);
    return sqrt(phi);
}

// Unresolved (Cox–Munk) slope covariance turned into world axes: (xx, yy, xy).
float3 oceanRemainder(constant OceanParams &o) {
    float2 w = o.remainder.zw;
    float up = o.remainder.x, across = o.remainder.y;
    return float3(up * w.x * w.x + across * w.y * w.y, up * w.y * w.y + across * w.x * w.x, (up - across) * w.x * w.y);
}

// The water seen at one point. `n`: shading normal; `slope`: its mean slope (−n.xy / n.z);
// `variance`: slope covariance (xx, yy, xy) around it that no normal resolves; `v`: towards
// the eye; `floorLight`: the sea floor's light already through the water (0 for open sea);
// `path`: metres the view travels underwater to that floor (huge for open sea); `sunLight`:
// sun visibility × cloud; `crest`: height above the mean sea (m). Pre-exposure radiance.
float3 oceanShade(constant Uniforms &u, constant OceanParams &o, texture2d<float> env, float3 n, float2 slope,
                  float3 variance, float3 v, float3 floorLight, float path, float sunLight, float crest) {
    float3 l = normalize(u.sunDirection);
    // Facets turned towards the eye cover more of what it sees, the more so the more grazing
    // the view: the mean seen facet leans towards the eye by about σ² tan θ (at most ~0.8σ,
    // the mean of the half that faces it). This is most of why a distant bay keeps its colour
    // instead of turning into a mirror of the horizon.
    float2 e = length(v.xy) > 1e-4 ? normalize(v.xy) : float2(1.0, 0.0);
    float along = variance.x * e.x * e.x + 2.0 * variance.z * e.x * e.y + variance.y * e.y * e.y;
    float tanView = length(v.xy) / max(v.z, 0.02);
    float lean = min(along * tanView, 0.8 * sqrt(max(along, 0.0)));
    n = normalize(float3(n.xy / max(n.z, 0.2) + e * lean, 1.0));
    float nv = max(dot(n, v), 1e-3);
    float fresnel = waterFresnel(nv);
    // Reflection: the sky environment along the mirrored ray, blurred by the unresolved slopes
    // (a reflected ray turns twice as far as the facet).
    float3 r = reflect(-v, n);
    r.z = max(r.z, 0.0);
    float sigma = sqrt(max(0.5 * (variance.x + variance.y), 1e-6));
    float3 color = fresnel * oceanEnvironment(env, r, 4.0 * sigma);

    // Sun glint (Cox & Munk): the share of facets whose slope mirrors the sun into the eye,
    // a Gaussian of their anisotropic slope covariance around the resolved mean slope.
    if (l.z > 0.0 && sunLight > 0.0) {
        float3 h = normalize(l + v);
        float2 hs = -h.xy / max(h.z, 0.05) - slope;
        float det = max(variance.x * variance.y - variance.z * variance.z, 1e-8);
        float q = (variance.y * hs.x * hs.x - 2.0 * variance.z * hs.x * hs.y + variance.x * hs.y * hs.y) / det;
        float p = exp(-0.5 * q) / (6.2831853 * sqrt(det));
        float hz = max(h.z, 0.05);
        float glint = waterFresnel(dot(h, v)) * p / (4.0 * max(v.z, 0.08) * hz * hz * hz * hz);
        color += u.sunRadiance * (glint * sunLight * smoothstep(0.0, 0.03, l.z));
    }

    // Light from below the surface: the floor, single scattering in the water column (sun
    // and sky, the sun dimming with depth), and the glow of sunlit crests seen from behind.
    float3 t = refract(-v, n, 1.0 / kWaterIndex);
    float down = max(-t.z, 0.05);
    float sunCos = sqrt(max(1.0 - (1.0 - l.z * l.z) / (kWaterIndex * kWaterIndex), 0.05));
    float3 a = o.absorption.rgb, bw = o.scattering.rgb;
    float bp = o.scattering.w;
    float3 c = a + bw + bp;
    // Downwelling light fades with depth by about absorption plus backscatter over the
    // refracted sun's cosine.
    float3 kd = (a + 0.5 * bw + 0.02 * bp) / sunCos;
    float3 kappa = c + kd * down;
    float3 column = (1.0 - exp(-kappa * min(path, 1e4))) / kappa;
    float3 sunDir = float3(-l.xy / kWaterIndex, -sunCos);
    float cosTheta = dot(sunDir, -t);
    // Water molecules: Rayleigh-like; particles: forward-peaked Henyey–Greenstein (g = 0.9).
    float rayleigh = 0.0596831 * (1.0 + cosTheta * cosTheta);
    float g = 0.9, hg = (1.0 - g * g) / (12.566371 * pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5));
    float sunIn = l.z > 0.0 ? (1.0 - waterFresnel(l.z)) * l.z / sunCos : 0.0;
    float3 sky = skyIrradiance(u, float3(0.0, 0.0, 1.0));
    float3 inscatter = (bw * rayleigh + bp * hg) * (u.sunRadiance * (sunIn * sunLight))
                     + (bw + 0.02 * bp) * sky * (2.0 / 12.566371);
    float3 below = floorLight + inscatter * column;
    // Crests lit from behind: thin water the sun shines through, green-blue.
    float2 toSun = length(l.xy) > 1e-3 ? normalize(l.xy) : float2(1.0, 0.0);
    float2 look = length(v.xy) > 1e-3 ? -normalize(v.xy) : float2(0.0, 1.0);
    float back = pow(saturate(dot(look, toSun)), 4.0) * saturate(1.2 - l.z);
    float lift = saturate(crest / max(o.grid.z, 0.1) + 0.2);
    below += u.sunRadiance * o.sss.rgb * (o.sss.w * back * lift * sunLight * pow(1.0 - nv, 2.0));
    return color + (1.0 - fresnel) * below;
}

// The open sea with no detail left: a flat mean surface whose slopes are all variance.
float3 oceanFlatVariance(constant OceanParams &o) {
    float3 v = oceanRemainder(o);
    for (int c = 0; c < 4; c++) v += float3(o.slopeXX[c], o.slopeYY[c], o.slopeXY[c]);
    return v;
}

"""

let oceanShaderSource = """
// ---- Ocean: spectrum, FFT, displacement maps (OceanRenderer.update) ----
constant uint kOceanN = 256;

float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// Inverse FFTs of 256 points in threadgroup memory: 4 fields at once, 128 threads, radix-2
// decimation in time in place from bit-reversed input; e^{iπm/128} from a table.
void oceanButterflies(threadgroup float2 (*buffer)[256], threadgroup float2 *twiddle, uint t) {
    for (uint s = 1u; s < kOceanN; s <<= 1u) {
        uint j = t % s, i0 = (t / s) * 2u * s + j, i1 = i0 + s;
        float2 w = twiddle[j * (128u / s)];
        for (uint f = 0u; f < 4u; f++) {
            float2 a = buffer[f][i0], b = cmul(buffer[f][i1], w);
            buffer[f][i0] = a + b;
            buffer[f][i1] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Pass 1, one threadgroup per row of a cascade (the cascades this frame updates): h(k, t) = h0(k) e^{−iωt} + conj(h0(−k)) e^{iωt}
// and from it four complex fields, each two real ones packed as f + ig (Tessendorf 2001):
// horizontal displacement x + iy, height + i·∂Dx/∂y, slope ∂h/∂x + i·∂h/∂y, and
// ∂Dx/∂x + i·∂Dy/∂y; then the row's inverse FFT of each.
kernel void oceanRows(constant OceanParams &o [[buffer(0)]],
                      const device float4 *h0 [[buffer(1)]],
                      const device float *omega [[buffer(2)]],
                      device float2 *spectrum [[buffer(3)]],
                      constant uint4 &cascades [[buffer(4)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
    threadgroup float2 buffer[4][256];
    threadgroup float2 twiddle[128];
    uint y = group.x, c = cascades[group.y], plane = kOceanN * kOceanN;
    float a = 3.14159265 * float(t) / 128.0;
    twiddle[t] = float2(cos(a), sin(a));
    float dk = 6.2831853 * o.invPatch[c];
    float ky = (float(y) - float(kOceanN / 2)) * dk;
    for (uint x = t; x < kOceanN; x += 128u) {
        uint idx = c * plane + y * kOceanN + x;
        float4 h00 = h0[idx];
        float phase = omega[idx] * o.time.x;
        float cw = cos(phase), sw = sin(phase);
        float2 h = cmul(h00.xy, float2(cw, -sw)) + cmul(float2(h00.z, -h00.w), float2(cw, sw));
        float kx = (float(x) - float(kOceanN / 2)) * dk;
        float k = sqrt(kx * kx + ky * ky);
        float ik = k > 1e-6 ? 1.0 / k : 0.0;
        float2 ih = float2(-h.y, h.x);
        uint r = reverse_bits(x) >> 24;
        buffer[0][r] = ih * (kx * ik) - h * (ky * ik);
        buffer[1][r] = h - ih * (kx * ky * ik);
        buffer[2][r] = ih * kx - h * ky;
        buffer[3][r] = -h * (kx * kx * ik) - ih * (ky * ky * ik);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    oceanButterflies(buffer, twiddle, t);
    for (uint x = t; x < kOceanN; x += 128u) {
        for (uint f = 0u; f < 4u; f++) spectrum[(c * 4u + f) * plane + y * kOceanN + x] = buffer[f][x];
    }
}

// Pass 2, one threadgroup per column of a cascade: the columns' inverse FFTs, then the maps:
// displacement (λDx, λDy, h, λ∂Dx/∂y), derivatives (∂h/∂x, ∂h/∂y, λ∂Dx/∂x, λ∂Dy/∂y; with the
// displacement's last, the Jacobian), and the slopes' second moments for filtering (LEAN:
// variance = E[s²] − E[s]² in any mip).
kernel void oceanColumns(constant OceanParams &o [[buffer(0)]],
                         const device float2 *spectrum [[buffer(1)]],
                         texture2d_array<float, access::write> displacement [[texture(0)]],
                         texture2d_array<float, access::write> derivatives [[texture(1)]],
                         texture2d_array<float, access::write> moments [[texture(2)]],
                         constant uint4 &cascades [[buffer(4)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
    threadgroup float2 buffer[4][256];
    threadgroup float2 twiddle[128];
    uint x = group.x, c = cascades[group.y], plane = kOceanN * kOceanN;
    float a = 3.14159265 * float(t) / 128.0;
    twiddle[t] = float2(cos(a), sin(a));
    for (uint y = t; y < kOceanN; y += 128u) {
        uint r = reverse_bits(y) >> 24;
        for (uint f = 0u; f < 4u; f++) buffer[f][r] = spectrum[(c * 4u + f) * plane + y * kOceanN + x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    oceanButterflies(buffer, twiddle, t);
    float lambda = o.grid.w;
    for (uint y = t; y < kOceanN; y += 128u) {
        // The spectrum is centred on k = 0: undo the half-period shift.
        float sign = ((x + y) & 1u) != 0u ? -1.0 : 1.0;
        float2 d = buffer[0][y] * sign, b = buffer[1][y] * sign;
        float2 s = buffer[2][y] * sign, e = buffer[3][y] * sign;
        uint2 p = uint2(x, y);
        displacement.write(float4(lambda * d.x, lambda * d.y, b.x, lambda * b.y), p, c);
        derivatives.write(float4(s.x, s.y, lambda * e.x, lambda * e.y), p, c);
        moments.write(float4(s.x * s.x, s.y * s.y, s.x * s.y, 0.0), p, c);
    }
}

// The sky environment the sea reflects: the sky-view LUT lifted to the horizon, with the
// cloud cover over the camera at the cloud base (lit grey bases, hazed with distance).
kernel void oceanSkyEnvironment(constant Uniforms &u [[buffer(0)]],
                                texture2d<float> skyView [[texture(0)]],
                                texture2d<float> cloudMap [[texture(1)]],
                                texture2d<float, access::write> env [[texture(2)]],
                                uint2 g [[thread_position_in_grid]]) {
    uint side = env.get_width();
    if (g.x >= side || g.y >= side) return;
    float3 d = hemiOctDecode((float2(g) + 0.5) / float(side));
    float3 color = skyAbove(skyView, d);
    float base = u.cloudLayer.x;
    if (d.z > 0.02 && u.cameraPosition.z < base) {
        float t = (base - u.cameraPosition.z) / d.z;
        float cover = cloudCoverage(u, cloudMap, u.cameraPosition.xy + d.xy * t);
        float3 l = normalize(u.sunDirection);
        // Seen from near the sea, a cumulus shows its sunlit flanks as much as its base.
        float3 cloud = (u.sunRadiance * (0.25 + 0.6 * max(l.z, 0.0)) + skyIrradiance(u, float3(0.0, 0.0, 1.0))) * (0.85 / 3.14159265);
        cloud = mix(cloud, skyAbove(skyView, d), kHazeMax * (1.0 - exp(-t / kHazeDistance)));
        color = mix(color, cloud, 0.75 * smoothstep(0.15, 0.55, cover));
    }
    env.write(float4(color, 1.0), g);
}

// ---- Ocean: the surface (OceanRenderer.draw*) ----
struct OceanOut {
    float4 position [[position, invariant]];
    float3 world;           // displaced surface
    float2 plane;           // the undisplaced point, where the maps are read
    float depth;            // water depth under it (m)
    float crest;            // its height above the mean sea (m)
};

struct OceanMotionOut {
    float4 position [[position, invariant]];
    float3 world;           // displaced surface
    float4 now;             // this frame's clip position (jittered)
    float4 before;          // where last frame's surface put this point, in last frame's clip space
};

// The water's clip position, pulled a hair towards the eye in depth (an NDC offset worth
// ~1 mm at 100 m and tens of metres at 8 km). A few kilometres out depth steps grow coarser
// than the shallows are deep, and interpolation error of a step would let the sea floor poke
// through; land that low by the shore is under a pixel at that range.
float4 oceanClip(constant Uniforms &u, float3 world) {
    float4 clip = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    clip.z -= 2e-7 * clip.w;
    return clip;
}

// Metres per pixel at `distance` (60° vertical field of view).
float oceanFootprint(constant Uniforms &u, float distance) {
    return distance * 1.1547 / u.frame.y;
}

// Water depth under xy from the streamed heights at a level as coarse as the grid; off the
// heightfield, the open ocean.
float oceanWaterDepth(constant Uniforms &u, constant TerrainParams &t, const device int *pages,
                      texture2d_array<ushort, access::read> tiles, float2 xy, float spacing) {
    if (any(xy < 0.0) || any(xy > float2(t.domainSize))) return 4000.0;
    int level = clamp(int(floor(log2(max(spacing / t.cellMetres, 1.0)))), 0, t.levels - 1);
    return u.seaLevel - levelHeight(t, pages, tiles, level, xy);
}

struct OceanPoint {
    float2 plane;
    float depth;
    float filter;           // metres the displacement is filtered over (mesh and pixel)
};

// A vertex of a sea node's grid: CDLOD morphing onto the next level's grid towards the
// ring's far edge, as terrainVertex does.
OceanPoint oceanPoint(constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t, TerrainNode node,
                      uint vid, const device int *pages, texture2d_array<ushort, access::read> tiles) {
    uint row = uint(node.quads + 1);
    float2 g = float2(float(vid % row), float(vid / row));
    float spacing = ldexp(o.grid.x, node.lod);
    float2 xy = node.origin + g * spacing;
    float distance0 = distance(float3(xy, u.seaLevel), u.cameraPosition);
    float ms = at(o.morphStart, node.lod), me = at(o.morphEnd, node.lod);
    float morph = saturate((distance0 - ms) / max(me - ms, 1.0));
    xy -= fmod(g, 2.0) * spacing * morph;
    OceanPoint p;
    p.plane = xy;
    p.depth = oceanWaterDepth(u, t, pages, tiles, xy, spacing);
    float footprint = oceanFootprint(u, distance(float3(xy, u.seaLevel), u.cameraPosition));
    p.filter = max(spacing * (1.0 + morph), footprint);
    return p;
}

// Displacement at a point from the four cascades, each filtered to the grid (its mip) and
// faded out where its waves drop under ~2 filter widths, each shrunk by the depth.
float3 oceanDisplacement(constant OceanParams &o, texture2d_array<float> maps, OceanPoint p) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::repeat);
    float3 d = 0.0;
    for (int c = 0; c < 4; c++) {
        float wavelength = 6.2831853 / max(o.meanK[c], 1e-4);
        float w = smoothstep(2.0, 4.0, wavelength / p.filter);
        if (w <= 0.0) continue;
        float mip = log2(max(p.filter * float(kOceanN) * o.invPatch[c], 1.0));
        float4 v = maps.sample(s, oceanUV(o, c, p.plane), uint(c), level(mip));
        d += v.xyz * (w * oceanDepthFactor(o.meanK[c], p.depth));
    }
    // A trough can't sink through the bottom (which would show the floor dry in the swash).
    d.z = max(d.z, -0.9 * max(p.depth, 0.0));
    return d;
}

vertex OceanOut oceanVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                            constant Uniforms &u [[buffer(1)]],
                            constant TerrainParams &t [[buffer(3)]],
                            const device TerrainNode *nodes [[buffer(4)]],
                            const device int *pages [[buffer(5)]],
                            constant OceanParams &o [[buffer(7)]],
                            texture2d_array<ushort, access::read> tiles [[texture(0)]],
                            texture2d_array<float> displacement [[texture(4)]]) {
    OceanPoint p = oceanPoint(u, o, t, nodes[iid], vid, pages, tiles);
    float3 d = oceanDisplacement(o, displacement, p);
    OceanOut out;
    out.world = float3(p.plane + d.xy, u.seaLevel + d.z);
    out.position = oceanClip(u, out.world);
    out.plane = p.plane;
    out.depth = p.depth;
    out.crest = d.z;
    return out;
}

// Motion of the water itself for temporal antialiasing: the same point under last frame's
// displacement and camera.
vertex OceanMotionOut oceanMotionVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                        constant Uniforms &u [[buffer(1)]],
                                        constant TerrainParams &t [[buffer(3)]],
                                        const device TerrainNode *nodes [[buffer(4)]],
                                        const device int *pages [[buffer(5)]],
                                        constant OceanParams &o [[buffer(7)]],
                                        texture2d_array<ushort, access::read> tiles [[texture(0)]],
                                        texture2d_array<float> displacement [[texture(4)]],
                                        texture2d_array<float> previous [[texture(5)]]) {
    OceanPoint p = oceanPoint(u, o, t, nodes[iid], vid, pages, tiles);
    float3 d = oceanDisplacement(o, displacement, p);
    float3 world = float3(p.plane + d.xy, u.seaLevel + d.z);
    float3 d0 = oceanDisplacement(o, previous, p);
    float3 before = float3(p.plane + d0.xy, u.seaLevel + d0.z);
    OceanMotionOut out;
    out.world = world;
    out.position = oceanClip(u, world);
    out.now = out.position;
    out.before = u.previousViewProjection * float4(before - u.previousCamera.xyz, 1.0);
    return out;
}

// Depth prepass: the water's surface for sunShadow (with ray tracing) and its own motion
// for temporal antialiasing, written straight into the motion texture (with this frame's
// parity, so motionVectors leaves it) by the water that passes the depth test: drawn after
// everything else in the prepass, that is the water left in view.
void oceanWriteMotion(constant Uniforms &u, OceanMotionOut in, texture2d<float, access::write> motion) {
    float2 m = in.before.w > 1e-3 ? clipToUV(in.before) - unjitteredUV(u, in.now) : float2(0.0);
    motion.write(float4(m.x, motionWithParity(m.y, uint(u.frame.z) & 1u), 0.0, 0.0), uint2(in.position.xy));
}

uint4 oceanSurfaceTexel(float3 p) {
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z),
                 pack_float_to_snorm2x16(octEncode(float3(0.0, 0.0, 1.0))) & ~(kTreeSurface | kImpostorSurface));
}

[[early_fragment_tests]]
fragment uint4 oceanSurfaceMotion(OceanMotionOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                  texture2d<float, access::write> motion [[texture(10)]]) {
    oceanWriteMotion(u, in, motion);
    return oceanSurfaceTexel(in.world);
}

[[early_fragment_tests]]
fragment void oceanMotion(OceanMotionOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                          texture2d<float, access::write> motion [[texture(10)]]) {
    oceanWriteMotion(u, in, motion);
}

fragment uint4 oceanSurface(OceanOut in [[stage_in]]) {
    return oceanSurfaceTexel(in.world);
}

// The sea floor seen from the water surface at `p` along the refracted view `t`: where that ray
// meets the floor (one refinement from the depth straight below, which is the refraction's
// offset), its depth below the surface (`below`) and the metres travelled underwater (`path`),
// and its light as it leaves the floor (pre-exposure): sand, darker on knolls and patch reefs
// (convex over ~20 m, as terrainVertex's ridge), lit by the sun through the surface's shadows.
struct OceanFloor {
    float3 light;
    float below;
    float path;
};

OceanFloor oceanFloor(constant Uniforms &u, constant TerrainParams &t, const device int *pages,
                      texture2d_array<ushort, access::read> tiles, float3 p, float3 dir, float depth,
                      float footprint, float sunLight) {
    int level = clamp(int(floor(log2(max(footprint / t.cellMetres, 1.0)))), 0, t.levels - 1);
    float down = max(-dir.z, 0.05);
    float below = max(depth + (p.z - u.seaLevel), 0.0);
    float2 xy = p.xy + dir.xy * (below / down);
    float h = levelHeight(t, pages, tiles, level, xy);
    OceanFloor f;
    f.below = max(p.z - h, 0.0);
    f.path = f.below / down;
    xy = p.xy + dir.xy * f.path;
    const float r = 20.0;
    float hx0 = levelHeight(t, pages, tiles, level, xy - float2(r, 0.0)), hx1 = levelHeight(t, pages, tiles, level, xy + float2(r, 0.0));
    float hy0 = levelHeight(t, pages, tiles, level, xy - float2(0.0, r)), hy1 = levelHeight(t, pages, tiles, level, xy + float2(0.0, r));
    float3 n = normalize(float3(hx0 - hx1, hy0 - hy1, 2.0 * r));
    float ridge = -(hx0 + hx1 + hy0 + hy1 - 4.0 * h) / r;
    float reef = saturate(ridge * 3.0) * (1.0 - smoothstep(8.0, 25.0, u.seaLevel - h));
    float3 albedo = mix(kSand, kReef, reef) * (0.88 + 0.24 * groundNoise(xy / 3.7 + 5.0));
    float3 l = normalize(u.sunDirection);
    f.light = litSurface(u, albedo, n, saturate(dot(n, l)), sunLight, 1.0, 1.0, 1.0);
    return f;
}

// The water's colour, drawn over the shaded scene in the same pass (after the terrain, trees
// and ground cover). Normals and their filtered variance from the cascades' derivative and
// moment maps (each faded out to pure variance as it drops below the pixel); the sea floor
// beneath (oceanFloor), seen through the water column; then haze and clouds. Records which
// cascades it used in `used` (a bit each, once per SIMD group and only when new), so the
// renderer can skip the ones no pixel needs (the short waves seen from a summit).
[[early_fragment_tests]]
fragment float4 oceanFragment(OceanOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                              constant OceanParams &o [[buffer(7)]],
                              texture2d<float> skyView [[texture(0)]],
                              texture2d<float> clouds [[texture(2)]],
                              texture2d<float> cloudDistance [[texture(3)]],
                              texture2d<float> cloudShadows [[texture(4)]],
                              texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                              texture2d_array<float> derivatives [[texture(6)]],
                              texture2d_array<float> moments [[texture(7)]],
                              texture2d<float> environment [[texture(8)]],
                              constant TerrainParams &t [[buffer(3)]],
                              const device int *pages [[buffer(5)]],
                              texture2d_array<ushort, access::read> tiles [[texture(9)]],
                              device atomic_uint *used [[buffer(8)]]) {
    constexpr sampler aniso(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(4));
    float3 toEye = u.cameraPosition - in.world;
    float dist = length(toEye);
    float3 v = toEye / max(dist, 1e-3);
    float footprint = oceanFootprint(u, dist) / max(abs(v.z), 0.05);
    float2 slope = 0.0, stretch = 0.0;
    float3 variance = oceanRemainder(o);
    uint mask = 0u;
    for (int c = 0; c < 4; c++) {
        float atten = oceanDepthFactor(o.meanK[c], in.depth);
        float3 whole = float3(o.slopeXX[c], o.slopeYY[c], o.slopeXY[c]) * (atten * atten);
        // Past ~7 mips the map is its own mean: just variance.
        float mip = log2(max(footprint * float(kOceanN) * o.invPatch[c], 1.0));
        float w = 1.0 - smoothstep(5.0, 7.0, mip);
        if (w > 0.0) mask |= 1u << uint(c);
        if (w <= 0.0) {
            variance += whole;
            continue;
        }
        float2 uv = oceanUV(o, c, in.plane);
        float4 dv = derivatives.sample(aniso, uv, uint(c)) * atten;
        float4 mo = moments.sample(aniso, uv, uint(c)) * (atten * atten);
        slope += w * dv.xy;
        stretch += w * dv.zw;
        float3 filtered = max(float3(mo.x - dv.x * dv.x, mo.y - dv.y * dv.y, 0.0), 0.0);
        filtered.z = mo.z - dv.x * dv.y;
        variance += w * w * filtered + (1.0 - w * w) * whole;
    }
    uint group = simd_or(mask);
    if (simd_is_first() && (atomic_load_explicit(used, memory_order_relaxed) & group) != group) {
        atomic_fetch_or_explicit(used, group, memory_order_relaxed);
    }
    float2 s = slope / max(1.0 + stretch, 0.2);
    float3 n = normalize(float3(-s, 1.0));
    float3 l = normalize(u.sunDirection);

    float sunLight = cloudSunlight(u, cloudShadows, in.world);
    if (kRayTraced) sunLight *= tracedSunAt(tracedSun, uint2(in.position.xy));

    // The floor along the refracted view: its light dimmed on the way down to it and back up.
    // Under more than ~120 m of water none shows; past the heightfield there is none.
    float3 floorLight = 0.0;
    float path = 1e5;
    if (in.depth < 120.0) {
        float3 refracted = refract(-v, n, 1.0 / kWaterIndex);
        OceanFloor f = oceanFloor(u, t, pages, tiles, in.world, refracted, in.depth, footprint, sunLight);
        path = f.path;
        float sunCos = sqrt(max(1.0 - (1.0 - l.z * l.z) / (kWaterIndex * kWaterIndex), 0.05));
        float3 kd = (o.absorption.rgb + 0.5 * o.scattering.rgb + 0.02 * o.scattering.w) / sunCos;
        float3 c = o.absorption.rgb + o.scattering.rgb + o.scattering.w;
        floorLight = f.light * exp(-kd * f.below - c * path);
    }
    float3 color = oceanShade(u, o, environment, n, -n.xy / n.z, variance, v, floorLight, path, sunLight, in.crest);
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}
"""
