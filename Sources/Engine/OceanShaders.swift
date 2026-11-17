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
    float4 time;            // seconds into the sea's loop, last frame's, surf waves' height scale, unused
    float4 absorption;      // water absorption per metre (rgb)
    float4 scattering;      // molecular scattering per metre (rgb); w: particle scattering per metre
    float4 sss;             // subsurface tint (rgb), strength
    float4 surf;            // significant height of the wind sea (with the trade swell) and of the swell (m),
                            // swell period (s), depth-limited Hs/h for the long waves (only w used now)
    float4 surfFrom;        // where the swell comes from (xy) and the wind sea (zw), unit vectors (unused)
    float4 waves;           // the surf's systems (the refraction texture's): trade swell and north
                            // swell periods (s), breaker index (Hs/h), waves per set
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

// Amplitude factor of cascade c over `depth` metres where the sea's shelter (SeaShelter) leaves
// it `shelter` of the open sea's (OceanSpectrum.shallowFactor for the long waves): the long-wave
// cascades keep their (sheltered) height until the depth limits the sea's significant height to
// surf.w × depth; the short ones (≤ ~6 m) by the TMA factor, which barely touches them.
float oceanWaveFactor(constant OceanParams &o, int c, float depth, float shelter) {
    if (c >= 2) return shelter * oceanDepthFactor(o.meanK[c], depth);
    return min(shelter, saturate(o.surf.w * max(depth, 0.0) / max(o.grid.z, 0.1)));
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
    // Crests hold thin, sunlit water that scatters more towards the eye; troughs sit in the
    // waves' shade and look deeper: the light from within the water rises and falls with the
    // height (neutral at the mean sea), so the swell's shapes read from a distance.
    float height = clamp(1.6 * crest / max(o.grid.z, 0.1), -1.0, 1.0);
    float3 below = floorLight + inscatter * column * (1.0 + 0.35 * height);
    // Crests lit from behind: thin water the sun shines through, green-blue.
    float2 toSun = length(l.xy) > 1e-3 ? normalize(l.xy) : float2(1.0, 0.0);
    float2 look = length(v.xy) > 1e-3 ? -normalize(v.xy) : float2(0.0, 1.0);
    float back = pow(saturate(dot(look, toSun)), 4.0) * saturate(1.2 - l.z);
    float lift = saturate(crest / max(o.grid.z, 0.1) + 0.2);
    below += u.sunRadiance * o.sss.rgb * (o.sss.w * back * lift * sunLight * pow(1.0 - nv, 2.0));
    // And a glow of the crests from any side, where the sun is up.
    float glow = saturate(height) * saturate(height);
    below += u.sunRadiance * o.sss.rgb * (0.35 * o.sss.w * glow * sunLight * saturate(l.z + 0.2));
    return color + (1.0 - fresnel) * below;
}

// The open sea with no detail left: a flat mean surface whose slopes are all variance.
float3 oceanFlatVariance(constant OceanParams &o) {
    float3 v = oceanRemainder(o);
    for (int c = 0; c < 4; c++) v += float3(o.slopeXX[c], o.slopeYY[c], o.slopeXY[c]);
    return v;
}

"""

let swashShaderSource = """
// ---- Swash (research/ocean.md, step O4): water running up the sand ----
// Each wave that reaches the shore runs up the beach as a thin sheet, slows, stops at its
// run-up, and drains back, over ~65% of the swell's period, timed by the same wave clock as the
// surf (OceanShaders' surfFoam). The terrain draws it over the sand (terrainFragment): the sheet
// (wet sand through a few centimetres of water, a sky sheen and glint, foam lace along the
// uprush's edge), sand the waves covered lately darker, foam stranded along the highest line.

// Beach slope for the run-up (Hawaiian sand beaches ~1:20), share of the period a swash
// lasts, and how far inland of the smoothed shoreline it is looked for (m).
constant float kSwashBeta = 0.05;
constant float kSwashShare = 0.65;
constant float kSwashReach = 80.0;

// The shore texture at uv, bilinear in full precision: the hardware's filter weights have 8
// bits, which on this coarse texture steps the shore distance by decimetres, and the swash's
// clock and the patterns it carries (×1/β) turn those steps into a lattice of metre-wide jumps.
float4 shoreAt(texture2d<float> shore, float2 uv) {
    float2 size = float2(shore.get_width(), shore.get_height());
    float2 s = clamp(uv * size - 0.5, float2(0.0), size - 1.001);
    float2 i = floor(s), f = s - i;
    uint2 a = uint2(i), b = min(a + 1u, uint2(size) - 1u);
    float4 t00 = shore.read(a), t10 = shore.read(uint2(b.x, a.y)), t01 = shore.read(uint2(a.x, b.y)), t11 = shore.read(b);
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

// A wave texel unpacked: each system's arrival, each one's height (the two halfs in z).
float4 surfWavesTexel(float4 t) {
    half2 h = as_type<half2>(t.z);
    return float4(t.xy, float2(h));
}

// The surf's wave texture (WaveRefraction: each system's arrival in its periods, its local
// significant height, and the water's depth) at xy, bilinear in full precision (the arrival
// runs to hundreds of periods): the arrivals and heights, and through `depth` the depth; with
// the arrivals' gradient per metre through `gradient` (x's in xy, y's in zw: the bilinear's own,
// so fronts drawn from it and their slopes agree).
float4 surfWavesAt(texture2d<float> waves, float2 xy, float domain, thread float4 &gradient, thread float &depth) {
    float2 size = float2(waves.get_width(), waves.get_height());
    float2 s = clamp(xy / domain * size - 0.5, float2(0.0), size - 1.001);
    float2 i = floor(s), f = s - i;
    uint2 a = uint2(i), b = min(a + 1u, uint2(size) - 1u);
    float4 r00 = waves.read(a), r10 = waves.read(uint2(b.x, a.y)), r01 = waves.read(uint2(a.x, b.y)), r11 = waves.read(b);
    depth = mix(mix(r00.w, r10.w, f.x), mix(r01.w, r11.w, f.x), f.y);
    float4 t00 = surfWavesTexel(r00), t10 = surfWavesTexel(r10), t01 = surfWavesTexel(r01), t11 = surfWavesTexel(r11);
    float scale = size.x / domain;
    gradient = float4(mix(t10.xy - t00.xy, t11.xy - t01.xy, f.y), mix(t01.xy - t00.xy, t11.xy - t10.xy, f.x)) * scale;
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

// The sets' timing along the coast (0…1, drifting over kilometres; surfSet), worked out once
// per vertex or pixel.
float surfSetDrift(float2 xy) {
    return groundNoise(xy / 3000.0 + 5.3);
}

// How big wave n of a system is within its set (0.45…1): the beat of two close periods, a set
// every o.waves.w waves, shifted along the coast by `drift` (surfSetDrift).
float surfSet(constant OceanParams &o, float n, float drift) {
    return 0.45 + 0.55 * abs(cos(3.14159265 * (n + o.waves.w * drift) / o.waves.w));
}

// The swash's wave clock at a beach point (from the wave texture there): the system bringing
// the most energy (its arrival in periods, period), and the height its bores reach the sand
// with as Stockdon's offshore height (the bake's local height on the sand is the breakers' last,
// depth-limited one: ×1.5, up to 2.5 m).
float3 swashClock(constant OceanParams &o, float4 waves, thread float &hs) {
    hs = min(1.5 * sqrt(waves.z * waves.z + waves.w * waves.w), 2.5);
    return waves.w > waves.z ? float3(waves.y, o.waves.y, 1.0) : float3(waves.x, o.waves.x, 0.0);
}

// Height (m above the sea) wave n runs up to at xy: ~0.4 of Stockdon et al.'s 2% run-up (a
// typical wave's) for waves of height hs and the swell's period on kSwashBeta, varying wave to wave
// in sections along the shore, with cusps every few metres.
float swashRunup(constant OceanParams &o, float hs, float period, float2 xy, float n, float drift) {
    float l0 = 9.81 * period * period / 6.2831853;
    float s = sqrt(max(hs, 0.05) * l0);
    float r2 = 1.1 * (0.35 * kSwashBeta * s + 0.5 * s * sqrt(0.563 * kSwashBeta * kSwashBeta + 0.004));
    float wave = 0.55 + 0.45 * groundNoise(xy / 120.0 + n * 1.37);
    float cusps = 1.0 + 0.3 * (groundNoise(xy / 6.0 + n * 0.53) - 0.5);
    return 0.38 * r2 * wave * cusps * surfSet(o, n, drift);
}

// Each wave's section along the shore (0…1, its own every wave; surfBreakers deals the
// whitewater's sections from it and the fronts peel by it). Turned off the noise's lattice,
// whose squares would show in the sections' ends.
float surfSection(float2 xy, float n, float period) {
    float2 r = float2(0.866 * xy.x - 0.5 * xy.y, 0.5 * xy.x + 0.866 * xy.y);
    return groundNoise(r / 160.0 + n * 1.37 + period);
}

// The surf's wave clock at xy: which wave's front the point is behind (the last to have passed
// it, its front peeled back 0.08 × its section: wave n's front is where wave + 0.08 s_n = n),
// with that wave's section and through `u0` how far behind it (in periods, 0 at the front and
// growing to the next front, ~1 on). Everything a wave carries (its set, sections, breaking)
// runs on unbroken from its front to the next one's: rounding `wave` instead would hand the
// trail over to the next wave half way, cutting it off in a seam along the crests. Through
// `next`, the next wave's section (surfCrest's face; unused, the compiler drops it).
float surfClock(float2 xy, float wave, float period, thread float &section, thread float &u0, thread float &next) {
    float k = floor(wave);
    // Peels are 0…0.08, so the front ahead of `wave` is k + 1's if it has already passed (with
    // its peel), and k's otherwise.
    float s = surfSection(xy, k + 1.0, period);
    float n = k + 1.0;
    if (wave + 0.08 * s < n) {
        n = k;
        next = s;
        s = surfSection(xy, k, period);
    } else {
        next = surfSection(xy, k + 2.0, period);
    }
    section = s;
    u0 = wave + 0.08 * s - n;
    return n;
}

// Where in its swash the wave at xy is (0…1 of the period since its bore reached the shore),
// and which wave (n): the surf's own clock (surfClock), so each swash starts as the bore that
// breaks on that stretch of shore arrives, sections peeling alike. `arrival` is the crests'
// arrival there in periods (the wave texture's, read at the waterline so the timing doesn't run
// on up the beach).
float swashPhase(float arrival, float period, float2 xy, float time, thread float &n) {
    float section, phase, next;
    n = surfClock(xy, time / period - arrival, period, section, phase, next);
    return phase;
}

// Sand `h` metres above the sea: (x) how wet the last few swashes left it (drying over ~20 s),
// (y) the foam they stranded along their highest line (fading over a few seconds), (z) metres
// the current sheet's front stands above it (≤ 0 where it hasn't reached), (w) the swash's
// phase; and through `front` and `runup0` the front's height now and this wave's run-up.
float4 swashSand(constant OceanParams &o, float hs, float arrival, float period, float2 xy, float h, float time,
                 float drift, thread float &front, thread float &runup0) {
    float n;
    float phase = swashPhase(arrival, period, xy, time, n);
    float wet = 0.0, foam = 0.0;
    runup0 = swashRunup(o, hs, period, xy, n, drift);
    float x0 = phase / kSwashShare;
    front = phase < kSwashShare ? runup0 * 4.0 * x0 * (1.0 - x0) : 0.0;
    for (int k = 0; k < 4; k++) {
        float runup = swashRunup(o, hs, period, xy, n - float(k), drift);
        float since = (phase + float(k) - 0.5 * kSwashShare) * period;   // s since its highest
        float reach = runup;
        if (k == 0 && since < 0.0) {
            float x = phase / kSwashShare;
            reach = runup * 4.0 * x * (1.0 - x);
        }
        float covered = 1.0 - smoothstep(reach - 0.04, reach + 0.02, h);
        wet = max(wet, covered * exp(-max(since, 0.0) / 20.0));
        if (since > 0.0) foam = max(foam, (1.0 - smoothstep(0.0, 0.012, abs(h - runup))) * exp(-since / 4.0));
    }
    return float4(wet, foam, front - h, phase);
}

// Value noise in 0…1 with a fully mixed hash (Murmur3's finaliser, high bits): groundNoise's
// keeps low bits that repeat in lattice patterns once thresholded into foam.
float swashNoise(float2 q) {
    float2 i = floor(q), f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    float v[4];
    for (int k = 0; k < 4; k++) {
        int2 c = int2(i) + int2(k & 1, k >> 1);
        uint h = uint(c.x) * 0x85ebca6bu + uint(c.y) * 0xc2b2ae35u + 0x9e3779b9u;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        v[k] = float(h >> 8) / 16777215.0;
    }
    return mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
}

// Cells: 1 at a jittered centre per unit cell, falling to 0 a cell away (Worley, nearest).
float foamCells(float2 q) {
    float2 c0 = floor(q);
    float best = 1e9;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 c = int2(c0) + int2(i, j);
            uint h = uint(c.x) * 0x85ebca6bu + uint(c.y) * 0xc2b2ae35u + 0x27d4eb2fu;
            h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
            float2 jitter = float2(float(h & 0xffffu), float(h >> 16)) / 65535.0;
            float2 d = q - float2(c) - jitter;
            best = min(best, dot(d, d));
        }
    }
    return 1.0 - saturate(sqrt(best));
}

// The swash sheet's look at a sand point (terrainFragment): coverage (x), the leading bore's
// foam (y), the foam and scum carried in the sheet (z), and its thickness in metres (w); and
// through `ripple` a slope for the moving ripples on it. `above` is the front's height over the
// sand (swashSand z), `phase` the swash's phase, `up` the up-beach direction, `footprint` metres
// per pixel.
float4 swashSheet(constant OceanParams &o, float hs, float period, float2 xy, float above, float front, float runup, float phase,
                  float2 up, float footprint, float time, thread float2 &ripple) {
    ripple = 0.0;
    float4 r = 0.0;
    // A crisp edge, antialiased over the pixel (the front is a contour of the sand's height).
    float aa = max(fwidth(above), 1e-4);
    r.x = saturate(above / aa + 0.5) * (phase < kSwashShare ? 1.0 : 0.0);
    if (r.x <= 0.0) return r;
    float x = phase / kSwashShare;
    // How far up the beach the water has been carried (m): the front's distance on the slope.
    // Patterns in coordinates wrapped every 4 km (seams that far apart) and turned off the
    // noise's lattice: at world coordinates tens of kilometres out, noise a few decimetres
    // across loses its precision and shows its cells.
    // The carried distance from the clock alone (the swash's phase without its arrival delay up
    // the beach): the delay shifts the front a metre per metre up, and would squeeze the patterns.
    float ca = time / period + 0.08 * swashNoise(xy / 160.0);
    float xa = fract(ca) / kSwashShare;
    // A smooth run-up (no cusps, no wave-to-wave sections): varying over a few metres, it would
    // shear the patterns into streaks up the beach.
    float smoothRunup = 0.75 * swashRunup(o, hs, period, float2(0.0), 0.0, 0.5) / max(0.55 + 0.45 * swashNoise(float2(0.0)), 0.1);
    float travelled = (xa < 1.0 ? smoothRunup * 4.0 * xa * (1.0 - xa) : 0.0) / kSwashBeta;
    float2 local = xy - floor(xy / 4096.0) * 4096.0;
    float2 q = local - up * travelled;
    q = float2(0.8 * q.x - 0.6 * q.y, 0.6 * q.x + 0.8 * q.y);
    float2 across = float2(-up.y, up.x);
    // The bore: white water along the leading edge, 0.5–2 m wide on the slope, strong early on
    // the uprush and thinning as it slows.
    float uprush = saturate(1.0 - x / 0.5);
    float width = mix(0.5, 2.0, uprush) * kSwashBeta;
    float rag = swashNoise(q / 1.1 + 6.1);
    float bore = (1.0 - smoothstep(0.0, width * (0.6 + 0.8 * rag), above)) * uprush;
    float churn = smoothstep(0.25, 0.65, swashNoise(q / 0.35 + 3.3));
    r.y = saturate(bore * (0.55 + 0.45 * churn));
    // Foam and scum riding in the sheet: patches carried up with it and back, thinner the
    // further behind the bore and the later in the swash; on the backwash streaked downslope.
    float behind = above / max(runup, 0.05);
    float2 sq = float2(dot(q, across), dot(q, up) * mix(1.0, 0.5, smoothstep(0.4, 0.7, x)));
    // Rounded blobs (a cell pattern, not thresholded value noise, whose bilinear lattice shows as
    // dashes), with a little finer lace through them.
    float blobs = smoothstep(0.45, 0.75, foamCells(sq / 1.4 + 1.9));
    float specks = smoothstep(0.55, 0.85, foamCells(sq / 0.45 + 8.8));
    float patches = 0.75 * blobs + 0.35 * specks * (1.0 - blobs);
    float scum = patches * (1.0 - smoothstep(0.1, 0.6, behind)) * mix(0.75, 0.12, saturate(x));
    r.z = mix(scum, 0.15 * mix(0.8, 0.25, x), smoothstep(0.05, 0.3, footprint));
    // Thickness: a few centimetres behind the bore, a film near the edge and on the backwash.
    r.w = min(above, 0.08) * mix(1.0, 0.4, smoothstep(0.5, 1.0, x));
    // Ripples: irregular, two scales, carried with the flow and drifting on their own; gentle
    // (slopes of a few hundredths), so the film stays glassy and its reflection only wavers.
    float2 p = local - up * travelled;
    p = float2(0.6 * p.x - 0.8 * p.y, 0.8 * p.x + 0.6 * p.y);
    float2 slope = 0.0;
    const float2 octave[2] = { float2(0.9, 0.025), float2(0.33, 0.012) };
    for (int k = 0; k < 2; k++) {
        float2 rq = p / octave[k].x + time * float2(0.21, -0.13) * float(k + 1) + float(k) * 17.3;
        float e = 0.2;
        float c0 = swashNoise(rq);
        slope += float2(swashNoise(rq + float2(e, 0.0)) - c0, swashNoise(rq + float2(0.0, e)) - c0) / (e * octave[k].x)
               * octave[k].y * octave[k].x * 4.0;
    }
    ripple = slope * (1.0 - smoothstep(0.03, 0.15, footprint));
    return r;
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
    // Near the camera, the floor's slope (xy) and convexity (z) under the vertex, valid where
    // w is 1 (oceanFloorShape); further out the fragment works them out itself.
    float4 floorShape;
    float4 surf;            // surfSetting: each system's height, smoothed depth, lace
    float4 crests;          // each system's arrival (periods) and crest spacing (m)
    float4 heading;         // each system's direction of travel
    float drift;            // the sets' timing (surfSetDrift)
    float4 shelter;         // OceanPoint's
    float calm;
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
    float4 shelter;         // each cascade's share of the open sea's amplitude (SeaShelter)
    float calm;             // the capillary roughness's share of the open sea's (the wind's lee)
};

// The sea's shelter at xy: each cascade's amplitude factor, and the capillaries' (shore.y).
// Past the domain it opens up to the open sea over a few kilometres.
float4 oceanShelter(constant TerrainParams &t, texture2d<float> shelter, texture2d<float> shore, float2 xy,
                    thread float &calm) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 size = float2(t.domainSize);
    float2 uv = xy / size;
    float outside = length(max(max(-xy, xy - size), 0.0));
    float open = smoothstep(0.0, 4000.0, outside);
    calm = mix(shore.sample(linear, uv, level(0)).y, 1.0, open);
    return mix(shelter.sample(linear, uv, level(0)), float4(1.0), open);
}

// A vertex of a sea node's grid: CDLOD morphing onto the next level's grid towards the
// ring's far edge, as terrainVertex does.
OceanPoint oceanPoint(constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t, TerrainNode node,
                      uint vid, const device int *pages, texture2d_array<ushort, access::read> tiles,
                      texture2d<float> shelter, texture2d<float> shore) {
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
    p.shelter = oceanShelter(t, shelter, shore, xy, p.calm);
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
        d += v.xyz * (w * oceanWaveFactor(o, c, p.depth, p.shelter[c]));
    }
    return d;
}

// ---- Surf (research/ocean.md, step O4, shore foam) ----
// Foam where waves break arriving at the shore: each swell system's crests as refracted and
// shoaled over the bathymetry (WaveRefraction, baked into the wave texture), breaking where a
// wave's height nears the depth; whitewater bands with a sharp front and a fading trail on
// the crests' own clock, crowding and bending as the crests do, broken into sections along the
// shore and grouped into sets; a lace of residual foam at the waterline. Advected bubbles, faded to their mean coverage where a pixel
// spans them.

// Gradient noise in about −1…1 for the foam, without trigonometry: each lattice point's
// gradient straight from its hash (gradientNoise's costs a sine and cosine a corner).
float foamNoise(float2 p, uint seed) {
    float2 i = floor(p), f = p - i;
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float4 g;
    for (int k = 0; k < 4; k++) {
        int2 o = int2(k & 1, k >> 1);
        int2 c = int2(i) + o;
        uint h = uint(c.x) * 0x85ebca6bu + uint(c.y) * 0xc2b2ae35u + seed * 0x9e3779b9u;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        float2 grad = float2(float(h & 0xffffu), float(h >> 16)) * (2.0 / 65535.0) - 1.0;
        g[k] = dot(grad, f - float2(o));
    }
    return 1.6 * mix(mix(g.x, g.y, u.x), mix(g.z, g.w, u.x), u.y);
}

// Two independent foamNoise fields at once (one hash a corner, its bits split between them).
float2 foamNoise2(float2 p, uint seed) {
    float2 i = floor(p), f = p - i;
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float4 gx, gy;
    for (int k = 0; k < 4; k++) {
        int2 o = int2(k & 1, k >> 1);
        int2 c = int2(i) + o;
        uint h = uint(c.x) * 0x85ebca6bu + uint(c.y) * 0xc2b2ae35u + seed * 0x9e3779b9u;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        float4 grad = float4(float(h & 0xffu), float((h >> 8) & 0xffu), float((h >> 16) & 0xffu), float(h >> 24))
                      * (2.0 / 255.0) - 1.0;
        float2 d = f - float2(o);
        gx[k] = dot(grad.xy, d);
        gy[k] = dot(grad.zw, d);
    }
    return 1.6 * float2(mix(mix(gx.x, gx.y, u.x), mix(gx.z, gx.w, u.x), u.y),
                        mix(mix(gy.x, gy.y, u.x), mix(gy.z, gy.w, u.x), u.y));
}

// Foam's rafts at q (in cell units, on a jittered lattice): 1 where there is foam. The cells
// are a power diagram (each centre weighted at random, so cells of mixed sizes). Foam is
// what's left around holes punched in them: `open` 0 (young, dense foam) leaves a blanket with
// a few small round holes in some cells; towards 1 (thinning, old) every cell's hole grows
// until only the walls between them are left, fat where three cells meet (the holes don't
// reach the corners), each wall its own thickness, and more and more of them gone (`gaps`),
// so cells merge into bigger holes. `aa`: the edges' softness (cell units).
float foamRafts(float2 q, float open, float aa) {
    float2 c0 = floor(q);
    float b1 = 1e9, b2 = 1e9;
    float2 p1 = 0.0, p2 = 0.0;
    uint h1 = 0u, h2 = 0u;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 c = int2(c0) + int2(i, j);
            uint h = (uint(c.x) * 1597334677u) ^ (uint(c.y) * 3812015801u);
            h *= 1597334677u;
            h ^= h >> 15;
            float2 p = float2(c) + 0.1 + 0.8 * float2(float(h & 0xffu), float((h >> 8) & 0xffu)) / 255.0;
            float2 d = q - p;
            // Power distance: a weight up to a quarter cell² grows a cell at its neighbours' cost.
            float pd = dot(d, d) - 0.3 * float((h >> 16) & 0xffu) / 255.0;
            if (pd < b1) {
                b2 = b1; p2 = p1; h2 = h1;
                b1 = pd; p1 = p; h1 = h;
            } else if (pd < b2) {
                b2 = pd; p2 = p; h2 = h;
            }
        }
    }
    // Distance to the wall between the two nearest cells (their power bisector).
    float wallDist = (b2 - b1) / max(2.0 * length(p2 - p1), 1e-3);
    // Holes: in a share of the cells when the foam is young, every one as it opens up, each
    // its own size.
    float2 rand1 = float2(float(h1 >> 24), float((h1 >> 4) & 0xffu)) / 255.0;
    float2 rand2 = float2(float(h2 >> 24), float((h2 >> 4) & 0xffu)) / 255.0;
    float share = mix(0.3, 1.05, open);
    float size = mix(0.05, 0.62, open);
    float r1 = rand1.y < share ? size * (0.4 + 0.9 * rand1.x) : -1.0;
    float r2 = rand2.y < share ? size * (0.4 + 0.9 * rand2.x) : -1.0;
    float hole = max(1.0 - smoothstep(r1 - aa, r1 + aa, length(q - p1)),
                     1.0 - smoothstep(r2 - aa, r2 + aa, length(q - p2)));
    // The wall between them: its own thickness, or gone.
    uint pair = (h1 ^ h2) * 2654435761u;
    pair ^= pair >> 13;
    // Past 1 (the oldest foam) the walls go on thinning and breaking until none are left.
    float thick = max(mix(0.09, 0.025, open), 0.0) * (0.4 + 1.2 * float(pair & 0xffu) / 255.0);
    bool gap = float((pair >> 8) & 0xffu) / 255.0 < 0.45 * open + 2.0 * max(open - 1.0, 0.0);
    float wall = gap ? 0.0 : 1.0 - smoothstep(thick, thick + aa, wallDist);
    // The oldest (open 1…1.4): whole cells' remains dissolve, at random, until none are left.
    float gone = step(rand1.x * 0.7 + rand1.y * 0.3, 2.5 * (open - 1.0));
    return max(1.0 - hole, wall) * (1.0 - gone);
}


// What the surf does around a water point, worked out per vertex from the wave texture
// (WaveRefraction: the crests refracted and shoaled over the bathymetry): the trade swell's and
// the north swell's local significant heights (x, y, m; none in pockets of sea a few tens of
// metres inland of the smoothed shoreline), the depth they break against (z: the water's depth
// smoothed over ~150 m, or the local one where shallower) and how much waterline
// lace waves leave (w); through `crests` each system's arrival (in its periods) and the spacing
// of its crests (m), through `heading` the direction each travels, and through `drift` the
// sets' timing (surfSetDrift). Far from any shallows
// (deeper than 8 m, fading in from there) there is none.
float4 surfSetting(constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t, texture2d<float> shore,
                   texture2d<float> waves, float2 xy, float depth, thread float4 &crests, thread float4 &heading,
                   thread float &drift) {
    crests = float4(0.0, 0.0, 1.0, 1.0);
    heading = float4(0.0, 1.0, 0.0, 1.0);
    drift = 0.0;
    if (depth > 8.0 || any(xy < 0.0) || any(xy > float2(t.domainSize))) return float4(0.0, 0.0, 1e4, 0.0);
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float4 setting = shore.sample(linear, xy / t.domainSize, level(0));
    float4 gradient;
    float local;
    float4 w = surfWavesAt(waves, xy, t.domainSize, gradient, local);
    // None in pockets of sea a few tens of metres inland of the smoothed shoreline, fading in
    // from 8 m deep.
    float inland = (1.0 - smoothstep(25.0, 60.0, -setting.r)) * (1.0 - smoothstep(6.0, 8.0, depth));
    float2 g1 = float2(gradient.x, gradient.z), g2 = float2(gradient.y, gradient.w);
    float l1 = length(g1), l2 = length(g2);
    // Spacing between crests: the wavelength over the smoothed depth (held to WaveRefraction's
    // 2 m, as the bake's speeds), L0·√tanh(k0·h) (Eckart). The texture's own gradient (1/|∇τ|,
    // the same by the eikonal) steps from texel to texel, and so would the fronts' widths.
    float2 l0 = 9.81 * o.waves.xy * o.waves.xy / 6.2831853;
    float hc = max(setting.a, 2.0);
    float2 spacing = l0 * sqrt(tanh(6.2831853 * hc / l0));
    crests = float4(w.x, w.y, spacing);
    heading = float4(l1 > 1e-5 ? g1 / l1 : float2(0.0, 1.0), l2 > 1e-5 ? g2 / l2 : float2(0.0, 1.0));
    float hs = length(w.zw);
    float lace = max(saturate(hs / 1.0), 0.35) * inland;
    drift = surfSetDrift(xy);
    // Breaking against the shallower of the smoothed depth and the bake's (~40 m): the bake held
    // the heights to its own, and the shore break and the reef's knolls are where it is less.
    return float4(w.z * inland, w.w * inland, min(setting.a, local), lace);
}

// One system's breakers at xy: a wave train coming in with crests `spacing` metres apart, each
// a solid whitewater roller a couple of wave heights wide with an irregular leading edge, in
// sections that peel a little, trailing clumpy foam that thins over ~35 m behind it. Each wave
// is its set's share of the local significant height `hs` (surfSet) and breaks where that
// nears the breaker depth over the smoothed depth `h`: the sets' big waves further out, the
// small ones only on the inside. Returns its foam cover; through `front`, how much is the roller.
float surfBreakers(constant OceanParams &o, float arrival, float spacing, float hs, float h, float period,
                   float2 xy, float ragged, float footprint, float drift, float weight, thread float &front,
                   thread float &age, thread float &wave) {
    front = 0.0;
    age = 0.0;
    wave = 0.0;
    if (hs < 0.05 || weight <= 0.0) return 0.0;
    // Sections 50–300 m long, re-dealt every wave, peeling: each one's front lags a little
    // towards one end. In world axes (xy is tens of kilometres: rotating it by the waves'
    // direction, which varies across a triangle, would scramble every pattern).
    float section, u0, next;
    float index = surfClock(xy, o.time.x / period - arrival + 2.2 * ragged / spacing, period, section, u0, next);
    // How long ago this wave's front passed (s), and which wave it was: its foam's own clock.
    age = u0 * period;
    wave = index;
    float set = surfSet(o, index, drift);
    // Waves break near H/h ≈ 0.78: a set's biggest (~1.8 Hs) from Hs/h ≈ 0.43, ~0.72 of the bake's
    // saturated Hs/h (o.waves.z), where all of them have broken. Breakers also need waves of some
    // size (over ~0.2 m).
    // The break line is ragged over tens of metres and dealt anew every wave, and the
    // whitewater thickens over some metres from it: a crisp line along the (smooth) heights'
    // contour would cut each wave's trail off in a straight edge across it.
    float depthHs = o.waves.z * max(h, 0.5);
    float2 r = float2(0.866 * xy.x - 0.5 * xy.y, 0.5 * xy.x + 0.866 * xy.y);
    float onset = 0.9 + 0.2 * groundNoise(r / 35.0 + index * 0.71 + 2.9);
    float breaking = smoothstep(0.7, 0.9, onset * hs * (0.6 + 0.5 * set) / depthHs) * smoothstep(0.15, 0.3, hs * set);
    // A solid roller 4–10 m wide.
    float width = clamp(4.0 * hs * set, 4.0, 10.0) / spacing;
    float f = 1.0 - smoothstep(0.6 * width, 1.4 * width, u0);
    // Dense whitewater right behind the roller, thinning into clumps over ~15 m, then a lace.
    float trail = 0.85 * exp(-u0 * spacing / 15.0) + 0.25 * exp(-u0 * spacing / 45.0);
    // Section ends ragged, and tapering over some tens of metres (the foam thinning into
    // clumps there), not cut straight across.
    float on = smoothstep(0.25, 0.5, section + 0.15 * ragged);
    // The leading edge feathered over half a metre (or a pixel), not cut.
    float lead = smoothstep(0.0, max(0.5, footprint) / spacing, u0);
    float bands = on * lead * max(f, trail);
    // A pixel wider than a fraction of the spacing sees the mean whitewater over a set.
    float spread = smoothstep(0.05, 0.25, footprint / spacing);
    float mean = 0.55 * (width + (0.85 * 15.0 + 0.25 * 45.0) / spacing);
    float meanBreaking = smoothstep(0.7, 0.9, 0.95 * hs / depthHs) * smoothstep(0.15, 0.3, 0.75 * hs);
    front = weight * breaking * on * lead * f * (1.0 - spread);
    return weight * mix(breaking * bands, meanBreaking * mean, spread);
}

// Foam cover (0…1) at a water point `depth` metres deep, from its surf setting, with the
// camera's `footprint` (m): both systems' breakers (surfBreakers) and a lace of residual foam at
// the waterline. Returns the cover, and through `roller` how much of it is the front (brighter).
// Each system's share of the surf (0…1): a system under ~30% of the other's height is left out
// (fading in to 50%), its breakers lost among the bigger ones'.
float2 surfWeights(float4 setting) {
    float top = max(max(setting.x, setting.y), 1e-3);
    return smoothstep(0.3, 0.5, setting.xy / top);
}

float surfFoam(constant OceanParams &o, float4 setting, float4 crests, float2 heading, float2 xy, float depth,
               float footprint, float drift, thread float &roller) {
    roller = 0.0;
    // The leading edge, a metre or two ragged.
    float ragged = groundNoise(xy / 2.5 + 4.1) - 0.5;
    float2 weight = surfWeights(setting);
    float f1, f2, age1, age2, wave1, wave2;
    float b1 = surfBreakers(o, crests.x, crests.z, setting.x, setting.z, o.waves.x, xy, ragged, footprint, drift, weight.x, f1,
                            age1, wave1);
    float b2 = surfBreakers(o, crests.y, crests.w, setting.y, setting.z, o.waves.y, xy, ragged, footprint, drift, weight.y, f2,
                            age2, wave2);
    float lace = 1.0 - smoothstep(0.02, 0.12, depth);
    // No breakers over the last few decimetres of water (the shallows' own depth): there the
    // front has become a sheet of swash and the waterline's lace.
    float cover = saturate(max(b1, b2) * smoothstep(0.25, 0.5, depth) + lace * 0.6 * setting.w);
    if (cover <= 0.02) return 0.0;
    roller = max(f1, f2);
    // A pixel this coarse sees only the foam's cover.
    if (footprint >= 0.5) return cover;
    // The foam's own clock: the wave whose whitewater this is (the bigger system's) and how long
    // since its front passed. Each wave leaves a fresh pattern, drifting on with it a little
    // and ageing: its rafts' walls thin and break, its cells open up. (Drifting by the sea's
    // whole time instead magnified the slightest turn of the waves' heading into streaks.)
    bool second = b2 > b1;
    float age = second ? age2 : age1, wave = second ? wave2 : wave1;
    float period = second ? o.waves.y : o.waves.x;
    float old = saturate(age / (0.8 * period));
    float2 q = xy - heading * (0.6 * age) + float2(fract(wave * 0.6180340), fract(wave * 0.3819660)) * 977.0;
    // In world axes, turned off every lattice, and warped.
    float2 r = float2(0.8 * q.x - 0.6 * q.y, 0.6 * q.x + 0.8 * q.y);
    float2 warp = foamNoise2(r / 2.3, 11u);
    // Patches and streaks 5–20 m across: dense foam beside thin (x), and where the rafts are
    // tight and bubbly or wide and open (y; more of them wide as the foam ages).
    float2 patch = foamNoise2(r / 11.0 + 0.6 * warp, 5u);
    float thick = saturate(cover * (0.75 + 0.55 * patch.x));
    // How far the foam has opened up: dense young whitewater a blanket with holes, thinning
    // foam a web of walls, the oldest a few broken filaments.
    // (Past 1 the last walls break away; under the roller, none.)
    float open = clamp(1.25 - 1.3 * thick + 0.35 * old - 1.5 * roller, 0.0, 1.5);
    float wide = smoothstep(-0.06, 0.06, patch.y + 0.5 * old - 0.2);
    float2 rw = r + 0.4 * warp;
    float aaFine = max(0.03, 1.2 * footprint / 0.5), aaWide = max(0.02, 1.2 * footprint / 1.5);
    float tight = wide < 1.0 ? foamRafts(rw / 0.5, open, aaFine) : 0.0;
    float loose = wide > 0.0 ? foamRafts(float2(0.5 * rw.x - 0.87 * rw.y, 0.87 * rw.x + 0.5 * rw.y) / 1.5 + 31.7, open, aaWide) : 0.0;
    // Where tight rafts meet wide ones both sets of holes are punched (a short way into each
    // side), not one faded through the other.
    float rafts = min(mix(1.0, tight, saturate(2.0 - 2.0 * wide)), mix(1.0, loose, saturate(2.0 * wide)));
    float lacy = rafts;
    // A pixel too coarse for the rafts sees their mean.
    lacy = mix(lacy, thick * (1.0 - 0.6 * open), smoothstep(0.12, 0.35, footprint / mix(0.5, 1.5, wide)));
    return mix(lacy, cover, smoothstep(0.08, 0.5, footprint));
}

// ---- Surf waves: the swell's crests shoaling in under the breakers ----
// One wave's height in surfCrest: its set's share of the local significant height, taller
// where its section breaks, faded out where `filter` (m) can't resolve the face.
float surfCrestHeight(constant OceanParams &o, float hs, float h, float n, float section, float spacing, float filter,
                      float drift, float weight) {
    float on = smoothstep(0.3, 0.45, section);
    float set = surfSet(o, n, drift);
    float breaking = smoothstep(0.75, 0.85, hs * (0.6 + 0.5 * set) / (o.waves.z * max(h, 0.5)));
    return weight * o.time.z * 0.45 * hs * set * mix(0.55, 1.0, on) * (0.6 + 0.4 * breaking)
           // Faded where the grid (or a pixel) is too coarse for the face (~3 m).
           * (1.0 - smoothstep(0.08, 0.2, filter / spacing));
}

// One system's crests (surfBreakers' clock): a wave its set's share of the local significant
// height, with a long gentle back behind the crest and a steep face ahead of it, taller where a
// section breaks, faded out where `filter` (m) can't resolve it. Its height (m, about the mean
// sea) at xy at `time`, and through `slope` its gradient (`heading` the way it travels).
float surfCrest(constant OceanParams &o, float arrival, float spacing, float2 heading, float hs, float h, float period,
                float2 xy, float filter, float time, float drift, float weight, thread float2 &slope) {
    slope = 0.0;
    if (hs < 0.05 || weight <= 0.0) return 0.0;
    // The crest the point is behind (surfClock), u0 periods back from it: its long gentle back;
    // and the next one's, v periods ahead: that wave's steep face, at its own height.
    float wave = time / period - arrival;
    float section, u0, next;
    float n = surfClock(xy, wave, period, section, u0, next);
    float v = max(n + 1.0 - 0.08 * next - wave, 0.0);
    float a0 = surfCrestHeight(o, hs, h, n, section, spacing, filter, drift, weight);
    float a1 = surfCrestHeight(o, hs, h, n + 1.0, next, spacing, filter, drift, weight);
    float back = a0 * exp(-u0 / 0.16), face = a1 * exp(-v / 0.05);
    // Less the profile's mean (∫f ≈ 0.16 + 0.05 of the height), so the sea keeps its level,
    // going over from one wave's height to the next's between their crests.
    float gap = max(u0 + v, 1e-3);
    float mean = 0.21 * mix(a0, a1, u0 / gap);
    float dz = (back > face ? -back / 0.16 : face / 0.05) - 0.21 * (a1 - a0) / gap;
    // u0 grows against the heading by 1/spacing per metre (and v shrinks).
    slope = dz * (-heading / spacing);
    return max(back, face) - mean;
}

// Both systems' surf crests (surfCrest), none deeper than ~6 m (the cascades carry the sea
// there) or in pockets of sea inland of the shoreline.
float surfWave(constant OceanParams &o, float4 setting, float4 crests, float4 heading, float drift, float2 xy, float depth,
               float filter, float time, thread float2 &slope) {
    slope = 0.0;
    if (setting.x + setting.y <= 0.0 || o.time.z <= 0.0) return 0.0;
    float2 weight = surfWeights(setting);
    float2 s1, s2;
    float z = surfCrest(o, crests.x, crests.z, heading.xy, setting.x, setting.z, o.waves.x, xy, filter, time, drift, weight.x, s1)
            + surfCrest(o, crests.y, crests.w, heading.zw, setting.y, setting.z, o.waves.y, xy, filter, time, drift, weight.y, s2);
    float fade = (1.0 - smoothstep(4.0, 6.0, depth)) * saturate(setting.w / 0.35);
    slope = (s1 + s2) * fade;
    return z * fade;
}

// The water's displacement at a sea grid point at `time`: the cascades' (from `maps`) and the
// surf's waves, kept off the bottom: a trough can't sink through it (which would show the floor
// dry in the swash), nor a crest heap up over the beach (water a few centimetres deep over the
// sand). Every pass that draws the water works it out the same way.
float3 oceanWater(constant OceanParams &o, texture2d_array<float> maps, OceanPoint p, float4 setting, float4 crests,
                  float4 heading, float drift, float time) {
    float3 d = oceanDisplacement(o, maps, p);
    float2 slope;
    d.z += surfWave(o, setting, crests, heading, drift, p.plane, p.depth, p.filter, time, slope);
    d.z = clamp(d.z, -0.9 * max(p.depth, 0.0), 0.5 * max(p.depth - 0.3, 0.0));
    return d;
}

// The floor's slope and convexity over ~20 m around xy (as terrainVertex's ridge): its
// normal is normalize(-slope, 1); reef darkens where convexity is positive.
float3 oceanFloorShape(constant TerrainParams &t, const device int *pages, texture2d_array<ushort, access::read> tiles,
                       int level, float2 xy, float h) {
    const float r = 20.0;
    float hx0 = levelHeight(t, pages, tiles, level, xy - float2(r, 0.0)), hx1 = levelHeight(t, pages, tiles, level, xy + float2(r, 0.0));
    float hy0 = levelHeight(t, pages, tiles, level, xy - float2(0.0, r)), hy1 = levelHeight(t, pages, tiles, level, xy + float2(0.0, r));
    return float3((hx1 - hx0) / (2.0 * r), (hy1 - hy0) / (2.0 * r), -(hx0 + hx1 + hy0 + hy1 - 4.0 * h) / r);
}

// Grid spacing under which the vertices work out the floor's shape (finer than its 20 m scale).
constant float kFloorShapeSpacing = 8.0;

vertex OceanOut oceanVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                            constant Uniforms &u [[buffer(1)]],
                            constant TerrainParams &t [[buffer(3)]],
                            const device TerrainNode *nodes [[buffer(4)]],
                            const device int *pages [[buffer(5)]],
                            constant OceanParams &o [[buffer(7)]],
                            texture2d_array<ushort, access::read> tiles [[texture(0)]],
                            texture2d_array<float> displacement [[texture(4)]],
                            texture2d<float> shore [[texture(6)]],
                            texture2d<float> waves [[texture(7)]],
                            texture2d<float> shelter [[texture(8)]]) {
    OceanPoint p = oceanPoint(u, o, t, nodes[iid], vid, pages, tiles, shelter, shore);
    OceanOut out;
    out.shelter = p.shelter;
    out.calm = p.calm;
    out.surf = surfSetting(u, o, t, shore, waves, p.plane, p.depth, out.crests, out.heading, out.drift);
    float3 d = oceanWater(o, displacement, p, out.surf, out.crests, out.heading, out.drift, o.time.x);
    out.world = float3(p.plane + d.xy, u.seaLevel + d.z);
    out.position = oceanClip(u, out.world);
    out.plane = p.plane;
    out.depth = p.depth;
    out.crest = d.z;
    out.floorShape = 0.0;
    if (p.filter < kFloorShapeSpacing && p.depth < 120.0 && all(p.plane > 0.0) && all(p.plane < float2(t.domainSize))) {
        out.floorShape = float4(oceanFloorShape(t, pages, tiles, 0, p.plane, u.seaLevel - p.depth), 1.0);
    }
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
                                        texture2d_array<float> previous [[texture(5)]],
                                        texture2d<float> shore [[texture(6)]],
                                        texture2d<float> waves [[texture(7)]],
                                        texture2d<float> shelter [[texture(8)]]) {
    OceanPoint p = oceanPoint(u, o, t, nodes[iid], vid, pages, tiles, shelter, shore);
    float4 crests, heading;
    float drift;
    float4 setting = surfSetting(u, o, t, shore, waves, p.plane, p.depth, crests, heading, drift);
    float3 d = oceanWater(o, displacement, p, setting, crests, heading, drift, o.time.x);
    float3 world = float3(p.plane + d.xy, u.seaLevel + d.z);
    float3 d0 = oceanWater(o, previous, p, setting, crests, heading, drift, o.time.y);
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
// and its light as it leaves the floor (pre-exposure): wet sand, darker on knolls and patch reefs
// (convex over ~20 m, as terrainVertex's ridge), lit by the sun through the surface's shadows.
struct OceanFloor {
    float3 light;
    float below;
    float path;
};

OceanFloor oceanFloor(constant Uniforms &u, constant TerrainParams &t, const device int *pages,
                      texture2d_array<ushort, access::read> tiles, float3 p, float3 dir, float depth,
                      float footprint, float sunLight, float cloudSun, float4 shape) {
    // Heights two levels finer than the pixel: a level as coarse as the pixel blends the land
    // into the floor along a shore and lifts the channels beside it.
    int level = clamp(int(floor(log2(max(footprint / t.cellMetres, 1.0)))) - 2, 0, t.levels - 1);
    float down = max(-dir.z, 0.05);
    float below = max(depth + (p.z - u.seaLevel), 0.0);
    float2 xy = p.xy + dir.xy * (below / down);
    float h = levelHeight(t, pages, tiles, level, xy);
    OceanFloor f;
    f.below = max(p.z - h, 0.0);
    f.path = f.below / down;
    xy = p.xy + dir.xy * f.path;
    // Near the camera the vertices have the floor's shape already (a few metres off along the
    // refracted ray, well inside its 20 m scale).
    float3 s = shape.w > 0.999 ? shape.xyz : oceanFloorShape(t, pages, tiles, level, xy, h);
    float3 n = normalize(float3(-s.xy, 1.0));
    float ridge = s.z;
    float reef = saturate(ridge * 3.0) * (1.0 - smoothstep(8.0, 25.0, u.seaLevel - h));
    // Sand under water is soaked, as the wet beach above the waterline is (terrainFragment).
    float3 albedo = mix(kSand * kWetSand, kReef, reef) * (0.88 + 0.24 * groundNoise(xy / 3.7 + 5.0));
    float3 l = normalize(u.sunDirection);
    f.light = litSurface(u, albedo, n, saturate(dot(n, l)), sunLight, 1.0, 1.0, cloudSun);
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
                              texture2d_array<float> displacement [[texture(11)]],
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
    float shear = 0.0;      // λ∂Dx/∂y, for the Jacobian (whitecaps)
    // Sheltered water is smoother at every scale: its capillaries follow the wind's lee, its
    // cascades their systems' shelter, so the glint narrows and calm water reads glassy.
    float3 variance = oceanRemainder(o) * in.calm;
    uint mask = 0u;
    for (int c = 0; c < 4; c++) {
        float atten = oceanWaveFactor(o, c, in.depth, in.shelter[c]);
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
        if (c < 3) shear += w * atten * displacement.sample(aniso, uv, uint(c)).w;
        float3 filtered = max(float3(mo.x - dv.x * dv.x, mo.y - dv.y * dv.y, 0.0), 0.0);
        filtered.z = mo.z - dv.x * dv.y;
        variance += w * w * filtered + (1.0 - w * w) * whole;
    }
    uint group = simd_or(mask);
    if (simd_is_first() && (atomic_load_explicit(used, memory_order_relaxed) & group) != group) {
        atomic_fetch_or_explicit(used, group, memory_order_relaxed);
    }
    float2 s = slope / max(1.0 + stretch, 0.2);
    // The surf's waves (the vertices lift the water by them, oceanWater).
    float4 heading = float4(length(in.heading.xy) > 1e-3 ? normalize(in.heading.xy) : float2(0.0, 1.0),
                            length(in.heading.zw) > 1e-3 ? normalize(in.heading.zw) : float2(0.0, 1.0));
    if (in.surf.x + in.surf.y > 0.0) {
        float2 surfSlope;
        surfWave(o, in.surf, in.crests, heading, in.drift, in.plane, in.depth, oceanFootprint(u, dist), o.time.x, surfSlope);
        s += surfSlope;
    }
    float3 n = normalize(float3(-s, 1.0));
    float3 l = normalize(u.sunDirection);

    float cloudSun = cloudSunlight(u, cloudShadows, in.world);
    float sunLight = cloudSun;
    if (kRayTraced) sunLight *= tracedSunAt(tracedSun, uint2(in.position.xy));

    // The floor along the refracted view: its light dimmed on the way down to it and back up.
    // Under more than ~120 m of water none shows; past the heightfield there is none.
    float3 floorLight = 0.0;
    float path = 1e5;
    // The water's depth here from the floor's own heights (the vertices' is coarser, and near
    // the shore reads part land), for the surf.
    float waterDepth = in.depth;
    if (in.depth < 120.0) {
        float3 refracted = refract(-v, n, 1.0 / kWaterIndex);
        OceanFloor f = oceanFloor(u, t, pages, tiles, in.world, refracted, in.depth, footprint, sunLight, cloudSun, in.floorShape);
        path = f.path;
        waterDepth = f.below;
        float sunCos = sqrt(max(1.0 - (1.0 - l.z * l.z) / (kWaterIndex * kWaterIndex), 0.05));
        float3 kd = (o.absorption.rgb + 0.5 * o.scattering.rgb + 0.02 * o.scattering.w) / sunCos;
        float3 c = o.absorption.rgb + o.scattering.rgb + o.scattering.w;
        floorLight = f.light * exp(-kd * f.below - c * path);
    }
    float3 color = oceanShade(u, o, environment, n, -n.xy / n.z, variance, v, floorLight, path, sunLight, in.crest);
    // Foam over the shallows (its own white: albedo ~0.8, lit by sun and sky, in the sun's and
    // the clouds' shadows), hiding the glint and the water beneath where it's dense.
    if (in.surf.x + in.surf.y > 0.05 || waterDepth < 0.12) {
        float roller;
        float foam = surfFoam(o, in.surf, in.crests, in.surf.y > in.surf.x ? heading.zw : heading.xy, in.plane, waterDepth,
                              oceanFootprint(u, dist), in.drift, roller);
        if (foam > 0.0) {
            // The roller's churned water brighter (0.88) than the foam it leaves (0.75).
            float albedo = mix(0.75, 0.88, roller);
            float3 white = albedo * (u.sunRadiance * (saturate(l.z) * sunLight) + skyIrradiance(u, float3(0.0, 0.0, 1.0)));
            color = mix(color, white, foam);
        }
    }
    // Whitecaps (research/ocean.md, step O3): where the choppy displacement folds the surface
    // (its Jacobian drops towards zero) a crest is breaking. Patchy, in world axes; out past
    // a few metres of water (the surf has the shallows).
    float jacobian = (1.0 + stretch.x) * (1.0 + stretch.y) - shear * shear;
    float cap = smoothstep(0.8, 0.3, jacobian) * smoothstep(3.0, 8.0, in.depth);
    if (cap > 0.0) {
        float breakup = groundNoise(in.plane / 3.1 + 7.3);
        cap *= smoothstep(0.25, 0.6, breakup + 0.4 * cap);
        float3 white = 0.8 * (u.sunRadiance * (saturate(l.z) * sunLight) + skyIrradiance(u, float3(0.0, 0.0, 1.0)));
        color = mix(color, white, 0.9 * cap);
    }
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}
"""
