/// Metal source for fine terrain detail (plan §4.1, TerrainDetail): the surface below the
/// bundle's 10 m grid, generated on the GPU into clipmap levels around the camera. Spliced
/// into terrainShaderSource after the streamed-height functions it builds on.
let detailShaderSource = """
// One clipmap level: `size`² samples, sample j at world j × spacing (vertices of the
// matching CDLOD level land on samples), stored at texel j mod size so moving the camera
// only rewrites the rows and columns that scroll in.
struct DetailLevel {
    int2 origin;            // first sample index held
    float spacing;          // metres between samples
    int valid;
};

struct DetailParams {
    int levels;             // 0 = finest; level k has spacing cell × 2^(k − levels)
    int size;
    uint seed;
    float seaLevel;
    DetailLevel level[8];
};

struct DetailUpdate {
    int level;
    int oldValid;
    int2 oldOrigin;
};

int2 wrapIndex(int2 j, int n) { return ((j % n) + n) % n; }

bool detailCovers(constant DetailParams &d, int l, float2 xy) {
    DetailLevel L = d.level[l];
    if (L.valid == 0) return false;
    int2 i = int2(floor(xy / L.spacing));
    return all(i >= L.origin) && all(i + 1 < L.origin + d.size);
}

float detailSample(constant DetailParams &d, texture2d_array<float, access::read> detail, int l, float2 xy) {
    float2 s = xy / d.level[l].spacing;
    float2 s0 = floor(s);
    float2 f = s - s0;
    int2 i = int2(s0);
    int n = d.size;
    float h00 = detail.read(uint2(wrapIndex(i, n)), uint(l)).r;
    float h10 = detail.read(uint2(wrapIndex(i + int2(1, 0), n)), uint(l)).r;
    float h01 = detail.read(uint2(wrapIndex(i + int2(0, 1), n)), uint(l)).r;
    float h11 = detail.read(uint2(wrapIndex(i + int2(1, 1), n)), uint(l)).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Surface `l` of the whole chain: detail level l while l < d.levels, else data level
// l − d.levels. A detail level that doesn't hold xy (yet) hands over to the next one up.
float chainHeight(constant TerrainParams &t, constant DetailParams &d,
                  texture2d_array<float, access::read> detail, const device int *pages,
                  texture2d_array<ushort, access::read> tiles, int l, float2 xy) {
    for (; l < d.levels; l++) {
        if (detailCovers(d, l, xy)) return detailSample(d, detail, l, xy);
    }
    return levelHeight(t, pages, tiles, min(l - d.levels, t.levels - 1), xy);
}

// The surface a LOD-`lod` vertex sits on, blended into the next LOD's as it morphs.
float surfaceHeight(constant TerrainParams &t, constant DetailParams &d,
                    texture2d_array<float, access::read> detail, const device int *pages,
                    texture2d_array<ushort, access::read> tiles, int lod, float morph, float2 xy) {
    if (lod >= d.levels) return terrainHeight(t, pages, tiles, lod - d.levels, morph, xy);
    float h = chainHeight(t, d, detail, pages, tiles, lod, xy);
    if (morph > 0.0) h = mix(h, chainHeight(t, d, detail, pages, tiles, lod + 1, xy), morph);
    return h;
}

// ---- Generating the detail ----

uint detailHash(int2 p, uint seed) {
    uint h = uint(p.x) * 0x8da6b343u + uint(p.y) * 0xd8163841u + seed * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float hashUnit(uint h) { return float(h >> 8) * (1.0 / 16777216.0); }

// Gradient noise, roughly in [−1, 1], lattice spacing 1.
float gradientNoise(float2 p, uint seed) {
    float2 i = floor(p);
    float2 f = p - i;
    int2 c = int2(i);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float4 g;
    for (int k = 0; k < 4; k++) {
        int2 o = int2(k & 1, k >> 1);
        float a = hashUnit(detailHash(c + o, seed)) * 6.2831853;
        g[k] = dot(float2(cos(a), sin(a)), f - float2(o));
    }
    return 1.4 * mix(mix(g.x, g.y, u.x), mix(g.z, g.w, u.x), u.y);
}

// How much of an octave of wavelength `lambda` a level can hold: none at or below
// `lambdaMin` (twice its spacing), all of it from twice that.
float band(float lambda, float lambdaMin) { return saturate(lambda / lambdaMin - 1.0); }

// Each octave turned by ~37° so lattice directions don't line up across octaves.
constant float2x2 kOctaveTurn = float2x2(float2(0.8, 0.6), float2(-0.6, 0.8));

// Broken rock and rough lava: ridged noise, sharp crests and blocky hollows, 12 m down.
float rockRelief(float2 xy, float lambdaMin, uint seed) {
    float sum = 0.0, amp = 1.0, lambda = 12.0;
    float2 p = xy;
    for (int o = 0; o < 7 && lambda > lambdaMin; o++) {
        float n = 1.0 - abs(gradientNoise(p / lambda, seed + uint(o)));
        sum += amp * band(lambda, lambdaMin) * (n * n - 0.5);
        amp *= 0.55;
        lambda *= 0.5;
        p = kOctaveTurn * p;
    }
    return sum;
}

// Soil hummocks and, from about 2 m down, clumpy tussocks: soft fBm turning billowy
// (rounded tops, sharp creases between) in the small octaves, 8 m down.
float soilRelief(float2 xy, float lambdaMin, uint seed) {
    float sum = 0.0, amp = 1.0, lambda = 8.0;
    float2 p = xy;
    for (int o = 0; o < 7 && lambda > lambdaMin; o++) {
        float n = gradientNoise(p / lambda, seed + 16u + uint(o));
        if (lambda < 2.5) n = 1.2 * (0.6 - abs(n));
        sum += amp * band(lambda, lambdaMin) * n;
        amp *= lambda < 2.5 ? 0.75 : 0.55;
        lambda *= 0.5;
        p = kOctaveTurn * p;
    }
    return sum;
}

// Rills and small gullies running straight downhill: noise stretched 4:1 along the
// slope, whose zero lines become narrow grooves. Coordinates turn with the slope only in
// eight fixed steps (blending the two nearest), since turning them continuously would
// shear the pattern by kilometres this far from the origin.
float gullies(float2 xy, float2 downhill, float lambdaMin, uint seed) {
    float k = atan2(downhill.y, downhill.x) / (M_PI_F / 8.0);
    float k0 = floor(k);
    float w = k - k0;
    float sum = 0.0;
    for (int j = 0; j < 2; j++) {
        int turn = ((int(k0) + j) % 8 + 8) % 8;
        float a = float(turn) * (M_PI_F / 8.0);
        float2 along = float2(cos(a), sin(a));
        float2 q = float2(dot(xy, along), dot(xy, float2(-along.y, along.x)));
        float groove = 0.0, amp = 1.0, lambda = 6.0;
        for (int o = 0; o < 3 && lambda > lambdaMin; o++) {
            float r = 1.0 - abs(gradientNoise(float2(q.x / (4.0 * lambda), q.y / lambda), seed + 32u + uint(turn * 4 + o)));
            groove -= amp * band(lambda, lambdaMin) * r * r * r * r;
            amp *= 0.5;
            lambda *= 0.5;
        }
        sum += (j == 0 ? 1.0 - w : w) * groove;
    }
    return sum;
}

// Loose boulders, 0.3–1.1 m in radius, mostly small: at most one per 3 m cell, kept where
// a hash falls under `density`. Each has a lumpy outline, a flattened top, a faceted
// surface and its base sunk into the ground; the tallest wins where they touch.
float boulders(float2 xy, float density, float lambdaMin, uint seed) {
    const float cell = 3.0;
    int2 c = int2(floor(xy / cell));
    float h = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 k = c + int2(dx, dy);
            uint r = detailHash(k, seed + 64u);
            if (hashUnit(r) >= density) continue;
            float2 centre = (float2(k) + float2(hashUnit(r * 0x9e3779b9u), hashUnit(r * 0x85ebca6bu))) * cell;
            float u = hashUnit(r * 0xc2b2ae35u);
            float radius = mix(0.3, 1.1, u * u);
            float weight = band(2.0 * radius, lambdaMin);
            if (weight <= 0.0) continue;
            float2 q = (xy - centre) / radius;
            if (length_squared(q) > 2.0) continue;
            uint own = r * 0x27d4eb2fu;
            // Lumpy outline and facets, both at the boulder's own scale.
            float d2 = length_squared(q) * (1.0 + 0.45 * gradientNoise(q * 1.3, own));
            if (d2 >= 1.0) continue;
            float facets = 1.0 - abs(gradientNoise(q * 2.2, own + 1u));
            float dome = min(sqrt(1.0 - d2), 0.75) * (0.8 + 0.3 * facets);
            float stone = radius * (0.75 * dome - 0.15);
            h = max(h, weight * stone);
        }
    }
    return h;
}

// Catmull-Rom weights and their derivatives for the four samples around t ∈ [0, 1).
float4 crWeights(float t) {
    float t2 = t * t, t3 = t2 * t;
    return float4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

float4 crSlopes(float t) {
    float t2 = t * t;
    return float4(-1.5 * t2 + 2.0 * t - 0.5, 4.5 * t2 - 5.0 * t, -4.5 * t2 + 4.0 * t + 0.5, 1.5 * t2 - t);
}

// The baked surface under fine detail: Catmull-Rom through the level-0 samples (smooth,
// and through every sample, unlike the bilinear surface drawn further out), with its
// gradient and the sum of second differences (m; > 0 in hollows, < 0 on crests).
struct BaseSurface {
    float height;
    float2 gradient;
    float curvature;
};

BaseSurface baseSurface(constant TerrainParams &t, const device int *pages,
                        texture2d_array<ushort, access::read> tiles, float2 xy) {
    float2 s = xy / t.cellMetres - 0.5;
    float2 s0 = floor(s);
    float2 f = s - s0;
    int2 i = int2(s0) - 1;
    float p[4][4];
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            p[y][x] = t.heightOffset + t.heightScale * rawHeight(t, pages, tiles, 0, i + int2(x, y));
        }
    }
    float4 wx = crWeights(f.x), wy = crWeights(f.y), dx = crSlopes(f.x), dy = crSlopes(f.y);
    BaseSurface b;
    b.height = 0.0;
    b.gradient = float2(0.0);
    for (int y = 0; y < 4; y++) {
        float4 row = float4(p[y][0], p[y][1], p[y][2], p[y][3]);
        b.height += wy[y] * dot(wx, row);
        b.gradient.x += wy[y] * dot(dx, row);
        b.gradient.y += dy[y] * dot(wx, row);
    }
    b.gradient /= t.cellMetres;
    float c11 = p[1][0] + p[1][2] + p[0][1] + p[2][1] - 4.0 * p[1][1];
    float c12 = p[1][1] + p[1][3] + p[0][2] + p[2][2] - 4.0 * p[1][2];
    float c21 = p[2][0] + p[2][2] + p[1][1] + p[3][1] - 4.0 * p[2][1];
    float c22 = p[2][1] + p[2][3] + p[1][2] + p[3][2] - 4.0 * p[2][2];
    b.curvature = mix(mix(c11, c12, f.x), mix(c21, c22, f.x), f.y);
    return b;
}

// The fine surface at xy, holding wavelengths down to `lambdaMin`. What it adds follows
// the baked island: broken rock on steep ground, crests, hard and dry rock; soft soil
// hummocks elsewhere; rills down hillsides, deepest where it rains most; smooth sediment
// on valley floors and along streams; nothing on beaches or under the sea.
float detailHeight(constant TerrainParams &t, constant DetailParams &d, const device int *pages,
                   texture2d_array<ushort, access::read> tiles,
                   texture2d<float> rainMap, texture2d<float> drainageMap, texture2d<float> hardnessMap,
                   float2 xy, float lambdaMin) {
    BaseSurface b = baseSurface(t, pages, tiles, xy);
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 uv = xy / t.domainSize;
    float rain = rainMap.sample(linear, uv, level(0)).r;          // m a year
    float logArea = drainageMap.sample(linear, uv, level(0)).r;   // log10 m² upstream
    float hardness = hardnessMap.sample(linear, uv, level(0)).r;

    float above = b.height - d.seaLevel;
    float slope = length(b.gradient);
    float steep = smoothstep(0.25, 1.0, slope);
    float crest = smoothstep(0.3, 3.0, -b.curvature);
    float hollow = smoothstep(0.3, 3.0, b.curvature);
    float stream = smoothstep(5.0, 6.5, logArea);
    float valleyFloor = saturate(max(stream, hollow) * (1.0 - steep));
    float wet = smoothstep(1.0, 4.0, rain);
    float rock = saturate(steep + 0.5 * crest + 0.35 * (1.0 - wet) * smoothstep(0.03, 0.2, slope));
    rock *= mix(0.6, 1.0, saturate((hardness - 0.4) / 0.8)) * (1.0 - 0.8 * valleyFloor);
    // Above the beach, or on a sea cliff; fading out a few metres under water.
    float land = max(smoothstep(0.5, 4.0, above), steep) * smoothstep(-3.0, 0.0, above);

    float rockAmp = 0.8 * rock;
    float soilAmp = 0.22 * (1.0 - rock) * (1.0 - 0.5 * valleyFloor);
    float gullyAmp = 0.6 * smoothstep(0.08, 0.4, slope) * (1.0 - smoothstep(1.3, 2.2, slope))
                   * mix(0.5, 1.0, wet) * (1.0 - valleyFloor);
    float boulderDensity = (0.015 + 0.3 * rock) * (1.0 - smoothstep(1.2, 2.0, slope));

    uint seed = d.seed;
    float h = 0.0;
    if (rockAmp > 0.01) h += rockAmp * rockRelief(xy, lambdaMin, seed);
    if (soilAmp > 0.01) h += soilAmp * soilRelief(xy, lambdaMin, seed);
    if (gullyAmp > 0.01 && slope > 1e-3) h += gullyAmp * gullies(xy, b.gradient, lambdaMin, seed);
    if (land > 0.0) h += boulders(xy, boulderDensity * land, lambdaMin, seed);
    return b.height + land * h;
}

// Fills the samples of detail level `u.level` that scrolled into it (all of them when it
// held nothing before). One thread per texel.
kernel void detailUpdate(uint2 gid [[thread_position_in_grid]],
                         constant TerrainParams &t [[buffer(0)]],
                         constant DetailParams &d [[buffer(1)]],
                         constant DetailUpdate &u [[buffer(2)]],
                         const device int *pages [[buffer(3)]],
                         texture2d_array<ushort, access::read> tiles [[texture(0)]],
                         texture2d_array<float, access::write> detail [[texture(1)]],
                         texture2d<float> rainMap [[texture(2)]],
                         texture2d<float> drainageMap [[texture(3)]],
                         texture2d<float> hardnessMap [[texture(4)]]) {
    int n = d.size;
    if (int(gid.x) >= n || int(gid.y) >= n) return;
    DetailLevel L = d.level[u.level];
    int2 j = L.origin + wrapIndex(int2(gid) - L.origin, n);
    if (u.oldValid != 0 && all(j >= u.oldOrigin) && all(j < u.oldOrigin + n)) return;
    float h = detailHeight(t, d, pages, tiles, rainMap, drainageMap, hardnessMap,
                           float2(j) * L.spacing, 2.0 * L.spacing);
    detail.write(float4(h, 0.0, 0.0, 0.0), gid, uint(u.level));
}
"""
