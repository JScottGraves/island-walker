/// Metal source for the lips of plunging breakers (SurfLip; research/waves.md, item 4), after
/// Thürey, Müller-Fischer, Schirm & Gross, "Real-time Breaking Waves for Shallow Water
/// Simulations" (2007): along the breaking crest a wave line, traced each row period from the
/// surf's own clock (OceanShaders: surfClock, surfProgress), emits a row of particles thrown
/// forward from the crest's tip faster than the wave runs; they fly ballistically and are joined
/// row to row and along the line into a thin sheet of water, the lip, curling down ahead of the
/// face. Where a particle hits the water it dies, throwing spray and leaving foam for the
/// whitewater (lipFoamAt). Appended after oceanShaderSource.
let surfLipShaderSource = """
// ---- Surf lips (SurfLip) ----
struct LipParams {
    float4 axis;            // the wave line's heading (xy: the way the waves run) and across it (zw)
    float4 frame;           // now, last frame (s, the sea's wrapped clock), row period (s), sea level
    float4 march;           // camera xy; where the trace starts along the heading (m) and how far it runs
    int4 slots;             // first slot's index across (1 m each), slots, rows, system (0 trade, 1 north)
    int4 rows;              // newest row's ring index, rows emitted this frame, first of them's ring index, reset (1)
    float4 emit;            // first emission's time (wrapped), particle lifetime (s), spray lifetime (s), foam fade
};

struct LipParticle {
    float4 start;           // where it left the crest (xyz), when (w; NaN: none)
    float4 velocity;        // its velocity then (xyz), when it hit the water (w; 1e30 while flying)
    float4 tag;             // slot (as int bits), wave, plunge, progress at emission
};

struct SprayParticle {
    float4 start;           // where (xyz), when (w; NaN: none)
    float4 velocity;        // velocity (xyz), size (w, m)
};

constant float kLipSlot = 2.0;      // metres between the wave line's points
constant float kLipStep = 8.0;      // the trace's step along the heading (m; crests are ≥ 25 m apart)
constant float kLipReach = 300.0;   // no lip further than this from the camera (m)
constant float3 kGravity = float3(0.0, 0.0, -9.81);
constant uint kSpraySides = 6u;
constant uint kSprayCapacity = 8192u;     // SurfLip.sprayCapacity

float lipHash(uint a, uint b) {
    uint h = a * 0x85ebca6bu + b * 0xc2b2ae35u + 0x27d4eb2fu;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return float(h >> 8) / 16777215.0;
}

// The surf's setting at a point of the wave line, worked out from the textures alone (the
// bake's depth for the water's).
float4 lipSetting(constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t, texture2d<float> shore,
                  texture2d<float> waves, texture2d<float> slopes, float2 xy, thread float4 &crests, thread float4 &heading, thread float &drift,
                  thread float2 &beta, thread float &depth) {
    float4 g;
    surfWavesAt(waves, xy, t.domainSize, g, depth);
    return surfSetting(u, o, t, shore, waves, slopes, xy, depth, crests, heading, drift, beta);
}

// The water's surface height (m above the sea) at xy and `time` from the surf's waves alone (the
// cascades' chop is left out), and its horizontal shift there.
float lipWater(constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t, texture2d<float> shore,
               texture2d<float> waves, texture2d<float> slopes, float2 xy, float time, thread float2 &shift) {
    float4 crests, heading;
    float drift, depth;
    float2 beta, slope;
    float stretch;
    float4 setting = lipSetting(u, o, t, shore, waves, slopes, xy, crests, heading, drift, beta, depth);
    shift = 0.0;
    if (setting.x + setting.y <= 0.0) return 0.0;
    heading = float4(length(heading.xy) > 1e-3 ? normalize(heading.xy) : float2(0.0, 1.0),
                     length(heading.zw) > 1e-3 ? normalize(heading.zw) : float2(0.0, 1.0));
    float z = surfWave(o, setting, crests, heading, beta, drift, xy, depth, 0.3, time, shift, slope, stretch);
    return clamp(z, -0.9 * max(depth, 0.0), 0.8 * max(max(depth, setting.z) - 0.3, 0.0));
}

// Slot `slot`'s ray across the surf at `time`: marching in along the heading, the first crest
// whose wave is throwing its lip (break progress 0.3…0.85, plunging enough, its section
// breaking, big enough). Its tip (the crest's top, thrown forward), the lip's launch velocity,
// and the wave's number, type and progress.
bool lipTrace(constant LipParams &l, constant Uniforms &u, constant OceanParams &o, constant TerrainParams &t,
              texture2d<float> shore, texture2d<float> waves, texture2d<float> slopes, int slot, float time, thread float3 &tip,
              thread float3 &velocity, thread float4 &tag) {
    float2 h = l.axis.xy, across = l.axis.zw * (float(slot) * kLipSlot);
    int system = l.slots.w;
    float period = system == 0 ? o.waves.x : o.waves.y;
    int steps = int(l.march.w / kLipStep);
    float last = 0.0;
    bool have = false;
    for (int k = 0; k <= steps; k++) {
        float s = l.march.z + float(k) * kLipStep;
        float2 xy = across + h * s;
        float4 g;
        float depth;
        float4 w4 = surfWavesAt(waves, xy, t.domainSize, g, depth);
        if (depth > 9.0 || depth < 0.05 || length(xy - l.march.xy) > kLipReach) {
            have = false;
            continue;
        }
        float w = time / period - (system == 0 ? w4.x : w4.y);
        // The waves run in along the heading: their phase falls; a crest where it crosses a whole
        // number (less its peel, surfClock's 0.08 of its section).
        if (have && floor(last) > floor(w)) {
            float n = floor(last);
            float lo = s - 2.0 * kLipStep, hi = s;
            for (int it = 0; it < 10; it++) {
                float mid = 0.5 * (lo + hi);
                float2 q = across + h * mid;
                float4 gq;
                float dq;
                float4 wq = surfWavesAt(waves, q, t.domainSize, gq, dq);
                float f = time / period - (system == 0 ? wq.x : wq.y) + 0.08 * surfSection(q, n, period) - n;
                if (f > 0.0) lo = mid; else hi = mid;
            }
            float2 c = across + h * (0.5 * (lo + hi));
            float4 crests, heading;
            float drift, cdepth;
            float2 beta;
            float4 setting = lipSetting(u, o, t, shore, waves, slopes, c, crests, heading, drift, beta, cdepth);
            float hs = system == 0 ? setting.x : setting.y;
            float weight = surfWeights(setting)[system];
            float b = system == 0 ? beta.x : beta.y;
            float set = surfSet(o, n, drift);
            float hb = surfBreakDepth(o, hs, set, surfOnset(c, n));
            float plunge = surfPlunge(o, hb, b, period);
            float p = surfProgress(hb, setting.z, b, plunge, 0.0);
            float section = surfSection(c, n, period);
            if (p > 0.3 && p < 0.85 && plunge > 0.15 && weight > 0.5 && hs * set > 0.25 && section > 0.4) {
                float2 dir = system == 0 ? heading.xy : heading.zw;
                dir = length(dir) > 1e-3 ? normalize(dir) : h;
                float2 shift;
                float z = lipWater(u, o, t, shore, waves, slopes, c, time, shift);
                tip = float3(c + shift, l.frame.w + z);
                // Thrown faster than the wave runs (Thürey et al.: u_s = (1 + p_v g (H − H_i)) u_l),
                // the more the more it plunges and the further on it is; a little upward at first.
                float speed = sqrt(9.81 * (max(setting.z, 0.3) + max(z, 0.0)));
                float throwSpeed = speed * (1.0 + 0.5 * plunge * smoothstep(0.3, 0.6, p));
                velocity = float3(dir * throwSpeed, 0.6 * plunge * (1.0 - smoothstep(0.35, 0.7, p)));
                tag = float4(as_type<float>(slot), n, plunge, p);
                return true;
            }
        }
        last = w;
        have = true;
    }
    return false;
}

// This frame's rows: one per row period since the last, each slot tracing its crest at that
// moment (so the rows keep the clock whatever the frame rate).
kernel void lipEmit(constant LipParams &l [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
                    constant OceanParams &o [[buffer(2)]], constant TerrainParams &t [[buffer(3)]],
                    device LipParticle *particles [[buffer(4)]],
                    texture2d<float> shore [[texture(0)]], texture2d<float> waves [[texture(1)]],
                    texture2d<float> slopes [[texture(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    int slots = l.slots.y, rows = l.slots.z;
    if (int(g.x) >= slots || int(g.y) >= l.rows.y) return;
    int slot = l.slots.x + int(g.x);
    int column = ((slot % slots) + slots) % slots;
    int row = (l.rows.z + int(g.y)) % rows;
    float te = l.emit.x + float(g.y) * l.frame.z;
    LipParticle q;
    float3 tip, velocity;
    float4 tag;
    if (lipTrace(l, u, o, t, shore, waves, slopes, slot, te, tip, velocity, tag)) {
        q.start = float4(tip, te);
        q.velocity = float4(velocity, 1e30);
        q.tag = tag;
    } else {
        q.start = float4(0.0, 0.0, 0.0, NAN);
        q.velocity = float4(0.0, 0.0, 0.0, 1e30);
        q.tag = float4(as_type<float>(slot), 0.0, 0.0, 0.0);
    }
    particles[row * slots + column] = q;
}

// A lip particle's position at `time` (ballistic from its launch).
float3 lipPosition(LipParticle q, float time) {
    float tau = time - q.start.w;
    return q.start.xyz + q.velocity.xyz * tau + 0.5 * kGravity * tau * tau;
}

// Every flying particle against the water: one under the surf's surface has hit it: it dies
// there, throwing spray (some of them) and leaving foam on the impact foam grid.
kernel void lipCollide(constant LipParams &l [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
                       constant OceanParams &o [[buffer(2)]], constant TerrainParams &t [[buffer(3)]],
                       device LipParticle *particles [[buffer(4)]], device SprayParticle *spray [[buffer(5)]],
                       device atomic_uint *sprayHead [[buffer(6)]], device atomic_uint *foam [[buffer(7)]],
                       texture2d<float> shore [[texture(0)]], texture2d<float> waves [[texture(1)]],
                    texture2d<float> slopes [[texture(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (int(i) >= l.slots.y * l.slots.z) return;
    LipParticle q = particles[i];
    float now = l.frame.x;
    if (isnan(q.start.w) || q.velocity.w < 1e29) return;
    float tau = now - q.start.w;
    if (tau < 0.0) return;
    float3 p = lipPosition(q, now);
    float2 shift;
    float water = l.frame.w + lipWater(u, o, t, shore, waves, slopes, p.xy, now, shift);
    bool old = tau > l.emit.y;
    if (p.z > water && !old) return;
    q.velocity.w = now;
    particles[i] = q;
    if (old || l.rows.w != 0) return;
    // Spray from about every third impact, thrown up and on with the jet.
    uint seed = i * 7919u + uint(now * 240.0);
    if (lipHash(seed, 1u) < 0.25) {
        uint k = atomic_fetch_add_explicit(sprayHead, 1u, memory_order_relaxed) % kSprayCapacity;
        float2 along = normalize(q.velocity.xy + 1e-4);
        float2 side = float2(-along.y, along.x);
        float r1 = lipHash(seed, 2u), r2 = lipHash(seed, 3u), r3 = lipHash(seed, 4u);
        float3 v = float3(along * length(q.velocity.xy) * mix(0.3, 0.8, r1) + side * (r2 - 0.5) * 2.0,
                          mix(1.0, 3.5, r3) * (0.5 + 0.5 * q.tag.z));
        spray[k].start = float4(p.xy, water, now);
        spray[k].velocity = float4(v, mix(0.12, 0.35, r2));
    }
    // Foam on the grid (o.breaking: cell metres, side), 1/4096 fixed point.
    float cell = o.breaking.z;
    int side = int(o.breaking.w);
    int2 c = int2(floor(p.xy / cell));
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 w = (((c + int2(dx, dy)) % side) + side) % side;
            uint amount = (dx == 0 && dy == 0) ? 1400u : 500u;
            atomic_fetch_add_explicit(&foam[w.y * side + w.x], amount, memory_order_relaxed);
        }
    }
}

// The impact foam fades (e-folding over a few seconds): `l.emit.w` its factor this frame.
kernel void lipFoamFade(constant LipParams &l [[buffer(0)]], device atomic_uint *foam [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
    uint v = atomic_load_explicit(&foam[i], memory_order_relaxed);
    if (v == 0u) return;
    atomic_store_explicit(&foam[i], uint(min(float(v), 6000.0) * l.emit.w), memory_order_relaxed);
}

// ---- The lip's sheet: rows (by age) × slots (along the line), two-sided ----
struct LipOut {
    float4 position [[position]];
    float3 world;
    float3 normal;
    float age;              // seconds since launch
    float plunge;
};

// The particle of the sheet's row `age` (0: the newest) at column `j` of this frame's slots.
LipParticle lipAt(constant LipParams &l, const device LipParticle *particles, int j, int age) {
    int slots = l.slots.y, rows = l.slots.z;
    int slot = l.slots.x + j;
    int column = ((slot % slots) + slots) % slots;
    int row = ((l.rows.x - age) % rows + rows) % rows;
    return particles[row * slots + column];
}

// Whether a particle is part of this frame's sheet: launched, at slot `slot`, from wave `wave`,
// and flying (or just landed: its last position held a moment, so the sheet meets the water).
bool lipLive(constant LipParams &l, LipParticle q, int slot, float wave) {
    if (isnan(q.start.w) || as_type<int>(q.tag.x) != slot || q.tag.y != wave) return false;
    float now = l.frame.x;
    return now >= q.start.w && (q.velocity.w > 1e29 || now - q.velocity.w < 0.05);
}

float3 lipLivePosition(constant LipParams &l, LipParticle q) {
    return lipPosition(q, min(l.frame.x, q.velocity.w));
}

// Whether quad (column j, row age k) of the sheet joins four flying particles of one wave line,
// and their positions now.
bool lipQuad(constant LipParams &l, const device LipParticle *particles, int j, int k, thread float3 *p,
             thread LipParticle *c) {
    c[0] = lipAt(l, particles, j, k);
    c[1] = lipAt(l, particles, j + 1, k);
    c[2] = lipAt(l, particles, j, k + 1);
    c[3] = lipAt(l, particles, j + 1, k + 1);
    float wave = c[0].tag.y;
    if (!(lipLive(l, c[0], l.slots.x + j, wave) && lipLive(l, c[1], l.slots.x + j + 1, wave)
          && lipLive(l, c[2], l.slots.x + j, wave) && lipLive(l, c[3], l.slots.x + j + 1, wave))) return false;
    for (int m = 0; m < 4; m++) p[m] = lipLivePosition(l, c[m]);
    // No bridges between wave lines that are not neighbours.
    float along = 2.0 * kLipSlot;
    return distance(p[0], p[1]) < along && distance(p[2], p[3]) < along && distance(p[0], p[2]) < 3.0
           && distance(p[1], p[3]) < 3.0;
}

// The sheet's live quads, listed for an indirect draw: each adds its six vertices to the draw's
// count (args[0], cleared before) and its index to `quads`.
kernel void lipQuads(constant LipParams &l [[buffer(0)]], const device LipParticle *particles [[buffer(4)]],
                     device atomic_uint *args [[buffer(8)]], device uint *quads [[buffer(9)]],
                     uint i [[thread_position_in_grid]]) {
    int columns = l.slots.y - 1;
    if (int(i) >= columns * (l.slots.z - 1)) return;
    float3 p[4];
    LipParticle c[4];
    if (!lipQuad(l, particles, int(i) % columns, int(i) / columns, p, c)) return;
    uint at = atomic_fetch_add_explicit(&args[0], 6u, memory_order_relaxed);
    quads[at / 6u] = i;
}

vertex LipOut lipVertex(uint vid [[vertex_id]], constant Uniforms &u [[buffer(1)]],
                        constant LipParams &l [[buffer(2)]], const device LipParticle *particles [[buffer(3)]],
                        const device uint *quads [[buffer(5)]]) {
    LipOut out;
    int columns = l.slots.y - 1;
    int quad = int(quads[vid / 6u]), corner = int(vid % 6u);
    int j = quad % columns, k = quad / columns;
    const int2 offsets[6] = { int2(0, 0), int2(1, 0), int2(1, 1), int2(0, 0), int2(1, 1), int2(0, 1) };
    float3 p[4];
    LipParticle c[4];
    out.position = float4(0.0, 0.0, 2.0, 1.0);
    out.world = 0.0;
    out.normal = float3(0.0, 0.0, 1.0);
    out.age = 0.0;
    out.plunge = 0.0;
    if (!lipQuad(l, particles, j, k, p, c)) return out;
    int2 o2 = offsets[corner];
    int m = o2.x + 2 * o2.y;
    float3 along = (p[1] - p[0]) + (p[3] - p[2]);
    float3 down = (p[2] - p[0]) + (p[3] - p[1]);
    out.world = p[m];
    out.normal = cross(along, down);
    out.age = l.frame.x - c[m].start.w;
    out.plunge = c[m].tag.z;
    out.position = u.viewProjection * float4(p[m] - u.cameraPosition, 1.0);
    return out;
}

// Thin water: the sky's reflection, sunlight through it (green-blue, brightest with the sun
// behind it), whitening as it aerates on the way down.
fragment float4 lipFragment(LipOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                            constant OceanParams &o [[buffer(7)]],
                            texture2d<float> skyView [[texture(0)]],
                            texture2d<float> clouds [[texture(2)]],
                            texture2d<float> cloudDistance [[texture(3)]],
                            texture2d<float> environment [[texture(8)]]) {
    float3 v = normalize(u.cameraPosition - in.world);
    float3 n = normalize(in.normal);
    if (dot(n, v) < 0.0) n = -n;
    float3 l = normalize(u.sunDirection);
    float nv = max(dot(n, v), 1e-3);
    float fresnel = waterFresnel(nv);
    float3 r = reflect(-v, n);
    r.z = max(r.z, 0.0);
    float3 reflection = oceanEnvironment(environment, r, 0.1);
    float3 sky = skyIrradiance(u, float3(0.0, 0.0, 1.0));
    float behind = pow(saturate(dot(-v, l)), 3.0);
    float3 tint = float3(0.12, 0.62, 0.52);
    float sunUp = smoothstep(0.0, 0.1, l.z);
    float3 through = u.sunRadiance * tint * ((0.06 + 0.5 * behind) * sunUp) + sky * tint * 0.3;
    float3 color = fresnel * reflection + (1.0 - fresnel) * through;
    float aerated = smoothstep(0.1, 0.55, in.age);
    float3 white = 0.8 * (u.sunRadiance * (saturate(l.z) * 0.9) + sky);
    color = mix(color, white, 0.8 * aerated);
    float alpha = mix(0.6, 0.92, aerated) * smoothstep(0.0, 0.02, in.age + 0.02);
    float4 c = finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
    return float4(c.rgb, alpha);
}

// ---- Spray: soft white billboards, rising, slowing and spreading as they fade ----
struct SprayOut {
    float4 position [[position]];
    float3 world;
    float2 corner;
    float fade;
};

vertex SprayOut sprayVertex(uint vid [[vertex_id]], constant Uniforms &u [[buffer(1)]],
                            constant LipParams &l [[buffer(2)]], const device SprayParticle *spray [[buffer(4)]]) {
    SprayOut out;
    SprayParticle s = spray[vid / kSpraySides];
    float tau = l.frame.x - s.start.w;
    float life = l.emit.z;
    out.position = float4(0.0, 0.0, 2.0, 1.0);
    out.world = 0.0;
    out.corner = 0.0;
    out.fade = 0.0;
    if (!(tau >= 0.0 && tau < life)) return out;
    // Air drag slows the droplets (over ~0.6 s) and the mist sinks slowly.
    float k = 1.6;
    float slow = (1.0 - exp(-k * tau)) / k;
    float3 p = s.start.xyz + s.velocity.xyz * slow + float3(0.0, 0.0, -2.5) * tau * tau;
    float size = s.velocity.w * (1.0 + 1.2 * tau);
    float3 toEye = u.cameraPosition - p;
    float3 f = normalize(-toEye);
    float3 right = normalize(cross(f, float3(0.0, 0.0, 1.0)) + 1e-5);
    float3 up = cross(right, f);
    const float2 corners[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float2 c = corners[vid % kSpraySides];
    float3 world = p + (right * c.x + up * c.y) * size;
    out.world = world;
    out.corner = c;
    out.fade = pow(1.0 - tau / life, 1.5);
    out.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    return out;
}

fragment float4 sprayFragment(SprayOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                              texture2d<float> skyView [[texture(0)]],
                              texture2d<float> clouds [[texture(2)]],
                              texture2d<float> cloudDistance [[texture(3)]]) {
    float d2 = dot(in.corner, in.corner);
    float alpha = exp(-3.0 * d2) * (1.0 - smoothstep(0.7, 1.0, d2)) * in.fade * 0.4;
    if (alpha < 0.004) discard_fragment();
    float3 l = normalize(u.sunDirection);
    float3 white = 0.85 * (u.sunRadiance * saturate(l.z) + skyIrradiance(u, float3(0.0, 0.0, 1.0)));
    float4 c = finishSurface(u, skyView, clouds, cloudDistance, white, in.world, in.position.xy);
    return float4(c.rgb, alpha);
}
"""
