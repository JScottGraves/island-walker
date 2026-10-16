/// Metal source for the trade-wind cloud layer (research/clouds.md): a 2.5D cloud layer
/// in the Horizon Zero Dawn / Frostbite style. A compute pass marches it at half
/// resolution up to the terrain depth, a second pass blends each frame into a history
/// (temporal accumulation, which turns the per-pixel step jitter into smooth cloud), and
/// the sky and terrain shaders composite the result in HDR before tone mapping.
///
/// Where clouds are comes from islandgen's baked orographic coverage (standing over the
/// windward slopes and crest) plus drifting ocean cumulus; a height profile between the
/// cloud base and the trade-wind inversion shapes them; value-noise fBm, advected with the
/// trade wind, erodes the detail. Lighting: sun through a short light march with dual-lobe
/// Henyey–Greenstein and Wrenninge multiple-scattering octaves, sky and ground ambient,
/// Hillaire's energy-conserving step integration. Inserted after `Uniforms` and the sky
/// helpers in the terrain shader.
let cloudShaderSource = """

constant float kEarthRadius = 6360000.0;
// Cumulus extinction per metre at full density (~40 m visibility inside a dense cloud core).
constant float kCloudExtinction = 0.025;
// March steps longer than these skip the detail noise and then the finest shape octave:
// features smaller than the step only alias into stipple.
constant float kCloudDetailStep = 250.0;
constant float kCloudShapeStep = 700.0;

float cloudHash(float3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise(float3 x) {
    float3 i = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(cloudHash(i + float3(0, 0, 0)), cloudHash(i + float3(1, 0, 0)), f.x),
                   mix(cloudHash(i + float3(0, 1, 0)), cloudHash(i + float3(1, 1, 0)), f.x), f.y),
               mix(mix(cloudHash(i + float3(0, 0, 1)), cloudHash(i + float3(1, 0, 1)), f.x),
                   mix(cloudHash(i + float3(0, 1, 1)), cloudHash(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

float fbm3(float3 p, int octaves) {
    float sum = 0.0, amplitude = 0.5, norm = 0.0;
    for (int k = 0; k < octaves; k++) {
        sum += amplitude * valueNoise(p);
        norm += amplitude;
        p = p * 2.03 + float3(17.1, 9.3, 4.7);
        amplitude *= 0.5;
    }
    return sum / norm;
}

float remapRange(float v, float lo, float hi, float newLo, float newHi) {
    return newLo + (v - lo) / max(hi - lo, 1e-4) * (newHi - newLo);
}

// Cloud cover at a ground position: baked orographic cloud, which stays put over the
// terrain, or drifting ocean cumulus, whichever is more.
float cloudCoverage(constant Uniforms &u, texture2d<float> cloudMap, float2 xy) {
    constexpr sampler s(filter::linear, address::clamp_to_zero);
    float clock = u.cloudLayer.z;
    float orographic = cloudMap.sample(s, xy / u.cloudLayer.w).r * u.cloudWind.z;
    float2 drift = xy - u.cloudWind.xy * clock;
    float field = fbm3(float3(drift / 6000.0, 0.37), 2);
    float cumulus = 0.55 * smoothstep(0.52, 0.72, field);
    // Even the cap cloud on the crest has breaks in it.
    return min(max(orographic, cumulus), 0.88);
}

// Height above the sea of a point, on a curved Earth centred below the camera's x, y.
float altitude(constant Uniforms &u, float3 p) {
    float2 d = p.xy - u.cameraPosition.xy;
    float r = length(float3(d, p.z + kEarthRadius));
    return r - kEarthRadius;
}

// Cloud density at a world position (0 outside the layer), with noise detail matched to
// `stepLength`, the length of the march step that samples it. The detail erosion fades
// out over a range of step lengths rather than at one, which would draw a seam.
float cloudDensity(constant Uniforms &u, texture2d<float> cloudMap, float3 p, float stepLength) {
    float base = u.cloudLayer.x, top = u.cloudLayer.y;
    float h = altitude(u, p);
    if (h < base || h > top) return 0.0;
    float coverage = cloudCoverage(u, cloudMap, p.xy);
    if (coverage < 0.02) return 0.0;
    // Heavier cover grows taller towards the inversion; tops flatten under it.
    float localTop = base + (top - base) * mix(0.35, 1.0, coverage);
    float hn = (h - base) / max(localTop - base, 1.0);
    if (hn > 1.0) return 0.0;
    float profile = saturate(hn / 0.12) * saturate((1.0 - hn) / 0.3);
    // The air streams through the standing cloud: only the texture moves with the wind.
    float3 q = float3(p.xy - u.cloudWind.xy * u.cloudLayer.z, h) / 1400.0;
    float shape = fbm3(q, stepLength < kCloudShapeStep ? 3 : 2) * profile;
    float d = saturate(remapRange(shape, 1.0 - coverage, 1.0, 0.0, 1.0));
    float detail = 1.0 - smoothstep(0.5 * kCloudDetailStep, kCloudDetailStep, stepLength);
    if (detail > 0.0 && d > 0.0) {
        float erode = fbm3(q * 7.0 + float3(3.1, 7.7, 1.3), 2);
        d = saturate(remapRange(d, 0.35 * detail * erode * (1.2 - hn), 1.0, 0.0, 1.0));
    }
    return d;
}

float henyeyGreenstein(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * M_PI_F * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

// Distances along a ray where it is inside the spherical shell between radii r0 and r1
// around the Earth's centre (the cloud layer): returns (enter, exit), exit < enter if none.
float2 shellSegment(float3 origin, float3 dir, float r0, float r1) {
    float b = dot(origin, dir);
    float c0 = dot(origin, origin) - r0 * r0;
    float c1 = dot(origin, origin) - r1 * r1;
    float d1 = b * b - c1;
    if (d1 < 0.0) return float2(1.0, 0.0);
    float outerFar = -b + sqrt(d1), outerNear = -b - sqrt(d1);
    float d0 = b * b - c0;
    float innerNear = 1e12, innerFar = -1e12;
    if (d0 >= 0.0) { innerNear = -b - sqrt(d0); innerFar = -b + sqrt(d0); }
    float r = length(origin);
    if (r < r0) {
        // Below the base: from the base up to the top.
        return float2(innerFar, outerFar);
    } else if (r < r1) {
        // Inside the layer: to the top, or down to the base if the ray hits it first.
        float exitAt = (d0 >= 0.0 && innerNear > 0.0) ? innerNear : outerFar;
        return float2(0.0, exitAt);
    } else {
        // Above: from the top down to the base (or out again).
        if (outerNear < 0.0) return float2(1.0, 0.0);
        float exitAt = (d0 >= 0.0 && innerNear > 0.0) ? innerNear : outerFar;
        return float2(outerNear, exitAt);
    }
}

struct CloudResult {
    float3 light;         // in-scattered radiance, before haze
    float transmittance;
    float distance;       // transmittance-weighted distance to the cloud
};

// Marches the cloud layer from the camera along `dir` up to `maxDistance` (the terrain,
// or far away for the sky). Radiance is in the renderer's display units (π · k · L).
CloudResult marchClouds(constant Uniforms &u, texture2d<float> cloudMap, float3 dir, float maxDistance, float2 pixel) {
    CloudResult result;
    result.light = 0.0;
    result.transmittance = 1.0;
    result.distance = maxDistance;
    float3 origin = float3(0.0, 0.0, u.cameraPosition.z + kEarthRadius);
    float2 seg = shellSegment(origin, dir, kEarthRadius + u.cloudLayer.x, kEarthRadius + u.cloudLayer.y);
    float start = max(seg.x, 0.0);
    float end = min(min(seg.y, maxDistance), 120000.0);
    if (end <= start) return result;

    float3 l = normalize(u.sunDirection);
    float cosTheta = dot(dir, l);
    float3 sun = u.sunRadiance * M_PI_F;
    float3 skyUp = skyIrradiance(u, float3(0.0, 0.0, 1.0));
    // What the ground below bounces up: green land or dark sea, lit by sun and sky.
    float3 ground = float3(0.05, 0.07, 0.06) * (u.sunRadiance * max(l.z, 0.0) + skyUp);
    // Light inside a cloud has scattered many times: most of the sky's blue is washed out.
    float skyLum = dot(skyUp, float3(0.2126, 0.7152, 0.0722));
    float3 skyNeutral = mix(skyUp, float3(skyLum), 0.6);

    end = min(end, start + 60000.0);
    if (end <= start) return result;
    // Few steps: the jitter moves every frame and the history averages it out.
    const int steps = maxDistance > 1e8 ? 32 : 24;
    float span = end - start;
    // Short steps near the camera (fog around a walker inside the cloud), longer far out.
    float jitter = fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
    float weightedDistance = 0.0, weight = 0.0;
    float tPrevious = start;
    for (int k = 0; k < steps; k++) {
        float f = (float(k) + jitter) / float(steps);
        float t = start + span * f * f;
        float dt = max(t - tPrevious, 1.0);
        tPrevious = t;
        float3 p = u.cameraPosition + dir * t;
        float density = cloudDensity(u, cloudMap, p, dt);
        if (density <= 0.0) continue;
        // Thin the cloud right around the camera so a walker inside it still sees the
        // ground at their feet: fog, not a grey wall.
        float sigma = density * kCloudExtinction * mix(0.3, 1.0, saturate(t / 400.0));

        // Optical depth towards the sun over a few coarse samples (~550 m in total).
        float tau = 0.0, tl = 0.0;
        float lightStep = 50.0;
        for (int j = 0; j < 3; j++) {
            tl += lightStep;
            tau += cloudDensity(u, cloudMap, p + l * (tl - 0.5 * lightStep), max(lightStep, kCloudDetailStep)) * kCloudExtinction * lightStep;
            lightStep *= 3.0;
        }
        // Sun: dual-lobe phase (silver linings towards the sun, a little back-scatter),
        // plus Wrenninge octaves standing in for multiple scattering.
        float3 inScatter = 0.0;
        float a = 1.0, b = 1.0, c = 1.0;
        for (int o = 0; o < 4; o++) {
            float phase = mix(henyeyGreenstein(cosTheta, -0.3 * c), henyeyGreenstein(cosTheta, 0.8 * c), 0.7);
            inScatter += sun * (a * phase * exp(-b * tau));
            a *= 0.5; b *= 0.5; c *= 0.5;
        }
        // Ambient: sky above (more of it near the tops), ground below.
        float hn = saturate((altitude(u, p) - u.cloudLayer.x) / (u.cloudLayer.y - u.cloudLayer.x));
        inScatter += 0.5 * skyNeutral * mix(0.5, 1.0, hn) + 0.5 * ground * (1.0 - 0.6 * hn);

        float stepT = exp(-sigma * dt);
        // Energy-conserving integration over the step (cloud albedo ~1: scattering = extinction).
        result.light += result.transmittance * inScatter * (1.0 - stepT);
        float absorbed = result.transmittance * (1.0 - stepT);
        weightedDistance += absorbed * t;
        weight += absorbed;
        result.transmittance *= stepT;
        if (result.transmittance < 0.01) break;
    }
    if (weight > 0.0) result.distance = weightedDistance / weight;
    return result;
}

// Where the view along `dir` ends with no terrain in the way: the sea below the horizon,
// else far away.
float skyViewEnd(constant Uniforms &u, float3 dir) {
    return dir.z < 0.0 ? max(u.cameraPosition.z - u.seaLevel, 1.0) / max(-dir.z, 1e-5) : 1e9;
}

// The full-resolution pixel that half-resolution cloud texel `g` is marched through
// (centre at 2g + 1.5 in pixel coordinates), and the view direction through it.
uint2 cloudPixel(constant Uniforms &u, uint2 g) {
    return min(g * 2 + 1, uint2(u.frame.xy) - 1);
}

float3 pixelDirection(constant Uniforms &u, float2 pixelCentre, float depth, thread float &distance) {
    float2 ndc = float2(pixelCentre.x / u.frame.x * 2.0 - 1.0, 1.0 - pixelCentre.y / u.frame.y * 2.0);
    float4 farPoint = u.inverseViewProjection * float4(ndc, 1.0, 1.0);
    float3 dir = normalize(farPoint.xyz / farPoint.w);
    if (depth >= 1.0) {
        distance = skyViewEnd(u, dir);
    } else {
        float4 q = u.inverseViewProjection * float4(ndc, depth, 1.0);
        distance = length(q.xyz / q.w);
    }
    return dir;
}

// Pass 1: march the clouds for each half-resolution texel up to the terrain (or sea, or
// sky) behind it. Writes hazed cloud light and transmittance, plus (distance to what lies
// behind, distance to the cloud) for upsampling and reprojection.
kernel void cloudMarch(constant Uniforms &u [[buffer(0)]],
                       depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<float> skyView [[texture(1)]],
                       texture2d<float> cloudMap [[texture(2)]],
                       texture2d<float, access::write> cloudOut [[texture(3)]],
                       texture2d<float, access::write> distanceOut [[texture(4)]],
                       uint2 g [[thread_position_in_grid]]) {
    if (g.x >= cloudOut.get_width() || g.y >= cloudOut.get_height()) return;
    uint2 pixel = cloudPixel(u, g);
    float behind;
    float3 dir = pixelDirection(u, float2(pixel) + 0.5, depth.read(pixel), behind);
    // Interleaved gradient noise shifted every frame (Jimenez 2014): over the history's
    // ~10 frames each texel sees evenly spread step offsets.
    float2 jitter = float2(g) + 5.588238 * fmod(u.frame.z, 64.0);
    CloudResult c = marchClouds(u, cloudMap, dir, behind, jitter);
    float3 light = 0.0;
    if (c.transmittance < 0.999) {
        // Aerial perspective over the cloud, at the cloud's own distance.
        float haze = kHazeMax * (1.0 - exp(-c.distance / kHazeDistance));
        light = mix(c.light, skyAbove(skyView, dir) * (1.0 - c.transmittance), haze);
    }
    cloudOut.write(float4(light, c.transmittance), g);
    distanceOut.write(float4(behind, c.transmittance < 0.999 ? c.distance : behind, 0.0, 0.0), g);
}

// Pass 2: blend this frame's clouds into the history. The history is reprojected through
// the cloud's own distance, kept only where it saw the same surface behind, and clamped to
// this frame's neighbourhood so moving clouds and a moving sun don't leave ghosts.
kernel void cloudResolve(constant Uniforms &u [[buffer(0)]],
                         texture2d<float> current [[texture(0)]],
                         texture2d<float> currentDistance [[texture(1)]],
                         texture2d<float> history [[texture(2)]],
                         texture2d<float> historyDistance [[texture(3)]],
                         texture2d<float, access::write> resolved [[texture(4)]],
                         texture2d<float, access::write> resolvedDistance [[texture(5)]],
                         uint2 g [[thread_position_in_grid]]) {
    int2 size = int2(current.get_width(), current.get_height());
    if (int(g.x) >= size.x || int(g.y) >= size.y) return;
    float4 c = current.read(g);
    float2 d = currentDistance.read(g).xy;
    float4 lo = c, hi = c;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float4 s = current.read(uint2(clamp(int2(g) + int2(x, y), int2(0), size - 1)));
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    float4 result = c;
    if (u.frame.w > 0.5) {
        uint2 pixel = cloudPixel(u, g);
        float unused;
        float3 dir = pixelDirection(u, float2(pixel) + 0.5, 1.0, unused);
        float3 world = u.cameraPosition + dir * min(d.y, 200000.0);
        float4 clip = u.previousViewProjection * float4(world - u.previousCamera.xyz, 1.0);
        if (clip.w > 0.0) {
            float2 ndc = clip.xy / clip.w;
            float2 previousPixel = float2((0.5 + 0.5 * ndc.x) * u.frame.x, (0.5 - 0.5 * ndc.y) * u.frame.y);
            float2 uv = ((previousPixel - 1.5) * 0.5 + 0.5) / float2(size);
            if (all(uv >= 0.0) && all(uv <= 1.0)) {
                constexpr sampler s(filter::linear, address::clamp_to_edge);
                float previousBehind = historyDistance.sample(s, uv).x;
                if (abs(previousBehind - d.x) < 0.05 * d.x) {
                    float4 h = clamp(history.sample(s, uv), lo, hi);
                    result = mix(h, c, 0.1);
                }
            }
        }
    }
    resolved.write(result, g);
    resolvedDistance.write(float4(d, 0.0, 0.0), g);
}

// Composites the resolved half-resolution clouds over a colour `distance` away, at full
// resolution: a bilinear upsample that favours texels which saw the same distance behind,
// so clouds don't bleed across the terrain's silhouette.
float3 compositeClouds(texture2d<float> clouds, texture2d<float> cloudDistance,
                       float3 color, float2 pixelCentre, float distance) {
    float2 h = (pixelCentre - 1.5) * 0.5;
    float2 g0 = floor(h);
    float2 f = h - g0;
    int2 size = int2(clouds.get_width(), clouds.get_height());
    float4 sum = 0.0;
    float weight = 0.0;
    float4 closest = 0.0;
    float closestError = 1e30;
    for (int k = 0; k < 4; k++) {
        int2 o = int2(k & 1, k >> 1);
        uint2 q = uint2(clamp(int2(g0) + o, int2(0), size - 1));
        float4 s = clouds.read(q);
        float error = abs(cloudDistance.read(q).x - distance) / max(distance, 1.0);
        float bilinear = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        float w = max(bilinear, 1e-3) * exp(-(error / 0.05) * (error / 0.05));
        sum += s * w;
        weight += w;
        if (error < closestError) { closestError = error; closest = s; }
    }
    float4 c = weight > 1e-4 ? sum / weight : closest;
    return color * c.a + c.rgb;
}

// Sunlight reaching a terrain point through the cloud layer (cloud shadows), from a
// coarse march along the sun direction.
float cloudShadow(constant Uniforms &u, texture2d<float> cloudMap, float3 p) {
    float3 l = normalize(u.sunDirection);
    if (l.z <= 0.02) return 1.0;
    float base = u.cloudLayer.x, top = u.cloudLayer.y;
    float h0 = max(p.z, 0.0);
    if (h0 >= top) return 1.0;
    float t0 = max(base - h0, 0.0) / l.z, t1 = (top - h0) / l.z;
    float tau = 0.0;
    const int samples = 6;
    float dt = (t1 - t0) / float(samples);
    for (int k = 0; k < samples; k++) {
        float3 q = p + l * (t0 + (float(k) + 0.5) * dt);
        tau += cloudDensity(u, cloudMap, q, dt) * kCloudExtinction * dt;
    }
    // Light scattered forward through the cloud keeps its shadow from going fully black.
    return mix(0.3, 1.0, exp(-tau));
}

"""
