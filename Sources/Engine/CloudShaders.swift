/// Metal source for the trade-wind cloud layer (research/clouds.md): a 2.5D cloud layer
/// in the Horizon Zero Dawn / Frostbite style, marched per pixel by the sky and terrain
/// shaders so clouds composite in HDR before tone mapping.
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
// Cumulus extinction per metre at full density (~25 m visibility inside a dense cloud core).
constant float kCloudExtinction = 0.04;

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
    float field = fbm3(float3(drift / 6000.0, 0.37), 3);
    float cumulus = 0.55 * smoothstep(0.52, 0.72, field);
    return saturate(max(orographic, cumulus));
}

// Height above the sea of a point, on a curved Earth centred below the camera's x, y.
float altitude(constant Uniforms &u, float3 p) {
    float2 d = p.xy - u.cameraPosition.xy;
    float r = length(float3(d, p.z + kEarthRadius));
    return r - kEarthRadius;
}

// Cloud density at a world position (0 outside the layer). `detail` adds the small-scale
// erosion; the light march and shadows leave it out.
float cloudDensity(constant Uniforms &u, texture2d<float> cloudMap, float3 p, bool detail) {
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
    float shape = fbm3(q, 3) * profile;
    float d = saturate(remapRange(shape, 1.0 - coverage, 1.0, 0.0, 1.0));
    if (detail && d > 0.0) {
        float erode = fbm3(q * 7.0 + float3(3.1, 7.7, 1.3), 2);
        d = saturate(remapRange(d, 0.35 * erode * (1.2 - hn), 1.0, 0.0, 1.0));
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

    const int steps = 48;
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
        float density = cloudDensity(u, cloudMap, p, true);
        if (density <= 0.0) continue;
        float sigma = density * kCloudExtinction;

        // Optical depth towards the sun over a few exponentially spaced samples.
        float tau = 0.0, tl = 0.0;
        float lightStep = 60.0;
        for (int j = 0; j < 4; j++) {
            tl += lightStep;
            tau += cloudDensity(u, cloudMap, p + l * tl, false) * kCloudExtinction * lightStep;
            lightStep *= 2.5;
        }
        // Sun: dual-lobe phase (silver linings towards the sun, a little back-scatter),
        // plus two Wrenninge octaves for multiple scattering.
        float3 inScatter = 0.0;
        float a = 1.0, b = 1.0, c = 1.0;
        for (int o = 0; o < 3; o++) {
            float phase = mix(henyeyGreenstein(cosTheta, -0.3 * c), henyeyGreenstein(cosTheta, 0.8 * c), 0.7);
            inScatter += sun * (a * phase * exp(-b * tau));
            a *= 0.5; b *= 0.5; c *= 0.5;
        }
        // Ambient: sky above (more of it near the tops), ground below.
        float hn = saturate((altitude(u, p) - u.cloudLayer.x) / (u.cloudLayer.y - u.cloudLayer.x));
        inScatter += 0.5 * skyUp * mix(0.35, 1.0, hn) + 0.5 * ground * (1.0 - 0.6 * hn);

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

// Composites clouds over a colour that lies `maxDistance` away along `dir`, with haze
// (aerial perspective) over the cloud at its own distance.
float3 applyClouds(constant Uniforms &u, texture2d<float> cloudMap, texture2d<float> skyView,
                   float3 color, float3 dir, float maxDistance, float2 pixel) {
    CloudResult c = marchClouds(u, cloudMap, dir, maxDistance, pixel);
    if (c.transmittance >= 0.999) return color;
    float haze = kHazeMax * (1.0 - exp(-c.distance / kHazeDistance));
    float3 cloud = mix(c.light, skyAbove(skyView, dir) * (1.0 - c.transmittance), haze);
    return color * c.transmittance + cloud;
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
        tau += cloudDensity(u, cloudMap, q, false) * kCloudExtinction * dt;
    }
    // Light scattered forward through thin cloud keeps shadows from going fully black.
    return mix(0.15, 1.0, exp(-tau));
}

"""
