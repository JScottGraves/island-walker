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

struct Vertex {
    packed_float3 position; // metres, world space
    packed_float3 normal;
    packed_float2 light;    // sky visibility, sun visibility at Look.sunDirection (TerrainLighting)
};

struct Uniforms {
    float4x4 viewProjection;
    float4x4 inverseViewProjection; // clip space → camera-relative world directions
    float3 cameraPosition;  // metres, world space
    float seaLevel;
    float3 sunDirection;    // towards the sun
    float maxHeight;
    float3 sunRadiance;     // SunLight.sun
    float exposure;
    float3 skyIrradiance;   // SunLight.sky
    float bounce;
    float3 haze;            // horizon sky colour, tinted for the sun's height
    float pad0;
    float3 zenith;
    float pad1;
};

struct VOut {
    float4 position [[position]];
    float3 world;
    float3 normal;
    float2 light;
    // Horizon elevation (radians) in 16 azimuth sectors, for the baked shadow path.
    float4 horizon0;
    float4 horizon1;
    float4 horizon2;
    float4 horizon3;
};

vertex VOut terrainVertex(const device Vertex *vertices [[buffer(0)]],
                          constant Uniforms &u [[buffer(1)]],
                          const device uchar4 *horizon [[buffer(2)]],
                          uint vid [[vertex_id]]) {
    Vertex v = vertices[vid];
    VOut o;
    float3 world = float3(v.position);
    // Camera-relative: subtract before projecting so precision stays high near the viewer.
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.normal = float3(v.normal);
    o.light = float2(v.light);
    const float toRadians = M_PI_2_F / 255.0;
    o.horizon0 = float4(horizon[vid * 4 + 0]) * toRadians;
    o.horizon1 = float4(horizon[vid * 4 + 1]) * toRadians;
    o.horizon2 = float4(horizon[vid * 4 + 2]) * toRadians;
    o.horizon3 = float4(horizon[vid * 4 + 3]) * toRadians;
    return o;
}

// The sun's disc is 0.53° across; shadows are softened over about a degree, like the
// CPU bake, which also hides the mesh's facets in shadow edges.
constant float kSunConeRadius = 0.01;

// Fraction of 4 rays towards the sun's disc that escape the terrain. A per-pixel rotation
// (interleaved gradient noise) spreads the 4 samples into a smooth penumbra.
float rayTracedSun(float3 p, float3 n, float3 l, float2 pixel, primitive_acceleration_structure terrain) {
    float3 t = normalize(cross(l, abs(l.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
    float3 b = cross(l, t);
    float spin = 6.2831853 * fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
    // Leave the surface along the normal so the ray can't hit its own triangle.
    float3 origin = p + n * 1.0;
    intersector<triangle_data> isect;
    isect.accept_any_intersection(true);
    float visible = 0.0;
    for (int k = 0; k < 4; k++) {
        float r = kSunConeRadius * sqrt((float(k) + 0.5) / 4.0);
        float a = float(k) * 2.3999632 + spin;
        float3 d = normalize(l + (t * cos(a) + b * sin(a)) * r);
        ray shadowRay(origin, d, 0.5, 200000.0);
        if (isect.intersect(shadowRay, terrain).type == intersection_type::none) visible += 0.25;
    }
    return visible;
}

// Sun visibility from the 16-sector horizon map: interpolate the horizon at the sun's
// azimuth and compare elevations, with the same soft edge as the CPU bake.
float horizonSun(VOut in, float3 l) {
    float h[16] = {
        in.horizon0.x, in.horizon0.y, in.horizon0.z, in.horizon0.w,
        in.horizon1.x, in.horizon1.y, in.horizon1.z, in.horizon1.w,
        in.horizon2.x, in.horizon2.y, in.horizon2.z, in.horizon2.w,
        in.horizon3.x, in.horizon3.y, in.horizon3.z, in.horizon3.w };
    float azimuth = atan2(l.y, l.x);
    if (azimuth < 0.0) azimuth += 6.2831853;
    float f = azimuth / (6.2831853 / 16.0) - 0.5;
    float f0 = floor(f);
    int a = (int(f0) + 16) % 16;
    int b = (a + 1) % 16;
    float horizonAngle = mix(h[a], h[b], f - f0);
    float margin = asin(clamp(l.z, -1.0, 1.0)) - horizonAngle;
    return smoothstep(0.0, 1.0, saturate((margin + 0.01) / 0.02));
}

// Mirrors Look.lit and GroundView.shade; constants come from Look.metalPrelude, the sun
// and sky from SunLight through the uniforms.
fragment float4 terrainFragment(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                primitive_acceleration_structure terrain [[buffer(2), function_constant(kRayTraced)]]) {
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
    float sunVisibility = 0.0;
    if (nDotL > 0.0 && l.z > -0.01) {
        if (kRayTraced) {
            sunVisibility = rayTracedSun(surface, n, l, in.position.xy, terrain);
        } else {
            sunVisibility = horizonSun(in, l);
        }
    }
    float3 direct = u.sunRadiance * (nDotL * sunVisibility);
    float3 ambient = u.skyIrradiance * (skyVisibility * (0.5 + 0.5 * n.z));
    float3 bounced = u.sunRadiance * (u.bounce * max(l.z, 0.0) * (0.5 - 0.5 * n.z));
    float3 color = albedo * (direct + ambient + bounced);

    float3 toEye = u.cameraPosition - in.world;
    float dist = length(toEye);
    if (h < 0.0) {
        // Flat water mirrors the sky at the view ray's depression angle.
        float cosView = toEye.z / max(dist, 1e-3);
        color = mix(color, skyColor(cosView, u.haze, u.zenith), waterReflectance(cosView, dist));
    }
    color = mix(color, u.haze, kHazeMax * (1.0 - exp(-dist / kHazeDistance)));

    color = pbrNeutral(color * u.exposure);
    // The render target is an _srgb format, which applies the sRGB encoding on write.
    return float4(color, 1.0);
}

// Sky: one full-screen triangle drawn before the terrain, coloured by the elevation of
// each pixel's view ray (mirrors GroundView's sky).
struct SkyOut {
    float4 position [[position]];
    float2 ndc;
};

vertex SkyOut skyVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    SkyOut o;
    o.position = float4(p, 0.0, 1.0);
    o.ndc = p;
    return o;
}

fragment float4 skyFragment(SkyOut in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    float4 far = u.inverseViewProjection * float4(in.ndc, 1.0, 1.0);
    float3 dir = normalize(far.xyz / far.w);
    return float4(pbrNeutral(skyColor(dir.z, u.haze, u.zenith) * u.exposure), 1.0);
}
"""
