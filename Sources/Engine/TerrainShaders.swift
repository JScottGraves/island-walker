import IslandCore

/// Metal source for the first terrain pass, compiled at runtime so `swift build` needs no
/// Xcode-only Metal toolchain step. Moves to .metal files once the app has an Xcode target.
let terrainShaderSource = """
#include <metal_stdlib>
using namespace metal;

""" + Look.metalPrelude + """
struct Vertex {
    packed_float3 position; // metres, world space
    packed_float3 normal;
    packed_float2 light;    // sky visibility, sun visibility (TerrainLighting)
};

struct Uniforms {
    float4x4 viewProjection;
    float4x4 inverseViewProjection; // clip space → camera-relative world directions
    float3 cameraPosition;  // metres, world space
    float seaLevel;
    float3 sunDirection;    // towards the sun
    float maxHeight;
};

struct VOut {
    float4 position [[position]];
    float3 world;
    float3 normal;
    float2 light;
};

vertex VOut terrainVertex(const device Vertex *vertices [[buffer(0)]],
                          constant Uniforms &u [[buffer(1)]],
                          uint vid [[vertex_id]]) {
    Vertex v = vertices[vid];
    VOut o;
    float3 world = float3(v.position);
    // Camera-relative: subtract before projecting so precision stays high near the viewer.
    o.position = u.viewProjection * float4(world - u.cameraPosition, 1.0);
    o.world = world;
    o.normal = float3(v.normal);
    o.light = float2(v.light);
    return o;
}

// Mirrors Look.lit and GroundView.shade; constants come from Look.metalPrelude.
fragment float4 terrainFragment(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    float slope = 1.0 - n.z;
    float h = in.world.z - u.seaLevel;

    float3 albedo;
    if (h < 0.0) {
        albedo = waterAlbedo(-h);
        n = float3(0.0, 0.0, 1.0);
    } else {
        albedo = terrainAlbedo(in.world.xy, h, slope, u.maxHeight);
    }

    float skyVisibility = saturate(in.light.x);
    float sunVisibility = saturate(in.light.y);
    float3 direct = kSun * (saturate(dot(n, normalize(u.sunDirection))) * sunVisibility);
    float3 ambient = kSky * (skyVisibility * (0.5 + 0.5 * n.z));
    float3 bounced = kSun * (kBounce * kSunDirection.z * (0.5 - 0.5 * n.z));
    float3 color = albedo * (direct + ambient + bounced);

    float3 toEye = u.cameraPosition - in.world;
    float dist = length(toEye);
    if (h < 0.0) {
        // Flat water mirrors the sky at the view ray's depression angle.
        float cosView = toEye.z / max(dist, 1e-3);
        color = mix(color, skyColor(cosView), waterReflectance(cosView, dist));
    }
    color = mix(color, kHaze, kHazeMax * (1.0 - exp(-dist / kHazeDistance)));

    color = pbrNeutral(color);
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
    return float4(pbrNeutral(skyColor(dir.z)), 1.0);
}
"""
