/// Metal source for the first terrain pass, compiled at runtime so `swift build` needs no
/// Xcode-only Metal toolchain step. Moves to .metal files once the app has an Xcode target.
let terrainShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float3 position; // metres, world space
    packed_float3 normal;
};

struct Uniforms {
    float4x4 viewProjection;
    float3 cameraPosition;  // metres, world space
    float seaLevel;
    float3 sunDirection;    // towards the sun
    float maxHeight;
};

struct VOut {
    float4 position [[position]];
    float3 world;
    float3 normal;
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
    return o;
}

fragment float4 terrainFragment(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    float slope = 1.0 - n.z;
    float h = in.world.z - u.seaLevel;

    float3 albedo;
    if (h < 0.0) {
        float d = saturate(-h / 400.0);
        albedo = mix(float3(0.20, 0.55, 0.60), float3(0.02, 0.10, 0.25), d);
    } else {
        float t = saturate(h / max(u.maxHeight, 1.0));
        float3 grass = mix(float3(0.20, 0.36, 0.14), float3(0.30, 0.38, 0.20), t);
        float3 rock = float3(0.32, 0.27, 0.24);                  // weathered basalt
        float3 soil = float3(0.45, 0.25, 0.15);                  // red lateritic soil
        albedo = mix(grass, soil, smoothstep(0.15, 0.30, slope));
        albedo = mix(albedo, rock, smoothstep(0.35, 0.55, slope));
        // Beach: narrow band just above the waterline on gentle slopes (matches GroundView).
        float land = max(smoothstep(1.0, 3.0, h), smoothstep(0.1, 0.25, slope));
        albedo = mix(float3(0.76, 0.70, 0.55), albedo, land);
    }

    float3 sunColor = float3(1.0, 0.95, 0.85) * 2.2;
    float3 skyColor = float3(0.45, 0.60, 0.85) * 0.5;
    float diffuse = saturate(dot(n, normalize(u.sunDirection)));
    float3 color = albedo * (sunColor * diffuse + skyColor * (0.5 + 0.5 * n.z));

    // Simple aerial perspective until the physically based sky lands.
    float dist = length(in.world - u.cameraPosition);
    float fog = 0.85 * (1.0 - exp(-dist / 60000.0));
    color = mix(color, float3(0.62, 0.72, 0.85), fog);

    color = color * (1.0 + color / 6.25) / (1.0 + color);   // extended Reinhard, white = 2.5
    // The render target is an _srgb format, which applies the sRGB encoding on write.
    return float4(color, 1.0);
}
"""
