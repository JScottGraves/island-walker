/// Buildings (BuildingRenderer): drawn after the terrain in the depth prepass and the
/// shading pass, their shadows traced by sunShadow against their own acceleration structure.
/// Spliced in after the terrain's shaders (Uniforms, litSurface, finishSurface, octEncode,
/// the structure map).
let buildingShaderSource = """
// ---- Buildings (BuildingRenderer) ----
struct BuildingVertex {
    float4 position;    // world xyz (m), material code
    float4 normal;      // world normal
    float4 pattern;     // the material's pattern coordinates (m), height above the floor (m)
};

struct BuildingOut {
    float4 position [[position]];
    float3 world;
    float3 relative;    // world − camera, for derivatives that keep their precision
    float3 normal;
    float3 pattern;
    int material [[flat]];
};

constant int kBlock = 0, kRoofIron = 1, kCladding = 2, kCastConcrete = 3, kSteel = 4;

vertex BuildingOut buildingVertex(uint vid [[vertex_id]],
                                  constant Uniforms &u [[buffer(1)]],
                                  const device BuildingVertex *vertices [[buffer(2)]]) {
    BuildingVertex v = vertices[vid];
    BuildingOut o;
    float3 relative = v.position.xyz - u.cameraPosition;
    o.position = u.viewProjection * float4(relative, 1.0);
    o.world = v.position.xyz;
    o.relative = relative;
    o.normal = v.normal.xyz;
    o.pattern = v.pattern.xyz;
    o.material = int(v.position.w);
    return o;
}

// Thin parts (roof sheets, cladding, door leaves) are seen from both sides: the normal
// turned towards the eye.
float3 buildingFacingNormal(BuildingOut in) {
    float3 n = normalize(in.normal);
    return dot(n, -in.relative) < 0.0 ? -n : n;
}

// Prepass with ray-traced shadows: the surface for sunShadow (neither tree nor impostor).
fragment uint4 buildingSurface(BuildingOut in [[stage_in]]) {
    float3 n = buildingFacingNormal(in);
    float3 p = in.world;
    return uint4(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z),
                 pack_float_to_snorm2x16(octEncode(n)) & ~(kTreeSurface | kImpostorSurface));
}

float buildingHash(float2 c) {
    uint h = uint(int(c.x)) * 0x8da6b343u ^ uint(int(c.y)) * 0xd8163841u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h >> 8) * (1.0 / 16777216.0);
}

// Two octaves of the structure map's value noise, 0…1.
float buildingNoise(float2 q) {
    return 0.65 * structureNoise(q) + 0.35 * structureNoise(q * 2.7 + 5.3);
}

struct BuildingLook {
    float3 albedo;
    float3 normal;
    float shine;        // share of bare metal that glints in the sun
};

// Galvanised iron gone to rust: patchy, heaviest along the laps and the lower edges, in
// streaks running down the sheet (`down` the pattern axis rust runs along).
float buildingRust(float2 q, float2 down, float edges) {
    float2 streak = float2(dot(q, down) * 0.15, dot(q, float2(down.y, -down.x)) * 3.0);
    float patch = buildingNoise(q / 1.9 + 1.7);
    float runs = structureNoise(streak + 9.1);
    return smoothstep(0.45, 0.8, 0.55 * patch + 0.35 * runs + 0.35 * edges);
}

float3 buildingRustColour(float2 q) {
    return mix(float3(0.16, 0.065, 0.03), float3(0.36, 0.15, 0.055), structureNoise(q / 0.7 + 3.3));
}

// Corrugations: the surface's tilt across them (a sine 19 mm deep every 76 mm), faded out
// before a pixel spans one.
float buildingCorrugation(float x, float footprint) {
    float pitch = 0.076;
    float fade = 1.0 - smoothstep(0.25, 0.7, footprint / pitch);
    return 0.0095 * 2.0 * M_PI_F / pitch * cos(x / pitch * 2.0 * M_PI_F) * fade;
}

BuildingLook buildingLook(int material, float3 pattern, float3 n, float3 tangent, float3 bitangent,
                          float footprint, float eaves, bool underside) {
    BuildingLook look;
    look.normal = n;
    look.shine = 0.0;
    float2 q = pattern.xy;
    float h = pattern.z;
    if (material == kBlock) {
        // 400 × 200 mm block in running bond, 10 mm mortar joints.
        float row = floor(q.y / 0.2);
        float shift = fmod(abs(row), 2.0) * 0.2;
        float a = q.x + shift;
        float col = floor(a / 0.4);
        float2 f = float2(a - col * 0.4, q.y - row * 0.2);
        float joint = min(min(f.x, 0.4 - f.x), min(f.y, 0.2 - f.y));
        float w = max(footprint, 0.002);
        float mortar = 1.0 - smoothstep(0.005 - w, 0.005 + w, joint);
        mortar = mix(mortar, 0.07, smoothstep(0.02, 0.06, footprint));
        float tint = 0.9 + 0.2 * buildingHash(float2(col, row));
        float3 block = float3(0.34, 0.33, 0.30) * tint * (0.92 + 0.16 * structureNoise(q / 0.09));
        // Whitewash, flaked off in patches, most of it gone near the ground.
        float wash = smoothstep(0.42, 0.62, buildingNoise(q / 1.6 + 2.2) + 0.25 * smoothstep(0.5, 2.5, h)
                                + 0.1 * structureNoise(q / 0.25));
        float3 albedo = mix(block, float3(0.60, 0.58, 0.52), 0.85 * wash);
        albedo = mix(albedo, float3(0.25, 0.24, 0.22), mortar);
        // Grime run down from the eaves and the window sills, and rising damp with black
        // algae along the foot.
        float runs = structureNoise(float2(q.x * 2.2, q.y * 0.18) + 4.1);
        float fromTop = smoothstep(eaves - 2.5, eaves, h) + 0.5 * structureNoise(float2(q.x * 0.7, 3.0));
        albedo *= 1.0 - 0.45 * saturate(fromTop * smoothstep(0.35, 0.75, runs));
        float damp = 1.0 - smoothstep(0.15, 0.7 + 0.6 * buildingNoise(q / 1.3 + 7.0), h);
        albedo = mix(albedo, float3(0.075, 0.08, 0.055), 0.75 * damp);
        // Recessed joints: the block faces' edges tilt into them, close up.
        if (footprint < 0.02) {
            float2 g = float2(f.x < 0.2 ? -1.0 : 1.0, f.y < 0.1 ? -1.0 : 1.0);
            float2 dist = float2(min(f.x, 0.4 - f.x), min(f.y, 0.2 - f.y));
            float2 tilt = g * (1.0 - smoothstep(0.004, 0.012, dist)) * 0.6 * (1.0 - smoothstep(0.008, 0.02, footprint));
            look.normal = normalize(n + tangent * tilt.x + bitangent * tilt.y);
        }
        look.albedo = albedo;
    } else if (material == kRoofIron || material == kCladding) {
        // Roof sheets: corrugations run down the arch (the profile varies along the depth,
        // pattern y), sheet laps every 0.9 m along the depth and 3 m down the arch. Cladding
        // and door leaves: vertical corrugations (the profile varies along pattern x).
        bool roof = material == kRoofIron;
        float across = roof ? q.y : q.x;
        float tilt = buildingCorrugation(across, footprint);
        look.normal = normalize(n - (roof ? bitangent : tangent) * tilt);
        float lap = roof ? min(abs(fract(q.y / 0.9 + 0.5) - 0.5) * 0.9, abs(fract(q.x / 3.0 + 0.5) - 0.5) * 3.0)
                         : abs(fract(q.x / 0.85 + 0.5) - 0.5) * 0.85;
        float edges = 1.0 - smoothstep(0.0, 0.25, lap);
        float2 down = roof ? float2(1.0, 0.0) : float2(0.0, 1.0);
        if (!roof) edges = max(edges, 1.0 - smoothstep(0.0, 0.9, h));   // the leaves' feet
        float rust = buildingRust(q, down, edges);
        float3 bare = roof ? float3(0.40, 0.41, 0.41) : float3(0.20, 0.27, 0.24);   // galvanised; faded green paint
        bare *= 0.88 + 0.24 * buildingNoise(q / 2.3 + 8.8);
        if (underside) {
            bare *= 0.75;
            rust *= 0.6;
        }
        look.albedo = mix(bare, buildingRustColour(q), rust);
        look.shine = (roof ? 0.35 : 0.08) * (1.0 - rust);
    } else if (material == kCastConcrete) {
        float stain = buildingNoise(q / 1.1 + 6.6);
        look.albedo = float3(0.43, 0.42, 0.39) * (0.8 + 0.3 * stain);
        look.albedo = mix(look.albedo, float3(0.12, 0.12, 0.10), 0.5 * smoothstep(0.55, 0.8, structureNoise(float2(q.x * 3.0, q.y * 0.3))));
    } else {
        look.albedo = mix(float3(0.07, 0.045, 0.03), float3(0.18, 0.08, 0.035), structureNoise(q / 0.3));
    }
    return look;
}

[[early_fragment_tests]]
fragment float4 buildingFragment(BuildingOut in [[stage_in]], constant Uniforms &u [[buffer(1)]],
                                 texture2d<float> skyView [[texture(0)]],
                                 texture2d<float> clouds [[texture(2)]],
                                 texture2d<float> cloudDistance [[texture(3)]],
                                 texture2d<float> cloudShadows [[texture(4)]],
                                 texture2d<float, access::read> tracedSun [[texture(5), function_constant(kRayTraced)]],
                                 texture2d<float> skyVisibility [[texture(8)]],
                                 texture2d_array<float, access::read> horizons [[texture(9)]],
                                 const device uchar *structures [[buffer(20)]]) {
    float3 n = buildingFacingNormal(in);
    // The pattern's axes on the surface, from the screen derivatives (camera-relative, so
    // they keep their precision far from the origin).
    float3 dp1 = dfdx(in.relative), dp2 = dfdy(in.relative);
    float2 duv1 = dfdx(in.pattern.xy), duv2 = dfdy(in.pattern.xy);
    float3 perp2 = cross(dp2, n), perp1 = cross(n, dp1);
    float3 tangent = perp2 * duv1.x + perp1 * duv2.x;
    float3 bitangent = perp2 * duv1.y + perp1 * duv2.y;
    float scale = rsqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-12));
    tangent *= scale;
    bitangent *= scale;
    float footprint = max(length(duv1), length(duv2));

    // Which building, for its eaves (the grime runs down from there).
    float eaves = 4.8;
    if (structureBuildingCount(structures) > 0u) eaves = structureBuilding(structures, 0u).b.w;
    BuildingLook look = buildingLook(in.material, in.pattern, n, tangent, bitangent, footprint, eaves, n.z < -0.2);
    float3 nm = look.normal;

    float3 l = normalize(u.sunDirection);
    float nDotL = saturate(dot(nm, l));
    float cloudSun = cloudSunlight(u, cloudShadows, in.world);
    float sunVisibility = 0.0;
    if (l.z > -0.01 && cloudSun > 0.01 && dot(n, l) > 0.0) {
        if (kRayTraced) {
            sunVisibility = tracedSunAt(tracedSun, uint2(in.position.xy));
        } else {
            int side = int(horizons.get_width());
            uint2 cell = uint2(clamp(int2(in.world.xy / u.cloudLayer.w * float(side)), int2(0), int2(side - 1)));
            float2 bounds;
            float margin = horizonMargin(horizons.read(cell, 0) * M_PI_2_F, horizons.read(cell, 1) * M_PI_2_F,
                                         horizons.read(cell, 2) * M_PI_2_F, horizons.read(cell, 3) * M_PI_2_F, l, bounds);
            sunVisibility = smoothstep(0.0, 1.0, saturate((margin + 0.01) / 0.02));
        }
    }
    sunVisibility *= cloudSun;
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float sky = saturate(skyVisibility.sample(linear, in.world.xy / u.cloudLayer.w, level(0)).r);
    // Indoors (a step off the surface along its facing normal is under the roof): only what
    // the door lets in. Outdoors, darker into the corner with the ground.
    float occlusion = buildingShelter(structures, in.world + n * 0.2);
    occlusion *= mix(0.55, 1.0, smoothstep(0.0, 1.2, in.pattern.z));
    float3 color = litSurface(u, look.albedo, nm, nDotL, sunVisibility, sky, occlusion, cloudSun);
    if (look.shine > 0.0) {
        float3 v = normalize(-in.relative);
        float3 hv = normalize(l + v);
        float fresnel = 0.3 + 0.7 * pow(1.0 - saturate(dot(nm, v)), 5.0);
        color += u.sunRadiance * (look.shine * fresnel * sunVisibility * pow(saturate(dot(nm, hv)), 48.0) * 6.0 * nDotL);
    }
    return finishSurface(u, skyView, clouds, cloudDistance, color, in.world, in.position.xy);
}
"""
