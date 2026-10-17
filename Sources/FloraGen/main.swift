import Foundation
import Metal
import Engine
import Flora
import IslandCore
import IslandGen

// floragen: specimen sheets of the native plants, drawn with the app's own Metal renderer
// (terrain, sun, sky, ray-traced shadows, haze and the tree shaders) on a flat, neutral
// ground at 13:00, each variant in a row with a 1.7 m figure for scale.
//
//   swift run -c release floragen --out DIR [--size 1280x720] [--time 13] [--plants ohia,koa]
//                                  [--lod 0|1|both] [--obj DIR]
//
// Writes DIR/flora-PLANT-lodN.png and DIR/flora-stats.csv (triangles per variant and LOD).
// `--obj DIR` also exports every mesh as Wavefront OBJ for inspection.

func option(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

/// Writes one mesh as OBJ (positions, normals, texture coordinates; one group per material).
func writeOBJ(_ mesh: PlantMesh, to url: URL) throws {
    var s = "# floragen \(mesh.vertices.count) vertices, \(mesh.triangleCount) triangles\n"
    for v in mesh.vertices { s += String(format: "v %.4f %.4f %.4f\n", v.x, v.z, -v.y) }
    for v in mesh.vertices { s += String(format: "vn %.4f %.4f %.4f\n", v.nx, v.nz, -v.ny) }
    for v in mesh.vertices { s += String(format: "vt %.4f %.4f\n", v.u, v.v) }
    var material = -1
    for t in stride(from: 0, to: mesh.indices.count, by: 3) {
        let m = Int(mesh.vertices[Int(mesh.indices[t])].material)
        if m != material {
            material = m
            s += "g \(FloraMaterial.table[m].name.replacingOccurrences(of: " ", with: "_"))\n"
        }
        let a = mesh.indices[t] + 1, b = mesh.indices[t + 1] + 1, c = mesh.indices[t + 2] + 1
        s += "f \(a)/\(a)/\(a) \(b)/\(b)/\(b) \(c)/\(c)/\(c)\n"
    }
    try s.write(to: url, atomically: true, encoding: .utf8)
}

do {
    guard let outPath = option("--out") else {
        print("usage: floragen --out DIR [--size WxH] [--time H] [--plants a,b] [--lod 0|1|both] [--obj DIR]")
        exit(2)
    }
    let out = URL(fileURLWithPath: outPath, isDirectory: true)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let dims = option("--size")?.split(separator: "x").compactMap { Int($0) } ?? []
    let (width, height) = dims.count == 2 ? (dims[0], dims[1]) : (1280, 720)
    let hour = option("--time").flatMap(Double.init) ?? 13
    let wanted = option("--plants")?.split(separator: ",").map(String.init)
    let plants = Plant.allCases.filter { wanted?.contains($0.id) ?? true }
    let lods: [Int] = option("--lod") == "1" ? [1] : option("--lod") == "0" ? [0] : [0, 1]

    var stats = "plant,variant,lod,triangles,vertices,height_m\n"
    for plant in Plant.allCases {
        for v in 0..<Flora.variants {
            for lod in 0..<Flora.lods {
                let m = Flora.mesh(plant, variant: v, lod: lod)
                stats += String(format: "%@,%d,%d,%d,%d,%.2f\n", plant.id, v, lod, m.triangleCount, m.vertices.count, m.bounds.max.z)
                if let objDir = option("--obj") {
                    let dir = URL(fileURLWithPath: objDir, isDirectory: true)
                    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                    try writeOBJ(m, to: dir.appendingPathComponent("\(plant.id)-v\(v)-lod\(lod).obj"))
                }
            }
        }
    }
    try stats.write(to: out.appendingPathComponent("flora-stats.csv"), atomically: true, encoding: .utf8)

    guard let device = MTLCreateSystemDefaultDevice() else { throw RendererError.setup("no Metal device") }
    // A flat plain 40 m above the sea, 2.56 km across, with no standing cloud, and a neutral
    // grey-brown ground in place of the plant-community colours.
    let side = 256
    let ground = Grid(width: side, height: side, cellSize: 10, fill: 40)
    let domain = Double(side) * 10
    let neutral = RGB.srgb(118, 112, 100)
    let canopy = ColourMap(width: 2, height: 2, domain: domain,
                           rgb: [Float](repeating: 0, count: 12).enumerated().map { i, _ in
                               Float([neutral.r, neutral.g, neutral.b][i % 3]) })
    let renderer = try TerrainRenderer(device: device, colorFormat: .bgra8Unorm_srgb, depthFormat: .depth32Float,
                                       terrain: GridTerrain(height: ground, seaLevel: 0),
                                       clouds: Grid(width: 1, height: 1, cellSize: domain),
                                       detailMaps: DetailMaps(canopy: canopy), fineDetail: false, floraSpecimens: true)
    renderer.light = SunPath.light(hour: hour, day: SunPath.defaultDay)
    renderer.hour = hour
    renderer.clock = 0
    renderer.windSpeed = 0

    for plant in plants {
        for lod in lods {
            // Variants side by side, spaced by the widest crown; the figure stands at the left.
            let meshes = (0..<Flora.variants).map { Flora.mesh(plant, variant: $0, lod: 0) }
            let spans = meshes.map { m -> Double in
                let b = m.bounds
                return Double(Swift.max(b.max.x - b.min.x, b.max.y - b.min.y))
            }
            let h = plant.referenceHeight
            let spacing = Swift.max(spans.max() ?? h, h * 0.45) * 1.05
            let cx = domain / 2, cy = domain / 2
            var list: [(plant: Plant?, variant: Int, lod: Int, x: Double, y: Double, yaw: Double, height: Double)] = []
            for v in 0..<Flora.variants {
                let x = cx + (Double(v) - 1.5) * spacing
                list.append((plant, v, lod, x, cy, 0.4 * Double(v), h))
            }
            list.append((nil, 0, 0, cx - 2.1 * spacing, cy - 0.25 * spacing, 0, 1.7))
            renderer.prepareGround(x: cx, y: cy)
            renderer.placeSpecimens(list)
            // Frame the row: wide enough for all four (horizontal field ~91° at 16:9), from
            // a little above the ground, looking slightly up at the crowns.
            let rowWidth = 4.6 * spacing
            let distance = Swift.max(rowWidth * 0.5, h * 1.7)
            let eyeZ = 40 + Swift.min(h * 0.35, 6) + 1.0
            let target = 40 + h * 0.45
            let pitch = atan2(target - eyeZ, distance)
            let forward = SIMD3<Float>(0, Float(cos(pitch)), Float(sin(pitch)))
            let rgb = try renderer.snapshot(width: width, height: height, eye: SIMD3(cx - 0.3 * spacing, cy - distance, eyeZ),
                                            forward: forward)
            let url = out.appendingPathComponent("flora-\(plant.id)-lod\(lod).png")
            try Data(PNG.encode(rgb: rgb, width: width, height: height)).write(to: url)
            print(String(format: "wrote %@ (GPU %.1f ms, %d triangles)", url.path, renderer.stats.gpuMilliseconds,
                         renderer.stats.floraTriangles))
        }
    }
} catch {
    print("floragen failed: \(error)")
    exit(1)
}
