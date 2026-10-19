#if os(macOS)
import Foundation
import Metal
import simd
import Engine
import IslandCore
import IslandGen

/// Headless mode: renders the app's own Metal view of the automatic viewpoints to PNGs,
/// so what the app draws can be checked without a window or screen-recording permission.
///
/// `--time 7,10.5,16` renders each viewpoint at those local solar times (app-NAME-HHMM.png)
/// on `--day N` (default mid-October); without it, at the fixed reference sun older
/// renders used (app-NAME.png). `--shadows horizon` uses the baked horizon map instead
/// of ray-traced shadows. `--size 2560x1440` sets the resolution (default 1280x720).
/// `--detail off` draws only the bundle's own grid, without the generated fine detail;
/// `--pitch DEG` looks that far below the horizon from every viewpoint (ground close-ups).
/// `--flora off` draws no trees (for A/B timing); `--cover off` no grasses and sedges,
/// `--cover-range M` and `--cover-density X` tune them. `--eye H` stands the camera H metres
/// above the ground instead of 1.7 (over the canopy, for distant views). `--camera X,Y,YAW,PITCH` (metres,
/// degrees; yaw clockwise from north, pitch up) adds a view named camN, or NAME=X,Y,YAW,PITCH
/// one named NAME; repeat it for more. A fifth value stands that view's camera that many metres
/// above the ground instead (a lookout platform above the trees). `--views a,b` renders only the views named.
/// Each view's GPU frame time is printed and written to timings.csv.
enum Snapshots {
    static func option(_ name: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// Every value given for an option that may repeat.
    static func options(_ name: String) -> [String] {
        let args = CommandLine.arguments
        return args.indices.filter { args[$0] == name && $0 + 1 < args.count }.map { args[$0 + 1] }
    }

    /// Views given with `--camera [NAME=]X,Y,YAW,PITCH`.
    static func cameras() -> [(name: String, camera: GroundView.Camera, eye: Double?)] {
        options("--camera").enumerated().compactMap { k, value in
            var name = "cam\(k + 1)", spec = Substring(value)
            if let eq = value.firstIndex(of: "=") {
                name = String(value[..<eq])
                spec = value[value.index(after: eq)...]
            }
            let v = spec.split(separator: ",").compactMap { Double($0) }
            guard v.count == 4 || v.count == 5 else {
                print("ignoring --camera \(value): expected X,Y,YAW,PITCH[,EYE]")
                return nil
            }
            return (name, GroundView.Camera(x: v[0], y: v[1], z: 0, yaw: v[2] * .pi / 180, pitch: v[3] * .pi / 180),
                    v.count == 5 ? v[4] : nil)
        }
    }

    /// The automatic viewpoints (summit, ridge, coast, pali…) picked on `height`.
    static func viewpoints(_ height: Grid) -> [Viewpoints.Named] {
        // Viewpoint picking needs drainage area; one routing step on a copy gives it.
        var copy = height
        var params = StreamPower.Params()
        params.randomRouting = false
        params.erodibility = 0
        params.diffusivity = 0
        let area = StreamPower.step(&copy, params: params).area

        // islandgen writes the flank collapse's scarp next to height.r32.
        var scarp: FlankCollapse.Scarp?
        if let dir = IslandSource.inputDirectory {
            let url = dir.appendingPathComponent("scarp.json")
            if let data = try? Data(contentsOf: url),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Double],
               let x = json["x"], let y = json["y"], let direction = json["direction"] {
                scarp = FlankCollapse.Scarp(x: x, y: y, direction: direction)
                scarp?.halfLength = json["halfLength"] ?? 9000
            }
        }
        return Viewpoints.automatic(height: height, drainageArea: area, scarp: scarp)
    }

    /// `--aa off` draws without antialiasing; `--aa taa` (the default) with our temporal
    /// antialiasing; `--scale X` at X of the output size, upscaled by MetalFX.
    static func antialiasing() -> TerrainRenderer.Antialiasing {
        if let scale = option("--scale").flatMap(Double.init) { return .upscaled(Swift.min(Swift.max(scale, 0.5), 1)) }
        return option("--aa") == "off" ? .off : .temporal
    }

    static func run(outDir: String) {
        let out = URL(fileURLWithPath: outDir, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            guard let device = MTLCreateSystemDefaultDevice() else { throw RendererError.setup("no Metal device") }
            let (tiles, clouds) = IslandSource.terrain()
            let requested: TerrainRenderer.Shadows = option("--shadows") == "horizon" ? .horizonMap : .rayTraced
            let renderer = try TerrainRenderer(device: device, colorFormat: .bgra8Unorm_srgb,
                                               depthFormat: .depth32Float, terrain: tiles,
                                               shadows: requested, clouds: clouds,
                                               detailMaps: IslandSource.detailMaps(),
                                               fineDetail: option("--detail") != "off",
                                               flora: IslandSource.floraMaps(),
                                               groundCover: GroundCoverSettings(arguments: CommandLine.arguments),
                                               antialiasing: antialiasing())
            renderer.jitterSign = option("--jitter-sign").flatMap(Float.init) ?? 1
            print("shadows: \(renderer.shadows.rawValue); antialiasing: \(renderer.antialiasing)")
            // Viewpoints are picked on the overview; the camera then stands on the full-detail ground.
            let height = try tiles.grid(level: tiles.overviewLevel())
            let dims = option("--size")?.split(separator: "x").compactMap { Int($0) } ?? []
            let (width, rows) = dims.count == 2 ? (dims[0], dims[1]) : (1280, 720)
            var timings = "view,time,gpu_ms,depth_ms,clouds_ms,shade_ms,nodes,tiles,trees,tree_tris\n"
            let day = option("--day").flatMap { Int($0) } ?? SunPath.defaultDay
            let hours = option("--time")?.split(separator: ",").compactMap { Double($0) } ?? []
            let only = option("--views")?.split(separator: ",").map(String.init)
            let views = viewpoints(height).map { (name: $0.name, camera: $0.camera, eye: Double?.none) } + cameras()
            for view in views where only?.contains(view.name) ?? true {
                var c = view.camera
                renderer.prepareGround(x: c.x, y: c.y)
                c.z = renderer.groundHeight(x: c.x, y: c.y) + (view.eye ?? option("--eye").flatMap(Double.init) ?? 1.7)
                if let degrees = option("--pitch").flatMap(Double.init) { c.pitch = -degrees * .pi / 180 }
                let forward = SIMD3<Float>(Float(sin(c.yaw) * cos(c.pitch)), Float(cos(c.yaw) * cos(c.pitch)), Float(sin(c.pitch)))
                // nil: the reference sun.
                for hour in hours.isEmpty ? [nil] : hours.map(Optional.some) {
                    var name = "app-\(view.name)"
                    if let hour {
                        renderer.light = SunPath.light(hour: hour, day: day)
                        renderer.hour = hour
                        // The same wind state for every viewpoint at a given hour.
                        renderer.clock = hour * 3600
                        let minutes = Int((hour * 60).rounded())
                        name += String(format: "-%02d%02d", minutes / 60, minutes % 60)
                    } else {
                        renderer.light = .reference
                    }
                    let rgb = try renderer.snapshot(width: width, height: rows, eye: SIMD3(c.x, c.y, c.z), forward: forward)
                    let url = out.appendingPathComponent(name + ".png")
                    try Data(PNG.encode(rgb: rgb, width: width, height: rows)).write(to: url)
                    let s = renderer.stats
                    let passes = s.passMilliseconds.count == 3 ? s.passMilliseconds : [0, 0, 0]
                    print(String(format: "wrote %@ (GPU %.1f ms: depth %.1f, clouds %.1f, shading %.1f; %d nodes, %d tiles, %d trees, %.1fM tree tris; shadow casters %@)",
                                 url.path, s.gpuMilliseconds, passes[0], passes[1], passes[2], s.nodes, s.residentTiles,
                                 s.floraInstances, Double(s.floraTriangles) / 1e6,
                                 s.shadowCasters.map { String(format: "%d/%.1fM", $0.instances, Double($0.triangles) / 1e6) }
                                     .joined(separator: " ")))
                    if s.coverBlades != (0, 0) { print("  cover blades \(s.coverBlades.0) + \(s.coverBlades.1)") }
                    print("  ocean nodes \(s.oceanNodes)")
                    timings += String(format: "%@,%@,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d\n", view.name, hour.map { String($0) } ?? "reference",
                                      s.gpuMilliseconds, passes[0], passes[1], passes[2], s.nodes, s.residentTiles,
                                      s.floraInstances, s.floraTriangles)
                }
            }
            try timings.write(to: out.appendingPathComponent("timings.csv"), atomically: true, encoding: .utf8)
        } catch {
            print("snapshot failed: \(error)")
            exit(1)
        }
    }
}
#endif
