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
/// Each view's GPU frame time is printed and written to timings.csv.
enum Snapshots {
    static func option(_ name: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
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
                                               fineDetail: option("--detail") != "off")
            print("shadows: \(renderer.shadows.rawValue)")
            // Viewpoints are picked on the overview; the camera then stands on the full-detail ground.
            let height = try tiles.grid(level: tiles.overviewLevel())
            let dims = option("--size")?.split(separator: "x").compactMap { Int($0) } ?? []
            let (width, rows) = dims.count == 2 ? (dims[0], dims[1]) : (1280, 720)
            var timings = "view,time,gpu_ms,depth_ms,clouds_ms,shade_ms,nodes,tiles\n"
            let day = option("--day").flatMap { Int($0) } ?? SunPath.defaultDay
            let hours = option("--time")?.split(separator: ",").compactMap { Double($0) } ?? []
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

            for view in Viewpoints.automatic(height: height, drainageArea: area, scarp: scarp) {
                var c = view.camera
                renderer.prepareGround(x: c.x, y: c.y)
                c.z = renderer.groundHeight(x: c.x, y: c.y) + 1.7
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
                    print(String(format: "wrote %@ (GPU %.1f ms: depth %.1f, clouds %.1f, shading %.1f; %d nodes, %d tiles)",
                                 url.path, s.gpuMilliseconds, passes[0], passes[1], passes[2], s.nodes, s.residentTiles))
                    timings += String(format: "%@,%@,%.2f,%.2f,%.2f,%.2f,%d,%d\n", view.name, hour.map { String($0) } ?? "reference",
                                      s.gpuMilliseconds, passes[0], passes[1], passes[2], s.nodes, s.residentTiles)
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
