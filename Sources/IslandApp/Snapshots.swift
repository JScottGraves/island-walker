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
/// of ray-traced shadows.
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
            let height = IslandSource.load()
            let requested: TerrainRenderer.Shadows = option("--shadows") == "horizon" ? .horizonMap : .rayTraced
            let renderer = try TerrainRenderer(device: device, colorFormat: .bgra8Unorm_srgb,
                                               depthFormat: .depth32Float, height: height, seaLevel: 0,
                                               shadows: requested)
            print("shadows: \(renderer.shadows.rawValue)")
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
            let args = CommandLine.arguments
            if args.count > 1, args[1].hasSuffix(".r32") {
                let url = URL(fileURLWithPath: args[1]).deletingLastPathComponent().appendingPathComponent("scarp.json")
                if let data = try? Data(contentsOf: url),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Double],
                   let x = json["x"], let y = json["y"], let direction = json["direction"] {
                    scarp = FlankCollapse.Scarp(x: x, y: y, direction: direction)
                    scarp?.halfLength = json["halfLength"] ?? 9000
                }
            }

            for view in Viewpoints.automatic(height: height, drainageArea: area, scarp: scarp) {
                let c = view.camera
                let forward = SIMD3<Float>(Float(sin(c.yaw) * cos(c.pitch)), Float(cos(c.yaw) * cos(c.pitch)), Float(sin(c.pitch)))
                // nil: the reference sun.
                for hour in hours.isEmpty ? [nil] : hours.map(Optional.some) {
                    var name = "app-\(view.name)"
                    if let hour {
                        renderer.light = SunPath.light(hour: hour, day: day)
                        let minutes = Int((hour * 60).rounded())
                        name += String(format: "-%02d%02d", minutes / 60, minutes % 60)
                    } else {
                        renderer.light = .reference
                    }
                    let rgb = try renderer.snapshot(width: 1280, height: 720, eye: SIMD3(c.x, c.y, c.z), forward: forward)
                    let url = out.appendingPathComponent(name + ".png")
                    try Data(PNG.encode(rgb: rgb, width: 1280, height: 720)).write(to: url)
                    print("wrote \(url.path)")
                }
            }
        } catch {
            print("snapshot failed: \(error)")
            exit(1)
        }
    }
}
#endif
