#if os(macOS)
import Foundation
import Metal
import simd
import Engine
import IslandCore
import IslandGen

/// Headless mode: renders the app's own Metal view of the automatic viewpoints to PNGs,
/// so what the app draws can be checked without a window or screen-recording permission.
enum Snapshots {
    static func run(outDir: String) {
        let out = URL(fileURLWithPath: outDir, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            guard let device = MTLCreateSystemDefaultDevice() else { throw RendererError.setup("no Metal device") }
            let height = IslandSource.load()
            let renderer = try TerrainRenderer(device: device, colorFormat: .bgra8Unorm_srgb,
                                               depthFormat: .depth32Float, height: height, seaLevel: 0)
            // Viewpoint picking needs drainage area; one routing step on a copy gives it.
            var copy = height
            var params = StreamPower.Params()
            params.randomRouting = false
            params.erodibility = 0
            params.diffusivity = 0
            let area = StreamPower.step(&copy, params: params).area

            for view in Viewpoints.automatic(height: height, drainageArea: area) {
                let c = view.camera
                let forward = SIMD3<Float>(Float(sin(c.yaw) * cos(c.pitch)), Float(cos(c.yaw) * cos(c.pitch)), Float(sin(c.pitch)))
                let rgb = try renderer.snapshot(width: 1280, height: 720, eye: SIMD3(c.x, c.y, c.z), forward: forward)
                let url = out.appendingPathComponent("app-\(view.name).png")
                try Data(PNG.encode(rgb: rgb, width: 1280, height: 720)).write(to: url)
                print("wrote \(url.path)")
            }
        } catch {
            print("snapshot failed: \(error)")
            exit(1)
        }
    }
}
#endif
