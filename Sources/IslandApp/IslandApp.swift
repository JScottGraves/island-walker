#if os(macOS)
import AppKit
import MetalKit
import SwiftUI
import simd
import Engine
import IslandCore
import IslandGen
import Walker

// Launch with `swift run -c release IslandApp [path/to/island.bundle | path/to/height.r32]`.
// With a bundle or .r32 from islandgen it shows that island; otherwise it bakes a quick 512² one.
// `--snapshot DIR` renders the automatic viewpoints offscreen to DIR/app-*.png and exits.
// Controls: WASD walk, Shift run, drag to look, F toggles fly mode, Space/C up/down in fly mode,
// [ and ] move the time of day, T lets time run (an hour every 20 s). `--time 16.5` starts at
// 16:30 local solar time, `--day N` picks the day of year, `--shadows horizon` skips ray tracing,
// `--view coast` starts at that snapshot viewpoint (summit, ridge, coast, pali), `--camera X,Y,YAW,PITCH`
// (metres, degrees) at that spot, and `--flora off` draws no trees. `--cover off` draws no grasses and
// sedges; `--cover-range M` (default 50) and `--cover-density X` (default 1) tune them.

@main
struct IslandApp: App {
    init() {
        if let i = CommandLine.arguments.firstIndex(of: "--snapshot"), i + 1 < CommandLine.arguments.count {
            Snapshots.run(outDir: CommandLine.arguments[i + 1])
            exit(0)
        }
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Island Walker") {
            TerrainView()
                .frame(minWidth: 960, minHeight: 600)
        }
    }
}

struct TerrainView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> InputMTKView {
        let view = InputMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = .bgra8Unorm_srgb
        // The renderer keeps its own depth buffer: the cloud pass reads it.
        view.depthStencilPixelFormat = .invalid
        view.clearColor = TerrainRenderer.skyClearColor
        view.preferredFramesPerSecond = 120
        context.coordinator.attach(view)
        return view
    }

    func updateNSView(_ nsView: InputMTKView, context: Context) {}
}

/// MTKView that records keyboard and mouse state for the walker.
final class InputMTKView: MTKView {
    var keys = Set<UInt16>()
    var shift = false
    var lookDelta = SIMD2<Double>(0, 0)

    override var acceptsFirstResponder: Bool { true }
    override func viewDidMoveToWindow() { window?.makeFirstResponder(self) }
    override func keyDown(with event: NSEvent) { keys.insert(event.keyCode) }
    override func keyUp(with event: NSEvent) { keys.remove(event.keyCode) }
    override func flagsChanged(with event: NSEvent) { shift = event.modifierFlags.contains(.shift) }
    override func mouseDragged(with event: NSEvent) {
        lookDelta += SIMD2(Double(event.deltaX), Double(event.deltaY))
    }
}

final class Coordinator: NSObject, MTKViewDelegate {
    var renderer: TerrainRenderer?
    var walker: Walker?
    var flying = false
    var flyHeight = 0.0
    var lastTime = CACurrentMediaTime()
    var fWasDown = false
    var tWasDown = false
    /// Local solar time of day and day of year; the sun follows them.
    var hour = 9.5
    var day = SunPath.defaultDay
    var timeRuns = false
    var shownMinute = -1
    var shownTick = -1
    var litHour = -1.0

    // macOS virtual key codes
    let kW: UInt16 = 13, kA: UInt16 = 0, kS: UInt16 = 1, kD: UInt16 = 2
    let kF: UInt16 = 3, kSpace: UInt16 = 49, kC: UInt16 = 8
    let kT: UInt16 = 17, kLeftBracket: UInt16 = 33, kRightBracket: UInt16 = 30

    func attach(_ view: InputMTKView) {
        view.delegate = self
        guard let device = view.device else { return }
        let (tiles, clouds) = IslandSource.terrain()
        let requested: TerrainRenderer.Shadows = Snapshots.option("--shadows") == "horizon" ? .horizonMap : .rayTraced
        do {
            renderer = try TerrainRenderer(device: device, colorFormat: view.colorPixelFormat,
                                           depthFormat: .depth32Float, terrain: tiles,
                                           shadows: requested, clouds: clouds,
                                           detailMaps: IslandSource.detailMaps(),
                                           fineDetail: Snapshots.option("--detail") != "off",
                                           flora: IslandSource.floraMaps(),
                                           groundCover: GroundCoverSettings(arguments: CommandLine.arguments),
                                           antialiasing: Snapshots.antialiasing())
            renderer?.jitterSign = Snapshots.option("--jitter-sign").flatMap(Float.init) ?? 1
        } catch {
            print("renderer failed: \(error)")
            return
        }
        hour = Snapshots.option("--time").flatMap { Double($0) } ?? hour
        day = Snapshots.option("--day").flatMap { Int($0) } ?? day
        guard let overview = try? tiles.grid(level: tiles.overviewLevel()) else { return }
        // `--camera X,Y,YAW,PITCH` or `--view coast` (or summit, ridge, pali…) starts there,
        // facing that way; otherwise at the highest point so the whole island is in view.
        let named = Snapshots.option("--view").flatMap { name in Snapshots.viewpoints(overview).first { $0.name == name } }
        if let view = Snapshots.cameras().first ?? named.map({ (name: $0.name, camera: $0.camera) }) {
            renderer?.prepareGround(x: view.camera.x, y: view.camera.y)
            var w = Walker(x: view.camera.x, y: view.camera.y, ground: ground)
            w.yaw = view.camera.yaw
            w.pitch = view.camera.pitch
            walker = w
            return
        }
        var best = 0
        for i in 0..<overview.count where overview.values[i] > overview.values[best] { best = i }
        let sx = (Double(best % overview.width) + 0.5) * overview.cellSize
        let sy = (Double(best / overview.width) + 0.5) * overview.cellSize
        renderer?.prepareGround(x: sx, y: sy)
        walker = Walker(x: sx, y: sy, ground: ground)
    }

    func ground(_ x: Double, _ y: Double) -> Double {
        renderer?.groundHeight(x: x, y: y) ?? 0
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let view = view as? InputMTKView, let renderer, var w = walker else { return }
        let now = CACurrentMediaTime()
        let dt = Swift.min(now - lastTime, 0.1)
        lastTime = now

        w.yaw += view.lookDelta.x * 0.003
        w.pitch = Swift.max(-1.4, Swift.min(1.4, w.pitch - view.lookDelta.y * 0.003))
        view.lookDelta = .zero

        let fDown = view.keys.contains(kF)
        if fDown && !fWasDown {
            flying.toggle()
            flyHeight = w.z + 50
        }
        fWasDown = fDown

        let tDown = view.keys.contains(kT)
        if tDown && !tWasDown { timeRuns.toggle() }
        tWasDown = tDown
        // Brackets scrub 1.5 hours a second; running time is an hour every 20 s.
        let scrub = (view.keys.contains(kRightBracket) ? 1.0 : 0) - (view.keys.contains(kLeftBracket) ? 1.0 : 0)
        let advance = scrub * 1.5 * dt + (timeRuns ? dt / 20 : 0)
        hour += advance
        // The wind keeps blowing in real time even while the sun stands still.
        renderer.clock += dt + Swift.max(advance, 0) * 3600
        renderer.hour = hour
        hour = (hour + 24).truncatingRemainder(dividingBy: 24)
        // Rebuilding the sky takes a few milliseconds: only when the sun has moved ~0.1°.
        if abs(hour - litHour) > 1.0 / 120 {
            renderer.light = SunPath.light(hour: hour, day: day)
            litHour = hour
        }
        // Debug overlay in the title bar: time of day, GPU frame time, what's streamed in.
        let minute = Int(hour * 60)
        let now10 = Int(now * 4)
        if minute != shownMinute || now10 != shownTick {
            shownMinute = minute
            shownTick = now10
            let s = renderer.stats
            view.window?.title = String(format: "Island Walker · %02d:%02d · GPU %.1f ms · %d nodes · %d tiles (%d loading)",
                                        minute / 60, minute % 60, s.gpuMilliseconds, s.nodes, s.residentTiles, s.loadingTiles)
        }

        let forward = (view.keys.contains(kW) ? 1.0 : 0) - (view.keys.contains(kS) ? 1.0 : 0)
        let strafe = (view.keys.contains(kD) ? 1.0 : 0) - (view.keys.contains(kA) ? 1.0 : 0)
        if flying {
            let speed = view.shift ? 400.0 : 60.0
            w.x += (sin(w.yaw) * forward + cos(w.yaw) * strafe) * speed * dt
            w.y += (cos(w.yaw) * forward - sin(w.yaw) * strafe) * speed * dt
            if view.keys.contains(kSpace) { flyHeight += speed * dt }
            if view.keys.contains(kC) { flyHeight -= speed * dt }
            flyHeight = Swift.max(flyHeight, ground(w.x, w.y) + 2)
            w.z = flyHeight
        } else {
            w.update(forward: forward, strafe: strafe, running: view.shift, dt: dt, ground: ground)
        }
        walker = w

        let dir = SIMD3<Float>(Float(sin(w.yaw) * cos(w.pitch)), Float(cos(w.yaw) * cos(w.pitch)), Float(sin(w.pitch)))
        let eye = SIMD3<Double>(w.x, w.y, flying ? w.z : w.z + w.eyeHeight)
        renderer.draw(in: view, eye: eye, forward: dir)
    }
}
#else
@main
struct IslandApp {
    static func main() { print("IslandApp runs on macOS only.") }
}
#endif
