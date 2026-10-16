import Foundation
import IslandCore
import IslandGen

// islandgen: bake an island from a seed and write preview images.
//
//   islandgen --seed 42 --size 1024 --steps 400 --out out/

func usage() -> Never {
    print("""
    usage: islandgen [--seed N] [--size N] [--steps N] [--domain METRES] [--out DIR]
      --seed    random seed (default 1)
      --size    grid samples per side (default 1024)
      --steps   erosion time steps (default 400)
      --domain  domain edge length in metres (default 80000)
      --out     output directory (default ./out)
    """)
    exit(2)
}

func parsed<T>(_ value: T?) -> T {
    guard let value else { usage() }
    return value
}

var params = Island.Params()
var outDir = "out"
var args = CommandLine.arguments.dropFirst()
while let flag = args.popFirst() {
    guard let value = args.popFirst() else { usage() }
    switch flag {
    case "--seed": params.seed = parsed(UInt64(value))
    case "--size": params.size = parsed(Int(value))
    case "--steps": params.steps = parsed(Int(value))
    case "--domain": params.domainSize = parsed(Double(value))
    case "--out": outDir = value
    default: usage()
    }
}

let out = URL(fileURLWithPath: outDir, isDirectory: true)
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

let start = Date()
var lastReported = -1
let island = Island.generate(params) { fraction in
    let percent = Int(fraction * 100)
    if percent / 10 != lastReported / 10 {
        lastReported = percent
        print("eroding… \(percent)%")
    }
}
let elapsed = Date().timeIntervalSince(start)

let sea = params.erosion.seaLevel
try Preview.write(Preview.hillshade(island.height, seaLevel: sea), to: out.appendingPathComponent("hillshade.png"))
try Preview.write(Preview.flow(island.drainageArea), to: out.appendingPathComponent("flow.png"))
try Preview.write(Preview.gray(island.rain), to: out.appendingPathComponent("rain.png"))
try Preview.write(Preview.gray(island.hardness), to: out.appendingPathComponent("hardness.png"))
try Preview.writeRaw(island.height, to: out.appendingPathComponent("height.r32"))

// Ground-level pictures from a walker's eye height, rendered on the CPU.
for view in Viewpoints.automatic(height: island.height, drainageArea: island.drainageArea, seaLevel: sea) {
    let image = GroundView.render(island.height, camera: view.camera, seaLevel: sea)
    try Preview.write(image, to: out.appendingPathComponent("view-\(view.name).png"))
}

let land = island.height.values.filter { $0 > sea }.count
let landKm2 = Double(land) * island.height.cellArea / 1e6
let info: [String: Any] = [
    "seed": params.seed,
    "size": params.size,
    "domainMetres": params.domainSize,
    "cellMetres": island.height.cellSize,
    "steps": params.steps,
    "landAreaKm2": (landKm2 * 10).rounded() / 10,
    "maxHeightMetres": Double(island.height.range.max),
    "seconds": (elapsed * 10).rounded() / 10,
]
let json = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
try json.write(to: out.appendingPathComponent("info.json"))
print(String(decoding: json, as: UTF8.self))
