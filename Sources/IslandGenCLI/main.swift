import Foundation
import IslandCore
import IslandGen

// islandgen: bake an island from a seed and write preview images.
//
//   islandgen --seed 42 --size 1024 --steps 400 --out out/

func usage() -> Never {
    print("""
    usage: islandgen [--seed N] [--size N] [--steps N] [--domain METRES] [--deposition G] [--out DIR]
      --seed    random seed (default 1)
      --size    grid samples per side (default 1024)
      --steps   erosion time steps (default 400)
      --domain  domain edge length in metres (default 80000)
      --deposition  sediment deposition coefficient G, 0 for none (default 1)
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
    case "--deposition": params.erosion.deposition = parsed(Double(value))
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
let lowland = island.height.values.filter { $0 > sea && $0 < sea + 20 }.count

// Mean land slope on each side of the island relative to the wind: the windward half
// should be the more dissected (steeper on average), not planed flat.
let wind = (cos(params.climate.windDirection), sin(params.climate.windDirection))
var summitIndex = 0
for i in 0..<island.height.count where island.height.values[i] > island.height.values[summitIndex] { summitIndex = i }
let sx = summitIndex % params.size, sy = summitIndex / params.size
var slopeSum = (windward: 0.0, leeward: 0.0), slopeCount = (windward: 0, leeward: 0)
var steep = 0 // land steeper than 35°: cliffs such as a pali
for y in 1..<(params.size - 1) {
    for x in 1..<(params.size - 1) {
        let g = island.height
        guard g[x, y] > sea + 1 else { continue }
        let gx = Double(g[x + 1, y] - g[x - 1, y]) / (2 * g.cellSize)
        let gy = Double(g[x, y + 1] - g[x, y - 1]) / (2 * g.cellSize)
        let slope = (gx * gx + gy * gy).squareRoot()
        if slope > 0.7 { steep += 1 }
        // Upwind of the summit means against the wind direction.
        if Double(x - sx) * wind.0 + Double(y - sy) * wind.1 < 0 {
            slopeSum.windward += slope; slopeCount.windward += 1
        } else {
            slopeSum.leeward += slope; slopeCount.leeward += 1
        }
    }
}
func rounded(_ v: Double, _ places: Double = 1000) -> Double { (v * places).rounded() / places }
let landKm2 = Double(land) * island.height.cellArea / 1e6
let info: [String: Any] = [
    "seed": params.seed,
    "size": params.size,
    "domainMetres": params.domainSize,
    "cellMetres": island.height.cellSize,
    "steps": params.steps,
    "landAreaKm2": (landKm2 * 10).rounded() / 10,
    "maxHeightMetres": Double(island.height.range.max),
    "steepFraction": rounded(Double(steep) / Double(Swift.max(land, 1)), 10000),
    "lowlandFraction": rounded(Double(lowland) / Double(Swift.max(land, 1))),
    "meanSlopeWindward": rounded(slopeSum.windward / Double(Swift.max(slopeCount.windward, 1))),
    "meanSlopeLeeward": rounded(slopeSum.leeward / Double(Swift.max(slopeCount.leeward, 1))),
    "seconds": (elapsed * 10).rounded() / 10,
]
let json = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
try json.write(to: out.appendingPathComponent("info.json"))
print(String(decoding: json, as: UTF8.self))
