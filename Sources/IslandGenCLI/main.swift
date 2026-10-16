import Foundation
import IslandCore
import IslandGen

// islandgen: bake an island from a seed and write preview images.
//
//   islandgen --seed 42 --size 1024 --steps 400 --out out/

func usage() -> Never {
    print("""
    usage: islandgen [--seed N] [--size N] [--steps N] [--domain METRES] [--deposition G] [--slope-exponent N] [--reference-slope S] [--critical-slope S] [--knickpoint-slope S] [--waterfall-retreat R] [--headwall-retreat R] [--waterfall-min-q Q] [--profile] [--out DIR]
      --seed    random seed (default 1)
      --size    grid samples per side (default 1024)
      --steps   erosion time steps (default 400)
      --domain  domain edge length in metres (default 80000)
      --deposition  sediment deposition coefficient G, 0 for none (default 0.3)
      --slope-exponent  stream power slope exponent n, area exponent follows as n/2 (default 1)
      --reference-slope  slope at which n leaves erosion rates unchanged (default 0.1)
      --critical-slope   steepest stable wall, rise over run; 0 disables (default 0)
      --knickpoint-slope  with n > 1: erode as n = 1 below this slope, boosted above it (default 0 = pure power law)
      --waterfall-retreat  headwall retreat in m/yr at the reference discharge; 0 disables (default 0)
      --waterfall-min-q    smallest rain-weighted drainage (m³/yr) that drives waterfall retreat (default 1e6)
      --headwall-retreat   retreat of every face steeper than 0.6, m/yr at 3 m/yr rain; 0 disables (default 0)
      --profile print time spent in each pipeline phase
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
var timer: PhaseTimer?
var args = CommandLine.arguments.dropFirst()
while let flag = args.popFirst() {
    if flag == "--profile" { timer = PhaseTimer(); continue }
    guard let value = args.popFirst() else { usage() }
    switch flag {
    case "--seed": params.seed = parsed(UInt64(value))
    case "--size": params.size = parsed(Int(value))
    case "--steps": params.steps = parsed(Int(value))
    case "--domain": params.domainSize = parsed(Double(value))
    case "--deposition": params.erosion.deposition = parsed(Double(value))
    case "--slope-exponent":
        params.erosion.slopeExponent = parsed(Double(value))
        params.erosion.areaExponent = params.erosion.slopeExponent / 2
    case "--reference-slope": params.erosion.referenceSlope = parsed(Double(value))
    case "--critical-slope": params.erosion.criticalSlope = parsed(Double(value))
    case "--knickpoint-slope": params.erosion.knickpointSlope = parsed(Double(value))
    case "--waterfall-retreat": params.erosion.waterfallRetreat = parsed(Double(value))
    case "--headwall-retreat": params.erosion.headwallRetreat = parsed(Double(value))
    case "--waterfall-min-q": params.erosion.waterfallMinDischarge = parsed(Double(value))
    case "--out": outDir = value
    default: usage()
    }
}

let out = URL(fileURLWithPath: outDir, isDirectory: true)
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

let start = Date()
var lastReported = -1
let island = Island.generate(params, timer: timer) { fraction in
    let percent = Int(fraction * 100)
    if percent / 10 != lastReported / 10 {
        lastReported = percent
        print("eroding… \(percent)%")
    }
}
let elapsed = Date().timeIntervalSince(start)
if let timer { print("time by phase (\(ProcessInfo.processInfo.activeProcessorCount) cores):\n" + timer.report) }

let sea = params.erosion.seaLevel
try Preview.write(Preview.hillshade(island.height, seaLevel: sea), to: out.appendingPathComponent("hillshade.png"))
try Preview.write(Preview.flow(island.drainageArea), to: out.appendingPathComponent("flow.png"))
try Preview.write(Preview.gray(island.rain), to: out.appendingPathComponent("rain.png"))
try Preview.write(Preview.gray(island.hardness), to: out.appendingPathComponent("hardness.png"))
try Preview.writeRaw(island.height, to: out.appendingPathComponent("height.r32"))
if let scarp = island.scarp {
    // Lets IslandApp's snapshots aim the pali view at the same cliff.
    let json: [String: Double] = ["x": scarp.x, "y": scarp.y, "direction": scarp.direction, "halfLength": scarp.halfLength]
    try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]).write(to: out.appendingPathComponent("scarp.json"))
}

// Ground-level pictures from a walker's eye height, rendered on the CPU.
for view in Viewpoints.automatic(height: island.height, drainageArea: island.drainageArea, seaLevel: sea, scarp: island.scarp) {
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
    "slopeExponent": params.erosion.slopeExponent,
    "deposition": params.erosion.deposition,
    "referenceSlope": params.erosion.referenceSlope,
    "criticalSlope": params.erosion.criticalSlope,
    "knickpointSlope": params.erosion.knickpointSlope,
    "waterfallRetreat": params.erosion.waterfallRetreat,
    "headwallRetreat": params.erosion.headwallRetreat,
    "waterfallMinDischarge": params.erosion.waterfallMinDischarge,
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
