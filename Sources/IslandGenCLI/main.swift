import Foundation
import IslandCore
import IslandGen

// islandgen: bake an island from a seed and write preview images.
//
//   islandgen --seed 42 --size 1024 --steps 400 --out out/

func usage() -> Never {
    print("""
    usage: islandgen --score HEIGHT.r32 --cell METRES [--width N] [--sea-level M]
             score an existing float32 heightfield (square unless --width; pits are filled first) and print realism metrics
           islandgen --import HEIGHT.r32 --cell METRES --bundle DIR [--sea-level M] [--out DIR] [--refine N] [--fine FILE] [refine options]
             write a square float32 heightfield, such as the real Oahu DEM, as an island bundle (height, rain and plant communities; heights kept as they are,
             or refined N times finer with --refine), and with --out a biomes.png preview and biomes.json
           islandgen [--seed N] [--size N] [--steps N] [--domain METRES] [--deposition G] [--slope-exponent N] [--reference-slope S] [--critical-slope S] [--knickpoint-slope S] [--waterfall-retreat R] [--headwall-retreat R] [--waterfall-min-q Q] [--land-exponent P] [--coast-slope A] [--rift-descent D] [--saddle-fill M] [--pond-time F] [--pond-hardness H] [--pond-share S] [--wave-retreat R] [--leeward-waves S] [--uplift U] [--flexure M] [--flexure-distance M] [--refine N] [--bundle DIR] [--profile] [--out DIR]
      --seed    random seed (default 1)
      --size    grid samples per side (default 1024)
      --steps   erosion time steps (default 400)
      --domain  domain edge length in metres (default 80000)
      --deposition  sediment deposition coefficient G, 0 for none (default 0.35)
      --slope-exponent  stream power slope exponent n, area exponent follows as n/2 (default 1)
      --reference-slope  slope at which n leaves erosion rates unchanged (default 0.1)
      --critical-slope   steepest stable wall, rise over run; 0 disables (default 0)
      --knickpoint-slope  with n > 1: erode as n = 1 below this slope, boosted above it (default 0 = pure power law)
      --waterfall-retreat  headwall retreat in m/yr at the reference discharge; 0 disables (default 0.05)
      --waterfall-min-q    smallest rain-weighted drainage (m³/yr) that drives waterfall retreat (default 3e5)
      --land-exponent      shield land profile top·(1 − r/shore)^P; 0 keeps the classic shape (default 2)
      --coast-slope        share of a straight cone mixed into the land profile, steepening the coast (default 0.35)
      --rift-descent       drop of each rift crest from summit to tip, as a share of its height (default 0.4)
      --saddle-fill        soft-maximum width in metres blending neighbouring shields, filling the saddle (default 0)
      --pond-time          share of the erosion run after which saddle lava floods in (default 0.9)
      --pond-hardness      erosion resistance multiplier of the ponded lava (default 2)
      --pond-share         share of the saddle the lava buries; the rest stands above it (default 0.7)
      --wave-retreat       sea-cliff retreat in m/yr on coasts facing the wind (default 0.004)
      --leeward-waves      share of wave retreat on coasts facing away from the wind (default 0.2)
      --uplift             uplift in m/yr, negative for subsidence (default 0)
      --flexure            flexural uplift in metres after erosion, growing inland from the coast (default 20)
      --refine             also refine the height N times finer (8 takes 1024² at 78 m to 8192² at 10 m) with detail and pit filling
    refine options (with --refine, here or with --import; --fine FILE also writes the refined height):
      --detail             RMS flute amplitude in metres per unit of --flute-base plus coarse slope (rise/run) on typically dissected ground (default 10)
      --flute-base         added to the coarse slope in the flute amplitude (default 0.35)
      --flute-start        coarse slope (rise/run) where flutes start (default 0.07)
      --flute-full         coarse slope (rise/run) where flutes are full (default 0.2)
      --dissection-cap     largest flute multiplier from coarse curvature relative to the island's mean at that slope (default 2)
      --planar-walls       least such multiplier on coarse ground of 40° and steeper, however planar (default 0.6)
      --flute-across       longest flute wavelength across the slope, metres (default 80)
      --flute-along        longest flute wavelength down the slope, metres (default 480)
      --roughness          amplitude ratio between successive flute octaves (default 0.65)
      --gentle             RMS of the rounded relief on all but the flattest ground, metres (default 2)
      --gentle-start       coarse slope (rise/run) where the rounded relief starts (default 0.02)
      --gentle-full        coarse slope (rise/run) where it is full (default 0.09)
      --gentle-across      its longest wavelength across the slope, metres (default 200)
      --gentle-along       its longest wavelength down the slope, metres (default 400)
      --gentle-roughness   amplitude ratio between its successive octaves (default 0.5)
      --channel-start      drainage area in m² where the detail start fading into a channel bed (default 1e5)
      --channel-full       drainage area in m² where the channel bed is smooth (default 1e7)
      --bundle             write the island bundle (tiled, mip-mapped maps for the runtime) to DIR
      --flexure-distance   distance inland in metres at which flexure reaches its full height (default 1500)
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

/// Plant community weights plus moisture and exposure, as bundle channels.
func vegetationChannels(_ v: Vegetation.Result) -> [(name: String, grid: Grid)] {
    var channels: [(name: String, grid: Grid)] = Vegetation.Community.allCases.map { (name: $0.channel, grid: v.weight($0)) }
    channels.append((name: "mai", grid: v.mai))
    channels.append((name: "exposure", grid: v.exposure))
    return channels
}

/// Share of land per plant community, rounded, for info.json.
func biomeShares(_ v: Vegetation.Result, height: Grid, seaLevel: Float) -> [String: Double] {
    var out = [String: Double]()
    for (c, share) in v.landShares(height: height, seaLevel: seaLevel) { out[c.name] = (share * 1000).rounded() / 1000 }
    return out
}

var params = Island.Params()
var outDir = "out"
var refineFactor = 0
var refineParams = Refine.Params()
var finePath: String? = nil

/// Refinement flags shared by the generator and import modes; false if `flag` isn't one.
func parseRefine(_ flag: String, _ value: String) -> Bool {
    switch flag {
    case "--refine": refineFactor = parsed(Int(value))
    case "--fine": finePath = value
    case "--detail": refineParams.detail = parsed(Double(value))
    case "--flute-start": refineParams.fluteStart = parsed(Double(value))
    case "--flute-full": refineParams.fluteFull = parsed(Double(value))
    case "--flute-base": refineParams.fluteBase = parsed(Double(value))
    case "--dissection-cap": refineParams.dissectionCap = parsed(Double(value))
    case "--planar-walls": refineParams.planarWalls = parsed(Double(value))
    case "--gentle": refineParams.gentle = parsed(Double(value))
    case "--gentle-start": refineParams.gentleStart = parsed(Double(value))
    case "--gentle-full": refineParams.gentleFull = parsed(Double(value))
    case "--gentle-across": refineParams.gentleAcross = parsed(Double(value))
    case "--gentle-along": refineParams.gentleAlong = parsed(Double(value))
    case "--gentle-roughness": refineParams.gentleRoughness = parsed(Double(value))
    case "--flute-across": refineParams.acrossWavelength = parsed(Double(value))
    case "--flute-along": refineParams.alongWavelength = parsed(Double(value))
    case "--roughness": refineParams.roughness = parsed(Double(value))
    case "--channel-start": refineParams.channelStart = parsed(Double(value))
    case "--channel-full": refineParams.channelFull = parsed(Double(value))
    default: return false
    }
    return true
}
var bundleDir: String? = nil
var timer: PhaseTimer?
var args = CommandLine.arguments.dropFirst()

// Scoring mode: realism metrics for any heightfield, such as a real DEM.
if args.first == "--score" {
    var path: String?, cellSize: Double?, width: Int?, seaLevel: Float = 0
    while let flag = args.popFirst() {
        guard let value = args.popFirst() else { usage() }
        switch flag {
        case "--score": path = value
        case "--cell": cellSize = parsed(Double(value))
        case "--width": width = parsed(Int(value))
        case "--sea-level": seaLevel = parsed(Float(value))
        default: usage()
        }
    }
    guard let path, let cellSize, let data = FileManager.default.contents(atPath: path) else { usage() }
    let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let w = width ?? Int(Double(values.count).squareRoot())
    guard w > 0, values.count % w == 0 else { print("size \(values.count) is not a multiple of width \(w)"); exit(1) }
    var grid = Grid(width: w, height: values.count / w, cellSize: cellSize)
    grid.values = values
    // A real DEM has flat floors, harbours and resampling noise that stop flow, which
    // would understate drainage density and Hack's exponent: fill pits first, as the
    // generator does, and report how many there were.
    let rawPits = Realism.measure(grid, seaLevel: seaLevel).pits
    PriorityFlood.fill(&grid, seaLevel: seaLevel)
    var scored = Realism.measure(grid, seaLevel: seaLevel).json
    scored["pitsBeforeFill"] = rawPits
    let json = try JSONSerialization.data(withJSONObject: scored, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: json, as: UTF8.self))
    exit(0)
}

// Import mode: a real DEM as an island bundle, to walk next to the generated islands.
if args.first == "--import" {
    var path: String?, cellSize: Double?, seaLevel: Float = 0, previewDir: String?
    while let flag = args.popFirst() {
        guard let value = args.popFirst() else { usage() }
        switch flag {
        case "--import": path = value
        case "--cell": cellSize = parsed(Double(value))
        case "--bundle": bundleDir = value
        case "--sea-level": seaLevel = parsed(Float(value))
        case "--out": previewDir = value
        default: if !parseRefine(flag, value) { usage() }
        }
    }
    guard let path, let cellSize, let bundleDir, let data = FileManager.default.contents(atPath: path) else { usage() }
    let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let w = Int(Double(values.count).squareRoot())
    guard w > 0, w * w == values.count else { print("\(path) is not a square float32 grid"); exit(1) }
    var grid = Grid(width: w, height: w, cellSize: cellSize)
    grid.values = values
    // Heights stay as surveyed (harbours and flat floors keep their pits), so the walk is
    // the real island. Rain comes from the same climate model as the generated islands so
    // moisture-driven materials compare like for like.
    let start = Date()
    let rain = Climate.precipitation(height: grid, seaLevel: seaLevel, params: params.climate)
    var height = grid
    if refineFactor > 1 {
        // Refining a coarse copy of a real DEM is how the refinement is calibrated: the
        // result is scored against the DEM at full resolution.
        var filled = grid
        PriorityFlood.fill(&filled, seaLevel: seaLevel)
        refineParams.factor = refineFactor
        height = Refine.refine(filled, drainageArea: Realism.drainageArea(filled, seaLevel: seaLevel),
                               seaLevel: seaLevel, params: refineParams)
        print(String(format: "refined to %d² at %.2f m in %.1f s", height.width, height.cellSize, Date().timeIntervalSince(start)))
        if let finePath { try Preview.writeRaw(height, to: URL(fileURLWithPath: finePath)) }
    }
    // Plant communities need no finer grid than a generated island's (~80 m), and the
    // cloud and drainage passes are much cheaper there.
    let coarse = Vegetation.downsample(grid, factor: Swift.max(1, Int((80 / cellSize).rounded())))
    let coarseRain = Climate.precipitation(height: coarse, seaLevel: seaLevel, params: params.climate)
    var cloudParams = Clouds.Params()
    cloudParams.windDirection = params.climate.windDirection
    let clouds = Clouds.coverage(height: coarse, seaLevel: seaLevel, params: cloudParams)
    var vegetationParams = Vegetation.Params()
    vegetationParams.windDirection = params.climate.windDirection
    let vegetation = Vegetation.bake(height: coarse, rain: coarseRain, clouds: clouds, seaLevel: seaLevel, params: vegetationParams)
    let channels: [(name: String, grid: Grid)] = [("height", height), ("rain", rain)] + vegetationChannels(vegetation)
    let bundle = try IslandBundle.write(channels, seaLevel: Double(seaLevel),
                                        to: URL(fileURLWithPath: bundleDir, isDirectory: true))
    let h = try bundle.channel("height")
    print(String(format: "wrote bundle %@: height %d² at %.2f m in %d levels, in %.1f s", bundleDir, h.size, h.cellMetres, h.levels, Date().timeIntervalSince(start)))
    let shares = biomeShares(vegetation, height: coarse, seaLevel: seaLevel)
    if let previewDir {
        let out = URL(fileURLWithPath: previewDir, isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try Preview.write(Preview.biomes(vegetation, height: coarse, seaLevel: seaLevel), to: out.appendingPathComponent("biomes.png"))
        try Preview.write(Preview.gray(coarseRain), to: out.appendingPathComponent("rain.png"))
        try JSONSerialization.data(withJSONObject: shares, options: [.prettyPrinted, .sortedKeys]).write(to: out.appendingPathComponent("biomes.json"))
    }
    print(String(decoding: try JSONSerialization.data(withJSONObject: shares, options: [.sortedKeys]), as: UTF8.self))
    exit(0)
}

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
    case "--land-exponent": params.landExponent = parsed(Double(value))
    case "--coast-slope": params.coastSlope = parsed(Double(value))
    case "--rift-descent": params.riftDescent = parsed(Double(value))
    case "--saddle-fill": params.saddleFill = parsed(Double(value))
    case "--pond-time": params.pondTime = parsed(Double(value))
    case "--pond-hardness": params.pondHardness = parsed(Float(value))
    case "--pond-share": params.pondShare = parsed(Double(value))
    case "--wave-retreat": params.erosion.waveRetreat = parsed(Double(value))
    case "--leeward-waves": params.erosion.leewardWaves = parsed(Double(value))
    case "--uplift": params.erosion.uplift = parsed(Double(value))
    case "--flexure": params.flexure = parsed(Double(value))
    case "--flexure-distance": params.flexureDistance = parsed(Double(value))
    case "--bundle": bundleDir = value
    case "--out": outDir = value
    default: if !parseRefine(flag, value) { usage() }
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

// Standing trade-wind cloud coverage for the renderer, for normal and strong trades
// (research/clouds.md §3). The runtime adds drifting cumulus and the daily cycle.
var cloudParams = Clouds.Params()
cloudParams.windDirection = params.climate.windDirection
var strongCloudParams = Clouds.Params.strongTrades
strongCloudParams.windDirection = params.climate.windDirection
let clouds = timer?.measure("clouds") { Clouds.coverage(height: island.height, seaLevel: sea, params: cloudParams) }
    ?? Clouds.coverage(height: island.height, seaLevel: sea, params: cloudParams)
let strongClouds = Clouds.coverage(height: island.height, seaLevel: sea, params: strongCloudParams)
try Preview.writeRaw(clouds, to: out.appendingPathComponent("clouds-trades.r32"))
try Preview.writeRaw(strongClouds, to: out.appendingPathComponent("clouds-strong.r32"))
try Preview.write(Preview.clouds(clouds, over: Preview.hillshade(island.height, seaLevel: sea)), to: out.appendingPathComponent("clouds.png"))
let cloudInfo: [String: Any] = [
    "size": clouds.width,
    "cellMetres": clouds.cellSize,
    "windDirection": params.climate.windDirection,
    "regimes": [
        "trades": ["file": "clouds-trades.r32", "windSpeed": cloudParams.windSpeed],
        "strong": ["file": "clouds-strong.r32", "windSpeed": strongCloudParams.windSpeed],
    ],
]
try JSONSerialization.data(withJSONObject: cloudInfo, options: [.prettyPrinted, .sortedKeys]).write(to: out.appendingPathComponent("clouds.json"))
let cloudStats = Clouds.stats(coverage: clouds, height: island.height, seaLevel: sea, windDirection: params.climate.windDirection)

// Native plant communities (research/vegetation.md §5).
var vegetationParams = Vegetation.Params()
vegetationParams.windDirection = params.climate.windDirection
vegetationParams.seed = params.seed
let vegetation = timer?.measure("vegetation") { Vegetation.bake(height: island.height, rain: island.rain, clouds: clouds, seaLevel: sea, params: vegetationParams) }
    ?? Vegetation.bake(height: island.height, rain: island.rain, clouds: clouds, seaLevel: sea, params: vegetationParams)
try Preview.write(Preview.biomes(vegetation, height: island.height, seaLevel: sea), to: out.appendingPathComponent("biomes.png"))

// Ground colour from the plant communities, for the ground views.
var communityWeights = [String: Grid]()
for c in Vegetation.Community.allCases { communityWeights[c.channel] = vegetation.weight(c) }
let canopy = Look.canopyMap(weights: communityWeights, domain: params.domainSize)

// Ground-level pictures from a walker's eye height, rendered on the CPU.
let lighting = timer?.measure("lighting") { TerrainLighting.bake(height: island.height, seaLevel: sea) }
    ?? TerrainLighting.bake(height: island.height, seaLevel: sea)
if let timer { print("time by phase (\(ProcessInfo.processInfo.activeProcessorCount) cores):\n" + timer.report) }
for view in Viewpoints.automatic(height: island.height, drainageArea: island.drainageArea, seaLevel: sea, scarp: island.scarp) {
    let image = GroundView.render(island.height, camera: view.camera, seaLevel: sea, lighting: lighting, canopy: canopy)
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
    "landExponent": params.landExponent,
    "coastSlope": params.coastSlope,
    "riftDescent": params.riftDescent,
    "saddleFill": params.saddleFill,
    "pondTime": params.pondTime,
    "pondHardness": params.pondHardness,
    "pondShare": params.pondShare,
    "waveRetreat": params.erosion.waveRetreat,
    "leewardWaves": params.erosion.leewardWaves,
    "uplift": params.erosion.uplift,
    "flexure": params.flexure,
    "flexureDistance": params.flexureDistance,
    "landAreaKm2": (landKm2 * 10).rounded() / 10,
    "maxHeightMetres": Double(island.height.range.max),
    "steepFraction": rounded(Double(steep) / Double(Swift.max(land, 1)), 10000),
    "lowlandFraction": rounded(Double(lowland) / Double(Swift.max(land, 1))),
    "meanSlopeWindward": rounded(slopeSum.windward / Double(Swift.max(slopeCount.windward, 1))),
    "meanSlopeLeeward": rounded(slopeSum.leeward / Double(Swift.max(slopeCount.leeward, 1))),
    "seconds": (elapsed * 10).rounded() / 10,
    "realism": Realism.measure(island.height, seaLevel: sea).json,
    "cloudLand": rounded(cloudStats.land),
    "cloudWindward": rounded(cloudStats.windward),
    "cloudLeeward": rounded(cloudStats.leeward),
    "biomes": biomeShares(vegetation, height: island.height, seaLevel: sea),
]
let json = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
try json.write(to: out.appendingPathComponent("info.json"))

// Final refinement to ~10 m (plan §3.8) and the island bundle for the runtime.
var bundleHeight = island.height
if refineFactor > 1 {
    refineParams.factor = refineFactor
    refineParams.seed = params.seed
    let refineStart = Date()
    let fine = Refine.refine(island.height, drainageArea: island.drainageArea, seaLevel: sea, params: refineParams)
    print(String(format: "refined to %d² at %.1f m in %.1f s", fine.width, fine.cellSize, Date().timeIntervalSince(refineStart)))
    try Preview.writeRaw(fine, to: out.appendingPathComponent("height-fine.r32"))
    bundleHeight = fine
}
if let bundleDir {
    let bundleStart = Date()
    let channels: [(name: String, grid: Grid)] = [
        ("height", bundleHeight),
        ("rain", island.rain),
        ("drainage", island.drainageArea),
        ("hardness", island.hardness),
        ("clouds-trades", clouds),
    ] + vegetationChannels(vegetation)
    let bundle = try IslandBundle.write(channels, seaLevel: Double(sea), to: URL(fileURLWithPath: bundleDir, isDirectory: true))
    let h = try bundle.channel("height")
    print(String(format: "wrote bundle %@: height %d² in %d levels, in %.1f s", bundleDir, h.size, h.levels, Date().timeIntervalSince(bundleStart)))
}
print(String(decoding: json, as: UTF8.self))
