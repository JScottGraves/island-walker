import Foundation
import IslandCore

/// The full generation pipeline from a seed to an eroded island.
public enum Island {
    public struct Params: Sendable {
        public var seed: UInt64 = 1
        /// Samples per side. 512 for quick looks, 2048 for review, 8192 for the final bake.
        public var size: Int = 1024
        /// Domain edge length in metres (80 km comfortably holds an Oahu-sized island).
        public var domainSize: Double = 80_000
        /// Geological time steps of erosion.
        public var steps: Int = 400
        /// Recompute rainfall from the evolving terrain every this many steps.
        public var rainInterval: Int = 50
        public var erosion = StreamPower.Params()
        public var climate = Climate.Params()
        /// Cut a giant landslide sector from the windward flank before erosion (see FlankCollapse).
        public var flankCollapse = true
        /// Shield land profile exponent (see VolcanicBase.generate); 0 is the classic shape.
        public var landExponent: Double = 2
        /// Straight-cone share of the land profile and rift crest drop (see VolcanicBase.generate).
        /// 0.3 and 0.4 score closest to Oahu with wave retreat on (realism sweep, 2026-10-16).
        public var coastSlope: Double = 0.3
        public var riftDescent: Double = 0.4
        /// Lava ponding in the saddles between shields (see VolcanicBase.generate): the
        /// soft-maximum width in metres, and the share of the erosion run after which the
        /// lava floods in. Ponded at the start, the fill was just more flank and eroded into
        /// ravines; Oahu's Schofield plateau is young Koolau lava banked against the older
        /// Waianae range, too young and permeable to be dissected much. The ponded lava is
        /// also `pondHardness` times harder to erode.
        ///
        /// Flooding all the way up to the uneroded saddle surface built a 200 m tableland
        /// over a third of the island. Instead the lava surface keeps the saddle's gentle
        /// shape but is lowered so that only the deepest `pondShare` of the saddle (by how
        /// far erosion has cut below that shape) is buried; ridges still stand above it.
        public var saddleFill: Double = 0
        public var pondTime: Double = 0.9
        public var pondHardness: Float = 2
        public var pondShare: Double = 0.5
        /// Flexural uplift after erosion, in metres, reached `flexureDistance` inland and
        /// held beyond it. Oahu rides the bulge raised by the Big Island's weight, which has
        /// lifted its old reefs and lowland a few tens of metres; tilting the coastal fringe
        /// this way gives the plains that erosion grades flat to sea level a gentle seaward
        /// slope instead. 20 m brings the 600 m coastal belt to about 4.2° (Oahu 4.8°) on
        /// seeds 7, 42 and 123. 0 disables it.
        public var flexure: Double = 20
        public var flexureDistance: Double = 1500
        public init() {}
    }

    public struct Result {
        public var height: Grid
        public var hardness: Grid
        public var rain: Grid
        public var drainageArea: Grid
        public var params: Params
        /// The flank collapse's headscarp, if one was cut.
        public var scarp: FlankCollapse.Scarp?
    }

    /// Generates an island. `progress` is called with values in 0...1; `timer`, if given,
    /// collects time per pipeline phase.
    public static func generate(_ p: Params, timer: PhaseTimer? = nil, progress: ((Double) -> Void)? = nil) -> Result {
        func phase<T>(_ name: String, _ body: () -> T) -> T { timer?.measure(name, body) ?? body() }
        var erosion = p.erosion
        erosion.seed = Hash.mix(p.seed ^ 0x524F_5554_45)
        erosion.windDirection = p.climate.windDirection
        // Keep diffusion stable at coarse and fine resolutions alike.
        let cell = p.domainSize / Double(p.size)
        let maxDt = 0.2 * cell * cell / Swift.max(erosion.diffusivity, 1e-9)
        erosion.timeStep = Swift.min(erosion.timeStep, maxDt)

        var base = phase("volcano") { VolcanicBase.generate(seed: p.seed, size: p.size, domainSize: p.domainSize, landExponent: p.landExponent,
                                                                coastSlope: p.coastSlope, riftDescent: p.riftDescent, saddleFill: p.saddleFill) }
        let scarp = p.flankCollapse
            ? FlankCollapse.choose(shields: base.shields, height: base.height, windDirection: p.climate.windDirection, seed: p.seed)
            : nil
        if let scarp {
            phase("collapse") { FlankCollapse.apply(scarp, height: &base.height, hardness: &base.hardness, seed: p.seed) }
        }
        var height = base.height
        var rain = phase("rain") { Climate.precipitation(height: height, seaLevel: erosion.seaLevel, params: p.climate) }
        var area = Grid(width: p.size, height: p.size, cellSize: cell)

        // Random routing makes any single step's drainage area noisy, so the reported
        // area is the mean over the last few steps.
        let averaged = Swift.min(20, p.steps)
        let pondStep = p.saddleFill > 0 ? Int((p.pondTime * Double(p.steps)).rounded()) : -1
        for s in 0..<p.steps {
            if s == pondStep {
                phase("ponding") { Self.pond(&height, hardness: &base.hardness, surface: base.ponded, p: p) }
            }
            if s > 0 && s % p.rainInterval == 0 {
                rain = phase("rain") { Climate.precipitation(height: height, seaLevel: erosion.seaLevel, params: p.climate) }
            }
            let stepArea = StreamPower.step(&height, rain: rain, hardness: base.hardness, params: erosion, stepIndex: s, timer: timer).area
            if s >= p.steps - averaged {
                for i in 0..<area.count { area.values[i] += stepArea.values[i] / Float(averaged) }
            }
            progress?(Double(s + 1) / Double(p.steps))
        }
        if p.flexure > 0 {
            phase("flexure") { Self.flex(&height, seaLevel: erosion.seaLevel, lift: p.flexure, distance: p.flexureDistance) }
        }
        if p.steps == 0 || p.flexure > 0 {
            PriorityFlood.fill(&height, seaLevel: erosion.seaLevel)
        }
        return Result(height: height, hardness: base.hardness, rain: rain, drainageArea: area, params: p, scarp: scarp)
    }

    /// Raises land by `lift` × min(1, distance to the sea / `distance`), using a chamfer
    /// distance transform from the sea cells.
    static func flex(_ h: inout Grid, seaLevel: Float, lift: Double, distance: Double) {
        let w = h.width, rows = h.height
        var d = [Double](repeating: .infinity, count: h.count)
        for i in 0..<h.count where h.values[i] <= seaLevel { d[i] = 0 }
        let diagonal = 2.0.squareRoot()
        for _ in 0..<2 {
            for y in 0..<rows {
                for x in 0..<w {
                    let i = y * w + x
                    if x > 0 { d[i] = Swift.min(d[i], d[i - 1] + 1) }
                    if y > 0 {
                        d[i] = Swift.min(d[i], d[i - w] + 1)
                        if x > 0 { d[i] = Swift.min(d[i], d[i - w - 1] + diagonal) }
                        if x < w - 1 { d[i] = Swift.min(d[i], d[i - w + 1] + diagonal) }
                    }
                }
            }
            for y in stride(from: rows - 1, through: 0, by: -1) {
                for x in stride(from: w - 1, through: 0, by: -1) {
                    let i = y * w + x
                    if x < w - 1 { d[i] = Swift.min(d[i], d[i + 1] + 1) }
                    if y < rows - 1 {
                        d[i] = Swift.min(d[i], d[i + w] + 1)
                        if x < w - 1 { d[i] = Swift.min(d[i], d[i + w + 1] + diagonal) }
                        if x > 0 { d[i] = Swift.min(d[i], d[i + w - 1] + diagonal) }
                    }
                }
            }
        }
        for i in 0..<h.count where h.values[i] > seaLevel && d[i].isFinite {
            h.values[i] += Float(lift * Swift.min(1, d[i] * h.cellSize / distance))
        }
    }

    /// Floods the saddle with lava: `surface` (the uneroded saddle) lowered by the
    /// `pondShare` quantile of how far erosion has cut below it, so that share of the
    /// saddle is buried under a gently sloping plain and the rest stands above it.
    static func pond(_ h: inout Grid, hardness: inout Grid, surface: Grid, p: Params) {
        var cut: [Float] = []
        for i in 0..<h.count where surface.values[i].isFinite && h.values[i] > 0 {
            cut.append(surface.values[i] - h.values[i])
        }
        guard !cut.isEmpty else { return }
        cut.sort()
        let k = Swift.min(cut.count - 1, Swift.max(0, Int(Double(cut.count) * (1 - p.pondShare))))
        let lowering = Swift.max(0, cut[k])
        for i in 0..<h.count where surface.values[i].isFinite {
            let lava = surface.values[i] - lowering
            if lava > h.values[i] {
                h.values[i] = lava
                hardness.values[i] *= p.pondHardness
            }
        }
    }
}
