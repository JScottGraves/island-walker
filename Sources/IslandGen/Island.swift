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
        public init() {}
    }

    public struct Result {
        public var height: Grid
        public var hardness: Grid
        public var rain: Grid
        public var drainageArea: Grid
        public var params: Params
    }

    /// Generates an island. `progress` is called with values in 0...1.
    public static func generate(_ p: Params, progress: ((Double) -> Void)? = nil) -> Result {
        var erosion = p.erosion
        erosion.seed = Hash.mix(p.seed ^ 0x524F_5554_45)
        // Keep diffusion stable at coarse and fine resolutions alike.
        let cell = p.domainSize / Double(p.size)
        let maxDt = 0.2 * cell * cell / Swift.max(erosion.diffusivity, 1e-9)
        erosion.timeStep = Swift.min(erosion.timeStep, maxDt)

        let base = VolcanicBase.generate(seed: p.seed, size: p.size, domainSize: p.domainSize)
        var height = base.height
        var rain = Climate.precipitation(height: height, seaLevel: erosion.seaLevel, params: p.climate)
        var area = Grid(width: p.size, height: p.size, cellSize: cell)

        // Random routing makes any single step's drainage area noisy, so the reported
        // area is the mean over the last few steps.
        let averaged = Swift.min(20, p.steps)
        for s in 0..<p.steps {
            if s > 0 && s % p.rainInterval == 0 {
                rain = Climate.precipitation(height: height, seaLevel: erosion.seaLevel, params: p.climate)
            }
            let stepArea = StreamPower.step(&height, rain: rain, hardness: base.hardness, params: erosion, stepIndex: s).area
            if s >= p.steps - averaged {
                for i in 0..<area.count { area.values[i] += stepArea.values[i] / Float(averaged) }
            }
            progress?(Double(s + 1) / Double(p.steps))
        }
        if p.steps == 0 {
            PriorityFlood.fill(&height, seaLevel: erosion.seaLevel)
        }
        return Result(height: height, hardness: base.hardness, rain: rain, drainageArea: area, params: p)
    }
}
