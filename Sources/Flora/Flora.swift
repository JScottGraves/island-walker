import IslandCore

/// Native trees and shrubs for the renderer (docs/flora.md): procedural meshes for each
/// plant, four variants each at two levels of detail. Pure CPU and deterministic: the same
/// plant, variant and LOD always give the same mesh.
public enum Flora {
    public static let variants = 4
    public static let lods = 2

    /// Seed of one variant (fixed, so meshes match between runs and machines).
    public static func seed(_ plant: Plant, variant: Int) -> UInt64 {
        Hash.mix(0xF10A_0000 &+ UInt64(plant.rawValue) &* 0x9E37 &+ UInt64(variant) &* 0x7F4A_7C15)
    }

    /// The mesh of one variant at one LOD (0 full, 1 about 15% of the triangles).
    public static func mesh(_ plant: Plant, variant: Int, lod: Int) -> PlantMesh {
        let seed = seed(plant, variant: variant)
        let h = Float(plant.referenceHeight)
        switch plant {
        case .ohia:
            return BranchingTree(
                height: h, stems: 1...3, stemLean: 0.35, trunkRadius: 0.32, forkHeight: 0.22...0.42,
                children: 2...3, branchAngle: 0.45...0.95, reach: 0.42...0.62, orders: 4, segments: 5,
                crooked: 0.32, upward: 0.05, crownCentre: 0.66, crownRadii: SIMD3(0.33, 0.33, 0.3),
                card: 0.42, cardsPerTip: 4...6, leaves: .ohiaLeaves, flowers: .lehua, flowerChance: 0.22,
                bark: .ohiaBark, sides: [9, 6, 5, 4, 3], sides1: [5, 4, 3]).build(seed: seed, lod: lod)
        case .ohiaDwarf:
            return BranchingTree(
                height: h, stems: 2...4, stemLean: 0.6, trunkRadius: 0.05, forkHeight: 0.1...0.25,
                children: 2...3, branchAngle: 0.5...1.0, reach: 0.45...0.7, orders: 3, segments: 4,
                crooked: 0.35, upward: 0.04, crownCentre: 0.55, crownRadii: SIMD3(0.5, 0.5, 0.42),
                card: 0.16, cardsPerTip: 4...6, leaves: .ohiaLeaves, flowers: .lehua, flowerChance: 0.3,
                bark: .ohiaBark, sides: [5, 4, 3, 3], sides1: [3, 3, 3], lod1Order: 1, barkRepeat: 0.4).build(seed: seed, lod: lod)
        case .koa:
            return BranchingTree(
                height: h, stems: 1...2, stemLean: 0.3, trunkRadius: 0.55, forkHeight: 0.18...0.3,
                children: 2...3, branchAngle: 0.6...1.05, reach: 0.45...0.65, orders: 4, segments: 5,
                crooked: 0.2, upward: 0.03, crownCentre: 0.66, crownRadii: SIMD3(0.46, 0.46, 0.26),
                card: 0.55, cardsPerTip: 4...6, leaves: .koaPhyllodes, bark: .koaBark,
                sides: [10, 7, 5, 4, 3], sides1: [6, 4, 3], barkRepeat: 2.0).build(seed: seed, lod: lod)
        case .aalii:
            return BranchingTree(
                height: h, stems: 1...3, stemLean: 0.4, trunkRadius: 0.08, forkHeight: 0.15...0.3,
                children: 2...3, branchAngle: 0.4...0.85, reach: 0.45...0.65, orders: 3, segments: 4,
                crooked: 0.22, upward: 0.06, crownCentre: 0.6, crownRadii: SIMD3(0.32, 0.32, 0.36),
                card: 0.26, cardsPerTip: 4...6, leaves: .aaliiLeaves, bark: .aaliiBark,
                sides: [6, 4, 3, 3], sides1: [4, 3, 3], lod1Order: 1, barkRepeat: 0.8).build(seed: seed, lod: lod)
        case .loulu:
            return Palm(height: h).build(seed: seed, lod: lod)
        case .hapuu:
            return TreeFern(height: h).build(seed: seed, lod: lod)
        case .naupaka:
            return Shrub(height: h, width: h * 1.8).build(seed: seed, lod: lod)
        }
    }

    /// Every plant's meshes: `[plant][variant][lod]`.
    public static func library() -> [[[PlantMesh]]] {
        Plant.allCases.map { plant in
            (0..<variants).map { v in (0..<lods).map { mesh(plant, variant: v, lod: $0) } }
        }
    }
}
