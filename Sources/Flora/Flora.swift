import IslandCore
import simd

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
                card: 0.5, cardsPerTip: 5...8, leaves: .ohiaLeaves, flowers: .lehua, flowerChance: 0.22,
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
                crooked: 0.2, upward: 0.03, crownCentre: 0.68, crownRadii: SIMD3(0.55, 0.55, 0.24),
                card: 0.75, cardsPerTip: 6...9, leaves: .koaPhyllodes, bark: .koaBark,
                sides: [10, 7, 5, 4, 3], sides1: [6, 4, 3], barkRepeat: 2.0).build(seed: seed, lod: lod)
        case .aalii:
            return BranchingTree(
                height: h, stems: 1...3, stemLean: 0.4, trunkRadius: 0.08, forkHeight: 0.15...0.3,
                children: 2...3, branchAngle: 0.4...0.85, reach: 0.45...0.65, orders: 3, segments: 4,
                crooked: 0.22, upward: 0.06, crownCentre: 0.6, crownRadii: SIMD3(0.32, 0.32, 0.36),
                card: 0.3, cardsPerTip: 4...6, leaves: .aaliiLeaves, bark: .aaliiBark,
                sides: [6, 4, 3, 3], sides1: [4, 3, 3], lod1Order: 1, barkRepeat: 0.8).build(seed: seed, lod: lod)
        case .loulu:
            return Palm(height: h).build(seed: seed, lod: lod)
        case .hapuu:
            return TreeFern(height: h).build(seed: seed, lod: lod)
        case .naupaka:
            return Shrub(height: h, width: h * 1.8).build(seed: seed, lod: lod)
        }
    }

    /// A 1.7 m standing figure for scale on specimen sheets.
    public static func figure() -> PlantMesh {
        var b = MeshBuilder(height: 1.7, crown: Crown(centre: SIMD3(0, 0, 1), radii: SIMD3(1, 1, 1)))
        func limb(_ points: [SIMD3<Float>], _ radii: [Float]) {
            let rings = zip(points, radii).enumerated().map { k, pr -> Ring in
                let next = points[Swift.min(k + 1, points.count - 1)], prev = points[Swift.max(k - 1, 0)]
                let d = simd_normalize(next - prev)
                return Ring(position: pr.0, direction: d, side: perpendicular(d), radius: pr.1)
            }
            b.tube(rings, sides: 8, material: .figure, repeatLength: 1)
        }
        for s: Float in [-1, 1] {
            limb([SIMD3(0.1 * s, 0, -0.05), SIMD3(0.1 * s, 0, 0.45), SIMD3(0.11 * s, 0, 0.9)], [0.06, 0.065, 0.08])
            limb([SIMD3(0.2 * s, 0, 1.42), SIMD3(0.24 * s, 0, 1.15), SIMD3(0.25 * s, 0.02, 0.85)], [0.045, 0.04, 0.035])
        }
        limb([SIMD3(0, 0, 0.85), SIMD3(0, 0, 1.15), SIMD3(0, 0, 1.45)], [0.17, 0.15, 0.19])
        limb([SIMD3(0, 0, 1.44), SIMD3(0, 0, 1.5), SIMD3(0, 0, 1.6), SIMD3(0, 0, 1.7)], [0.05, 0.1, 0.1, 0.02])
        for i in b.mesh.vertices.indices { b.mesh.vertices[i].sway = 0 }
        b.finish()
        return b.mesh
    }

    /// Every plant's meshes: `[plant][variant][lod]`.
    public static func library() -> [[[PlantMesh]]] {
        Plant.allCases.map { plant in
            (0..<variants).map { v in (0..<lods).map { mesh(plant, variant: v, lod: $0) } }
        }
    }
}
