import Foundation

/// Whether terrain hides points from the eye, from a coarse heightfield (the overview the
/// renderer keeps): for culling sea nodes behind the mountains, which would otherwise cost
/// their vertices twice a frame. Conservative: a ray counts as blocked only where the terrain
/// stands `margin` metres above it (the coarse grid smooths over notches the drawn terrain
/// has), and only if every point's ray is blocked.
public struct TerrainOcclusion: Sendable {
    let height: Grid
    public var margin: Double = 30

    public init(height: Grid) {
        self.height = height
    }

    /// True when terrain blocks the line of sight from `eye` to every one of `points`.
    public func hides(eye: SIMD3<Double>, points: [SIMD3<Double>]) -> Bool {
        for p in points where !blocked(eye, p) { return false }
        return !points.isEmpty
    }

    /// Marches the ray a cell at a time, from two cells out (the camera's own ground) to the
    /// target, over the part of it inside the grid.
    func blocked(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Bool {
        let cell = height.cellSize
        let dx = b.x - a.x, dy = b.y - a.y
        let length = (dx * dx + dy * dy).squareRoot()
        let steps = Int(length / cell)
        guard steps > 2 else { return false }
        let w = Double(height.width), h = Double(height.height)
        for k in 2..<steps {
            let t = Double(k) / Double(steps)
            let i = (a.x + dx * t) / cell, j = (a.y + dy * t) / cell
            if i < 0 || j < 0 || i >= w || j >= h { continue }
            let ground = Double(height[Int(i), Int(j)])
            let ray = a.z + (b.z - a.z) * t
            if ground - margin > ray { return true }
        }
        return false
    }
}
