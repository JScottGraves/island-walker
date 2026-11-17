import Foundation
import IslandCore

/// Everything a structure type may look at when it chooses its sites. Grids may have
/// different resolutions but all span the same square domain; read them with `at`.
public struct SiteContext {
    /// The heights the bundle will carry (10 m after refinement, else the bake's grid).
    public var fine: Grid
    /// The bake's ~78 m heights, for fast first-pass scoring.
    public var coarse: Grid
    public var seaLevel: Float
    /// Upstream drainage area, m², on any grid over the domain.
    public var drainage: Grid?
    /// Standing trade-wind cloud coverage 0…1.
    public var clouds: Grid?
    /// Plant community weights, in `Vegetation.Community.allCases` order.
    public var communities: [Grid]
    /// The direction the trade wind blows toward, radians from +x counter-clockwise
    /// (Climate.Params.windDirection: 5π/4, from the north-east).
    public var windDirection: Double
    /// Distance to the sea, metres, on the coarse grid.
    public var seaDistance: Grid
    /// Seeds per-island choices (ties, variants).
    public var seed: UInt64
    /// Structures already placed, which later ones must keep clear of.
    public var placed: [Structure] = []

    public var domain: Double { Double(coarse.width) * coarse.cellSize }

    public init(fine: Grid, coarse: Grid, seaLevel: Float, drainage: Grid?, clouds: Grid?, communities: [Grid],
                windDirection: Double, seed: UInt64) {
        self.fine = fine
        self.coarse = coarse
        self.seaLevel = seaLevel
        self.drainage = drainage
        self.clouds = clouds
        self.communities = communities
        self.windDirection = windDirection
        self.seed = seed
        let d = Island.seaDistance(coarse, seaLevel: seaLevel)
        var g = Grid(width: coarse.width, height: coarse.height, cellSize: coarse.cellSize)
        for i in 0..<g.count { g.values[i] = Float(d[i].isFinite ? d[i] * coarse.cellSize : 1e7) }
        seaDistance = g
    }

    /// Bilinear value of `g` at world position (x, y), metres.
    @inline(__always)
    public func at(_ g: Grid, _ x: Double, _ y: Double) -> Float {
        let scale = Double(g.width) / domain
        return g.sample(x: x * scale - 0.5, y: y * scale - 0.5)
    }

    /// Weight of one plant community at (x, y), or 0 when the bake has none.
    public func community(_ c: Vegetation.Community, _ x: Double, _ y: Double) -> Float {
        c.rawValue < communities.count ? at(communities[c.rawValue], x, y) : 0
    }

    /// Whether a footprint of `radius` metres at (x, y) keeps `margin` clear of every placed structure.
    public func isClear(x: Double, y: Double, radius: Double, margin: Double = 100) -> Bool {
        placed.allSatisfy { s in
            let dx = x - s.x, dy = y - s.y, r = radius + s.radius + s.embankment + margin
            return dx * dx + dy * dy > r * r
        }
    }
}

/// One type of structure: how many to place, where, and what it does to the ground.
/// Adding a type is one conforming file plus an entry in `SitePlanner.standardKinds`.
public protocol StructureKind {
    /// The `kind` written on its structures.
    var name: String { get }
    /// Placed in ascending priority; later types keep clear of earlier ones.
    var priority: Int { get }
    /// Chooses this type's sites. Must not overlap `context.placed`.
    func place(in context: SiteContext) -> [Structure]
}

public enum SitePlanner {
    /// The structure types every island gets, in any order (sorted by priority).
    public static var standardKinds: [StructureKind] { [Airstrip()] }

    /// Places every kind in priority order and grades the ground under them into `context.fine`.
    /// Returns the structures and the graded heights.
    public static func plan(_ context: SiteContext, kinds: [StructureKind] = standardKinds,
                            log: (String) -> Void = { print($0) }) -> (structures: [Structure], height: Grid) {
        var context = context
        var counts = [String: Int]()
        for kind in kinds.sorted(by: { $0.priority < $1.priority }) {
            let start = Date()
            var placed = kind.place(in: context)
            for i in placed.indices {
                counts[kind.name, default: 0] += 1
                placed[i].id = "\(kind.name)-\(counts[kind.name]!)"
                Grading.apply(&placed[i], to: &context.fine, seaLevel: context.seaLevel)
                // Keep the coarse heights in step, so later kinds score the graded ground.
                Grading.apply(placed[i], toCoarse: &context.coarse)
            }
            context.placed += placed
            log(String(format: "structures: %d %@ in %.1f s", placed.count, kind.name, Date().timeIntervalSince(start)))
        }
        return (context.placed, context.fine)
    }
}

/// Cuts and fills the ground to a structure's grade line.
public enum Grading {
    /// Side slope of cuts and fills: 1 m of height per `run` metres outward.
    public static let run = 3.0

    /// Grades `height` under `s`'s graded zones: exactly the grade line inside them, then a
    /// 1:3 cut or fill slope down (or up) to the natural ground, its toe rounded off. Sets
    /// `s.embankment` to how far out the slope reaches.
    public static func apply(_ s: inout Structure, to height: inout Grid, seaLevel: Float) {
        let hl = s.gradedHalfLength, hw = s.gradedHalfWidth
        guard hl > 0, hw > 0 else { return }
        let cell = height.cellSize
        // The deepest cut or fill sets how far out the slopes can reach.
        let reachLimit = 60.0 * run
        let r = (hl + reachLimit) * (hl + reachLimit) + (hw + reachLimit) * (hw + reachLimit)
        let reach = r.squareRoot()
        let x0 = Swift.max(0, Int((s.x - reach) / cell)), x1 = Swift.min(height.width - 1, Int((s.x + reach) / cell) + 1)
        let y0 = Swift.max(0, Int((s.y - reach) / cell)), y1 = Swift.min(height.height - 1, Int((s.y + reach) / cell) + 1)
        guard x0 <= x1, y0 <= y1 else { return }
        var out = height
        var embankment = 0.0
        let soft = 1.5  // metres of height over which the toe rounds into the natural slope
        for j in y0...y1 {
            for i in x0...x1 {
                let px = (Double(i) + 0.5) * cell, py = (Double(j) + 0.5) * cell
                let (u, v) = s.local(x: px, y: py)
                let du = Swift.max(abs(u) - hl, 0), dv = Swift.max(abs(v) - hw, 0)
                let d = (du * du + dv * dv).squareRoot()
                if d > reachLimit { continue }
                let target = s.gradeHeight(u: u)
                let natural = Double(height[i, j])
                let diff = natural - target
                let limit = d / run
                // Soft clamp of diff to ±limit: the surface leaves the grade at 1:3 and
                // eases into the natural ground instead of meeting it at a kink.
                let graded: Double
                if d == 0 {
                    graded = target
                } else {
                    graded = target + (diff < 0 ? -1 : 1) * softMin(abs(diff), limit, knee: Swift.min(soft, limit))
                }
                if abs(graded - natural) > 0.05 { embankment = Swift.max(embankment, d) }
                out[i, j] = Float(graded)
            }
        }
        height = out
        s.embankment = Swift.max(s.embankment, (embankment + 5).rounded())
    }

    /// min(a, limit) with the corner rounded over ±knee (a, limit ≥ 0): smooth in a.
    static func softMin(_ a: Double, _ limit: Double, knee k: Double) -> Double {
        if k <= 0 || a <= limit - k { return Swift.min(a, limit) }
        if a >= limit + k { return limit }
        let t = a - (limit - k)
        return a - t * t / (4 * k)
    }

    /// The same grading on a coarse grid, cells whose centre falls in the graded zones only
    /// (enough for later structure types to see the ground as flat there).
    public static func apply(_ s: Structure, toCoarse height: inout Grid) {
        let hl = s.gradedHalfLength, hw = s.gradedHalfWidth
        guard hl > 0, hw > 0 else { return }
        let cell = height.cellSize, reach = s.radius
        let x0 = Swift.max(0, Int((s.x - reach) / cell)), x1 = Swift.min(height.width - 1, Int((s.x + reach) / cell) + 1)
        let y0 = Swift.max(0, Int((s.y - reach) / cell)), y1 = Swift.min(height.height - 1, Int((s.y + reach) / cell) + 1)
        guard x0 <= x1, y0 <= y1 else { return }
        for j in y0...y1 {
            for i in x0...x1 {
                let (u, v) = s.local(x: (Double(i) + 0.5) * cell, y: (Double(j) + 0.5) * cell)
                if abs(u) <= hl && abs(v) <= hw { height[i, j] = Float(s.gradeHeight(u: u)) }
            }
        }
    }
}
