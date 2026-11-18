import Foundation
import IslandCore

/// Exactly one paved airstrip per island: a runway roughly along the trade wind, on low,
/// gentle ground that needs the least earthwork, with clear approaches past both ends.
///
/// Search: every `spacing` metres and `headingStep` degrees, scored on the coarse grid;
/// the best `refineCount` distinct sites are then shifted and turned a little and scored
/// on the fine grid. When no site passes the hard rules the rules relax rung by rung
/// (`ladder`), so every island with land gets its airstrip.
///
/// Beside the runway it lays a concrete apron just past the grass strip, joined to the
/// runway by a short taxiway across the strip, on whichever side and at whichever point
/// along the runway needs the least earthwork. Both follow the runway's grade line, so
/// grading them leaves the runway and its strip as they were. They are structures of their
/// own (kinds "apron" and "taxiway", every zone a box around its own centre).
public struct Airstrip: StructureKind {
    public struct Params: Sendable {
        /// A small island field's runway (Hāna, on Maui, is about 1,100 m).
        public var runwayWidth = 23.0
        /// Graded grass strip: this wide, and this far past each runway end.
        public var stripWidth = 75.0
        public var stripOverrun = 60.0
        /// Steepest grade line along the runway (rise per metre).
        public var maxGrade = 0.015
        /// The whole strip at least this far above the sea, metres.
        public var minAboveSea = 3.0
        /// Streams draining more than this (m²) may not cross the strip.
        public var maxStreamArea = 5e6
        /// Obstacle surface past each end: rising 1 in 20 out to 3 km.
        public var approachSlope = 0.05
        public var approachLength = 3000.0
        public var spacing = 150.0
        public var headingStep = 5.0
        public var refineCount = 50
        /// Runway length (m) and deepest allowed cut or fill (m), tried in order.
        public var ladder: [(length: Double, maxCut: Double)] = [(1100, 8), (1000, 8), (900, 12), (800, 16)]
        /// Concrete apron beside the strip: this long along the runway and this deep away from it.
        public var apronLength = 150.0
        public var apronDepth = 90.0
        /// Taxiway from the runway's edge to the apron.
        public var taxiwayWidth = 15.0
        /// Deepest cut or fill the apron may need; past it there is no apron.
        public var apronMaxCut = 10.0
        public init() {}
    }

    public var params = Params()
    public var name: String { "airstrip" }
    public var priority: Int { 0 }

    public init(params: Params = Params()) { self.params = params }

    /// A scored site.
    public struct Site: Sendable {
        public var x: Double, y: Double, heading: Double
        public var score: Double
        public var elevation: Double, slope: Double
        public var meanCut: Double, maxCut: Double
        public var crosswind: Double
    }

    struct Rules {
        var length: Double
        var maxCut: Double
        var minAboveSea: Double
        var maxStreamArea: Double
    }

    public func place(in context: SiteContext) -> [Structure] {
        var rungs = params.ladder.map { Rules(length: $0.length, maxCut: $0.maxCut, minAboveSea: params.minAboveSea,
                                              maxStreamArea: params.maxStreamArea) }
        // Last resort: anywhere on land.
        rungs.append(Rules(length: 600, maxCut: 1000, minAboveSea: 0.5, maxStreamArea: .infinity))
        for rules in rungs {
            if let site = search(context, rules: rules) {
                let runway = structure(site, length: rules.length)
                return [runway] + apron(beside: runway, context)
            }
        }
        print("structures: no land for an airstrip")
        return []
    }

    func structure(_ site: Site, length: Double) -> Structure {
        let p = params
        var heading = site.heading.truncatingRemainder(dividingBy: 360)
        if heading < 0 { heading += 360 }
        func designator(_ h: Double) -> Int {
            let n = Int((h / 10).rounded()) % 36
            return n == 0 ? 36 : n
        }
        let a = designator(heading), b = designator(heading + 180)
        let low = Swift.min(a, b), high = Swift.max(a, b)
        let zones = [
            StructureZone(name: "runway", halfLength: length / 2, halfWidth: p.runwayWidth / 2,
                          surface: .paved, plants: .none, graded: true),
            StructureZone(name: "strip", halfLength: length / 2 + p.stripOverrun, halfWidth: p.stripWidth / 2,
                          surface: .mownGrass, plants: .grassOnly, graded: true),
        ]
        return Structure(id: "airstrip", kind: name, x: site.x, y: site.y, heading: heading,
                         elevation: site.elevation, slope: site.slope, zones: zones,
                         properties: [
                            "designators": String(format: "%02d/%02d", low, high),
                            // Long out of use: cracked paving, worn paint, the strip gone to seed.
                            "condition": "abandoned",
                            "length": String(format: "%.0f", length),
                            "width": String(format: "%.0f", p.runwayWidth),
                            "meanCut": String(format: "%.2f", site.meanCut),
                            "maxCut": String(format: "%.2f", site.maxCut),
                            "crosswind": String(format: "%.0f", site.crosswind),
                            "score": String(format: "%.3f", site.score),
                         ])
    }

    /// The apron and its taxiway beside `runway`: the side and the point along it (the middle
    /// or a quarter of the way from either end) whose apron needs the least earthwork on the
    /// fine heights. None when every choice runs into the sea or past `apronMaxCut`.
    func apron(beside runway: Structure, _ ctx: SiteContext) -> [Structure] {
        let p = params
        let runwayHalfWidth = runway.zones.first { $0.surface == .paved }?.halfWidth ?? p.runwayWidth / 2
        let stripHalfWidth = runway.gradedHalfWidth
        let length = runway.zones.first { $0.surface == .paved }.map { 2 * $0.halfLength } ?? 1000
        let sea = Double(ctx.seaLevel)
        var best: (cost: Double, u: Double, side: Double)?
        for side in [-1.0, 1.0] {
            for along in [0.0, -0.25, 0.25] {
                let u0 = along * length
                let v0 = side * (stripHalfWidth + p.apronDepth / 2)
                var sum = 0.0, worst = 0.0, n = 0.0
                var wet = false
                var a = -p.apronLength / 2
                while a <= p.apronLength / 2 + 1e-6 {
                    var b = -p.apronDepth / 2
                    while b <= p.apronDepth / 2 + 1e-6 {
                        let (x, y) = runway.world(u: u0 + a, v: v0 + b)
                        if x < 0 || y < 0 || x > ctx.domain || y > ctx.domain { wet = true; break }
                        let h = Double(ctx.at(ctx.fine, x, y))
                        if h < sea + 0.5 { wet = true; break }
                        let d = abs(h - runway.gradeHeight(u: u0 + a))
                        sum += d; worst = Swift.max(worst, d); n += 1
                        b += 10
                    }
                    if wet { break }
                    a += 10
                }
                if wet || worst > p.apronMaxCut || n == 0 { continue }
                let cost = sum / n + 0.3 * worst
                if best == nil || cost < best!.cost { best = (cost, u0, side) }
            }
        }
        guard let pick = best else {
            print("structures: no room for an apron beside the runway")
            return []
        }
        let condition = runway.properties["condition"] ?? "abandoned"
        func piece(_ kind: String, u: Double, v: Double, halfLength: Double, halfWidth: Double,
                   edges: String?) -> Structure {
            let (x, y) = runway.world(u: u, v: v)
            var properties = ["condition": condition]
            if let edges { properties["edges"] = edges }
            return Structure(id: kind, kind: kind, x: x, y: y, heading: runway.heading,
                             elevation: runway.gradeHeight(u: u), slope: runway.slope,
                             zones: [StructureZone(name: kind, halfLength: halfLength, halfWidth: halfWidth,
                                                   surface: .concrete, plants: .none, graded: true)],
                             properties: properties)
        }
        // The taxiway runs from 2 m inside the runway's edge to 2 m inside the apron, so its
        // ends meet paving (they don't crumble: only its sides do).
        let inner = runwayHalfWidth - 2, outer = stripHalfWidth + 2
        let taxiway = piece("taxiway", u: pick.u, v: pick.side * (inner + outer) / 2,
                            halfLength: p.taxiwayWidth / 2, halfWidth: (outer - inner) / 2, edges: "sides")
        let apron = piece("apron", u: pick.u, v: pick.side * (stripHalfWidth + p.apronDepth / 2),
                          halfLength: p.apronLength / 2, halfWidth: p.apronDepth / 2, edges: nil)
        return [taxiway, apron]
    }

    /// The best site under `rules`, or nil when none passes them.
    func search(_ ctx: SiteContext, rules: Rules) -> Site? {
        let p = params
        let domain = ctx.domain
        let steps = Int(domain / p.spacing)
        let headings = stride(from: 0.0, to: 180.0, by: p.headingStep).map { $0 }
        let hl = rules.length / 2 + p.stripOverrun, hw = p.stripWidth / 2
        let sea = Double(ctx.seaLevel)
        let lock = NSLock()
        var found: [Site] = []
        Parallel.forRanges(steps) { rows in
            var local: [Site] = []
            for j in rows {
                let y = (Double(j) + 0.5) * p.spacing
                for i in 0..<steps {
                    let x = (Double(i) + 0.5) * p.spacing
                    let h = Double(ctx.at(ctx.coarse, x, y))
                    if h < sea + rules.minAboveSea || h > sea + 600 { continue }
                    if Double(ctx.at(ctx.seaDistance, x, y)) < hw { continue }
                    if !ctx.isClear(x: x, y: y, radius: (hl * hl + hw * hw).squareRoot()) { continue }
                    for heading in headings {
                        if let s = evaluate(ctx, x: x, y: y, heading: heading, rules: rules, fine: false) { local.append(s) }
                    }
                }
            }
            lock.lock(); found += local; lock.unlock()
        }
        guard !found.isEmpty else { return nil }
        found.sort { $0.score < $1.score }
        // The best few distinct sites (apart in place or heading).
        var picks: [Site] = []
        for s in found {
            let distinct = picks.allSatisfy { q in
                let d = ((s.x - q.x) * (s.x - q.x) + (s.y - q.y) * (s.y - q.y)).squareRoot()
                let turn = abs(s.heading - q.heading).truncatingRemainder(dividingBy: 180)
                return d > 600 || Swift.min(turn, 180 - turn) > 15
            }
            if distinct { picks.append(s) }
            if picks.count >= p.refineCount { break }
        }
        // Refine on the fine grid: small shifts and turns around each pick.
        let offsets = [-60.0, -30, 0, 30, 60]
        let turns = [-4.0, -2, 0, 2, 4]
        var best: Site?
        Parallel.forRanges(picks.count, chunks: picks.count) { range in
            var localBest: Site?
            for k in range {
                let s = picks[k]
                for dy in offsets {
                    for dx in offsets {
                        for dh in turns {
                            guard ctx.isClear(x: s.x + dx, y: s.y + dy, radius: (hl * hl + hw * hw).squareRoot()) else { continue }
                            if let r = evaluate(ctx, x: s.x + dx, y: s.y + dy, heading: s.heading + dh, rules: rules, fine: true),
                               r.score < (localBest?.score ?? .infinity) {
                                localBest = r
                            }
                        }
                    }
                }
            }
            if let localBest {
                lock.lock()
                if localBest.score < (best?.score ?? .infinity) { best = localBest }
                lock.unlock()
            }
        }
        return best
    }

    /// Scores a runway centred at (x, y) along compass `heading`, or nil when it breaks a hard rule.
    func evaluate(_ ctx: SiteContext, x: Double, y: Double, heading: Double, rules: Rules, fine: Bool) -> Site? {
        let p = params
        let grid = fine ? ctx.fine : ctx.coarse
        let sea = Double(ctx.seaLevel)
        let hl = rules.length / 2 + p.stripOverrun, hw = p.stripWidth / 2
        let h = heading * Double.pi / 180
        let ax = sin(h), ay = cos(h)        // along
        let cx = ay, cy = -ax               // across (right of along)
        @inline(__always) func pos(_ u: Double, _ v: Double) -> (Double, Double) { (x + u * ax + v * cx, y + u * ay + v * cy) }

        // Heights over the strip, and the grade line fitted to them.
        let along = fine ? Swift.max(grid.cellSize, 10) : Swift.max(grid.cellSize, 75)
        let nu = Int((2 * hl / along).rounded()) + 1
        let across: [Double] = fine ? stride(from: -hw, through: hw, by: Swift.max(grid.cellSize, 10)).map { $0 } : [-hw, 0, hw]
        var us: [Double] = [], hs: [Double] = []
        us.reserveCapacity(nu * across.count)
        hs.reserveCapacity(nu * across.count)
        var minCoast = Double.infinity
        for a in 0..<nu {
            let u = -hl + 2 * hl * Double(a) / Double(nu - 1)
            for v in across {
                let (px, py) = pos(u, v)
                if px < 0 || py < 0 || px > ctx.domain || py > ctx.domain { return nil }
                let z = Double(ctx.at(grid, px, py))
                if z < sea + rules.minAboveSea { return nil }
                us.append(u)
                hs.append(z)
                if v == across[0] || v == across[across.count - 1] {
                    minCoast = Swift.min(minCoast, Double(ctx.at(ctx.seaDistance, px, py)))
                }
            }
        }
        let n = Double(us.count)
        let meanU = us.reduce(0, +) / n, meanH = hs.reduce(0, +) / n
        var suu = 0.0, suh = 0.0
        for k in 0..<us.count {
            suu += (us[k] - meanU) * (us[k] - meanU)
            suh += (us[k] - meanU) * (hs[k] - meanH)
        }
        let slope = Swift.min(Swift.max(suu > 0 ? suh / suu : 0, -p.maxGrade), p.maxGrade)
        var elevation = meanH - slope * meanU
        // Both ends of the grade line above the sea rule.
        let lowEnd = elevation - abs(slope) * hl
        if lowEnd < sea + rules.minAboveSea { elevation += sea + rules.minAboveSea - lowEnd }
        var sumCut = 0.0, maxCut = 0.0
        for k in 0..<us.count {
            let d = abs(hs[k] - (elevation + slope * us[k]))
            sumCut += d
            maxCut = Swift.max(maxCut, d)
        }
        if maxCut > rules.maxCut { return nil }
        let meanCut = sumCut / n

        // Streams crossing the strip.
        var stream = 0.0
        if let drainage = ctx.drainage {
            var maxArea = 0.0
            for a in 0..<9 {
                let (px, py) = pos(-hl + 2 * hl * Double(a) / 8, 0)
                maxArea = Swift.max(maxArea, Double(ctx.at(drainage, px, py)))
            }
            if maxArea > rules.maxStreamArea { return nil }
            if maxArea > 5e5 { stream = Swift.min(1, log10(maxArea / 5e5)) }
        }

        // Crosswind: angle between the runway and the trade wind's line.
        let runwayMath = 90 - heading
        let windMath = ctx.windDirection * 180 / Double.pi
        var delta = abs(runwayMath - windMath).truncatingRemainder(dividingBy: 180)
        delta = Swift.min(delta, 180 - delta)
        let cross = Swift.min(1, Swift.max(0, (delta - 15) / 45))

        // Approaches: ground above a 1:20 climb surface out to 3 km past each end.
        var penetration = 0.0
        let approachStep = fine ? 100.0 : 150.0
        for end in [-1.0, 1.0] {
            let threshold = end * rules.length / 2
            let z0 = elevation + slope * Swift.min(Swift.max(threshold, -hl), hl)
            var worst = 0.0
            var t = p.stripOverrun
            while t <= p.approachLength {
                let surface = z0 + p.approachSlope * t
                let spread = Swift.min(hw + 0.15 * t, 300)
                for v in [-spread, 0, spread] {
                    let (px, py) = pos(threshold + end * t, v)
                    if px < 0 || py < 0 || px > ctx.domain || py > ctx.domain { continue }
                    worst = Swift.max(worst, Double(ctx.at(ctx.coarse, px, py)) - surface)
                }
                t += approachStep
            }
            penetration += worst
        }

        // Weather, ground and coast, sampled along the centreline.
        var cloud = 0.0, wet = 0.0
        for a in 0..<5 {
            let (px, py) = pos(-hl + 2 * hl * Double(a) / 4, 0)
            if let clouds = ctx.clouds { cloud += Double(ctx.at(clouds, px, py)) / 5 }
            let w = ctx.community(.wetForest, px, py) + ctx.community(.bog, px, py) + ctx.community(.riparian, px, py)
                + ctx.community(.marsh, px, py) + ctx.community(.cloudShrubland, px, py) + ctx.community(.cliff, px, py)
            wet += Double(w) / 5
        }
        let centreCoast = Double(ctx.at(ctx.seaDistance, x, y))
        var coast = 0.0
        if minCoast < 250 { coast += (250 - minCoast) / 250 }
        if centreCoast > 3000 { coast += Swift.min(1, (centreCoast - 3000) / 4000) }
        let altitude = Swift.max(0, elevation - sea - 150) / 300

        let earthwork = meanCut / 1.5 + 0.3 * maxCut / Swift.max(rules.maxCut, 1)
        let score = earthwork + 0.8 * cross + 0.8 * Swift.min(penetration / 60, 3) + 0.5 * cloud + 0.4 * wet
            + 0.4 * coast + 0.4 * altitude + 0.4 * stream
        return Site(x: x, y: y, heading: heading, score: score, elevation: elevation, slope: slope,
                    meanCut: meanCut, maxCut: maxCut, crosswind: delta)
    }
}
