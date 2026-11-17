import Foundation

/// Man-made structures (docs/architecture §structures): what islandgen's site planner placed
/// on the island, written next to the bundle's channels as `structures.json`.
///
/// A structure is a pose (centre and compass heading) plus a stack of zones, each an
/// oriented box around the centre, listed innermost first. A zone says what the ground is
/// (surface), what may grow on it (plants) and whether the ground follows the structure's
/// grade line there exactly. The bundle's height channel is already graded; the runtime
/// asks the structure map what lies at a point, so terrain shading, fine ground detail,
/// trees and ground cover agree without knowing any structure type.
///
/// World coordinates are metres from the domain's south-west corner, x east, y north,
/// like the bundle's grids (cell centres at (i + 0.5) × cell).

/// What the ground of a zone is made of. The raw values index the renderer's surface table.
public enum StructureSurface: String, Codable, Sendable, CaseIterable {
    case natural
    case paved
    case mownGrass
    case gravel
    case earth

    public var code: Int {
        switch self {
        case .natural: return 0
        case .paved: return 1
        case .mownGrass: return 2
        case .gravel: return 3
        case .earth: return 4
        }
    }
}

/// What may grow on a zone.
public enum PlantRule: String, Codable, Sendable {
    /// Whatever the biomes grow.
    case untouched
    /// Grasses only (no trees, shrubs, ferns or sedges), kept short.
    case grassOnly
    /// Nothing at all.
    case none

    public var code: Int {
        switch self {
        case .untouched: return 0
        case .grassOnly: return 1
        case .none: return 2
        }
    }
}

public struct StructureZone: Codable, Sendable, Equatable {
    public var name: String
    /// Half extents of the box along and across the structure's axis, metres.
    public var halfLength: Double
    public var halfWidth: Double
    public var surface: StructureSurface
    public var plants: PlantRule
    /// The ground inside follows the grade line exactly (the renderer's fine detail is
    /// replaced by it, so a runway is smooth to the centimetre).
    public var graded: Bool

    public init(name: String, halfLength: Double, halfWidth: Double, surface: StructureSurface,
                plants: PlantRule, graded: Bool) {
        self.name = name
        self.halfLength = halfLength
        self.halfWidth = halfWidth
        self.surface = surface
        self.plants = plants
        self.graded = graded
    }

    public func contains(u: Double, v: Double) -> Bool { abs(u) <= halfLength && abs(v) <= halfWidth }
}

public struct Structure: Codable, Sendable, Equatable {
    /// Unique within the island, such as "airstrip-1".
    public var id: String
    /// The structure type that placed it, such as "airstrip".
    public var kind: String
    /// Centre, metres (x east, y north).
    public var x: Double
    public var y: Double
    /// Compass bearing of the structure's +u axis, degrees clockwise from north.
    public var heading: Double
    /// Ground height at the centre and its rise per metre along +u (the grade line); level across.
    public var elevation: Double
    public var slope: Double
    /// Innermost first; the first zone containing a point is the one that applies.
    public var zones: [StructureZone]
    /// Width of the cut or fill slope around the graded zones, metres (1:3 down to where
    /// it meets the natural ground), for the ground cover that only grows grass there.
    public var embankment: Double
    /// Type-specific facts, such as the runway designators.
    public var properties: [String: String]

    public init(id: String, kind: String, x: Double, y: Double, heading: Double, elevation: Double, slope: Double,
                zones: [StructureZone], embankment: Double = 0, properties: [String: String] = [:]) {
        self.id = id
        self.kind = kind
        self.x = x
        self.y = y
        self.heading = heading
        self.elevation = elevation
        self.slope = slope
        self.zones = zones
        self.embankment = embankment
        self.properties = properties
    }

    /// Unit vector of +u (along) in world xy.
    public var axis: (x: Double, y: Double) {
        let h = heading * Double.pi / 180
        return (sin(h), cos(h))
    }

    /// World point → (along, across) metres from the centre; +v is to the right of +u.
    public func local(x px: Double, y py: Double) -> (u: Double, v: Double) {
        let a = axis, dx = px - x, dy = py - y
        return (dx * a.x + dy * a.y, dx * a.y - dy * a.x)
    }

    /// (along, across) → world point.
    public func world(u: Double, v: Double) -> (x: Double, y: Double) {
        let a = axis
        return (x + u * a.x + v * a.y, y + u * a.y - v * a.x)
    }

    /// The grade line's height at `u` metres along the axis (clamped to the graded zones' ends).
    public func gradeHeight(u: Double) -> Double {
        let reach = gradedHalfLength
        return elevation + slope * Swift.min(Swift.max(u, -reach), reach)
    }

    /// The outermost graded zone's half extents (0 when nothing is graded).
    public var gradedHalfLength: Double { zones.filter(\.graded).map(\.halfLength).max() ?? 0 }
    public var gradedHalfWidth: Double { zones.filter(\.graded).map(\.halfWidth).max() ?? 0 }

    /// Radius of a circle around the centre holding every zone, metres.
    public var radius: Double {
        zones.map { ($0.halfLength * $0.halfLength + $0.halfWidth * $0.halfWidth).squareRoot() }.max() ?? 0
    }

    public func zone(atU u: Double, v: Double) -> StructureZone? { zones.first { $0.contains(u: u, v: v) } }
}

public struct StructureManifest: Codable, Sendable, Equatable {
    public static let formatVersion = 1
    public static let fileName = "structures.json"

    public var format: Int
    public var structures: [Structure]

    public init(structures: [Structure]) {
        format = Self.formatVersion
        self.structures = structures
    }

    public func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url)
    }

    public static func read(from url: URL) throws -> StructureManifest {
        try JSONDecoder().decode(StructureManifest.self, from: Data(contentsOf: url))
    }
}

/// What the structures say about one point.
public struct StructureSample: Sendable, Equatable {
    public var surface: StructureSurface
    public var plants: PlantRule
    /// The grade line's height when the point lies in a graded zone, else nil.
    public var gradeHeight: Double?
    /// The structure it belongs to, or nil for open country.
    public var structure: Int?

    public static let open = StructureSample(surface: .natural, plants: .untouched, gradeHeight: nil, structure: nil)
}

/// Point queries over an island's structures (CPU side; the renderer's StructureMap mirrors it).
public struct StructureMap: Sendable {
    public let structures: [Structure]

    public init(_ structures: [Structure]) { self.structures = structures }

    public func sample(x: Double, y: Double) -> StructureSample {
        for (k, s) in structures.enumerated() {
            let dx = x - s.x, dy = y - s.y
            let r = s.radius + s.embankment
            guard dx * dx + dy * dy <= r * r else { continue }
            let (u, v) = s.local(x: x, y: y)
            if let z = s.zone(atU: u, v: v) {
                return StructureSample(surface: z.surface, plants: z.plants,
                                       gradeHeight: z.graded ? s.gradeHeight(u: u) : nil, structure: k)
            }
            // The cut or fill slope around the graded zones grows grass only.
            let du = abs(u) - s.gradedHalfLength, dv = abs(v) - s.gradedHalfWidth
            if s.embankment > 0, s.gradedHalfLength > 0, du <= s.embankment, dv <= s.embankment {
                return StructureSample(surface: .natural, plants: .grassOnly, gradeHeight: nil, structure: k)
            }
        }
        return .open
    }
}

public extension IslandBundle {
    /// The structures islandgen placed, or none for a bundle baked before structures.
    func structures() -> [Structure] {
        let file = url.appendingPathComponent(StructureManifest.fileName)
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        do { return try StructureManifest.read(from: file).structures } catch {
            print("can't read \(StructureManifest.fileName): \(error)")
            return []
        }
    }
}
