import IslandCore

/// Native grasses and sedges drawn as blades near the walker (docs/flora.md §12): which
/// grows in each plant community, how densely and in what shape. Pure data, shared by the
/// renderer (GroundCoverRenderer, its shaders through `metalTable`) and the tests.
public enum GroundCover {
    public enum Species: Int, CaseIterable, Sendable {
        /// ʻAkiʻaki (Sporobolus virginicus): short wiry grey-green runs on sand and rock by the sea.
        case akiaki
        /// Kawelu (Eragrostis variabilis): tall arching bunches, sparse in dry lowlands.
        case kawelu
        /// Makaloa (Cyperus laevigatus): upright round stems in brackish marsh.
        case makaloa
        /// ʻUki (Machaerina angustifolia): stiff sword-leaved sedge tussocks in mesic and wet
        /// forest and on the summits.
        case uki

        public var shape: Shape {
            switch self {
            case .akiaki:
                return Shape(height: 0.15...0.35, width: 0.022, bend: 0.35, maxSlope: 0.6, clump: 0.8, bunch: 0,
                             shadeTolerance: 0.2, round: false,
                             tip: RGB.srgb(132, 146, 108), root: RGB.srgb(96, 98, 70))
            case .kawelu:
                return Shape(height: 0.45...0.95, width: 0.016, bend: 0.85, maxSlope: 1.0, clump: 1.8, bunch: 0.15,
                             shadeTolerance: 0.3, round: false,
                             tip: RGB.srgb(168, 158, 104), root: RGB.srgb(118, 108, 64))
            case .makaloa:
                return Shape(height: 0.4...1.4, width: 0.012, bend: 0.12, maxSlope: 0.3, clump: 0.7, bunch: 0,
                             shadeTolerance: 0.4, round: true,
                             tip: RGB.srgb(86, 108, 56), root: RGB.srgb(66, 74, 44))
            case .uki:
                return Shape(height: 0.4...1.1, width: 0.05, bend: 0.45, maxSlope: 1.43, clump: 1.3, bunch: 0.25,
                             shadeTolerance: 0.8, round: false,
                             tip: RGB.srgb(78, 108, 48), root: RGB.srgb(58, 70, 36))
            }
        }
    }

    /// A species' blades, each standing for a few leaves of a tuft. `height` metres; `width`
    /// metres at the root; `bend` how far the
    /// tip arches over (0 upright, 1 lying over); `maxSlope` the steepest ground (tan) it
    /// grows on; `clump` the spacing (m) of the Voronoi clumps that share a lean and height;
    /// `bunch` > 0 keeps only blades within that fraction of the spacing from a clump's
    /// centre (tussocks); `shadeTolerance` how much density survives under a full canopy;
    /// `round` stems face the camera rather than a fixed way. Colours are linear albedo.
    public struct Shape: Sendable {
        public var height: ClosedRange<Double>
        public var width: Double
        public var bend: Double
        public var maxSlope: Double
        public var clump: Double
        public var bunch: Double
        public var shadeTolerance: Double
        public var round: Bool
        public var tip: RGB
        public var root: RGB
    }

    /// What one plant community grows: its species, blades per m² where it grows (inside the
    /// tussocks for bunch species), how patchy (0 everywhere … 1 nowhere), and the share of
    /// the ground the cover hides as seen from afar (for the ground texture's tint).
    public struct Community: Sendable {
        public var channel: String
        public var species: Species
        public var density: Double
        public var patchiness: Double
        public var cover: Double
    }

    /// In the bundle's biome channel order (Look.communities).
    public static let communities: [Community] = [
        Community(channel: "biome-strand", species: .akiaki, density: 56, patchiness: 0.3, cover: 0.7),
        Community(channel: "biome-rocky-shore", species: .akiaki, density: 18, patchiness: 0.45, cover: 0.35),
        Community(channel: "biome-marsh", species: .makaloa, density: 24, patchiness: 0.35, cover: 0.7),
        // Dry lowlands are mostly bare red soil between kawelu bunches.
        Community(channel: "biome-dry-forest", species: .kawelu, density: 60, patchiness: 0.5, cover: 0.2),
        Community(channel: "biome-mesic-forest", species: .uki, density: 50, patchiness: 0.35, cover: 0.35),
        Community(channel: "biome-wet-forest", species: .uki, density: 50, patchiness: 0.3, cover: 0.35),
        Community(channel: "biome-riparian", species: .uki, density: 40, patchiness: 0.3, cover: 0.3),
        Community(channel: "biome-cloud-shrubland", species: .uki, density: 45, patchiness: 0.3, cover: 0.4),
        Community(channel: "biome-bog", species: .uki, density: 35, patchiness: 0.25, cover: 0.4),
        Community(channel: "biome-cliff", species: .uki, density: 0, patchiness: 1, cover: 0),
        Community(channel: "biome-montane", species: .uki, density: 30, patchiness: 0.4, cover: 0.3),
    ]

    /// Tiles the cover is generated in (metres), and candidates per tile on a jittered grid
    /// (`tileGrid`² of them): one blade at most per candidate.
    public static let tileSize = 8.0
    public static let tileGrid = 64
    public static var tileCandidates: Int { tileGrid * tileGrid }

    /// The app's and snapshots' flags: `--cover off`, `--cover-range M` (where blades end;
    /// full density to half of it), `--cover-density X` (scales every community's).
    public struct Settings: Sendable {
        public var enabled = true
        public var range: Float = 50
        public var density: Float = 1

        public init(enabled: Bool = true, range: Float = 50, density: Float = 1) {
            self.enabled = enabled
            self.range = range
            self.density = density
        }

        public init(arguments: [String]) {
            func value(_ name: String) -> String? {
                guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
                return arguments[i + 1]
            }
            enabled = value("--cover") != "off"
            range = value("--cover-range").flatMap(Float.init) ?? 50
            density = value("--cover-density").flatMap(Float.init) ?? 1
        }
    }

    /// The tables as Metal constants (GroundCoverShaders).
    public static var metalTable: String {
        func f(_ v: Double) -> String { String(format: "%.5f", v) }
        func rgb(_ c: RGB) -> String { "float3(\(f(c.r)), \(f(c.g)), \(f(c.b)))" }
        let shapes = Species.allCases.map(\.shape)
        let a = shapes.map { "float4(\(f($0.height.lowerBound)), \(f($0.height.upperBound)), \(f($0.width)), \(f($0.bend)))" }
        let b = shapes.map { "float4(\(f($0.maxSlope)), \(f($0.clump)), \(f($0.bunch)), \(f($0.shadeTolerance)))" }
        let round = shapes.map { $0.round ? "1" : "0" }
        let tips = shapes.map { rgb($0.tip) }, roots = shapes.map { rgb($0.root) }
        let c = communities.map { "float4(\($0.species.rawValue), \(f($0.density)), \(f($0.patchiness)), \(f($0.cover)))" }
        let n = Species.allCases.count
        return """
        constant int kCoverSpecies = \(n);
        constant float4 kCoverShapeA[\(n)] = { \(a.joined(separator: ", ")) };   // height min, max (m), width (m), bend
        constant float4 kCoverShapeB[\(n)] = { \(b.joined(separator: ", ")) };   // max slope, clump spacing (m), bunch, shade tolerance
        constant int kCoverRound[\(n)] = { \(round.joined(separator: ", ")) };
        constant float3 kCoverTip[\(n)] = { \(tips.joined(separator: ", ")) };
        constant float3 kCoverRoot[\(n)] = { \(roots.joined(separator: ", ")) };
        constant float4 kCoverCommunity[\(communities.count)] = { \(c.joined(separator: ", ")) };  // species, blades/m², patchiness, cover
        constant float kCoverTile = \(f(tileSize));
        constant int kCoverGrid = \(tileGrid);

        """
    }
}
