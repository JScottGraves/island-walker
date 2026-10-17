import IslandCore

/// The first six native species (docs/flora.md §2). Native and indigenous Hawaiian flora
/// only, as it was before human contact.
public enum Species: Int, CaseIterable, Sendable {
    case ohia, koa, loulu, hapuu, naupaka, aalii

    /// Hawaiian name, with ʻokina and kahakō.
    public var name: String {
        switch self {
        case .ohia: return "ʻōhiʻa lehua"
        case .koa: return "koa"
        case .loulu: return "loulu"
        case .hapuu: return "hāpuʻu"
        case .naupaka: return "naupaka kahakai"
        case .aalii: return "ʻaʻaliʻi"
        }
    }

    public var scientificName: String {
        switch self {
        case .ohia: return "Metrosideros polymorpha"
        case .koa: return "Acacia koa"
        case .loulu: return "Pritchardia martii"
        case .hapuu: return "Cibotium chamissoi"
        case .naupaka: return "Scaevola taccada"
        case .aalii: return "Dodonaea viscosa"
        }
    }
}

/// What the renderer draws: a species in one growth form. ʻŌhiʻa grows as a tall tree in
/// the forests and as a dwarf shrub on wind-blasted summits and bogs; the same species,
/// built at small scale with denser branching.
public enum Plant: Int, CaseIterable, Sendable {
    case ohia, ohiaDwarf, koa, loulu, hapuu, naupaka, aalii

    public var species: Species {
        switch self {
        case .ohia, .ohiaDwarf: return .ohia
        case .koa: return .koa
        case .loulu: return .loulu
        case .hapuu: return .hapuu
        case .naupaka: return .naupaka
        case .aalii: return .aalii
        }
    }

    /// File-name style identifier.
    public var id: String {
        switch self {
        case .ohia: return "ohia"
        case .ohiaDwarf: return "ohia-dwarf"
        case .koa: return "koa"
        case .loulu: return "loulu"
        case .hapuu: return "hapuu"
        case .naupaka: return "naupaka"
        case .aalii: return "aalii"
        }
    }

    /// Mature height in good conditions, metres; the scatter picks within it by site.
    public var heightRange: ClosedRange<Double> {
        switch self {
        case .ohia: return 6...20
        case .ohiaDwarf: return 0.5...3
        case .koa: return 12...25
        case .loulu: return 5...10
        case .hapuu: return 2...6
        case .naupaka: return 1...3
        case .aalii: return 1.5...6
        }
    }

    /// Height the meshes are built at (metres); instances scale from it.
    public var referenceHeight: Double {
        switch self {
        case .ohia: return 14
        case .ohiaDwarf: return 1.6
        case .koa: return 18
        case .loulu: return 7.5
        case .hapuu: return 4
        case .naupaka: return 1.8
        case .aalii: return 3.5
        }
    }
}

/// How a mesh surface is shaded: which texture it samples, whether it is an alpha-tested
/// card, a tint, and how much sunlight passes through it (leaves seen against the sun).
/// The shader gets this table as Metal constants (`metalTable`), indexed by
/// `FloraVertex.material`.
public struct FloraMaterial: Sendable {
    public var name: String
    /// Alpha-tested card (leaves, flowers, fronds) rather than opaque bark.
    public var card: Bool
    /// Layer in the card or bark texture array.
    public var layer: Int
    /// Multiplies the texture (linear).
    public var tint: RGB
    /// Share of sunlight transmitted through to the far side.
    public var transmission: Double

    public init(_ name: String, card: Bool, layer: Int, tint: RGB = RGB(1, 1, 1), transmission: Double = 0) {
        self.name = name; self.card = card; self.layer = layer; self.tint = tint; self.transmission = transmission
    }

    /// Card texture layers, in texture-array order (Assets/<name>.png).
    public static let cardTextures = ["ohia-leaves", "ohia-flowers", "koa-phyllodes", "loulu-fan",
                                      "hapuu-frond", "naupaka-leaves", "aalii-leaves"]
    /// Bark texture layers, in texture-array order.
    public static let barkTextures = ["ohia-bark", "koa-bark", "loulu-bark", "hapuu-bark", "aalii-bark"]

    public enum ID: Int, CaseIterable, Sendable {
        case ohiaBark, koaBark, louluTrunk, hapuuTrunk, pulu, aaliiBark, naupakaStem
        case ohiaLeaves, lehua, koaPhyllodes, louluFan, louluDead, hapuuFrond, naupakaLeaves, aaliiLeaves
        /// floragen's 1.7 m scale figure.
        case figure
    }

    /// Indexed by `ID`.
    public static let table: [FloraMaterial] = [
        FloraMaterial("ohia bark", card: false, layer: 0),
        FloraMaterial("koa bark", card: false, layer: 1),
        FloraMaterial("loulu trunk", card: false, layer: 2),
        FloraMaterial("hapuu trunk", card: false, layer: 3),
        // Golden pulu wool on the hāpuʻu crown and fiddleheads.
        FloraMaterial("pulu", card: false, layer: 3, tint: RGB(3.2, 2.2, 1.0)),
        FloraMaterial("aalii bark", card: false, layer: 4),
        // Young naupaka stems are pale green.
        FloraMaterial("naupaka stem", card: false, layer: 1, tint: RGB(0.9, 1.25, 0.8)),
        FloraMaterial("ohia leaves", card: true, layer: 0, transmission: 0.25),
        FloraMaterial("lehua", card: true, layer: 1, transmission: 0.35),
        FloraMaterial("koa phyllodes", card: true, layer: 2, transmission: 0.2),
        FloraMaterial("loulu fan", card: true, layer: 3, transmission: 0.15),
        FloraMaterial("loulu dead fan", card: true, layer: 3, tint: RGB(1.7, 1.35, 1.0), transmission: 0.05),
        FloraMaterial("hapuu frond", card: true, layer: 4, transmission: 0.4),
        FloraMaterial("naupaka leaves", card: true, layer: 5, transmission: 0.3),
        FloraMaterial("aalii leaves", card: true, layer: 6, transmission: 0.25),
        FloraMaterial("figure", card: false, layer: 2, tint: RGB(0.55, 0.62, 0.9)),
    ]

    /// The table as Metal constants: per material, (tint rgb, transmission) and
    /// (layer, card 0/1).
    public static var metalTable: String {
        let tints = table.map { m in
            String(format: "float4(%.4f, %.4f, %.4f, %.3f)", m.tint.r, m.tint.g, m.tint.b, m.transmission)
        }
        let layers = table.map { "int2(\($0.layer), \($0.card ? 1 : 0))" }
        return """
        constant int kFloraMaterials = \(table.count);
        constant float4 kFloraTint[\(table.count)] = { \(tints.joined(separator: ", ")) };
        constant int2 kFloraLayer[\(table.count)] = { \(layers.joined(separator: ", ")) };

        """
    }
}
