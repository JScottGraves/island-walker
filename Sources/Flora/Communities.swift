/// Which plants grow in each native plant community, and how densely (docs/flora.md §3).
/// First guesses from research/vegetation.md, to be tuned by eye against reference photos;
/// kept in this one table so they are easy to change.
public enum Communities {
    /// One community's canopy and shrub layers. Densities are mature stems per hectare;
    /// shares are each plant's part of its layer's stems and sum to 1.
    public struct Entry: Sendable {
        /// The bundle channel of its weight map (islandgen's Vegetation).
        public var channel: String
        public var canopy: Double
        public var canopyShares: [(plant: Plant, share: Double)]
        public var shrubs: Double
        public var shrubShares: [(plant: Plant, share: Double)]
    }

    /// In the bundle's biome channel order (Look.communities, Vegetation.Community).
    public static let table: [Entry] = [
        // Naupaka dense along the shore, ʻaʻaliʻi sparse behind it.
        Entry(channel: "biome-strand", canopy: 0, canopyShares: [],
              shrubs: 900, shrubShares: [(.naupaka, 0.88), (.aalii, 0.12)]),
        Entry(channel: "biome-rocky-shore", canopy: 0, canopyShares: [],
              shrubs: 250, shrubShares: [(.naupaka, 1)]),
        // Sedges are ground cover (step 3).
        Entry(channel: "biome-marsh", canopy: 0, canopyShares: [], shrubs: 0, shrubShares: []),
        Entry(channel: "biome-dry-forest", canopy: 120,
              canopyShares: [(.loulu, 0.40), (.aalii, 0.45), (.koa, 0.10), (.ohia, 0.05)],
              shrubs: 300, shrubShares: [(.aalii, 1)]),
        Entry(channel: "biome-mesic-forest", canopy: 350,
              canopyShares: [(.koa, 0.45), (.ohia, 0.35), (.loulu, 0.10), (.aalii, 0.10)],
              shrubs: 150, shrubShares: [(.aalii, 1)]),
        // Hāpuʻu fill the subcanopy of wet forest and stream banks.
        Entry(channel: "biome-wet-forest", canopy: 450,
              canopyShares: [(.ohia, 0.80), (.koa, 0.12), (.loulu, 0.08)],
              shrubs: 450, shrubShares: [(.hapuu, 1)]),
        Entry(channel: "biome-riparian", canopy: 400,
              canopyShares: [(.ohia, 0.60), (.koa, 0.25), (.loulu, 0.15)],
              shrubs: 450, shrubShares: [(.hapuu, 1)]),
        // Dwarf ʻōhiʻa are small enough for the shrub layer's finer grid.
        Entry(channel: "biome-cloud-shrubland", canopy: 90, canopyShares: [(.loulu, 1)],
              shrubs: 600, shrubShares: [(.ohiaDwarf, 1)]),
        Entry(channel: "biome-bog", canopy: 0, canopyShares: [],
              shrubs: 80, shrubShares: [(.ohiaDwarf, 1)]),
        Entry(channel: "biome-cliff", canopy: 40,
              canopyShares: [(.ohia, 0.5), (.loulu, 0.3), (.aalii, 0.2)],
              shrubs: 60, shrubShares: [(.aalii, 1)]),
        Entry(channel: "biome-montane", canopy: 300,
              canopyShares: [(.koa, 0.55), (.ohia, 0.45)],
              shrubs: 0, shrubShares: []),
    ]

    /// Canopy and shrub cell sizes of the scatter's hash grid (metres). One candidate per
    /// cell caps a layer at 10⁴ / cell² stems per hectare: 494 for canopy, 1600 for shrubs.
    public static let canopyCell = 4.5
    public static let shrubCell = 2.5
}
