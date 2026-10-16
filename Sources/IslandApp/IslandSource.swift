import Foundation
import IslandCore
import IslandGen

/// Where the app's island comes from: an island bundle (islandgen --bundle) or a baked
/// `height.r32` passed on the command line, or a quick in-app bake when none is given.
enum IslandSource {
    /// The island bundle named on the command line, if the first argument is one.
    static func bundle() -> IslandBundle? {
        let args = CommandLine.arguments
        guard args.count > 1, !args[1].hasPrefix("--") else { return nil }
        let url = URL(fileURLWithPath: args[1], isDirectory: true)
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent(IslandBundle.manifestName).path) else { return nil }
        do { return try IslandBundle(url: url) } catch {
            print("can't read the bundle at \(args[1]): \(error)")
            return nil
        }
    }

    /// The directory the island came from (for scarp.json and other side files).
    static var inputDirectory: URL? {
        let args = CommandLine.arguments
        guard args.count > 1, !args[1].hasPrefix("--") else { return nil }
        return URL(fileURLWithPath: args[1]).deletingLastPathComponent()
    }

    /// The terrain to render and its standing cloud coverage.
    static func terrain() -> (tiles: TerrainTiles, clouds: Grid) {
        if let bundle = bundle(), let tiles = try? BundleTerrain(bundle: bundle) {
            let size = tiles.size
            print("bundle: height \(size)² at \(tiles.cellMetres) m in \(tiles.levels) levels")
            if let clouds = try? bundle.grid("clouds-trades") {
                // Rescale the cell so the coverage spans the same domain as the heights.
                var spanning = Grid(width: clouds.width, height: clouds.height,
                                    cellSize: tiles.domainMetres / Double(clouds.width))
                spanning.values = clouds.values
                return (tiles, spanning)
            }
            let overview = (try? tiles.grid(level: tiles.overviewLevel())) ?? Grid(width: 1, height: 1, cellSize: tiles.domainMetres)
            return (tiles, Clouds.coverage(height: overview, seaLevel: tiles.seaLevel))
        }
        let grid = load()
        return (GridTerrain(height: grid, seaLevel: 0), clouds(for: grid))
    }

    /// Standing trade-wind cloud coverage: islandgen's clouds-trades.r32 next to the
    /// height map, or baked here from the heightfield if there is none.
    static func clouds(for height: Grid) -> Grid {
        let args = CommandLine.arguments
        let domain = Double(height.width) * height.cellSize
        if args.count > 1, args[1].hasSuffix(".r32") {
            let url = URL(fileURLWithPath: args[1]).deletingLastPathComponent().appendingPathComponent("clouds-trades.r32")
            if let data = FileManager.default.contents(atPath: url.path) {
                let count = data.count / MemoryLayout<Float>.size
                let side = Int(Double(count).squareRoot())
                if side * side == count, side > 0 {
                    var grid = Grid(width: side, height: side, cellSize: domain / Double(side))
                    data.withUnsafeBytes { raw in
                        for i in 0..<count { grid.values[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
                    }
                    return grid
                }
            }
        }
        return Clouds.coverage(height: height, seaLevel: 0)
    }

    static func load() -> Grid {
        let args = CommandLine.arguments
        if args.count > 1, args[1].hasSuffix(".r32"), let data = FileManager.default.contents(atPath: args[1]) {
            let count = data.count / MemoryLayout<Float>.size
            let side = Int(Double(count).squareRoot())
            if side * side == count {
                let domain = args.count > 2 ? Double(args[2]) ?? 80_000 : 80_000
                var grid = Grid(width: side, height: side, cellSize: domain / Double(side))
                data.withUnsafeBytes { raw in
                    for i in 0..<count { grid.values[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
                }
                return grid
            }
            print("\(args[1]) is not a square Float32 heightfield; baking a quick island instead")
        }
        var params = Island.Params()
        params.size = 512
        params.steps = 150
        print("baking a quick 512² island (pass a height.r32 from islandgen for a better one)")
        return Island.generate(params).height
    }
}
