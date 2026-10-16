import Foundation
import IslandCore
import IslandGen

/// Where the app's island comes from: a baked `height.r32` passed on the command line,
/// or a quick in-app bake when none is given.
enum IslandSource {
    static func load() -> Grid {
        let args = CommandLine.arguments
        if args.count > 1, args[1].hasSuffix(".r32"), let data = FileManager.default.contents(atPath: args[1]) {
            let count = data.count / MemoryLayout<Float>.size
            let side = Int(Double(count).squareRoot())
            if side * side == count {
                let domain = args.count > 2 ? Double(args[2]) ?? 80_000 : 80_000
                var grid = Grid(width: side, height: side, cellSize: domain / Double(side))
                data.withUnsafeBytes { raw in
                    for i in 0..<count { grid.values[i] = raw.load(fromByteOffset: i * 4, as: Float.self) }
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
