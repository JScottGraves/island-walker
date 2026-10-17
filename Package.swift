// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "IslandWalker",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "islandgen", targets: ["IslandGenCLI"]),
        .executable(name: "IslandApp", targets: ["IslandApp"]),
        .executable(name: "floragen", targets: ["FloraGen"]),
    ],
    targets: [
        .target(name: "IslandCore"),
        .target(name: "IslandGen", dependencies: ["IslandCore"]),
        // Not named "islandgen": macOS file systems are case-insensitive, so its build
        // directory would collide with the IslandGen library's.
        .executableTarget(name: "IslandGenCLI", dependencies: ["IslandCore", "IslandGen"]),
        // Native trees: species, procedural meshes, community table and textures.
        .target(name: "Flora", dependencies: ["IslandCore"], resources: [.copy("Assets")]),
        .target(name: "Engine", dependencies: ["IslandCore", "Flora"]),
        .target(name: "Walker", dependencies: ["IslandCore"]),
        .executableTarget(name: "IslandApp", dependencies: ["IslandCore", "IslandGen", "Engine", "Walker"]),
        .executableTarget(name: "FloraGen", dependencies: ["IslandCore", "IslandGen", "Flora", "Engine"]),
        .testTarget(name: "IslandCoreTests", dependencies: ["IslandCore"]),
        .testTarget(name: "IslandGenTests", dependencies: ["IslandCore", "IslandGen"]),
        .testTarget(name: "FloraTests", dependencies: ["IslandCore", "Flora"]),
    ]
)
