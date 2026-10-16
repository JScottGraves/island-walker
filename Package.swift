// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "IslandWalker",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "islandgen", targets: ["IslandGenCLI"]),
        .executable(name: "IslandApp", targets: ["IslandApp"]),
    ],
    targets: [
        .target(name: "IslandCore"),
        .target(name: "IslandGen", dependencies: ["IslandCore"]),
        // Not named "islandgen": macOS file systems are case-insensitive, so its build
        // directory would collide with the IslandGen library's.
        .executableTarget(name: "IslandGenCLI", dependencies: ["IslandCore", "IslandGen"]),
        .target(name: "Engine", dependencies: ["IslandCore"]),
        .target(name: "Walker", dependencies: ["IslandCore"]),
        .executableTarget(name: "IslandApp", dependencies: ["IslandCore", "IslandGen", "Engine", "Walker"]),
        .testTarget(name: "IslandCoreTests", dependencies: ["IslandCore"]),
        .testTarget(name: "IslandGenTests", dependencies: ["IslandCore", "IslandGen"]),
    ]
)
