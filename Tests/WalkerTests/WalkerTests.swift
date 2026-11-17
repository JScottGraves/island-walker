import XCTest
@testable import Walker

final class WalkerTests: XCTestCase {
    /// A wall rising 5 m per metre east of x = 1, flat ground west of it.
    static func wall(_ x: Double, _ y: Double) -> Double { max(0, (x - 1) * 5) }

    func testEyeOnFlatGroundIsEyeHeightUp() {
        let w = Walker(x: 0, y: 0, ground: { _, _ in 10 })
        XCTAssertEqual(w.eyeZ(ground: { _, _ in 10 }), 11.7, accuracy: 1e-9)
    }

    func testEyeStaysClearOfAWallBesideIt() {
        // Feet on the flat 0.2 m from the wall's foot: the wall is 1.7 m high just 0.2 m
        // further on, inside the near plane; the eye must rise above it there.
        let w = Walker(x: 0.8, y: 0, ground: Self.wall)
        let z = w.eyeZ(ground: Self.wall)
        for k in 0..<16 {
            let a = Double(k) * .pi / 8
            for r in stride(from: 0.1, through: Walker.nearClearance, by: 0.1) {
                XCTAssertGreaterThan(z, Self.wall(0.8 + r * sin(a), r * cos(a)) + 0.3)
            }
        }
    }

    func testWalksAlongTheFootOfAWall() {
        // Standing beside a wall (the eye raised clear of it), walking along it still moves.
        var w = Walker(x: 0.8, y: 0, ground: Self.wall)
        w.yaw = 0   // north, along the wall
        for _ in 0..<60 { w.update(forward: 1, strafe: 0, running: false, dt: 1.0 / 60, ground: Self.wall) }
        XCTAssertGreaterThan(w.y, 1.0)
        XCTAssertGreaterThan(w.eyeZ(ground: Self.wall), w.z + w.eyeHeight)
    }

    func testWalksOnRoughGentleGround() {
        // 0.5 m bumps every 2 m: the eye stays at eye height and walking never stalls.
        let bumps: (Double, Double) -> Double = { x, y in 0.25 * sin(x * .pi) * cos(y * .pi) }
        var w = Walker(x: 0, y: 0, ground: bumps)
        w.yaw = .pi / 4
        for _ in 0..<120 { w.update(forward: 1, strafe: 0, running: false, dt: 1.0 / 60, ground: bumps) }
        XCTAssertGreaterThan(hypot(w.x, w.y), 2.5)
        XCTAssertEqual(w.eyeZ(ground: bumps), w.z + w.eyeHeight, accuracy: 1e-9)
    }
}
