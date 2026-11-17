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

    func testCannotWalkUpToTheFootOfAWall() {
        var w = Walker(x: -2, y: 0, ground: Self.wall)
        w.yaw = .pi / 2   // east, into the wall
        for _ in 0..<400 { w.update(forward: 1, strafe: 0, running: false, dt: 1.0 / 60, ground: Self.wall) }
        // Stops while the wall above eye level is still beyond the clearance.
        XCTAssertLessThan(w.x, 1 + 1.7 / 5 - Walker.nearClearance + 0.05)
        XCTAssertEqual(w.eyeZ(ground: Self.wall), w.z + w.eyeHeight, accuracy: 1e-9)
    }
}
