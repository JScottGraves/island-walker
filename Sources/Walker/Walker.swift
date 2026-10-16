import Foundation
import IslandCore

/// First-person walker on a heightfield. World units are metres; x east, y north, z up.
/// Positions are Double so precision holds tens of kilometres from the origin.
public struct Walker: Sendable {
    public var x: Double
    public var y: Double
    /// Height of the feet above sea level.
    public var z: Double
    /// Heading in radians, 0 = north, increasing clockwise.
    public var yaw: Double = 0
    public var pitch: Double = 0

    public var eyeHeight = 1.7
    public var walkSpeed = 1.4
    public var runSpeed = 4.0
    /// Steepest slope (rise over run) the walker can climb.
    public var maxSlope = 1.0

    public init(x: Double, y: Double, ground: (Double, Double) -> Double) {
        self.x = x
        self.y = y
        self.z = ground(x, y)
    }

    public var eye: (x: Double, y: Double, z: Double) { (x, y, z + eyeHeight) }

    /// Moves the walker. `forward` and `strafe` are in [-1, 1].
    public mutating func update(forward: Double, strafe: Double, running: Bool, dt: Double,
                                ground: (Double, Double) -> Double) {
        let speed = running ? runSpeed : walkSpeed
        let dx = (sin(yaw) * forward + cos(yaw) * strafe) * speed * dt
        let dy = (cos(yaw) * forward - sin(yaw) * strafe) * speed * dt
        let distance = (dx * dx + dy * dy).squareRoot()
        guard distance > 0 else { return }
        let target = ground(x + dx, y + dy)
        if (target - z) / distance > maxSlope { return } // too steep to climb
        x += dx
        y += dy
        z = target
    }
}
