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

    /// How far round the eye the ground must stay below it: past the camera's near plane
    /// (0.3 m) at its corners, or a steep wall beside the eye is clipped away and the view
    /// looks through the ground.
    public static let nearClearance = 0.6
    /// How far the ground within `nearClearance` must stay below the eye.
    static let headroom = 0.4

    /// The eye's height: `eyeHeight` over the feet, raised where ground within
    /// `nearClearance` would come nearer the eye than `headroom` (spawned against a wall).
    public func eyeZ(ground: (Double, Double) -> Double) -> Double {
        Self.eyeZ(x: x, y: y, feet: z, eyeHeight: eyeHeight, ground: ground)
    }

    static func eyeZ(x: Double, y: Double, feet: Double, eyeHeight: Double,
                     ground: (Double, Double) -> Double) -> Double {
        var eye = feet + eyeHeight
        for r in [nearClearance / 3, 2 * nearClearance / 3, nearClearance] {
            for k in 0..<16 {
                let a = Double(k) * .pi / 8
                eye = max(eye, ground(x + r * sin(a), y + r * cos(a)) + headroom)
            }
        }
        return eye
    }

    /// Moves the walker. `forward` and `strafe` are in [-1, 1]. `blocked` says where the
    /// walker may not stand (a building's walls); against one it slides along it.
    public mutating func update(forward: Double, strafe: Double, running: Bool, dt: Double,
                                ground: (Double, Double) -> Double,
                                blocked: ((Double, Double) -> Bool)? = nil) {
        let speed = running ? runSpeed : walkSpeed
        let dx = (sin(yaw) * forward + cos(yaw) * strafe) * speed * dt
        let dy = (cos(yaw) * forward - sin(yaw) * strafe) * speed * dt
        // The whole step, else only its east or north part (sliding along a wall).
        for (sx, sy) in [(dx, dy), (dx, 0), (0, dy)] {
            let distance = (sx * sx + sy * sy).squareRoot()
            guard distance > 0 else { continue }
            if let blocked, blocked(x + sx, y + sy) { continue }
            let target = ground(x + sx, y + sy)
            if (target - z) / distance > maxSlope { return } // too steep to climb
            x += sx
            y += sy
            z = target
            return
        }
    }
}
