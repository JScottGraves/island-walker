import Foundation
import simd

/// Right-handed matrices with z up in world space, mapped to Metal clip space (z in 0...1).
public enum Matrix {
    public static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4(x, 0, 0, 0),
            SIMD4(0, y, 0, 0),
            SIMD4(0, 0, z, -1),
            SIMD4(0, 0, z * near, 0)))
    }

    /// View matrix looking along `forward` from the origin (camera-relative rendering:
    /// the camera position is subtracted from vertices before this is applied).
    public static func lookDirection(_ forward: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 0, 1)) -> simd_float4x4 {
        let f = simd_normalize(forward)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(columns: (
            SIMD4(s.x, u.x, -f.x, 0),
            SIMD4(s.y, u.y, -f.y, 0),
            SIMD4(s.z, u.z, -f.z, 0),
            SIMD4(0, 0, 0, 1)))
    }
}
