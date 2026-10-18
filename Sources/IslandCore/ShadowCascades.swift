/// Cascaded shadow maps for the trees' sun shadows (docs/flora.md §8): orthographic views
/// from the sun, one per range of distance from the camera. Each map is a fixed size and
/// sits on a fixed world-space texel grid, so shadow edges don't crawl as the camera moves.
public enum ShadowCascades {
    public struct Cascade: Sendable {
        /// Light-space axes: across the map (x right, y up) and towards the sun.
        public var right: SIMD3<Double>
        public var up: SIMD3<Double>
        public var toSun: SIMD3<Double>
        /// The map's centre in light space, relative to the camera (metres along right, up).
        public var offset: SIMD2<Double>
        /// Half the map's width, metres; one texel is `texel` metres.
        public var radius: Double
        public var texel: Double
        /// Camera-relative distance along `toSun` of depth 0 (nearest the sun) and depth 1.
        public var near: Double
        public var far: Double
        /// Distance from the camera this cascade is meant for.
        public var split: Double

        public var depthRange: Double { near - far }

        /// Clip-space rows (x, y, depth in 0…1) for a camera-relative point: dot each with
        /// (p, 1). Metal clip space: y up, so texture v = 0.5 − 0.5 y.
        public var clipRows: [SIMD4<Double>] {
            [SIMD4(right / radius, -offset.x / radius),
             SIMD4(up / radius, -offset.y / radius),
             SIMD4(-toSun / depthRange, near / depthRange)]
        }

        /// The same map seen from a camera moved from `old` to `new` (world positions): for a
        /// map drawn on an earlier frame and looked up on this one.
        public func rebased(from old: SIMD3<Double>, to new: SIMD3<Double>) -> Cascade {
            let shift = old - new
            var c = self
            c.offset += SIMD2(ShadowCascades.dot(shift, right), ShadowCascades.dot(shift, up))
            c.near += ShadowCascades.dot(shift, toSun)
            c.far += ShadowCascades.dot(shift, toSun)
            return c
        }

        /// Texture coordinates (u, v) and depth (0 nearest the sun) of a camera-relative point.
        public func project(_ p: SIMD3<Double>) -> SIMD3<Double> {
            let r = clipRows
            func row(_ k: Int) -> Double { dot(SIMD3(r[k].x, r[k].y, r[k].z), p) + r[k].w }
            return SIMD3(0.5 + 0.5 * row(0), 0.5 - 0.5 * row(1), row(2))
        }
    }

    /// Light-space axes for a sun direction (towards the sun).
    public static func basis(sun: SIMD3<Double>) -> (right: SIMD3<Double>, up: SIMD3<Double>, toSun: SIMD3<Double>) {
        let l = normalized(sun)
        let reference: SIMD3<Double> = abs(l.z) < 0.999 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        let right = normalized(cross(l, reference))
        return (right, cross(right, l), l)
    }

    /// One cascade per split distance. Cascade i holds the part of the view frustum within
    /// splits[i] of the camera (the shaders pick cascades by that distance): a cone-like
    /// region whose widest directions, the frustum's corners, are φ off the view axis. The
    /// sphere through the camera and the far rim of that cone is centred on the view axis at
    /// split / (2 cos φ), which holds it all when φ ≥ 45° (a wide lens); a narrower lens
    /// takes the sphere centred at half the split. So the map's size depends only on the
    /// split and the lens, never on where the camera looks, and its centre is snapped to
    /// whole texels in world space (`resolutions`: each map's side in texels). `casterReach`: how far beyond the sphere towards the sun
    /// casters are kept (metres).
    public static func fit(eye: SIMD3<Double>, forward: SIMD3<Double>, sun: SIMD3<Double>, splits: [Double],
                           tanHalfFovY: Double, aspect: Double, resolutions: [Int], casterReach: Double) -> [Cascade] {
        let f = normalized(forward)
        let (right, up, toSun) = basis(sun: sun)
        let tanX = tanHalfFovY * aspect
        let cosCorner = 1 / (1 + tanX * tanX + tanHalfFovY * tanHalfFovY).squareRoot()
        let wide = cosCorner * cosCorner <= 0.5
        return zip(splits, resolutions).map { split, resolution in
            let along = wide ? split / (2 * cosCorner) : split / 2
            let radius = wide ? along : split * (1.25 - cosCorner).squareRoot()
            let texel = 2 * radius / Double(resolution)
            let centre = eye + f * along
            func snapped(_ axis: SIMD3<Double>) -> Double {
                (dot(centre, axis) / texel).rounded() * texel - dot(eye, axis)
            }
            let depth = dot(f * along, toSun)
            return Cascade(right: right, up: up, toSun: toSun, offset: SIMD2(snapped(right), snapped(up)),
                           radius: radius, texel: texel, near: depth + radius + casterReach, far: depth - radius,
                           split: split)
        }
    }

    public static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { (a * b).sum() }

    public static func cross(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    public static func normalized(_ a: SIMD3<Double>) -> SIMD3<Double> { a / dot(a, a).squareRoot() }
}
