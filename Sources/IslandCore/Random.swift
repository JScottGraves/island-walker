/// Deterministic randomness. Everything in generation derives from a single seed so the
/// same seed yields a bit-identical island on any machine.

/// SplitMix64: a small, fast, well-distributed sequential generator.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform Double in [0, 1).
    public mutating func unit() -> Double {
        Double(next() >> 11) * 0x1.0p-53
    }

    public mutating func uniform(_ lo: Double, _ hi: Double) -> Double {
        lo + (hi - lo) * unit()
    }
}

/// Stateless hashing for lattice noise: same inputs always give the same output,
/// independent of evaluation order, which keeps parallel generation deterministic.
public enum Hash {
    @inline(__always)
    public static func mix(_ a: UInt64) -> UInt64 {
        var z = a &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform value in [-1, 1) for an integer lattice point.
    @inline(__always)
    public static func lattice(_ x: Int, _ y: Int, seed: UInt64) -> Double {
        let h = mix(seed ^ mix(UInt64(bitPattern: Int64(x)) ^ (mix(UInt64(bitPattern: Int64(y))) << 1)))
        return Double(h >> 11) * 0x1.0p-52 - 1.0
    }
}

/// Smooth value noise and fractal sums over it.
public enum Noise {
    /// Value noise in roughly [-1, 1] at continuous coordinates (lattice spacing 1).
    public static func value(_ x: Double, _ y: Double, seed: UInt64) -> Double {
        let xf = x.rounded(.down), yf = y.rounded(.down)
        let xi = Int(xf), yi = Int(yf)
        let tx = x - xf, ty = y - yf
        let sx = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
        let sy = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
        let a = Hash.lattice(xi, yi, seed: seed)
        let b = Hash.lattice(xi + 1, yi, seed: seed)
        let c = Hash.lattice(xi, yi + 1, seed: seed)
        let d = Hash.lattice(xi + 1, yi + 1, seed: seed)
        let ab = a + (b - a) * sx
        let cd = c + (d - c) * sx
        return ab + (cd - ab) * sy
    }

    /// Fractal Brownian motion: `octaves` layers of value noise, each at double frequency
    /// and `gain` times the amplitude of the previous one. Output roughly in [-1, 1].
    public static func fbm(_ x: Double, _ y: Double, seed: UInt64, octaves: Int = 6, gain: Double = 0.5) -> Double {
        var sum = 0.0, amp = 1.0, freq = 1.0, norm = 0.0
        for o in 0..<octaves {
            sum += amp * value(x * freq, y * freq, seed: seed &+ UInt64(o) &* 0x632B_E59B_D9B4_E019)
            norm += amp
            amp *= gain
            freq *= 2
        }
        return sum / norm
    }
}
