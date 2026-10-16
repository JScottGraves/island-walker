/// A square-celled 2D grid of Float samples, stored row-major.
/// Used for heightfields and every derived map (rainfall, hardness, drainage area, ...).
public struct Grid: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// Size of one cell edge in metres.
    public let cellSize: Double
    public var values: [Float]

    public init(width: Int, height: Int, cellSize: Double, fill: Float = 0) {
        precondition(width > 0 && height > 0, "grid must be non-empty")
        self.width = width
        self.height = height
        self.cellSize = cellSize
        self.values = [Float](repeating: fill, count: width * height)
    }

    public var count: Int { width * height }
    public var cellArea: Double { cellSize * cellSize }

    @inline(__always)
    public func index(_ x: Int, _ y: Int) -> Int { y * width + x }

    @inline(__always)
    public func contains(_ x: Int, _ y: Int) -> Bool {
        x >= 0 && y >= 0 && x < width && y < height
    }

    @inline(__always)
    public func isBoundary(_ i: Int) -> Bool {
        let x = i % width, y = i / width
        return x == 0 || y == 0 || x == width - 1 || y == height - 1
    }

    public subscript(x: Int, y: Int) -> Float {
        get { values[y * width + x] }
        set { values[y * width + x] = newValue }
    }

    public var range: (min: Float, max: Float) {
        var lo = Float.infinity, hi = -Float.infinity
        for v in values {
            lo = Swift.min(lo, v)
            hi = Swift.max(hi, v)
        }
        return (lo, hi)
    }

    /// Bilinear sample at continuous cell coordinates, clamped to the grid edge.
    public func sample(x: Double, y: Double) -> Float {
        let cx = Swift.min(Swift.max(x, 0), Double(width - 1))
        let cy = Swift.min(Swift.max(y, 0), Double(height - 1))
        let x0 = Int(cx), y0 = Int(cy)
        let x1 = Swift.min(x0 + 1, width - 1), y1 = Swift.min(y0 + 1, height - 1)
        let fx = Float(cx - Double(x0)), fy = Float(cy - Double(y0))
        let a = self[x0, y0] + (self[x1, y0] - self[x0, y0]) * fx
        let b = self[x0, y1] + (self[x1, y1] - self[x0, y1]) * fx
        return a + (b - a) * fy
    }

    /// Bicubic (Catmull-Rom) sample at continuous cell coordinates, clamped to the grid edge.
    /// Smooth first derivatives, so close-up terrain shows no bilinear creases.
    public func sampleCubic(x: Double, y: Double) -> Float {
        let cx = Swift.min(Swift.max(x, 0), Double(width - 1))
        let cy = Swift.min(Swift.max(y, 0), Double(height - 1))
        let x1 = Int(cx), y1 = Int(cy)
        let tx = Float(cx - Double(x1)), ty = Float(cy - Double(y1))
        @inline(__always) func at(_ x: Int, _ y: Int) -> Float {
            self[Swift.min(Swift.max(x, 0), width - 1), Swift.min(Swift.max(y, 0), height - 1)]
        }
        @inline(__always) func cr(_ p0: Float, _ p1: Float, _ p2: Float, _ p3: Float, _ t: Float) -> Float {
            let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
            let b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
            let c = -0.5 * p0 + 0.5 * p2
            return ((a * t + b) * t + c) * t + p1
        }
        @inline(__always) func row(_ yy: Int) -> Float {
            cr(at(x1 - 1, yy), at(x1, yy), at(x1 + 1, yy), at(x1 + 2, yy), tx)
        }
        return cr(row(y1 - 1), row(y1), row(y1 + 1), row(y1 + 2), ty)
    }
}

/// The eight D8 neighbour offsets and their distances in cell units.
public enum D8 {
    public static let dx: [Int] = [1, 1, 0, -1, -1, -1, 0, 1]
    public static let dy: [Int] = [0, 1, 1, 1, 0, -1, -1, -1]
    public static let distance: [Double] = [1, 2.0.squareRoot(), 1, 2.0.squareRoot(), 1, 2.0.squareRoot(), 1, 2.0.squareRoot()]
}
