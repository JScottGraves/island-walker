import IslandCore

/// Binary min-heap keyed on (height, index). Ties break on index so the
/// processing order, and therefore the output, is fully deterministic.
/// Each entry packs an order-preserving encoding of the height above the cell index
/// into one UInt64, so comparing entries is a single integer compare.
struct MinHeap {
    private var items: [UInt64] = []

    var isEmpty: Bool { items.isEmpty }

    init(reserve: Int) {
        items.reserveCapacity(reserve)
    }

    /// Maps a Float to a UInt32 whose unsigned order matches the Float's order.
    @inline(__always)
    private static func orderedBits(_ f: Float) -> UInt32 {
        let b = f.bitPattern
        return b & 0x8000_0000 != 0 ? ~b : b | 0x8000_0000
    }

    mutating func push(_ key: Float, _ item: Int) {
        // -0 and +0 compare equal as Floats; give them one encoding so ties still break on index.
        let entry = UInt64(Self.orderedBits(key == 0 ? 0 : key)) << 32 | UInt64(UInt32(item))
        items.append(entry)
        items.withUnsafeMutableBufferPointer { a in
            var c = a.count - 1
            while c > 0 {
                let p = (c - 1) / 2
                guard entry < a[p] else { break }
                a[c] = a[p]
                c = p
            }
            a[c] = entry
        }
    }

    mutating func pop() -> Int {
        let top = items[0]
        let last = items.removeLast()
        if !items.isEmpty {
            items.withUnsafeMutableBufferPointer { a in
                let n = a.count
                var p = 0
                while true {
                    let l = 2 * p + 1
                    if l >= n { break }
                    var m = l
                    if l + 1 < n && a[l + 1] < a[l] { m = l + 1 }
                    guard a[m] < last else { break }
                    a[p] = a[m]
                    p = m
                }
                a[p] = last
            }
        }
        return Int(UInt32(truncatingIfNeeded: top))
    }
}

/// Depression filling (Barnes, Lehman & Mulla 2014, "Priority-Flood + Epsilon").
/// Raises every closed basin on land so that each land cell has a strictly
/// descending path to the sea or the grid edge. Cells at or below sea level and
/// cells on the grid edge are outlets and are never modified.
public enum PriorityFlood {
    public static func fill(_ h: inout Grid, seaLevel: Float, epsilon: Float = 1e-3) {
        let n = h.count
        var visited = [Bool](repeating: false, count: n)
        var heap = MinHeap(reserve: n / 4)

        let w = h.width
        for i in 0..<n where h.isBoundary(i) || h.values[i] <= seaLevel { visited[i] = true }
        // Seed the heap only with outlets that touch land. Open sea (outlets surrounded
        // by outlets) can never raise anything, and on an island it is most of the grid,
        // so leaving it out shrinks the heap several-fold. The pop order of the remaining
        // cells, and so the result, is unchanged.
        for i in 0..<n where visited[i] {
            let x = i % w, y = i / w
            for k in 0..<8 {
                let nx = x + D8.dx[k], ny = y + D8.dy[k]
                if h.contains(nx, ny) && !visited[ny * w + nx] {
                    heap.push(h.values[i], i)
                    break
                }
            }
        }

        while !heap.isEmpty {
            let c = heap.pop()
            let cx = c % w, cy = c / w
            let hc = h.values[c]
            for k in 0..<8 {
                let nx = cx + D8.dx[k], ny = cy + D8.dy[k]
                guard h.contains(nx, ny) else { continue }
                let j = ny * w + nx
                if visited[j] { continue }
                visited[j] = true
                let minimum = hc + epsilon
                if h.values[j] < minimum { h.values[j] = minimum }
                heap.push(h.values[j], j)
            }
        }
    }
}
