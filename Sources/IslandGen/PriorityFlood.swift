import IslandCore

/// Binary min-heap keyed on (height, index). Ties break on index so the
/// processing order, and therefore the output, is fully deterministic.
struct MinHeap {
    private var keys: [Float] = []
    private var items: [Int32] = []

    var isEmpty: Bool { items.isEmpty }

    init(reserve: Int) {
        keys.reserveCapacity(reserve)
        items.reserveCapacity(reserve)
    }

    @inline(__always)
    private func less(_ a: Int, _ b: Int) -> Bool {
        keys[a] < keys[b] || (keys[a] == keys[b] && items[a] < items[b])
    }

    mutating func push(_ key: Float, _ item: Int) {
        keys.append(key)
        items.append(Int32(item))
        var c = items.count - 1
        while c > 0 {
            let p = (c - 1) / 2
            guard less(c, p) else { break }
            keys.swapAt(c, p)
            items.swapAt(c, p)
            c = p
        }
    }

    mutating func pop() -> Int {
        let top = Int(items[0])
        let last = items.count - 1
        keys.swapAt(0, last)
        items.swapAt(0, last)
        keys.removeLast()
        items.removeLast()
        var p = 0
        let n = items.count
        while true {
            let l = 2 * p + 1, r = l + 1
            var m = p
            if l < n && less(l, m) { m = l }
            if r < n && less(r, m) { m = r }
            if m == p { break }
            keys.swapAt(p, m)
            items.swapAt(p, m)
            p = m
        }
        return top
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

        for i in 0..<n where h.isBoundary(i) || h.values[i] <= seaLevel {
            visited[i] = true
            heap.push(h.values[i], i)
        }

        let w = h.width
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
