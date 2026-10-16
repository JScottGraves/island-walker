import Dispatch
import Foundation

/// Runs independent chunks of work on all cores. Every caller splits its work so that
/// chunks touch disjoint cells, which keeps results bit-identical to a serial run.
public enum Parallel {
    /// Number of chunks to split row- or basin-parallel work into.
    public static let chunkCount = Swift.max(1, ProcessInfo.processInfo.activeProcessorCount * 4)

    /// Calls `body` once per chunk of `0..<count`, concurrently.
    public static func forRanges(_ count: Int, chunks: Int = chunkCount, _ body: (Range<Int>) -> Void) {
        guard count > 0 else { return }
        let k = Swift.min(chunks, count)
        DispatchQueue.concurrentPerform(iterations: k) { c in
            body((c * count / k)..<((c + 1) * count / k))
        }
    }
}

/// Wall-clock time spent in named phases, summed over calls. Not thread-safe: record from
/// one thread (the phases themselves may run in parallel inside).
public final class PhaseTimer {
    public private(set) var seconds: [String: Double] = [:]
    public private(set) var order: [String] = []
    public init() {}

    @discardableResult
    public func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        defer {
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            if seconds[name] == nil { order.append(name) }
            seconds[name, default: 0] += elapsed
        }
        return try body()
    }

    public var report: String {
        let total = order.reduce(0) { $0 + seconds[$1, default: 0] }
        return order.map { name in
            let s = seconds[name, default: 0]
            let label = name.padding(toLength: 12, withPad: " ", startingAt: 0)
            return label + String(format: " %7.2f s  %5.1f%%", s, total > 0 ? 100 * s / total : 0)
        }.joined(separator: "\n")
    }
}

/// Lets disjoint-index writes from concurrent chunks share one array.
public struct SharedBuffer<T>: @unchecked Sendable {
    public let base: UnsafeMutablePointer<T>
    public init(_ buffer: UnsafeMutableBufferPointer<T>) { base = buffer.baseAddress! }
    @inline(__always) public subscript(i: Int) -> T {
        get { base[i] }
        nonmutating set { base[i] = newValue }
    }
}
