/// Minimal dependency-free PNG writer (8-bit RGB, uncompressed deflate blocks).
/// Works on macOS and Linux alike, so previews come out the same everywhere.
/// Files are larger than a compressing encoder would make; fine for previews.
public enum PNG {
    public static func encode(rgb: [UInt8], width: Int, height: Int) -> [UInt8] {
        precondition(rgb.count == width * height * 3, "pixel buffer size mismatch")

        // Scanlines, each prefixed with filter type 0 (none).
        let stride = width * 3
        var raw = [UInt8]()
        raw.reserveCapacity((stride + 1) * height)
        for y in 0..<height {
            raw.append(0)
            raw.append(contentsOf: rgb[(y * stride)..<((y + 1) * stride)])
        }

        // zlib stream made of stored (uncompressed) deflate blocks.
        var z: [UInt8] = [0x78, 0x01]
        z.reserveCapacity(raw.count + raw.count / 65535 * 5 + 16)
        var i = 0
        repeat {
            let n = Swift.min(65535, raw.count - i)
            z.append(i + n == raw.count ? 1 : 0)
            z.append(UInt8(n & 0xFF))
            z.append(UInt8(n >> 8))
            z.append(UInt8(~n & 0xFF))
            z.append(UInt8((~n >> 8) & 0xFF))
            z.append(contentsOf: raw[i..<(i + n)])
            i += n
        } while i < raw.count
        appendBigEndian(&z, adler32(raw))

        var ihdr = [UInt8]()
        appendBigEndian(&ihdr, UInt32(width))
        appendBigEndian(&ihdr, UInt32(height))
        ihdr += [8, 2, 0, 0, 0] // 8-bit, truecolour RGB, deflate, no filter, no interlace

        var out: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        chunk(&out, "IHDR", ihdr)
        chunk(&out, "IDAT", z)
        chunk(&out, "IEND", [])
        return out
    }

    private static func chunk(_ out: inout [UInt8], _ type: String, _ data: [UInt8]) {
        appendBigEndian(&out, UInt32(data.count))
        let typeBytes = Array(type.utf8)
        out += typeBytes
        out += data
        appendBigEndian(&out, crc32(typeBytes + data))
    }

    private static func appendBigEndian(_ out: inout [UInt8], _ v: UInt32) {
        out += [UInt8(v >> 24), UInt8((v >> 16) & 0xFF), UInt8((v >> 8) & 0xFF), UInt8(v & 0xFF)]
    }

    private static let crcTable: [UInt32] = (0..<256).map { n -> UInt32 in
        var c = UInt32(n)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32(_ bytes: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in bytes { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }

    static func adler32(_ bytes: [UInt8]) -> UInt32 {
        var a: UInt32 = 1, b: UInt32 = 0
        var i = 0
        while i < bytes.count {
            // 5552 is the largest block that cannot overflow before the modulo.
            let end = Swift.min(i + 5552, bytes.count)
            while i < end {
                a += UInt32(bytes[i])
                b += a
                i += 1
            }
            a %= 65521
            b %= 65521
        }
        return (b << 16) | a
    }
}
