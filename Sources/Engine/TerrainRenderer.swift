#if canImport(MetalKit)
import Metal
import MetalKit
import simd
import IslandCore

/// First terrain renderer: one static mesh of the whole island at a fixed resolution.
/// It exists to look at baked islands while the generator is tuned; the streaming,
/// LOD-morphing renderer from the plan replaces it.
public final class TerrainRenderer {
    struct Uniforms {
        var viewProjection: simd_float4x4
        var cameraPosition: SIMD3<Float>
        var seaLevel: Float
        var sunDirection: SIMD3<Float>
        var maxHeight: Float
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let vertices: MTLBuffer
    let indices: MTLBuffer
    let indexCount: Int
    let seaLevel: Float
    let maxHeight: Float

    /// - Parameter meshResolution: vertices per side of the rendered mesh (the height grid
    ///   is resampled to this). 1024 keeps the mesh around 2 million triangles.
    public init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat,
                height: Grid, seaLevel: Float, meshResolution: Int = 1024) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw RendererError.setup("command queue") }
        self.queue = queue
        self.seaLevel = seaLevel
        self.maxHeight = height.range.max

        let library = try device.makeLibrary(source: terrainShaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "terrainVertex")
        desc.fragmentFunction = library.makeFunction(name: "terrainFragment")
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = depthFormat
        pipeline = try device.makeRenderPipelineState(descriptor: desc)

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("depth state") }
        depthState = ds

        // Build the mesh: positions in metres, normals from central differences.
        let n = Swift.min(meshResolution, height.width)
        let scale = Double(height.width - 1) / Double(n - 1)
        let spacing = Float(height.cellSize * scale)
        var verts = [Float]()
        verts.reserveCapacity(n * n * 6)
        func h(_ i: Int, _ j: Int) -> Float {
            height.sample(x: Double(Swift.min(Swift.max(i, 0), n - 1)) * scale,
                          y: Double(Swift.min(Swift.max(j, 0), n - 1)) * scale)
        }
        for j in 0..<n {
            for i in 0..<n {
                let z = h(i, j)
                let nx = (h(i - 1, j) - h(i + 1, j)) / (2 * spacing)
                let ny = (h(i, j - 1) - h(i, j + 1)) / (2 * spacing)
                let normal = simd_normalize(SIMD3<Float>(nx, ny, 1))
                let px: Float = Float(i) * spacing
                let py: Float = Float(j) * spacing
                verts.append(contentsOf: [px, py, z, normal.x, normal.y, normal.z])
            }
        }
        var idx = [UInt32]()
        idx.reserveCapacity((n - 1) * (n - 1) * 6)
        for j in 0..<(n - 1) {
            for i in 0..<(n - 1) {
                let a = UInt32(j * n + i), b = a + 1, c = a + UInt32(n), d = c + 1
                idx.append(contentsOf: [a, b, d, a, d, c])
            }
        }
        guard let vb = device.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared)
        else { throw RendererError.setup("buffers") }
        vertices = vb
        indices = ib
        indexCount = idx.count
    }

    /// Renders one frame from the given eye position (metres) and view direction.
    public func draw(in view: MTKView, eye: SIMD3<Double>, forward: SIMD3<Float>) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }

        let size = view.drawableSize
        let aspect = Float(size.width / Swift.max(size.height, 1))
        let proj = Matrix.perspective(fovY: 60 * .pi / 180, aspect: aspect, near: 0.3, far: 150_000)
        var u = Uniforms(
            viewProjection: proj * Matrix.lookDirection(forward),
            cameraPosition: SIMD3<Float>(Float(eye.x), Float(eye.y), Float(eye.z)),
            seaLevel: seaLevel,
            sunDirection: simd_normalize(SIMD3<Float>(-0.4, 0.5, 0.6)),
            maxHeight: maxHeight)

        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)
        enc.setVertexBuffer(vertices, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint32,
                                  indexBuffer: indices, indexBufferOffset: 0)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

public enum RendererError: Error {
    case setup(String)
}
#endif
