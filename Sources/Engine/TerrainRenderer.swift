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
        var inverseViewProjection: simd_float4x4
        var cameraPosition: SIMD3<Float>
        var seaLevel: Float
        var sunDirection: SIMD3<Float>
        var maxHeight: Float
        var sunRadiance: SIMD3<Float>
        var exposure: Float
        var skyIrradiance: SIMD3<Float>
        var bounce: Float
        var haze: SIMD3<Float>
        var pad0: Float = 0
        var zenith: SIMD3<Float>
        var pad1: Float = 0
    }

    /// How terrain casts sun shadows.
    public enum Shadows: String, Sendable {
        /// Rays traced against the terrain mesh in the fragment shader (Metal ray tracing;
        /// hardware-accelerated on M3 and later).
        case rayTraced
        /// The baked 16-sector horizon map: cheaper, coarser, works on any GPU.
        case horizonMap
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    let skyPipeline: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let skyDepthState: MTLDepthStencilState
    let vertices: MTLBuffer
    let indices: MTLBuffer
    let indexCount: Int
    let horizons: MTLBuffer
    let terrainAccel: MTLAccelerationStructure?
    let seaLevel: Float
    let maxHeight: Float
    /// Which shadow path this renderer uses (ray tracing unless unsupported or declined).
    public let shadows: Shadows
    /// The sun and sky to light the next frame with; set it to move the sun.
    public var light = SunLight.reference

    /// - Parameter meshResolution: vertices per side of the rendered mesh (the height grid
    ///   is resampled to this). 1024 keeps the mesh around 2 million triangles.
    public init(device: MTLDevice, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat,
                height: Grid, seaLevel: Float, meshResolution: Int = 1024,
                shadows requested: Shadows = .rayTraced) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw RendererError.setup("command queue") }
        self.queue = queue
        self.seaLevel = seaLevel
        self.maxHeight = height.range.max

        let shadows: Shadows = requested == .rayTraced && device.supportsRaytracingFromRender ? .rayTraced : .horizonMap
        self.shadows = shadows
        let library = try device.makeLibrary(source: terrainShaderSource, options: nil)
        let constants = MTLFunctionConstantValues()
        var rayTraced = shadows == .rayTraced
        constants.setConstantValue(&rayTraced, type: .bool, index: 0)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "terrainVertex")
        desc.fragmentFunction = try library.makeFunction(name: "terrainFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = depthFormat
        pipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.vertexFunction = library.makeFunction(name: "skyVertex")
        desc.fragmentFunction = library.makeFunction(name: "skyFragment")
        skyPipeline = try device.makeRenderPipelineState(descriptor: desc)

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("depth state") }
        depthState = ds
        let skyDepthDesc = MTLDepthStencilDescriptor()
        skyDepthDesc.depthCompareFunction = .always
        skyDepthDesc.isDepthWriteEnabled = false
        guard let sds = device.makeDepthStencilState(descriptor: skyDepthDesc) else { throw RendererError.setup("sky depth state") }
        skyDepthState = sds

        // Build the mesh: positions in metres, normals from central differences.
        let n = Swift.min(meshResolution, height.width)
        let scale = Double(height.width - 1) / Double(n - 1)
        let spacing = Float(height.cellSize * scale)
        // Sky and sun visibility (and, without ray tracing, the horizon map) from the
        // full-resolution heightfield, sampled per vertex.
        let lighting = TerrainLighting.bake(height: height, seaLevel: seaLevel, horizons: shadows == .horizonMap)
        let sectors = TerrainLighting.sectors
        var verts = [Float]()
        verts.reserveCapacity(n * n * 8)
        // 16 bytes per vertex either way; all zero (open horizon) when ray tracing.
        var horizonBytes = [UInt8](repeating: 0, count: n * n * sectors)
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
                // Grid sample k sits at the centre of cell k: (k + 0.5) · cellSize.
                let px = Float((Double(i) * scale + 0.5) * height.cellSize)
                let py = Float((Double(j) * scale + 0.5) * height.cellSize)
                let lx = Double(i) * scale, ly = Double(j) * scale
                verts.append(contentsOf: [px, py, z, normal.x, normal.y, normal.z,
                                          lighting.sky.sample(x: lx, y: ly), lighting.sun.sample(x: lx, y: ly)])
                if !lighting.horizon.isEmpty {
                    // Nearest cell: horizon angles don't interpolate meaningfully across a ridge.
                    let cx = Swift.min(height.width - 1, Int(lx.rounded())), cy = Swift.min(height.height - 1, Int(ly.rounded()))
                    let src = (cy * height.width + cx) * sectors, dst = (j * n + i) * sectors
                    for k in 0..<sectors { horizonBytes[dst + k] = lighting.horizon[src + k] }
                }
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
              let ib = device.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared),
              let hb = device.makeBuffer(bytes: horizonBytes, length: horizonBytes.count, options: .storageModeShared)
        else { throw RendererError.setup("buffers") }
        vertices = vb
        indices = ib
        horizons = hb
        indexCount = idx.count
        terrainAccel = shadows == .rayTraced
            ? try TerrainRenderer.buildAccelerationStructure(device: device, queue: queue, vertices: vb,
                                                              stride: 8 * 4, indices: ib, triangles: idx.count / 3)
            : nil
    }

    /// One bottom-level acceleration structure over the whole terrain mesh, built once and
    /// compacted. Shares the render vertex buffer: positions are its first 12 bytes.
    static func buildAccelerationStructure(device: MTLDevice, queue: MTLCommandQueue, vertices: MTLBuffer,
                                           stride: Int, indices: MTLBuffer, triangles: Int) throws -> MTLAccelerationStructure {
        let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometry.vertexBuffer = vertices
        geometry.vertexStride = stride
        geometry.vertexFormat = .float3
        geometry.indexBuffer = indices
        geometry.indexType = .uint32
        geometry.triangleCount = triangles
        geometry.opaque = true
        let desc = MTLPrimitiveAccelerationStructureDescriptor()
        desc.geometryDescriptors = [geometry]

        let sizes = device.accelerationStructureSizes(descriptor: desc)
        guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
              let compactedSize = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(), let enc = cmd.makeAccelerationStructureCommandEncoder()
        else { throw RendererError.setup("acceleration structure") }
        enc.build(accelerationStructure: accel, descriptor: desc, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.writeCompactedSize(accelerationStructure: accel, buffer: compactedSize, offset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let size = Int(compactedSize.contents().load(as: UInt32.self))
        guard let compact = device.makeAccelerationStructure(size: size),
              let cmd2 = queue.makeCommandBuffer(), let enc2 = cmd2.makeAccelerationStructureCommandEncoder()
        else { throw RendererError.setup("compacted acceleration structure") }
        enc2.copyAndCompact(sourceAccelerationStructure: accel, destinationAccelerationStructure: compact)
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return compact
    }

    /// Clear colour behind the sky pass (which covers every pixel): the horizon haze, tone
    /// mapped like the terrain (linear, since the target applies the sRGB encoding on write).
    public static var skyClearColor: MTLClearColor {
        // Only shows if the sky pass is skipped; the reference sun's haze is close enough.
        let c = Look.toneMap(Look.haze)
        return MTLClearColor(red: c.r, green: c.g, blue: c.b, alpha: 1)
    }

    /// Renders one frame into the view's drawable.
    public func draw(in view: MTKView, eye: SIMD3<Double>, forward: SIMD3<Float>) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }
        let size = view.drawableSize
        encode(cmd, pass: pass, aspect: Float(size.width / Swift.max(size.height, 1)), eye: eye, forward: forward)
        cmd.present(drawable)
        cmd.commit()
    }

    /// Renders one frame offscreen and returns it as 8-bit RGB, top row first.
    /// Lets tools and CI inspect what the app draws without a window or screen capture.
    public func snapshot(width: Int, height: Int, eye: SIMD3<Double>, forward: SIMD3<Float>,
                         colorFormat: MTLPixelFormat = .bgra8Unorm_srgb) throws -> [UInt8] {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: colorFormat, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget]
        colorDesc.storageMode = .shared
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .private
        guard let color = device.makeTexture(descriptor: colorDesc), let depth = device.makeTexture(descriptor: depthDesc),
              let cmd = queue.makeCommandBuffer() else { throw RendererError.setup("snapshot targets") }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = TerrainRenderer.skyClearColor
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        encode(cmd, pass: pass, aspect: Float(width) / Float(height), eye: eye, forward: forward)
        cmd.commit()
        cmd.waitUntilCompleted()

        var bgra = [UInt8](repeating: 0, count: width * height * 4)
        bgra.withUnsafeMutableBytes { buf in
            color.getBytes(buf.baseAddress!, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) {
            rgb[p * 3] = bgra[p * 4 + 2]
            rgb[p * 3 + 1] = bgra[p * 4 + 1]
            rgb[p * 3 + 2] = bgra[p * 4]
        }
        return rgb
    }

    private func encode(_ cmd: MTLCommandBuffer, pass: MTLRenderPassDescriptor, aspect: Float,
                        eye: SIMD3<Double>, forward: SIMD3<Float>) {
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        let proj = Matrix.perspective(fovY: 60 * .pi / 180, aspect: aspect, near: 0.3, far: 150_000)
        let viewProjection = proj * Matrix.lookDirection(forward)
        func f3(_ c: RGB) -> SIMD3<Float> { SIMD3(Float(c.r), Float(c.g), Float(c.b)) }
        let d = light.direction
        var u = Uniforms(
            viewProjection: viewProjection,
            inverseViewProjection: viewProjection.inverse,
            cameraPosition: SIMD3<Float>(Float(eye.x), Float(eye.y), Float(eye.z)),
            seaLevel: seaLevel,
            sunDirection: SIMD3<Float>(Float(d.x), Float(d.y), Float(d.z)),
            maxHeight: maxHeight,
            sunRadiance: f3(light.sun),
            exposure: Float(light.exposure),
            skyIrradiance: f3(light.sky),
            bounce: Float(Look.bounce),
            haze: f3(Look.haze * light.hazeTint),
            zenith: f3(Look.zenith * light.zenithTint))

        enc.setRenderPipelineState(skyPipeline)
        enc.setDepthStencilState(skyDepthState)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)
        enc.setVertexBuffer(vertices, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setVertexBuffer(horizons, offset: 0, index: 2)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        if let terrainAccel {
            enc.setFragmentAccelerationStructure(terrainAccel, bufferIndex: 2)
            enc.useResource(terrainAccel, usage: .read, stages: .fragment)
        }
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint32,
                                  indexBuffer: indices, indexBufferOffset: 0)
        enc.endEncoding()
    }
}

public enum RendererError: Error {
    case setup(String)
}
#endif
