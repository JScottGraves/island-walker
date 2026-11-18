#if canImport(MetalKit)
import Metal
import simd
import IslandCore

/// The structures that stand up off the ground (IslandCore Buildings: so far the airfield's
/// hangar), built once at load into one world-space mesh. Drawn after the terrain in the
/// depth prepass (writing the surface the sun rays trace from) and shaded with equal depth
/// like the trees; with ray tracing, `sunShadow` also traces each pixel near them against
/// their own acceleration structure, so they shade the ground, each other and themselves
/// (the sun through the door and the gaps in the roof).
final class BuildingRenderer {
    /// One vertex as BuildingVertex in MSL: world position and material, normal, pattern
    /// coordinates and height above the floor. Positions first, for the acceleration structure.
    struct Vertex {
        var position: SIMD4<Float>
        var normal: SIMD4<Float>
        var pattern: SIMD4<Float>
    }

    let vertices: MTLBuffer
    let indices: MTLBuffer
    let indexCount: Int
    let prepassPipeline: MTLRenderPipelineState
    let shadePipeline: MTLRenderPipelineState
    let shadeDepthState: MTLDepthStencilState
    /// Their triangles for the sun rays (nil without ray tracing).
    let accel: MTLAccelerationStructure?

    /// World-space triangles of every building among `structures`, or nil when there are none.
    static func mesh(_ structures: [Structure]) -> (vertices: [Vertex], indices: [UInt32])? {
        var vertices: [Vertex] = [], indices: [UInt32] = []
        for s in structures {
            guard let hangar = Hangar(s) else { continue }
            let m = hangar.mesh()
            let axis = s.axis
            let base = UInt32(vertices.count)
            for v in m.vertices {
                let u = Double(v.position.x), w = Double(v.position.y)
                let p = s.world(u: u, v: w)
                // +u along the axis, +v to its right: (ay, −ax).
                let nx = Double(v.normal.x) * axis.x + Double(v.normal.y) * axis.y
                let ny = Double(v.normal.x) * axis.y - Double(v.normal.y) * axis.x
                vertices.append(Vertex(position: SIMD4(Float(p.x), Float(p.y), Float(s.elevation) + v.position.z,
                                                       Float(v.material.rawValue)),
                                       normal: SIMD4(Float(nx), Float(ny), v.normal.z, 0),
                                       pattern: SIMD4(v.uv.x, v.uv.y, v.position.z, 0)))
            }
            indices += m.indices.map { $0 + base }
        }
        return vertices.isEmpty ? nil : (vertices, indices)
    }

    init?(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, constants: MTLFunctionConstantValues,
          colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat, rayTraced: Bool, structures: [Structure]) throws {
        guard let mesh = Self.mesh(structures) else { return nil }
        guard let vb = device.makeBuffer(bytes: mesh.vertices, length: mesh.vertices.count * MemoryLayout<Vertex>.stride,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: mesh.indices, length: mesh.indices.count * 4, options: .storageModeShared)
        else { throw RendererError.setup("building mesh") }
        vb.label = "buildings"
        vertices = vb
        indices = ib
        indexCount = mesh.indices.count

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "buildingVertex")
        desc.depthAttachmentPixelFormat = depthFormat
        if rayTraced {
            desc.fragmentFunction = library.makeFunction(name: "buildingSurface")
            desc.colorAttachments[0].pixelFormat = .rgba32Uint
        } else {
            desc.fragmentFunction = nil
            desc.colorAttachments[0].pixelFormat = .invalid
        }
        prepassPipeline = try device.makeRenderPipelineState(descriptor: desc)
        desc.fragmentFunction = try library.makeFunction(name: "buildingFragment", constantValues: constants)
        desc.colorAttachments[0].pixelFormat = colorFormat
        shadePipeline = try device.makeRenderPipelineState(descriptor: desc)
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .equal
        depthDesc.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { throw RendererError.setup("building depth state") }
        shadeDepthState = ds
        accel = rayTraced
            ? try TerrainRenderer.buildAccelerationStructure(device: device, queue: queue, vertices: vb,
                                                             stride: MemoryLayout<Vertex>.stride, indices: ib,
                                                             triangles: mesh.indices.count / 3)
            : nil
    }

    var triangleCount: Int { indexCount / 3 }

    private func draw(_ enc: MTLRenderCommandEncoder) {
        enc.setVertexBuffer(vertices, offset: 0, index: 2)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint32, indexBuffer: indices,
                                  indexBufferOffset: 0)
    }

    /// In the depth prepass, after the terrain (same depth state and uniforms).
    func drawPrepass(_ enc: MTLRenderCommandEncoder) {
        enc.setRenderPipelineState(prepassPipeline)
        draw(enc)
    }

    /// In the shading pass, after the terrain, whose fragment textures (sky view, clouds,
    /// cloud shadows, traced sun) and structure map stay bound; this adds the ground's sky
    /// visibility and horizon map.
    func drawShading(_ enc: MTLRenderCommandEncoder, skyVisibility: MTLTexture, horizons: MTLTexture) {
        enc.setRenderPipelineState(shadePipeline)
        enc.setDepthStencilState(shadeDepthState)
        enc.setFragmentTexture(skyVisibility, index: 8)
        enc.setFragmentTexture(horizons, index: 9)
        draw(enc)
    }
}
#endif
