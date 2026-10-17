import Foundation
import IslandCore

/// Where each native plant community grows (research/vegetation.md §2, §5).
///
/// Hawaiian vegetation sorts by moisture first, then height, exposure and landform.
/// Moisture follows Price et al. (USGS 2012): annual rain minus potential evaporation,
/// with the dry/mesic line at 0 and the mesic/wet line at 1,661 mm. Evaporation falls
/// with height and under standing cloud, so the same rain is wetter higher up.
///
/// Each land cell gets a weight per community, summing to 1, with soft edges and a
/// wandering low-frequency offset so boundaries read as ecotones, not contour lines.
/// Weights (not labels) are baked because they mip-map correctly in the bundle.
public enum Vegetation {
    public enum Community: Int, CaseIterable, Sendable {
        case strand, rockyShore, marsh, dryForest, mesicForest, wetForest
        case riparian, cloudShrubland, bog, cliff, montane

        /// Bundle channel name.
        public var channel: String {
            switch self {
            case .strand: return "biome-strand"
            case .rockyShore: return "biome-rocky-shore"
            case .marsh: return "biome-marsh"
            case .dryForest: return "biome-dry-forest"
            case .mesicForest: return "biome-mesic-forest"
            case .wetForest: return "biome-wet-forest"
            case .riparian: return "biome-riparian"
            case .cloudShrubland: return "biome-cloud-shrubland"
            case .bog: return "biome-bog"
            case .cliff: return "biome-cliff"
            case .montane: return "biome-montane"
            }
        }

        /// Short name for info.json.
        public var name: String { String(channel.dropFirst("biome-".count)) }

        /// Preview colour, picked to read as the community from above (sRGB 0...1).
        var colour: (Double, Double, Double) {
            switch self {
            case .strand: return (0.93, 0.86, 0.62)
            case .rockyShore: return (0.55, 0.52, 0.48)
            case .marsh: return (0.45, 0.70, 0.62)
            case .dryForest: return (0.72, 0.62, 0.38)
            case .mesicForest: return (0.48, 0.60, 0.30)
            case .wetForest: return (0.13, 0.42, 0.20)
            case .riparian: return (0.05, 0.55, 0.45)
            case .cloudShrubland: return (0.55, 0.62, 0.70)
            case .bog: return (0.70, 0.55, 0.65)
            case .cliff: return (0.30, 0.26, 0.24)
            case .montane: return (0.20, 0.32, 0.40)
            }
        }
    }

    public struct Params: Sendable {
        /// Direction the wind blows towards, in radians (same convention as `Climate`).
        public var windDirection: Double = 5 * Double.pi / 4
        /// Potential evaporation at sea level under clear sky, mm/yr.
        public var seaLevelPET: Double = 1700
        /// Drop in potential evaporation per metre of height, mm/yr. Gives ~800 mm at
        /// 1,000 m, where Price et al. put the dry/mesic line.
        public var petLapse: Double = 0.9
        /// Share of potential evaporation removed under full standing cloud.
        public var cloudPETCut: Double = 0.4
        /// Moisture availability (rain − PET, mm/yr) at the dry/mesic and mesic/wet lines.
        public var mesicMAI: Double = 0
        public var wetMAI: Double = 1661
        /// Half-width of each moisture ecotone, mm/yr.
        public var moistureBlend: Double = 200
        /// Trade-wind cloud base in metres: summit shrubland and montane forest start here.
        public var cloudBase: Double = 600
        /// Height above which mountain forest turns low montane, metres.
        public var montaneHeight: Double = 900
        /// Width of the salt-spray strand from the shoreline, metres.
        public var strandWidth: Double = 150
        /// Slope in degrees where pali cliff vegetation starts and where it is complete.
        public var cliffSlope: (start: Double, full: Double) = (32, 45)
        /// Drainage area (m²) from which a valley floor counts as riparian.
        public var riparianArea: Double = 2e6
        /// Wavelength of the boundary wander, metres, and its size in MAI (mm/yr) and height (m).
        public var wanderWavelength: Double = 1500
        public var wanderMAI: Double = 300
        public var wanderHeight: Double = 80
        public var seed: UInt64 = 1
        public init() {}
    }

    public struct Result: Sendable {
        /// One weight grid per community, in `Community.allCases` order; zero over the sea.
        public var weights: [Grid]
        /// Moisture availability, rain − PET, mm/yr.
        public var mai: Grid
        /// Wind exposure of crests in 0...1.
        public var exposure: Grid

        public func weight(_ c: Community) -> Grid { weights[c.rawValue] }

        /// Share of land area per community.
        public func landShares(height: Grid, seaLevel: Float) -> [Community: Double] {
            var sums = [Double](repeating: 0, count: Community.allCases.count)
            var land = 0.0
            for i in 0..<height.count where height.values[i] > seaLevel {
                land += 1
                for k in 0..<sums.count { sums[k] += Double(weights[k].values[i]) }
            }
            var out = [Community: Double]()
            for c in Community.allCases { out[c] = sums[c.rawValue] / Swift.max(land, 1) }
            return out
        }
    }

    /// Community weights on `height`'s grid. `rain` (m/yr) and `clouds` (0...1) may be on
    /// other grids covering the same domain; they are sampled by position.
    public static func bake(height: Grid, rain: Grid, clouds: Grid, seaLevel: Float, params p: Params = Params()) -> Result {
        let w = height.width, rows = height.height, n = height.count
        let cell = height.cellSize
        let coast = Island.seaDistance(height, seaLevel: seaLevel)
        let area = drainageArea(height, seaLevel: seaLevel)

        // Crests: height above the surrounding ~600 m mean, which marks ridgelines and
        // summits the trade wind hits whatever their aspect.
        var surface = height
        for i in 0..<n { surface.values[i] = Swift.max(surface.values[i], seaLevel) }
        let r = Swift.max(1, Int(250 / cell))
        let around = Climate.boxBlur(Climate.boxBlur(Climate.boxBlur(surface, radius: r), radius: r), radius: r)

        var weights = [Grid](repeating: Grid(width: w, height: rows, cellSize: cell), count: Community.allCases.count)
        var mai = Grid(width: w, height: rows, cellSize: cell)
        var exposure = Grid(width: w, height: rows, cellSize: cell)
        let wanderSeed = Hash.mix(p.seed ^ 0x5645_4745)
        let toRain = (x: Double(rain.width) / Double(w), y: Double(rain.height) / Double(rows))
        let toCloud = (x: Double(clouds.width) / Double(w), y: Double(clouds.height) / Double(rows))
        let windX = cos(p.windDirection), windY = sin(p.windDirection)
        var zonal = [Double](repeating: 0, count: Community.allCases.count)
        var overlay = zonal

        for y in 0..<rows {
            for x in 0..<w {
                let i = y * w + x
                let z = Double(height.values[i] - seaLevel)
                guard z > 0 else { continue }
                let xl = Swift.max(x - 1, 0), xr = Swift.min(x + 1, w - 1)
                let yd = Swift.max(y - 1, 0), yu = Swift.min(y + 1, rows - 1)
                let gx = Double(height[xr, y] - height[xl, y]) / (Double(xr - xl) * cell)
                let gy = Double(height[x, yu] - height[x, yd]) / (Double(yu - yd) * cell)
                let slope = atan((gx * gx + gy * gy).squareRoot()) * 180 / .pi

                let rainMM = 1000 * Double(rain.sample(x: (Double(x) + 0.5) * toRain.x - 0.5, y: (Double(y) + 0.5) * toRain.y - 0.5))
                let cloud = Double(clouds.sample(x: (Double(x) + 0.5) * toCloud.x - 0.5, y: (Double(y) + 0.5) * toCloud.y - 0.5))

                // Boundary wander: two independent noise fields so moisture and height
                // edges do not move in lockstep.
                let nx = Double(x) * cell / p.wanderWavelength, ny = Double(y) * cell / p.wanderWavelength
                let wanderM = p.wanderMAI * Noise.fbm(nx, ny, seed: wanderSeed, octaves: 3)
                let wanderZ = p.wanderHeight * Noise.fbm(nx, ny, seed: wanderSeed &+ 1, octaves: 3)
                let zw = z + wanderZ

                let pet = Swift.max(300, p.seaLevelPET - p.petLapse * z) * (1 - p.cloudPETCut * cloud)
                let m = rainMM - pet
                mai.values[i] = Float(m)
                let mw = m + wanderM

                // Crests facing into the wind, or level along it, take the full trades; the
                // lee side of a ridge is sheltered.
                let crest = smoothstep(5, 40, Double(height.values[i] - around.values[i]))
                let ax = Double(around[xr, y] - around[xl, y]) / (Double(xr - xl) * cell)
                let ay = Double(around[x, yu] - around[x, yd]) / (Double(yu - yd) * cell)
                let windward = smoothstep(-0.1, 0.05, ax * windX + ay * windY)
                let exposed = crest * windward * smoothstep(p.cloudBase - 200, p.cloudBase + 200, zw) * (0.3 + 0.7 * cloud)
                exposure.values[i] = Float(exposed)

                // Zonal forest by moisture and height.
                let dry = 1 - smoothstep(p.mesicMAI - p.moistureBlend, p.mesicMAI + p.moistureBlend, mw)
                let wet = smoothstep(p.wetMAI - p.moistureBlend, p.wetMAI + p.moistureBlend, mw)
                let mesic = Swift.max(0, 1 - dry - wet)
                let montane = smoothstep(p.montaneHeight - 100, p.montaneHeight + 100, zw) * (1 - dry)
                for k in 0..<zonal.count { zonal[k] = 0; overlay[k] = 0 }
                zonal[Community.dryForest.rawValue] = dry
                zonal[Community.mesicForest.rawValue] = mesic * (1 - montane)
                zonal[Community.wetForest.rawValue] = wet * (1 - montane)
                zonal[Community.montane.rawValue] = montane

                // Landform overlays, which take their share before the zonal forest.
                let d = coast[i] * cell
                let nearShore = 1 - smoothstep(p.strandWidth * 0.5, p.strandWidth * 1.5, d)
                let gentle = 1 - smoothstep(4, 12, slope)
                let low = 1 - smoothstep(8, 25, z)
                let logArea = log10(Swift.max(Double(area[i]), 1))
                let riparianLog = log10(p.riparianArea)
                overlay[Community.strand.rawValue] = nearShore * gentle * low
                overlay[Community.rockyShore.rawValue] = nearShore * (1 - gentle * low)
                overlay[Community.marsh.rawValue] = (1 - smoothstep(2, 6, z)) * (1 - smoothstep(1, 3, slope))
                    * smoothstep(6.3, 7, logArea) * (1 - smoothstep(400, 1200, d))
                overlay[Community.cliff.rawValue] = smoothstep(p.cliffSlope.start, p.cliffSlope.full, slope)
                overlay[Community.riparian.rawValue] = smoothstep(riparianLog - 0.3, riparianLog + 0.3, logArea)
                    * (1 - smoothstep(10, 25, slope)) * smoothstep(p.mesicMAI, p.mesicMAI + 2 * p.moistureBlend, mw)
                    * smoothstep(p.strandWidth, 2 * p.strandWidth, d)
                overlay[Community.cloudShrubland.rawValue] = smoothstep(0.15, 0.5, exposed) * wet
                overlay[Community.bog.rawValue] = smoothstep(p.montaneHeight - 200, p.montaneHeight, zw)
                    * (1 - smoothstep(2, 6, slope)) * smoothstep(p.wetMAI, p.wetMAI + 1000, mw) * crest

                let overlaySum = overlay.reduce(0, +)
                let overlayScale = overlaySum > 1 ? 1 / overlaySum : 1
                let zonalShare = 1 - Swift.min(overlaySum, 1)
                let zonalSum = Swift.max(zonal.reduce(0, +), 1e-9)
                for k in 0..<zonal.count {
                    weights[k].values[i] = Float(overlay[k] * overlayScale + zonal[k] / zonalSum * zonalShare)
                }
            }
        }
        return Result(weights: weights, mai: mai, exposure: exposure)
    }

    @inline(__always)
    static func smoothstep(_ a: Double, _ b: Double, _ x: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    }

    /// Plain (not rain-weighted) D8 drainage area in m² on a pit-filled copy of `h`, so
    /// generated islands and imported DEMs are treated alike.
    public static func drainageArea(_ h: Grid, seaLevel: Float) -> [Float] {
        var filled = h
        PriorityFlood.fill(&filled, seaLevel: seaLevel)
        let w = h.width, rows = h.height, n = h.count
        let order = (0..<n).sorted { filled.values[$0] > filled.values[$1] }
        var area = [Float](repeating: Float(h.cellArea), count: n)
        for i in order where filled.values[i] > seaLevel {
            let x = i % w, y = i / w
            var best = 0.0, target = -1
            for k in 0..<8 {
                let nx = x + D8.dx[k], ny = y + D8.dy[k]
                guard nx >= 0, ny >= 0, nx < w, ny < rows else { continue }
                let drop = Double(filled.values[i] - filled[nx, ny]) / D8.distance[k]
                if drop > best { best = drop; target = ny * w + nx }
            }
            if target >= 0 { area[target] += area[i] }
        }
        return area
    }

    /// Mean of `factor`×`factor` blocks, for baking biomes of a fine DEM on a coarse grid.
    public static func downsample(_ g: Grid, factor f: Int) -> Grid {
        guard f > 1 else { return g }
        let w = g.width / f, rows = g.height / f
        var out = Grid(width: w, height: rows, cellSize: g.cellSize * Double(f))
        let norm = 1 / Float(f * f)
        for y in 0..<rows {
            for x in 0..<w {
                var sum: Float = 0
                for dy in 0..<f { for dx in 0..<f { sum += g[x * f + dx, y * f + dy] } }
                out[x, y] = sum * norm
            }
        }
        return out
    }
}

extension Preview {
    /// Community weights blended into one colour per cell over a soft hillshade.
    public static func biomes(_ v: Vegetation.Result, height: Grid, seaLevel: Float) -> Image {
        var out = hillshade(height, seaLevel: seaLevel)
        let w = height.width, rows = height.height
        for y in 0..<rows {
            for x in 0..<w {
                let i = y * w + x
                guard height.values[i] > seaLevel else { continue }
                var c = (0.0, 0.0, 0.0)
                for community in Vegetation.Community.allCases {
                    let k = Double(v.weights[community.rawValue].values[i])
                    c.0 += community.colour.0 * k; c.1 += community.colour.1 * k; c.2 += community.colour.2 * k
                }
                let o = ((rows - 1 - y) * w + x) * 3
                // Keep some relief: scale the colour by the hillshade's brightness.
                let base = (Double(out.rgb[o]) + Double(out.rgb[o + 1]) + Double(out.rgb[o + 2])) / (3 * 255)
                let shade = 0.45 + 0.9 * base
                out.rgb[o] = UInt8(Swift.max(0, Swift.min(255, (c.0 * shade * 255).rounded())))
                out.rgb[o + 1] = UInt8(Swift.max(0, Swift.min(255, (c.1 * shade * 255).rounded())))
                out.rgb[o + 2] = UInt8(Swift.max(0, Swift.min(255, (c.2 * shade * 255).rounded())))
            }
        }
        return out
    }
}
