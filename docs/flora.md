# Native trees: floragen and the tree renderer (vegetation step 2)

Step 2 of the plan in `research/vegetation.md` §6.5, which Scott approved on 2026-10-17. This
step adds the first six native species, built procedurally, scattered from the baked biome
weights and drawn by the renderer. Ground cover is step 3.

Scope rules carried over from step 1:
- Native and indigenous Hawaiian flora only, as it was before human contact. No canoe
  plants, no introduced species.
- Hybrid assets: shapes are generated procedurally, and leaf, bark and frond textures come
  from CC-licensed photos (CC0 / CC BY / CC BY-SA). Credits go in `Assets/flora/CREDITS.md`.
- Frame budget: at most **2.0 ms GPU at 2560×1440 on the M3** in the worst forest view.
  Time it with lighting and clouds on.

## 1. Pieces

| Piece | Where | What it does |
|---|---|---|
| `Flora` library | `Sources/Flora` (depends on IslandCore) | Species table, procedural plant builder, LOD meshes, community composition table. Pure CPU, deterministic for a seed. |
| `floragen` | `Sources/FloraGen` (executable) | Specimen sheets: each species and variant rendered with the app's own Metal shaders on a neutral ground, next to a scale figure. Also `--obj DIR` to export meshes for inspection. |
| Textures | `Assets/flora/` (SwiftPM resources of `Flora`) | Alpha-cut leaf, leaf-cluster and frond textures plus tileable bark, cropped from CC photos and kept small (≤512² per texture). `CREDITS.md` gives source URL, author and licence for each one. |
| Scatter | `Engine/FloraScatter.swift`, compute kernel | Builds this frame's tree instances on the GPU from a world-space hash grid and the biome weights, and fills indirect draw arguments. |
| Tree renderer | `Engine/FloraRenderer.swift` + `FloraShaders.swift` | Instanced, alpha-tested meshes in the depth prepass and in the shading pass. |

The app builds the species meshes at start-up from the `Flora` library. That takes a
fraction of a second and needs no asset pack on disk beyond the textures. If start-up
cost grows, move the meshes into an offline pack written by `floragen`.

## 2. Species (first six)

Heights are for mature plants in good conditions. The scatter scales them by site (§4).

| Species | Form | Height | Builder notes |
|---|---|---|---|
| **ʻŌhiʻa lehua** (*Metrosideros polymorpha*) | Gnarled, often multi-stemmed tree. Irregular, open, rounded crown made of tufts of small leaves at the branch tips. Red pompom lehua flowers. Grey-brown flaky bark. | 15–20 m in wet valleys, 5–10 m mesic, 0.5–3 m dwarf on the summit and bog | Branching tree, sympodial, with crooked segments. Leaf-cluster cards at the tips, with a few red flower clusters. The **dwarf form** is the same species at small scale with denser, more shrub-like branching. |
| **Koa** (*Acacia koa*) | Large tree. Thick trunk that often forks low, with a broad, spreading, flat-topped crown. Grey-green sickle-shaped phyllodes. Rough grey bark. | 15–25 m | Branching tree with wide branch angles and a dense crown. Phyllode-cluster cards. |
| **Loulu** (*Pritchardia*; Oʻahu's *P. martii*, *P. kaalae*) | Fan palm. Single smooth grey trunk about 25 cm across. Crown of 20–30 stiff palmate leaves 1–1.5 m wide on 1 m petioles. Sometimes a short skirt of dead leaves. | 5–10 m | Palm rule: a slightly curved trunk plus a crown of fan leaves. Each fan is a pleated, radially split disc mesh with an alpha-cut texture. |
| **Hāpuʻu** (*Cibotium chamissoi*) | Tree fern. Thick, fibrous dark-brown trunk with golden pulu wool at the top. Crown of 8–15 arching bipinnate fronds 2–3 m long, light green. | 2–6 m | Rosette rule: a trunk plus arching frond strips (a curved ribbon mesh with a frond texture), with uncurling fiddleheads at the centre. |
| **Naupaka kahakai** (*Scaevola taccada*) | Dense mounding shrub, wider than tall. Glossy, succulent, bright-green spatulate leaves crowded at the stem tips. White half-flowers. | 1–3 m tall, 2–5 m wide | Shrub rule: many short stems from the base, leaf rosettes at the tips, and a dense hemispherical outer shell of leaf cards. |
| **ʻAʻaliʻi** (*Dodonaea viscosa*) | Shrub or small tree. Narrow, glossy lanceolate leaves. Papery winged seed capsules in red, yellow or brown. Stringy grey bark. | 1–5 m (up to 8 m) | Branching shrub or small tree. Leaf-cluster cards with capsule clusters. |

Each species has **4 variants** (seeds). Each variant has two meshes:
- **LOD0**: full branches and every leaf card. Roughly 8–20k triangles for a tree.
- **LOD1**: about 15% of the triangles. Branches only down to order 2, merged larger
  cluster cards, and trunk tubes with 4–6 sides.

Drawing beyond the mesh range is §5.

## 3. Communities → species

Abundance is the share of stems in a community's canopy layer. Density is mature stems
per hectare. Shrub species are scattered on their own finer grid (§4).

| Community | Canopy (stems/ha) | Species shares | Shrub layer |
|---|---|---|---|
| strand | — | — | naupaka dense near the shore, ʻaʻaliʻi sparse inland |
| rocky shore | — | — | naupaka sparse |
| marsh | — | — | none yet (sedges are ground cover, step 3) |
| dry forest | 120 | loulu .40, ʻaʻaliʻi (tree form) .45, koa .10, ʻōhiʻa .05 | ʻaʻaliʻi shrubs |
| mesic forest | 350 | koa .45, ʻōhiʻa .35, loulu .10, ʻaʻaliʻi .10 | ʻaʻaliʻi |
| wet forest | 450 | ʻōhiʻa .80, koa .12, loulu .08 | hāpuʻu dense (subcanopy) |
| riparian | 400 | ʻōhiʻa .60 (tallest), koa .25, loulu .15 | hāpuʻu dense |
| cloud shrubland | 600 (dwarf) | ʻōhiʻa dwarf .85, loulu .15 | — |
| bog | 80 (dwarf) | ʻōhiʻa dwarf 1.0 | — |
| cliff | 40 | ʻōhiʻa .5, loulu .3, ʻaʻaliʻi .2 | sparse |
| montane | 300 | koa .55, ʻōhiʻa .45 | — |

These are first guesses from the research doc and get tuned by eye against reference
photos. Keep the table in one place (`Flora.Communities`) so it is easy to change.

## 4. Scatter

- **Hash grid in world space**: one candidate per cell, jittered inside its cell and
  seeded from the cell's integer coordinates and a layer id. A tree therefore always
  stands in the same place on revisits, and nothing is stored per tree.
  - Canopy layer: 5 m cells. Shrub/subcanopy layer: 2.5 m cells, nearer range.
- For each candidate:
  1. Sample the 11 biome weights (bilinear, packed into 3 RGBA textures at the bundle's
     ≤1024² mip). Pick a community by weighted random. The weights already carry the
     ecotone noise.
  2. Accept it with probability density × cell area. Then pick a species from that
     community's shares, and a variant by hash.
  3. Reject on steep slope (no canopy trees above ~40°, shrubs to ~55°) and below sea level.
  4. Set scale from the species' height range × the site factor: wetter (higher MAI) and
     more sheltered (lower exposure) gives taller trees. Add a random yaw and a slight lean.
  5. Take ground height from the same surface the terrain draws (`surfaceHeight`, with
     the detail clipmap near the camera), so trunks meet the ground.
  6. Cull against the frustum. Pick a LOD by distance (with screen-size bias), then append
     to that (species, variant, LOD) bucket's instance range with an atomic counter, and
     write the indirect draw arguments.
- The scatter runs as a compute pass at the start of the frame. The same instances feed
  the prepass and the shading pass, so depth matches exactly.
- `--flora off` (app and snapshots) skips all of it, for A/B timing.

## 5. Drawing

- **Prepass**: trees draw after the terrain, alpha-tested, and write depth plus the
  `rgba32Uint` surface (position + oct normal). The ray-traced sun pass therefore shades
  tree pixels with terrain shadows at no extra cost, and the cloud march stops at trees.
- **Shading pass**: after the terrain, with depth `lessEqual` and `invariant` positions
  (same vertex code as the prepass).
  - Lighting reuses the terrain's tail: sun × traced visibility × cloud sunlight, sky SH,
    haze, `compositeClouds` and tone mapping. Factor that tail into one shared function so
    the terrain and the trees can't drift apart. The lighting thread owns that code, so
    coordinate with it.
  - Foliage: two-sided cards, normals blended toward the crown's ellipsoid normal (soft,
    volumetric crowns), and a transmission term for back-lit leaves.
  - Self-occlusion: AO is baked per vertex from depth inside the crown. Ambient is darkened
    by that AO.
- **Alpha mips** preserve coverage (scale each mip's alpha to keep the alpha-test
  coverage of mip 0). Otherwise distant crowns thin out.
- **Wind**: gentle sway along the trade wind. Branch and leaf weights are stored per vertex.
- **Distance**:
  - LOD0 out to about 40 m.
  - LOD1 out to about 400 m, with a dithered cross-fade between them.
  - Beyond 400 m the terrain's biome colour carries the look. If the edge shows, extend
    LOD1 for the large canopy species only, or add a cheap crossed-card LOD2 out to
    ~1.5 km, within the budget.
  - Octahedral impostors to 1.5 km are a later PR; tree-cast ray-traced shadows are §8.

## 6. Checking it

- `swift test` gets a `Flora` test: determinism (same seed → same mesh), triangle budgets
  per LOD, and community shares summing to 1.
- `floragen --out DIR`: specimen sheet per species (4 variants, a 1.7 m figure for scale)
  at 13:00, compared with reference photos.
- `IslandApp BUNDLE --snapshot DIR --size 2560x1440 --time 13 --flora on|off` over the
  standard views, plus forest close-ups, on seed 42 and real Oʻahu. `--camera X,Y,YAW,PITCH`
  (metres, degrees) adds a view. Report the GPU delta per view.

## 7. As built (first version)

What changed from the plan above while building it:

- **Assets** live in `Sources/Flora/Assets/` (SwiftPM resources must sit inside the
  target), cut by `tools/flora_textures.py` from the photos credited in `CREDITS.md`.
  Drawn procedurally: the loulu fan (colour sampled from a *P. martii* photo), the hāpuʻu
  trunk and the ʻaʻaliʻi capsules.
- **Grid**: canopy cells are 4.5 m (one candidate per cell caps a layer at 10⁴/cell²
  stems/ha, so 5 m cells could not reach wet forest's 450). Dwarf ʻōhiʻa grow on the
  2.5 m shrub grid.
- **Ground**: the scatter puts each trunk on the CDLOD level its distance falls in, on
  that level's triangles (blended into the next as it morphs), and trunks reach 0.6 m
  below their base.
- **LODs**: LOD0 to 30 m, LOD1 to 80 m (both scaled by √(height/12 m)), then an
  impostor (LOD2) to 600 m for the canopy and 120 m for shrubs, with dithered
  cross-fades. Each impostor is the variant's LOD0 drawn once at start-up from the side
  into albedo and normal/AO arrays (coverage-preserving mips), shown as a camera-facing
  quad and lit with the same tail as the meshes. Without it the trees were
  geometry-bound (2.6M triangles twice a frame, +5–11 ms); the octahedral impostors of §5
  remain the follow-up.
- **Shading pass**: depth `equal` against the prepass with `[[early_fragment_tests]]` and
  no discard. Only the fragment that wrote a pixel's depth matches it, so the prepass's
  alpha test is the only one and each tree pixel is shaded once.
- **Sun rays**: tree pixels set the lowest bit of the surface buffer's packed normal.
  `sunShadow` skips the n·l early-out for them. Those standing more than 1 m above the
  coarse shadow mesh (crowns) also skip the ground-horizon "shadowed" early-out and offset
  their rays along the sun direction; below it (trunks and low leaves on a valley floor
  the mesh bridges over) a ray would leak out through the backs of the ridges, so they
  take the ground's early-outs.
- **Budget levers** that got the worst view (Oʻahu's ridge, in wet forest) near 2 ms:
  LOD1 draws branches to order 1 only, with one large card per nine LOD0 tufts;
  hāpuʻu ribbons follow the frond's outline; hāpuʻu at 450/ha rather than 700; the shrub
  layer ends at 120 m; the prepass draws LOD0 before LOD1 before impostors.
- `floragen --sites BUNDLE` prints a close-up `--camera` in each community's purest
  gentle stand; `IslandApp --views a,b` renders only the named views.

## 8. Tree shadows (step 2b)

Trees cast ray-traced sun shadows on the ground and on each other within 200 m of the
camera. Beyond that the canopy stand-in from PR #33 still shades the ground.

- **Proxies** (`Flora.shadowProxy`, `ShadowProxy.swift`). Each variant gets a closed,
  outward-wound stand-in, fitted to its LOD0 mesh, of at most 80 triangles:
  - a 5-sided trunk prism from 0.6 m below the ground to the lowest crown lobe, following
    the trunk's lean;
  - a crown of three icosahedron lobes. The lobes come from k-means over the leaf-card
    vertices and are shrunk to 70% of each cluster's 90th-percentile spread, so the
    outer, sunlit leaves stay outside their own proxy;
  - for loulu and hāpuʻu, one flattened 10-sided bicone (a disc) instead of the lobes.

  `ShadowProxyTests` checks that every proxy is closed, consistently and outwardly wound,
  within budget, inside its mesh's bounds, and deterministic.
- **Acceleration structures.**
  - The terrain's primitive structure is unchanged: the same back-face culling,
    clockwise winding and maxHeight exit.
  - The trees get an instance structure of their own (`FloraRenderer.Shadows`), built on
    the GPU with `MTLIndirectInstanceAccelerationStructureDescriptor`. `floraShadows` fills
    the descriptors and `floraShadowCount` clamps the count the build reads.
  - That scatter shares `floraCandidate` with the drawing scatter, so each proxy stands
    exactly where its tree is drawn (wind sway aside), with no view culling.
  - It covers trees within 260 m: the 200 m range plus a 60 m margin for slanting shadows
    and the camera's travel between rebuilds. Shrubs are covered to 110 m, inside the
    120 m to which they are drawn.
  - Each tree is two instances: its trunk (2k) and its crown (2k + 1), each in a tight box.
    The parity tells the shader which one a ray hit.
  - The structure is rebuilt when the camera has moved 10 m, or when the last build stood
    on tiles still loading.
- **Rays.** `sunShadow` keeps the terrain's own rays as they were, and adds a second ray
  per pixel into the tree structure.
  - The tree ray starts from the true surface, not from the shadow-mesh lift: on a valley
    floor the coarse mesh bridges, the lifted origin would sit above or inside the trees.
  - It uses any-hit and no culling, so a leaf inside its own crown's proxy is shaded by it.
  - It records only what it met: nothing, a trunk or a crown.
  - Crowns let light through in patches: `crownTransmission` is two octaves of value noise
    about a metre across, sampled 12 m towards the sun from each pixel. Any-hit may report
    a crown's far side, so the hit point would scatter the patches pixel by pixel.
  - The ray bends by up to 0.08 rad with smooth noise over the surface, so the
    icosahedra's straight facets cast ragged edges.
  - There is no sun-disc jitter. Real canopy shadows are nearly sharp, and one jittered
    ray per pixel left speckle.
- **Passes.** Rays are traced coarse to fine:
  1. One pixel per 8×8 block traces the trees.
  2. One pixel per 4×4 block takes the coarse pixels' answer where all four show the same
     surface and agree, and traces otherwise.
  3. Pass 0 traces the terrain per 2×2 block. Its tree answer is taken the same way from
     the 4×4 pixels around it, or traced, so shadow edges are traced per 2×2 block.
  4. Pass 1 takes the mean of its 2×2 neighbours' tree answers, which softens edges over
     two pixels.

  The sun texture (rgba8: read-write textures can't be rg8) holds:
  - r: the terrain's visibility;
  - g, b: the trees' cover (the share of rays that met nothing, and that met a trunk);
  - a: the light the trees let through, which adds each pixel's own crown gaps.

  Shading multiplies r by a. Impostor pixels (bit 16 of the surface normal) skip the trees:
  their quads stand inside their own proxy.
- **Crown rays at low sun** (§7's remaining cost). In pass 1, crown pixels take their 2×2
  anchors' mean for the terrain answer too, whatever the anchors' depths.
- **Range edge.** The traced shadows fade out over 150–200 m, and the stand-in's sun
  dimming fades in over the same distance. The stand-in's sky occlusion (half the sky
  hidden under a full canopy) stays everywhere within the canopy range, since the proxies
  only block the sun. Nothing is darkened twice and nothing pops at the edge.
- **Stable while walking.** Each proxy stands on the finest ground, so it doesn't move with
  the camera between rebuilds. Rays take the nearest proxy hit: any-hit reports whichever
  proxy the traversal meets first, which a rebuild changes, so pixels flipped between a
  trunk's shadow and a crown's (up to 1.6% of the frame). Rendering a 40 m walk at 2 m
  steps with a stale structure and with a fresh one now differs in at most 0.005% of
  pixels. The edge bend shrinks with the sun's height; at low sun it was drawn out into
  snakes along the shadows.
- **Cost.** Rays run at about 200 Mrays/s against this structure on the M3. At one ray
  per 2×2 block, shadows cost 3–5 ms at 1440p; the 8×8 pass and the per-pixel crown gaps
  are what fit them in. Things that didn't help:
  - a shorter range (the cost is in near pixels);
  - capping ray length;
  - `preferFastIntersection`;
  - interpolating distant edges.

  2560×1440 GPU ms, median of 5:

  | View | Hour | Trees off | #33 trees on | Tree shadows | Δ vs off | Δ vs #33 |
  |---|---|---|---|---|---|---|
  | Oʻahu ridge | 07 / 13 / 17:30 | 8.22 / 8.37 / 8.71 | 11.96 / 9.53 / 11.24 | 13.19 / 11.30 / 12.57 | +5.0 / +2.9 / +3.9 | +1.2 / +1.8 / +1.3 |
  | Oʻahu riparian | | 7.16 / 7.32 / 7.27 | 8.73 / 8.04 / 8.63 | 9.50 / 9.82 / 9.49 | +2.3 / +2.5 / +2.2 | +0.8 / +1.8 / +0.9 |
  | Oʻahu mesic | | 7.54 / 7.96 / 7.70 | 7.95 / 8.07 / 9.36 | 9.99 / 9.53 / 10.13 | +2.5 / +1.6 / +2.4 | +2.0 / +1.5 / +0.8 |
  | Oʻahu wet forest | | 6.65 / 6.63 / 7.48 | 7.05 / 7.42 / 9.50 | 8.67 / 8.21 / 8.12 | +2.0 / +1.6 / +0.6 | +1.6 / +0.8 / −1.4 |
  | Oʻahu pali | | 11.56 / 7.27 / 7.80 | 13.30 / 7.25 / 8.21 | 13.92 / 8.22 / 9.88 | +2.4 / +1.0 / +2.1 | +0.6 / +1.0 / +1.7 |
  | Oʻahu coast | | 6.93 / 7.18 / 7.81 | 7.27 / 7.73 / 8.99 | 8.64 / 8.99 / 10.95 | +1.7 / +1.8 / +3.1 | +1.4 / +1.3 / +2.0 |
  | seed 42 pali | | 8.74 / 8.93 / 8.59 | 11.95 / 8.29 / 11.80 | 14.41 / 10.12 / 12.26 | +5.7 / +1.2 / +3.7 | +2.5 / +1.8 / +0.5 |
  | seed 42 mesic | | 7.59 / 7.60 / 7.61 | 7.77 / 7.87 / 10.02 | 9.97 / 9.69 / 10.90 | +2.4 / +2.1 / +3.3 | +2.2 / +1.8 / +0.9 |
  | seed 42 riparian | | 8.14 / 6.49 / 6.51 | 10.81 / 6.77 / 6.72 | 13.76 / 8.43 / 7.78 | +5.6 / +1.9 / +1.3 | +3.0 / +1.7 / +1.1 |

  Every view stays under 16.7 ms (worst 14.4). Against the targets of +2.0 ms at noon and
  +2.5 at low sun, measured from trees off, Oʻahu ridge, riparian and seed 42 mesic are
  over at noon, and most views are over at low sun. In several of those the trees alone
  already cost more than the target before any shadows (#33: ridge +3.7 at 07:00, seed 42
  pali +3.2).

