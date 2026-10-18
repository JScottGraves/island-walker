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
  - Octahedral impostors to 1.5 km are a later PR; tree shadows are §8.

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

## 8. Tree shadows: cascaded shadow maps

Trees cast sun shadows on the ground and on each other within 220 m of the camera,
through three shadow maps drawn from the sun each frame. Beyond that the canopy stand-in
from PR #33 shades the ground, fading in over the last 40 m as the maps fade out. This
replaces PR #34's ray-traced proxies, whose icosahedra and prisms (traced per 8×8 to
2×2 block) cast octagons and straight edges.

- **Cascades** (`ShadowCascades`, IslandCore; tested in `ShadowCascadeTests`): 0–25,
  25–80 and 80–220 m from the camera, orthographic from the sun, each 1024² in one
  Depth32Float array.
  - Each cascade holds the part of the view frustum within its split *by distance*, the
    same measure the shaders pick cascades by. That region is a cone out to the frustum's
    corners, 50° off axis for a 16:9, 60° lens, so it fits a sphere of radius 0.77 × the
    split centred on the view axis. Fitting by view depth instead needs 1.18 × the split
    (the slice's corners lie 1.55 × further out), with 2.3× the texels.
  - The map's size depends only on the split and the lens, and its centre is snapped to
    whole texels in world space, so edges don't crawl while walking or turning.
  - Texels are 3.8, 12 and 33 cm. 2048² maps cost 1.5–3 ms more and looked no
    different in the close-ups: the cost is the alpha-tested leaf cards' fill, which grows
    with texels, not with the area a map covers.
- **Casters** (`floraCasters`, `FloraRenderer.renderShadows`):
  - Every tree within 300 m (shrubs 120 m) whose bounding sphere overlaps a cascade goes
    into it, with no view culling, since trees behind and beside the camera shade what it
    sees. Same `floraCandidate` and ground as the drawn trees, so trunks meet their shadows.
  - Casters are kept at most 45 / sin(elevation) m (≤ 300 m) beyond a cascade towards
    the sun: no tree top is higher above what it shades.
  - Drawn with `floraVertex` (wind and all) and the prepass's alpha test.
  - LOD by the tree's distance, never finer than its cascade's: LOD0 within 20 m in the
    near cascade, LOD1 to 60 m, beyond that and in the far cascade the impostor. A far
    tree casting into a near cascade (at low sun, shadows reach the camera from 100 m away)
    stands high above what it shades, so its penumbra is wide and LOD0 detail would be lost.
    The casters' cost is in their vertices, not their fill (timed in a command buffer of
    their own): on Oʻahu's ridge at 07:00, LOD0 to 35 m and LOD1 to 150 m cost 2.4 ms and
    1.6 ms for the near two maps, 512² maps saved little, and 20 / 60 m halve it.
  - Impostor casters face the sun and pivot about the middle of the crown, so a crown's
    shadow lands where the mesh's would and is drawn out by 1 / sin(elevation) along the
    sun, as a round crown's is.
  - A slope-scaled raster bias (≤ 0.3 m) for cards seen edge-on from the sun.
  - The near map is drawn every frame, the other two on alternate frames. A map not
    redrawn is looked up through its cascade as drawn, re-based to the current camera
    (`Cascade.rebased`, tested), so its shadows stay put in the world; the trees' sway in it
    is a frame old. All three are redrawn when the camera has moved 2 m or turned 3° since,
    or the sun has moved.
- **Lookup** (`treeSunlight`, in `sunShadow`): each pixel of the prepass surface, after
  the terrain's rays, writes the sun the trees let through next to the terrain's
  visibility (r, g of the sun texture; shading multiplies them). Terrain and tree pixels
  alike, so crowns shade themselves and each other.
  - Normal offset of 1.5 texels (along the normal for ground, towards the sun for leaves,
    which are two-sided) and a one-texel depth bias: no acne, no peter-panning.
  - PCSS: 8 blocker taps over the area the sun's disc sees through occluders up to 30 m
    above, then 16 rotated Poisson taps of hardware 2×2 comparisons over the penumbra the
    disc (0.53°) gives at that height. Trunk bases stay sharp; leaf shadows from a crown
    15 m up soften over ~15 cm, and long low-sun shadows blur as real ones do.
  - Cascades blend over the last fifth of each split.
  - Shadowed light floors at 15% (`kCrownLeak`): leaves pass some, and the LOD meshes'
    cluster cards close gaps a real crown has. Without it the floor of a closed stand was
    black.
  - Impostor pixels (80–600 m) take no tree shadows: their quads stand inside their own
    crown. They keep the baked AO.
- **Kept from PR #33/#34**: the terrain's two sun-ray passes, the ray-leak fixes for
  valley floors (ground rays start no lower than the shadow mesh; crowns start at the
  leaf and among tree pixels take their neighbours' mean), and the canopy stand-in.
- **Removed**: the proxies (`ShadowProxy.swift` and its tests), the instance acceleration
  structure and its rebuilds, `proxyHit`, `treeSun`, `crownTransmission`, the coarse and
  fine tree passes and the cover channels.
- **Looks**: under a closed canopy the floor is now mostly shade with leaf-shaped
  sunflecks, 20–30% darker on average than PR #34, whose shrunken proxies let more sun
  through. Open dry forest shows each tree's shadow with its branches and gaps.

- **Checked** against origin/main at 07:00, 10:00 and 16:00 in the forest close-ups
  (`floragen --sites` cameras, pitched −25°) and mid-range views, on Oʻahu:
  - leaf and branch silhouettes with sunflecks, no octagons or straight facet edges;
  - no acne, trunks meeting their shadows;
  - no cascade seams.

  A 40 m walk at 0.5 m steps shows no crawl. Snapping makes that hold by construction
  (tested), but snapshots let the maps settle at every step, so the alternate-frame
  redraw is only seen in the app. Screen-space contact shadows weren't needed: close-ups
  don't read flat.
- **Cost.** 2560×1440 GPU ms, median of 5, every automatic view plus the forest cameras,
  at 07:00, 10:00, 13:00 and 16:00. Builds compared: origin/main (eb0098a, PR #34's proxy
  rays), these shadow maps, and the shadow maps plus §9's ground textures. "Trees off" is
  the last build with `--flora off`, so it includes the ground textures.

| View | Hour | Trees off | origin/main | Shadow maps | Δ vs main | + ground textures | Δ textures |
|---|---|---|---|---|---|---|---|
| Oʻahu coast | 07:00 | 8.10 | 9.46 | 8.96 | -0.50 | 10.05 | +1.09 |
| Oʻahu coast | 10:00 | 8.19 | 9.40 | 8.75 | -0.65 | 9.75 | +1.00 |
| Oʻahu coast | 13:00 | 8.66 | 9.81 | 9.05 | -0.76 | 9.97 | +0.92 |
| Oʻahu coast | 16:00 | 8.84 | 9.97 | 9.31 | -0.66 | 10.33 | +1.02 |
| Oʻahu dry-forest | 07:00 | 8.36 | 9.71 | 9.51 | -0.20 | 9.45 | -0.06 |
| Oʻahu dry-forest | 10:00 | 8.01 | 9.28 | 9.24 | -0.04 | 9.34 | +0.10 |
| Oʻahu dry-forest | 13:00 | 8.13 | 9.77 | 9.30 | -0.47 | 9.31 | +0.01 |
| Oʻahu dry-forest | 16:00 | 8.13 | 9.24 | 9.18 | -0.06 | 9.08 | -0.10 |
| Oʻahu mesic | 07:00 | 8.68 | 10.66 | 11.23 | +0.57 | 12.01 | +0.78 |
| Oʻahu mesic | 10:00 | 8.90 | 10.28 | 10.83 | +0.55 | 11.61 | +0.78 |
| Oʻahu mesic | 13:00 | 9.08 | 10.23 | 10.66 | +0.43 | 11.33 | +0.67 |
| Oʻahu mesic | 16:00 | 9.25 | 10.52 | 11.01 | +0.49 | 11.70 | +0.69 |
| Oʻahu pali | 07:00 | 12.25 | 14.52 | 13.26 | -1.26 | 13.50 | +0.24 |
| Oʻahu pali | 10:00 | 8.14 | 8.90 | 8.41 | -0.49 | 8.82 | +0.41 |
| Oʻahu pali | 13:00 | 8.11 | 8.88 | 8.31 | -0.57 | 8.86 | +0.55 |
| Oʻahu pali | 16:00 | 8.21 | 8.94 | 8.32 | -0.62 | 8.96 | +0.64 |
| Oʻahu ridge | 07:00 | 8.79 | 13.39 | 14.72 | +1.33 | 14.54 | -0.18 |
| Oʻahu ridge | 10:00 | 8.80 | 11.93 | 12.72 | +0.79 | 12.53 | -0.19 |
| Oʻahu ridge | 13:00 | 8.87 | 11.80 | 12.65 | +0.85 | 12.64 | -0.01 |
| Oʻahu ridge | 16:00 | 9.01 | 11.96 | 12.88 | +0.92 | 12.77 | -0.11 |
| Oʻahu riparian | 07:00 | 9.33 | 9.94 | 9.94 | +0.00 | 10.67 | +0.73 |
| Oʻahu riparian | 10:00 | 9.48 | 10.47 | 10.84 | +0.37 | 11.60 | +0.76 |
| Oʻahu riparian | 13:00 | 9.46 | 10.39 | 10.73 | +0.34 | 11.65 | +0.92 |
| Oʻahu riparian | 16:00 | 9.72 | 11.15 | 11.46 | +0.31 | 12.15 | +0.69 |
| Oʻahu summit | 07:00 | 10.37 | 10.47 | 12.58 | +2.11 | 10.46 | -2.12 |
| Oʻahu summit | 10:00 | 9.83 | 10.58 | 9.83 | -0.75 | 10.07 | +0.24 |
| Oʻahu summit | 13:00 | 9.32 | 9.90 | 9.34 | -0.56 | 9.47 | +0.13 |
| Oʻahu summit | 16:00 | 11.68 | 11.36 | 11.01 | -0.35 | 11.30 | +0.29 |
| Oʻahu wet-forest | 07:00 | 7.75 | 9.30 | 9.31 | +0.01 | 10.15 | +0.84 |
| Oʻahu wet-forest | 10:00 | 7.79 | 8.83 | 8.95 | +0.12 | 9.66 | +0.71 |
| Oʻahu wet-forest | 13:00 | 7.61 | 8.58 | 8.91 | +0.33 | 9.66 | +0.75 |
| Oʻahu wet-forest | 16:00 | 7.58 | 8.32 | 8.36 | +0.04 | 9.08 | +0.72 |
| seed 42 coast | 07:00 | 8.18 | 9.35 | 8.87 | -0.48 | 9.59 | +0.72 |
| seed 42 coast | 10:00 | 8.23 | 9.33 | 8.71 | -0.62 | 9.59 | +0.88 |
| seed 42 coast | 13:00 | 8.38 | 9.16 | 8.75 | -0.41 | 9.62 | +0.87 |
| seed 42 coast | 16:00 | 9.12 | 10.04 | 9.60 | -0.44 | 10.39 | +0.79 |
| seed 42 dry-forest | 07:00 | 9.10 | 10.94 | 10.38 | -0.56 | 11.11 | +0.73 |
| seed 42 dry-forest | 10:00 | 8.55 | 9.17 | 8.64 | -0.53 | 9.65 | +1.01 |
| seed 42 dry-forest | 13:00 | 8.62 | 9.02 | 8.43 | -0.59 | 9.33 | +0.90 |
| seed 42 dry-forest | 16:00 | 8.45 | 9.02 | 8.52 | -0.50 | 9.30 | +0.78 |
| seed 42 mesic | 07:00 | 8.31 | 10.82 | 10.99 | +0.17 | 11.49 | +0.50 |
| seed 42 mesic | 10:00 | 8.13 | 10.37 | 10.91 | +0.54 | 11.45 | +0.54 |
| seed 42 mesic | 13:00 | 8.42 | 10.36 | 10.71 | +0.35 | 11.29 | +0.58 |
| seed 42 mesic | 16:00 | 8.13 | 10.51 | 11.09 | +0.58 | 11.60 | +0.51 |
| seed 42 pali | 07:00 | 9.48 | 14.86 | 15.33 | +0.47 | 15.39 | +0.06 |
| seed 42 pali | 10:00 | 9.42 | 10.85 | 11.50 | +0.65 | 11.72 | +0.22 |
| seed 42 pali | 13:00 | 9.73 | 10.73 | 11.33 | +0.60 | 11.49 | +0.16 |
| seed 42 pali | 16:00 | 9.66 | 11.00 | 11.56 | +0.56 | 11.68 | +0.12 |
| seed 42 ridge | 07:00 | 11.03 | 11.26 | 11.53 | +0.27 | 12.03 | +0.50 |
| seed 42 ridge | 10:00 | 9.83 | 10.14 | 11.13 | +0.99 | 11.64 | +0.51 |
| seed 42 ridge | 13:00 | 10.36 | 10.13 | 11.08 | +0.95 | 11.43 | +0.35 |
| seed 42 ridge | 16:00 | 10.03 | 10.32 | 10.94 | +0.62 | 11.49 | +0.55 |
| seed 42 riparian | 07:00 | 10.83 | 13.84 | 12.77 | -1.07 | 12.99 | +0.22 |
| seed 42 riparian | 10:00 | 9.29 | 8.87 | 9.57 | +0.70 | 9.85 | +0.28 |
| seed 42 riparian | 13:00 | 9.32 | 8.73 | 9.44 | +0.71 | 9.75 | +0.31 |
| seed 42 riparian | 16:00 | 11.64 | 15.09 | 15.06 | -0.03 | 15.12 | +0.06 |
| seed 42 summit | 07:00 | 10.06 | 11.54 | 14.14 | +2.60 | 11.58 | -2.56 |
| seed 42 summit | 10:00 | 8.04 | 9.35 | 10.33 | +0.98 | 10.71 | +0.38 |
| seed 42 summit | 13:00 | 7.94 | 9.22 | 10.06 | +0.84 | 10.47 | +0.41 |
| seed 42 summit | 16:00 | 7.96 | 9.85 | 10.14 | +0.29 | 10.49 | +0.35 |
| seed 42 wet-forest | 07:00 | 9.43 | 10.99 | 11.31 | +0.32 | 11.51 | +0.20 |
| seed 42 wet-forest | 10:00 | 9.74 | 10.44 | 11.28 | +0.84 | 11.45 | +0.17 |
| seed 42 wet-forest | 13:00 | 10.54 | 10.26 | 10.98 | +0.72 | 11.35 | +0.37 |
| seed 42 wet-forest | 16:00 | 10.96 | 10.91 | 11.24 | +0.33 | 11.41 | +0.17 |

  - Every view stays under 16.7 ms. The worst is seed 42's pali at 07:00 at 15.39 ms
    (origin/main's worst: 15.09).
  - Against origin/main the shadow maps cost −1.3 to +1.3 ms, mostly +0.3–0.9. The
    shading pass is 0.5–1.3 ms cheaper without the tree rays; the casters cost 0.5–2 ms.
  - The 07:00 summit views' +2.1 / +2.6 for the shadow maps come from a few slow rounds
    of that build: the textured build, which draws the same shadows, is back at
    origin/main's figure.

## 9. Ground textures

No ground cover or rocks are drawn as geometry. Instead the ground near the camera shows a
tileable texture for its plant community, where `terrainAlbedo` had given it the
community's canopy colour seen from above. That colour was right before the trees were
drawn; the ground under and between them now looks like ground.

- **Layers** (`GroundTextures`, `Assets/ground`, cut by `tools/ground_textures.py` from
  CC0 ambientCG materials; `Assets/ground/CREDITS.md` gives each source). One per
  community, chosen for pre-contact Oʻahu, no lawn grass:

  | Community | Material | Ground |
  |---|---|---|
  | strand | Ground033 | pale calcareous sand |
  | rocky shore | Gravel024 | black basalt clinker (ʻaʻā) |
  | marsh | Ground051 | wet mud |
  | dry forest | Ground067 | dry red-brown soil, litter, stones |
  | mesic forest | Ground023 | brown leaf litter over soil |
  | wet forest | Ground024 | dark wet litter with moss |
  | riparian | Ground038 | dark wet litter, twigs, fern debris |
  | cloud shrubland | Ground037 | mossy ground |
  | bog | Moss003 | wet moss over peat |
  | cliff | Rock029 | weathered red basalt, lateritic soil |
  | montane | Ground068 | moss over red soil and rock |
  | (steep slopes) | Rock031 | bare dark basalt |

  - Each is 512² (2 m across, so ~4 mm texels).
  - Its mean is set in linear light to a ground albedo near the luminance of its
    community's canopy colour, so the fade doesn't jump in brightness.
  - Albedo is floored at 0.8%: crevices photographed black.
  - Displacement goes in the normal map's blue channel, z rebuilt from x and y. Not alpha:
    the app decodes PNGs through premultiplied CoreGraphics, which blacked out every texel
    whose alpha (height) was 0.
- **Blending** (`groundTexture` in `terrainFragment`):
  - The bundle's 11 community weights (the same 1024² maps the trees are scattered from,
    now also in `DetailMaps.communities`, so they work with `--flora off`).
  - The two heaviest communities per pixel, the weights jittered by a 9 m noise so the
    bilinear ramps between 80 m texels break into patches.
  - Height-based blending: a layer's stones and litter poke through the other's. It needs
    about a quarter share before they show.
  - Bare rock by slope, as `terrainAlbedo` puts it.
  - The waterline's sand band only where strand or rocky shore weigh: dry forest on the
    low ʻEwa plain stood on beach sand before. The far albedo gets the same gate, so near
    and far agree.
- **Tiling**: each layer is laid at two scales (2 m, and 5.3 m turned 37°), switched
  between by a noise field ~23 m across, with a ±12% brightness drift over a few metres.
  Repeats don't line up into a grid.
- **Shading normal**: the maps' normals perturb it at 60%.
- **Distance**: full texture to 150 m, fading to the canopy colour by 300 m, where the
  forest canopy is what's seen from above.
- The detail clipmap's levels share `terrainFragment`, so the close-up ground shows the
  textures too. floragen's specimen sheets have no community maps and keep their plain
  ground.

- **Cost**: the last column of §8's table. 0.1–0.9 ms on most views, at most 1.1 ms
  (Oʻahu's coast, wide open ground near the camera), about nothing where trees cover
  the near ground (Oʻahu's ridge and dry forest). Two things got it there from 1.1–1.9 ms:
  - one noise value spread over the communities by golden-ratio steps for the weight
    jitter, instead of 11 noise lookups;
  - sampling the second tiling scale only where the switch between them is under way.
