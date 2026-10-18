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
  (Since 2026-10-18 the whole frame's budget is **25 ms at 2560×1440, 40 fps**, with
  ~21–22 ms as the working ceiling for the worst view; the trees' share is set by §10.)
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
  cross-fades. (§10 replaced the fades and moved the distances out.) Each impostor is the variant's LOD0 drawn once at start-up from the side
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
  - Deep in shadow (every blocker tap found leaves and the penumbra is narrower than the
    search) the filter is skipped: most of a closed forest's floor.
  - Crown pixels trace the terrain's rays at one per 4×4 block and take their neighbours'
    mean, where everything else traces per 2×2 (§10).
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

  - Every view stayed under the 16.7 ms budget of the time (now 25 ms). The worst is
    seed 42's pali at 07:00 at 15.39 ms (origin/main's worst: 15.09). §10's timings are
    for the build after it.
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

## 10. LOD fades and distances

Scott, walking in the app: a screen door on the trees, and LOD switches too close and too
obvious.

- **The screen door** was the LOD cross-fade. Each plant inside a fade band (8 m wide
  at every LOD boundary and at the end of the range) was drawn twice with complementary
  screen-space dithers. With no TAA, a fixed dither reads as a stipple, on every plant in
  the band at once. With the fades switched off, the same snapshot was clean. Alpha-test
  mips and shadow filtering weren't involved.
- **Fades in time, not distance** (`floraScatter`):
  - Each plant's LOD is kept from frame to frame in a GPU table keyed by its grid cell
    (`FloraRenderer.lodStates`, 2¹⁷ open-addressed slots). Each slot holds the current
    LOD, the one it is fading from, the progress and the frame last seen.
  - A plant changes LOD only once its distance is 4% past the boundary (hysteresis), then
    dithers between the two over 0.3 s. So the dither is on the few plants crossing,
    briefly, not on a band of plants.
  - Leaving the range is a LOD too (not drawn): the scatter looks 8% past the range so
    plants can fade out.
  - Plants seen for the first time, or back in view after being culled, take their LOD
    at once, so pops happen off screen.
  - Snapshots step the fades 1/60 s per frame, as the app does at 60 fps.
- **Impostors that match the mesh**:
  - Each variant's LOD0 is baked from 8 sides (256² each, 224 layers).
  - The impostor shows the two views nearest the direction it's seen from, in the
    plant's own frame (its yaw included), blending colour, normal and coverage.
  - Before, one side view was turned to face the camera and mirrored at random, so the
    plant changed shape at the swap.
  - The normals now point towards the viewer; the old basis had them facing away.
  - Shadow-caster impostors use the view nearest the sun.
  - Start-up bakes the views and builds their mips on all cores in about a second.
- **Distances**: LOD0 to 80 m, LOD1 to 250 m (both × √(height/12 m), 0.63–1.41), then
  impostors to 600 m. Measured at 2560×1440, 07:00, worst views:

  | LOD0 / LOD1 | seed 42 ridge | seed 42 pali | Oʻahu ridge | Oʻahu mesic |
  |---|---|---|---|---|
  | 30 / 80 m | 12.1–12.5 | 15.1–15.2 | 14.0–14.1 | 12.3–12.5 |
  | 80 / 250 m | 17.8–20.8 | 17.2–17.3 | 19.4–20.7 | 14.6–14.9 |
  | 100 / 300 m | 18.6–23.8 | 18.8 | 20.3–32.0 | 16.2–16.4 |

  At 100 / 300 m the ridge views came near or past the 25 ms budget; 80 / 250 m keeps
  the worst near 21 ms. The ridge views are geometry-bound (2–2.5M tree triangles,
  drawn in both the prepass and the shading pass). No bucket came near its 4096
  instances (the largest held 1657).
- **Made room along the way** (before the budget went to 25 ms; kept because they cost
  no quality):
  - the shadow lookup's deep-shadow early out (§8);
  - crowns' terrain rays at one per 4×4 block: at low sun every crown pixel needs rays,
    and LOD0 crowns cover much of a forest view.

  Together about 0.8 ms on seed 42's pali at 07:00.

- **Cost**: 2560×1440 GPU ms, median of 5, every automatic view plus the forest cameras.
  "PR #36 before §10" is §9's build (shadow maps and ground textures).

  | View | Hour | origin/main | PR #36 before §10 | §10 | Δ |
  |---|---|---|---|---|---|
  | Oʻahu coast | 07:00 | 9.46 | 10.05 | 10.62 | +0.57 |
  | Oʻahu coast | 10:00 | 9.40 | 9.75 | 10.37 | +0.62 |
  | Oʻahu coast | 13:00 | 9.81 | 9.97 | 10.53 | +0.56 |
  | Oʻahu coast | 16:00 | 9.97 | 10.33 | 10.90 | +0.57 |
  | Oʻahu dry-forest | 07:00 | 9.71 | 9.45 | 9.97 | +0.52 |
  | Oʻahu dry-forest | 10:00 | 9.28 | 9.34 | 9.77 | +0.43 |
  | Oʻahu dry-forest | 13:00 | 9.77 | 9.31 | 9.73 | +0.42 |
  | Oʻahu dry-forest | 16:00 | 9.24 | 9.08 | 9.95 | +0.87 |
  | Oʻahu mesic | 07:00 | 10.66 | 12.01 | 14.67 | +2.66 |
  | Oʻahu mesic | 10:00 | 10.28 | 11.61 | 13.42 | +1.81 |
  | Oʻahu mesic | 13:00 | 10.23 | 11.33 | 13.38 | +2.05 |
  | Oʻahu mesic | 16:00 | 10.52 | 11.70 | 14.36 | +2.66 |
  | Oʻahu pali | 07:00 | 14.52 | 13.50 | 14.56 | +1.06 |
  | Oʻahu pali | 10:00 | 8.90 | 8.82 | 9.72 | +0.90 |
  | Oʻahu pali | 13:00 | 8.88 | 8.86 | 9.73 | +0.87 |
  | Oʻahu pali | 16:00 | 8.94 | 8.96 | 9.56 | +0.60 |
  | Oʻahu ridge | 07:00 | 13.39 | 14.54 | 19.87 | +5.33 |
  | Oʻahu ridge | 10:00 | 11.93 | 12.53 | 16.65 | +4.12 |
  | Oʻahu ridge | 13:00 | 11.80 | 12.64 | 16.14 | +3.50 |
  | Oʻahu ridge | 16:00 | 11.96 | 12.77 | 16.42 | +3.65 |
  | Oʻahu riparian | 07:00 | 9.94 | 10.67 | 12.65 | +1.98 |
  | Oʻahu riparian | 10:00 | 10.47 | 11.60 | 13.02 | +1.42 |
  | Oʻahu riparian | 13:00 | 10.39 | 11.65 | 13.04 | +1.39 |
  | Oʻahu riparian | 16:00 | 11.15 | 12.15 | 14.14 | +1.99 |
  | Oʻahu summit | 07:00 | 10.47 | 10.46 | 14.68 | +4.22 |
  | Oʻahu summit | 10:00 | 10.58 | 10.07 | 10.62 | +0.55 |
  | Oʻahu summit | 13:00 | 9.90 | 9.47 | 10.01 | +0.54 |
  | Oʻahu summit | 16:00 | 11.36 | 11.30 | 11.92 | +0.62 |
  | Oʻahu wet-forest | 07:00 | 9.30 | 10.15 | 11.47 | +1.32 |
  | Oʻahu wet-forest | 10:00 | 8.83 | 9.66 | 10.90 | +1.24 |
  | Oʻahu wet-forest | 13:00 | 8.58 | 9.66 | 11.05 | +1.39 |
  | Oʻahu wet-forest | 16:00 | 8.32 | 9.08 | 10.62 | +1.54 |
  | seed 42 coast | 07:00 | 9.35 | 9.59 | 10.49 | +0.90 |
  | seed 42 coast | 10:00 | 9.33 | 9.59 | 10.28 | +0.69 |
  | seed 42 coast | 13:00 | 9.16 | 9.62 | 10.26 | +0.64 |
  | seed 42 coast | 16:00 | 10.04 | 10.39 | 11.12 | +0.73 |
  | seed 42 dry-forest | 07:00 | 10.94 | 11.11 | 12.12 | +1.01 |
  | seed 42 dry-forest | 10:00 | 9.17 | 9.65 | 10.36 | +0.71 |
  | seed 42 dry-forest | 13:00 | 9.02 | 9.33 | 9.99 | +0.66 |
  | seed 42 dry-forest | 16:00 | 9.02 | 9.30 | 10.16 | +0.86 |
  | seed 42 mesic | 07:00 | 10.82 | 11.49 | 13.92 | +2.43 |
  | seed 42 mesic | 10:00 | 10.37 | 11.45 | 13.33 | +1.88 |
  | seed 42 mesic | 13:00 | 10.36 | 11.29 | 12.94 | +1.65 |
  | seed 42 mesic | 16:00 | 10.51 | 11.60 | 13.78 | +2.18 |
  | seed 42 pali | 07:00 | 14.86 | 15.39 | 17.88 | +2.49 |
  | seed 42 pali | 10:00 | 10.85 | 11.72 | 14.12 | +2.40 |
  | seed 42 pali | 13:00 | 10.73 | 11.49 | 13.63 | +2.14 |
  | seed 42 pali | 16:00 | 11.00 | 11.68 | 14.23 | +2.55 |
  | seed 42 ridge | 07:00 | 11.26 | 12.03 | 16.04 | +4.01 |
  | seed 42 ridge | 10:00 | 10.14 | 11.64 | 15.27 | +3.63 |
  | seed 42 ridge | 13:00 | 10.13 | 11.43 | 14.72 | +3.29 |
  | seed 42 ridge | 16:00 | 10.32 | 11.49 | 15.61 | +4.12 |
  | seed 42 riparian | 07:00 | 13.84 | 12.99 | 14.81 | +1.82 |
  | seed 42 riparian | 10:00 | 8.87 | 9.85 | 10.87 | +1.02 |
  | seed 42 riparian | 13:00 | 8.73 | 9.75 | 10.83 | +1.08 |
  | seed 42 riparian | 16:00 | 15.09 | 15.12 | 16.59 | +1.47 |
  | seed 42 summit | 07:00 | 11.54 | 11.58 | 16.35 | +4.77 |
  | seed 42 summit | 10:00 | 9.35 | 10.71 | 13.57 | +2.86 |
  | seed 42 summit | 13:00 | 9.22 | 10.47 | 13.28 | +2.81 |
  | seed 42 summit | 16:00 | 9.85 | 10.49 | 13.33 | +2.84 |
  | seed 42 wet-forest | 07:00 | 10.99 | 11.51 | 16.17 | +4.66 |
  | seed 42 wet-forest | 10:00 | 10.44 | 11.45 | 15.36 | +3.91 |
  | seed 42 wet-forest | 13:00 | 10.26 | 11.35 | 14.94 | +3.59 |
  | seed 42 wet-forest | 16:00 | 10.91 | 11.41 | 15.64 | +4.23 |

  - The worst view is Oʻahu's ridge at 07:00 at 19.87 ms, under the 21–22 ms working
    ceiling and the 25 ms budget.
  - Views looking over forest from above (ridges, pali, summits, seed 42's wet forest)
    pay 2.5–5 ms for the farther meshes. Views inside a stand or over open ground pay
    0.4–2 ms.

## 11. Far foliage

Scott wanted to see foliage on distant pali. Research notes (Crimson Desert, Nanite
Foliage in the Witcher 4 demo, Ghost of Tsushima and Yōtei, Bruneton & Neyret 2012) are in
`flora-out/far-foliage-research.md` on the Mac. The common shape is three tiers:
individual trees near, aggregated stands in the middle distance, and forest folded into the
terrain far away. Built here in that order of value.

- **Canopy shading on far terrain** (`terrainFragment`, from the ground textures' 150 m
  outwards, under the drawn trees once they end):
  - Each community's crown cover and spacing come from its stems per hectare (as
    `canopyCoverTexture` has them), varied by a clumping noise tens of metres across.
  - Crowns while they span a pixel or more: domes on a jittered grid, each lit with its own
    normal, its flanks in its neighbours' shade at low sun, the gaps dark.
  - Then clumps of crowns, four spacings across, then stands about 100 m across. Each
    tilts the normal and shades the averaged canopy, so relief carries out to where single
    crowns are long gone.
  - Past all of them the average: the crowns' mean colour, the share in sun (less at low
    sun, all of it in the hotspot looking down-sun) and mean occlusion.
- **Pali vegetation**:
  - Canopy trees are now placed on slopes to ~50° and shrubs and ferns to ~70°.
  - The cliff community has a dense layer of ʻaʻaliʻi, dwarf ʻōhiʻa and hāpuʻu.
  - Further out, steep vegetated ground carries a fern-green mat (uluhe) over most of the
    far albedo's soil streaks and half its rock, with the same crown shading. The
    Koʻolau's walls read green to the crest, with red erosion still breaking through.
- **Occlusion culling** (`floraOccluded`, `FloraRenderer.buildHzb`):
  - Last frame's depth prepass becomes a pyramid of farthest depths.
  - A tree (or patch) whose nearest point lies behind everything in its screen rectangle
    there isn't drawn.
  - Trees outside last frame's view count as visible. The pyramid isn't used after the
    camera moves 2 m or turns 3°.
  - Renders match unculled ones to ≤0.06% of pixels. A fifth to a half of the trees are
    drawn in forest views, saving 0.5–2 ms there.
- **Stand patches** (`floraPatches`, `FloraRenderer.bakePatches`), 600 m to 2.5 km:
  - At start-up each community's stand is baked as a 24 m patch, from LOD0 trees at a
    middling site, in the impostors' eight views.
  - Shrubs are included only where they are the cover seen from afar (no canopy, cloud
    shrubland, pali). Elsewhere single shrubs end at 120 m and a canopy hides them.
  - Each forested 24 m cell is one camera-facing card of its community's patch: random
    yaw, ±15% size, sunk 15% so slopes show no daylight beneath.
  - Cells are kept by the patch's crown cover.
  - Same view blend, time fades (at both ends of the range) and occlusion as the trees.
  - They give ridgelines and pali edges a broken silhouette and continue the stands
    seamlessly where the single trees end.
- **After Scott's first look** (red cliffs showing, canopy too low in contrast):
  - On pali the mat now covers nearly all soil streaks and most of the far albedo's rock.
    It stays near-continuous, and covers the ground wherever plants grow at all, since
    the canopy shading draws its own gaps. Open woodland on gentle ground still shows its
    soil, by crown cover.
  - Contrast: gaps at 35% of the canopy colour with deep occlusion, crowns rounder than
    hemispheres, and the clump and stand layers at nearly full strength.
  - Cost: seed 42's pali at 07:00 went from ~16.7 to ~19.3 ms. Single runs of the
    ridge and summit views at 07:00 ranged 14.7–23.6 ms: all within 25, but above the
    21–22 ms working ceiling at times.
- **After Scott's second look** (ridge trees too big, slivers on the walls, marbled walls):
  - Stand patches are 12 m (two to six trees) instead of 24 m, so ridgeline silhouettes are
    tree-sized. None stand on slopes over 45°, where a card stood off the wall as a sliver;
    the canopy shading covers those.
  - The marbling was the crown spacing varying smoothly over the ground (the mat's share
    follows the slope): a pattern whose scale drifts across a surface is squeezed into
    rings. Spacing now comes in half-octave steps and is the mat's or the trees', never a
    mix.
  - Single crowns are laid out in 3D (a sphere round each cuts any surface into round
    discs), so they don't stretch down steep walls. Clumps and stands stay on the map's
    plane, which is cheaper, and their normals are gentle so they read as light and shade,
    not bubbles.
  - The crowns' footprint test counts the viewing angle, so walls seen at a slant fall
    back to the averaged canopy before they alias.
  - Cost: single runs of the heaviest views at 07:00 reach ~24 ms (Oʻahu ridge, seed 42
    summit): within 25, above the 22 ms aim.
- **After Scott's third look** (scaly cells and red earth on a wall up close, a dark seam):
  - Pali walls no longer use crowns at all. The fern mat is its own soft texture
    (`canopyMat`): three octaves of 3D value noise at 1.5, 6 and 30 m shading light and
    shade, each faded out as a pixel spans it, at every distance. With no crown outlines
    there's nothing to read as scales, and no switch between regimes to leave a seam.
  - The mat is coloured fern green outright (hue drifting over tens of metres), not the
    far albedo's soil, so no red returns.
  - Crowns, clumps and stands remain for forest on gentler ground.
  - Cost: single runs of the heaviest 07:00 views are 16–24 ms.
- **Scott's photo of the windward Koʻolau** as the guide (2026-10-18):
  - No accent trees on the walls: canopy trees back to slopes ≤40°, shrubs ≤55°, stand
    patches ≤40°. Past 40° the ground is the wall's, whatever community the weights name.
  - The wall is coloured by its shape (`paliWall`), not by noise. Bright yellow-green
    uluhe and moss on the less steep facets and convex ribs; dark blue-grey-green wet
    basalt on the steepest faces and in concave gullies; grey rock on the steepest
    concave faces. Curvature is the 10 m heights' Laplacian over ±20 m, from the vertex
    shader (`VOut.ridge`).
  - Fine streaks run down the fall line: noise stretched 10× along it, at 2 m and 9 m,
    each filtered by footprint. Gullies are darker through occlusion.
  - The forest at the foot keeps its crowns, blended into the wall's texture over a
    range of slope, with no seam.
  - The wall's shading starts 30 m from the camera rather than at the ground textures'
    150 m, so close walls aren't red soil texture.
  - Lookout used for the comparison: `--camera koolau=55200,34000,-125,7 --eye 120`
    (Kāneʻohe side, ~4 km from the crest).
  - Cost: Oʻahu ridge at 07:00 17–22 ms (one 27 ms outlier in eight runs); seed 42
    summit 22.5.
- `floragen --sites` is unchanged.
- Snapshots take `--eye H` to stand the camera over the canopy for distant views.

- **Cost**: 2560×1440 GPU ms, median of 5, every automatic view plus the forest cameras,
  against §10's build (main after PR #36).

  | View | Hour | main (PR #36) | far foliage | Δ |
  |---|---|---|---|---|
  | Oʻahu coast | 07:00 | 10.62 | 11.39 | +0.77 |
  | Oʻahu coast | 10:00 | 10.37 | 11.35 | +0.98 |
  | Oʻahu coast | 13:00 | 10.53 | 11.57 | +1.04 |
  | Oʻahu coast | 16:00 | 10.90 | 11.71 | +0.81 |
  | Oʻahu dry-forest | 07:00 | 9.97 | 10.37 | +0.40 |
  | Oʻahu dry-forest | 10:00 | 9.77 | 10.53 | +0.76 |
  | Oʻahu dry-forest | 13:00 | 9.73 | 10.36 | +0.63 |
  | Oʻahu dry-forest | 16:00 | 9.95 | 10.26 | +0.31 |
  | Oʻahu mesic | 07:00 | 14.67 | 14.51 | -0.16 |
  | Oʻahu mesic | 10:00 | 13.42 | 13.09 | -0.33 |
  | Oʻahu mesic | 13:00 | 13.38 | 13.05 | -0.33 |
  | Oʻahu mesic | 16:00 | 14.36 | 13.85 | -0.51 |
  | Oʻahu pali | 07:00 | 14.56 | 15.30 | +0.74 |
  | Oʻahu pali | 10:00 | 9.72 | 10.89 | +1.17 |
  | Oʻahu pali | 13:00 | 9.73 | 10.98 | +1.25 |
  | Oʻahu pali | 16:00 | 9.56 | 10.94 | +1.38 |
  | Oʻahu ridge | 07:00 | 19.87 | 21.09 | +1.22 |
  | Oʻahu ridge | 10:00 | 16.65 | 15.18 | -1.47 |
  | Oʻahu ridge | 13:00 | 16.14 | 15.11 | -1.03 |
  | Oʻahu ridge | 16:00 | 16.42 | 15.22 | -1.20 |
  | Oʻahu riparian | 07:00 | 12.65 | 12.87 | +0.22 |
  | Oʻahu riparian | 10:00 | 13.02 | 13.05 | +0.03 |
  | Oʻahu riparian | 13:00 | 13.04 | 13.11 | +0.07 |
  | Oʻahu riparian | 16:00 | 14.14 | 14.20 | +0.06 |
  | Oʻahu summit | 07:00 | 14.68 | 14.38 | -0.30 |
  | Oʻahu summit | 10:00 | 10.62 | 11.86 | +1.24 |
  | Oʻahu summit | 13:00 | 10.01 | 11.13 | +1.12 |
  | Oʻahu summit | 16:00 | 11.92 | 12.85 | +0.93 |
  | Oʻahu wet-forest | 07:00 | 11.47 | 12.90 | +1.43 |
  | Oʻahu wet-forest | 10:00 | 10.90 | 12.59 | +1.69 |
  | Oʻahu wet-forest | 13:00 | 11.05 | 12.70 | +1.65 |
  | Oʻahu wet-forest | 16:00 | 10.62 | 12.07 | +1.45 |
  | seed 42 coast | 07:00 | 10.49 | 10.79 | +0.30 |
  | seed 42 coast | 10:00 | 10.28 | 10.79 | +0.51 |
  | seed 42 coast | 13:00 | 10.26 | 10.64 | +0.38 |
  | seed 42 coast | 16:00 | 11.12 | 11.84 | +0.72 |
  | seed 42 dry-forest | 07:00 | 12.12 | 12.55 | +0.43 |
  | seed 42 dry-forest | 10:00 | 10.36 | 10.92 | +0.56 |
  | seed 42 dry-forest | 13:00 | 9.99 | 11.01 | +1.02 |
  | seed 42 dry-forest | 16:00 | 10.16 | 11.28 | +1.12 |
  | seed 42 mesic | 07:00 | 13.92 | 13.86 | -0.06 |
  | seed 42 mesic | 10:00 | 13.33 | 12.79 | -0.54 |
  | seed 42 mesic | 13:00 | 12.94 | 12.75 | -0.19 |
  | seed 42 mesic | 16:00 | 13.78 | 13.59 | -0.19 |
  | seed 42 pali | 07:00 | 17.88 | 16.66 | -1.22 |
  | seed 42 pali | 10:00 | 14.12 | 13.97 | -0.15 |
  | seed 42 pali | 13:00 | 13.63 | 13.23 | -0.40 |
  | seed 42 pali | 16:00 | 14.23 | 13.74 | -0.49 |
  | seed 42 ridge | 07:00 | 16.04 | 14.58 | -1.46 |
  | seed 42 ridge | 10:00 | 15.27 | 13.90 | -1.37 |
  | seed 42 ridge | 13:00 | 14.72 | 13.20 | -1.52 |
  | seed 42 ridge | 16:00 | 15.61 | 13.92 | -1.69 |
  | seed 42 riparian | 07:00 | 14.81 | 14.17 | -0.64 |
  | seed 42 riparian | 10:00 | 10.87 | 10.79 | -0.08 |
  | seed 42 riparian | 13:00 | 10.83 | 10.85 | +0.02 |
  | seed 42 riparian | 16:00 | 16.59 | 16.23 | -0.36 |
  | seed 42 summit | 07:00 | 16.35 | 21.08 | +4.73 |
  | seed 42 summit | 10:00 | 13.57 | 13.11 | -0.46 |
  | seed 42 summit | 13:00 | 13.28 | 12.73 | -0.55 |
  | seed 42 summit | 16:00 | 13.33 | 12.81 | -0.52 |
  | seed 42 wet-forest | 07:00 | 16.17 | 14.47 | -1.70 |
  | seed 42 wet-forest | 10:00 | 15.36 | 13.55 | -1.81 |
  | seed 42 wet-forest | 13:00 | 14.94 | 13.45 | -1.49 |
  | seed 42 wet-forest | 16:00 | 15.64 | 14.38 | -1.26 |

  - The worst view is Oʻahu's ridge at 07:00 at 21.09 ms, within the 21–22 ms working
    ceiling of the 25 ms budget.
  - Forest views come out cheaper than before (−0.2 to −1.8 ms): occlusion culling saves
    more than the far shading and the patches cost.
  - Open views over far terrain pay +0.3 to +1.7 ms for the canopy shading and patches.
  - Seed 42's summit at 07:00 (+4.7) is an outlier of that low-sun view, as in §8's
    summit figures; its other hours are cheaper than before.

- **Ring without trees when flying** (2026-10-18): a high camera saw a band a few hundred
  metres out with neither single trees nor stand patches. The trees end by 3D distance;
  the patches cut their inner edge by horizontal distance, so cells horizontally inside
  the trees' range but 3D beyond it got nothing. Patches now cut only their outer edge
  horizontally and take their inner edge from the same 3D distance (with the time fades).
  Also fixed: `floraCandidate` chose the trunk's CDLOD ground ring from the camera's
  height above the sea, not above the ground, which put trees on far rings' surfaces
  under a high camera.
