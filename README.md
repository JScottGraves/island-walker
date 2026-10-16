# Island Walker

A native macOS walking simulator for procedurally generated volcanic islands about the
size of Oahu (~1,500 km²). Swift + Metal, custom engine, terrain realism first.

The architecture and Milestone 1 plan live in the project's plan document
(`plan/terrain-architecture-and-milestone-1.md` in the project files).

## Layout

| Module | What it does |
|---|---|
| `IslandCore` | Grids, deterministic RNG and noise shared by everything |
| `IslandGen` | Generation: shield volcanoes, rainfall, priority-flood, stream power erosion, previews |
| `islandgen` | Command-line bake tool |
| `Engine` | Metal renderer (currently one whole-island mesh; streaming LOD comes next) |
| `Walker` | First-person controller on the heightfield |
| `IslandApp` | macOS app window |

## Try it

Requires an Apple Silicon Mac with macOS 14+ and Xcode 16.

```sh
swift build -c release
# Bake an island and write hillshade.png, flow.png, rain.png, hardness.png, height.r32
.build/release/islandgen --seed 42 --size 1024 --steps 400 --out out
# Walk it (WASD, Shift to run, drag to look, F to fly, Space/C up/down while flying)
swift run -c release IslandApp out/height.r32
```

`--size 512` is quick for iterating on look; `--size 2048` is the review resolution.
