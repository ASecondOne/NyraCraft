# NyraCraft

NyraCraft is an experimental voxel engine/game prototype written in Rust on top of `wgpu` + `winit`, with custom chunk streaming, LOD meshing, inventory/crafting, and data-driven block/item content.

## What Is In This Project

- Procedural world generation (normal, flat, grid modes)
- Chunked voxel meshing and GPU rendering
- Runtime world editing (break/place blocks)
- Inventory + hotbar + crafting (2x2 player, 3x3 crafting table)
- Dropped items with physics, pickup, stacking, despawn
- Command/chat console (`/help`, `/items`, `/give`)
- Hot reload of textures/content (`F1 + R`)
- Debug overlays and runtime performance stats

## Tech Stack

- Rust 2024 edition
- `wgpu` 0.19
- `winit` 0.29
- `glam` for math
- `noise` for terrain/caves
- `serde` + `serde_json` for content and recipes
- `image` for atlas/texture loading
- Python 3 + Pillow (`PIL`) for texture atlas generation script

## Quick Start

### Prerequisites

1. Rust toolchain + Cargo
2. Python 3 available as `python3` (or `python`)
3. Pillow for atlas generation:

```bash
pip install pillow
```

### Run

```bash
cargo run --release -- 12345
```

Seed argument behavior:
- Number/text: deterministic normal world seed
- `FLAT`: flat world mode
- `GRID`: flat world + debug marker columns
- No argument: prompts in terminal for seed/mode (blank = random)

### Test

```bash
cargo test
```

Current test set: 5 tests (worldgen + block/tool logic).

## Controls

- `W/A/S/D`: move
- `Space`: jump (or ascend in fly mode)
- `Shift`: sprint
- `C`: sneak (and descend in fly mode)
- `Mouse`: look
- `Left Mouse`: mine (hold for hardness-based breaking)
- `Shift + Left Mouse`: multi-break up to 12 matching nearby blocks (3x3x3 neighborhood)
- `Right Mouse`: place block / use crafting table
- `E`: inventory
- `1..9` / mouse wheel: hotbar select
- `Q`: drop one item from selected hotbar slot
- `Escape`: pause menu
- `/` or `T`: open chat/command console
- `F1`: keybind overlay
- `F3`: detailed debug stats
- `F1 + F`: face debug colors
- `F1 + W`: chunk wireframe
- `F1 + P`: pause/resume streaming
- `F1 + V`: pause/resume rendering
- `F1 + R`: hot reload atlases/content
- `F1 + M`: toggle fly mode
- `F1 + X`: toggle fullscreen
- `F1 + D`: stats overlay toggle

## Console Commands

- `/help`
- `/items`
- `/give <item> [count]`

`<item>` accepts names/aliases and namespaced ids:
- blocks: `1:<block_id>`
- items: `2:<item_id>`

## Project Layout

```text
src/
  main.rs                    # app loop, input, streaming tick, worker setup
  app/
    bootstrap.rs             # seed/mode parsing, atlas generation, CPU label
    controls.rs              # input helpers and keybind text
    console.rs               # chat/command parsing
    dropped_items.rs         # drop physics, merge, pickup, render data
    menu.rs                  # pause menu layout/hit testing
    streaming.rs             # request queue, LOD policy, streaming/apply/caching
    logger.rs                # logs/latest.log + panic hook
  world/
    worldgen.rs              # terrain, caves, biomes, trees, modes
    mesher.rs                # full/LOD meshing + greedy meshing + AO/light
    blocks.rs                # block/item registry, tool rules, drop rules
    lightengine/             # sky visibility + face light
  render/
    gpu.rs                   # pipelines, chunk buffers, culling, UI rendering
    mesh.rs                  # raw/packed vertex formats
    texture.rs               # atlas/colormap/texture upload
  player/
    movement.rs              # movement, collision response, sneak edge-guard
    block_edit.rs            # break/place edits, dirty chunks, leaf decay
    inventory.rs             # inventory/hotbar/craft interaction and layout
    crafting.rs              # recipe loading + shaped/shapeless matcher
  content/
    blocks/*.json            # block definitions
    items/*.json             # item definitions
    recipes/*.json           # crafting recipes
  texturing/
    atlas_gen.py             # builds atlas_output/atls_blocks.png + atlas_items.png
```

## How Runtime Works

### Startup

1. Initializes logger + panic hook (`logs/latest.log`)
2. Resolves seed/mode and creates `WorldGen`
3. Regenerates texture atlases only if sources changed
4. Loads block/item/recipe registries
5. Creates GPU device/pipelines
6. Spawns:
   - mesh worker threads (`available_parallelism/2`, clamped `1..=6`)
   - cache writer thread for far-LOD mesh cache

### Main Loop

- Fixed simulation tick every `50 ms` (~20 TPS)
- Frame pacing at ~`16 ms` target
- Updates:
  - player movement/physics
  - dropped item simulation/pickup
  - leaf decay and block edit remesh invalidation
  - chunk stream requests + apply worker results
  - adaptive budget tuning from live FPS/TPS

### Render

- Two chunk pipelines:
  - raw chunk vertex path (near/high detail)
  - packed far-vertex path (far LOD)
- Separate dynamic meshes for:
  - dropped items
  - block break overlay stages
  - sun billboard
- UI pass draws hotbar/inventory/chat/keybind/stats/pause menu

## Content System (Data Driven)

### Block and Item IDs

- Namespace `1:n` => block ids
- Namespace `2:n` => item ids
- Item ids are internally `i8`; namespaced `2:n` become negative ids to avoid collisions with block-backed items.

### Blocks (`src/content/blocks/*.json`)

Defines:
- `id`, `register_name`, `aliases`
- hardness + required tool
- texture slots per face (`texture_slot_1based` or `textures_slot_1based`)
- UV rotation per face
- transparency mode per face
- optional drop rules (`drop_item` / `drop_items`)

### Items (`src/content/items/*.json`)

Defines:
- `id`, `register_name`, `aliases`
- optional placeable block link
- atlas icon slot
- tool type, breaktime, durability, stack size
- optional edible metadata

### Recipes (`src/content/recipes/*.json`)

Supports:
- `shaped`
- `shapeless`
- per-file recipe arrays
- recipe aliases/variants
- craft output preview + atomic consume on craft

## Optimization and Performance Features

This project already includes substantial optimization work, including LOD.

### 1) LOD Meshing

- Mesh modes in `world::mesher`:
  - `Full`
  - `SurfaceSides`
  - `SurfaceOnly`
- Distance-based selection in `stream_tick`
- Step size scaling (`1/4/8/16/32/64`) for farther chunks
- LOD coverage logic avoids requesting lower quality if higher quality is already loaded/requested

### 2) Greedy Meshing

- Full-detail chunks (`Full`, `step=1`) use greedy face merging
- Merges across planes with light bin constraints to reduce triangles/draw cost

### 3) Packed Far Geometry

- Far LOD can be packed to `PackedFarVertex` (`i16` positions/uv, compact flags, `u8` color)
- Greatly reduces far mesh memory and upload size

### 4) Mesh Cache (Disk)

- Far packed chunks cached to disk:
  - `target/mesh_cache/world_<world_id>/...`
- Cache key includes coord + mode + step
- Binary format with magic header (`MSH3`)
- Only used for unedited chunks (dirty/override chunks bypass cache)
- Hot reload clears cache directory

### 5) Request Queue Prioritization

- Three classes:
  - `Edit` (highest)
  - `Near`
  - `Far`
- Priority combines distance, view facing, and LOD quality
- Queue replacement logic keeps better/newer work and prevents stale tasks

### 6) Time-Budgeted Streaming/Application

- Per-tick request budget (`STREAM_REQUEST_TIME_BUDGET = 3 ms`)
- Per-tick apply budget (`APPLY_RESULTS_TIME_BUDGET = 4 ms`)
- Dirty fast lane when edited chunks are pending

### 7) Adaptive Runtime Budgets

- If FPS/TPS drop:
  - lower request/apply/rebuild/pregen budgets
  - reduce draw radius cap
- If performance recovers:
  - budgets climb back up toward base values

### 8) Super-Chunk GPU Batching

- Chunks grouped into super-chunks of size `4x4x4` chunks
- One raw and one packed mesh buffer per super-chunk
- Slot-based in-place updates when capacity allows
- Rebuild only dirty super-chunks (distance-prioritized)

### 9) Visibility Culling

- Radius cull using draw distance
- Directional cull against camera forward + chunk bounding-sphere margin
- Extra horizon/underground cull for far terrain
- Visible set refreshed only when camera changes or periodic refresh needed

### 10) Memory Budget Enforcement

- Loaded chunk cap (`loaded_chunk_cap`)
- Mesh memory cap in MB (`mesh_memory_cap_mb`)
- Evicts farthest chunks when caps are exceeded

### 11) CPU Caches

- Column max-height cache for stream planning
- Height/tree caches for block solidity queries during meshing
- CPU-side cached item atlas sampling for dropped item side tinting

### 12) Other Runtime Guards

- Dropped item caps + merge in-cell stacks
- Leaf decay BFS cap (`LEAF_DECAY_SEARCH_CAP`) to avoid expensive canopy scans
- Sneak edge-guard prevents accidental ledge stepping

## Important Tunables

Most tuning constants live in:
- `src/main.rs` (radii, budgets, worker count, caps)
- `src/app/streaming.rs` (LOD policy, queue/apply budgets, request logic)
- `src/world/worldgen.rs` (terrain/cave/climate/tree parameters)
- `src/world/mesher.rs` (mesh mode behavior and AO/light work)

Notable current defaults:
- `CHUNK_SIZE = 18`
- world width in normal mode: `1000 x 1000` chunks
- base render radius: `512` (dynamic with altitude)
- base draw radius: `192`
- LOD near/mid radii: `16 / 32` chunks
- mesh memory cap: `4096 MB`

## Texture Atlas Workflow

`src/texturing/atlas_gen.py`:
- scans `textures_blocks/` and `textures_items/`
- file names must start with numeric prefix (`<index>_name.png` or `.tga`)
- packs 16x16 tiles into generated atlases:
  - `src/texturing/atlas_output/atls_blocks.png`
  - `src/texturing/atlas_output/atlas_items.png`

Atlases are regenerated automatically on startup only if source files are newer.

## Current Limitations

- No persistent world save/load for edited blocks yet (edits are runtime/session state)
- Single-player only
- No networking, entities/AI, or chunk disk persistence beyond far mesh cache

## License

MIT (`LICENSE`)
