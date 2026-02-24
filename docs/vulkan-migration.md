# Vulkan Migration Plan

This project currently renders through `wgpu` with a Vulkan backend lock.

Goal: move from `wgpu` abstractions to a native Vulkan backend incrementally, without halting gameplay development.

## Principles
- Keep the game loop and streaming pipeline backend-agnostic.
- Port one subsystem at a time behind stable interfaces.
- Require parity checks (visual + perf + memory) before deleting old paths.

## Phase 0: Stabilize Abstractions (in progress)
- [x] Lock `wgpu` to Vulkan backend at runtime.
- [x] Introduce `MeshUploadBackend` trait for streaming-facing renderer operations.
- [ ] Add a backend selector (`wgpu-vulkan` / `vulkan-native`) at startup.

## Phase 1: Native Vulkan Bootstrap
- [ ] Create `render/vk/` module with `ash` instance/device/swapchain setup.
- [ ] Implement window surface creation via `ash-window` + `winit`.
- [ ] Add robust device/queue selection and required extensions/layers checks.
- [ ] Build a minimal frame loop that clears + presents.

Exit criteria:
- `vulkan-native` launches and presents frames on the same platforms as current branch.

## Phase 2: Buffer + Upload Path
- [ ] Port chunk vertex/index buffer allocation and staging uploads.
- [ ] Port super-chunk slot management and dirty rebuild uploads.
- [ ] Keep current meshing output format unchanged.

Exit criteria:
- World geometry renders with equivalent chunk visibility and memory behavior.

## Phase 3: Pipeline Port
- [ ] Port main cube pipeline (vertex layout, depth, blend, descriptors).
- [ ] Port packed-far LOD pipeline.
- [ ] Port line pipeline (selection/debug chunk bounds).

Exit criteria:
- All world geometry paths match current visuals within acceptable tolerance.

## Phase 4: UI + Dynamic Effects
- [ ] Port UI quads/textures pipeline.
- [ ] Port dropped-item mesh path.
- [ ] Port break overlay and sun billboard paths.

Exit criteria:
- HUD/menu/chat/stats overlays and dynamic overlays match existing behavior.

## Phase 5: Cleanup + Removal
- [ ] Capture perf baseline and compare frame time + memory.
- [ ] Remove `wgpu` renderer path after parity and stability sign-off.
- [ ] Update docs/build instructions for Vulkan-native runtime requirements.

Exit criteria:
- Native Vulkan is default and only render backend in mainline branch.

## Immediate Next Step
Implement Phase 0 backend selector so `main` can request a renderer backend without coupling to `wgpu` internals.
