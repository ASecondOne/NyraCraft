use glam::{IVec3, Vec3};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::render::block::{BlockTexture, RENDER_SHAPE_CROSS};
use crate::render::mesh::ChunkVertex;
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{block_is_collidable, block_texture_by_id, core_block_ids};
use crate::world::lightengine::compute_face_light;
use crate::world::worldgen::{WorldGen, WorldMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    Full = 0,
    SurfaceSides = 1,
    SurfaceOnly = 2,
}

#[derive(Clone)]
pub struct MeshData {
    pub coord: IVec3,
    pub step: i32,
    pub mode: MeshMode,
    pub center: Vec3,
    pub radius: f32,
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

const BLOCK_LIGHT_MAX_LEVEL: u8 = 15;
const BLOCK_LIGHT_MARGIN: i32 = BLOCK_LIGHT_MAX_LEVEL as i32 + 1;
const BLOCK_LIGHT_GAMMA: f32 = 0.82;
const ENABLE_STATIC_BLOCK_LIGHT: bool = false;

struct BlockLightField {
    min: IVec3,
    span: i32,
    levels: Vec<u8>,
}

impl BlockLightField {
    #[inline]
    fn level_at_world(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        let lx = wx - self.min.x;
        let ly = wy - self.min.y;
        let lz = wz - self.min.z;
        if lx < 0 || ly < 0 || lz < 0 || lx >= self.span || ly >= self.span || lz >= self.span {
            return 0;
        }
        self.levels[local_idx(lx, ly, lz, self.span)]
    }
}

pub fn generate_chunk_mesh<F>(
    coord: IVec3,
    blocks: &[BlockTexture],
    worldgen: &WorldGen,
    block_at: &F,
    edited_y_range: Option<(i32, i32)>,
    step: i32,
    mode: MeshMode,
    use_sky_shading: bool,
) -> MeshData
where
    F: Fn(i32, i32, i32) -> i8,
{
    let origin = IVec3::new(
        coord.x * CHUNK_SIZE,
        coord.y * CHUNK_SIZE,
        coord.z * CHUNK_SIZE,
    );
    let size = CHUNK_SIZE;
    let half = CHUNK_SIZE / 2;
    let height_cache = build_height_cache(worldgen, origin, size, half);
    let step = step.max(1);
    let grid = (size / step).max(1);
    let est_faces = match mode {
        MeshMode::Full => (grid * grid * 12).max(1024),
        MeshMode::SurfaceSides => (grid * grid * 8).max(512),
        MeshMode::SurfaceOnly => (grid * grid * 2).max(256),
    } as usize;
    let mut vertices = Vec::with_capacity(est_faces * 4);
    let mut indices = Vec::with_capacity(est_faces * 6);
    let core_ids = core_block_ids();
    let tree_log_idx = usize::try_from(core_ids.log)
        .ok()
        .filter(|&idx| idx < blocks.len());
    let tree_leaves_idx = usize::try_from(core_ids.leaves)
        .ok()
        .filter(|&idx| idx < blocks.len());

    let use_texture = if mode == MeshMode::SurfaceOnly { 0 } else { 1 };
    let use_sky_shading = use_sky_shading && mode == MeshMode::Full && step == 1;
    let chunk_min_y = origin.y - half;
    let chunk_max_y = origin.y + half;

    if mode == MeshMode::Full && step == 1 && edited_y_range.is_none() {
        let mut max_feature_top = i32::MIN;
        let tree_extra = if worldgen.mode == WorldMode::Normal { 10 } else { 0 };
        let scan_margin = if worldgen.mode == WorldMode::Normal { 3 } else { 0 };
        let mut z = -scan_margin;
        while z < size + scan_margin {
            let mut x = -scan_margin;
            while x < size + scan_margin {
                let h = if (0..size).contains(&x) && (0..size).contains(&z) {
                    height_cache.height_at_local(x, z)
                } else {
                    let wx = origin.x + x - half;
                    let wz = origin.z + z - half;
                    worldgen.height_at(wx, wz)
                };
                let top = match worldgen.mode {
                    WorldMode::Grid => h + 2,
                    _ => h + tree_extra,
                };
                if top > max_feature_top {
                    max_feature_top = top;
                }
                x += 1;
            }
            z += 1;
        }
        if chunk_min_y > max_feature_top {
            return MeshData {
                coord,
                step: 1,
                mode: MeshMode::Full,
                center: Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32),
                radius: chunk_radius(),
                vertices: Vec::new(),
                indices: Vec::new(),
            };
        }
    }

    if mode == MeshMode::Full && step == 1 {
        return generate_chunk_mesh_greedy(
            coord,
            origin,
            half,
            size,
            blocks,
            block_at,
            use_texture,
            use_sky_shading,
        );
    }

    if mode != MeshMode::Full {
        if mode == MeshMode::SurfaceOnly {
            let cell = step.max(1);
            let grid = size / cell;
            let mut cz = 0;
            while cz < grid {
                let mut cx = 0;
                while cx < grid {
                    let mut max_h = i32::MIN;
                    let mut oz = 0;
                    while oz < cell {
                        let mut ox = 0;
                        while ox < cell {
                            let hx = cx * cell + ox;
                            let hz = cz * cell + oz;
                            let h = height_cache.height_at_local(hx, hz);
                            if h > max_h {
                                max_h = h;
                            }
                            ox += 1;
                        }
                        oz += 1;
                    }
                    if max_h >= chunk_min_y && max_h <= chunk_max_y {
                        let wx = origin.x + cx * cell - half;
                        let wz = origin.z + cz * cell - half;
                        let block_id = block_at(wx, max_h, wz);
                        if block_id >= 0 {
                            let block = &blocks[block_id as usize];
                            let wx = origin.x + cx * cell - half;
                            let wz = origin.z + cz * cell - half;
                            let wy = max_h;
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                block,
                                2,
                                wx,
                                wy,
                                wz,
                                cell,
                                1,
                                cell,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                    }
                    cx += 1;
                }
                cz += 1;
            }

            let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
            let radius = chunk_radius();

            return MeshData {
                coord,
                step,
                mode,
                center,
                radius,
                vertices,
                indices,
            };
        }

        let cell = step.max(1);
        let grid = size / cell;
        let grid_usize = grid as usize;
        let mut cells = vec![-1_i32; grid_usize * grid_usize * grid_usize];
        let block_count = blocks.len();

        let mut counts = vec![0_i32; block_count];
        let mut cz = 0;
        while cz < grid {
            let mut cy = 0;
            while cy < grid {
                let mut cx = 0;
                while cx < grid {
                    counts.fill(0);
                    let mut oz = 0;
                    while oz < cell {
                        let mut oy = 0;
                        while oy < cell {
                            let mut ox = 0;
                            while ox < cell {
                                let wx = origin.x + cx * cell + ox - half;
                                let wy = origin.y + cy * cell + oy - half;
                                let wz = origin.z + cz * cell + oz - half;
                                if wy < chunk_min_y || wy > chunk_max_y {
                                    ox += 1;
                                    continue;
                                }
                                let block_id = block_at(wx, wy, wz);
                                if block_id >= 0 {
                                    counts[block_id as usize] += 1;
                                }
                                ox += 1;
                            }
                            oy += 1;
                        }
                        oz += 1;
                    }
                    let mut best = -1;
                    let mut best_count = 0;
                    for (i, &c) in counts.iter().enumerate() {
                        if c > best_count {
                            best_count = c;
                            best = i as i32;
                        }
                    }
                    if best_count > 0 {
                        let idx = (cx + cy * grid + cz * grid * grid) as usize;
                        cells[idx] = best;
                    }
                    cx += 1;
                }
                cy += 1;
            }
            cz += 1;
        }

        let mut cz = 0;
        while cz < grid {
            let mut cy = 0;
            while cy < grid {
                let mut cx = 0;
                while cx < grid {
                    let idx = (cx + cy * grid + cz * grid * grid) as usize;
                    let id = cells[idx];
                    if id < 0 {
                        cx += 1;
                        continue;
                    }
                    let block = &blocks[id as usize];
                    let wx = origin.x + cx * cell - half;
                    let wy = origin.y + cy * cell - half;
                    let wz = origin.z + cz * cell - half;

                    let nx = if cx + 1 < grid {
                        cells[(cx + 1 + cy * grid + cz * grid * grid) as usize]
                    } else {
                        -1
                    };
                    let px = if cx > 0 {
                        cells[(cx - 1 + cy * grid + cz * grid * grid) as usize]
                    } else {
                        -1
                    };
                    let ny = if cy + 1 < grid {
                        cells[(cx + (cy + 1) * grid + cz * grid * grid) as usize]
                    } else {
                        -1
                    };
                    let py = if cy > 0 {
                        cells[(cx + (cy - 1) * grid + cz * grid * grid) as usize]
                    } else {
                        -1
                    };
                    let nz = if cz + 1 < grid {
                        cells[(cx + cy * grid + (cz + 1) * grid * grid) as usize]
                    } else {
                        -1
                    };
                    let pz = if cz > 0 {
                        cells[(cx + cy * grid + (cz - 1) * grid * grid) as usize]
                    } else {
                        -1
                    };

                    if mode != MeshMode::SurfaceOnly && nx < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            0,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if mode != MeshMode::SurfaceOnly && px < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            1,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if ny < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            2,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if mode != MeshMode::SurfaceOnly && py < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            3,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if mode != MeshMode::SurfaceOnly && nz < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            4,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if mode != MeshMode::SurfaceOnly && pz < 0 {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            5,
                            wx,
                            wy,
                            wz,
                            cell,
                            cell,
                            cell,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }

                    cx += 1;
                }
                cy += 1;
            }
            cz += 1;
        }

        let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
        let radius = chunk_radius();

        return MeshData {
            coord,
            step,
            mode,
            center,
            radius,
            vertices,
            indices,
        };
    }

    let mut z = 0;
    while z < size {
        let mut x = 0;
        while x < size {
            let wx = origin.x + x - half;
            let wz = origin.z + z - half;
            let height = height_cache
                .height_at_world(wx, wz)
                .unwrap_or_else(|| worldgen.height_at(wx, wz));
            let feature_top = worldgen.highest_solid_y_at(wx, wz);
            if feature_top < chunk_min_y && edited_y_range.is_none() {
                x += step;
                continue;
            }

            let mut y_start = chunk_min_y;
            let mut y_end = feature_top.min(chunk_max_y);
            if let Some((_, edit_max_y)) = edited_y_range {
                // Dirty remeshes must include placed blocks above terrain height.
                y_end = y_end.max(edit_max_y + 1);
            }
            y_start = y_start.max(chunk_min_y);
            y_end = y_end.min(chunk_max_y);
            if y_end < y_start {
                x += step;
                continue;
            }

            let mut y = y_start;
            while y <= y_end {
                let block_id = block_at(wx, y, wz);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let wy = y;

                    let sx = (size - x).min(step);
                    let sy = ((y - chunk_min_y) % step) + 1;
                    let sz = (size - z).min(step);

                    if is_air(IVec3::new(wx + sx, wy, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            0,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx - 1, wy, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            1,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy + 1, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            2,
                            wx,
                            wy,
                            wz,
                            sx,
                            1,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy - 1, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            3,
                            wx,
                            wy,
                            wz,
                            sx,
                            1,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy, wz + sz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            4,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy, wz - 1), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            5,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                }
                y += step;
            }

            if height >= chunk_min_y && height <= chunk_max_y {
                let block_id = block_at(wx, height, wz);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let wy = height;
                    let sx = 1;
                    let sy = 1;
                    let sz = 1;
                    if is_air(IVec3::new(wx + 1, wy, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            0,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx - 1, wy, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            1,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy + 1, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            2,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy - 1, wz), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            3,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy, wz + 1), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            4,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                    if is_air(IVec3::new(wx, wy, wz - 1), block_at) {
                        emit_face(
                            &mut vertices,
                            &mut indices,
                            block,
                            5,
                            wx,
                            wy,
                            wz,
                            sx,
                            sy,
                            sz,
                            use_texture,
                            use_sky_shading,
                            block_at,
                        );
                    }
                }
            }
            x += step;
        }
        z += step;
    }

    let mut z = 0;
    while z < size {
        let mut x = 0;
        while x < size {
            let wx = origin.x + x - half;
            let wz = origin.z + z - half;
            let height = height_cache
                .height_at_world(wx, wz)
                .unwrap_or_else(|| worldgen.height_at(wx, wz));
            if let Some(tree) = worldgen.tree_at(wx, wz, height)
                && let (Some(log_idx), Some(leaves_idx)) = (tree_log_idx, tree_leaves_idx)
            {
                let log_block = &blocks[log_idx];
                let leaves_block = &blocks[leaves_idx];
                let trunk_end = tree.base_y + tree.trunk_h;
                let top = trunk_end;
                let r = tree.leaf_r;
                let mut ty = tree.base_y;
                while ty < trunk_end {
                    if ty >= chunk_min_y && ty <= chunk_max_y {
                        let wy = ty;
                        if block_at(wx, wy, wz) != core_ids.log {
                            ty += 1;
                            continue;
                        }
                        let n = IVec3::new(wx + 1, wy, wz);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                0,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                        let n = IVec3::new(wx - 1, wy, wz);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                1,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                        let n = IVec3::new(wx, wy + 1, wz);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                2,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                        let n = IVec3::new(wx, wy - 1, wz);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                3,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                        let n = IVec3::new(wx, wy, wz + 1);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                4,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                        let n = IVec3::new(wx, wy, wz - 1);
                        if is_air(n, block_at) {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                log_block,
                                5,
                                wx,
                                wy,
                                wz,
                                1,
                                1,
                                1,
                                use_texture,
                                use_sky_shading,
                                block_at,
                            );
                        }
                    }
                    ty += 1;
                }

                let mut dy = -r;
                while dy <= r {
                    let mut dz = -r;
                    while dz <= r {
                        let mut dx = -r;
                        while dx <= r {
                            if dx == 0 && dz == 0 && dy < 0 {
                                dx += 1;
                                continue;
                            }
                            let dist2 = dx * dx + dy * dy + dz * dz;
                            if dist2 <= r * r {
                                let lx = wx + dx;
                                let ly = top + dy;
                                let lz = wz + dz;
                                if ly >= chunk_min_y && ly <= chunk_max_y {
                                    if lx == wx && lz == wz && ly >= tree.base_y && ly < trunk_end {
                                        dx += 1;
                                        continue;
                                    }
                                    if block_at(lx, ly, lz) != core_ids.leaves {
                                        dx += 1;
                                        continue;
                                    }
                                    let n = IVec3::new(lx + 1, ly, lz);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            0,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                    let n = IVec3::new(lx - 1, ly, lz);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            1,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                    let n = IVec3::new(lx, ly + 1, lz);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            2,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                    let n = IVec3::new(lx, ly - 1, lz);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            3,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                    let n = IVec3::new(lx, ly, lz + 1);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            4,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                    let n = IVec3::new(lx, ly, lz - 1);
                                    if is_air(n, block_at) {
                                        emit_face(
                                            &mut vertices,
                                            &mut indices,
                                            leaves_block,
                                            5,
                                            lx,
                                            ly,
                                            lz,
                                            1,
                                            1,
                                            1,
                                            use_texture,
                                            use_sky_shading,
                                            block_at,
                                        );
                                    }
                                }
                            }
                            dx += 1;
                        }
                        dz += 1;
                    }
                    dy += 1;
                }
            }
            x += 1;
        }
        z += 1;
    }

    let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
    let radius = chunk_radius();

    MeshData {
        coord,
        step,
        mode,
        center,
        radius,
        vertices,
        indices,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceMergeKey {
    block_id: i8,
    light_bin: u8,
}

fn quantize_light(light: f32) -> u8 {
    (light.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn dequantize_light(light_bin: u8) -> f32 {
    light_bin as f32 / 255.0
}

#[inline]
fn light_level_to_factor(level: u8) -> f32 {
    if level == 0 {
        0.0
    } else {
        (level as f32 / BLOCK_LIGHT_MAX_LEVEL as f32)
            .powf(BLOCK_LIGHT_GAMMA)
            .clamp(0.0, 1.0)
    }
}

#[inline]
fn face_light_sample_cell(
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
) -> (i32, i32, i32) {
    match face {
        0 => (wx + sx, wy, wz),
        1 => (wx - 1, wy, wz),
        2 => (wx, wy + sy, wz),
        3 => (wx, wy - 1, wz),
        4 => (wx, wy, wz + sz),
        _ => (wx, wy, wz - 1),
    }
}

#[inline]
fn sample_face_block_light(
    block_light_field: Option<&BlockLightField>,
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
) -> f32 {
    let Some(field) = block_light_field else {
        return 0.0;
    };
    let (sx, sy, sz) = face_light_sample_cell(face, wx, wy, wz, sx, sy, sz);
    light_level_to_factor(field.level_at_world(sx, sy, sz))
}

#[inline]
fn combine_sky_and_block_light(sky_light: f32, block_light: f32) -> f32 {
    sky_light.max(block_light).clamp(0.0, 1.0)
}

#[inline]
fn ao_axis_samples(min: i32, span: i32, high: bool) -> (i32, i32) {
    if high {
        (min + span - 1, min + span)
    } else {
        (min, min - 1)
    }
}

#[inline]
fn ao_occlusion_weight(block_id: i8, leaves_id: i8) -> f32 {
    if block_id < 0 {
        0.0
    } else if block_id == leaves_id {
        0.35
    } else {
        1.0
    }
}

#[inline]
fn ao_corner_factor(side_a: f32, side_b: f32, corner: f32) -> f32 {
    let occlusion = if side_a >= 0.99 && side_b >= 0.99 {
        3.0
    } else {
        side_a + side_b + corner
    };
    (1.0 - 0.14 * occlusion).clamp(0.58, 1.0)
}

fn compute_face_corner_ao<F>(
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
    block_at: &F,
) -> [f32; 4]
where
    F: Fn(i32, i32, i32) -> i8,
{
    let leaves_id = core_block_ids().leaves;
    let sample_yz = |x: i32, high_y: bool, high_z: bool| {
        let (y_base, y_side) = ao_axis_samples(wy, sy, high_y);
        let (z_base, z_side) = ao_axis_samples(wz, sz, high_z);
        let side_y = ao_occlusion_weight(block_at(x, y_side, z_base), leaves_id);
        let side_z = ao_occlusion_weight(block_at(x, y_base, z_side), leaves_id);
        let corner = ao_occlusion_weight(block_at(x, y_side, z_side), leaves_id);
        ao_corner_factor(side_y, side_z, corner)
    };
    let sample_xz = |y: i32, high_x: bool, high_z: bool| {
        let (x_base, x_side) = ao_axis_samples(wx, sx, high_x);
        let (z_base, z_side) = ao_axis_samples(wz, sz, high_z);
        let side_x = ao_occlusion_weight(block_at(x_side, y, z_base), leaves_id);
        let side_z = ao_occlusion_weight(block_at(x_base, y, z_side), leaves_id);
        let corner = ao_occlusion_weight(block_at(x_side, y, z_side), leaves_id);
        ao_corner_factor(side_x, side_z, corner)
    };
    let sample_xy = |z: i32, high_x: bool, high_y: bool| {
        let (x_base, x_side) = ao_axis_samples(wx, sx, high_x);
        let (y_base, y_side) = ao_axis_samples(wy, sy, high_y);
        let side_x = ao_occlusion_weight(block_at(x_side, y_base, z), leaves_id);
        let side_y = ao_occlusion_weight(block_at(x_base, y_side, z), leaves_id);
        let corner = ao_occlusion_weight(block_at(x_side, y_side, z), leaves_id);
        ao_corner_factor(side_x, side_y, corner)
    };

    match face {
        0 => {
            let x = wx + sx;
            [
                sample_yz(x, false, false),
                sample_yz(x, true, false),
                sample_yz(x, true, true),
                sample_yz(x, false, true),
            ]
        }
        1 => {
            let x = wx - 1;
            [
                sample_yz(x, false, true),
                sample_yz(x, true, true),
                sample_yz(x, true, false),
                sample_yz(x, false, false),
            ]
        }
        2 => {
            let y = wy + sy;
            [
                sample_xz(y, false, false),
                sample_xz(y, false, true),
                sample_xz(y, true, true),
                sample_xz(y, true, false),
            ]
        }
        3 => {
            let y = wy - 1;
            [
                sample_xz(y, false, true),
                sample_xz(y, false, false),
                sample_xz(y, true, false),
                sample_xz(y, true, true),
            ]
        }
        4 => {
            let z = wz + sz;
            [
                sample_xy(z, false, false),
                sample_xy(z, true, false),
                sample_xy(z, true, true),
                sample_xy(z, false, true),
            ]
        }
        _ => {
            let z = wz - 1;
            [
                sample_xy(z, true, false),
                sample_xy(z, false, false),
                sample_xy(z, false, true),
                sample_xy(z, true, true),
            ]
        }
    }
}

fn local_idx(x: i32, y: i32, z: i32, size: i32) -> usize {
    (x + y * size + z * size * size) as usize
}

fn local_block(voxels: &[i8], x: i32, y: i32, z: i32, size: i32) -> i8 {
    voxels[local_idx(x, y, z, size)]
}

fn local_to_world(origin: IVec3, half: i32, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
    (
        origin.x + x - half,
        origin.y + y - half,
        origin.z + z - half,
    )
}

fn build_block_light_field<F>(
    chunk_min: IVec3,
    size: i32,
    blocks: &[BlockTexture],
    block_at: &F,
) -> Option<BlockLightField>
where
    F: Fn(i32, i32, i32) -> i8,
{
    let span = size + BLOCK_LIGHT_MARGIN * 2;
    if span <= 0 {
        return None;
    }
    let min = chunk_min - IVec3::splat(BLOCK_LIGHT_MARGIN);
    let volume = (span * span * span) as usize;
    let mut ids = vec![-1_i8; volume];
    let mut levels = vec![0_u8; volume];
    let mut queue = VecDeque::<(i32, i32, i32)>::new();

    let mut z = 0;
    while z < span {
        let mut y = 0;
        while y < span {
            let mut x = 0;
            while x < span {
                let wx = min.x + x;
                let wy = min.y + y;
                let wz = min.z + z;
                let idx = local_idx(x, y, z, span);
                let id = block_at(wx, wy, wz);
                ids[idx] = id;
                if id >= 0
                    && let Some(block) = blocks.get(id as usize)
                {
                    let emission = block
                        .light_emission
                        .clamp(0.0, BLOCK_LIGHT_MAX_LEVEL as f32)
                        .round() as u8;
                    if emission > 0 {
                        levels[idx] = emission;
                        queue.push_back((x, y, z));
                    }
                }
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }

    if queue.is_empty() {
        return None;
    }

    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    while let Some((x, y, z)) = queue.pop_front() {
        let idx = local_idx(x, y, z, span);
        let level = levels[idx];
        if level <= 1 {
            continue;
        }
        let next = level - 1;
        for &(dx, dy, dz) in &DIRS {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if nx < 0 || ny < 0 || nz < 0 || nx >= span || ny >= span || nz >= span {
                continue;
            }
            let nidx = local_idx(nx, ny, nz, span);
            if block_is_collidable(ids[nidx]) || levels[nidx] >= next {
                continue;
            }
            levels[nidx] = next;
            queue.push_back((nx, ny, nz));
        }
    }

    Some(BlockLightField { min, span, levels })
}

#[inline]
fn compute_face_light_with_block<F>(
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
    use_sky_shading: bool,
    block_at: &F,
    block_light_field: Option<&BlockLightField>,
) -> f32
where
    F: Fn(i32, i32, i32) -> i8,
{
    let sky_light = compute_face_light(face, wx, wy, wz, sx, sy, sz, use_sky_shading, block_at);
    let block_light = sample_face_block_light(block_light_field, face, wx, wy, wz, sx, sy, sz);
    combine_sky_and_block_light(sky_light, block_light)
}

fn generate_chunk_mesh_greedy<F>(
    coord: IVec3,
    origin: IVec3,
    half: i32,
    size: i32,
    blocks: &[BlockTexture],
    block_at: &F,
    use_texture: u32,
    use_sky_shading: bool,
) -> MeshData
where
    F: Fn(i32, i32, i32) -> i8,
{
    let volume = (size * size * size) as usize;
    let area = (size * size) as usize;
    let halo_span = size + 2;
    let halo_min = IVec3::new(origin.x - half - 1, origin.y - half - 1, origin.z - half - 1);
    let mut halo = vec![-1_i8; (halo_span * halo_span * halo_span) as usize];
    let mut hz = 0;
    while hz < halo_span {
        let mut hy = 0;
        while hy < halo_span {
            let mut hx = 0;
            while hx < halo_span {
                let wx = halo_min.x + hx;
                let wy = halo_min.y + hy;
                let wz = halo_min.z + hz;
                halo[local_idx(hx, hy, hz, halo_span)] = block_at(wx, wy, wz);
                hx += 1;
            }
            hy += 1;
        }
        hz += 1;
    }
    let chunk_min = IVec3::new(origin.x - half, origin.y - half, origin.z - half);
    let block_at_raw = block_at;
    let spill_cache = RefCell::new(HashMap::<(i32, i32, i32), i8>::with_capacity(1024));
    let cached_block_at = |wx: i32, wy: i32, wz: i32| -> i8 {
        let lx = wx - chunk_min.x;
        let ly = wy - chunk_min.y;
        let lz = wz - chunk_min.z;
        if (-1..=size).contains(&lx) && (-1..=size).contains(&ly) && (-1..=size).contains(&lz) {
            return halo[local_idx(lx + 1, ly + 1, lz + 1, halo_span)];
        }
        let key = (wx, wy, wz);
        if let Some(id) = spill_cache.borrow().get(&key).copied() {
            return id;
        }
        let id = block_at_raw(wx, wy, wz);
        spill_cache.borrow_mut().insert(key, id);
        id
    };
    let block_at = &cached_block_at;
    let block_light_field = if ENABLE_STATIC_BLOCK_LIGHT {
        let has_emissive_in_halo = halo.iter().copied().any(|id| {
            id >= 0
                && blocks
                    .get(id as usize)
                    .map(|block| block.light_emission > 0.0)
                    .unwrap_or(false)
        });
        if has_emissive_in_halo {
            build_block_light_field(chunk_min, size, blocks, block_at)
        } else {
            None
        }
    } else {
        None
    };

    let mut voxels = vec![-1_i8; volume];
    let mut z = 0;
    while z < size {
        let mut y = 0;
        while y < size {
            let mut x = 0;
            while x < size {
                let id = halo[local_idx(x + 1, y + 1, z + 1, halo_span)];
                voxels[local_idx(x, y, z, size)] = if id >= 0
                    && blocks
                        .get(id as usize)
                        .map(|block| block.render_shape == RENDER_SHAPE_CROSS)
                        .unwrap_or(false)
                {
                    -1
                } else {
                    id
                };
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }

    let mut vertices = Vec::with_capacity(area * 12);
    let mut indices = Vec::with_capacity(area * 18);

    // +X faces (face 0), merge on Y/Z plane.
    let mut x = 0;
    while x < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut y = 0;
        while y < size {
            let mut z = 0;
            while z < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if x + 1 < size {
                        local_block(&voxels, x + 1, y, z, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x + 1, y, z);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            0,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(y * size + z) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                z += 1;
            }
            y += 1;
        }

        let mut y0 = 0;
        while y0 < size {
            let mut z0 = 0;
            while z0 < size {
                let idx = (y0 * size + z0) as usize;
                let Some(key) = mask[idx] else {
                    z0 += 1;
                    continue;
                };

                let mut width = 1;
                while z0 + width < size && mask[(y0 * size + z0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_xp: while y0 + height < size {
                    let mut dz = 0;
                    while dz < width {
                        if mask[((y0 + height) * size + z0 + dz) as usize] != Some(key) {
                            break 'grow_xp;
                        }
                        dz += 1;
                    }
                    height += 1;
                }

                let mut dy = 0;
                while dy < height {
                    let mut dz = 0;
                    while dz < width {
                        mask[((y0 + dy) * size + z0 + dz) as usize] = None;
                        dz += 1;
                    }
                    dy += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x, y0, z0);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    0,
                    wx,
                    wy,
                    wz,
                    1,
                    height,
                    width,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                z0 += width;
            }
            y0 += 1;
        }
        x += 1;
    }

    // -X faces (face 1), merge on Y/Z plane.
    let mut x = 0;
    while x < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut y = 0;
        while y < size {
            let mut z = 0;
            while z < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if x > 0 {
                        local_block(&voxels, x - 1, y, z, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x - 1, y, z);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            1,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(y * size + z) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                z += 1;
            }
            y += 1;
        }

        let mut y0 = 0;
        while y0 < size {
            let mut z0 = 0;
            while z0 < size {
                let idx = (y0 * size + z0) as usize;
                let Some(key) = mask[idx] else {
                    z0 += 1;
                    continue;
                };

                let mut width = 1;
                while z0 + width < size && mask[(y0 * size + z0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_xn: while y0 + height < size {
                    let mut dz = 0;
                    while dz < width {
                        if mask[((y0 + height) * size + z0 + dz) as usize] != Some(key) {
                            break 'grow_xn;
                        }
                        dz += 1;
                    }
                    height += 1;
                }

                let mut dy = 0;
                while dy < height {
                    let mut dz = 0;
                    while dz < width {
                        mask[((y0 + dy) * size + z0 + dz) as usize] = None;
                        dz += 1;
                    }
                    dy += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x, y0, z0);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    1,
                    wx,
                    wy,
                    wz,
                    1,
                    height,
                    width,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                z0 += width;
            }
            y0 += 1;
        }
        x += 1;
    }

    // +Y faces (face 2), merge on X/Z plane.
    let mut y = 0;
    while y < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut x = 0;
        while x < size {
            let mut z = 0;
            while z < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if y + 1 < size {
                        local_block(&voxels, x, y + 1, z, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y + 1, z);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            2,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(x * size + z) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                z += 1;
            }
            x += 1;
        }

        let mut x0 = 0;
        while x0 < size {
            let mut z0 = 0;
            while z0 < size {
                let idx = (x0 * size + z0) as usize;
                let Some(key) = mask[idx] else {
                    z0 += 1;
                    continue;
                };

                let mut width = 1;
                while z0 + width < size && mask[(x0 * size + z0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_yp: while x0 + height < size {
                    let mut dz = 0;
                    while dz < width {
                        if mask[((x0 + height) * size + z0 + dz) as usize] != Some(key) {
                            break 'grow_yp;
                        }
                        dz += 1;
                    }
                    height += 1;
                }

                let mut dx = 0;
                while dx < height {
                    let mut dz = 0;
                    while dz < width {
                        mask[((x0 + dx) * size + z0 + dz) as usize] = None;
                        dz += 1;
                    }
                    dx += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x0, y, z0);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    2,
                    wx,
                    wy,
                    wz,
                    height,
                    1,
                    width,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                z0 += width;
            }
            x0 += 1;
        }
        y += 1;
    }

    // -Y faces (face 3), merge on X/Z plane.
    let mut y = 0;
    while y < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut x = 0;
        while x < size {
            let mut z = 0;
            while z < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if y > 0 {
                        local_block(&voxels, x, y - 1, z, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y - 1, z);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            3,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(x * size + z) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                z += 1;
            }
            x += 1;
        }

        let mut x0 = 0;
        while x0 < size {
            let mut z0 = 0;
            while z0 < size {
                let idx = (x0 * size + z0) as usize;
                let Some(key) = mask[idx] else {
                    z0 += 1;
                    continue;
                };

                let mut width = 1;
                while z0 + width < size && mask[(x0 * size + z0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_yn: while x0 + height < size {
                    let mut dz = 0;
                    while dz < width {
                        if mask[((x0 + height) * size + z0 + dz) as usize] != Some(key) {
                            break 'grow_yn;
                        }
                        dz += 1;
                    }
                    height += 1;
                }

                let mut dx = 0;
                while dx < height {
                    let mut dz = 0;
                    while dz < width {
                        mask[((x0 + dx) * size + z0 + dz) as usize] = None;
                        dz += 1;
                    }
                    dx += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x0, y, z0);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    3,
                    wx,
                    wy,
                    wz,
                    height,
                    1,
                    width,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                z0 += width;
            }
            x0 += 1;
        }
        y += 1;
    }

    // +Z faces (face 4), merge on Y/X plane.
    let mut z = 0;
    while z < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut y = 0;
        while y < size {
            let mut x = 0;
            while x < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if z + 1 < size {
                        local_block(&voxels, x, y, z + 1, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z + 1);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            4,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(y * size + x) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                x += 1;
            }
            y += 1;
        }

        let mut y0 = 0;
        while y0 < size {
            let mut x0 = 0;
            while x0 < size {
                let idx = (y0 * size + x0) as usize;
                let Some(key) = mask[idx] else {
                    x0 += 1;
                    continue;
                };

                let mut width = 1;
                while x0 + width < size && mask[(y0 * size + x0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_zp: while y0 + height < size {
                    let mut dx = 0;
                    while dx < width {
                        if mask[((y0 + height) * size + x0 + dx) as usize] != Some(key) {
                            break 'grow_zp;
                        }
                        dx += 1;
                    }
                    height += 1;
                }

                let mut dy = 0;
                while dy < height {
                    let mut dx = 0;
                    while dx < width {
                        mask[((y0 + dy) * size + x0 + dx) as usize] = None;
                        dx += 1;
                    }
                    dy += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x0, y0, z);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    4,
                    wx,
                    wy,
                    wz,
                    width,
                    height,
                    1,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                x0 += width;
            }
            y0 += 1;
        }
        z += 1;
    }

    // -Z faces (face 5), merge on Y/X plane.
    let mut z = 0;
    while z < size {
        let mut mask: Vec<Option<FaceMergeKey>> = vec![None; area];
        let mut y = 0;
        while y < size {
            let mut x = 0;
            while x < size {
                let block_id = local_block(&voxels, x, y, z, size);
                if block_id >= 0 {
                    let neighbor = if z > 0 {
                        local_block(&voxels, x, y, z - 1, size)
                    } else {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z - 1);
                        block_at(wx, wy, wz)
                    };
                    if !is_face_occluder(neighbor) {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light = compute_face_light_with_block(
                            5,
                            wx,
                            wy,
                            wz,
                            1,
                            1,
                            1,
                            use_sky_shading,
                            block_at,
                            block_light_field.as_ref(),
                        );
                        mask[(y * size + x) as usize] = Some(FaceMergeKey {
                            block_id,
                            light_bin: quantize_light(light),
                        });
                    }
                }
                x += 1;
            }
            y += 1;
        }

        let mut y0 = 0;
        while y0 < size {
            let mut x0 = 0;
            while x0 < size {
                let idx = (y0 * size + x0) as usize;
                let Some(key) = mask[idx] else {
                    x0 += 1;
                    continue;
                };

                let mut width = 1;
                while x0 + width < size && mask[(y0 * size + x0 + width) as usize] == Some(key) {
                    width += 1;
                }

                let mut height = 1;
                'grow_zn: while y0 + height < size {
                    let mut dx = 0;
                    while dx < width {
                        if mask[((y0 + height) * size + x0 + dx) as usize] != Some(key) {
                            break 'grow_zn;
                        }
                        dx += 1;
                    }
                    height += 1;
                }

                let mut dy = 0;
                while dy < height {
                    let mut dx = 0;
                    while dx < width {
                        mask[((y0 + dy) * size + x0 + dx) as usize] = None;
                        dx += 1;
                    }
                    dy += 1;
                }

                let block = &blocks[key.block_id as usize];
                let (wx, wy, wz) = local_to_world(origin, half, x0, y0, z);
                emit_face_with_light(
                    &mut vertices,
                    &mut indices,
                    block,
                    5,
                    wx,
                    wy,
                    wz,
                    width,
                    height,
                    1,
                    use_texture,
                    dequantize_light(key.light_bin),
                    use_sky_shading,
                    block_at,
                );
                x0 += width;
            }
            y0 += 1;
        }
        z += 1;
    }

    let mut z = 0;
    while z < size {
        let mut y = 0;
        while y < size {
            let mut x = 0;
            while x < size {
                let id = halo[local_idx(x + 1, y + 1, z + 1, halo_span)];
                if id >= 0
                    && let Some(block) = blocks.get(id as usize)
                    && block.render_shape == RENDER_SHAPE_CROSS
                {
                    let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                    emit_cross_plant(
                        &mut vertices,
                        &mut indices,
                        block,
                        wx,
                        wy,
                        wz,
                        use_texture,
                        use_sky_shading,
                        block_at,
                    );
                }
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }

    let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
    let radius = chunk_radius();
    MeshData {
        coord,
        step: 1,
        mode: MeshMode::Full,
        center,
        radius,
        vertices,
        indices,
    }
}

fn emit_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    tile: u32,
    face: u32,
    rotation: u32,
    use_texture: u32,
    transparent_mode: u32,
    color: [f32; 4],
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
) {
    let base = vertices.len() as u32;
    vertices.push(ChunkVertex {
        position: p0,
        uv: [0.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [1.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [1.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn emit_cross_plant<F>(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    block: &BlockTexture,
    wx: i32,
    wy: i32,
    wz: i32,
    use_texture: u32,
    use_sky_shading: bool,
    block_at: &F,
) where
    F: Fn(i32, i32, i32) -> i8,
{
    let light = compute_face_light_with_block(
        2,
        wx,
        wy,
        wz,
        1,
        1,
        1,
        use_sky_shading,
        block_at,
        None,
    );
    let emissive = (block.light_emission / 15.0).clamp(0.0, 1.0);
    let shade = light.max(emissive);
    let color = [shade, shade, shade, 1.0];
    let tile = block.tiles[2];
    let rotation = block.rotations[2];
    let transparent_mode = block.transparent_mode[2];
    let x = wx as f32;
    let y = wy as f32;
    let z = wz as f32;

    let a0 = [x, y, z];
    let a1 = [x + 1.0, y, z + 1.0];
    let a2 = [x + 1.0, y + 1.0, z + 1.0];
    let a3 = [x, y + 1.0, z];

    let b0 = [x + 1.0, y, z];
    let b1 = [x, y, z + 1.0];
    let b2 = [x, y + 1.0, z + 1.0];
    let b3 = [x + 1.0, y + 1.0, z];

    emit_quad(
        vertices,
        indices,
        tile,
        4,
        rotation,
        use_texture,
        transparent_mode,
        color,
        a0,
        a1,
        a2,
        a3,
    );
    emit_quad(
        vertices,
        indices,
        tile,
        5,
        rotation,
        use_texture,
        transparent_mode,
        color,
        a1,
        a0,
        a3,
        a2,
    );
    emit_quad(
        vertices,
        indices,
        tile,
        0,
        rotation,
        use_texture,
        transparent_mode,
        color,
        b0,
        b1,
        b2,
        b3,
    );
    emit_quad(
        vertices,
        indices,
        tile,
        1,
        rotation,
        use_texture,
        transparent_mode,
        color,
        b1,
        b0,
        b3,
        b2,
    );
}

fn emit_face<F>(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    block: &BlockTexture,
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
    use_texture: u32,
    use_sky_shading: bool,
    block_at: &F,
) where
    F: Fn(i32, i32, i32) -> i8,
{
    let light = compute_face_light_with_block(
        face,
        wx,
        wy,
        wz,
        sx,
        sy,
        sz,
        use_sky_shading,
        block_at,
        None,
    );
    emit_face_with_light(
        vertices,
        indices,
        block,
        face,
        wx,
        wy,
        wz,
        sx,
        sy,
        sz,
        use_texture,
        light,
        use_sky_shading,
        block_at,
    );
}

fn emit_face_with_light<F>(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    block: &BlockTexture,
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
    use_texture: u32,
    light: f32,
    use_ambient_occlusion: bool,
    block_at: &F,
) where
    F: Fn(i32, i32, i32) -> i8,
{
    let base = vertices.len() as u32;
    let ao = if use_ambient_occlusion {
        compute_face_corner_ao(face, wx, wy, wz, sx, sy, sz, block_at)
    } else {
        [1.0; 4]
    };
    let emissive = (block.light_emission / 15.0).clamp(0.0, 1.0);
    let color0 = [
        (light * ao[0]).max(emissive),
        (light * ao[0]).max(emissive),
        (light * ao[0]).max(emissive),
        1.0,
    ];
    let color1 = [
        (light * ao[1]).max(emissive),
        (light * ao[1]).max(emissive),
        (light * ao[1]).max(emissive),
        1.0,
    ];
    let color2 = [
        (light * ao[2]).max(emissive),
        (light * ao[2]).max(emissive),
        (light * ao[2]).max(emissive),
        1.0,
    ];
    let color3 = [
        (light * ao[3]).max(emissive),
        (light * ao[3]).max(emissive),
        (light * ao[3]).max(emissive),
        1.0,
    ];
    let tile = block.tiles[face as usize];
    let rotation = block.rotations[face as usize];
    let transparent_mode = block.transparent_mode[face as usize];

    let fx = sx as f32;
    let fy = sy as f32;
    let fz = sz as f32;
    let (p0, p1, p2, p3) = match face {
        0 => (
            // +X
            [wx as f32 + fx, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32 + fy, wz as f32],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
        ),
        1 => (
            // -X
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32],
            [wx as f32, wy as f32, wz as f32],
        ),
        2 => (
            // +Y
            [wx as f32, wy as f32 + fy, wz as f32],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32],
        ),
        3 => (
            // -Y
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
        ),
        4 => (
            // +Z
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
        ),
        _ => (
            // -Z
            [wx as f32 + fx, wy as f32, wz as f32],
            [wx as f32, wy as f32, wz as f32],
            [wx as f32, wy as f32 + fy, wz as f32],
            [wx as f32 + fx, wy as f32 + fy, wz as f32],
        ),
    };

    let (u_scale, v_scale) = match face {
        0 | 1 => (fy, fz),
        2 | 3 => (fz, fx),
        _ => (fx, fy),
    };
    vertices.push(ChunkVertex {
        position: p0,
        uv: [0.0, v_scale],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color: color0,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [u_scale, v_scale],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color: color1,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [u_scale, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color: color2,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color: color3,
    });

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_air<F>(world: IVec3, block_at: &F) -> bool
where
    F: Fn(i32, i32, i32) -> i8,
{
    let id = block_at(world.x, world.y, world.z);
    id < 0 || !block_is_collidable(id)
}

#[inline]
fn is_face_occluder(id: i8) -> bool {
    if id < 0 || !block_is_collidable(id) {
        return false;
    }
    if let Some(block) = block_texture_by_id(id) {
        if block.render_shape == RENDER_SHAPE_CROSS {
            return false;
        }
        // Fully cutout blocks (e.g. leaves) should not cull adjacent faces.
        if block.transparent_mode.iter().all(|&mode| mode != 0) {
            return false;
        }
        return true;
    }
    true
}

fn chunk_radius() -> f32 {
    let half = (CHUNK_SIZE as f32) * 0.5;
    half * (3.0f32).sqrt()
}

struct HeightCache {
    min_x: i32,
    min_z: i32,
    size: i32,
    heights: Vec<i32>,
}

impl HeightCache {
    fn height_at_local(&self, lx: i32, lz: i32) -> i32 {
        let idx = (lz * self.size + lx) as usize;
        self.heights[idx]
    }

    fn height_at_world(&self, wx: i32, wz: i32) -> Option<i32> {
        if wx < self.min_x || wz < self.min_z {
            return None;
        }
        let lx = wx - self.min_x;
        let lz = wz - self.min_z;
        if lx < 0 || lz < 0 || lx >= self.size || lz >= self.size {
            return None;
        }
        Some(self.height_at_local(lx, lz))
    }
}

fn build_height_cache(worldgen: &WorldGen, origin: IVec3, size: i32, half: i32) -> HeightCache {
    let min_x = origin.x - half;
    let min_z = origin.z - half;
    let mut heights = Vec::with_capacity((size * size) as usize);
    let mut z = 0;
    while z < size {
        let mut x = 0;
        while x < size {
            let h = worldgen.height_at(min_x + x, min_z + z);
            heights.push(h);
            x += 1;
        }
        z += 1;
    }
    HeightCache {
        min_x,
        min_z,
        size,
        heights,
    }
}
