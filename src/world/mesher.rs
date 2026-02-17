use glam::{IVec3, Vec3};

use crate::render::block::BlockTexture;
use crate::render::mesh::ChunkVertex;
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{BLOCK_LEAVES, BLOCK_LOG};
use crate::world::lightengine::compute_face_light;
use crate::world::worldgen::WorldGen;

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

    let use_texture = if mode == MeshMode::SurfaceOnly { 0 } else { 1 };
    let use_sky_shading = use_sky_shading && mode == MeshMode::Full && step == 1;

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

    let chunk_min_y = origin.y - half;
    let chunk_max_y = origin.y + half;

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
            if let Some(tree) = worldgen.tree_at(wx, wz, height) {
                let log_block = &blocks[BLOCK_LOG];
                let leaves_block = &blocks[BLOCK_LEAVES];
                let trunk_end = tree.base_y + tree.trunk_h;
                let top = trunk_end;
                let r = tree.leaf_r;
                let mut ty = tree.base_y;
                while ty < trunk_end {
                    if ty >= chunk_min_y && ty <= chunk_max_y {
                        let wy = ty;
                        if block_at(wx, wy, wz) != BLOCK_LOG as i8 {
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
                                    if block_at(lx, ly, lz) != BLOCK_LEAVES as i8 {
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
    let mut voxels = vec![-1_i8; volume];
    let mut z = 0;
    while z < size {
        let mut y = 0;
        while y < size {
            let mut x = 0;
            while x < size {
                let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                voxels[local_idx(x, y, z, size)] = block_at(wx, wy, wz);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(0, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(1, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(2, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(3, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(4, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                    if neighbor < 0 {
                        let (wx, wy, wz) = local_to_world(origin, half, x, y, z);
                        let light =
                            compute_face_light(5, wx, wy, wz, 1, 1, 1, use_sky_shading, block_at);
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
                );
                x0 += width;
            }
            y0 += 1;
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
    let light = compute_face_light(face, wx, wy, wz, sx, sy, sz, use_sky_shading, block_at);
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
    );
}

fn emit_face_with_light(
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
) {
    let base = vertices.len() as u32;
    let color = [light, light, light, 1.0];
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
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [u_scale, v_scale],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [u_scale, 0.0],
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

fn is_air<F>(world: IVec3, block_at: &F) -> bool
where
    F: Fn(i32, i32, i32) -> i8,
{
    block_at(world.x, world.y, world.z) < 0
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
