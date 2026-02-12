use glam::{IVec3, Vec3};

use crate::render::block::BlockTexture;
use crate::render::mesh::ChunkVertex;
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{BLOCK_LEAVES, BLOCK_LOG};
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
            if height < chunk_min_y && edited_y_range.is_none() {
                x += step;
                continue;
            }

            let mut y_start = chunk_min_y;
            let mut y_end = height.min(chunk_max_y);
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

fn emit_face(
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
) {
    let base = vertices.len() as u32;
    let color = [1.0, 1.0, 1.0, 1.0];
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
        0 | 1 => (fz, fy),
        2 | 3 => (fx, fz),
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
