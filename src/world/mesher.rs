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

pub fn generate_chunk_mesh(
    coord: IVec3,
    blocks: &[BlockTexture],
    worldgen: &WorldGen,
    step: i32,
    mode: MeshMode,
) -> MeshData {
    let origin = IVec3::new(
        coord.x * CHUNK_SIZE,
        coord.y * CHUNK_SIZE,
        coord.z * CHUNK_SIZE,
    );
    let size = CHUNK_SIZE as i32;
    let half = CHUNK_SIZE / 2;
    let height_cache = build_height_cache(worldgen, origin, size, half);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let step = step.max(1);
    let allow_caves = step == 1 && mode == MeshMode::Full;
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
                        let block_id = worldgen.block_id_at(wx, max_h, wz, max_h);
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
                                let height = height_cache
                                    .height_at_world(wx, wz)
                                    .unwrap_or_else(|| worldgen.height_at(wx, wz));
                                if wy <= height {
                                let block_id = worldgen.block_id_at(wx, wy, wz, height);
                                if block_id >= 0 {
                                    counts[block_id as usize] += 1;
                                }
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
            if height < chunk_min_y {
                x += step;
                continue;
            }

            let y_start = chunk_min_y;
            let y_end = height.min(chunk_max_y);
            if y_end < y_start {
                x += step;
                continue;
            }

            let mut y = y_start;
            while y <= y_end {
                if worldgen.is_cave(wx, y, wz, height) {
                    y += step;
                    continue;
                }
                let block_id = worldgen.block_id_at(wx, y, wz, height);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let wy = y;

                    let sx = (size - x).min(step);
                    let sy = ((y - chunk_min_y) % step) + 1;
                    let sz = (size - z).min(step);

                    if is_air(
                        IVec3::new(wx + sx, wy, wz),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                    if is_air(
                        IVec3::new(wx - 1, wy, wz),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                    if is_air(
                        IVec3::new(wx, wy + 1, wz),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                    if is_air(
                        IVec3::new(wx, wy - 1, wz),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                    if is_air(
                        IVec3::new(wx, wy, wz + sz),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                    if is_air(
                        IVec3::new(wx, wy, wz - 1),
                        worldgen,
                        allow_caves,
                        Some(&height_cache),
                    ) {
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
                if !worldgen.is_cave(wx, height, wz, height) {
                    let block_id = worldgen.block_id_at(wx, height, wz, height);
                    if block_id >= 0 {
                        let block = &blocks[block_id as usize];
                        let wy = height;
                        let sx = 1;
                        let sy = 1;
                        let sz = 1;
                        if is_air(
                            IVec3::new(wx + 1, wy, wz),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                        if is_air(
                            IVec3::new(wx - 1, wy, wz),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                        if is_air(
                            IVec3::new(wx, wy + 1, wz),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                        if is_air(
                            IVec3::new(wx, wy - 1, wz),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                        if is_air(
                            IVec3::new(wx, wy, wz + 1),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                        if is_air(
                            IVec3::new(wx, wy, wz - 1),
                            worldgen,
                            allow_caves,
                            Some(&height_cache),
                        ) {
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
                let tree_contains = |x: i32, y: i32, z: i32| -> bool {
                    if x == wx && z == wz && y >= tree.base_y && y < trunk_end {
                        return true;
                    }
                    let dy = y - top;
                    if dy < -r || dy > r {
                        return false;
                    }
                    let dx = x - wx;
                    let dz = z - wz;
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if dist2 <= r * r {
                        if dx == 0 && dz == 0 && y < trunk_end {
                            return false;
                        }
                        return true;
                    }
                    false
                };
                let mut ty = tree.base_y;
                while ty < trunk_end {
                    if ty >= chunk_min_y && ty <= chunk_max_y {
                        let wy = ty;
                        let n = IVec3::new(wx + 1, wy, wz);
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                        if !tree_contains(n.x, n.y, n.z)
                            && is_air(n, worldgen, allow_caves, Some(&height_cache))
                        {
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
                                    let n = IVec3::new(lx + 1, ly, lz);
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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
                                    if !tree_contains(n.x, n.y, n.z)
                                        && is_air(n, worldgen, allow_caves, Some(&height_cache))
                                    {
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

fn is_air(
    world: IVec3,
    worldgen: &WorldGen,
    allow_caves: bool,
    cache: Option<&HeightCache>,
) -> bool {
    let height = if let Some(cache) = cache {
        cache
            .height_at_world(world.x, world.z)
            .unwrap_or_else(|| worldgen.height_at(world.x, world.z))
    } else {
        worldgen.height_at(world.x, world.z)
    };
    if world.y > height {
        return true;
    }
    allow_caves && worldgen.is_cave(world.x, world.y, world.z, height)
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
