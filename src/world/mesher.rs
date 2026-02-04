use glam::{IVec3, Vec3};

use crate::render::block::BlockTexture;
use crate::render::mesh::ChunkVertex;
use crate::world::worldgen::WorldGen;
use crate::world::CHUNK_SIZE;

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
    pub is_super: bool,
    pub super_size: i32,
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
    let origin = IVec3::new(coord.x * CHUNK_SIZE, coord.y * CHUNK_SIZE, coord.z * CHUNK_SIZE);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let size = CHUNK_SIZE as i32;
    let half = CHUNK_SIZE / 2;
    let step = step.max(1);
    let allow_caves = step == 1 && mode == MeshMode::Full;
    let use_texture = if mode == MeshMode::SurfaceOnly { 0 } else { 1 };

    let chunk_min_y = origin.y - half;
    let chunk_max_y = origin.y + half;

    if mode != MeshMode::Full {
        let mut z = 0;
        while z < size {
            let mut x = 0;
            while x < size {
                let wx = origin.x + x - half;
                let wz = origin.z + z - half;
                let height = worldgen.height_at(wx, wz);
                if height < chunk_min_y || height > chunk_max_y {
                    x += step;
                    continue;
                }
                let block_id = worldgen.block_id_for_height(height, height);
                if block_id < 0 {
                    x += step;
                    continue;
                }
                let block = &blocks[block_id as usize];
                let wy = height;
                let sx = (size - x).min(step);
                let sz = (size - z).min(step);
                emit_face(&mut vertices, &mut indices, block, 2, wx, wy, wz, sx, 1, sz, use_texture);

                if mode == MeshMode::SurfaceSides {
                    let h_px = worldgen.height_at(wx + sx, wz);
                    let h_mx = worldgen.height_at(wx - 1, wz);
                    let h_pz = worldgen.height_at(wx, wz + sz);
                    let h_mz = worldgen.height_at(wx, wz - 1);

                    if h_px < height {
                        let base = (h_px + 1).max(chunk_min_y);
                        let top = height.min(chunk_max_y);
                        if base <= top {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                block,
                                0,
                                wx,
                                base,
                                wz,
                                sx,
                                top - base + 1,
                                sz,
                                use_texture,
                            );
                        }
                    }
                    if h_mx < height {
                        let base = (h_mx + 1).max(chunk_min_y);
                        let top = height.min(chunk_max_y);
                        if base <= top {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                block,
                                1,
                                wx,
                                base,
                                wz,
                                sx,
                                top - base + 1,
                                sz,
                                use_texture,
                            );
                        }
                    }
                    if h_pz < height {
                        let base = (h_pz + 1).max(chunk_min_y);
                        let top = height.min(chunk_max_y);
                        if base <= top {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                block,
                                4,
                                wx,
                                base,
                                wz,
                                sx,
                                top - base + 1,
                                sz,
                                use_texture,
                            );
                        }
                    }
                    if h_mz < height {
                        let base = (h_mz + 1).max(chunk_min_y);
                        let top = height.min(chunk_max_y);
                        if base <= top {
                            emit_face(
                                &mut vertices,
                                &mut indices,
                                block,
                                5,
                                wx,
                                base,
                                wz,
                                sx,
                                top - base + 1,
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

        let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
        let radius = chunk_radius();

        return MeshData {
            coord,
            step,
            mode,
            is_super: false,
            super_size: 1,
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
            let height = worldgen.height_at(wx, wz);
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
                let block_id = worldgen.block_id_for_height(y, height);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let wy = y;

                    let sx = (size - x).min(step);
                    let sy = ((y - chunk_min_y) % step) + 1;
                    let sz = (size - z).min(step);

                    if is_air(IVec3::new(wx + sx, wy, wz), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 0, wx, wy, wz, sx, sy, sz, use_texture);
                    }
                    if is_air(IVec3::new(wx - 1, wy, wz), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 1, wx, wy, wz, sx, sy, sz, use_texture);
                    }
                    if is_air(IVec3::new(wx, wy + 1, wz), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 2, wx, wy, wz, sx, 1, sz, use_texture);
                    }
                    if is_air(IVec3::new(wx, wy - 1, wz), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 3, wx, wy, wz, sx, 1, sz, use_texture);
                    }
                    if is_air(IVec3::new(wx, wy, wz + sz), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 4, wx, wy, wz, sx, sy, sz, use_texture);
                    }
                    if is_air(IVec3::new(wx, wy, wz - 1), worldgen, allow_caves) {
                        emit_face(&mut vertices, &mut indices, block, 5, wx, wy, wz, sx, sy, sz, use_texture);
                    }
                }
                y += step;
            }

            if height >= chunk_min_y && height <= chunk_max_y {
                if !worldgen.is_cave(wx, height, wz, height) {
                    let block_id = worldgen.block_id_for_height(height, height);
                    if block_id >= 0 {
                        let block = &blocks[block_id as usize];
                        let wy = height;
                        let sx = 1;
                        let sy = 1;
                        let sz = 1;
                        if is_air(IVec3::new(wx + 1, wy, wz), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 0, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                        if is_air(IVec3::new(wx - 1, wy, wz), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 1, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                        if is_air(IVec3::new(wx, wy + 1, wz), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 2, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                        if is_air(IVec3::new(wx, wy - 1, wz), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 3, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                        if is_air(IVec3::new(wx, wy, wz + 1), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 4, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                        if is_air(IVec3::new(wx, wy, wz - 1), worldgen, allow_caves) {
                            emit_face(&mut vertices, &mut indices, block, 5, wx, wy, wz, sx, sy, sz, use_texture);
                        }
                    }
                }
            }
            x += step;
        }
        z += step;
    }

    let center = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
    let radius = chunk_radius();

    MeshData {
        coord,
        step,
        mode,
        is_super: false,
        super_size: 1,
        center,
        radius,
        vertices,
        indices,
    }
}

pub fn generate_super_chunk_mesh(
    min_chunk: IVec3,
    blocks: &[BlockTexture],
    worldgen: &WorldGen,
    step: i32,
    super_size: i32,
) -> MeshData {
    let step = step.max(1);
    let size = CHUNK_SIZE * super_size;
    let half = CHUNK_SIZE / 2;
    let min_world_x = min_chunk.x * CHUNK_SIZE - half;
    let min_world_y = min_chunk.y * CHUNK_SIZE - half;
    let min_world_z = min_chunk.z * CHUNK_SIZE - half;
    let max_world_y = min_world_y + CHUNK_SIZE - 1;
    let use_texture = 0;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut z = 0;
    while z < size {
        let mut x = 0;
        while x < size {
            let wx = min_world_x + x;
            let wz = min_world_z + z;
            let height = worldgen.height_at(wx, wz);
            if height >= min_world_y && height <= max_world_y {
                let block_id = worldgen.block_id_for_height(height, height);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let sx = (size - x).min(step);
                    let sz = (size - z).min(step);
                    emit_face(&mut vertices, &mut indices, block, 2, wx, height, wz, sx, 1, sz, use_texture);
                }
            }
            x += step;
        }
        z += step;
    }

    let center = Vec3::new(
        min_world_x as f32 + (size as f32) * 0.5,
        min_world_y as f32 + (CHUNK_SIZE as f32) * 0.5,
        min_world_z as f32 + (size as f32) * 0.5,
    );
    let half_xz = (size as f32) * 0.5;
    let half_y = (CHUNK_SIZE as f32) * 0.5;
    let radius = Vec3::new(half_xz, half_y, half_xz).length();

    MeshData {
        coord: min_chunk,
        step,
        mode: MeshMode::SurfaceOnly,
        is_super: true,
        super_size,
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
    let color = block.colors[face as usize].as_f32_rgba();
    let tile = block.tiles[face as usize];
    let rotation = block.rotations[face as usize];

    let fx = sx as f32;
    let fy = sy as f32;
    let fz = sz as f32;
    let (p0, p1, p2, p3) = match face {
        0 => ( // +X
            [wx as f32 + fx, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32 + fy, wz as f32],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
        ),
        1 => ( // -X
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32],
            [wx as f32, wy as f32, wz as f32],
        ),
        2 => ( // +Y
            [wx as f32, wy as f32 + fy, wz as f32],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32],
        ),
        3 => ( // -Y
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32, wz as f32],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
        ),
        4 => ( // +Z
            [wx as f32, wy as f32, wz as f32 + fz],
            [wx as f32 + fx, wy as f32, wz as f32 + fz],
            [wx as f32 + fx, wy as f32 + fy, wz as f32 + fz],
            [wx as f32, wy as f32 + fy, wz as f32 + fz],
        ),
        _ => ( // -Z
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
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [u_scale, v_scale],
        tile,
        face,
        rotation,
        use_texture,
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [u_scale, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        _pad0: 0,
        color,
    });

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_air(world: IVec3, worldgen: &WorldGen, allow_caves: bool) -> bool {
    let height = worldgen.height_at(world.x, world.z);
    if world.y > height {
        return true;
    }
    allow_caves && worldgen.is_cave(world.x, world.y, world.z, height)
}

fn chunk_radius() -> f32 {
    let half = (CHUNK_SIZE as f32) * 0.5;
    half * (3.0f32).sqrt()
}
