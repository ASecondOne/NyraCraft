use glam::{IVec3, Vec3};

use crate::render::block::BlockTexture;
use crate::render::mesh::ChunkVertex;
use crate::world::worldgen::WorldGen;
use crate::world::CHUNK_SIZE;

#[derive(Clone)]
pub struct MeshData {
    pub coord: IVec3,
    pub step: i32,
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
) -> MeshData {
    let origin = IVec3::new(coord.x * CHUNK_SIZE, coord.y * CHUNK_SIZE, coord.z * CHUNK_SIZE);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let size = CHUNK_SIZE as i32;
    let half = CHUNK_SIZE / 2;
    let step = step.max(1);
    let dirt_depth = 3;

    let mut z = 0;
    while z < size {
        let mut x = 0;
        while x < size {
            let wx = origin.x + x - half;
            let wz = origin.z + z - half;
            let height = worldgen.height_at(wx, wz);

            let y_top = height;
            let y_bottom = height - dirt_depth;

            let mut y = y_bottom;
            while y <= y_top {
                let block_id = worldgen.block_id_for_height(y, height);
                if block_id >= 0 {
                    let block = &blocks[block_id as usize];
                    let wy = y;

                    if is_air(IVec3::new(wx + step, wy, wz), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 0, wx, wy, wz, step);
                    }
                    if is_air(IVec3::new(wx - step, wy, wz), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 1, wx, wy, wz, step);
                    }
                    if is_air(IVec3::new(wx, wy + step, wz), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 2, wx, wy, wz, step);
                    }
                    if is_air(IVec3::new(wx, wy - step, wz), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 3, wx, wy, wz, step);
                    }
                    if is_air(IVec3::new(wx, wy, wz + step), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 4, wx, wy, wz, step);
                    }
                    if is_air(IVec3::new(wx, wy, wz - step), worldgen) {
                        emit_face(&mut vertices, &mut indices, block, 5, wx, wy, wz, step);
                    }
                }
                y += step;
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
    step: i32,
) {
    let base = vertices.len() as u32;
    let color = block.colors[face as usize].as_f32_rgba();
    let tile = block.tiles[face as usize];
    let rotation = block.rotations[face as usize];

    let s = step as f32;
    let (p0, p1, p2, p3) = match face {
        0 => ( // +X
            [wx as f32 + s, wy as f32, wz as f32],
            [wx as f32 + s, wy as f32 + s, wz as f32],
            [wx as f32 + s, wy as f32 + s, wz as f32 + s],
            [wx as f32 + s, wy as f32, wz as f32 + s],
        ),
        1 => ( // -X
            [wx as f32, wy as f32, wz as f32 + s],
            [wx as f32, wy as f32 + s, wz as f32 + s],
            [wx as f32, wy as f32 + s, wz as f32],
            [wx as f32, wy as f32, wz as f32],
        ),
        2 => ( // +Y
            [wx as f32, wy as f32 + s, wz as f32],
            [wx as f32, wy as f32 + s, wz as f32 + s],
            [wx as f32 + s, wy as f32 + s, wz as f32 + s],
            [wx as f32 + s, wy as f32 + s, wz as f32],
        ),
        3 => ( // -Y
            [wx as f32, wy as f32, wz as f32 + s],
            [wx as f32, wy as f32, wz as f32],
            [wx as f32 + s, wy as f32, wz as f32],
            [wx as f32 + s, wy as f32, wz as f32 + s],
        ),
        4 => ( // +Z
            [wx as f32, wy as f32, wz as f32 + s],
            [wx as f32 + s, wy as f32, wz as f32 + s],
            [wx as f32 + s, wy as f32 + s, wz as f32 + s],
            [wx as f32, wy as f32 + s, wz as f32 + s],
        ),
        _ => ( // -Z
            [wx as f32 + s, wy as f32, wz as f32],
            [wx as f32, wy as f32, wz as f32],
            [wx as f32, wy as f32 + s, wz as f32],
            [wx as f32 + s, wy as f32 + s, wz as f32],
        ),
    };

    let uv_scale = step as f32;
    vertices.push(ChunkVertex {
        position: p0,
        uv: [0.0, uv_scale],
        tile,
        face,
        rotation,
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [uv_scale, uv_scale],
        tile,
        face,
        rotation,
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [uv_scale, 0.0],
        tile,
        face,
        rotation,
        _pad0: 0,
        color,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        _pad0: 0,
        color,
    });

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_air(world: IVec3, worldgen: &WorldGen) -> bool {
    world.y > worldgen.height_at(world.x, world.z)
}

fn chunk_radius() -> f32 {
    let half = (CHUNK_SIZE as f32) * 0.5;
    half * (3.0f32).sqrt()
}
