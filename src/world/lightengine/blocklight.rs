use glam::IVec3;
use std::collections::VecDeque;

use crate::world::blocks::{block_is_collidable, block_light_emission};

use super::compute_face_light;

const BLOCK_LIGHT_MAX_LEVEL: u8 = 15;
const BLOCK_LIGHT_MARGIN: i32 = BLOCK_LIGHT_MAX_LEVEL as i32 + 1;
const BLOCK_LIGHT_GAMMA: f32 = 0.82;

pub const ENABLE_STATIC_BLOCK_LIGHT: bool = false;

pub struct BlockLightField {
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

#[inline]
fn local_idx(x: i32, y: i32, z: i32, size: i32) -> usize {
    (x + y * size + z * size * size) as usize
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
pub fn sample_face_block_light(
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
pub fn combine_sky_and_block_light(sky_light: f32, block_light: f32) -> f32 {
    sky_light.max(block_light).clamp(0.0, 1.0)
}

pub fn build_block_light_field<F>(
    chunk_min: IVec3,
    size: i32,
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
                if id >= 0 {
                    let emission = block_light_emission(id)
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
pub fn compute_face_light_with_block<F>(
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
