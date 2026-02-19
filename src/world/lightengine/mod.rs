use crate::world::blocks::BLOCK_LEAVES;

pub fn compute_face_light<F>(
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
    use_sky_shading: bool,
    block_at: &F,
) -> f32
where
    F: Fn(i32, i32, i32) -> i8,
{
    let base = match face {
        2 => 1.00f32, // top
        3 => 0.36f32, // bottom
        _ => 0.62f32, // sides
    };
    if !use_sky_shading {
        // Fallback shading for non-sky passes (lower contrast, but not over-bright).
        return (0.02f32 + base * 0.45f32).clamp(0.01f32, 0.55f32);
    }
    let (cx, cy, cz) = face_center_sample(face, wx, wy, wz, sx, sy, sz);
    let mut sky = sample_sky_visibility_fast(face, cx, cy, cz, block_at);
    if face == 3 {
        // Bottom faces receive less direct sky lighting.
        sky *= 0.74;
    }
    // Square visibility so enclosed spaces drop off harder.
    let sky_curve = 0.28f32 * sky + 0.72f32 * sky * sky;
    let ambient = 0.004f32 + 0.022f32 * sky;
    (ambient + base * sky_curve).clamp(0.0f32, 1.0f32)
}

fn face_center_sample(
    face: u32,
    wx: i32,
    wy: i32,
    wz: i32,
    sx: i32,
    sy: i32,
    sz: i32,
) -> (f32, f32, f32) {
    let fx = sx as f32;
    let fy = sy as f32;
    let fz = sz as f32;
    let px = wx as f32;
    let py = wy as f32;
    let pz = wz as f32;
    let e = 0.01f32;
    match face {
        0 => (px + fx + e, py + fy * 0.5, pz + fz * 0.5),
        1 => (px - e, py + fy * 0.5, pz + fz * 0.5),
        2 => (px + fx * 0.5, py + fy + e, pz + fz * 0.5),
        3 => (px + fx * 0.5, py - e, pz + fz * 0.5),
        4 => (px + fx * 0.5, py + fy * 0.5, pz + fz + e),
        _ => (px + fx * 0.5, py + fy * 0.5, pz - e),
    }
}

fn sample_sky_visibility_fast<F>(face: u32, x: f32, y: f32, z: f32, block_at: &F) -> f32
where
    F: Fn(i32, i32, i32) -> i8,
{
    let bx = x.floor() as i32;
    let bz = z.floor() as i32;
    let by = y.floor() as i32;

    let above0 = sun_visibility_mul(block_at(bx, by, bz));
    let above1 = sun_visibility_mul(block_at(bx, by + 1, bz));
    let above2 = sun_visibility_mul(block_at(bx, by + 2, bz));
    let above3 = if face == 2 {
        sun_visibility_mul(block_at(bx, by + 4, bz))
    } else {
        1.0
    };

    let mut visibility = above0 * above1 * above2 * above3;

    let ring_ids_near = [
        block_at(bx + 1, by, bz),
        block_at(bx - 1, by, bz),
        block_at(bx, by, bz + 1),
        block_at(bx, by, bz - 1),
    ];
    let ring_ids_far = [
        block_at(bx + 1, by + 1, bz),
        block_at(bx - 1, by + 1, bz),
        block_at(bx, by + 1, bz + 1),
        block_at(bx, by + 1, bz - 1),
    ];
    let mut roof_cover = 0.0f32;
    for id in ring_ids_near {
        roof_cover += if id < 0 {
            0.0
        } else if id == BLOCK_LEAVES as i8 {
            0.06
        } else {
            0.12
        };
    }
    for id in ring_ids_far {
        roof_cover += if id < 0 {
            0.0
        } else if id == BLOCK_LEAVES as i8 {
            0.04
        } else {
            0.08
        };
    }
    visibility *= (1.0 - roof_cover).clamp(0.02, 1.0);

    let side_y = y.floor() as i32;
    let side_ids = [
        block_at(bx + 1, side_y, bz),
        block_at(bx - 1, side_y, bz),
        block_at(bx, side_y, bz + 1),
        block_at(bx, side_y, bz - 1),
    ];
    let mut enclosure = 0.0f32;
    for id in side_ids {
        enclosure += if id < 0 {
            0.0
        } else if id == BLOCK_LEAVES as i8 {
            0.07
        } else {
            0.15
        };
    }
    visibility *= (1.0 - enclosure).clamp(0.02, 1.0);

    visibility.clamp(0.0, 1.0)
}

fn sun_visibility_mul(id: i8) -> f32 {
    if id < 0 {
        1.0
    } else if id == BLOCK_LEAVES as i8 {
        0.72
    } else {
        0.42
    }
}
