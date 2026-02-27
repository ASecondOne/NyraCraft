use crate::world::blocks::{block_is_collidable, core_block_ids};

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
    let leaves_id = core_block_ids().leaves;
    let base = match face {
        2 => 1.00f32, // top
        3 => 0.36f32, // bottom
        _ => 0.62f32, // sides
    };
    if !use_sky_shading {
        // Fallback shading for low-detail passes where full sky probing is skipped.
        return (0.015f32 + base * 0.34f32).clamp(0.01f32, 0.42f32);
    }
    let (cx, cy, cz) = face_center_sample(face, wx, wy, wz, sx, sy, sz);
    let mut sky = sample_sky_visibility_fast(face, cx, cy, cz, leaves_id, block_at);
    if face == 3 {
        // Bottom faces receive less direct sky lighting.
        sky *= 0.66;
    }
    // Bias toward darker low-visibility results so caves stop looking washed out.
    let sky_curve = 0.18f32 * sky + 0.82f32 * sky * sky;
    let ambient = 0.003f32 + 0.018f32 * sky;
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

fn sample_sky_visibility_fast<F>(
    face: u32,
    x: f32,
    y: f32,
    z: f32,
    leaves_id: i8,
    block_at: &F,
) -> f32
where
    F: Fn(i32, i32, i32) -> i8,
{
    let bx = x.floor() as i32;
    let bz = z.floor() as i32;
    let by = y.floor() as i32;

    // Near-probe: catches immediate ceilings and overhangs around the current face.
    let mut visibility = 1.0f32;
    visibility *= sun_visibility_mul_near(block_at(bx, by, bz), leaves_id);
    visibility *= sun_visibility_mul_near(block_at(bx, by + 1, bz), leaves_id);
    visibility *= sun_visibility_mul_near(block_at(bx, by + 2, bz), leaves_id);
    if visibility <= 0.01 {
        return 0.0;
    }

    // Far-probe: adaptive vertical sampling to detect cave roofs deeper above the face.
    const SKY_FAR_PROBE_OFFSETS: [i32; 6] = [4, 7, 11, 16, 24, 36];
    for dy in SKY_FAR_PROBE_OFFSETS {
        visibility *= sun_visibility_mul_far(block_at(bx, by + dy, bz), leaves_id);
        if visibility <= 0.01 {
            return 0.0;
        }
    }
    if face == 2 {
        visibility *= sun_visibility_mul_far(block_at(bx, by + 52, bz), leaves_id);
    }

    let ring_ids_near = [
        block_at(bx + 1, by, bz),
        block_at(bx - 1, by, bz),
        block_at(bx, by, bz + 1),
        block_at(bx, by, bz - 1),
    ];
    let ring_ids_above = [
        block_at(bx + 1, by + 1, bz),
        block_at(bx - 1, by + 1, bz),
        block_at(bx, by + 1, bz + 1),
        block_at(bx, by + 1, bz - 1),
    ];

    // Local overhang penalty around the light shaft.
    let mut roof_cover = 0.0f32;
    for id in ring_ids_near {
        roof_cover += local_cover_weight(id, leaves_id, 0.05, 0.16);
    }
    for id in ring_ids_above {
        roof_cover += local_cover_weight(id, leaves_id, 0.03, 0.10);
    }
    visibility *= (1.0 - roof_cover).clamp(0.02, 1.0);

    // Tunnel enclosure penalty to suppress bright side walls deep in caves.
    let mut enclosure = 0.0f32;
    for id in ring_ids_near {
        enclosure += local_cover_weight(id, leaves_id, 0.04, 0.12);
    }
    let enclosure_scale = match face {
        2 => 0.35, // top
        3 => 0.95, // bottom
        _ => 0.75, // sides
    };
    if enclosure > 0.0 {
        visibility *= (1.0 - enclosure * enclosure_scale).clamp(0.03, 1.0);
    }

    visibility.clamp(0.0, 1.0)
}

#[inline]
fn local_cover_weight(id: i8, leaves_id: i8, leaves_weight: f32, solid_weight: f32) -> f32 {
    if id < 0 {
        0.0
    } else if !block_is_collidable(id) {
        0.0
    } else if id == leaves_id {
        leaves_weight
    } else {
        solid_weight
    }
}

#[inline]
fn sun_visibility_mul_near(id: i8, leaves_id: i8) -> f32 {
    if id < 0 {
        1.0
    } else if !block_is_collidable(id) {
        1.0
    } else if id == leaves_id {
        0.80
    } else {
        0.14
    }
}

#[inline]
fn sun_visibility_mul_far(id: i8, leaves_id: i8) -> f32 {
    if id < 0 {
        1.0
    } else if !block_is_collidable(id) {
        1.0
    } else if id == leaves_id {
        0.90
    } else {
        0.46
    }
}

#[cfg(test)]
mod tests {
    use super::compute_face_light;

    #[test]
    fn open_sky_is_brighter_than_cave_roof() {
        let open = |_: i32, _: i32, _: i32| -> i8 { -1 };
        let roofed = |_: i32, y: i32, _: i32| -> i8 { if y >= 8 { 0 } else { -1 } };
        let open_light = compute_face_light(2, 0, 0, 0, 1, 1, 1, true, &open);
        let roofed_light = compute_face_light(2, 0, 0, 0, 1, 1, 1, true, &roofed);
        assert!(open_light > roofed_light);
        assert!(roofed_light < 0.12);
    }

    #[test]
    fn near_ceiling_darkens_side_faces() {
        let near_roof = |_: i32, y: i32, _: i32| -> i8 { if y >= 2 { 0 } else { -1 } };
        let side_light = compute_face_light(0, 0, 0, 0, 1, 1, 1, true, &near_roof);
        assert!(side_light < 0.07);
    }
}
