use glam::{IVec3, Vec3};
use std::time::Duration;

use crate::world::blocks::{
    block_is_collidable, block_light_emission, block_texture_by_id, parse_block_id,
    placeable_block_id_for_item,
};

pub const DAY_CYCLE_SECONDS: f32 = 48.0 * 60.0; // 2 min per in-game hour, 48 min full day.
pub const MAX_POINT_LIGHTS: usize = 20;
pub const EMISSIVE_SCAN_RADIUS_BLOCKS: i32 = 20;
pub const EMISSIVE_SCAN_MAX_SOURCES: usize = 96;
pub const EMISSIVE_SCAN_INTERVAL: Duration = Duration::from_millis(120);
pub const EMISSIVE_SCAN_MOVE_THRESHOLD_SQ: f32 = 1.25 * 1.25;

const POINT_LIGHT_CULL_DISTANCE: f32 = 128.0;
const POINT_LIGHT_DROPPED_SAMPLE_CAP: usize = 192;
const GLOWSTONE_RAY_MAX_STEPS: i32 = 20;
const GLOWSTONE_RAY_STEP_STRIDE: i32 = 2;
const GLOWSTONE_RAY_EMISSION_SCALE: f32 = 0.38;
const GLOWSTONE_RAY_MIN_TINT: f32 = 0.04;
const GLOWSTONE_RAY_SCORE_WEIGHT: f32 = 0.55;
const GLOWSTONE_GLASS_FACE_ESCAPE_WEIGHT: f32 = 0.28;
const GLOWSTONE_ENCLOSED_GLASS_ESCAPE_SCALE: f32 = 0.16;
const GLOWSTONE_RAY_DIRECTIONS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[derive(Clone, Copy)]
pub struct DayCycleState {
    pub day_progress: f32,
    pub daylight: f32,
    pub sky_mix: f32,
    pub sun_height: f32,
}

#[derive(Clone, Copy)]
struct PointLight {
    position: Vec3,
    radius: f32,
    color: Vec3,
    intensity: f32,
    // 0.0 = fully omnidirectional priority, 1.0 = strong view-direction priority.
    view_bias: f32,
}

#[derive(Clone, Copy)]
pub struct CulledPointLights {
    pub count: u32,
    pub pos_radius: [[f32; 4]; MAX_POINT_LIGHTS],
    pub color_intensity: [[f32; 4]; MAX_POINT_LIGHTS],
}

#[derive(Clone, Copy)]
pub struct WorldEmissiveLight {
    pub position: Vec3,
    pub block_id: i8,
    pub emission: f32,
    pub tint: Option<Vec3>,
}

pub fn sample_day_cycle(elapsed_seconds: f32) -> DayCycleState {
    let day_progress = (elapsed_seconds / DAY_CYCLE_SECONDS).rem_euclid(1.0);
    // Noon at t=0, midnight at 0.5 cycle.
    let cycle_angle = day_progress * std::f32::consts::TAU;
    let sun_height = cycle_angle.cos();
    let day01 = ((sun_height + 1.0) * 0.5).clamp(0.0, 1.0);
    // Keep a small ambient floor so nights stay readable.
    let daylight = (0.05 + 0.95 * day01.powf(1.22)).clamp(0.05, 1.0);
    let sky_mix = day01.powf(0.72);
    DayCycleState {
        day_progress,
        daylight,
        sky_mix,
        sun_height,
    }
}

pub fn sun_direction(day_progress: f32, sun_height: f32) -> Vec3 {
    let orbit_angle = day_progress * std::f32::consts::TAU;
    let dir =
        Vec3::new(orbit_angle.sin(), sun_height, orbit_angle.cos() * 0.55).normalize_or_zero();
    if dir.length_squared() > 1.0e-8 {
        dir
    } else {
        Vec3::Y
    }
}

fn point_light_tint(block_id: i8) -> Vec3 {
    let tint = block_texture_by_id(block_id)
        .map(|texture| {
            Vec3::new(
                texture.overlay[0].clamp(0.0, 1.0),
                texture.overlay[1].clamp(0.0, 1.0),
                texture.overlay[2].clamp(0.0, 1.0),
            )
        })
        .unwrap_or(Vec3::ONE);
    tint.max(Vec3::splat(0.22)).min(Vec3::splat(1.0))
}

fn item_light_emission(item_id: i8) -> Option<(i8, f32)> {
    let block_id = placeable_block_id_for_item(item_id)?;
    let emission = block_light_emission(block_id);
    (emission > 0.0).then_some((block_id, emission))
}

fn normalize_light_emission(emission: f32) -> f32 {
    (emission / 15.0).clamp(0.0, 1.0)
}

fn truncate_top_scored<T>(scored: &mut Vec<(f32, T)>, max_keep: usize) {
    if max_keep == 0 {
        scored.clear();
        return;
    }
    if scored.len() > max_keep {
        scored.select_nth_unstable_by(max_keep, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(max_keep);
    }
}

pub fn build_culled_point_lights<I>(
    camera_pos: Vec3,
    camera_forward: Vec3,
    held_light_origin: Vec3,
    held_light_forward: Vec3,
    day_cycle: DayCycleState,
    dropped_items: I,
    break_overlay: Option<(IVec3, u32)>,
    held_item_id: Option<i8>,
    world_emissive_lights: &[WorldEmissiveLight],
) -> CulledPointLights
where
    I: IntoIterator<Item = (Vec3, i8)>,
{
    let camera_forward = if camera_forward.length_squared() > 1.0e-8 {
        camera_forward.normalize()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    let held_forward = if held_light_forward.length_squared() > 1.0e-8 {
        held_light_forward.normalize()
    } else {
        camera_forward
    };
    let night_factor = (1.0 - day_cycle.daylight).clamp(0.0, 1.0);
    let mut candidates: Vec<PointLight> =
        Vec::with_capacity(POINT_LIGHT_DROPPED_SAMPLE_CAP + world_emissive_lights.len() + 4);

    if let Some((coord, _stage)) = break_overlay {
        candidates.push(PointLight {
            position: Vec3::new(
                coord.x as f32 + 0.5,
                coord.y as f32 + 0.5,
                coord.z as f32 + 0.5,
            ),
            radius: 7.5 + 3.0 * night_factor,
            color: Vec3::new(1.0, 0.8, 0.42),
            intensity: 0.35 + 0.9 * night_factor,
            view_bias: 0.55,
        });
    }

    for (position, block_id) in dropped_items
        .into_iter()
        .take(POINT_LIGHT_DROPPED_SAMPLE_CAP)
    {
        let emissive_factor = item_light_emission(block_id)
            .map(|(_, emission)| normalize_light_emission(emission))
            .unwrap_or(0.0);
        if night_factor <= 0.2 && emissive_factor <= 0.0 {
            continue;
        }
        candidates.push(PointLight {
            position: position + Vec3::new(0.0, 0.12, 0.0),
            radius: 3.2 + 3.2 * night_factor + 6.0 * emissive_factor,
            color: point_light_tint(block_id),
            intensity: 0.06 + 0.28 * night_factor + 0.85 * emissive_factor,
            view_bias: 0.70,
        });
    }

    if let Some((block_id, emission)) = held_item_id.and_then(item_light_emission) {
        let emissive_factor = normalize_light_emission(emission);
        candidates.push(PointLight {
            position: held_light_origin + held_forward * 0.42 + Vec3::new(0.0, -0.18, 0.0),
            radius: 3.0 + 11.0 * emissive_factor,
            color: point_light_tint(block_id),
            intensity: 0.25 + 1.75 * emissive_factor,
            view_bias: 0.78,
        });
    }

    for light in world_emissive_lights {
        let emissive_factor = normalize_light_emission(light.emission);
        if emissive_factor <= 0.01 {
            continue;
        }
        candidates.push(PointLight {
            position: light.position,
            radius: 2.5 + 24.0 * emissive_factor,
            color: light
                .tint
                .unwrap_or_else(|| point_light_tint(light.block_id)),
            intensity: 0.02 + 3.8 * emissive_factor,
            view_bias: 0.12,
        });
    }

    let mut scored: Vec<(f32, PointLight)> = Vec::with_capacity(candidates.len());
    for light in candidates {
        let to_light = light.position - camera_pos;
        let dist = to_light.length();
        if dist > POINT_LIGHT_CULL_DISTANCE + light.radius {
            continue;
        }
        let dir = if dist > 1.0e-5 {
            to_light / dist
        } else {
            camera_forward
        };
        let facing = ((camera_forward.dot(dir) + 0.35) * 0.74).clamp(0.0, 1.0);
        let attenuation = 1.0 / (1.0 + dist * 0.18 + dist * dist * 0.018);
        let view_weight = ((1.0 - light.view_bias) + light.view_bias * facing).clamp(0.05, 1.0);
        let score = light.intensity * view_weight * attenuation;
        if score <= 0.001 {
            continue;
        }
        scored.push((score, light));
    }
    truncate_top_scored(&mut scored, MAX_POINT_LIGHTS);

    let mut result = CulledPointLights {
        count: 0,
        pos_radius: [[0.0; 4]; MAX_POINT_LIGHTS],
        color_intensity: [[0.0; 4]; MAX_POINT_LIGHTS],
    };

    for (_, light) in scored.into_iter().take(MAX_POINT_LIGHTS) {
        let idx = result.count as usize;
        result.pos_radius[idx] = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.radius.max(0.05),
        ];
        result.color_intensity[idx] = [
            light.color.x,
            light.color.y,
            light.color.z,
            light.intensity.max(0.0),
        ];
        result.count += 1;
    }

    result
}

pub fn collect_world_emissive_lights<FEdited, FWorld>(
    center: Vec3,
    radius_blocks: i32,
    max_sources: usize,
    edited_sources: &[(IVec3, i8)],
    edited_block_at: FEdited,
    world_block_at: FWorld,
) -> Vec<WorldEmissiveLight>
where
    FEdited: Fn(i32, i32, i32) -> Option<i8>,
    FWorld: Fn(i32, i32, i32) -> i8,
{
    if edited_sources.is_empty() {
        return Vec::new();
    }

    let radius = radius_blocks.max(1);
    let r2 = (radius * radius) as f32;
    let mut scored: Vec<(f32, WorldEmissiveLight)> = Vec::new();
    let glowstone_id = parse_block_id("glowstone").or_else(|| parse_block_id("1:11"));
    let glass_tiles = parse_block_id("1:13")
        .and_then(block_texture_by_id)
        .map(|texture| texture.tiles);
    let block_id_at = |x: i32, y: i32, z: i32| -> i8 {
        edited_block_at(x, y, z).unwrap_or_else(|| world_block_at(x, y, z))
    };
    let glass_filter_tint = |id: i8| -> Option<Vec3> {
        if id < 0 || !block_is_collidable(id) {
            return None;
        }
        let Some(expected_tiles) = glass_tiles else {
            return None;
        };
        let texture = block_texture_by_id(id)?;
        if texture.tiles != expected_tiles || texture.transparent_mode.iter().any(|&mode| mode == 0)
        {
            return None;
        }
        Some(Vec3::new(
            texture.overlay[0].clamp(0.0, 1.0),
            texture.overlay[1].clamp(0.0, 1.0),
            texture.overlay[2].clamp(0.0, 1.0),
        ))
    };

    for &(pos, block_id) in edited_sources {
        if block_id < 0 {
            continue;
        }
        let x = pos.x;
        let y = pos.y;
        let z = pos.z;
        let dx = x as f32 + 0.5 - center.x;
        let dy = y as f32 + 0.5 - center.y;
        let dz = z as f32 + 0.5 - center.z;
        let dist2 = dx * dx + dy * dy + dz * dz;
        if dist2 > r2 {
            continue;
        }

        let emission = block_light_emission(block_id);
        if emission <= 0.0 {
            continue;
        }

        let mut source_emission = emission;
        if glowstone_id == Some(block_id) {
            let mut open_faces = 0_u32;
            let mut glass_faces = 0_u32;
            for (nx, ny, nz) in GLOWSTONE_RAY_DIRECTIONS {
                let neighbor_id = block_id_at(x + nx, y + ny, z + nz);
                if neighbor_id < 0 || !block_is_collidable(neighbor_id) {
                    open_faces += 1;
                    continue;
                }
                if glass_filter_tint(neighbor_id).is_some() {
                    glass_faces += 1;
                }
            }
            let faces_total = GLOWSTONE_RAY_DIRECTIONS.len() as f32;
            let open_escape = open_faces as f32 / faces_total;
            let glass_escape =
                (glass_faces as f32 / faces_total) * GLOWSTONE_GLASS_FACE_ESCAPE_WEIGHT;
            let mut escape = open_escape.max(glass_escape).clamp(0.0, 1.0);
            if open_faces == 0 {
                // Fully enclosed glowstone should not emit a central omnidirectional
                // point light through walls; only traced transmission rays may escape.
                source_emission = 0.0;
            } else {
                if glass_faces > 0 {
                    // Glass-adjacent openings keep a softer core source because
                    // directional rays are handling most of the transmitted light.
                    escape *= GLOWSTONE_ENCLOSED_GLASS_ESCAPE_SCALE;
                }
                source_emission *= escape;
            }
        }

        if source_emission > 0.03 {
            let score = (source_emission * source_emission) / (dist2 + 1.0);
            scored.push((
                score,
                WorldEmissiveLight {
                    position: Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
                    block_id,
                    emission: source_emission,
                    tint: None,
                },
            ));
        }

        if glowstone_id == Some(block_id) {
            for (step_x, step_y, step_z) in GLOWSTONE_RAY_DIRECTIONS {
                let mut ray_tint = Vec3::new(1.0, 1.0, 1.0);
                let mut filtered_by_glass = false;
                let mut step = 1;
                while step <= GLOWSTONE_RAY_MAX_STEPS {
                    let sx = x + step_x * step;
                    let sy = y + step_y * step;
                    let sz = z + step_z * step;
                    let sample_id = block_id_at(sx, sy, sz);
                    if sample_id >= 0 && block_is_collidable(sample_id) {
                        if let Some(glass_tint) = glass_filter_tint(sample_id) {
                            ray_tint *= glass_tint;
                            filtered_by_glass = true;
                            if ray_tint.max_element() <= GLOWSTONE_RAY_MIN_TINT {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if !filtered_by_glass || step % GLOWSTONE_RAY_STEP_STRIDE != 0 {
                        step += 1;
                        continue;
                    }

                    let ray_dx = sx as f32 + 0.5 - center.x;
                    let ray_dy = sy as f32 + 0.5 - center.y;
                    let ray_dz = sz as f32 + 0.5 - center.z;
                    let ray_dist2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
                    if ray_dist2 <= r2 {
                        let distance_falloff =
                            1.0 - ((step - 1) as f32 / GLOWSTONE_RAY_MAX_STEPS as f32);
                        let ray_base_emission = emission
                            * GLOWSTONE_RAY_EMISSION_SCALE
                            * distance_falloff.clamp(0.0, 1.0).powf(1.3);
                        let color_strength =
                            ((ray_tint.x + ray_tint.y + ray_tint.z) / 3.0).clamp(0.0, 1.0);
                        let ray_emission = ray_base_emission * (0.28 + 0.72 * color_strength);
                        if ray_emission > 0.05 {
                            let ray_score = ((ray_emission * ray_emission) / (ray_dist2 + 1.0))
                                * GLOWSTONE_RAY_SCORE_WEIGHT;
                            scored.push((
                                ray_score,
                                WorldEmissiveLight {
                                    position: Vec3::new(
                                        sx as f32 + 0.5,
                                        sy as f32 + 0.5,
                                        sz as f32 + 0.5,
                                    ),
                                    block_id,
                                    emission: ray_emission,
                                    tint: Some(ray_tint.clamp(Vec3::splat(0.0), Vec3::splat(1.0))),
                                },
                            ));
                        }
                    }
                    step += 1;
                }
            }
        }
    }

    truncate_top_scored(&mut scored, max_sources.max(1));
    scored.into_iter().map(|(_, light)| light).collect()
}
