use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use glam::Vec3;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::CursorGrabMode,
    window::WindowBuilder,
};

mod camera;
mod colorthing;
mod render;
mod world;

use render::{CubeStyle, TextureAtlas};
use camera::Camera;
use render::Gpu;
use world::CHUNK_SIZE;
use glam::IVec3;
use world::worldgen::WorldGen;
use world::mesher::{generate_chunk_mesh, MeshData, MeshMode};
use world::blocks::{default_blocks, DEFAULT_TILES_X};

enum WorkerMsg {
    Request {
        coord: IVec3,
        step: i32,
        mode: MeshMode,
    },
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("CraftCraft")
        .build(&event_loop)
        .unwrap();

    let style = CubeStyle { use_texture: true };

    let atlas = TextureAtlas {
        path: "src/textures/atlas.png".to_string(),
        tile_size: 16,
    };

    let tiles_x = DEFAULT_TILES_X;
    let blocks = default_blocks(tiles_x);

    let seed: u32 = rand::random();
    let world_gen = WorldGen::new(seed)
        .with_height(0, 12.0, 0.04)
        .with_mountains(36.0, 0.015)
        .with_caves(0.08, 0.6)
        .with_cave_open(0.04, 0.7)
        .with_cave_depth(4, 64)
        .with_cave_open_depth(3);
    eprintln!(
        "World seed: {}, world_id: {}",
        world_gen.seed, world_gen.world_id
    );

    let mut gpu = Gpu::new(window, style, Some(atlas));

    let delay = Duration::from_millis(16);
    let move_speed = 4.0f32;
    let tick_dt = Duration::from_millis(50);
    let mut tick_accum = Duration::ZERO;
    let mut next_tick = Instant::now() + delay;
    let mut last_update = Instant::now();
    let mut last_title_update = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut fps_last = Instant::now();
    let mut fps_value: u32 = 0;
    let mut tps_ticks: u32 = 0;
    let mut tps_last = Instant::now();
    let mut tps_value: u32 = 0;

    let spawn_height = world_gen.height_at(0, 0) + 2;
    let mut player_pos = Vec3::new(0.0, spawn_height as f32, 0.0);
    let mut velocity = Vec3::ZERO;
    let mut grounded = false;
    let player_height = 2.0f32;
    let player_radius = 0.35f32;
    let eye_height = 1.6f32;

    let mut camera = Camera {
        position: player_pos + Vec3::new(0.0, eye_height, 0.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
    };
    let forward = camera.forward.normalize();
    let mut yaw = forward.z.atan2(forward.x);
    let mut pitch = forward.y.asin();
    let mut move_forward = false;
    let mut move_back = false;
    let mut move_left = false;
    let mut move_right = false;
    let mut jump = false;
    let mut move_up = false;
    let mut move_down = false;
    let mouse_sensitivity = 0.0015f32;
    let mut mouse_enabled = false;
    let mut debug_faces = false;
    let mut debug_chunks = false;
    let mut pause_stream = false;
    let mut pause_render = false;
    let mut debug_ui = false;
    let mut debug_dump = false;
    let mut fly_mode = false;
    let mut force_reload = false;

    let is_solid = {
        let worldgen = world_gen.clone();
        move |x: i32, y: i32, z: i32| -> bool { y <= worldgen.height_at(x, z) }
    };

    let (tx_req, rx_req) = mpsc::channel::<WorkerMsg>();
    let (tx_res, rx_res) = mpsc::channel::<MeshData>();
    let blocks_gen = blocks.clone();
    let gen_thread = world_gen.clone();
    let rx_req = Arc::new(Mutex::new(rx_req));
    let worker_count = 8;
    for _ in 0..worker_count {
        let rx_req = Arc::clone(&rx_req);
        let tx_res = tx_res.clone();
        let blocks_gen = blocks_gen.clone();
        let gen_thread = gen_thread.clone();
        thread::spawn(move || {
            loop {
                let msg = {
                    let lock = rx_req.lock().unwrap();
                    lock.recv()
                };
                let msg = match msg {
                    Ok(msg) => msg,
                    Err(_) => break,
                };
                match msg {
                    WorkerMsg::Request { coord, step, mode } => {
                        let mesh = generate_chunk_mesh(coord, &blocks_gen, &gen_thread, step, mode);
                        let _ = tx_res.send(mesh);
                    }
                }
            }
        });
    }

    let base_render_radius = 256;
    let max_render_radius = 256;
    let base_draw_radius = 128;
    let lod_near_radius = 16;
    let lod_mid_radius = 64;
    let request_budget = 96;
    let max_inflight = 2048usize;
    let surface_depth_chunks = 8;
    let initial_burst_radius = 4;
    let mut current_chunk = chunk_coord_from_pos(player_pos);
    let mut loaded: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut requested: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            elwt.set_control_flow(ControlFlow::WaitUntil(next_tick));

            let now = Instant::now();
            if now >= next_tick {
                let dt = (now - last_update).as_secs_f32();
                let dt = dt.min(0.05);
                last_update = now;

                let forward = camera.forward.normalize();
                let forward_flat = Vec3::new(forward.x, 0.0, forward.z);
                let forward_flat = if forward_flat.length_squared() > 0.0 {
                    forward_flat.normalize()
                } else {
                    Vec3::new(0.0, 0.0, -1.0)
                };
                let right = forward_flat.cross(camera.up).normalize();

                let speed = move_speed;
                let mut move_dir = Vec3::ZERO;
                if move_forward {
                    move_dir += forward_flat;
                }
                if move_back {
                    move_dir -= forward_flat;
                }
                if move_right {
                    move_dir += right;
                }
                if move_left {
                    move_dir -= right;
                }
                if fly_mode {
                    if move_up {
                        move_dir += camera.up;
                    }
                    if move_down {
                        move_dir -= camera.up;
                    }
                }

                let move_dir = if move_dir.length_squared() > 0.0 {
                    move_dir.normalize()
                } else {
                    Vec3::ZERO
                };

                velocity.x = move_dir.x * speed;
                velocity.z = move_dir.z * speed;

                if fly_mode {
                    velocity.y = move_dir.y * speed;
                    grounded = false;
                    jump = false;
                } else {
                    if jump && grounded {
                        velocity.y = 7.0;
                        grounded = false;
                    }
                    jump = false;
                    velocity.y += -18.0 * dt;
                }

                let mut new_pos = player_pos;
                // X axis
                new_pos.x += velocity.x * dt;
                if collides(new_pos, player_height, player_radius, &is_solid) {
                    if velocity.x > 0.0 {
                        let max_x = new_pos.x + player_radius;
                        let block_x = max_x.floor() as i32;
                        new_pos.x = block_x as f32 - player_radius - 0.001;
                    } else if velocity.x < 0.0 {
                        let min_x = new_pos.x - player_radius;
                        let block_x = min_x.floor() as i32;
                        new_pos.x = block_x as f32 + 1.0 + player_radius + 0.001;
                    }
                    velocity.x = 0.0;
                }
                // Z axis
                new_pos.z += velocity.z * dt;
                if collides(new_pos, player_height, player_radius, &is_solid) {
                    if velocity.z > 0.0 {
                        let max_z = new_pos.z + player_radius;
                        let block_z = max_z.floor() as i32;
                        new_pos.z = block_z as f32 - player_radius - 0.001;
                    } else if velocity.z < 0.0 {
                        let min_z = new_pos.z - player_radius;
                        let block_z = min_z.floor() as i32;
                        new_pos.z = block_z as f32 + 1.0 + player_radius + 0.001;
                    }
                    velocity.z = 0.0;
                }
                // Y axis
                new_pos.y += velocity.y * dt;
                if collides(new_pos, player_height, player_radius, &is_solid) {
                    if velocity.y < 0.0 {
                        let min_y = new_pos.y;
                        let block_y = min_y.floor() as i32;
                        new_pos.y = block_y as f32 + 1.0 + 0.001;
                        grounded = true;
                    } else if velocity.y > 0.0 {
                        let max_y = new_pos.y + player_height;
                        let block_y = max_y.floor() as i32;
                        new_pos.y = block_y as f32 - player_height - 0.001;
                    }
                    velocity.y = 0.0;
                } else {
                    grounded = false;
                }

                player_pos = new_pos;
                camera.position = player_pos + Vec3::new(0.0, eye_height, 0.0);

                tick_accum = tick_accum.saturating_add(Duration::from_secs_f32(dt));
                while tick_accum >= tick_dt {
                    tick_accum -= tick_dt;
                    if pause_stream {
                        continue;
                    }
                    let height_chunks = (player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32;
                    let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                    let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
                    stream_tick(
                        &mut gpu,
                        player_pos,
                        &mut current_chunk,
                        &mut ring_r,
                        &mut ring_i,
                        &mut loaded,
                        &mut requested,
                        &tx_req,
                        &rx_res,
                        base_render_radius,
                        max_render_radius,
                        request_budget,
                        initial_burst_radius,
                        max_inflight,
                        &mut force_reload,
                        &world_gen,
                        surface_depth_chunks,
                        draw_radius,
                        lod_near_radius,
                        lod_mid_radius,
                    );
                    gpu.update_visible(&camera, draw_radius);
                    tps_ticks += 1;
                }

                fps_frames += 1;
                if fps_last.elapsed() >= Duration::from_secs(1) {
                    fps_value = fps_frames;
                    fps_frames = 0;
                    fps_last = Instant::now();
                }
                if tps_last.elapsed() >= Duration::from_secs(1) {
                    tps_value = tps_ticks;
                    tps_ticks = 0;
                    tps_last = Instant::now();
                }

                if last_title_update.elapsed() >= Duration::from_millis(250) {
                    let height_chunks = (player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32;
                    let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                    let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
                    if debug_ui {
                        let title = format!(
                            "loaded:{} requested:{} tps:{} fps:{} paused_stream:{} paused_render:{}",
                            loaded.len(),
                            requested.len(),
                            tps_value,
                            fps_value,
                            pause_stream,
                            pause_render
                        );
                        gpu.window().set_title(&title);
                        eprintln!(
                            "loaded={} requested={} tps={} fps={} chunk=({}, {}, {}) radius={} paused_stream={} paused_render={}",
                            loaded.len(),
                            requested.len(),
                            tps_value,
                            fps_value,
                            current_chunk.x,
                            current_chunk.y,
                            current_chunk.z,
                            draw_radius,
                            pause_stream,
                            pause_render
                        );
                    } else {
                        gpu.window().set_title("wgpu_try");
                    }
                    last_title_update = Instant::now();
                }

                if debug_dump {
                    dump_surface_bands(
                        &world_gen,
                        current_chunk,
                        &loaded,
                        &requested,
                        surface_depth_chunks,
                    );
                    debug_dump = false;
                }
                gpu.window().request_redraw();
                next_tick = now + delay;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            if !pause_render {
                let height_chunks = (player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32;
                let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
                gpu.render(&camera, debug_faces, debug_chunks, draw_radius);
            }
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state,
                            ..
                        },
                    ..
                },
            ..
        } => {
            let pressed = matches!(state, ElementState::Pressed);
            match key {
                KeyCode::KeyW => move_forward = pressed,
                KeyCode::KeyS => move_back = pressed,
                KeyCode::KeyA => move_left = pressed,
                KeyCode::KeyD => move_right = pressed,
                KeyCode::Space => {
                    if fly_mode {
                        move_up = pressed;
                    } else if pressed {
                        jump = true;
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                    if fly_mode {
                        move_down = pressed;
                    }
                }
                KeyCode::F1 if pressed => debug_faces = !debug_faces,
                KeyCode::F2 if pressed => {
                    pause_stream = !pause_stream;
                }
                KeyCode::F3 if pressed => fly_mode = !fly_mode,
                KeyCode::F5 if pressed => {
                    force_reload = true;
                }
                KeyCode::F9 if pressed => {
                    pause_render = !pause_render;
                }
                KeyCode::F6 if pressed => {
                    debug_ui = !debug_ui;
                }
                KeyCode::F7 if pressed => {
                    debug_chunks = !debug_chunks;
                }
                KeyCode::F8 if pressed => {
                    debug_dump = true;
                }
                KeyCode::F11 if pressed => {
                    let window = gpu.window();
                    if window.fullscreen().is_some() {
                        window.set_fullscreen(None);
                    } else {
                        window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                    }
                }
                _ => {}
            }
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            if !mouse_enabled {
                return;
            }
            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
            let dx = dx.clamp(-100.0, 100.0);
            let dy = dy.clamp(-100.0, 100.0);
            yaw += dx * mouse_sensitivity;
            pitch -= dy * mouse_sensitivity;

            let max_pitch = 1.5533f32;
            if pitch > max_pitch {
                pitch = max_pitch;
            } else if pitch < -max_pitch {
                pitch = -max_pitch;
            }

            let (sy, cy) = yaw.sin_cos();
            let (sp, cp) = pitch.sin_cos();
            camera.forward = Vec3::new(cy * cp, sp, sy * cp).normalize();
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            gpu.resize(size);
        }
        Event::WindowEvent {
            event: WindowEvent::Focused(true),
            ..
        } => {
            if gpu.window().set_cursor_grab(CursorGrabMode::Locked).is_ok()
                || gpu.window().set_cursor_grab(CursorGrabMode::Confined).is_ok()
            {
                gpu.window().set_cursor_visible(false);
                mouse_enabled = true;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::Focused(false),
            ..
        } => {
            let _ = gpu.window().set_cursor_grab(CursorGrabMode::None);
            gpu.window().set_cursor_visible(true);
            mouse_enabled = false;
        }
        Event::WindowEvent {
            event: WindowEvent::ScaleFactorChanged { .. },
            ..
        } => {
            let size = gpu.window().inner_size();
            gpu.resize(size);
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            elwt.exit();
        }
        _ => {}
    });
}

fn chunk_coord_from_pos(pos: Vec3) -> IVec3 {
    let half = (CHUNK_SIZE / 2) as f32;
    IVec3::new(
        ((pos.x + half) / CHUNK_SIZE as f32).floor() as i32,
        ((pos.y + half) / CHUNK_SIZE as f32).floor() as i32,
        ((pos.z + half) / CHUNK_SIZE as f32).floor() as i32,
    )
}

fn ring_length(r: i32) -> i32 {
    if r == 0 { 1 } else { 8 * r }
}

fn ring_coord(r: i32, i: i32) -> (i32, i32) {
    if r == 0 {
        return (0, 0);
    }
    let side = 2 * r;
    let edge = i / side;
    let offset = i % side;
    match edge {
        0 => (-r + offset, -r),
        1 => (r, -r + offset),
        2 => (r - offset, r),
        _ => (-r, r - offset),
    }
}

fn distance_2d(center: IVec3, coord: (i32, i32, i32)) -> i32 {
    let dx = (coord.0 - center.x).abs();
    let dz = (coord.2 - center.z).abs();
    dx.max(dz)
}

fn max_height_in_chunk(world_gen: &WorldGen, chunk_x: i32, chunk_z: i32) -> i32 {
    let half = CHUNK_SIZE / 2;
    let base_x = chunk_x * CHUNK_SIZE - half;
    let base_z = chunk_z * CHUNK_SIZE - half;
    let mut max_h = i32::MIN;
    let mut z = 0;
    while z < CHUNK_SIZE {
        let mut x = 0;
        while x < CHUNK_SIZE {
            let h = world_gen.height_at(base_x + x, base_z + z);
            if h > max_h {
                max_h = h;
            }
            x += 1;
        }
        z += 1;
    }
    max_h
}

fn world_y_to_chunk_y(y: i32) -> i32 {
    let half = CHUNK_SIZE / 2;
    ((y + half) as f32 / CHUNK_SIZE as f32).floor() as i32
}

fn lod_div(dist: i32, base: i32) -> i32 {
    if dist <= base {
        3
    } else if dist <= base * 2 {
        6
    } else if dist <= base * 3 {
        9
    } else {
        18
    }
}

fn pack_lod(mode: MeshMode, step: i32) -> i32 {
    ((mode as i32) << 16) | (step & 0xFFFF)
}

fn depth_for_dist(base_depth: i32, dist: i32) -> i32 {
    let mut depth = base_depth;
    let tiers = dist / 16;
    depth -= tiers * 3;
    depth.clamp(2, base_depth)
}

fn dump_surface_bands(
    world_gen: &WorldGen,
    current_chunk: IVec3,
    loaded: &HashMap<(i32, i32, i32), i32>,
    requested: &HashMap<(i32, i32, i32), i32>,
    surface_depth_chunks: i32,
) {
    eprintln!("--- surface band dump ---");
    eprintln!(
        "loaded_total={} requested_total={}",
        loaded.len(),
        requested.len()
    );
    for dz in -2..=2 {
        for dx in -2..=2 {
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            let surface_y = max_height_in_chunk(world_gen, cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let dist_xz = distance_2d(current_chunk, (cx, surface_chunk_y, cz));
            let depth = depth_for_dist(surface_depth_chunks, dist_xz);
            let y_start = surface_chunk_y - depth;
            let y_end = surface_chunk_y + 1;
            let mut missing = Vec::new();
            let mut loaded_count = 0;
            let mut requested_count = 0;
            for cy in y_start..=y_end {
                let key = (cx, cy, cz);
                if loaded.contains_key(&key) {
                    loaded_count += 1;
                } else if requested.contains_key(&key) {
                    requested_count += 1;
                } else {
                    missing.push(cy);
                }
            }
            eprintln!(
                "xz=({}, {}) surface_y={} band=[{}..{}] loaded={} requested={} missing_count={}",
                cx,
                cz,
                surface_y,
                y_start,
                y_end,
                loaded_count,
                requested_count,
                missing.len()
            );
        }
    }
    eprintln!("--- end dump ---");
}

fn stream_tick(
    gpu: &mut Gpu,
    player_pos: Vec3,
    current_chunk: &mut IVec3,
    ring_r: &mut i32,
    ring_i: &mut i32,
    loaded: &mut HashMap<(i32, i32, i32), i32>,
    requested: &mut HashMap<(i32, i32, i32), i32>,
    tx_req: &mpsc::Sender<WorkerMsg>,
    rx_res: &mpsc::Receiver<MeshData>,
    base_render_radius: i32,
    max_render_radius: i32,
    request_budget: i32,
    initial_burst_radius: i32,
    max_inflight: usize,
    force_reload: &mut bool,
    world_gen: &WorldGen,
    surface_depth_chunks: i32,
    draw_radius: i32,
    lod_near_radius: i32,
    lod_mid_radius: i32,
) {
    let new_chunk = chunk_coord_from_pos(player_pos);
    if new_chunk != *current_chunk {
        *current_chunk = new_chunk;
        *ring_r = 0;
        *ring_i = 0;
        requested.clear();
    }

    let dynamic_radius = ((player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32) * 2;
    let render_radius = (base_render_radius + dynamic_radius).min(max_render_radius);
    let height_chunks = (player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32;
    let height_factor = height_chunks / 8;
    let budget = (request_budget - height_factor * 8).max(16);
    if *force_reload {
        gpu.clear_chunks();
        loaded.clear();
        requested.clear();
        *ring_r = 0;
        *ring_i = 0;
        *force_reload = false;
    }

    while let Ok(mesh) = rx_res.try_recv() {
        let k = (mesh.coord.x, mesh.coord.y, mesh.coord.z);
        gpu.upsert_chunk(mesh.coord, mesh.center, mesh.radius, &mesh.vertices, &mesh.indices);
        loaded.insert(k, pack_lod(mesh.mode, mesh.step));
        requested.remove(&k);
    }

    if requested.len() >= max_inflight {
        return;
    }

    // Safe column: 2x2 chunks around player are always requested
    for dz in 0..=1 {
        for dx in 0..=1 {
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            let surface_y = max_height_in_chunk(world_gen, cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let y_start = surface_chunk_y - surface_depth_chunks;
            let y_end = surface_chunk_y + 1;
            for dy in y_start..=y_end {
                let coord = (cx, dy, cz);
                let step = 1;
                let mode = MeshMode::Full;
                let lod = pack_lod(mode, step);
                let key = coord;
                let mut needs_request = true;
                if let Some(&req_step) = requested.get(&key) {
                    needs_request = req_step != lod;
                } else if let Some(&loaded_step) = loaded.get(&key) {
                    needs_request = loaded_step != lod;
                }
                if needs_request {
                    let _ = tx_req.send(WorkerMsg::Request {
                        coord: IVec3::new(coord.0, coord.1, coord.2),
                        step,
                        mode,
                    });
                    requested.insert(key, lod);
                }
            }
        }
    }

    for dz in -initial_burst_radius..=initial_burst_radius {
        for dx in -initial_burst_radius..=initial_burst_radius {
            if requested.len() >= max_inflight {
                return;
            }
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            let surface_y = max_height_in_chunk(world_gen, cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let y_start = surface_chunk_y - surface_depth_chunks;
            let y_end = surface_chunk_y + 1;

            for dy in y_start..=y_end {
                if requested.len() >= max_inflight {
                    return;
                }
                let coord = (cx, dy, cz);
                let dist = distance_2d(*current_chunk, coord);
                if dist > render_radius {
                    continue;
                }
                let step = 1;
                let mode = MeshMode::Full;
                let lod = pack_lod(mode, step);
                let key = coord;
                let mut needs_request = true;
                if let Some(&req_step) = requested.get(&key) {
                    needs_request = req_step != lod;
                } else if let Some(&loaded_step) = loaded.get(&key) {
                    needs_request = loaded_step != lod;
                }
                if needs_request {
                    let _ = tx_req.send(WorkerMsg::Request {
                        coord: IVec3::new(coord.0, coord.1, coord.2),
                        step,
                        mode,
                    });
                    requested.insert(key, lod);
                }
            }
        }
    }

    let mut budget = budget;
    while budget > 0 && *ring_r <= render_radius {
        if requested.len() >= max_inflight {
            return;
        }
        let (dx, dz) = ring_coord(*ring_r, *ring_i);
        *ring_i += 1;
        let ring_len = ring_length(*ring_r);
        if *ring_i >= ring_len {
            *ring_i = 0;
            *ring_r += 1;
        }
        let cx = current_chunk.x + dx;
        let cz = current_chunk.z + dz;
        let surface_y = max_height_in_chunk(world_gen, cx, cz);
        let surface_chunk_y = world_y_to_chunk_y(surface_y);
        let dist_xz = distance_2d(*current_chunk, (cx, surface_chunk_y, cz));
        let depth = depth_for_dist(surface_depth_chunks, dist_xz);
        let y_start = surface_chunk_y - depth;
        let y_end = surface_chunk_y + 1;

        for dy in y_start..=y_end {
            if budget <= 0 {
                break;
            }
            let coord = (
                cx,
                dy,
                cz,
            );
            let dist = distance_2d(*current_chunk, coord);
            if dist > render_radius {
                continue;
            }
            let mode = if dist <= lod_near_radius {
                MeshMode::Full
            } else if dist <= lod_mid_radius {
                MeshMode::SurfaceSides
            } else {
                MeshMode::SurfaceOnly
            };
            let step = if mode == MeshMode::Full {
                1
            } else if mode == MeshMode::SurfaceSides {
                3
            } else {
                lod_div(dist, lod_mid_radius)
            };
            let lod = pack_lod(mode, step);
            let key = coord;
            let mut needs_request = true;
            if let Some(&req_step) = requested.get(&key) {
                needs_request = req_step != lod;
            } else if let Some(&loaded_step) = loaded.get(&key) {
                needs_request = loaded_step != lod;
            }
            if needs_request {
                let _ = tx_req.send(WorkerMsg::Request {
                    coord: IVec3::new(coord.0, coord.1, coord.2),
                    step,
                    mode,
                });
                requested.insert(key, lod);
                budget -= 1;
                if requested.len() >= max_inflight {
                    return;
                }
            }
        }

        // surface chunk already included in y_start..=y_end
    }

    let to_remove: Vec<(i32, i32, i32)> = loaded
        .keys()
        .filter(|(x, y, z)| {
            let dx = x - current_chunk.x;
            let _dy = y - current_chunk.y;
            let dz = z - current_chunk.z;
            dx.abs() > render_radius
                || dz.abs() > render_radius
                || dx.abs() > draw_radius + 8
                || dz.abs() > draw_radius + 8
        })
        .cloned()
        .collect();
    for coord in to_remove {
        gpu.remove_chunk(IVec3::new(coord.0, coord.1, coord.2));
        loaded.remove(&coord);
        requested.remove(&coord);
    }

    // no super chunks
}

fn collides<F>(pos: Vec3, height: f32, radius: f32, is_solid: &F) -> bool
where
    F: Fn(i32, i32, i32) -> bool,
{
    let min = Vec3::new(pos.x - radius, pos.y, pos.z - radius);
    let max = Vec3::new(pos.x + radius, pos.y + height, pos.z + radius);

    let min_x = min.x.floor() as i32;
    let max_x = (max.x - 0.001).floor() as i32;
    let min_y = min.y.floor() as i32;
    let max_y = (max.y - 0.001).floor() as i32;
    let min_z = min.z.floor() as i32;
    let max_z = (max.z - 0.001).floor() as i32;

    for z in min_z..=max_z {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if is_solid(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}
