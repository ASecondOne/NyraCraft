use glam::Vec3;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::CursorGrabMode,
    window::WindowBuilder,
};

mod colorthing;
mod physics;
mod player;
mod render;
mod world;

use glam::IVec3;
use player::{Camera, PlayerConfig, PlayerInput, PlayerState, update_player};
use render::Gpu;
use render::{CubeStyle, TextureAtlas};
use render::GpuStats;
use world::CHUNK_SIZE;
use world::blocks::{DEFAULT_TILES_X, default_blocks};
use world::mesher::{MeshData, MeshMode, generate_chunk_mesh};
use world::worldgen::WorldGen;

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
    let world_gen = WorldGen::new(seed);
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
    let mut last_stats_print = Instant::now();
    let start_time = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut fps_last = Instant::now();
    let mut fps_value: u32 = 0;
    let mut tps_ticks: u32 = 0;
    let mut tps_last = Instant::now();
    let mut tps_value: u32 = 0;

    let spawn_height = (world_gen.height_at(0, 0) + 4).max(4);
    let mut player = PlayerState {
        position: Vec3::new(0.0, spawn_height as f32, 0.0),
        velocity: Vec3::ZERO,
        grounded: false,
    };
    let player_config = PlayerConfig {
        height: 2.0,
        radius: 0.35,
        eye_height: 1.6,
        move_speed,
        jump_speed: 7.0,
        gravity: -18.0,
    };

    let mut camera = Camera {
        position: player.position + Vec3::new(0.0, player_config.eye_height, 0.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
    };
    let forward = camera.forward.normalize();
    let mut yaw = forward.z.atan2(forward.x);
    let mut pitch = forward.y.asin();
    let mut input = PlayerInput {
        move_forward: false,
        move_back: false,
        move_left: false,
        move_right: false,
        move_up: false,
        move_down: false,
        jump: false,
        fly_mode: false,
    };
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
    let mut f1_down = false;

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
    let base_draw_radius = 256;
    let lod_near_radius = 16;
    let lod_mid_radius = 64;
    let request_budget = 192;
    let max_inflight = 2048usize;
    let surface_depth_chunks = 8;
    let initial_burst_radius = 4;
    let mut current_chunk = chunk_coord_from_pos(player.position);
    let mut loaded: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut requested: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;
    let mut emergency_budget: i32 = 16;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            elwt.set_control_flow(ControlFlow::WaitUntil(next_tick));

            let now = Instant::now();
            if now >= next_tick {
                let dt = (now - last_update).as_secs_f32();
                let dt = dt.min(0.05);
                last_update = now;

                input.fly_mode = fly_mode;
                camera.position = update_player(
                    &mut player,
                    &mut input,
                    camera.forward,
                    camera.up,
                    dt,
                    &player_config,
                    &is_solid,
                );

                tick_accum = tick_accum.saturating_add(Duration::from_secs_f32(dt));
                while tick_accum >= tick_dt {
                    tick_accum -= tick_dt;
                    if pause_stream {
                        continue;
                    }
                    let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                    let render_radius =
                        (base_render_radius + height_chunks * 2).min(max_render_radius);
                    let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
                    stream_tick(
                        &mut gpu,
                        player.position,
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
                        &mut emergency_budget,
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
                    } else {
                        gpu.window().set_title("wgpu_try");
                    }
                    last_title_update = Instant::now();
                }

                if debug_ui && last_stats_print.elapsed() >= Duration::from_millis(500) {
                    let stats = gpu.stats();
                    print_stats(
                        &stats,
                        &loaded,
                        &requested,
                        tps_value,
                        fps_value,
                        current_chunk,
                        player.position,
                        start_time.elapsed(),
                        pause_stream,
                        pause_render,
                        base_render_radius,
                        base_draw_radius,
                    );
                    last_stats_print = Instant::now();
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
                if !pause_render {
                    gpu.window().request_redraw();
                }
                next_tick = now + delay;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            if !pause_render {
                let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
                let selection = pick_block(camera.position, camera.forward, &world_gen, 8.0);
                gpu.set_selection(selection);
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
            if key == KeyCode::F1 {
                f1_down = pressed;
                return;
            }
            if pressed && f1_down {
                match key {
                    KeyCode::KeyF => debug_faces = !debug_faces,
                    KeyCode::KeyW => debug_chunks = !debug_chunks,
                    KeyCode::KeyP => pause_stream = !pause_stream,
                    KeyCode::KeyV => pause_render = !pause_render,
                    KeyCode::KeyD => debug_ui = !debug_ui,
                    KeyCode::KeyG => debug_dump = true,
                    KeyCode::KeyR => force_reload = true,
                    KeyCode::KeyM => fly_mode = !fly_mode,
                    KeyCode::KeyX => {
                        let window = gpu.window();
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        } else {
                            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
                    }
                    _ => {}
                }
                return;
            }
            match key {
                KeyCode::KeyW => input.move_forward = pressed,
                KeyCode::KeyS => input.move_back = pressed,
                KeyCode::KeyA => input.move_left = pressed,
                KeyCode::KeyD => input.move_right = pressed,
                KeyCode::Space => {
                    if fly_mode {
                        input.move_up = pressed;
                    } else if pressed {
                        input.jump = true;
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                    if fly_mode {
                        input.move_down = pressed;
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

fn pick_block(
    camera_pos: Vec3,
    forward: Vec3,
    world_gen: &WorldGen,
    max_dist: f32,
) -> Option<IVec3> {
    let dir = forward.normalize();
    let step = 0.1f32;
    let mut t = 0.0f32;
    let mut last_block = IVec3::new(i32::MIN, i32::MIN, i32::MIN);
    while t <= max_dist {
        let pos = camera_pos + dir * t;
        let block = IVec3::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        if block != last_block {
            let height = world_gen.height_at(block.x, block.z);
            if block.y <= height {
                return Some(block);
            }
            last_block = block;
        }
        t += step;
    }
    None
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

fn print_stats(
    stats: &GpuStats,
    loaded: &HashMap<(i32, i32, i32), i32>,
    requested: &HashMap<(i32, i32, i32), i32>,
    tps: u32,
    fps: u32,
    current_chunk: IVec3,
    player_pos: Vec3,
    world_time: Duration,
    pause_stream: bool,
    pause_render: bool,
    base_render_radius: i32,
    base_draw_radius: i32,
) {
    let triangles = stats.total_indices / 3;
    let seconds = world_time.as_secs_f32();
    print!("\x1B[2J\x1B[H");
    println!("F3 STATS (refresh 0.5s)");
    println!(
        "time: {:>8.2}s | tps: {:>3} | fps: {:>3} | paused_stream: {} | paused_render: {}",
        seconds, tps, fps, pause_stream, pause_render
    );
    println!(
        "player: ({:>7.2}, {:>7.2}, {:>7.2}) | chunk: ({:>4}, {:>4}, {:>4})",
        player_pos.x, player_pos.y, player_pos.z, current_chunk.x, current_chunk.y, current_chunk.z
    );
    println!(
        "loaded: {:>6} | requested: {:>6} | base_render_radius: {:>4} | base_draw_radius: {:>4}",
        loaded.len(),
        requested.len(),
        base_render_radius,
        base_draw_radius
    );
    println!(
        "gpu: chunks={} super_chunks={} visible_supers={} dirty_supers={}",
        stats.chunks, stats.super_chunks, stats.visible_supers, stats.dirty_supers
    );
    println!(
        "gpu: pending_updates={} pending_queue={} total_triangles={} total_index_count={}",
        stats.pending_updates, stats.pending_queue, triangles, stats.total_indices
    );
    println!(
        "gpu: total_vertex_capacity={}",
        stats.total_vertices_capacity
    );
    let _ = io::stdout().flush();
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
    emergency_budget: &mut i32,
) {
    let mut height_cache: HashMap<(i32, i32), i32> = HashMap::new();
    let mut max_height_cached = |cx: i32, cz: i32| -> i32 {
        if let Some(h) = height_cache.get(&(cx, cz)) {
            *h
        } else {
            let h = max_height_in_chunk(world_gen, cx, cz);
            height_cache.insert((cx, cz), h);
            h
        }
    };

    let new_chunk = chunk_coord_from_pos(player_pos);
    if new_chunk != *current_chunk {
        *current_chunk = new_chunk;
        *ring_r = 0;
        *ring_i = 0;
        requested.clear();
        *emergency_budget = 16;
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

    let max_apply_per_tick = 8;
    for _ in 0..max_apply_per_tick {
        let Ok(mesh) = rx_res.try_recv() else { break; };
        let k = (mesh.coord.x, mesh.coord.y, mesh.coord.z);
        let MeshData {
            coord,
            step,
            mode,
            center,
            radius,
            vertices,
            indices,
        } = mesh;
        gpu.upsert_chunk(
            coord,
            center,
            radius,
            vertices,
            indices,
        );
        loaded.insert(k, pack_lod(mode, step));
        requested.remove(&k);
    }
    let max_rebuilds_per_tick = 2;
    gpu.rebuild_dirty_superchunks(player_pos, max_rebuilds_per_tick);

    if requested.len() >= max_inflight {
        return;
    }

    // Safe column: 2x2 chunks around player are always requested
    for dz in 0..=1 {
        for dx in 0..=1 {
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            let surface_y = max_height_cached(cx, cz);
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

    if *emergency_budget > 0 {
        let mut count = *emergency_budget;
        'emergency: for dz in -1..=1 {
            for dx in -1..=1 {
                if count <= 0 || requested.len() >= max_inflight {
                    break 'emergency;
                }
                let cx = current_chunk.x + dx;
                let cz = current_chunk.z + dz;
                let surface_y = max_height_cached(cx, cz);
                let surface_chunk_y = world_y_to_chunk_y(surface_y);
                let y_start = surface_chunk_y - surface_depth_chunks;
                let y_end = surface_chunk_y + 1;
                for dy in y_start..=y_end {
                    if count <= 0 || requested.len() >= max_inflight {
                        break 'emergency;
                    }
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
                        count -= 1;
                    }
                }
            }
        }
        *emergency_budget = count;
    }

    for dz in -initial_burst_radius..=initial_burst_radius {
        for dx in -initial_burst_radius..=initial_burst_radius {
            if requested.len() >= max_inflight {
                return;
            }
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            let surface_y = max_height_cached(cx, cz);
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
        let surface_y = max_height_cached(cx, cz);
        let surface_chunk_y = world_y_to_chunk_y(surface_y);
        let dist_xz = distance_2d(*current_chunk, (cx, surface_chunk_y, cz));
        let depth = depth_for_dist(surface_depth_chunks, dist_xz);
        let y_start = surface_chunk_y - depth;
        let y_end = surface_chunk_y + 1;

        for dy in y_start..=y_end {
            if budget <= 0 {
                break;
            }
            let coord = (cx, dy, cz);
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
