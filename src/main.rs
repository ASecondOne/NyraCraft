use glam::{IVec3, Vec3};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::CursorGrabMode,
    window::WindowBuilder,
};

mod app;
mod physics;
mod player;
mod render;
mod world;

use app::streaming::{
    CacheMeshView, CacheWriteMsg, EditedChunkRanges, PackedMeshData, SharedRequestQueue,
    WorkerResult, apply_stream_results, block_id_full_cached, chunk_coord_from_pos,
    estimate_mesh_memory_mb, is_solid_cached, mesh_cache_dir, new_request_queue, pick_block,
    pop_request, print_stats, should_pack_far_lod, stream_tick, try_load_cached_mesh,
    write_cached_mesh,
};
use player::{
    Camera, EditedBlocks, PlayerConfig, PlayerInput, PlayerState, block_id_with_edits,
    handle_block_mouse_input, new_edited_blocks, update_player,
};
use render::Gpu;
use render::mesh::pack_far_vertices;
use render::{CubeStyle, TextureAtlas};
use world::CHUNK_SIZE;
use world::blocks::{DEFAULT_TILES_X, build_block_index, default_blocks};
use world::mesher::generate_chunk_mesh;
use world::worldgen::{TreeSpec, WorldGen};

type CoordKey = (i32, i32, i32);
type DirtyChunks = Arc<Mutex<HashMap<CoordKey, u64>>>;
const DIRTY_EDIT_CHUNK_HALO: i32 = 1;

fn newest_atlas_input_mtime(script: &str, texture_dir: &str) -> Option<SystemTime> {
    let mut newest = fs::metadata(script).ok()?.modified().ok()?;
    let entries = fs::read_dir(texture_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !ext.eq_ignore_ascii_case("png") && !ext.eq_ignore_ascii_case("tga") {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if modified > newest {
            newest = modified;
        }
    }
    Some(newest)
}

fn atlas_is_up_to_date(script: &str, texture_dir: &str, atlas_path: &str) -> bool {
    let Ok(atlas_mtime) = fs::metadata(atlas_path).and_then(|m| m.modified()) else {
        return false;
    };
    let Some(input_mtime) = newest_atlas_input_mtime(script, texture_dir) else {
        return false;
    };
    atlas_mtime >= input_mtime
}

fn generate_texture_atlas() {
    let script = "src/texturing/atlas_gen.py";
    let texture_dir = "src/texturing/textures";
    let atlas_path = "src/texturing/atlas_output/atlas.png";

    if atlas_is_up_to_date(script, texture_dir, atlas_path) {
        return;
    }

    let try_run = |exe: &str| -> Result<(), String> {
        let status = Command::new(exe)
            .arg(script)
            .status()
            .map_err(|e| format!("failed to run {exe}: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{exe} exited with status {status}"))
        }
    };

    if try_run("python3").is_err()
        && let Err(e) = try_run("python")
    {
        panic!("atlas generation failed: {e}");
    }

    if !Path::new(atlas_path).exists() {
        panic!("atlas generation completed but atlas was not found at {atlas_path}");
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("NyraCraft")
        .build(&event_loop)
        .unwrap();

    generate_texture_atlas();

    let style = CubeStyle { use_texture: true };

    let atlas = TextureAtlas {
        path: "src/texturing/atlas_output/atlas.png".to_string(),
        tile_size: 16,
    };

    let tiles_x = DEFAULT_TILES_X;
    let blocks = default_blocks(tiles_x);
    let block_index = build_block_index(tiles_x);
    if cfg!(debug_assertions) {
        eprintln!("Indexed {} block types:", block_index.len());
        for entry in &block_index {
            eprintln!(
                "  item_id={} block_id={} name={} texture_index={}",
                entry.item_id, entry.block_id, entry.name, entry.texture_index
            );
        }
    }

    let seed: u32 = rand::random();
    let world_gen = WorldGen::new(seed);
    eprintln!(
        "World seed: {}, world_id: {}",
        world_gen.seed, world_gen.world_id
    );

    let edited_blocks: EditedBlocks = new_edited_blocks();
    let dirty_chunks: DirtyChunks = Arc::new(Mutex::new(HashMap::new()));
    let edited_chunk_ranges: EditedChunkRanges = Arc::new(Mutex::new(HashMap::new()));

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
        height: 1.9,
        radius: 0.35,
        eye_height: 1.62,
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
    let mut last_cursor_pos: Option<(f64, f64)> = None;
    let mut debug_faces = false;
    let mut debug_chunks = false;
    let mut pause_stream = false;
    let mut pause_render = false;
    let mut debug_ui = false;
    let mut fly_mode = false;
    let mut force_reload = false;
    let mut f1_down = false;
    let mut f1_combo_used = false;

    let is_solid = {
        let worldgen = world_gen.clone();
        let edited_blocks = Arc::clone(&edited_blocks);
        let height_cache = RefCell::new(HashMap::<(i32, i32), i32>::new());
        let tree_cache = RefCell::new(HashMap::<(i32, i32), Option<TreeSpec>>::new());
        move |x: i32, y: i32, z: i32| -> bool {
            if !worldgen.in_world_bounds(x, z) {
                return true;
            }
            if let Some(id) = edited_blocks.read().unwrap().get(x, y, z) {
                return id >= 0;
            }
            is_solid_cached(&worldgen, &height_cache, &tree_cache, x, y, z)
        }
    };

    let (tx_res, rx_res) = mpsc::channel::<WorkerResult>();
    let (tx_cache, rx_cache) = mpsc::channel::<CacheWriteMsg>();
    let request_queue: SharedRequestQueue = new_request_queue();
    let cache_dir = mesh_cache_dir(world_gen.world_id);
    let _ = fs::create_dir_all(&cache_dir);
    let cache_dir_writer = cache_dir.clone();
    thread::spawn(move || {
        let _ = fs::create_dir_all(&cache_dir_writer);
        while let Ok(msg) = rx_cache.recv() {
            match msg {
                CacheWriteMsg::Write {
                    coord,
                    step,
                    mode,
                    center,
                    radius,
                    vertices,
                    indices,
                } => {
                    write_cached_mesh(
                        &cache_dir_writer,
                        coord,
                        step,
                        mode,
                        CacheMeshView {
                            center,
                            radius,
                            vertices: &vertices,
                            indices: &indices,
                        },
                    );
                }
            }
        }
    });
    let blocks_gen = blocks.clone();
    let gen_thread = world_gen.clone();
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).clamp(1, 12))
        .unwrap_or(1);
    for _ in 0..worker_count {
        let request_queue = Arc::clone(&request_queue);
        let tx_res = tx_res.clone();
        let tx_cache = tx_cache.clone();
        let blocks_gen = blocks_gen.clone();
        let gen_thread = gen_thread.clone();
        let cache_dir = cache_dir.clone();
        let edited_blocks = Arc::clone(&edited_blocks);
        let edited_chunk_ranges = Arc::clone(&edited_chunk_ranges);
        let dirty_chunks = Arc::clone(&dirty_chunks);
        thread::spawn(move || {
            while let Some(task) = pop_request(&request_queue) {
                let coord = task.coord;
                let step = task.step;
                let mode = task.mode;
                let coord_key = (coord.x, coord.y, coord.z);
                let dirty_rev = dirty_chunks.lock().unwrap().get(&coord_key).copied();
                let (overrides, override_range) = {
                    let store = edited_blocks.read().unwrap();
                    if store.has_any_edits() {
                        (
                            store.collect_chunk_halo(coord, DIRTY_EDIT_CHUNK_HALO),
                            store.chunk_override_y_range(coord),
                        )
                    } else {
                        (HashMap::new(), None)
                    }
                };
                let has_overrides = !overrides.is_empty();
                if should_pack_far_lod(mode, step)
                    && dirty_rev.is_none()
                    && !has_overrides
                    && let Some(cached) = try_load_cached_mesh(&cache_dir, coord, step, mode)
                {
                    let _ = tx_res.send(cached.into_worker_result(coord, step, mode));
                    continue;
                }

                let mut edit_range = if dirty_rev.is_some() {
                    edited_chunk_ranges.lock().unwrap().get(&coord_key).copied()
                } else {
                    None
                };
                if edit_range.is_none() {
                    edit_range = override_range;
                }

                let height_cache = RefCell::new(HashMap::<(i32, i32), i32>::new());
                let tree_cache = RefCell::new(HashMap::<(i32, i32), Option<TreeSpec>>::new());
                let block_at = |x: i32, y: i32, z: i32| -> i8 {
                    overrides.get(&(x, y, z)).copied().unwrap_or_else(|| {
                        block_id_full_cached(&gen_thread, &height_cache, &tree_cache, x, y, z)
                    })
                };
                let mesh = generate_chunk_mesh(
                    coord,
                    &blocks_gen,
                    &gen_thread,
                    &block_at,
                    edit_range,
                    step,
                    mode,
                );
                if should_pack_far_lod(mode, step) {
                    let packed_vertices = Arc::<[_]>::from(pack_far_vertices(&mesh.vertices));
                    let packed_indices = Arc::<[_]>::from(mesh.indices);
                    let packed = PackedMeshData {
                        coord: mesh.coord,
                        step: mesh.step,
                        mode: mesh.mode,
                        center: mesh.center,
                        radius: mesh.radius,
                        vertices: Arc::clone(&packed_vertices),
                        indices: Arc::clone(&packed_indices),
                    };
                    if dirty_rev.is_none() && !has_overrides {
                        let _ = tx_cache.send(CacheWriteMsg::Write {
                            coord: mesh.coord,
                            step: mesh.step,
                            mode: mesh.mode,
                            center: mesh.center,
                            radius: mesh.radius,
                            vertices: packed_vertices,
                            indices: packed_indices,
                        });
                    }
                    let _ = tx_res.send(WorkerResult::Packed { packed, dirty_rev });
                } else {
                    let _ = tx_res.send(WorkerResult::Raw { mesh, dirty_rev });
                }
            }
        });
    }

    let base_render_radius = 192;
    let max_render_radius = 320;
    let base_draw_radius = 72;
    let lod_near_radius = 16;
    let lod_mid_radius = 32;
    let request_budget = 112;
    let max_inflight = 1024usize;
    let surface_depth_chunks = 4;
    let initial_burst_radius = 4;
    let loaded_chunk_cap = 12000usize;
    let mesh_memory_cap_mb = 6144usize;
    let mut current_chunk = chunk_coord_from_pos(player.position);
    let pregen_center_chunk = current_chunk;
    let requested_pregen_span_chunks = 1000;
    let pregen_radius_chunks = (requested_pregen_span_chunks / 2).clamp(8, 64);
    if pregen_radius_chunks * 2 + 1 < requested_pregen_span_chunks {
        eprintln!(
            "pregen span {}x{} chunks requested, capped to {}x{} for memory/perf safety",
            requested_pregen_span_chunks,
            requested_pregen_span_chunks,
            pregen_radius_chunks * 2 + 1,
            pregen_radius_chunks * 2 + 1,
        );
    }
    let pregen_budget_per_tick = 64;
    let pregen_total_columns = ((pregen_radius_chunks * 2 + 1) as usize).pow(2);
    let pregen_est_chunks_total = pregen_total_columns * (surface_depth_chunks as usize + 2);
    let mut pregen_active = false;
    let mut pregen_ring_r: i32 = 0;
    let mut pregen_ring_i: i32 = 0;
    let mut pregen_columns_done: usize = 0;
    let mut pregen_chunks_requested: usize = 0;
    let mut pregen_chunks_created: usize = 0;
    let mut loaded: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut requested: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut column_height_cache: HashMap<(i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;
    let mut emergency_budget: i32 = 16;
    let mut looked_block: Option<(IVec3, i8)> = None;
    let mut looked_hit: Option<(IVec3, i8, IVec3)> = None;
    let mut adaptive_request_budget = request_budget;
    let mut adaptive_pregen_budget = pregen_budget_per_tick;
    let mut adaptive_max_apply_per_tick: usize = 16;
    let mut adaptive_max_rebuilds_per_tick: usize = 3;
    let mut adaptive_draw_radius_cap = base_draw_radius;
    let mut last_camera_pos = camera.position;
    let mut last_camera_forward = camera.forward;

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
                    let draw_radius_base = (base_draw_radius + height_chunks).min(render_radius);
                    let draw_radius = draw_radius_base.min(adaptive_draw_radius_cap.max(16));
                    stream_tick(
                        &mut gpu,
                        player.position,
                        camera.forward,
                        &mut current_chunk,
                        &mut ring_r,
                        &mut ring_i,
                        &mut loaded,
                        &mut requested,
                        &mut column_height_cache,
                        &request_queue,
                        &rx_res,
                        &dirty_chunks,
                        &edited_chunk_ranges,
                        base_render_radius,
                        max_render_radius,
                        adaptive_request_budget,
                        initial_burst_radius,
                        max_inflight,
                        &mut force_reload,
                        &world_gen,
                        surface_depth_chunks,
                        draw_radius,
                        lod_near_radius,
                        lod_mid_radius,
                        &mut emergency_budget,
                        pregen_center_chunk,
                        pregen_radius_chunks,
                        adaptive_pregen_budget,
                        &mut pregen_active,
                        &mut pregen_ring_r,
                        &mut pregen_ring_i,
                        &mut pregen_columns_done,
                        &mut pregen_chunks_requested,
                        &mut pregen_chunks_created,
                        loaded_chunk_cap,
                        mesh_memory_cap_mb,
                        adaptive_max_apply_per_tick,
                        adaptive_max_rebuilds_per_tick,
                    );
                    let camera_changed = (camera.position - last_camera_pos).length_squared() > 0.0009
                        || camera.forward.dot(last_camera_forward) < 0.9998;
                    if camera_changed || !requested.is_empty() || pregen_active {
                        gpu.update_visible(&camera, draw_radius);
                        last_camera_pos = camera.position;
                        last_camera_forward = camera.forward;
                    }
                    looked_hit = pick_block(camera.position, camera.forward, 8.0, |x, y, z| {
                        block_id_with_edits(&world_gen, &edited_blocks, x, y, z)
                    })
                    .map(|hit| (hit.block, hit.block_id, hit.place));
                    looked_block = looked_hit.map(|(block, block_id, _)| (block, block_id));
                    tps_ticks += 1;
                }

                if !pause_stream {
                    let dirty_pending = dirty_chunks.lock().unwrap().len();
                    let apply_budget = if dirty_pending > 0 {
                        adaptive_max_apply_per_tick.max(16)
                    } else if !requested.is_empty() {
                        (adaptive_max_apply_per_tick / 2).max(4)
                    } else {
                        2
                    };
                    let rebuild_budget = if dirty_pending > 0 {
                        adaptive_max_rebuilds_per_tick + 4
                    } else {
                        adaptive_max_rebuilds_per_tick.max(1)
                    };
                    apply_stream_results(
                        &mut gpu,
                        &rx_res,
                        &dirty_chunks,
                        &edited_chunk_ranges,
                        &mut loaded,
                        &mut requested,
                        player.position,
                        pregen_center_chunk,
                        pregen_radius_chunks,
                        &mut pregen_chunks_created,
                        loaded_chunk_cap,
                        mesh_memory_cap_mb,
                        apply_budget,
                        rebuild_budget,
                        true,
                    );
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

                    if tps_value < 18 || fps_value < 35 {
                        adaptive_request_budget = (adaptive_request_budget * 85 / 100).max(48);
                        adaptive_pregen_budget = (adaptive_pregen_budget * 80 / 100).max(24);
                        adaptive_max_apply_per_tick = adaptive_max_apply_per_tick.saturating_sub(2).max(8);
                        adaptive_max_rebuilds_per_tick = adaptive_max_rebuilds_per_tick.saturating_sub(1).max(1);
                        adaptive_draw_radius_cap = (adaptive_draw_radius_cap - 4).max(24);
                    } else if tps_value >= 20 && fps_value > 60 {
                        adaptive_request_budget = (adaptive_request_budget + 12).min(request_budget);
                        adaptive_pregen_budget = (adaptive_pregen_budget + 8).min(pregen_budget_per_tick);
                        adaptive_max_apply_per_tick = (adaptive_max_apply_per_tick + 1).min(24);
                        adaptive_max_rebuilds_per_tick = (adaptive_max_rebuilds_per_tick + 1).min(4);
                        adaptive_draw_radius_cap = (adaptive_draw_radius_cap + 2).min(base_draw_radius + 32);
                    }
                }

                if last_title_update.elapsed() >= Duration::from_millis(250) {
                    if debug_ui || pregen_active {
                        let mesh_mem_mb = estimate_mesh_memory_mb(&gpu.stats());
                        let title = format!(
                            "loaded:{} requested:{} tps:{} fps:{} mem:{}MB paused_stream:{} paused_render:{} pregen:{}",
                            loaded.len(),
                            requested.len(),
                            tps_value,
                            fps_value,
                            mesh_mem_mb,
                            pause_stream,
                            pause_render,
                            pregen_active,
                        );
                        gpu.window().set_title(&title);
                    } else {
                        gpu.window().set_title("NyraCraft");
                    }
                    last_title_update = Instant::now();
                }

                if (debug_ui || pregen_active) && last_stats_print.elapsed() >= Duration::from_millis(500) {
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
                        looked_block,
                        pregen_active,
                        pregen_ring_r,
                        pregen_radius_chunks,
                        pregen_columns_done,
                        pregen_total_columns,
                        pregen_chunks_requested,
                        pregen_chunks_created,
                        pregen_est_chunks_total,
                        loaded_chunk_cap,
                        mesh_memory_cap_mb,
                    );
                    last_stats_print = Instant::now();
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
            if !pause_render && !pregen_active {
                let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                let draw_radius_base = (base_draw_radius + height_chunks).min(render_radius);
                let draw_radius = draw_radius_base.min(adaptive_draw_radius_cap.max(16));
                gpu.set_selection(looked_block.map(|(coord, _)| coord));
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
                if pressed {
                    f1_down = true;
                    f1_combo_used = false;
                } else {
                    if !f1_combo_used {
                        print_keybinds();
                    }
                    f1_down = false;
                    f1_combo_used = false;
                }
                return;
            }
            if pressed && f1_down {
                f1_combo_used = true;
                match key {
                    KeyCode::KeyF => debug_faces = !debug_faces,
                    KeyCode::KeyW => debug_chunks = !debug_chunks,
                    KeyCode::KeyP => pause_stream = !pause_stream,
                    KeyCode::KeyV => pause_render = !pause_render,
                    KeyCode::KeyD => debug_ui = !debug_ui,
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
        Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button,
                    ..
                },
            ..
        } => {
            if !mouse_enabled {
                return;
            }
            let had_target = looked_hit.is_some();
            handle_block_mouse_input(
                button,
                looked_hit,
                &world_gen,
                &edited_blocks,
                &edited_chunk_ranges,
                &dirty_chunks,
                &request_queue,
                player.position,
                player_config.height,
                player_config.radius,
            );
            if had_target && matches!(button, MouseButton::Left | MouseButton::Right) && !pause_stream {
                apply_stream_results(
                    &mut gpu,
                    &rx_res,
                    &dirty_chunks,
                    &edited_chunk_ranges,
                    &mut loaded,
                    &mut requested,
                    player.position,
                    pregen_center_chunk,
                    pregen_radius_chunks,
                    &mut pregen_chunks_created,
                    loaded_chunk_cap,
                    mesh_memory_cap_mb,
                    24,
                    adaptive_max_rebuilds_per_tick.max(8),
                    true,
                );
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
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            if !mouse_enabled {
                return;
            }
            if let Some((lx, ly)) = last_cursor_pos {
                let dx = (position.x - lx) as f32;
                let dy = (position.y - ly) as f32;
                // Fallback path for platforms where raw device motion gets throttled with key combos.
                if dx != 0.0 || dy != 0.0 {
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
            }
            last_cursor_pos = Some((position.x, position.y));
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
                last_cursor_pos = None;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::Focused(false),
            ..
        } => {
            let _ = gpu.window().set_cursor_grab(CursorGrabMode::None);
            gpu.window().set_cursor_visible(true);
            mouse_enabled = false;
            last_cursor_pos = None;
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

fn print_keybinds() {
    println!("--- NyraCraft Keybinds ---");
    println!("Movement: W/A/S/D | Jump: Space | Descend (fly): Shift");
    println!("Mouse Left: Break block | Mouse Right: Place stone block");
    println!("F1 + F : Toggle face debug colors");
    println!("F1 + W : Toggle chunk wireframe");
    println!("F1 + P : Pause/resume chunk streaming");
    println!("F1 + V : Pause/resume rendering");
    println!("F1 + D : Toggle debug stats UI");
    println!("F1 + R : Force chunk reload");
    println!("F1 + M : Toggle fly mode");
    println!("F1 + X : Toggle borderless fullscreen");
}
