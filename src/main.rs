use glam::{IVec3, Vec3};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

mod app;
mod physics;
mod player;
mod render;
mod world;

use app::bootstrap::{detect_cpu_label, generate_texture_atlases, resolve_launch_seed};
use app::console::{
    CommandConsoleState, append_console_input, close_command_console,
    execute_and_close_command_console, execute_console_command, is_console_open_shortcut,
};
use app::controls::{
    apply_look_delta, clear_movement_input, disable_mouse_look, print_keybinds,
    try_enable_mouse_look,
};
use app::dropped_items::{
    DroppedItem, build_dropped_item_render_data, nudge_items_from_placed_block, spawn_block_drop,
    spawn_leaf_loot_drops, throw_hotbar_item, update_dropped_items,
};
use app::streaming::{
    CacheMeshView, CacheWriteMsg, EditedChunkRanges, PackedMeshData, SharedRequestQueue,
    WorkerResult, apply_stream_results, block_id_full_cached, chunk_coord_from_pos,
    is_solid_cached, mesh_cache_dir, new_request_queue, pick_block,
    pop_request, print_stats, request_queue_stats, should_pack_far_lod, stream_tick,
    try_load_cached_mesh, write_cached_mesh, DebugPerfSnapshot,
};
use player::inventory::{InventorySlotRef, InventoryState, hit_test_slot};
use player::{
    Camera, EditedBlocks, LeafDecayQueue, PlayerConfig, PlayerInput, PlayerState,
    block_id_with_edits, handle_block_mouse_input, new_edited_blocks, new_leaf_decay_queue,
    tick_leaf_decay, update_player,
};
use render::Gpu;
use render::mesh::pack_far_vertices;
use render::{CubeStyle, TextureAtlas};
use world::CHUNK_SIZE;
use world::blocks::{
    BLOCK_CRAFTING_TABLE, BLOCK_LEAVES, DEFAULT_TILES_X, HOTBAR_SLOTS, block_break_stage,
    block_break_time_with_strength_seconds, build_block_index, default_blocks,
};
use world::mesher::generate_chunk_mesh;
use world::worldgen::{TreeSpec, WorldGen};

type CoordKey = (i32, i32, i32);
type DirtyChunks = Arc<Mutex<HashMap<CoordKey, i64>>>;
const DIRTY_EDIT_CHUNK_HALO: i32 = 1;
const HAND_BREAK_STRENGTH: f32 = 1.0;

fn ema_ms(previous: f32, sample_ms: f32) -> f32 {
    if previous <= 0.001 {
        sample_ms
    } else {
        previous * 0.82 + sample_ms * 0.18
    }
}

fn main() {
    let (seed, world_mode, seed_input) = resolve_launch_seed();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("NyraCraft")
        .build(&event_loop)
        .unwrap();

    generate_texture_atlases();

    let style = CubeStyle { use_texture: true };

    let atlas = TextureAtlas {
        path: "src/texturing/atlas_output/atls_blocks.png".to_string(),
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

    let world_gen = WorldGen::new(seed, world_mode);
    eprintln!(
        "World seed input: {} | resolved_seed: {} | mode: {} | world_id: {}",
        seed_input,
        world_gen.seed,
        world_gen.mode_name(),
        world_gen.world_id
    );

    let edited_blocks: EditedBlocks = new_edited_blocks();
    let leaf_decay_queue: LeafDecayQueue = new_leaf_decay_queue();
    let dirty_chunks: DirtyChunks = Arc::new(Mutex::new(HashMap::new()));
    let edited_chunk_ranges: EditedChunkRanges = Arc::new(Mutex::new(HashMap::new()));

    let mut gpu = Gpu::new(window, style, Some(atlas));
    let gpu_label = gpu.adapter_summary();
    let cpu_label = detect_cpu_label();
    gpu.window().set_ime_allowed(false);

    let delay = Duration::from_millis(16);
    let move_speed = 4.0f32;
    let tick_dt = Duration::from_millis(50);
    let mut tick_accum = Duration::ZERO;
    let mut next_tick = Instant::now() + delay;
    let mut last_update = Instant::now();
    let mut last_stats_print = Instant::now();
    let start_time = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut fps_last = Instant::now();
    let mut fps_value: u32 = 0;
    let mut tps_ticks: u32 = 0;
    let mut tps_last = Instant::now();
    let mut tps_value: u32 = 0;
    let mut perf = DebugPerfSnapshot::default();
    let (tx_thread_report, rx_thread_report) = mpsc::channel::<String>();
    let mut thread_reports = vec![format!(
        "main(loop/input/player/render): tid={:?}",
        thread::current().id()
    )];

    let spawn_height = (world_gen.highest_solid_y_at(0, 0) + 4).max(4);
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
        sprint_multiplier: 1.65,
        sneak_multiplier: 0.45,
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
        sprint: false,
        sneak: false,
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
    let mut selected_hotbar_slot: u8 = 0;
    let mut inventory = InventoryState::new();
    let mut dropped_items: Vec<DroppedItem> = Vec::new();
    let mut command_console = CommandConsoleState::default();
    let mut ui_cursor_pos: Option<(f32, f32)> = None;
    let mut f1_down = false;
    let mut f1_combo_used = false;
    let mut shift_down = false;
    let mut inventory_left_mouse_down = false;
    let mut inventory_right_mouse_down = false;
    let mut inventory_drag_last_slot: Option<InventorySlotRef> = None;
    let mut mine_left_down = false;
    let mut mining_target: Option<IVec3> = None;
    let mut mining_progress = 0.0f32;
    let mut mining_overlay: Option<(IVec3, u32)> = None;
    let (tx_console_stdin, rx_console_stdin) = mpsc::channel::<String>();
    let tx_thread_report_stdin = tx_thread_report.clone();
    let _ = thread::Builder::new()
        .name("nyra-stdin".to_string())
        .spawn(move || {
            let _ = tx_thread_report_stdin.send(format!("stdin: tid={:?}", thread::current().id()));
            let stdin = io::stdin();
            loop {
                let mut line = String::new();
                match stdin.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if tx_console_stdin.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

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
    let tx_thread_report_cache = tx_thread_report.clone();
    let _ = thread::Builder::new()
        .name("nyra-cache-writer".to_string())
        .spawn(move || {
            let _ = tx_thread_report_cache
                .send(format!("cache_writer: tid={:?}", thread::current().id()));
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
        // Keep one core free for render/input, but allow enough workers so edits remesh quickly.
        .map(|n| n.get().saturating_sub(1).clamp(2, 12))
        .unwrap_or(2);
    for worker_idx in 0..worker_count {
        let request_queue = Arc::clone(&request_queue);
        let tx_res = tx_res.clone();
        let tx_cache = tx_cache.clone();
        let blocks_gen = blocks_gen.clone();
        let gen_thread = gen_thread.clone();
        let cache_dir = cache_dir.clone();
        let edited_blocks = Arc::clone(&edited_blocks);
        let edited_chunk_ranges = Arc::clone(&edited_chunk_ranges);
        let dirty_chunks = Arc::clone(&dirty_chunks);
        let tx_thread_report_worker = tx_thread_report.clone();
        let _ = thread::Builder::new()
            .name(format!("nyra-worker-{worker_idx}"))
            .spawn(move || {
            let _ = tx_thread_report_worker.send(format!(
                "worker-{worker_idx}(mesh/gen/cache-read): tid={:?}",
                thread::current().id()
            ));
            while let Some(task) = pop_request(&request_queue) {
                let coord = task.coord;
                let step = task.step;
                let mode = task.mode;
                let lighting_pass = task.lighting_pass;
                let coord_key = (coord.x, coord.y, coord.z);
                let dirty_rev = dirty_chunks
                    .lock()
                    .unwrap()
                    .get(&coord_key)
                    .copied()
                    .and_then(|state| {
                        let rev = state.abs();
                        if rev == 0 { None } else { Some(rev as u64) }
                    });
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

                let edit_range = if dirty_rev.is_some() {
                    let recent_range = edited_chunk_ranges.lock().unwrap().get(&coord_key).copied();
                    match (recent_range, override_range) {
                        (Some((a_min, a_max)), Some((b_min, b_max))) => {
                            Some((a_min.min(b_min), a_max.max(b_max)))
                        }
                        (Some(range), None) | (None, Some(range)) => Some(range),
                        (None, None) => None,
                    }
                } else {
                    None
                };

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
                    lighting_pass,
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

    let base_render_radius = 512;
    let max_render_radius = 1024;
    let base_draw_radius = 192;
    let lod_near_radius = 16;
    let lod_mid_radius = 32;
    let request_budget = 112;
    let max_inflight = 1024usize;
    let surface_depth_chunks = 4;
    let initial_burst_radius = 4;
    let loaded_chunk_cap = 100000usize; // important
    let mesh_memory_cap_mb = 12288usize; // 12 GB cap
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

            while let Ok(report) = rx_thread_report.try_recv() {
                if !thread_reports.iter().any(|existing| existing == &report) {
                    thread_reports.push(report);
                    thread_reports.sort();
                }
            }

            while let Ok(line) = rx_console_stdin.try_recv() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if command_console.open || line.starts_with('/') {
                    let feedback = execute_console_command(line, &mut inventory);
                    let printed = line.trim_start_matches('/');
                    println!("/{} -> {}", printed, feedback);
                    if command_console.open {
                        close_command_console(
                            &mut command_console,
                            inventory.open,
                            gpu.window(),
                            &mut mouse_enabled,
                            &mut last_cursor_pos,
                        );
                    }
                }
            }

            let now = Instant::now();
            if now >= next_tick {
                let dt = (now - last_update).as_secs_f32();
                let dt = dt.min(0.05);
                last_update = now;
                let tick_start = Instant::now();

                let player_phase_start = Instant::now();
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
                update_dropped_items(
                    &mut dropped_items,
                    dt,
                    player.position,
                    player_config.height,
                    &world_gen,
                    &edited_blocks,
                    &mut inventory,
                );
                perf.player_ms = ema_ms(
                    perf.player_ms,
                    player_phase_start.elapsed().as_secs_f32() * 1000.0,
                );

                tick_accum = tick_accum.saturating_add(Duration::from_secs_f32(dt));
                let mut stream_ms_accum = 0.0f32;
                let mut stream_calls = 0u32;
                while tick_accum >= tick_dt {
                    tick_accum -= tick_dt;
                    if pause_stream {
                        continue;
                    }
                    if let Some(decayed_leaf) = tick_leaf_decay(
                        &world_gen,
                        &edited_blocks,
                        &leaf_decay_queue,
                        &edited_chunk_ranges,
                        &dirty_chunks,
                        &request_queue,
                    ) {
                        spawn_leaf_loot_drops(&mut dropped_items, decayed_leaf);
                    }
                    let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                    let render_radius =
                        (base_render_radius + height_chunks * 2).min(max_render_radius);
                    let draw_radius_base = (base_draw_radius + height_chunks).min(render_radius);
                    let draw_radius = draw_radius_base.min(adaptive_draw_radius_cap.max(16));
                    let stream_phase_start = Instant::now();
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
                    stream_ms_accum += stream_phase_start.elapsed().as_secs_f32() * 1000.0;
                    stream_calls += 1;
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
                if stream_calls > 0 {
                    perf.stream_ms = ema_ms(perf.stream_ms, stream_ms_accum / stream_calls as f32);
                }

                let mining_phase_start = Instant::now();
                mining_overlay = None;
                if mine_left_down
                    && mouse_enabled
                    && !inventory.open
                    && !command_console.open
                    && !pause_stream
                {
                    if let Some((target_block, _target_id, _place_block)) = looked_hit {
                        let current_target_id = block_id_with_edits(
                            &world_gen,
                            &edited_blocks,
                            target_block.x,
                            target_block.y,
                            target_block.z,
                        );
                        if current_target_id >= 0 {
                            if mining_target != Some(target_block) {
                                mining_target = Some(target_block);
                                mining_progress = 0.0;
                            }
                            if let Some(break_time) = block_break_time_with_strength_seconds(
                                current_target_id,
                                HAND_BREAK_STRENGTH,
                            ) {
                                mining_progress += dt;
                                let stage = block_break_stage(mining_progress, break_time);
                                mining_overlay = Some((target_block, stage));
                                if mining_progress >= break_time {
                                    let edit_result = handle_block_mouse_input(
                                        MouseButton::Left,
                                        looked_hit,
                                        &world_gen,
                                        &edited_blocks,
                                        &leaf_decay_queue,
                                        &edited_chunk_ranges,
                                        &dirty_chunks,
                                        &request_queue,
                                        player.position,
                                        player_config.height,
                                        player_config.radius,
                                        inventory.selected_hotbar_block(selected_hotbar_slot),
                                    );
                                    if let Some((broken_pos, broken_id)) = edit_result.broke {
                                        if broken_id == BLOCK_LEAVES as i8 {
                                            spawn_leaf_loot_drops(&mut dropped_items, broken_pos);
                                        } else {
                                            spawn_block_drop(&mut dropped_items, broken_pos, broken_id);
                                        }
                                        mining_target = None;
                                        mining_progress = 0.0;
                                        mining_overlay = None;
                                        if !pause_stream {
                                            apply_stream_results(
                                                &mut gpu,
                                            &rx_res,
                                            &dirty_chunks,
                                            &edited_chunk_ranges,
                                            &request_queue,
                                            &mut loaded,
                                            &mut requested,
                                                player.position,
                                                pregen_center_chunk,
                                                pregen_radius_chunks,
                                                &mut pregen_chunks_created,
                                                loaded_chunk_cap,
                                                mesh_memory_cap_mb,
                                                128,
                                                adaptive_max_rebuilds_per_tick.max(16),
                                                true,
                                            );
                                        }
                                    }
                                }
                            } else {
                                mining_target = None;
                                mining_progress = 0.0;
                                mining_overlay = None;
                            }
                        } else {
                            mining_target = None;
                            mining_progress = 0.0;
                            mining_overlay = None;
                        }
                    } else {
                        mining_target = None;
                        mining_progress = 0.0;
                        mining_overlay = None;
                    }
                } else {
                    mining_target = None;
                    mining_progress = 0.0;
                    mining_overlay = None;
                }
                perf.mining_ms = ema_ms(
                    perf.mining_ms,
                    mining_phase_start.elapsed().as_secs_f32() * 1000.0,
                );

                let (dirty_pending_debug, apply_budget_debug, rebuild_budget_debug) =
                    if !pause_stream {
                        let dirty_pending = dirty_chunks
                            .lock()
                            .unwrap()
                            .values()
                            .filter(|state| **state > 0)
                            .count();
                        let apply_budget = if dirty_pending > 0 {
                            adaptive_max_apply_per_tick.max(64)
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
                        let apply_phase_start = Instant::now();
                        apply_stream_results(
                            &mut gpu,
                            &rx_res,
                            &dirty_chunks,
                            &edited_chunk_ranges,
                            &request_queue,
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
                        perf.apply_ms = ema_ms(
                            perf.apply_ms,
                            apply_phase_start.elapsed().as_secs_f32() * 1000.0,
                        );
                        (dirty_pending, apply_budget, rebuild_budget)
                    } else {
                        let dirty_pending = dirty_chunks
                            .lock()
                            .unwrap()
                            .values()
                            .filter(|state| **state > 0)
                            .count();
                        (dirty_pending, 0, 0)
                    };

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

                if (debug_ui || pregen_active) && last_stats_print.elapsed() >= Duration::from_millis(500) {
                    let stats = gpu.stats();
                    let req_stats = request_queue_stats(&request_queue);
                    print_stats(
                        &stats,
                        &loaded,
                        &requested,
                        tps_value,
                        fps_value,
                        &cpu_label,
                        &gpu_label,
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
                        HAND_BREAK_STRENGTH,
                        mining_target,
                        mining_progress,
                        dirty_pending_debug,
                        req_stats,
                        perf,
                        adaptive_request_budget,
                        adaptive_pregen_budget,
                        apply_budget_debug,
                        rebuild_budget_debug,
                        adaptive_draw_radius_cap,
                        worker_count,
                        &thread_reports,
                    );
                    last_stats_print = Instant::now();
                }

                if !pause_render {

                    gpu.window().request_redraw();
                }
                perf.tick_cpu_ms = ema_ms(perf.tick_cpu_ms, tick_start.elapsed().as_secs_f32() * 1000.0);
                next_tick = now + delay;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            if !pause_render && !pregen_active {
                let window_size = gpu.window().inner_size();
                let hovered_inventory_slot = if inventory.open {
                    ui_cursor_pos.and_then(|(mx, my)| {
                        hit_test_slot(
                            window_size.width,
                            window_size.height,
                            mx,
                            my,
                            true,
                            inventory.craft_grid_side(),
                        )
                    })
                } else {
                    None
                };
                let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                let draw_radius_base = (base_draw_radius + height_chunks).min(render_radius);
                let draw_radius = draw_radius_base.min(adaptive_draw_radius_cap.max(16));
                let render_time = start_time.elapsed().as_secs_f32();
                let dropped_render_items =
                    build_dropped_item_render_data(&dropped_items, render_time);
                gpu.set_selection(looked_block.map(|(coord, _)| coord));
                let render_phase_start = Instant::now();
                gpu.render(
                    &camera,
                    debug_faces,
                    debug_chunks,
                    draw_radius,
                    selected_hotbar_slot,
                    &inventory.hotbar,
                    inventory.open,
                    &inventory.storage,
                    inventory.craft_grid_side(),
                    &inventory.craft_input,
                    inventory.craft_output_preview(),
                    hovered_inventory_slot,
                    ui_cursor_pos,
                    inventory.dragged_item,
                    mining_overlay,
                    &dropped_render_items,
                );
                perf.render_cpu_ms = ema_ms(
                    perf.render_cpu_ms,
                    render_phase_start.elapsed().as_secs_f32() * 1000.0,
                );
            }
        }
        Event::WindowEvent {
            event: WindowEvent::Ime(Ime::Commit(text)),
            ..
        } => {
            if command_console.open {
                let has_newline = text.contains('\n') || text.contains('\r');
                append_console_input(&mut command_console, &text);
                if has_newline {
                    let inventory_open = inventory.open;
                    execute_and_close_command_console(
                        &mut command_console,
                        &mut inventory,
                        inventory_open,
                        gpu.window(),
                        &mut mouse_enabled,
                        &mut last_cursor_pos,
                    );
                }
            }
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { event, .. },
            ..
        } => {
            let pressed = matches!(event.state, ElementState::Pressed);
            let key = match event.physical_key {
                PhysicalKey::Code(code) => Some(code),
                _ => None,
            };
            if let Some(KeyCode::ShiftLeft | KeyCode::ShiftRight) = key {
                shift_down = pressed;
            }
            if command_console.open {
                if pressed {
                    if let Some(text) = event.text.as_ref()
                        && (text.contains('\n') || text.contains('\r'))
                    {
                        let inventory_open = inventory.open;
                        execute_and_close_command_console(
                            &mut command_console,
                            &mut inventory,
                            inventory_open,
                            gpu.window(),
                            &mut mouse_enabled,
                            &mut last_cursor_pos,
                        );
                        return;
                    }
                    match key {
                        Some(KeyCode::Escape) => {
                            close_command_console(
                                &mut command_console,
                                inventory.open,
                                gpu.window(),
                                &mut mouse_enabled,
                                &mut last_cursor_pos,
                            );
                        }
                        Some(KeyCode::Enter) | Some(KeyCode::NumpadEnter) => {
                            let inventory_open = inventory.open;
                            execute_and_close_command_console(
                                &mut command_console,
                                &mut inventory,
                                inventory_open,
                                gpu.window(),
                                &mut mouse_enabled,
                                &mut last_cursor_pos,
                            );
                        }
                        Some(KeyCode::Backspace) => {
                            command_console.input.pop();
                        }
                        _ => {
                            if let Some(text) = event.text.as_ref() {
                                append_console_input(&mut command_console, text.as_str());
                            }
                        }
                    }
                }
                return;
            }

            let Some(key) = key else {
                return;
            };

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
            if !inventory.open && is_console_open_shortcut(&event) {
                let cached_cursor = last_cursor_pos.map(|(x, y)| (x as f32, y as f32));
                disable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                ui_cursor_pos = cached_cursor;
                clear_movement_input(&mut input);
                mine_left_down = false;
                mining_target = None;
                mining_progress = 0.0;
                inventory_left_mouse_down = false;
                inventory_right_mouse_down = false;
                inventory_drag_last_slot = None;
                command_console.open = true;
                command_console.input.clear();
                gpu.window().set_ime_allowed(true);
                println!(
                    "Console opened. Type command in window or terminal, then Enter. Example: give apple 1"
                );
                return;
            }
            if pressed && key == KeyCode::KeyE {
                inventory.toggle_open();
                if inventory.open {
                    let cached_cursor = last_cursor_pos.map(|(x, y)| (x as f32, y as f32));
                    disable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                    ui_cursor_pos = cached_cursor;
                    clear_movement_input(&mut input);
                    mine_left_down = false;
                    mining_target = None;
                    mining_progress = 0.0;
                    inventory_left_mouse_down = false;
                    inventory_right_mouse_down = false;
                    inventory_drag_last_slot = None;
                } else {
                    try_enable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                    inventory_left_mouse_down = false;
                    inventory_right_mouse_down = false;
                    inventory_drag_last_slot = None;
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
                    KeyCode::KeyM => {
                        fly_mode = !fly_mode;
                        input.sneak = false;
                        input.move_down = false;
                    }
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
                KeyCode::Digit1 if pressed => selected_hotbar_slot = 0,
                KeyCode::Digit2 if pressed => selected_hotbar_slot = 1,
                KeyCode::Digit3 if pressed => selected_hotbar_slot = 2,
                KeyCode::Digit4 if pressed => selected_hotbar_slot = 3,
                KeyCode::Digit5 if pressed => selected_hotbar_slot = 4,
                KeyCode::Digit6 if pressed => selected_hotbar_slot = 5,
                KeyCode::Digit7 if pressed => selected_hotbar_slot = 6,
                KeyCode::Digit8 if pressed => selected_hotbar_slot = 7,
                KeyCode::Digit9 if pressed => selected_hotbar_slot = 8,
                KeyCode::KeyQ if pressed && !inventory.open && !command_console.open => {
                    if let Some(stack) = inventory.take_one_selected_hotbar(selected_hotbar_slot) {
                        throw_hotbar_item(
                            &mut dropped_items,
                            player.position,
                            player_config.height,
                            camera.forward,
                            stack,
                        );
                    }
                }
                KeyCode::KeyW if !inventory.open && !command_console.open => {
                    input.move_forward = pressed
                }
                KeyCode::KeyS if !inventory.open && !command_console.open => input.move_back = pressed,
                KeyCode::KeyA if !inventory.open && !command_console.open => input.move_left = pressed,
                KeyCode::KeyD if !inventory.open && !command_console.open => {
                    input.move_right = pressed
                }
                KeyCode::Space => {
                    if inventory.open || command_console.open {
                        return;
                    }
                    if fly_mode {
                        input.move_up = pressed;
                    } else if pressed {
                        input.jump = true;
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                    if inventory.open || command_console.open {
                        input.sprint = false;
                        return;
                    }
                    input.sprint = pressed;
                }
                KeyCode::KeyC => {
                    if inventory.open || command_console.open {
                        input.sneak = false;
                        if fly_mode {
                            input.move_down = false;
                        }
                        return;
                    }
                    if fly_mode {
                        input.move_down = pressed;
                    } else {
                        input.sneak = pressed;
                    }
                }
                _ => {}
            }
        }
        Event::WindowEvent {
            event: WindowEvent::MouseWheel { delta, .. },
            ..
        } => {
            if command_console.open {
                return;
            }
            let wheel_y = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
            };
            let step = if wheel_y > 0.0 {
                -1
            } else if wheel_y < 0.0 {
                1
            } else {
                0
            };
            if step != 0 {
                let slots = HOTBAR_SLOTS as i32;
                selected_hotbar_slot =
                    (selected_hotbar_slot as i32 + step).rem_euclid(slots) as u8;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::MouseInput { state, button, .. },
            ..
        } => {
            let pressed = matches!(state, ElementState::Pressed);
            if command_console.open {
                return;
            }
            if inventory.open {
                if matches!(button, MouseButton::Left | MouseButton::Right) {
                    if ui_cursor_pos.is_none() {
                        ui_cursor_pos = last_cursor_pos.map(|(x, y)| (x as f32, y as f32));
                    }
                    let size = gpu.window().inner_size();
                    let hovered = ui_cursor_pos.and_then(|(mx, my)| {
                        hit_test_slot(
                            size.width,
                            size.height,
                            mx,
                            my,
                            true,
                            inventory.craft_grid_side(),
                        )
                    });
                    if matches!(button, MouseButton::Left) {
                        inventory_left_mouse_down = pressed && !shift_down;
                        if pressed {
                            if shift_down {
                                inventory.quick_move_slot(hovered);
                            } else {
                                inventory.left_click_slot(hovered);
                            }
                            inventory_drag_last_slot = hovered;
                        }
                    } else {
                        inventory_right_mouse_down = pressed && !shift_down;
                        if pressed {
                            if shift_down {
                                inventory.quick_move_slot(hovered);
                            } else {
                                inventory.right_click_slot(hovered);
                            }
                            inventory_drag_last_slot = hovered;
                        }
                    }
                    if !inventory_left_mouse_down && !inventory_right_mouse_down {
                        inventory_drag_last_slot = None;
                    }
                }
                return;
            }

            if matches!(button, MouseButton::Left) {
                mine_left_down = pressed;
                if !pressed {
                    mining_target = None;
                    mining_progress = 0.0;
                }
            }

            if !pressed {
                return;
            }

            if !mouse_enabled {
                return;
            }
            if matches!(button, MouseButton::Right)
                && looked_hit.is_some_and(|(_, block_id, _)| block_id == BLOCK_CRAFTING_TABLE as i8)
            {
                inventory.open_crafting_table();
                let cached_cursor = last_cursor_pos.map(|(x, y)| (x as f32, y as f32));
                disable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                ui_cursor_pos = cached_cursor;
                clear_movement_input(&mut input);
                mine_left_down = false;
                mining_target = None;
                mining_progress = 0.0;
                inventory_left_mouse_down = false;
                inventory_right_mouse_down = false;
                inventory_drag_last_slot = None;
                return;
            }
            if !matches!(button, MouseButton::Right) {
                return;
            }
            let edit_result = handle_block_mouse_input(
                button,
                looked_hit,
                &world_gen,
                &edited_blocks,
                &leaf_decay_queue,
                &edited_chunk_ranges,
                &dirty_chunks,
                &request_queue,
                player.position,
                player_config.height,
                player_config.radius,
                inventory.selected_hotbar_block(selected_hotbar_slot),
            );
            if let Some((broken_pos, broken_id)) = edit_result.broke {
                if broken_id == BLOCK_LEAVES as i8 {
                    spawn_leaf_loot_drops(&mut dropped_items, broken_pos);
                } else {
                    spawn_block_drop(&mut dropped_items, broken_pos, broken_id);
                }
            }
            if let Some((placed_pos, _placed_id)) = edit_result.placed {
                let _ = inventory.consume_selected_hotbar(selected_hotbar_slot, 1);
                nudge_items_from_placed_block(
                    &mut dropped_items,
                    placed_pos,
                    &world_gen,
                    &edited_blocks,
                );
            }
            let edited_world = edit_result.broke.is_some() || edit_result.placed.is_some();
            if edited_world && !pause_stream {
                apply_stream_results(
                    &mut gpu,
                    &rx_res,
                    &dirty_chunks,
                    &edited_chunk_ranges,
                    &request_queue,
                    &mut loaded,
                    &mut requested,
                    player.position,
                    pregen_center_chunk,
                    pregen_radius_chunks,
                    &mut pregen_chunks_created,
                    loaded_chunk_cap,
                    mesh_memory_cap_mb,
                    128,
                    adaptive_max_rebuilds_per_tick.max(16),
                    true,
                );
            }
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            if !mouse_enabled || inventory.open || command_console.open {
                return;
            }
            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
            apply_look_delta(
                &mut camera,
                &mut yaw,
                &mut pitch,
                dx,
                dy,
                mouse_sensitivity,
            );
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            ui_cursor_pos = Some((position.x as f32, position.y as f32));
            if inventory.open {
                if inventory_right_mouse_down {
                    let size = gpu.window().inner_size();
                    let hovered = hit_test_slot(
                        size.width,
                        size.height,
                        position.x as f32,
                        position.y as f32,
                        true,
                        inventory.craft_grid_side(),
                    );
                    if hovered != inventory_drag_last_slot {
                        if shift_down {
                            inventory.quick_move_slot(hovered);
                        } else {
                            inventory.right_click_slot(hovered);
                        }
                        inventory_drag_last_slot = hovered;
                    }
                }
                last_cursor_pos = Some((position.x, position.y));
                return;
            }
            if !mouse_enabled || command_console.open {
                last_cursor_pos = Some((position.x, position.y));
                return;
            }
            if let Some((lx, ly)) = last_cursor_pos {
                let dx = (position.x - lx) as f32;
                let dy = (position.y - ly) as f32;
                // Fallback path for platforms where raw device motion gets throttled with key combos.
                if dx != 0.0 || dy != 0.0 {
                    apply_look_delta(
                        &mut camera,
                        &mut yaw,
                        &mut pitch,
                        dx,
                        dy,
                        mouse_sensitivity,
                    );
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
            if !inventory.open && !command_console.open {
                try_enable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
            }
        }
        Event::WindowEvent {
            event: WindowEvent::Focused(false),
            ..
        } => {
            disable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
            clear_movement_input(&mut input);
            shift_down = false;
            inventory.close();
            mine_left_down = false;
            mining_target = None;
            mining_progress = 0.0;
            inventory_left_mouse_down = false;
            inventory_right_mouse_down = false;
            inventory_drag_last_slot = None;
            command_console.open = false;
            command_console.input.clear();
            gpu.window().set_ime_allowed(false);
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
