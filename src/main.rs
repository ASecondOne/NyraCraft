use glam::{IVec3, Vec3};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
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
    execute_and_close_command_console, is_console_open_shortcut,
};
use app::controls::{
    apply_look_delta, clear_movement_input, disable_mouse_look, keybind_lines,
    try_enable_mouse_look,
};
use app::dropped_items::{
    DroppedItem, build_dropped_item_render_data, nudge_items_from_placed_block, spawn_block_drop,
    throw_hotbar_item, update_dropped_items,
};
use app::logger::{init_logger, install_panic_hook, log_info, log_warn};
use app::menu::{PauseMenuButton, hit_test_pause_menu_button};
use app::save::{SaveIo, SavedPlayerState};
use app::streaming::{
    ApplyDebugStats, CacheMeshView, CacheWriteMsg, DebugPerfSnapshot, EditedChunkRanges,
    PackedMeshData,
    SharedRequestQueue, WorkerResult, apply_stream_results, block_id_full_cached,
    build_stats_lines, chunk_coord_from_pos, is_solid_cached, mesh_cache_dir, new_request_queue,
    pick_block, pop_request, request_queue_stats, should_pack_far_lod, stream_tick,
    try_load_cached_mesh, write_cached_mesh,
};
use player::crafting::reload_recipes;
use player::inventory::{HOTBAR_SLOTS, InventorySlotRef, InventoryState, hit_test_slot};
use player::{
    Camera, EditedBlockEntry, EditedBlocks, LeafDecayQueue, PlayerConfig, PlayerInput,
    PlayerState, restore_loaded_edit_metadata,
    block_id_with_edits, break_blocks_batch, handle_block_mouse_input, leaf_decay_stats,
    new_edited_blocks, new_leaf_decay_queue, tick_leaf_decay, update_player,
};
use render::Gpu;
use render::gpu::reload_cpu_item_atlas_cache;
use render::mesh::pack_far_vertices;
use render::{CubeStyle, TextureAtlas};
use world::CHUNK_SIZE;
use world::blocks::{
    DEFAULT_TILES_X, block_break_stage, block_break_time_with_item_seconds, build_block_index,
    core_block_ids, default_blocks, reload_registry,
};
use world::mesher::generate_chunk_mesh;
use world::worldgen::{TreeSpec, WorldGen};

type CoordKey = (i32, i32, i32);
type DirtyChunks = Arc<Mutex<HashMap<CoordKey, i64>>>;
const DIRTY_EDIT_CHUNK_HALO: i32 = 1;
const HAND_BREAK_STRENGTH: f32 = 1.0;
const CHAT_HIDE_DELAY: Duration = Duration::from_secs(10);
const MULTI_BREAK_MAX_BLOCKS: usize = 12;
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(2);

fn ema_ms(previous: f32, sample_ms: f32) -> f32 {
    if previous <= 0.001 {
        sample_ms
    } else {
        previous * 0.82 + sample_ms * 0.18
    }
}

#[derive(Clone, Copy, Debug)]
struct LodLogSummary {
    full: usize,
    sides: usize,
    surface: usize,
    step_min: i32,
    step_max: i32,
    step_1: usize,
    step_2_4: usize,
    step_5_16: usize,
    step_17p: usize,
}

impl Default for LodLogSummary {
    fn default() -> Self {
        Self {
            full: 0,
            sides: 0,
            surface: 0,
            step_min: 0,
            step_max: 0,
            step_1: 0,
            step_2_4: 0,
            step_5_16: 0,
            step_17p: 0,
        }
    }
}

fn summarize_lod_map(map: &HashMap<(i32, i32, i32), i32>) -> LodLogSummary {
    if map.is_empty() {
        return LodLogSummary::default();
    }
    let mut out = LodLogSummary {
        step_min: i32::MAX,
        ..LodLogSummary::default()
    };
    for &packed in map.values() {
        let mode = (packed >> 16) & 0xFFFF;
        let step = (packed & 0xFFFF).max(1);
        match mode {
            0 => out.full += 1,
            1 => out.sides += 1,
            _ => out.surface += 1,
        }
        out.step_min = out.step_min.min(step);
        out.step_max = out.step_max.max(step);
        if step <= 1 {
            out.step_1 += 1;
        } else if step <= 4 {
            out.step_2_4 += 1;
        } else if step <= 16 {
            out.step_5_16 += 1;
        } else {
            out.step_17p += 1;
        }
    }
    if out.step_min == i32::MAX {
        out.step_min = 0;
    }
    out
}

fn collect_break_targets(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    target_block: IVec3,
    target_id: i8,
    multi_break: bool,
) -> Vec<IVec3> {
    let mut out = vec![target_block];
    if !multi_break {
        return out;
    }

    let mut nearby = Vec::<(i32, IVec3)>::new();
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let candidate = IVec3::new(
                    target_block.x + dx,
                    target_block.y + dy,
                    target_block.z + dz,
                );
                let id = block_id_with_edits(
                    world_gen,
                    edited_blocks,
                    candidate.x,
                    candidate.y,
                    candidate.z,
                );
                if id != target_id {
                    continue;
                }
                let dist_sq = dx * dx + dy * dy + dz * dz;
                nearby.push((dist_sq, candidate));
            }
        }
    }
    nearby.sort_by_key(|entry| entry.0);
    for (_, candidate) in nearby
        .into_iter()
        .take(MULTI_BREAK_MAX_BLOCKS.saturating_sub(1))
    {
        out.push(candidate);
    }
    out
}

fn current_saved_player_state(
    player: &PlayerState,
    camera: &Camera,
    selected_hotbar_slot: u8,
    fly_mode: bool,
) -> SavedPlayerState {
    SavedPlayerState {
        position: player.position.to_array(),
        velocity: player.velocity.to_array(),
        grounded: player.grounded,
        camera_forward: camera.forward.to_array(),
        camera_up: camera.up.to_array(),
        selected_hotbar_slot,
        fly_mode,
    }
}

fn save_runtime_state(
    save_io: &SaveIo,
    player: &PlayerState,
    camera: &Camera,
    selected_hotbar_slot: u8,
    fly_mode: bool,
    inventory: &InventoryState,
    edited_blocks: &EditedBlocks,
    include_edits: bool,
) -> std::io::Result<usize> {
    let player_state = current_saved_player_state(player, camera, selected_hotbar_slot, fly_mode);
    save_io.save_player(&player_state)?;
    save_io.save_inventory(&inventory.snapshot())?;
    if include_edits {
        let edits = edited_blocks.read().unwrap().snapshot_entries();
        let count = edits.len();
        save_io.save_edited_blocks(&edits)?;
        Ok(count)
    } else {
        Ok(0)
    }
}

fn main() {
    match init_logger() {
        Ok(path) => {
            eprintln!("NyraCraft log: {}", path.display());
        }
        Err(err) => {
            eprintln!("failed to initialize logger: {err}");
        }
    }
    install_panic_hook();
    log_info("startup: NyraCraft launching");

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
    let runtime_blocks = Arc::new(RwLock::new(default_blocks(tiles_x)));
    let mut core_ids = core_block_ids();
    let content_reload_epoch = Arc::new(AtomicU64::new(1));
    let block_index = build_block_index(tiles_x);
    if cfg!(debug_assertions) {
        eprintln!("Indexed {} block types:", block_index.len());
        log_info(format!(
            "startup: indexed {} block types",
            block_index.len()
        ));
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
    log_info(format!(
        "startup: seed_input={} resolved_seed={} mode={} world_id={}",
        seed_input,
        world_gen.seed,
        world_gen.mode_name(),
        world_gen.world_id
    ));

    let edited_blocks: EditedBlocks = new_edited_blocks();
    let leaf_decay_queue: LeafDecayQueue = new_leaf_decay_queue();
    let dirty_chunks: DirtyChunks = Arc::new(Mutex::new(HashMap::new()));
    let edited_chunk_ranges: EditedChunkRanges = Arc::new(Mutex::new(HashMap::new()));
    let save_io = SaveIo::new(&world_gen);
    let loaded_save = save_io.load();
    for warning in &loaded_save.warnings {
        log_warn(format!("save-load: {warning}"));
    }
    let loaded_player_state = loaded_save.player;
    let loaded_inventory_snapshot = loaded_save.inventory;
    let loaded_edited_entries: Vec<EditedBlockEntry> = loaded_save.edited_blocks;
    if !loaded_edited_entries.is_empty() {
        edited_blocks
            .write()
            .unwrap()
            .replace_entries(&loaded_edited_entries);
        let (edit_count, dirty_chunk_count) =
            restore_loaded_edit_metadata(&edited_blocks, &edited_chunk_ranges, &dirty_chunks);
        log_info(format!(
            "save-load: restored {edit_count} edited blocks across {dirty_chunk_count} chunks from {}",
            save_io.save_dir().display()
        ));
    }

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
    let mut player_config = PlayerConfig {
        height: 1.9,
        radius: 0.35,
        eye_height: 1.62,
        move_speed,
        sprint_multiplier: 1.65,
        sneak_multiplier: 0.45,
        jump_speed: 7.0,
        gravity: -18.0,
        acceleration: 15.0,
        friction: 20.0,
        air_control: 0.3,
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
    let mut pause_menu_open = false;
    let mut pause_stream = false;
    let mut pause_render = false;
    let mut debug_ui = false;
    let mut fly_mode = false;
    let mut force_reload = false;
    let mut selected_hotbar_slot: u8 = 0;
    let mut inventory = InventoryState::new();
    if let Some(saved_player) = loaded_player_state {
        player.position = Vec3::from_array(saved_player.position);
        player.velocity = Vec3::from_array(saved_player.velocity);
        player.grounded = saved_player.grounded;
        camera.forward = Vec3::from_array(saved_player.camera_forward).normalize_or_zero();
        if camera.forward.length_squared() == 0.0 {
            camera.forward = Vec3::new(0.0, 0.0, -1.0);
        }
        camera.up = Vec3::from_array(saved_player.camera_up).normalize_or_zero();
        if camera.up.length_squared() == 0.0 {
            camera.up = Vec3::Y;
        }
        selected_hotbar_slot = saved_player
            .selected_hotbar_slot
            .min((HOTBAR_SLOTS.saturating_sub(1)) as u8);
        fly_mode = saved_player.fly_mode;
        let restored_forward = camera.forward.normalize();
        yaw = restored_forward.z.atan2(restored_forward.x);
        pitch = restored_forward.y.asin();
        log_info("save-load: restored player state");
    }
    if let Some(snapshot) = loaded_inventory_snapshot.as_ref() {
        inventory.apply_snapshot(snapshot);
        log_info("save-load: restored inventory");
    }
    camera.position = player.position + Vec3::new(0.0, player_config.eye_height, 0.0);
    let mut dropped_items: Vec<DroppedItem> = Vec::new();
    let mut command_console = CommandConsoleState::default();
    let keybind_overlay_lines = keybind_lines();
    let mut keybind_overlay_visible = false;
    let mut stats_overlay_lines: Vec<String> = Vec::new();
    let mut chat_visible_until = Instant::now();
    let mut last_runtime_log = Instant::now();
    let mut last_autosave = Instant::now();
    let mut edits_save_dirty = false;
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
    let blocks_gen = Arc::clone(&runtime_blocks);
    let gen_thread = world_gen.clone();
    let worker_count = std::thread::available_parallelism()
        // Keep several cores free for render/input/OS and avoid startup/load saturation.
        .map(|n| (n.get() / 2).clamp(1, 6))
        .unwrap_or(1);
    log_info(format!("startup: worker threads = {worker_count}"));
    for worker_idx in 0..worker_count {
        let request_queue = Arc::clone(&request_queue);
        let tx_res = tx_res.clone();
        let tx_cache = tx_cache.clone();
        let blocks_gen = Arc::clone(&blocks_gen);
        let content_reload_epoch = Arc::clone(&content_reload_epoch);
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
                    let task_epoch = content_reload_epoch.load(Ordering::Acquire);
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
                        let _ =
                            tx_res.send(cached.into_worker_result(coord, step, mode, task_epoch));
                        continue;
                    }

                    let edit_range = if dirty_rev.is_some() {
                        let recent_range =
                            edited_chunk_ranges.lock().unwrap().get(&coord_key).copied();
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
                    let blocks_for_task = {
                        let guard = blocks_gen.read().unwrap();
                        guard.clone()
                    };
                    let mesh = generate_chunk_mesh(
                        coord,
                        &blocks_for_task,
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
                        let _ = tx_res.send(WorkerResult::Packed {
                            packed,
                            dirty_rev,
                            epoch: task_epoch,
                        });
                    } else {
                        let _ = tx_res.send(WorkerResult::Raw {
                            mesh,
                            dirty_rev,
                            epoch: task_epoch,
                        });
                    }
                }
            });
    }

    let base_render_radius = 384;
    let max_render_radius = 768;
    let base_draw_radius = 160;
    let lod_near_radius = 16;
    let lod_mid_radius = 32;
    let request_budget = 56;
    let max_inflight = 384usize;
    let surface_depth_chunks = 4;
    let initial_burst_radius = 2;
    let loaded_chunk_cap = 36000usize; // trim residency so render doesn't drown in far chunks
    let mesh_memory_cap_mb = 3072usize; // 3 GB cap
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
        log_warn(format!(
            "startup: pregen span capped from {}x{} to {}x{}",
            requested_pregen_span_chunks,
            requested_pregen_span_chunks,
            pregen_radius_chunks * 2 + 1,
            pregen_radius_chunks * 2 + 1,
        ));
    }
    let pregen_budget_per_tick = 24;
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
    let mut deferred_apply_results: VecDeque<WorkerResult> = VecDeque::new();
    let mut column_height_cache: HashMap<(i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;
    let mut stream_unload_counter: u8 = 0;
    let mut emergency_budget: i32 = 16;
    let mut looked_block: Option<(IVec3, i8)> = None;
    let mut looked_hit: Option<(IVec3, i8, IVec3)> = None;
    let mut adaptive_request_budget = request_budget;
    let mut adaptive_pregen_budget = pregen_budget_per_tick;
    let mut adaptive_max_apply_per_tick: usize = 12;
    let mut adaptive_max_rebuilds_per_tick: usize = 2;
    let mut adaptive_draw_radius_cap = base_draw_radius;
    let mut last_camera_pos = camera.position;
    let mut last_camera_forward = camera.forward;
    let mut last_visibility_refresh = Instant::now();

    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            elwt.set_control_flow(ControlFlow::WaitUntil(next_tick));

            while let Ok(report) = rx_thread_report.try_recv() {
                if !thread_reports.iter().any(|existing| existing == &report) {
                    log_info(format!("thread-report: {report}"));
                    thread_reports.push(report);
                    thread_reports.sort();
                }
            }

            let now = Instant::now();
            if now >= next_tick {
                if pause_menu_open {
                    last_update = now;
                    tick_accum = Duration::ZERO;
                    gpu.window().request_redraw();
                    next_tick = now + delay;
                    return;
                }

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
                    &mut player_config,
                    &is_solid,
                );
                perf.player_ms = ema_ms(
                    perf.player_ms,
                    player_phase_start.elapsed().as_secs_f32() * 1000.0,
                );

                let dropped_phase_start = Instant::now();
                update_dropped_items(
                    &mut dropped_items,
                    dt,
                    player.position,
                    player_config.height,
                    &world_gen,
                    &edited_blocks,
                    &mut inventory,
                );
                perf.dropped_ms = ema_ms(
                    perf.dropped_ms,
                    dropped_phase_start.elapsed().as_secs_f32() * 1000.0,
                );

                tick_accum = tick_accum.saturating_add(Duration::from_secs_f32(dt));
                let mut stream_ms_accum = 0.0f32;
                let mut stream_calls = 0u32;
                let mut leaf_decay_ms_accum = 0.0f32;
                let mut leaf_decay_calls = 0u32;
                let mut visibility_ms_accum = 0.0f32;
                let mut visibility_calls = 0u32;
                while tick_accum >= tick_dt {
                    tick_accum -= tick_dt;
                    if pause_stream {
                        continue;
                    }
                    let leaf_decay_phase_start = Instant::now();
                    if let Some(decayed_leaf) = tick_leaf_decay(
                        &world_gen,
                        &edited_blocks,
                        &leaf_decay_queue,
                        &edited_chunk_ranges,
                        &dirty_chunks,
                        &request_queue,
                    ) {
                        spawn_block_drop(&mut dropped_items, decayed_leaf, core_ids.leaves);
                        edits_save_dirty = true;
                    }
                    leaf_decay_ms_accum +=
                        leaf_decay_phase_start.elapsed().as_secs_f32() * 1000.0;
                    leaf_decay_calls += 1;
                    let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                    let render_radius =
                        (base_render_radius + height_chunks * 2).min(max_render_radius);
                    let draw_radius_base = (base_draw_radius + height_chunks).min(render_radius);
                    let draw_radius = draw_radius_base.min(adaptive_draw_radius_cap.max(16));
                    if force_reload {
                        deferred_apply_results.clear();
                    }
                    let stream_phase_start = Instant::now();
                    stream_tick(
                        &mut gpu,
                        player.position,
                        camera.forward,
                        &mut current_chunk,
                        &mut ring_r,
                        &mut ring_i,
                        &mut stream_unload_counter,
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
                    let camera_changed = (camera.position - last_camera_pos).length_squared() > 0.16
                        || camera.forward.dot(last_camera_forward) < 0.9992;
                    let visibility_refresh_due =
                        last_visibility_refresh.elapsed() >= Duration::from_millis(66);
                    if visibility_refresh_due
                        && (camera_changed || !requested.is_empty() || pregen_active)
                    {
                        let visibility_phase_start = Instant::now();
                        gpu.update_visible(&camera, draw_radius);
                        visibility_ms_accum +=
                            visibility_phase_start.elapsed().as_secs_f32() * 1000.0;
                        visibility_calls += 1;
                        last_camera_pos = camera.position;
                        last_camera_forward = camera.forward;
                        last_visibility_refresh = Instant::now();
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
                if leaf_decay_calls > 0 {
                    perf.leaf_decay_ms =
                        ema_ms(perf.leaf_decay_ms, leaf_decay_ms_accum / leaf_decay_calls as f32);
                }
                if visibility_calls > 0 {
                    perf.visibility_ms = ema_ms(
                        perf.visibility_ms,
                        visibility_ms_accum / visibility_calls as f32,
                    );
                }

                let active_break_strength = inventory
                    .selected_hotbar_break_strength(selected_hotbar_slot)
                    .unwrap_or(HAND_BREAK_STRENGTH);
                let active_break_item_id = inventory.selected_hotbar_item_id(selected_hotbar_slot);
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
                            if let Some(break_time) = block_break_time_with_item_seconds(
                                current_target_id,
                                active_break_item_id,
                                active_break_strength,
                            ) {
                                mining_progress += dt;
                                let stage = block_break_stage(mining_progress, break_time);
                                mining_overlay = Some((target_block, stage));
                                if mining_progress >= break_time {
                                    let break_targets = collect_break_targets(
                                        &world_gen,
                                        &edited_blocks,
                                        target_block,
                                        current_target_id,
                                        input.sneak && !input.fly_mode,
                                    );
                                    let edit_result = break_blocks_batch(
                                        &break_targets,
                                        &world_gen,
                                        &edited_blocks,
                                        &leaf_decay_queue,
                                        &edited_chunk_ranges,
                                        &dirty_chunks,
                                        &request_queue,
                                    );

                                    if !edit_result.broke.is_empty() {
                                        edits_save_dirty = true;
                                        for (broken_pos, broken_id) in edit_result.broke.iter().copied() {
                                            spawn_block_drop(&mut dropped_items, broken_pos, broken_id);
                                            let _ = inventory.apply_selected_hotbar_durability_loss(
                                                selected_hotbar_slot,
                                                1,
                                            );
                                        }
                                        mining_target = None;
                                        mining_progress = 0.0;
                                        mining_overlay = None;
                                        if !pause_stream {
                                            let _ = apply_stream_results(
                                                &mut gpu,
                                                &rx_res,
                                                &mut deferred_apply_results,
                                                content_reload_epoch.load(Ordering::Acquire),
                                                &dirty_chunks,
                                                1,
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
                                                32,
                                                adaptive_max_rebuilds_per_tick.max(5),
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

                let mut apply_debug_tick = ApplyDebugStats::default();
                let (dirty_pending_debug, apply_budget_debug, rebuild_budget_debug) =
                    if !pause_stream {
                        let dirty_pending = dirty_chunks
                            .lock()
                            .unwrap()
                            .values()
                            .filter(|state| **state > 0)
                            .count();
                        let apply_budget = if dirty_pending > 0 {
                            let burst_floor = if dirty_pending > 48 {
                                24
                            } else if dirty_pending > 12 {
                                18
                            } else {
                                12
                            };
                            adaptive_max_apply_per_tick.max(burst_floor)
                        } else if !requested.is_empty() {
                            (adaptive_max_apply_per_tick / 2).max(3)
                        } else {
                            1
                        };
                        let rebuild_budget = if dirty_pending > 0 {
                            let extra = if dirty_pending > 48 { 2 } else { 1 };
                            (adaptive_max_rebuilds_per_tick + extra).min(4)
                        } else {
                            adaptive_max_rebuilds_per_tick.max(1)
                        };
                        let apply_phase_start = Instant::now();
                        apply_debug_tick = apply_stream_results(
                            &mut gpu,
                            &rx_res,
                            &mut deferred_apply_results,
                            content_reload_epoch.load(Ordering::Acquire),
                            &dirty_chunks,
                            dirty_pending,
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

                    if tps_value < 18 || fps_value < 45 {
                        adaptive_request_budget = (adaptive_request_budget * 82 / 100).max(28);
                        adaptive_pregen_budget = (adaptive_pregen_budget * 75 / 100).max(8);
                        adaptive_max_apply_per_tick =
                            adaptive_max_apply_per_tick.saturating_sub(2).max(6);
                        adaptive_max_rebuilds_per_tick = adaptive_max_rebuilds_per_tick.saturating_sub(1).max(1);
                        adaptive_draw_radius_cap = (adaptive_draw_radius_cap - 8).max(24);
                    } else if tps_value >= 20 && fps_value > 72 {
                        adaptive_request_budget = (adaptive_request_budget + 6).min(request_budget);
                        adaptive_pregen_budget =
                            (adaptive_pregen_budget + 4).min(pregen_budget_per_tick);
                        adaptive_max_apply_per_tick = (adaptive_max_apply_per_tick + 2).min(20);
                        adaptive_max_rebuilds_per_tick =
                            (adaptive_max_rebuilds_per_tick + 1).min(3);
                        adaptive_draw_radius_cap =
                            (adaptive_draw_radius_cap + 2).min(base_draw_radius + 16);
                    }
                }

                let stats_overlay_visible = debug_ui || pregen_active;
                if stats_overlay_visible && last_stats_print.elapsed() >= Duration::from_millis(500) {
                    let stats = gpu.stats();
                    let req_stats = request_queue_stats(&request_queue);
                    stats_overlay_lines = build_stats_lines(
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
                        active_break_item_id,
                        active_break_strength,
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
                } else if !stats_overlay_visible {
                    stats_overlay_lines.clear();
                }

                if last_runtime_log.elapsed() >= Duration::from_secs(1) {
                    let req_stats = request_queue_stats(&request_queue);
                    let gpu_stats = gpu.stats();
                    let leaf_stats = leaf_decay_stats(&leaf_decay_queue);
                    let loaded_lod = summarize_lod_map(&loaded);
                    let requested_lod = summarize_lod_map(&requested);
                    let tick_ms = perf.tick_cpu_ms.max(0.0001);
                    let mut bottleneck_label = "render";
                    let mut bottleneck_ms = perf.render_cpu_ms;
                    for (label, ms) in [
                        ("stream", perf.stream_ms),
                        ("apply", perf.apply_ms),
                        ("player", perf.player_ms),
                        ("drops", perf.dropped_ms),
                        ("leaf", perf.leaf_decay_ms),
                        ("vis", perf.visibility_ms),
                        ("mining", perf.mining_ms),
                        ("other", perf.untracked_ms),
                    ] {
                        if ms > bottleneck_ms {
                            bottleneck_label = label;
                            bottleneck_ms = ms;
                        }
                    }
                    let render_pct = (perf.render_cpu_ms / tick_ms * 100.0).max(0.0);
                    let stream_pct = (perf.stream_ms / tick_ms * 100.0).max(0.0);
                    let apply_pct = (perf.apply_ms / tick_ms * 100.0).max(0.0);
                    let player_pct = (perf.player_ms / tick_ms * 100.0).max(0.0);
                    let dropped_pct = (perf.dropped_ms / tick_ms * 100.0).max(0.0);
                    let leaf_pct = (perf.leaf_decay_ms / tick_ms * 100.0).max(0.0);
                    let visibility_pct = (perf.visibility_ms / tick_ms * 100.0).max(0.0);
                    let mining_pct = (perf.mining_ms / tick_ms * 100.0).max(0.0);
                    let other_pct = (perf.untracked_ms / tick_ms * 100.0).max(0.0);
                    let bottleneck_pct = (bottleneck_ms / tick_ms * 100.0).max(0.0);
                    let render_ms = perf.render_cpu_ms;
                    let stream_ms = perf.stream_ms;
                    let apply_ms = perf.apply_ms;
                    let player_ms = perf.player_ms;
                    let dropped_ms = perf.dropped_ms;
                    let leaf_ms = perf.leaf_decay_ms;
                    let visibility_ms = perf.visibility_ms;
                    let mining_ms = perf.mining_ms;
                    let other_ms = perf.untracked_ms;
                    log_info(format!(
                        "runtime-bottleneck: fps={fps_value} tps={tps_value} bottleneck={bottleneck_label}({bottleneck_ms:.2}ms,{bottleneck_pct:.0}%tick) tick_ms={tick_ms:.2} phases_ms[r/s/a/p/d/l/v/m/o]={render_ms:.2}/{stream_ms:.2}/{apply_ms:.2}/{player_ms:.2}/{dropped_ms:.2}/{leaf_ms:.2}/{visibility_ms:.2}/{mining_ms:.2}/{other_ms:.2} phases_pct[r/s/a/p/d/l/v/m/o]={render_pct:.0}/{stream_pct:.0}/{apply_pct:.0}/{player_pct:.0}/{dropped_pct:.0}/{leaf_pct:.0}/{visibility_pct:.0}/{mining_pct:.0}/{other_pct:.0} chunks[loaded/requested/dirty]={loaded_len}/{requested_len}/{dirty_pending_debug} req[edit/near/far/inflight]={req_edit}/{req_near}/{req_far}/{req_inflight} stream_cursor[r/i]={stream_ring_r}/{stream_ring_i} deferred={deferred_len} lod_loaded[f/s/o]={lod_loaded_full}/{lod_loaded_sides}/{lod_loaded_surface} lod_req[f/s/o]={lod_req_full}/{lod_req_sides}/{lod_req_surface} lod_step_loaded[min/max/1/2_4/5_16/17p]={lod_loaded_step_min}/{lod_loaded_step_max}/{lod_loaded_step_1}/{lod_loaded_step_2_4}/{lod_loaded_step_5_16}/{lod_loaded_step_17p} lod_step_req[min/max/1/2_4/5_16/17p]={lod_req_step_min}/{lod_req_step_max}/{lod_req_step_1}/{lod_req_step_2_4}/{lod_req_step_5_16}/{lod_req_step_17p} apply_dbg[ok/raw/packed/skip_epoch/skip_rev/defer_push/defer_pop/fast_scan/fast_ok/rebuild]={apply_ok}/{apply_raw}/{apply_packed}/{apply_skip_epoch}/{apply_skip_rev}/{apply_defer_push}/{apply_defer_pop}/{apply_fast_scan}/{apply_fast_ok}/{apply_rebuild_budget} leaf_q[check/break/check_p/break_p/tick]={leaf_check}/{leaf_break}/{leaf_check_pending}/{leaf_break_pending}/{leaf_tick} gpu[vis_supers={gpu_vis_supers}/{gpu_total_supers} vis_draw={gpu_vis_draw}/{gpu_total_draw} vis_idx={gpu_vis_idx} raw_idx={gpu_vis_raw_idx} packed_idx={gpu_vis_packed_idx} dirty_supers={gpu_dirty_supers} pending_upd={gpu_pending_updates} pending_q={gpu_pending_queue}] budgets[req/pregen/apply/rebuild/apply_now/rebuild_now/draw_cap]={adaptive_request_budget}/{adaptive_pregen_budget}/{adaptive_max_apply_per_tick}/{adaptive_max_rebuilds_per_tick}/{apply_budget_debug}/{rebuild_budget_debug}/{adaptive_draw_radius_cap} flags[pause_stream/pause_render/pregen]={pause_stream}/{pause_render}/{pregen_active}",
                        loaded_len = loaded.len(),
                        requested_len = requested.len(),
                        req_edit = req_stats.edit,
                        req_near = req_stats.near,
                        req_far = req_stats.far,
                        req_inflight = req_stats.inflight_chunks,
                        stream_ring_r = ring_r,
                        stream_ring_i = ring_i,
                        deferred_len = deferred_apply_results.len(),
                        leaf_check = leaf_stats.check_queue,
                        leaf_break = leaf_stats.break_queue,
                        leaf_check_pending = leaf_stats.check_pending,
                        leaf_break_pending = leaf_stats.break_pending,
                        leaf_tick = leaf_stats.tick,
                        gpu_vis_supers = gpu_stats.visible_supers,
                        gpu_total_supers = gpu_stats.super_chunks,
                        gpu_vis_draw = gpu_stats.visible_draw_calls_est,
                        gpu_total_draw = gpu_stats.total_draw_calls_est,
                        gpu_vis_idx = gpu_stats.visible_indices,
                        gpu_vis_raw_idx = gpu_stats.visible_raw_indices,
                        gpu_vis_packed_idx = gpu_stats.visible_packed_indices,
                        gpu_dirty_supers = gpu_stats.dirty_supers,
                        gpu_pending_updates = gpu_stats.pending_updates,
                        gpu_pending_queue = gpu_stats.pending_queue,
                        lod_loaded_full = loaded_lod.full,
                        lod_loaded_sides = loaded_lod.sides,
                        lod_loaded_surface = loaded_lod.surface,
                        lod_req_full = requested_lod.full,
                        lod_req_sides = requested_lod.sides,
                        lod_req_surface = requested_lod.surface,
                        lod_loaded_step_min = loaded_lod.step_min,
                        lod_loaded_step_max = loaded_lod.step_max,
                        lod_loaded_step_1 = loaded_lod.step_1,
                        lod_loaded_step_2_4 = loaded_lod.step_2_4,
                        lod_loaded_step_5_16 = loaded_lod.step_5_16,
                        lod_loaded_step_17p = loaded_lod.step_17p,
                        lod_req_step_min = requested_lod.step_min,
                        lod_req_step_max = requested_lod.step_max,
                        lod_req_step_1 = requested_lod.step_1,
                        lod_req_step_2_4 = requested_lod.step_2_4,
                        lod_req_step_5_16 = requested_lod.step_5_16,
                        lod_req_step_17p = requested_lod.step_17p,
                        apply_ok = apply_debug_tick.applied_total,
                        apply_raw = apply_debug_tick.applied_raw,
                        apply_packed = apply_debug_tick.applied_packed,
                        apply_skip_epoch = apply_debug_tick.skipped_epoch,
                        apply_skip_rev = apply_debug_tick.skipped_dirty_rev,
                        apply_defer_push = apply_debug_tick.deferred_pushed,
                        apply_defer_pop = apply_debug_tick.deferred_popped,
                        apply_fast_scan = apply_debug_tick.fast_lane_scanned,
                        apply_fast_ok = apply_debug_tick.fast_lane_applied,
                        apply_rebuild_budget = apply_debug_tick.rebuild_budget,
                    ));
                    last_runtime_log = Instant::now();
                }

                if last_autosave.elapsed() >= AUTOSAVE_INTERVAL {
                    match save_runtime_state(
                        &save_io,
                        &player,
                        &camera,
                        selected_hotbar_slot,
                        fly_mode,
                        &inventory,
                        &edited_blocks,
                        edits_save_dirty,
                    ) {
                        Ok(saved_edit_count) => {
                            if edits_save_dirty {
                                log_info(format!(
                                    "save: autosaved {} edited blocks to {}",
                                    saved_edit_count,
                                    save_io.save_dir().display()
                                ));
                                edits_save_dirty = false;
                            }
                        }
                        Err(err) => {
                            log_warn(format!("save: autosave failed: {err}"));
                        }
                    }
                    last_autosave = Instant::now();
                }

                if !pause_render || pause_menu_open {
                    gpu.window().request_redraw();
                }
                let tick_ms_sample = tick_start.elapsed().as_secs_f32() * 1000.0;
                let measured_sum_ms = perf.render_cpu_ms
                    + perf.stream_ms
                    + perf.apply_ms
                    + perf.player_ms
                    + perf.dropped_ms
                    + perf.leaf_decay_ms
                    + perf.visibility_ms
                    + perf.mining_ms;
                let untracked_ms = (tick_ms_sample - measured_sum_ms).max(0.0);
                perf.untracked_ms = ema_ms(perf.untracked_ms, untracked_ms);
                perf.tick_cpu_ms = ema_ms(perf.tick_cpu_ms, tick_ms_sample);
                next_tick = now + delay;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            if pause_menu_open || (!pause_render && !pregen_active) {
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
                let chat_recently_visible = Instant::now() < chat_visible_until;
                let stats_overlay_visible = debug_ui || pregen_active;
                let draw_world = !pause_menu_open && !pregen_active;
                gpu.set_selection(if draw_world {
                    looked_block.map(|(coord, _)| coord)
                } else {
                    None
                });
                let render_phase_start = Instant::now();
                gpu.render(
                    &camera,
                    render_time,
                    draw_world,
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
                    command_console.open,
                    chat_recently_visible,
                    keybind_overlay_visible,
                    keybind_overlay_lines,
                    stats_overlay_visible,
                    &stats_overlay_lines,
                    &command_console.input,
                    &command_console.chat_lines,
                    &dropped_render_items,
                    pause_menu_open,
                    ui_cursor_pos,
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
                    if execute_and_close_command_console(
                        &mut command_console,
                        &mut inventory,
                        inventory_open,
                        gpu.window(),
                        &mut mouse_enabled,
                        &mut last_cursor_pos,
                    ) {
                        chat_visible_until = Instant::now() + CHAT_HIDE_DELAY;
                    }
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
                        if execute_and_close_command_console(
                            &mut command_console,
                            &mut inventory,
                            inventory_open,
                            gpu.window(),
                            &mut mouse_enabled,
                            &mut last_cursor_pos,
                        ) {
                            chat_visible_until = Instant::now() + CHAT_HIDE_DELAY;
                        }
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
                            if execute_and_close_command_console(
                                &mut command_console,
                                &mut inventory,
                                inventory_open,
                                gpu.window(),
                                &mut mouse_enabled,
                                &mut last_cursor_pos,
                            ) {
                                chat_visible_until = Instant::now() + CHAT_HIDE_DELAY;
                            }
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

            if pressed && key == KeyCode::Escape {
                if pause_menu_open {
                    pause_menu_open = false;
                    try_enable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                    clear_movement_input(&mut input);
                    mine_left_down = false;
                    mining_target = None;
                    mining_progress = 0.0;
                    mining_overlay = None;
                    inventory_left_mouse_down = false;
                    inventory_right_mouse_down = false;
                    inventory_drag_last_slot = None;
                    gpu.window().set_ime_allowed(false);
                    log_info("ui: pause menu closed");
                } else {
                    pause_menu_open = true;
                    inventory.close();
                    command_console.open = false;
                    command_console.input.clear();
                    let cached_cursor = last_cursor_pos.map(|(x, y)| (x as f32, y as f32));
                    disable_mouse_look(gpu.window(), &mut mouse_enabled, &mut last_cursor_pos);
                    ui_cursor_pos = cached_cursor;
                    clear_movement_input(&mut input);
                    mine_left_down = false;
                    mining_target = None;
                    mining_progress = 0.0;
                    mining_overlay = None;
                    inventory_left_mouse_down = false;
                    inventory_right_mouse_down = false;
                    inventory_drag_last_slot = None;
                    gpu.window().set_ime_allowed(false);
                    log_info("ui: pause menu opened");
                }
                gpu.window().request_redraw();
                return;
            }
            if pause_menu_open {
                return;
            }

            if key == KeyCode::F1 {
                if pressed {
                    f1_down = true;
                    f1_combo_used = false;
                } else {
                    if !f1_combo_used {
                        keybind_overlay_visible = !keybind_overlay_visible;
                        log_info(format!(
                            "ui: keybind overlay {}",
                            if keybind_overlay_visible {
                                "enabled"
                            } else {
                                "disabled"
                            }
                        ));
                    }
                    f1_down = false;
                    f1_combo_used = false;
                }
                return;
            }
            if pressed && key == KeyCode::F3 {
                debug_ui = !debug_ui;
                if debug_ui {
                    last_stats_print = Instant::now() - Duration::from_secs(1);
                } else {
                    stats_overlay_lines.clear();
                }
                log_info(format!(
                    "ui: stats overlay {} via F3",
                    if debug_ui { "enabled" } else { "disabled" }
                ));
                return;
            }
            if !inventory.open && is_console_open_shortcut(&event) {
                let opens_with_slash = event
                    .text
                    .as_ref()
                    .is_some_and(|text| text.as_str() == "/")
                    || matches!(key, KeyCode::Slash);
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
                command_console.input = if opens_with_slash {
                    "/".to_string()
                } else {
                    String::new()
                };
                gpu.window().set_ime_allowed(true);
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
                    KeyCode::KeyF => {
                        debug_faces = !debug_faces;
                        log_info(format!(
                            "ui: face debug {}",
                            if debug_faces { "enabled" } else { "disabled" }
                        ));
                    }
                    KeyCode::KeyW => {
                        debug_chunks = !debug_chunks;
                        log_info(format!(
                            "ui: chunk wireframe {}",
                            if debug_chunks { "enabled" } else { "disabled" }
                        ));
                    }
                    KeyCode::KeyP => {
                        pause_stream = !pause_stream;
                        log_info(format!(
                            "runtime: stream {}",
                            if pause_stream { "paused" } else { "resumed" }
                        ));
                    }
                    KeyCode::KeyV => {
                        pause_render = !pause_render;
                        log_info(format!(
                            "runtime: render {}",
                            if pause_render { "paused" } else { "resumed" }
                        ));
                    }
                    KeyCode::KeyD => {
                        debug_ui = !debug_ui;
                        if debug_ui {
                            last_stats_print = Instant::now() - Duration::from_secs(1);
                        } else {
                            stats_overlay_lines.clear();
                        }
                        log_info(format!(
                            "ui: stats overlay {} via F1+D",
                            if debug_ui { "enabled" } else { "disabled" }
                        ));
                    }
                    KeyCode::KeyR => {
                        // Phase 1: move to a transient epoch so in-flight work becomes stale.
                        let next_epoch = content_reload_epoch.fetch_add(1, Ordering::AcqRel) + 1;
                        log_info("runtime: hot reload requested (textures/content)");
                        generate_texture_atlases();
                        reload_registry(DEFAULT_TILES_X);
                        core_ids = core_block_ids();
                        reload_recipes();
                        {
                            let mut shared_blocks = runtime_blocks.write().unwrap();
                            *shared_blocks = default_blocks(DEFAULT_TILES_X);
                        }
                        gpu.reload_textures(Some(TextureAtlas {
                            path: "src/texturing/atlas_output/atls_blocks.png".to_string(),
                            tile_size: 16,
                        }));
                        reload_cpu_item_atlas_cache();
                        let _ = fs::remove_dir_all(&cache_dir);
                        let _ = fs::create_dir_all(&cache_dir);
                        // Phase 2: publish the final epoch for freshly generated work.
                        content_reload_epoch.store(next_epoch + 1, Ordering::Release);
                        force_reload = true;
                        log_info("runtime: hot reload complete");
                    }
                    KeyCode::KeyM => {
                        fly_mode = !fly_mode;
                        input.sneak = false;
                        input.move_down = false;
                        log_info(format!(
                            "player: fly mode {}",
                            if fly_mode { "enabled" } else { "disabled" }
                        ));
                    }
                    KeyCode::KeyX => {
                        let window = gpu.window();
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                            log_info("ui: fullscreen disabled");
                        } else {
                            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                            log_info("ui: fullscreen enabled");
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
            if command_console.open || pause_menu_open {
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
            if pause_menu_open {
                if pressed && matches!(button, MouseButton::Left) {
                    let size = gpu.window().inner_size();
                    let cursor = ui_cursor_pos.or_else(|| {
                        last_cursor_pos.map(|(x, y)| (x as f32, y as f32))
                    });
                    if let Some((mx, my)) = cursor {
                        match hit_test_pause_menu_button(size.width, size.height, mx, my) {
                            Some(PauseMenuButton::ReturnToGame) => {
                                pause_menu_open = false;
                                try_enable_mouse_look(
                                    gpu.window(),
                                    &mut mouse_enabled,
                                    &mut last_cursor_pos,
                                );
                                clear_movement_input(&mut input);
                                mine_left_down = false;
                                mining_target = None;
                                mining_progress = 0.0;
                                mining_overlay = None;
                                inventory_left_mouse_down = false;
                                inventory_right_mouse_down = false;
                                inventory_drag_last_slot = None;
                                gpu.window().set_ime_allowed(false);
                                log_info("ui: pause menu closed");
                            }
                            Some(PauseMenuButton::Quit) => {
                                log_info("shutdown: quit selected from pause menu");
                                elwt.exit();
                            }
                            None => {}
                        }
                    }
                    gpu.window().request_redraw();
                }
                return;
            }
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
                && looked_hit.is_some_and(|(_, block_id, _)| block_id == core_ids.crafting_table)
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
            for (broken_pos, broken_id) in edit_result.broke.iter().copied() {
                spawn_block_drop(&mut dropped_items, broken_pos, broken_id);
            }
            for (placed_pos, _placed_id) in edit_result.placed.iter().copied() {
                let _ = inventory.consume_selected_hotbar(selected_hotbar_slot, 1);
                nudge_items_from_placed_block(
                    &mut dropped_items,
                    placed_pos,
                    &world_gen,
                    &edited_blocks,
                );
            }
            let edited_world = edit_result.edited_world();
            if edited_world {
                edits_save_dirty = true;
            }
            if edited_world && !pause_stream {
                let _ = apply_stream_results(
                    &mut gpu,
                    &rx_res,
                    &mut deferred_apply_results,
                    content_reload_epoch.load(Ordering::Acquire),
                    &dirty_chunks,
                    1,
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
                    32,
                    adaptive_max_rebuilds_per_tick.max(5),
                    true,
                );
            }
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            if !mouse_enabled || inventory.open || command_console.open || pause_menu_open {
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
            if pause_menu_open {
                last_cursor_pos = Some((position.x, position.y));
                gpu.window().request_redraw();
                return;
            }
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
            if !inventory.open && !command_console.open && !pause_menu_open {
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
        Event::LoopExiting => match save_runtime_state(
            &save_io,
            &player,
            &camera,
            selected_hotbar_slot,
            fly_mode,
            &inventory,
            &edited_blocks,
            true,
        ) {
            Ok(saved_edit_count) => {
                log_info(format!(
                    "save: final flush complete (edited_blocks={saved_edit_count})"
                ));
            }
            Err(err) => {
                log_warn(format!("save: final flush failed: {err}"));
            }
        },
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            log_info("shutdown: close requested");
            elwt.exit();
        }
        _ => {}
    });
}
