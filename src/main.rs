use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::cmp::Ordering;
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
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
use render::mesh::ChunkVertex;
use render::Gpu;
use render::{CubeStyle, TextureAtlas};
use render::GpuStats;
use world::CHUNK_SIZE;
use world::blocks::{
    DEFAULT_TILES_X, block_item_id_by_id, block_name_by_id, build_block_index, default_blocks,
};
use world::mesher::{MeshData, MeshMode, generate_chunk_mesh};
use world::worldgen::{TreeSpec, WorldGen, WORLD_HALF_SIZE_CHUNKS};

#[derive(Clone)]
enum WorkerResult {
    Raw(MeshData),
    Packed(PackedMeshData),
}

#[derive(Clone)]
struct PackedMeshData {
    coord: IVec3,
    step: i32,
    mode: MeshMode,
    center: Vec3,
    radius: f32,
    vertices: Vec<PackedFarVertex>,
    indices: Vec<u32>,
}

enum CacheWriteMsg {
    Write {
        coord: IVec3,
        step: i32,
        mode: MeshMode,
        center: Vec3,
        radius: f32,
        vertices: Vec<PackedFarVertex>,
        indices: Vec<u32>,
    },
}

#[derive(Clone, Copy)]
struct RequestTask {
    priority: i32,
    seq: u64,
    coord: IVec3,
    step: i32,
    mode: MeshMode,
}

impl PartialEq for RequestTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl Eq for RequestTask {}

impl PartialOrd for RequestTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RequestTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct RequestQueueState {
    heap: BinaryHeap<RequestTask>,
    next_seq: u64,
    closed: bool,
}

type SharedRequestQueue = Arc<(Mutex<RequestQueueState>, Condvar)>;

const MESH_CACHE_MAGIC: &[u8; 4] = b"MSH1";
const CACHE_ENCODING_PACKED_FAR: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedFarVertex {
    position: [i16; 3],
    uv: [u16; 2],
    tile: u16,
    face: u8,
    rotation: u8,
    use_texture: u8,
    transparent_mode: u8,
    color: [u8; 4],
}

struct MeshCacheEntry {
    center: Vec3,
    radius: f32,
    vertices: Vec<PackedFarVertex>,
    indices: Vec<u32>,
}

impl MeshCacheEntry {
    fn into_worker_result(self, coord: IVec3, step: i32, mode: MeshMode) -> WorkerResult {
        WorkerResult::Packed(PackedMeshData {
            coord,
            step,
            mode,
            center: self.center,
            radius: self.radius,
            vertices: self.vertices,
            indices: self.indices,
        })
    }
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
    let block_index = build_block_index(tiles_x);
    eprintln!("Indexed {} block types:", block_index.len());
    for entry in &block_index {
        eprintln!(
            "  item_id={} block_id={} name={} texture_index={}",
            entry.item_id, entry.block_id, entry.name, entry.texture_index
        );
    }

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
        let height_cache = RefCell::new(HashMap::<(i32, i32), i32>::new());
        let tree_cache = RefCell::new(HashMap::<(i32, i32), Option<TreeSpec>>::new());
        move |x: i32, y: i32, z: i32| -> bool {
            is_solid_cached(&worldgen, &height_cache, &tree_cache, x, y, z)
        }
    };

    let (tx_res, rx_res) = mpsc::channel::<WorkerResult>();
    let (tx_cache, rx_cache) = mpsc::channel::<CacheWriteMsg>();
    let request_queue: SharedRequestQueue = Arc::new((
        Mutex::new(RequestQueueState {
            heap: BinaryHeap::new(),
            next_seq: 0,
            closed: false,
        }),
        Condvar::new(),
    ));
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
                    let entry = MeshCacheEntry {
                        center,
                        radius,
                        vertices,
                        indices,
                    };
                    write_cached_mesh(&cache_dir_writer, coord, step, mode, &entry);
                }
            }
        }
    });
    let blocks_gen = blocks.clone();
    let gen_thread = world_gen.clone();
    let worker_count = std::thread::available_parallelism()
        .map(|n| (n.get() * 2).clamp(8, 32))
        .unwrap_or(8);
    for _ in 0..worker_count {
        let request_queue = Arc::clone(&request_queue);
        let tx_res = tx_res.clone();
        let tx_cache = tx_cache.clone();
        let blocks_gen = blocks_gen.clone();
        let gen_thread = gen_thread.clone();
        let cache_dir = cache_dir.clone();
        thread::spawn(move || {
            loop {
                let Some(task) = pop_request(&request_queue) else {
                    break;
                };
                let coord = task.coord;
                let step = task.step;
                let mode = task.mode;
                if should_pack_far_lod(mode, step) {
                    if let Some(cached) = try_load_cached_mesh(&cache_dir, coord, step, mode) {
                        let _ = tx_res.send(cached.into_worker_result(coord, step, mode));
                        continue;
                    }
                }
                let mesh = generate_chunk_mesh(coord, &blocks_gen, &gen_thread, step, mode);
                if should_pack_far_lod(mode, step) {
                    let packed_vertices = pack_far_vertices(&mesh.vertices);
                    let packed_for_write = packed_vertices.clone();
                    let indices_for_write = mesh.indices.clone();
                    let packed = PackedMeshData {
                        coord: mesh.coord,
                        step: mesh.step,
                        mode: mesh.mode,
                        center: mesh.center,
                        radius: mesh.radius,
                        vertices: packed_vertices,
                        indices: mesh.indices,
                    };
                    let _ = tx_cache.send(CacheWriteMsg::Write {
                        coord: mesh.coord,
                        step: mesh.step,
                        mode: mesh.mode,
                        center: mesh.center,
                        radius: mesh.radius,
                        vertices: packed_for_write,
                        indices: indices_for_write,
                    });
                    let _ = tx_res.send(WorkerResult::Packed(packed));
                } else {
                    let _ = tx_res.send(WorkerResult::Raw(mesh));
                }
            }
        });
    }

    let base_render_radius = 128;
    let max_render_radius = 128;
    let base_draw_radius = 128;
    let lod_near_radius = 16;
    let lod_mid_radius = 64;
    let request_budget = 768;
    let max_inflight = 8192usize;
    let surface_depth_chunks = 8;
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
    let pregen_budget_per_tick = 1024;
    let pregen_total_columns = ((pregen_radius_chunks * 2 + 1) as usize).pow(2);
    let pregen_est_chunks_total = pregen_total_columns * (surface_depth_chunks as usize + 2);
    let mut pregen_active = true;
    let mut pregen_ring_r: i32 = 0;
    let mut pregen_ring_i: i32 = 0;
    let mut pregen_columns_done: usize = 0;
    let mut pregen_chunks_requested: usize = 0;
    let mut pregen_chunks_created: usize = 0;
    let mut loaded: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut requested: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;
    let mut emergency_budget: i32 = 16;
    let mut looked_block: Option<(IVec3, i8)> = None;

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
                        camera.forward,
                        &mut current_chunk,
                        &mut ring_r,
                        &mut ring_i,
                        &mut loaded,
                        &mut requested,
                        &request_queue,
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
                        pregen_center_chunk,
                        pregen_radius_chunks,
                        pregen_budget_per_tick,
                        &mut pregen_active,
                        &mut pregen_ring_r,
                        &mut pregen_ring_i,
                        &mut pregen_columns_done,
                        &mut pregen_chunks_requested,
                        &mut pregen_chunks_created,
                        loaded_chunk_cap,
                        mesh_memory_cap_mb,
                    );
                    gpu.update_visible(&camera, draw_radius);
                    looked_block =
                        pick_block(camera.position, camera.forward, &world_gen, 8.0);
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
                        gpu.window().set_title("wgpu_try");
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
            if !pause_render && !pregen_active {
                let height_chunks = (player.position.y / CHUNK_SIZE as f32).max(0.0) as i32;
                let render_radius = (base_render_radius + height_chunks * 2).min(max_render_radius);
                let draw_radius = (base_draw_radius + height_chunks).min(render_radius);
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

fn div_floor_i32(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn chunk_coord_from_block(block: IVec3) -> IVec3 {
    let half = CHUNK_SIZE / 2;
    IVec3::new(
        div_floor_i32(block.x + half, CHUNK_SIZE),
        div_floor_i32(block.y + half, CHUNK_SIZE),
        div_floor_i32(block.z + half, CHUNK_SIZE),
    )
}

fn super_chunk_coord_from_chunk(chunk: IVec3) -> IVec3 {
    const SUPER_CHUNK_SIZE: i32 = 4;
    IVec3::new(
        div_floor_i32(chunk.x, SUPER_CHUNK_SIZE),
        div_floor_i32(chunk.y, SUPER_CHUNK_SIZE),
        div_floor_i32(chunk.z, SUPER_CHUNK_SIZE),
    )
}

fn in_world_chunk_bounds(cx: i32, cz: i32) -> bool {
    cx >= -WORLD_HALF_SIZE_CHUNKS
        && cx < WORLD_HALF_SIZE_CHUNKS
        && cz >= -WORLD_HALF_SIZE_CHUNKS
        && cz < WORLD_HALF_SIZE_CHUNKS
}

fn height_at_cached(
    world_gen: &WorldGen,
    cache: &RefCell<HashMap<(i32, i32), i32>>,
    x: i32,
    z: i32,
) -> i32 {
    if let Some(&h) = cache.borrow().get(&(x, z)) {
        return h;
    }
    let h = world_gen.height_at(x, z);
    cache.borrow_mut().insert((x, z), h);
    h
}

fn tree_at_cached(
    world_gen: &WorldGen,
    height_cache: &RefCell<HashMap<(i32, i32), i32>>,
    tree_cache: &RefCell<HashMap<(i32, i32), Option<TreeSpec>>>,
    x: i32,
    z: i32,
) -> Option<TreeSpec> {
    if let Some(tree) = tree_cache.borrow().get(&(x, z)).copied() {
        return tree;
    }
    let h = height_at_cached(world_gen, height_cache, x, z);
    let tree = world_gen.tree_at(x, z, h);
    tree_cache.borrow_mut().insert((x, z), tree);
    tree
}

fn is_solid_cached(
    world_gen: &WorldGen,
    height_cache: &RefCell<HashMap<(i32, i32), i32>>,
    tree_cache: &RefCell<HashMap<(i32, i32), Option<TreeSpec>>>,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    if !world_gen.in_world_bounds(x, z) {
        return true;
    }

    let height = height_at_cached(world_gen, height_cache, x, z);
    if y <= height {
        return true;
    }

    let max_leaf_r = 3;
    for tz in (z - max_leaf_r)..=(z + max_leaf_r) {
        for tx in (x - max_leaf_r)..=(x + max_leaf_r) {
            let Some(tree) = tree_at_cached(world_gen, height_cache, tree_cache, tx, tz) else {
                continue;
            };

            let trunk_end = tree.base_y + tree.trunk_h;
            if x == tx && z == tz && y >= tree.base_y && y < trunk_end {
                return true;
            }

            let dy = y - trunk_end;
            if dy < -tree.leaf_r || dy > tree.leaf_r {
                continue;
            }
            let dx = x - tx;
            let dz = z - tz;
            if dx * dx + dy * dy + dz * dz <= tree.leaf_r * tree.leaf_r {
                if dx == 0 && dz == 0 && y >= tree.base_y && y < trunk_end {
                    continue;
                }
                return true;
            }
        }
    }

    false
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

fn should_pack_far_lod(mode: MeshMode, step: i32) -> bool {
    mode == MeshMode::SurfaceOnly || step >= 6
}

fn quantize_unorm01_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn quantize_uv_to_u16(v: f32) -> u16 {
    (v.clamp(-8.0, 8.0) * 4096.0).round() as i32 as u16
}

fn dequantize_uv_from_u16(v: u16) -> f32 {
    (v as i16 as f32) / 4096.0
}

fn pack_far_vertices(vertices: &[ChunkVertex]) -> Vec<PackedFarVertex> {
    let mut packed = Vec::with_capacity(vertices.len());
    for v in vertices {
        packed.push(PackedFarVertex {
            position: [
                v.position[0].round() as i16,
                v.position[1].round() as i16,
                v.position[2].round() as i16,
            ],
            uv: [quantize_uv_to_u16(v.uv[0]), quantize_uv_to_u16(v.uv[1])],
            tile: v.tile.min(u16::MAX as u32) as u16,
            face: v.face.min(u8::MAX as u32) as u8,
            rotation: v.rotation.min(u8::MAX as u32) as u8,
            use_texture: v.use_texture.min(u8::MAX as u32) as u8,
            transparent_mode: v.transparent_mode.min(u8::MAX as u32) as u8,
            color: [
                quantize_unorm01_to_u8(v.color[0]),
                quantize_unorm01_to_u8(v.color[1]),
                quantize_unorm01_to_u8(v.color[2]),
                quantize_unorm01_to_u8(v.color[3]),
            ],
        });
    }
    packed
}

fn unpack_far_vertices(vertices: &[PackedFarVertex]) -> Vec<ChunkVertex> {
    let mut unpacked = Vec::with_capacity(vertices.len());
    for v in vertices {
        unpacked.push(ChunkVertex {
            position: [
                v.position[0] as f32,
                v.position[1] as f32,
                v.position[2] as f32,
            ],
            uv: [dequantize_uv_from_u16(v.uv[0]), dequantize_uv_from_u16(v.uv[1])],
            tile: v.tile as u32,
            face: v.face as u32,
            rotation: v.rotation as u32,
            use_texture: v.use_texture as u32,
            transparent_mode: v.transparent_mode as u32,
            color: [
                v.color[0] as f32 / 255.0,
                v.color[1] as f32 / 255.0,
                v.color[2] as f32 / 255.0,
                v.color[3] as f32 / 255.0,
            ],
        });
    }
    unpacked
}

fn mesh_cache_dir(world_id: u64) -> PathBuf {
    Path::new("target")
        .join("mesh_cache")
        .join(format!("world_{world_id:016x}"))
}

fn mesh_cache_path(dir: &Path, coord: IVec3, step: i32, mode: MeshMode) -> PathBuf {
    dir.join(format!(
        "{}_{}_{}_m{}_s{}.bin",
        coord.x, coord.y, coord.z, mode as i32, step
    ))
}

fn write_u32_le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_f32_le(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32_le(data: &[u8], cursor: &mut usize) -> Option<u32> {
    let end = cursor.saturating_add(4);
    if end > data.len() {
        return None;
    }
    let mut buf = [0_u8; 4];
    buf.copy_from_slice(&data[*cursor..end]);
    *cursor = end;
    Some(u32::from_le_bytes(buf))
}

fn read_f32_le(data: &[u8], cursor: &mut usize) -> Option<f32> {
    let end = cursor.saturating_add(4);
    if end > data.len() {
        return None;
    }
    let mut buf = [0_u8; 4];
    buf.copy_from_slice(&data[*cursor..end]);
    *cursor = end;
    Some(f32::from_le_bytes(buf))
}

fn serialize_cache_entry(entry: &MeshCacheEntry) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        48 + entry.vertices.len() * std::mem::size_of::<PackedFarVertex>()
            + entry.indices.len() * std::mem::size_of::<u32>(),
    );
    out.extend_from_slice(MESH_CACHE_MAGIC);
    out.push(CACHE_ENCODING_PACKED_FAR);
    out.extend_from_slice(&[0_u8; 3]);
    write_f32_le(&mut out, entry.center.x);
    write_f32_le(&mut out, entry.center.y);
    write_f32_le(&mut out, entry.center.z);
    write_f32_le(&mut out, entry.radius);
    write_u32_le(&mut out, entry.vertices.len() as u32);
    write_u32_le(&mut out, entry.indices.len() as u32);
    out.extend_from_slice(bytemuck::cast_slice(&entry.vertices));
    out.extend_from_slice(bytemuck::cast_slice(&entry.indices));
    out
}

fn parse_cache_entry(bytes: &[u8]) -> Option<MeshCacheEntry> {
    if bytes.len() < 32 || &bytes[0..4] != MESH_CACHE_MAGIC {
        return None;
    }
    if bytes[4] != CACHE_ENCODING_PACKED_FAR {
        return None;
    }
    let mut cursor = 8;
    let cx = read_f32_le(bytes, &mut cursor)?;
    let cy = read_f32_le(bytes, &mut cursor)?;
    let cz = read_f32_le(bytes, &mut cursor)?;
    let radius = read_f32_le(bytes, &mut cursor)?;
    let vertex_count = read_u32_le(bytes, &mut cursor)? as usize;
    let index_count = read_u32_le(bytes, &mut cursor)? as usize;
    let vertex_bytes = vertex_count.checked_mul(std::mem::size_of::<PackedFarVertex>())?;
    let index_bytes = index_count.checked_mul(std::mem::size_of::<u32>())?;
    let total_needed = cursor.checked_add(vertex_bytes)?.checked_add(index_bytes)?;
    if total_needed > bytes.len() {
        return None;
    }
    let vertices_end = cursor + vertex_bytes;
    let indices_end = vertices_end + index_bytes;
    let vertices: Vec<PackedFarVertex> =
        bytemuck::cast_slice(&bytes[cursor..vertices_end]).to_vec();
    let indices: Vec<u32> = bytemuck::cast_slice(&bytes[vertices_end..indices_end]).to_vec();
    Some(MeshCacheEntry {
        center: Vec3::new(cx, cy, cz),
        radius,
        vertices,
        indices,
    })
}

fn try_load_cached_mesh(
    cache_dir: &Path,
    coord: IVec3,
    step: i32,
    mode: MeshMode,
) -> Option<MeshCacheEntry> {
    let path = mesh_cache_path(cache_dir, coord, step, mode);
    let bytes = fs::read(path).ok()?;
    parse_cache_entry(&bytes)
}

fn write_cached_mesh(cache_dir: &Path, coord: IVec3, step: i32, mode: MeshMode, entry: &MeshCacheEntry) {
    let path = mesh_cache_path(cache_dir, coord, step, mode);
    let tmp_path = path.with_extension("tmp");
    let bytes = serialize_cache_entry(entry);
    if fs::write(&tmp_path, bytes).is_ok() {
        let _ = fs::rename(&tmp_path, &path);
    }
}

fn enqueue_request(queue: &SharedRequestQueue, coord: IVec3, step: i32, mode: MeshMode, priority: i32) {
    let (lock, cvar) = &**queue;
    let mut state = lock.lock().unwrap();
    let seq = state.next_seq;
    state.next_seq = state.next_seq.wrapping_add(1);
    state.heap.push(RequestTask {
        priority,
        seq,
        coord,
        step,
        mode,
    });
    cvar.notify_one();
}

fn pop_request(queue: &SharedRequestQueue) -> Option<RequestTask> {
    let (lock, cvar) = &**queue;
    let mut state = lock.lock().unwrap();
    loop {
        if let Some(task) = state.heap.pop() {
            return Some(task);
        }
        if state.closed {
            return None;
        }
        state = cvar.wait(state).unwrap();
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
) -> Option<(IVec3, i8)> {
    let dir = forward.normalize();
    let step = 0.1f32;
    let mut t = 0.0f32;
    let mut last_block = IVec3::new(i32::MIN, i32::MIN, i32::MIN);
    while t <= max_dist {
        let pos = camera_pos + dir * t;
        let block = IVec3::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        if block != last_block {
            let block_id = world_gen.block_id_full_at(block.x, block.y, block.z);
            if block_id >= 0 {
                return Some((block, block_id));
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

fn unpack_lod(packed: i32) -> (MeshMode, i32) {
    let mode = match (packed >> 16) & 0xFFFF {
        0 => MeshMode::Full,
        1 => MeshMode::SurfaceSides,
        _ => MeshMode::SurfaceOnly,
    };
    let step = (packed & 0xFFFF).max(1);
    (mode, step)
}

fn lod_quality_rank(mode: MeshMode) -> i32 {
    match mode {
        MeshMode::Full => 3,
        MeshMode::SurfaceSides => 2,
        MeshMode::SurfaceOnly => 1,
    }
}

fn lod_covers(existing_lod: i32, required_lod: i32) -> bool {
    let (existing_mode, existing_step) = unpack_lod(existing_lod);
    let (required_mode, required_step) = unpack_lod(required_lod);
    let existing_rank = lod_quality_rank(existing_mode);
    let required_rank = lod_quality_rank(required_mode);
    if existing_rank > required_rank {
        return true;
    }
    if existing_rank < required_rank {
        return false;
    }
    existing_step <= required_step
}

fn needs_chunk_request(
    requested: &HashMap<(i32, i32, i32), i32>,
    loaded: &HashMap<(i32, i32, i32), i32>,
    key: (i32, i32, i32),
    desired_lod: i32,
) -> bool {
    if let Some(&requested_lod) = requested.get(&key) {
        return !lod_covers(requested_lod, desired_lod);
    }
    if let Some(&loaded_lod) = loaded.get(&key) {
        return !lod_covers(loaded_lod, desired_lod);
    }
    true
}

fn request_priority(player_pos: Vec3, camera_forward: Vec3, coord: (i32, i32, i32), lod: i32) -> i32 {
    let center = Vec3::new(
        (coord.0 * CHUNK_SIZE) as f32,
        (coord.1 * CHUNK_SIZE) as f32,
        (coord.2 * CHUNK_SIZE) as f32,
    );
    let to_chunk = center - player_pos;
    let dist_sq = to_chunk.length_squared().max(1.0);
    let dist_weight = (200_000.0 / dist_sq.sqrt()).round() as i32;
    let facing = if to_chunk.length_squared() > 0.0001 {
        camera_forward.normalize().dot(to_chunk.normalize())
    } else {
        0.0
    };
    let facing_weight = ((facing + 1.0) * 50_000.0).round() as i32;
    let (mode, _) = unpack_lod(lod);
    let detail_weight = match mode {
        MeshMode::Full => 80_000,
        MeshMode::SurfaceSides => 40_000,
        MeshMode::SurfaceOnly => 10_000,
    };
    dist_weight + facing_weight + detail_weight
}

fn submit_chunk_request(
    req_queue: &SharedRequestQueue,
    requested: &mut HashMap<(i32, i32, i32), i32>,
    player_pos: Vec3,
    camera_forward: Vec3,
    coord: (i32, i32, i32),
    step: i32,
    mode: MeshMode,
    lod: i32,
) {
    let priority = request_priority(player_pos, camera_forward, coord, lod);
    enqueue_request(
        req_queue,
        IVec3::new(coord.0, coord.1, coord.2),
        step,
        mode,
        priority,
    );
    requested.insert(coord, lod);
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
    looked_block: Option<(IVec3, i8)>,
    pregen_active: bool,
    pregen_ring_r: i32,
    pregen_radius_chunks: i32,
    pregen_columns_done: usize,
    pregen_total_columns: usize,
    pregen_chunks_requested: usize,
    pregen_chunks_created: usize,
    pregen_est_chunks_total: usize,
    loaded_chunk_cap: usize,
    mesh_memory_cap_mb: usize,
) {
    let triangles = stats.total_indices / 3;
    let seconds = world_time.as_secs_f32();
    let mesh_mem_mb = estimate_mesh_memory_mb(stats);
    let mem_fill = if mesh_memory_cap_mb > 0 {
        (mesh_mem_mb as f32 / mesh_memory_cap_mb as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cols_fill = if pregen_total_columns > 0 {
        (pregen_columns_done as f32 / pregen_total_columns as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let chunks_fill = if pregen_est_chunks_total > 0 {
        (pregen_chunks_created as f32 / pregen_est_chunks_total as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    print!("\x1B[2J\x1B[H");
    println!("F3 STATS (refresh 0.5s)");
    println!(
        "time: {:>8.2}s | tps: {:>3} | fps: {:>3} | paused_stream: {} | paused_render: {}",
        seconds, tps, fps, pause_stream, pause_render
    );
    println!(
        "player: ({:>7.2}, {:>7.2}, {:>7.2}) | chunk: ({:>4}, {:>4}, {:>4}) | super_chunk: ({:>4}, {:>4}, {:>4})",
        player_pos.x,
        player_pos.y,
        player_pos.z,
        current_chunk.x,
        current_chunk.y,
        current_chunk.z,
        super_chunk_coord_from_chunk(current_chunk).x,
        super_chunk_coord_from_chunk(current_chunk).y,
        super_chunk_coord_from_chunk(current_chunk).z,
    );
    match looked_block {
        Some((block, block_id)) => {
            let look_chunk = chunk_coord_from_block(block);
            let look_super = super_chunk_coord_from_chunk(look_chunk);
            let block_name = block_name_by_id(block_id);
            let item_id = block_item_id_by_id(block_id)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "looking_at_block: ({:>5}, {:>5}, {:>5}) | block_id: {:>3} | item_id: {:>3} | type: {} | look_chunk: ({:>4}, {:>4}, {:>4}) | look_super_chunk: ({:>4}, {:>4}, {:>4})",
                block.x,
                block.y,
                block.z,
                block_id,
                item_id,
                block_name,
                look_chunk.x,
                look_chunk.y,
                look_chunk.z,
                look_super.x,
                look_super.y,
                look_super.z
            );
        }
        None => {
            println!("looking_at_block: none | look_chunk: n/a | look_super_chunk: n/a");
        }
    }
    println!(
        "loaded: {:>6} | requested: {:>6} | base_render_radius: {:>4} | base_draw_radius: {:>4}",
        loaded.len(),
        requested.len(),
        base_render_radius,
        base_draw_radius
    );
    println!(
        "pregen: active={} progress_radius={:>4}/{:>4}",
        pregen_active, pregen_ring_r, pregen_radius_chunks
    );
    println!(
        "pregen columns [{}] {:>7}/{:<7} ({:>5.1}%)",
        progress_bar(cols_fill, 32),
        pregen_columns_done,
        pregen_total_columns,
        cols_fill * 100.0
    );
    println!(
        "pregen chunks  [{}] created={:>7} requested={:>7} est_total={:<7} ({:>5.1}%)",
        progress_bar(chunks_fill, 32),
        pregen_chunks_created,
        pregen_chunks_requested,
        pregen_est_chunks_total,
        chunks_fill * 100.0
    );
    println!(
        "memory fill    [{}] mesh_est={:>6}MB cap={:>6}MB | loaded_cap={}",
        progress_bar(mem_fill, 32),
        mesh_mem_mb,
        mesh_memory_cap_mb,
        loaded_chunk_cap
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
    println!("culling: camera_face_occlusion=off | backface_cull=on");
    let _ = io::stdout().flush();
}

fn progress_bar(fill: f32, width: usize) -> String {
    let fill = fill.clamp(0.0, 1.0);
    let filled = (fill * width as f32).round() as usize;
    let mut s = String::with_capacity(width);
    for i in 0..width {
        if i < filled {
            s.push('#');
        } else {
            s.push('-');
        }
    }
    s
}

fn estimate_mesh_memory_mb(stats: &GpuStats) -> usize {
    let vertex_bytes = stats.total_vertices_capacity * std::mem::size_of::<ChunkVertex>() as u64;
    let index_bytes = stats.total_indices * std::mem::size_of::<u32>() as u64;
    ((vertex_bytes + index_bytes) / (1024 * 1024)) as usize
}

fn in_pregen_bounds(coord: (i32, i32, i32), center: IVec3, radius: i32) -> bool {
    (coord.0 - center.x).abs() <= radius && (coord.2 - center.z).abs() <= radius
}

fn enforce_memory_budget(
    gpu: &mut Gpu,
    loaded: &mut HashMap<(i32, i32, i32), i32>,
    requested: &mut HashMap<(i32, i32, i32), i32>,
    current_chunk: IVec3,
    loaded_chunk_cap: usize,
    mesh_memory_cap_mb: usize,
) {
    let mut mesh_mem_mb = estimate_mesh_memory_mb(&gpu.stats());
    if loaded.len() <= loaded_chunk_cap && mesh_mem_mb <= mesh_memory_cap_mb {
        return;
    }

    let mut coords: Vec<(i32, i32, i32)> = loaded.keys().copied().collect();
    coords.sort_by_key(|(x, y, z)| {
        let dx = x - current_chunk.x;
        let dy = y - current_chunk.y;
        let dz = z - current_chunk.z;
        -(dx * dx + dy * dy + dz * dz)
    });

    for coord in coords {
        if loaded.len() <= loaded_chunk_cap && mesh_mem_mb <= mesh_memory_cap_mb {
            break;
        }
        gpu.remove_chunk(IVec3::new(coord.0, coord.1, coord.2));
        loaded.remove(&coord);
        requested.remove(&coord);
        mesh_mem_mb = estimate_mesh_memory_mb(&gpu.stats());
    }
}

fn stream_tick(
    gpu: &mut Gpu,
    player_pos: Vec3,
    camera_forward: Vec3,
    current_chunk: &mut IVec3,
    ring_r: &mut i32,
    ring_i: &mut i32,
    loaded: &mut HashMap<(i32, i32, i32), i32>,
    requested: &mut HashMap<(i32, i32, i32), i32>,
    req_queue: &SharedRequestQueue,
    rx_res: &mpsc::Receiver<WorkerResult>,
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
    pregen_center_chunk: IVec3,
    pregen_radius_chunks: i32,
    pregen_budget_per_tick: i32,
    pregen_active: &mut bool,
    pregen_ring_r: &mut i32,
    pregen_ring_i: &mut i32,
    pregen_columns_done: &mut usize,
    pregen_chunks_requested: &mut usize,
    pregen_chunks_created: &mut usize,
    loaded_chunk_cap: usize,
    mesh_memory_cap_mb: usize,
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
        let keep_radius = max_render_radius + 8;
        requested.retain(|(x, _, z), _| {
            in_world_chunk_bounds(*x, *z)
                && (*x - new_chunk.x).abs() <= keep_radius
                && (*z - new_chunk.z).abs() <= keep_radius
        });
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

    let max_apply_per_tick = 64;
    for _ in 0..max_apply_per_tick {
        let Ok(result) = rx_res.try_recv() else { break; };
        let mesh = match result {
            WorkerResult::Raw(mesh) => mesh,
            WorkerResult::Packed(packed) => MeshData {
                coord: packed.coord,
                step: packed.step,
                mode: packed.mode,
                center: packed.center,
                radius: packed.radius,
                vertices: unpack_far_vertices(&packed.vertices),
                indices: packed.indices,
            },
        };
        let k = (mesh.coord.x, mesh.coord.y, mesh.coord.z);
        gpu.upsert_chunk(
            mesh.coord,
            mesh.center,
            mesh.radius,
            mesh.vertices,
            mesh.indices,
        );
        let was_loaded = loaded.insert(k, pack_lod(mesh.mode, mesh.step)).is_some();
        if !was_loaded && in_pregen_bounds(k, pregen_center_chunk, pregen_radius_chunks) {
            *pregen_chunks_created += 1;
        }
        requested.remove(&k);
    }
    let max_rebuilds_per_tick = 8;
    gpu.rebuild_dirty_superchunks(player_pos, max_rebuilds_per_tick);
    enforce_memory_budget(
        gpu,
        loaded,
        requested,
        *current_chunk,
        loaded_chunk_cap,
        mesh_memory_cap_mb,
    );

    if *pregen_active {
        let mut budget = pregen_budget_per_tick.max(0);
        while budget > 0 && *pregen_ring_r <= pregen_radius_chunks {
            if requested.len() >= max_inflight {
                return;
            }

            let (dx, dz) = ring_coord(*pregen_ring_r, *pregen_ring_i);
            *pregen_ring_i += 1;
            let ring_len = ring_length(*pregen_ring_r);
            if *pregen_ring_i >= ring_len {
                *pregen_ring_i = 0;
                *pregen_ring_r += 1;
            }
            *pregen_columns_done += 1;

            let cx = pregen_center_chunk.x + dx;
            let cz = pregen_center_chunk.z + dz;
            if !in_world_chunk_bounds(cx, cz) {
                continue;
            }
            let surface_y = max_height_cached(cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let y_start = surface_chunk_y - surface_depth_chunks;
            let y_end = surface_chunk_y + 1;

            for cy in y_start..=y_end {
                if budget <= 0 || requested.len() >= max_inflight {
                    break;
                }
                let coord = (cx, cy, cz);
                let step = 6;
                let mode = MeshMode::SurfaceOnly;
                let lod = pack_lod(mode, step);
                if needs_chunk_request(requested, loaded, coord, lod) {
                    let was_requested = requested.contains_key(&coord);
                    submit_chunk_request(
                        req_queue,
                        requested,
                        player_pos,
                        camera_forward,
                        coord,
                        step,
                        mode,
                        lod,
                    );
                    if !was_requested {
                        *pregen_chunks_requested += 1;
                    }
                    budget -= 1;
                }
            }
        }

        if *pregen_ring_r > pregen_radius_chunks {
            *pregen_active = false;
        }
    }

    if requested.len() >= max_inflight {
        return;
    }

    // Safe column: 2x2 chunks around player are always requested
    for dz in 0..=1 {
        for dx in 0..=1 {
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            if !in_world_chunk_bounds(cx, cz) {
                continue;
            }
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
                if needs_chunk_request(requested, loaded, key, lod) {
                    submit_chunk_request(
                        req_queue,
                        requested,
                        player_pos,
                        camera_forward,
                        key,
                        step,
                        mode,
                        lod,
                    );
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
                if !in_world_chunk_bounds(cx, cz) {
                    continue;
                }
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
                    if needs_chunk_request(requested, loaded, key, lod) {
                        submit_chunk_request(
                            req_queue,
                            requested,
                            player_pos,
                            camera_forward,
                            key,
                            step,
                            mode,
                            lod,
                        );
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
            if !in_world_chunk_bounds(cx, cz) {
                continue;
            }
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
                if needs_chunk_request(requested, loaded, key, lod) {
                    submit_chunk_request(
                        req_queue,
                        requested,
                        player_pos,
                        camera_forward,
                        key,
                        step,
                        mode,
                        lod,
                    );
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
        if !in_world_chunk_bounds(cx, cz) {
            continue;
        }
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
            if needs_chunk_request(requested, loaded, key, lod) {
                submit_chunk_request(
                    req_queue,
                    requested,
                    player_pos,
                    camera_forward,
                    key,
                    step,
                    mode,
                    lod,
                );
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
            if !in_world_chunk_bounds(*x, *z) {
                return true;
            }
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
