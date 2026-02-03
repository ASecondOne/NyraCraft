use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::mpsc;
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

use colorthing::ColorThing;
use render::{BlockTexture, CubeStyle, TextureAtlas};
use camera::Camera;
use render::Gpu;
use world::CHUNK_SIZE;
use glam::IVec3;
use world::worldgen::WorldGen;
use world::mesher::{generate_chunk_mesh, MeshData};

enum WorkerMsg {
    Reset(u64),
    Request {
        coord: IVec3,
        step: i32,
        priority: i32,
        epoch: u64,
    },
}

#[derive(Eq, PartialEq)]
struct Job {
    priority: i32,
    step: i32,
    coord: IVec3,
    order: u64,
    epoch: u64,
}

impl Job {
    fn new(priority: i32, step: i32, coord: IVec3, order: u64, epoch: u64) -> Self {
        Self {
            priority,
            step,
            coord,
            order,
            epoch,
        }
    }
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.step.cmp(&self.step))
            .then_with(|| other.order.cmp(&self.order))
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn handle_worker_msg(
    msg: WorkerMsg,
    heap: &mut BinaryHeap<Job>,
    order: &mut u64,
    epoch: &mut u64,
) {
    match msg {
        WorkerMsg::Reset(new_epoch) => {
            *epoch = new_epoch;
            heap.clear();
        }
        WorkerMsg::Request {
            coord,
            step,
            priority,
            epoch: msg_epoch,
        } => {
            if msg_epoch != *epoch {
                return;
            }
            heap.push(Job::new(priority, step, coord, *order, msg_epoch));
            *order = order.wrapping_add(1);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Window HEHEHAHA fuck JavaFX")
        .build(&event_loop)
        .unwrap();

    let style = CubeStyle { use_texture: true };

    let atlas = TextureAtlas {
        path: "src/textures/atlas.png".to_string(),
        tile_size: 16,
    };

    let tiles_x = 16;
    let stone = BlockTexture::solid(
        tile_index_1based(2, 1, tiles_x),
        ColorThing::new(1.0, 1.0, 1.0),
    );
    let dirt = BlockTexture::solid(
        tile_index_1based(3, 1, tiles_x),
        ColorThing::new(1.0, 1.0, 1.0),
    );
    let grass = BlockTexture {
        colors: [
            ColorThing::new(0.62, 0.82, 0.35), // +X
            ColorThing::new(0.62, 0.82, 0.35), // -X
            ColorThing::new(0.62, 0.82, 0.35), // +Y (top)
            ColorThing::new(1.0, 1.0, 1.0),    // -Y (bottom)
            ColorThing::new(0.62, 0.82, 0.35), // +Z
            ColorThing::new(0.62, 0.82, 0.35), // -Z
        ],
        tiles: [
            tile_index_1based(4, 1, tiles_x), // sides
            tile_index_1based(4, 1, tiles_x),
            tile_index_1based(9, 3, tiles_x), // top (tinted)
            tile_index_1based(3, 1, tiles_x), // bottom
            tile_index_1based(4, 1, tiles_x),
            tile_index_1based(4, 1, tiles_x),
        ],
        rotations: [3, 3, 0, 0, 0, 0],
    };

    let blocks = vec![stone, dirt, grass];

    let seed: u32 = rand::random();
    let world_gen = WorldGen::new(seed).with_height(0, 12.0, 0.04);
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
    let mut pause_stream = false;
    let mut pause_render = false;
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
    thread::spawn(move || {
        let mut heap: BinaryHeap<Job> = BinaryHeap::new();
        let mut order: u64 = 0;
        let mut epoch: u64 = 0;
        loop {
            if heap.is_empty() {
                match rx_req.recv() {
                    Ok(msg) => {
                        handle_worker_msg(msg, &mut heap, &mut order, &mut epoch);
                    }
                    Err(_) => break,
                };
            }

            while let Ok(msg) = rx_req.try_recv() {
                handle_worker_msg(msg, &mut heap, &mut order, &mut epoch);
            }

            if let Some(job) = heap.pop() {
                if job.epoch != epoch {
                    continue;
                }
                let mesh = generate_chunk_mesh(job.coord, &blocks_gen, &gen_thread, job.step);
                let _ = tx_res.send(mesh);
            }
        }
    });

    let base_render_radius = 256;
    let max_render_radius = 256;
    let request_budget = 64;
    let max_inflight = 2048usize;
    let y_range_up = 32;
    let y_range_down = 16;
    let initial_burst_radius = 4;
    let mut current_chunk = chunk_coord_from_pos(player_pos);
    let mut loaded: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut requested: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut ring_r: i32 = 0;
    let mut ring_i: i32 = 0;
    let mut stream_epoch: u64 = 0;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            elwt.set_control_flow(ControlFlow::WaitUntil(next_tick));

            let now = Instant::now();
            if now >= next_tick {
                let dt = (now - last_update).as_secs_f32();
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
                        y_range_down,
                        y_range_up,
                        initial_burst_radius,
                        max_inflight,
                        &mut force_reload,
                        &mut stream_epoch,
                    );
                }

                if last_title_update.elapsed() >= Duration::from_millis(250) {
                    let title = format!(
                        "chunks loaded:{} requested:{} paused_stream:{} paused_render:{}",
                        loaded.len(),
                        requested.len(),
                        pause_stream,
                        pause_render
                    );
                    gpu.window().set_title(&title);
                    last_title_update = Instant::now();
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
                gpu.render(&camera, debug_faces);
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
                KeyCode::F4 if pressed => {
                    force_reload = true;
                }
                KeyCode::F5 if pressed => {
                    pause_render = !pause_render;
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

fn tile_index_1based(x: u32, y: u32, tiles_x: u32) -> u32 {
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    x0 + y0 * tiles_x
}

fn chunk_coord_from_pos(pos: Vec3) -> IVec3 {
    IVec3::new(
        (pos.x / CHUNK_SIZE as f32).floor() as i32,
        (pos.y / CHUNK_SIZE as f32).floor() as i32,
        (pos.z / CHUNK_SIZE as f32).floor() as i32,
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

fn distance_3d(center: IVec3, coord: (i32, i32, i32)) -> i32 {
    let dx = (coord.0 - center.x).abs();
    let dy = (coord.1 - center.y).abs();
    let dz = (coord.2 - center.z).abs();
    dx.max(dy).max(dz)
}

fn dy_order(range_down: i32, range_up: i32) -> Vec<i32> {
    let max = range_down.max(range_up);
    let mut out = Vec::with_capacity((max * 2 + 1) as usize);
    out.push(0);
    for offset in 1..=max {
        if offset <= range_up {
            out.push(offset);
        }
        if offset <= range_down {
            out.push(-offset);
        }
    }
    out
}

fn lod_step(dist: i32) -> i32 {
    let mut step = 1;
    let mut threshold = 16;
    let d = dist;
    while d >= threshold {
        step *= 2;
        threshold *= 2;
    }
    step
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
    y_range_down: i32,
    y_range_up: i32,
    initial_burst_radius: i32,
    max_inflight: usize,
    force_reload: &mut bool,
    stream_epoch: &mut u64,
) {
    let new_chunk = chunk_coord_from_pos(player_pos);
    let mut reset_worker = false;
    if new_chunk != *current_chunk {
        *current_chunk = new_chunk;
        *ring_r = 0;
        *ring_i = 0;
        requested.clear();
        reset_worker = true;
    }

    let dynamic_radius = ((player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32) * 2;
    let render_radius = (base_render_radius + dynamic_radius).min(max_render_radius);
    let height_chunks = (player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32;
    let height_factor = height_chunks / 8;
    let budget = (request_budget - height_factor * 8).max(16);
    let dy_list = dy_order(y_range_down, y_range_up);

    if *force_reload {
        gpu.clear_chunks();
        loaded.clear();
        requested.clear();
        *ring_r = 0;
        *ring_i = 0;
        *force_reload = false;
        reset_worker = true;
    }

    if reset_worker {
        *stream_epoch = stream_epoch.wrapping_add(1);
        let _ = tx_req.send(WorkerMsg::Reset(*stream_epoch));
    }

    if requested.len() >= max_inflight {
        while let Ok(mesh) = rx_res.try_recv() {
            let k = (mesh.coord.x, mesh.coord.y, mesh.coord.z);
            let dx = (k.0 - current_chunk.x).abs();
            let dy = (k.1 - current_chunk.y).abs();
            let dz = (k.2 - current_chunk.z).abs();
            if dx > render_radius || dz > render_radius || dy < -y_range_down || dy > y_range_up {
                requested.remove(&k);
                continue;
            }
            gpu.upsert_chunk(mesh.coord, mesh.center, mesh.radius, &mesh.vertices, &mesh.indices);
            loaded.insert(k, mesh.step);
            requested.remove(&k);
        }
        return;
    }

    for dz in -initial_burst_radius..=initial_burst_radius {
        for dx in -initial_burst_radius..=initial_burst_radius {
            for &dy in &dy_list {
                let coord = (
                    current_chunk.x + dx,
                    current_chunk.y + dy,
                    current_chunk.z + dz,
                );
                let dist = distance_2d(*current_chunk, coord);
                if dist > render_radius {
                    continue;
                }
                let step = 1;
                let priority = distance_3d(*current_chunk, coord);
                let key = coord;
                let mut needs_request = true;
                if let Some(&req_step) = requested.get(&key) {
                    needs_request = req_step != step;
                } else if let Some(&loaded_step) = loaded.get(&key) {
                    needs_request = loaded_step != step;
                }
                if needs_request {
                    let _ = tx_req.send(WorkerMsg::Request {
                        coord: IVec3::new(coord.0, coord.1, coord.2),
                        step,
                        priority,
                        epoch: *stream_epoch,
                    });
                    requested.insert(key, step);
                }
            }
        }
    }

    let mut budget = budget;
    while budget > 0 && *ring_r <= render_radius {
        let (dx, dz) = ring_coord(*ring_r, *ring_i);
        *ring_i += 1;
        let ring_len = ring_length(*ring_r);
        if *ring_i >= ring_len {
            *ring_i = 0;
            *ring_r += 1;
        }
        for &dy in &dy_list {
            if budget <= 0 {
                break;
            }
            let coord = (
                current_chunk.x + dx,
                current_chunk.y + dy,
                current_chunk.z + dz,
            );
            let dist = distance_2d(*current_chunk, coord);
            if dist > render_radius {
                continue;
            }
            let step = if dx.abs() <= 1 && dz.abs() <= 1 {
                1
            } else {
                lod_step(dist)
            };
            let priority = distance_3d(*current_chunk, coord);
            let key = coord;
            let mut needs_request = true;
            if let Some(&req_step) = requested.get(&key) {
                needs_request = req_step != step;
            } else if let Some(&loaded_step) = loaded.get(&key) {
                needs_request = loaded_step != step;
            }
            if needs_request {
                let _ = tx_req.send(WorkerMsg::Request {
                    coord: IVec3::new(coord.0, coord.1, coord.2),
                    step,
                    priority,
                    epoch: *stream_epoch,
                });
                requested.insert(key, step);
                budget -= 1;
                if requested.len() >= max_inflight {
                    break;
                }
            }
        }
    }

    let to_remove: Vec<(i32, i32, i32)> = loaded
        .keys()
        .filter(|(x, y, z)| {
            let dx = x - current_chunk.x;
            let dy = y - current_chunk.y;
            let dz = z - current_chunk.z;
            dx.abs() > render_radius
                || dz.abs() > render_radius
                || dy < -y_range_down
                || dy > y_range_up
        })
        .cloned()
        .collect();
    for coord in to_remove {
        gpu.remove_chunk(IVec3::new(coord.0, coord.1, coord.2));
        loaded.remove(&coord);
        requested.remove(&coord);
    }

    while let Ok(mesh) = rx_res.try_recv() {
        let k = (mesh.coord.x, mesh.coord.y, mesh.coord.z);
        let dx = (k.0 - current_chunk.x).abs();
        let dy = (k.1 - current_chunk.y).abs();
        let dz = (k.2 - current_chunk.z).abs();
        if dx > render_radius || dz > render_radius || dy < -y_range_down || dy > y_range_up {
            requested.remove(&k);
            continue;
        }
        gpu.upsert_chunk(mesh.coord, mesh.center, mesh.radius, &mesh.vertices, &mesh.indices);
        loaded.insert(k, mesh.step);
        requested.remove(&k);

        let dist = dx.max(dz);
        let desired_step = if dx <= 1 && dz <= 1 {
            1
        } else {
            lod_step(dist)
        };
        if desired_step != mesh.step {
            let priority = dx.max(dy).max(dz);
            let _ = tx_req.send(WorkerMsg::Request {
                coord: mesh.coord,
                step: desired_step,
                priority,
                epoch: *stream_epoch,
            });
            requested.insert(k, desired_step);
        }
    }
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
