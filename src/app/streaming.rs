use glam::{IVec3, Vec3};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::render::mesh::{ChunkVertex, PackedFarVertex};
use crate::render::{Gpu, GpuStats};
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{
    block_break_time_with_item_seconds, block_hardness, block_name_by_id, can_break_block_with_item,
};
use crate::world::mesher::{MeshData, MeshMode};
use crate::world::worldgen::{TreeSpec, WORLD_HALF_SIZE_CHUNKS, WorldGen, WorldMode};

pub enum WorkerResult {
    Raw {
        mesh: MeshData,
        dirty_rev: Option<u64>,
    },
    Packed {
        packed: PackedMeshData,
        dirty_rev: Option<u64>,
    },
}

#[derive(Clone)]
pub struct PackedMeshData {
    pub coord: IVec3,
    pub step: i32,
    pub mode: MeshMode,
    pub center: Vec3,
    pub radius: f32,
    pub vertices: Arc<[PackedFarVertex]>,
    pub indices: Arc<[u32]>,
}

pub enum CacheWriteMsg {
    Write {
        coord: IVec3,
        step: i32,
        mode: MeshMode,
        center: Vec3,
        radius: f32,
        vertices: Arc<[PackedFarVertex]>,
        indices: Arc<[u32]>,
    },
}

#[derive(Clone, Copy)]
pub struct RequestTask {
    pub priority: i32,
    pub seq: u64,
    pub coord: IVec3,
    pub step: i32,
    pub mode: MeshMode,
    pub lighting_pass: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestClass {
    Edit,
    Near,
    Far,
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

pub struct RequestQueueState {
    pub edit_heap: BinaryHeap<RequestTask>,
    pub near_heap: BinaryHeap<RequestTask>,
    pub far_heap: BinaryHeap<RequestTask>,
    pub latest_task_by_chunk: HashMap<ChunkKey, PendingTaskMeta>,
    pub next_seq: u64,
    pub closed: bool,
}

#[derive(Clone, Copy)]
pub struct PendingTaskMeta {
    pub seq: u64,
    pub class: RequestClass,
    pub mode: MeshMode,
    pub step: i32,
    pub lighting_pass: bool,
}

pub type SharedRequestQueue = Arc<(Mutex<RequestQueueState>, Condvar)>;
type ChunkKey = (i32, i32, i32);
pub type DirtyChunks = Arc<Mutex<HashMap<ChunkKey, i64>>>;
pub type EditedChunkRanges = Arc<Mutex<HashMap<ChunkKey, (i32, i32)>>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestQueueStats {
    pub edit: usize,
    pub near: usize,
    pub far: usize,
    pub inflight_chunks: usize,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugPerfSnapshot {
    pub tick_cpu_ms: f32,
    pub render_cpu_ms: f32,
    pub player_ms: f32,
    pub stream_ms: f32,
    pub apply_ms: f32,
    pub mining_ms: f32,
}

fn dirty_state_rev(state: i64) -> Option<u64> {
    let rev = state.abs();
    if rev == 0 { None } else { Some(rev as u64) }
}

pub fn new_request_queue() -> SharedRequestQueue {
    Arc::new((
        Mutex::new(RequestQueueState {
            edit_heap: BinaryHeap::new(),
            near_heap: BinaryHeap::new(),
            far_heap: BinaryHeap::new(),
            latest_task_by_chunk: HashMap::new(),
            next_seq: 0,
            closed: false,
        }),
        Condvar::new(),
    ))
}

pub fn request_queue_stats(queue: &SharedRequestQueue) -> RequestQueueStats {
    let (lock, _) = &**queue;
    let state = lock.lock().unwrap();
    RequestQueueStats {
        edit: state.edit_heap.len(),
        near: state.near_heap.len(),
        far: state.far_heap.len(),
        inflight_chunks: state.latest_task_by_chunk.len(),
        closed: state.closed,
    }
}

const MESH_CACHE_MAGIC: &[u8; 4] = b"MSH3";
const CACHE_ENCODING_PACKED_FAR: u8 = 1;

pub struct MeshCacheEntry {
    pub center: Vec3,
    pub radius: f32,
    pub vertices: Vec<PackedFarVertex>,
    pub indices: Vec<u32>,
}

pub struct CacheMeshView<'a> {
    pub center: Vec3,
    pub radius: f32,
    pub vertices: &'a [PackedFarVertex],
    pub indices: &'a [u32],
}

impl MeshCacheEntry {
    pub fn into_worker_result(self, coord: IVec3, step: i32, mode: MeshMode) -> WorkerResult {
        WorkerResult::Packed {
            packed: PackedMeshData {
                coord,
                step,
                mode,
                center: self.center,
                radius: self.radius,
                vertices: self.vertices.into(),
                indices: self.indices.into(),
            },
            dirty_rev: None,
        }
    }
}

pub fn chunk_coord_from_pos(pos: Vec3) -> IVec3 {
    let half = (CHUNK_SIZE / 2) as f32;
    IVec3::new(
        ((pos.x + half) / CHUNK_SIZE as f32).floor() as i32,
        ((pos.y + half) / CHUNK_SIZE as f32).floor() as i32,
        ((pos.z + half) / CHUNK_SIZE as f32).floor() as i32,
    )
}

fn in_world_chunk_bounds(cx: i32, cz: i32) -> bool {
    (-WORLD_HALF_SIZE_CHUNKS..WORLD_HALF_SIZE_CHUNKS).contains(&cx)
        && (-WORLD_HALF_SIZE_CHUNKS..WORLD_HALF_SIZE_CHUNKS).contains(&cz)
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

pub fn is_solid_cached(
    world_gen: &WorldGen,
    height_cache: &RefCell<HashMap<(i32, i32), i32>>,
    tree_cache: &RefCell<HashMap<(i32, i32), Option<TreeSpec>>>,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    if world_gen.mode == WorldMode::Normal && !world_gen.in_world_bounds(x, z) {
        return true;
    }
    block_id_full_cached(world_gen, height_cache, tree_cache, x, y, z) >= 0
}

pub fn block_id_full_cached(
    world_gen: &WorldGen,
    height_cache: &RefCell<HashMap<(i32, i32), i32>>,
    tree_cache: &RefCell<HashMap<(i32, i32), Option<TreeSpec>>>,
    x: i32,
    y: i32,
    z: i32,
) -> i8 {
    if world_gen.mode != WorldMode::Normal {
        return world_gen.block_id_full_at(x, y, z);
    }
    if !world_gen.in_world_bounds(x, z) {
        return -1;
    }

    let height = height_at_cached(world_gen, height_cache, x, z);
    if y <= height {
        return world_gen.block_id_at(x, y, z, height);
    }

    let max_leaf_r = 3;
    for tz in (z - max_leaf_r)..=(z + max_leaf_r) {
        for tx in (x - max_leaf_r)..=(x + max_leaf_r) {
            let Some(tree) = tree_at_cached(world_gen, height_cache, tree_cache, tx, tz) else {
                continue;
            };

            let trunk_end = tree.base_y + tree.trunk_h;
            if x == tx && z == tz && y >= tree.base_y && y < trunk_end {
                return crate::world::blocks::BLOCK_LOG as i8;
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
                return crate::world::blocks::BLOCK_LEAVES as i8;
            }
        }
    }

    -1
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

pub fn should_pack_far_lod(mode: MeshMode, step: i32) -> bool {
    mode == MeshMode::SurfaceOnly || step >= 6
}

pub fn mesh_cache_dir(world_id: u64) -> PathBuf {
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

fn serialize_cache_entry(
    center: Vec3,
    radius: f32,
    vertices: &[PackedFarVertex],
    indices: &[u32],
) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(48 + std::mem::size_of_val(vertices) + std::mem::size_of_val(indices));
    out.extend_from_slice(MESH_CACHE_MAGIC);
    out.push(CACHE_ENCODING_PACKED_FAR);
    out.extend_from_slice(&[0_u8; 3]);
    write_f32_le(&mut out, center.x);
    write_f32_le(&mut out, center.y);
    write_f32_le(&mut out, center.z);
    write_f32_le(&mut out, radius);
    write_u32_le(&mut out, vertices.len() as u32);
    write_u32_le(&mut out, indices.len() as u32);
    out.extend_from_slice(bytemuck::cast_slice(vertices));
    out.extend_from_slice(bytemuck::cast_slice(indices));
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

pub fn try_load_cached_mesh(
    cache_dir: &Path,
    coord: IVec3,
    step: i32,
    mode: MeshMode,
) -> Option<MeshCacheEntry> {
    let path = mesh_cache_path(cache_dir, coord, step, mode);
    let bytes = fs::read(path).ok()?;
    parse_cache_entry(&bytes)
}

pub fn write_cached_mesh(
    cache_dir: &Path,
    coord: IVec3,
    step: i32,
    mode: MeshMode,
    mesh: CacheMeshView<'_>,
) {
    let path = mesh_cache_path(cache_dir, coord, step, mode);
    let tmp_path = path.with_extension("tmp");
    let bytes = serialize_cache_entry(mesh.center, mesh.radius, mesh.vertices, mesh.indices);
    if fs::write(&tmp_path, bytes).is_ok() {
        let _ = fs::rename(&tmp_path, &path);
    }
}

fn enqueue_request(
    queue: &SharedRequestQueue,
    coord: IVec3,
    step: i32,
    mode: MeshMode,
    lighting_pass: bool,
    class: RequestClass,
    priority: i32,
) {
    let (lock, cvar) = &**queue;
    let mut state = lock.lock().unwrap();
    let key = (coord.x, coord.y, coord.z);
    if let Some(existing) = state.latest_task_by_chunk.get(&key).copied()
        && !should_replace_pending(existing, class, mode, step, lighting_pass)
    {
        return;
    }
    let seq = state.next_seq;
    state.next_seq = state.next_seq.wrapping_add(1);
    state.latest_task_by_chunk.insert(
        key,
        PendingTaskMeta {
            seq,
            class,
            mode,
            step,
            lighting_pass,
        },
    );
    let task = RequestTask {
        priority,
        seq,
        coord,
        step,
        mode,
        lighting_pass,
    };
    match class {
        RequestClass::Edit => state.edit_heap.push(task),
        RequestClass::Near => state.near_heap.push(task),
        RequestClass::Far => state.far_heap.push(task),
    }
    cvar.notify_one();
}

fn request_chunk_remesh_with_priority(queue: &SharedRequestQueue, coord: IVec3, priority: i32) {
    enqueue_request(
        queue,
        coord,
        1,
        MeshMode::Full,
        false,
        RequestClass::Edit,
        priority,
    );
}

pub fn request_chunk_remesh_now(queue: &SharedRequestQueue, coord: IVec3) {
    request_chunk_remesh_with_priority(queue, coord, i32::MAX);
}

pub fn request_chunk_remesh_decay(queue: &SharedRequestQueue, coord: IVec3) {
    enqueue_request(
        queue,
        coord,
        1,
        MeshMode::Full,
        true,
        RequestClass::Near,
        i32::MAX,
    );
}

pub fn request_chunk_remesh_lighting(queue: &SharedRequestQueue, coord: IVec3) {
    enqueue_request(
        queue,
        coord,
        1,
        MeshMode::Full,
        true,
        RequestClass::Edit,
        i32::MIN / 2,
    );
}

fn pop_fresh_from_heap(
    heap: &mut BinaryHeap<RequestTask>,
    latest_task_by_chunk: &mut HashMap<ChunkKey, PendingTaskMeta>,
) -> Option<RequestTask> {
    while let Some(task) = heap.pop() {
        let key = (task.coord.x, task.coord.y, task.coord.z);
        let Some(latest) = latest_task_by_chunk.get(&key).copied() else {
            continue;
        };
        if task.seq != latest.seq {
            continue;
        }
        latest_task_by_chunk.remove(&key);
        return Some(task);
    }
    None
}

fn class_rank(class: RequestClass) -> i32 {
    match class {
        RequestClass::Edit => 3,
        RequestClass::Near => 2,
        RequestClass::Far => 1,
    }
}

fn mode_rank(mode: MeshMode) -> i32 {
    match mode {
        MeshMode::Full => 3,
        MeshMode::SurfaceSides => 2,
        MeshMode::SurfaceOnly => 1,
    }
}

fn should_replace_pending(
    existing: PendingTaskMeta,
    new_class: RequestClass,
    new_mode: MeshMode,
    new_step: i32,
    new_lighting_pass: bool,
) -> bool {
    let existing_class_rank = class_rank(existing.class);
    let new_class_rank = class_rank(new_class);
    if new_class_rank > existing_class_rank {
        return true;
    }
    if new_class_rank < existing_class_rank {
        return false;
    }

    let existing_mode_rank = mode_rank(existing.mode);
    let new_mode_rank = mode_rank(new_mode);
    if new_mode_rank > existing_mode_rank {
        return true;
    }
    if new_mode_rank < existing_mode_rank {
        return false;
    }

    // Lower step means higher quality.
    if new_step < existing.step {
        return true;
    }
    if new_step > existing.step {
        return false;
    }

    // Geometry-first policy: a non-lighting update must never be replaced by a lighting pass.
    if new_lighting_pass && !existing.lighting_pass {
        return false;
    }
    if !new_lighting_pass && existing.lighting_pass {
        return true;
    }

    // Same class/quality: allow refreshing priority.
    true
}

fn pop_adaptive(state: &mut RequestQueueState) -> Option<RequestTask> {
    let edit_len = state.edit_heap.len();
    let near_len = state.near_heap.len();
    let far_len = state.far_heap.len();

    if edit_len + near_len + far_len == 0 {
        return None;
    }

    // Hard top priority for edits.
    if edit_len > 0
        && let Some(task) =
            pop_fresh_from_heap(&mut state.edit_heap, &mut state.latest_task_by_chunk)
    {
        return Some(task);
    }

    // Near must always win over far.
    if near_len > 0
        && let Some(task) =
            pop_fresh_from_heap(&mut state.near_heap, &mut state.latest_task_by_chunk)
    {
        return Some(task);
    }

    if far_len > 0
        && let Some(task) =
            pop_fresh_from_heap(&mut state.far_heap, &mut state.latest_task_by_chunk)
    {
        return Some(task);
    }

    if edit_len > 0
        && let Some(task) =
            pop_fresh_from_heap(&mut state.edit_heap, &mut state.latest_task_by_chunk)
    {
        return Some(task);
    }
    None
}

pub fn pop_request(queue: &SharedRequestQueue) -> Option<RequestTask> {
    let (lock, cvar) = &**queue;
    let mut state = lock.lock().unwrap();
    loop {
        if let Some(task) = pop_adaptive(&mut state) {
            return Some(task);
        }
        if state.closed {
            return None;
        }
        state = cvar.wait(state).unwrap();
    }
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

fn column_stream_y_range(
    surface_chunk_y: i32,
    player_chunk_y: i32,
    base_surface_depth_chunks: i32,
    dist_xz: i32,
) -> (i32, i32) {
    const SURFACE_STICKY_MARGIN_CHUNKS: i32 = 1;
    const CAVE_WINDOW_DEPTH_CHUNKS: i32 = 3;
    const CAVE_WINDOW_ABOVE_CHUNKS: i32 = 2;

    let surface_depth = depth_for_dist(base_surface_depth_chunks, dist_xz);
    let surface_start = surface_chunk_y - surface_depth;
    let surface_end = surface_chunk_y + 1;
    if player_chunk_y >= surface_start - SURFACE_STICKY_MARGIN_CHUNKS {
        return (surface_start, surface_end);
    }

    let cave_depth = depth_for_dist(CAVE_WINDOW_DEPTH_CHUNKS, dist_xz).max(1);
    (
        player_chunk_y - cave_depth,
        player_chunk_y + CAVE_WINDOW_ABOVE_CHUNKS,
    )
}

fn radius_cap_from_loaded_limit(surface_depth_chunks: i32, loaded_chunk_cap: usize) -> i32 {
    if loaded_chunk_cap == 0 {
        return 0;
    }
    // Conservative estimate: each column can occupy up to depth+2 chunks.
    let chunks_per_column = (surface_depth_chunks + 2).max(1) as usize;
    let max_columns = (loaded_chunk_cap / chunks_per_column).max(1) as f32;
    let side = max_columns.sqrt().floor() as i32;
    ((side - 1) / 2).max(4)
}

pub fn pick_block(
    camera_pos: Vec3,
    forward: Vec3,
    max_dist: f32,
    mut block_at: impl FnMut(i32, i32, i32) -> i8,
) -> Option<PickHit> {
    let dir_len_sq = forward.length_squared();
    if dir_len_sq <= 1.0e-8 {
        return None;
    }
    let dir = forward / dir_len_sq.sqrt();

    let mut block = IVec3::new(
        camera_pos.x.floor() as i32,
        camera_pos.y.floor() as i32,
        camera_pos.z.floor() as i32,
    );

    let step_x = if dir.x > 0.0 {
        1
    } else if dir.x < 0.0 {
        -1
    } else {
        0
    };
    let step_y = if dir.y > 0.0 {
        1
    } else if dir.y < 0.0 {
        -1
    } else {
        0
    };
    let step_z = if dir.z > 0.0 {
        1
    } else if dir.z < 0.0 {
        -1
    } else {
        0
    };
    if step_x == 0 && step_y == 0 && step_z == 0 {
        return None;
    }

    let mut t_max_x = if step_x > 0 {
        (block.x as f32 + 1.0 - camera_pos.x) / dir.x
    } else if step_x < 0 {
        (block.x as f32 - camera_pos.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if step_y > 0 {
        (block.y as f32 + 1.0 - camera_pos.y) / dir.y
    } else if step_y < 0 {
        (block.y as f32 - camera_pos.y) / dir.y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if step_z > 0 {
        (block.z as f32 + 1.0 - camera_pos.z) / dir.z
    } else if step_z < 0 {
        (block.z as f32 - camera_pos.z) / dir.z
    } else {
        f32::INFINITY
    };

    if t_max_x < 0.0 {
        t_max_x = 0.0;
    }
    if t_max_y < 0.0 {
        t_max_y = 0.0;
    }
    if t_max_z < 0.0 {
        t_max_z = 0.0;
    }

    let t_delta_x = if step_x != 0 {
        1.0 / dir.x.abs()
    } else {
        f32::INFINITY
    };
    let t_delta_y = if step_y != 0 {
        1.0 / dir.y.abs()
    } else {
        f32::INFINITY
    };
    let t_delta_z = if step_z != 0 {
        1.0 / dir.z.abs()
    } else {
        f32::INFINITY
    };

    let mut t = 0.0_f32;
    while t <= max_dist {
        let prev_block = block;
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            t = t_max_x;
            t_max_x += t_delta_x;
            block.x += step_x;
        } else if t_max_y <= t_max_z {
            t = t_max_y;
            t_max_y += t_delta_y;
            block.y += step_y;
        } else {
            t = t_max_z;
            t_max_z += t_delta_z;
            block.z += step_z;
        }

        if t > max_dist {
            break;
        }

        let block_id = block_at(block.x, block.y, block.z);
        if block_id >= 0 {
            return Some(PickHit {
                block,
                block_id,
                place: prev_block,
            });
        }
    }

    None
}

#[derive(Clone, Copy)]
pub struct PickHit {
    pub block: IVec3,
    pub block_id: i8,
    pub place: IVec3,
}

fn lod_div(dist: i32, base: i32) -> i32 {
    if dist <= base {
        4
    } else if dist <= base * 2 {
        8
    } else if dist <= base * 4 {
        16
    } else if dist <= base * 8 {
        32
    } else {
        64
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

fn request_priority(
    player_pos: Vec3,
    camera_forward: Vec3,
    coord: (i32, i32, i32),
    lod: i32,
) -> i32 {
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
    class: RequestClass,
) {
    let lod = pack_lod(mode, step);
    let priority = request_priority(player_pos, camera_forward, coord, lod);
    enqueue_request(
        req_queue,
        IVec3::new(coord.0, coord.1, coord.2),
        step,
        mode,
        true,
        class,
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

#[allow(clippy::too_many_arguments)]
pub fn build_stats_lines(
    stats: &GpuStats,
    loaded: &HashMap<(i32, i32, i32), i32>,
    requested: &HashMap<(i32, i32, i32), i32>,
    tps: u32,
    fps: u32,
    cpu_label: &str,
    gpu_label: &str,
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
    held_item_id: Option<i8>,
    active_break_strength: f32,
    mining_target: Option<IVec3>,
    mining_progress: f32,
    dirty_pending: usize,
    request_queue: RequestQueueStats,
    perf: DebugPerfSnapshot,
    adaptive_request_budget: i32,
    adaptive_pregen_budget: i32,
    adaptive_max_apply_per_tick: usize,
    adaptive_max_rebuilds_per_tick: usize,
    adaptive_draw_radius_cap: i32,
    worker_count: usize,
    thread_reports: &[String],
) -> Vec<String> {
    let seconds = world_time.as_secs_f32();
    let mesh_mem_mb = estimate_mesh_memory_mb(stats);
    let mem_fill = if mesh_memory_cap_mb > 0 {
        (mesh_mem_mb as f32 / mesh_memory_cap_mb as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let queue_total = request_queue.edit + request_queue.near + request_queue.far;
    let total_tris = stats.total_indices / 3;
    let visible_tris = stats.visible_indices / 3;
    let mut perf_rank = [
        ("render", perf.render_cpu_ms),
        ("stream", perf.stream_ms),
        ("apply", perf.apply_ms),
        ("player", perf.player_ms),
        ("mining", perf.mining_ms),
    ];
    perf_rank.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut lines = Vec::new();
    lines.push("F3 STATS+ (refresh 0.5s)".to_string());
    lines.push(format!(
        "time: {:>8.2}s | tps: {:>3} | fps: {:>3} | paused_stream: {} | paused_render: {}",
        seconds, tps, fps, pause_stream, pause_render
    ));
    lines.push(format!("hardware: cpu={} | gpu={}", cpu_label, gpu_label));
    lines.push(format!(
        "player: ({:>7.2}, {:>7.2}, {:>7.2}) | chunk: ({:>4}, {:>4}, {:>4})",
        player_pos.x, player_pos.y, player_pos.z, current_chunk.x, current_chunk.y, current_chunk.z,
    ));

    if let Some((block, block_id)) = looked_block {
        let block_name = block_name_by_id(block_id);
        let hardness = block_hardness(block_id);
        let breakable = can_break_block_with_item(block_id, held_item_id, active_break_strength);
        let break_time =
            block_break_time_with_item_seconds(block_id, held_item_id, active_break_strength);
        let mining_percent = if mining_target == Some(block) {
            break_time
                .map(|t| (mining_progress / t.max(0.001)).clamp(0.0, 1.0) * 100.0)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let break_time_label = break_time
            .map(|v| format!("{v:.2}s"))
            .unwrap_or_else(|| "blocked".to_string());
        lines.push(format!(
            "looking_at: ({:>5}, {:>5}, {:>5}) | block_id: {:>3} | type: {} | hardness:{:>4.2} | breakable:{} | break_time:{} | mining:{:>5.1}%",
            block.x,
            block.y,
            block.z,
            block_id,
            block_name,
            hardness,
            breakable,
            break_time_label,
            mining_percent,
        ));
    } else {
        lines.push("looking_at: none".to_string());
    }

    lines.push(format!(
        "chunks: loaded={:>6} requested={:>6} dirty_pending={:>5} | render_radius={} draw_radius={} draw_cap={}",
        loaded.len(),
        requested.len(),
        dirty_pending,
        base_render_radius,
        base_draw_radius,
        adaptive_draw_radius_cap,
    ));
    lines.push(format!(
        "gpu: super_chunks={} visible_supers={} dirty_supers={} pending={} queue={}",
        stats.super_chunks,
        stats.visible_supers,
        stats.dirty_supers,
        stats.pending_updates,
        stats.pending_queue
    ));
    lines.push(format!(
        "geometry: tris(total/visible)={:>9}/{:>9} | vis_indices(raw/packed/all)={:>9}/{:>9}/{:>9} | draw_calls(vis/total)={:>4}/{:>4}",
        total_tris,
        visible_tris,
        stats.visible_raw_indices,
        stats.visible_packed_indices,
        stats.visible_indices,
        stats.visible_draw_calls_est,
        stats.total_draw_calls_est,
    ));
    lines.push(format!(
        "queues: req(edit/near/far/total)={:>4}/{:>4}/{:>4}/{:>4} inflight_chunks={:>5} closed={} | budgets req/pregen/apply/rebuild={:>3}/{:>3}/{:>3}/{:>3}",
        request_queue.edit,
        request_queue.near,
        request_queue.far,
        queue_total,
        request_queue.inflight_chunks,
        request_queue.closed,
        adaptive_request_budget,
        adaptive_pregen_budget,
        adaptive_max_apply_per_tick,
        adaptive_max_rebuilds_per_tick,
    ));
    lines.push(format!(
        "memory: [{}] mesh_est={:>6}MB cap={:>6}MB | loaded_cap={}",
        progress_bar(mem_fill, 32),
        mesh_mem_mb,
        mesh_memory_cap_mb,
        loaded_chunk_cap
    ));
    lines.push(format!(
        "perf(ms ema): tick={:>6.2} render={:>6.2} stream={:>6.2} apply={:>6.2} player={:>6.2} mining={:>6.2}",
        perf.tick_cpu_ms,
        perf.render_cpu_ms,
        perf.stream_ms,
        perf.apply_ms,
        perf.player_ms,
        perf.mining_ms,
    ));
    lines.push(format!(
        "bottleneck_now: #1 {} {:>6.2}ms | #2 {} {:>6.2}ms | #3 {} {:>6.2}ms",
        perf_rank[0].0,
        perf_rank[0].1,
        perf_rank[1].0,
        perf_rank[1].1,
        perf_rank[2].0,
        perf_rank[2].1,
    ));
    lines.push(format!(
        "threading: workers={} | break_strength={:.2} | held_item={:?}",
        worker_count, active_break_strength, held_item_id
    ));
    for report in thread_reports {
        lines.push(format!("thread: {}", report));
    }

    if pregen_active {
        let cols_fill = if pregen_total_columns > 0 {
            (pregen_columns_done as f32 / pregen_total_columns as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        lines.push(format!(
            "pregen: radius={:>4}/{:>4} | columns [{}] {:>7}/{:<7}",
            pregen_ring_r,
            pregen_radius_chunks,
            progress_bar(cols_fill, 24),
            pregen_columns_done,
            pregen_total_columns
        ));
        lines.push(format!(
            "pregen chunks: created={} requested={} est_total={}",
            pregen_chunks_created, pregen_chunks_requested, pregen_est_chunks_total
        ));
    }

    lines
}

#[allow(clippy::too_many_arguments, dead_code)]
pub fn print_stats(
    stats: &GpuStats,
    loaded: &HashMap<(i32, i32, i32), i32>,
    requested: &HashMap<(i32, i32, i32), i32>,
    tps: u32,
    fps: u32,
    cpu_label: &str,
    gpu_label: &str,
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
    held_item_id: Option<i8>,
    active_break_strength: f32,
    mining_target: Option<IVec3>,
    mining_progress: f32,
    dirty_pending: usize,
    request_queue: RequestQueueStats,
    perf: DebugPerfSnapshot,
    adaptive_request_budget: i32,
    adaptive_pregen_budget: i32,
    adaptive_max_apply_per_tick: usize,
    adaptive_max_rebuilds_per_tick: usize,
    adaptive_draw_radius_cap: i32,
    worker_count: usize,
    thread_reports: &[String],
) {
    let lines = build_stats_lines(
        stats,
        loaded,
        requested,
        tps,
        fps,
        cpu_label,
        gpu_label,
        current_chunk,
        player_pos,
        world_time,
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
        held_item_id,
        active_break_strength,
        mining_target,
        mining_progress,
        dirty_pending,
        request_queue,
        perf,
        adaptive_request_budget,
        adaptive_pregen_budget,
        adaptive_max_apply_per_tick,
        adaptive_max_rebuilds_per_tick,
        adaptive_draw_radius_cap,
        worker_count,
        thread_reports,
    );
    print!("\x1B[2J\x1B[H");
    for line in lines {
        println!("{line}");
    }
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

fn estimate_mesh_memory_bytes(stats: &GpuStats) -> u64 {
    let vertex_bytes = stats.total_raw_vertices_capacity
        * std::mem::size_of::<ChunkVertex>() as u64
        + stats.total_packed_vertices_capacity * std::mem::size_of::<PackedFarVertex>() as u64;
    let index_bytes = stats.total_indices * std::mem::size_of::<u32>() as u64;
    vertex_bytes + index_bytes
}

pub fn estimate_mesh_memory_mb(stats: &GpuStats) -> usize {
    (estimate_mesh_memory_bytes(stats) / (1024 * 1024)) as usize
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
    let mesh_cap_bytes = mesh_memory_cap_mb.saturating_mul(1024 * 1024) as u64;
    let mut mesh_mem_bytes = estimate_mesh_memory_bytes(&gpu.stats());
    if loaded.len() <= loaded_chunk_cap && mesh_mem_bytes <= mesh_cap_bytes {
        return;
    }

    let mut eviction_heap: BinaryHeap<(i64, ChunkKey)> = BinaryHeap::with_capacity(loaded.len());
    for &coord in loaded.keys() {
        let dx = (coord.0 - current_chunk.x) as i64;
        let dy = (coord.1 - current_chunk.y) as i64;
        let dz = (coord.2 - current_chunk.z) as i64;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        eviction_heap.push((dist_sq, coord));
    }

    while let Some((_, coord)) = eviction_heap.pop() {
        if loaded.len() <= loaded_chunk_cap && mesh_mem_bytes <= mesh_cap_bytes {
            break;
        }
        if !loaded.contains_key(&coord) {
            continue;
        }
        let chunk_coord = IVec3::new(coord.0, coord.1, coord.2);
        let removed_bytes = gpu.chunk_memory_bytes(chunk_coord);
        gpu.remove_chunk(chunk_coord);
        loaded.remove(&coord);
        requested.remove(&coord);
        if let Some(bytes) = removed_bytes {
            mesh_mem_bytes = mesh_mem_bytes.saturating_sub(bytes);
        } else {
            mesh_mem_bytes = estimate_mesh_memory_bytes(&gpu.stats());
        }
    }
}

fn worker_result_key_and_rev(result: &WorkerResult) -> (ChunkKey, Option<u64>) {
    match result {
        WorkerResult::Raw { mesh, dirty_rev } => {
            ((mesh.coord.x, mesh.coord.y, mesh.coord.z), *dirty_rev)
        }
        WorkerResult::Packed { packed, dirty_rev } => {
            ((packed.coord.x, packed.coord.y, packed.coord.z), *dirty_rev)
        }
    }
}

fn apply_worker_result(
    gpu: &mut Gpu,
    result: WorkerResult,
    dirty_chunks: &DirtyChunks,
    edited_chunk_ranges: &EditedChunkRanges,
    request_queue: &SharedRequestQueue,
    loaded: &mut HashMap<ChunkKey, i32>,
    requested: &mut HashMap<ChunkKey, i32>,
    pregen_center_chunk: IVec3,
    pregen_radius_chunks: i32,
    pregen_chunks_created: &mut usize,
) -> bool {
    let (k, dirty_rev) = worker_result_key_and_rev(&result);

    let mut clear_edit_range = false;
    {
        let mut dirty_guard = dirty_chunks.lock().unwrap();
        let current_state = dirty_guard.get(&k).copied();
        let current_rev = current_state.and_then(dirty_state_rev);

        if current_rev != dirty_rev {
            // Result does not match the latest chunk revision snapshot.
            return false;
        }

        if let Some(state) = current_state
            && state > 0
        {
            // Pending dirty edit resolved; keep revision but mark it clean.
            dirty_guard.insert(k, -state);
            clear_edit_range = true;
        }
    }

    if clear_edit_range {
        edited_chunk_ranges.lock().unwrap().remove(&k);
        request_chunk_remesh_lighting(request_queue, IVec3::new(k.0, k.1, k.2));
    }

    let lod = match result {
        WorkerResult::Raw { mesh, .. } => {
            gpu.upsert_chunk(
                mesh.coord,
                mesh.center,
                mesh.radius,
                mesh.vertices,
                mesh.indices,
            );
            pack_lod(mesh.mode, mesh.step)
        }
        WorkerResult::Packed { packed, .. } => {
            gpu.upsert_chunk_packed(
                packed.coord,
                packed.center,
                packed.radius,
                packed.vertices,
                packed.indices,
            );
            pack_lod(packed.mode, packed.step)
        }
    };
    let was_loaded = loaded.insert(k, lod).is_some();
    if !was_loaded && in_pregen_bounds(k, pregen_center_chunk, pregen_radius_chunks) {
        *pregen_chunks_created += 1;
    }
    requested.remove(&k);
    true
}

pub fn apply_stream_results(
    gpu: &mut Gpu,
    rx_res: &mpsc::Receiver<WorkerResult>,
    dirty_chunks: &DirtyChunks,
    dirty_pending_hint: usize,
    edited_chunk_ranges: &EditedChunkRanges,
    request_queue: &SharedRequestQueue,
    loaded: &mut HashMap<ChunkKey, i32>,
    requested: &mut HashMap<ChunkKey, i32>,
    player_pos: Vec3,
    pregen_center_chunk: IVec3,
    pregen_radius_chunks: i32,
    pregen_chunks_created: &mut usize,
    loaded_chunk_cap: usize,
    mesh_memory_cap_mb: usize,
    max_apply_per_tick: usize,
    max_rebuilds_per_tick: usize,
    prioritize_dirty: bool,
) {
    let fast_lane_enabled = prioritize_dirty && dirty_pending_hint > 0;
    let mut deferred_non_dirty: Vec<WorkerResult> = Vec::with_capacity(max_apply_per_tick);
    let mut pending_dirty_keys: HashSet<ChunkKey> = if fast_lane_enabled {
        dirty_chunks
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(&k, &state)| if state > 0 { Some(k) } else { None })
            .collect()
    } else {
        HashSet::new()
    };

    if fast_lane_enabled {
        let fast_lane_budget = max_apply_per_tick.max(12);
        let mut dirty_applied = 0usize;
        let mut scanned = 0usize;
        let scan_limit = max_apply_per_tick.saturating_mul(6).max(32);

        while dirty_applied < fast_lane_budget
            && scanned < scan_limit
            && deferred_non_dirty.len() < max_apply_per_tick
        {
            let Ok(result) = rx_res.try_recv() else {
                break;
            };
            scanned += 1;
            let (k, _) = worker_result_key_and_rev(&result);
            let pending_now = pending_dirty_keys.contains(&k);
            if pending_now {
                if apply_worker_result(
                    gpu,
                    result,
                    dirty_chunks,
                    edited_chunk_ranges,
                    request_queue,
                    loaded,
                    requested,
                    pregen_center_chunk,
                    pregen_radius_chunks,
                    pregen_chunks_created,
                ) {
                    pending_dirty_keys.remove(&k);
                    dirty_applied += 1;
                }
            } else {
                deferred_non_dirty.push(result);
            }
        }
    }

    let mut budget = max_apply_per_tick;
    for result in deferred_non_dirty {
        if budget == 0 {
            break;
        }
        if apply_worker_result(
            gpu,
            result,
            dirty_chunks,
            edited_chunk_ranges,
            request_queue,
            loaded,
            requested,
            pregen_center_chunk,
            pregen_radius_chunks,
            pregen_chunks_created,
        ) {
            budget -= 1;
        }
    }

    while budget > 0 {
        let Ok(result) = rx_res.try_recv() else {
            break;
        };
        if apply_worker_result(
            gpu,
            result,
            dirty_chunks,
            edited_chunk_ranges,
            request_queue,
            loaded,
            requested,
            pregen_center_chunk,
            pregen_radius_chunks,
            pregen_chunks_created,
        ) {
            budget -= 1;
        }
    }

    let dirty_pending_after = fast_lane_enabled
        && (!pending_dirty_keys.is_empty()
            || dirty_chunks.lock().unwrap().values().any(|state| *state > 0));
    let rebuild_budget = if prioritize_dirty && dirty_pending_after {
        max_rebuilds_per_tick.max(8)
    } else {
        max_rebuilds_per_tick
    };
    gpu.rebuild_dirty_superchunks(player_pos, rebuild_budget);
    enforce_memory_budget(
        gpu,
        loaded,
        requested,
        chunk_coord_from_pos(player_pos),
        loaded_chunk_cap,
        mesh_memory_cap_mb,
    );
}

pub fn stream_tick(
    gpu: &mut Gpu,
    player_pos: Vec3,
    camera_forward: Vec3,
    current_chunk: &mut IVec3,
    ring_r: &mut i32,
    ring_i: &mut i32,
    stream_unload_counter: &mut u8,
    loaded: &mut HashMap<ChunkKey, i32>,
    requested: &mut HashMap<ChunkKey, i32>,
    column_height_cache: &mut HashMap<(i32, i32), i32>,
    req_queue: &SharedRequestQueue,
    _rx_res: &mpsc::Receiver<WorkerResult>,
    _dirty_chunks: &DirtyChunks,
    _edited_chunk_ranges: &EditedChunkRanges,
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
    _pregen_chunks_created: &mut usize,
    loaded_chunk_cap: usize,
    _mesh_memory_cap_mb: usize,
    _max_apply_per_tick: usize,
    _max_rebuilds_per_tick: usize,
) {
    let dynamic_radius = ((player_pos.y / CHUNK_SIZE as f32).max(0.0) as i32) * 2;
    let render_radius = (base_render_radius + dynamic_radius).min(max_render_radius);
    let stream_radius = render_radius
        .min(draw_radius + 8)
        .min(radius_cap_from_loaded_limit(
            surface_depth_chunks,
            loaded_chunk_cap,
        ))
        .max(initial_burst_radius.max(8));

    *stream_unload_counter = stream_unload_counter.wrapping_add(1);
    let new_chunk = chunk_coord_from_pos(player_pos);
    let chunk_changed = new_chunk != *current_chunk;
    if chunk_changed {
        *current_chunk = new_chunk;
        *ring_r = 0;
        *ring_i = 0;
        let keep_radius = stream_radius + 12;
        requested.retain(|(x, _, z), _| {
            in_world_chunk_bounds(*x, *z)
                && (*x - new_chunk.x).abs() <= keep_radius
                && (*z - new_chunk.z).abs() <= keep_radius
        });
        let cache_keep_radius = keep_radius + 48;
        column_height_cache.retain(|(cx, cz), _| {
            (*cx - new_chunk.x).abs() <= cache_keep_radius
                && (*cz - new_chunk.z).abs() <= cache_keep_radius
        });
        *emergency_budget = 16;
    }

    let mut max_height_cached = |cx: i32, cz: i32| -> i32 {
        if let Some(h) = column_height_cache.get(&(cx, cz)) {
            *h
        } else {
            let h = max_height_in_chunk(world_gen, cx, cz);
            column_height_cache.insert((cx, cz), h);
            h
        }
    };

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

    if !*pregen_active
        && requested.is_empty()
        && *ring_r > stream_radius
        && (player_pos
            - Vec3::new(
                current_chunk.x as f32 * CHUNK_SIZE as f32,
                current_chunk.y as f32 * CHUNK_SIZE as f32,
                current_chunk.z as f32 * CHUNK_SIZE as f32,
            ))
        .length_squared()
            < 4.0
    {
        return;
    }

    if *pregen_active {
        // Keep headroom for near-player requests so far pregen cannot starve them.
        let near_reserve = (max_inflight / 4).clamp(64, 512);
        let pregen_inflight_cap = max_inflight.saturating_sub(near_reserve).max(1);
        let mut budget = pregen_budget_per_tick.max(0);
        let pregen_lod = pack_lod(MeshMode::SurfaceOnly, 16);
        while budget > 0 && *pregen_ring_r <= pregen_radius_chunks {
            if requested.len() >= pregen_inflight_cap {
                break;
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
                if budget <= 0 || requested.len() >= pregen_inflight_cap {
                    break;
                }
                let coord = (cx, cy, cz);
                let step = 16;
                let mode = MeshMode::SurfaceOnly;
                let lod = pregen_lod;
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
                        RequestClass::Far,
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

    let full_lod = pack_lod(MeshMode::Full, 1);

    // Safe column: 2x2 chunks around player are always requested
    for dz in 0..=1 {
        for dx in 0..=1 {
            let cx = current_chunk.x + dx;
            let cz = current_chunk.z + dz;
            if !in_world_chunk_bounds(cx, cz) {
                continue;
            }
            let column_dist = (cx - current_chunk.x).abs().max((cz - current_chunk.z).abs());
            let surface_y = max_height_cached(cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let (y_start, y_end) =
                column_stream_y_range(surface_chunk_y, current_chunk.y, surface_depth_chunks, column_dist);
            for dy in y_start..=y_end {
                let coord = (cx, dy, cz);
                let step = 1;
                let mode = MeshMode::Full;
                let lod = full_lod;
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
                        RequestClass::Near,
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
                let column_dist = (cx - current_chunk.x).abs().max((cz - current_chunk.z).abs());
                let surface_y = max_height_cached(cx, cz);
                let surface_chunk_y = world_y_to_chunk_y(surface_y);
                let (y_start, y_end) = column_stream_y_range(
                    surface_chunk_y,
                    current_chunk.y,
                    surface_depth_chunks,
                    column_dist,
                );
                for dy in y_start..=y_end {
                    if count <= 0 || requested.len() >= max_inflight {
                        break 'emergency;
                    }
                    let coord = (cx, dy, cz);
                    let step = 1;
                    let mode = MeshMode::Full;
                    let lod = full_lod;
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
                            RequestClass::Near,
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
            let column_dist = (cx - current_chunk.x).abs().max((cz - current_chunk.z).abs());
            if column_dist > stream_radius {
                continue;
            }
            let surface_y = max_height_cached(cx, cz);
            let surface_chunk_y = world_y_to_chunk_y(surface_y);
            let (y_start, y_end) =
                column_stream_y_range(surface_chunk_y, current_chunk.y, surface_depth_chunks, column_dist);

            for dy in y_start..=y_end {
                if requested.len() >= max_inflight {
                    return;
                }
                let coord = (cx, dy, cz);
                let step = 1;
                let mode = MeshMode::Full;
                let lod = full_lod;
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
                        RequestClass::Near,
                    );
                }
            }
        }
    }

    let mut budget = budget;
    while budget > 0 && *ring_r <= stream_radius {
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
        let column_dist = (cx - current_chunk.x).abs().max((cz - current_chunk.z).abs());
        if column_dist > stream_radius {
            continue;
        }
        let surface_y = max_height_cached(cx, cz);
        let surface_chunk_y = world_y_to_chunk_y(surface_y);
        let (y_start, y_end) =
            column_stream_y_range(surface_chunk_y, current_chunk.y, surface_depth_chunks, column_dist);

        for dy in y_start..=y_end {
            if budget <= 0 {
                break;
            }
            let coord = (cx, dy, cz);
            if column_dist > lod_mid_radius * 2 && ((cx + cz) & 1) != 0 {
                continue;
            }
            let mode = if column_dist <= lod_near_radius {
                MeshMode::Full
            } else if column_dist <= lod_mid_radius {
                MeshMode::SurfaceSides
            } else {
                MeshMode::SurfaceOnly
            };
            let step = if mode == MeshMode::Full {
                1
            } else {
                lod_div(column_dist, lod_mid_radius)
            };
            let lod = pack_lod(mode, step);
            let key = coord;
            if needs_chunk_request(requested, loaded, key, lod) {
                let class = if column_dist <= lod_mid_radius {
                    RequestClass::Near
                } else {
                    RequestClass::Far
                };
                submit_chunk_request(
                    req_queue,
                    requested,
                    player_pos,
                    camera_forward,
                    key,
                    step,
                    mode,
                    class,
                );
                budget -= 1;
                if requested.len() >= max_inflight {
                    return;
                }
            }
        }

        // surface chunk already included in y_start..=y_end
    }

    let should_run_unload_scan =
        chunk_changed || loaded.len() > loaded_chunk_cap || (*stream_unload_counter & 0b111) == 0;
    if should_run_unload_scan {
        let to_remove: Vec<(i32, i32, i32)> = loaded
            .keys()
            .filter(|(x, y, z)| {
                if !in_world_chunk_bounds(*x, *z) {
                    return true;
                }
                let dx = x - current_chunk.x;
                let _dy = y - current_chunk.y;
                let dz = z - current_chunk.z;
                dx.abs() > stream_radius + 8 || dz.abs() > stream_radius + 8
            })
            .cloned()
            .collect();
        for coord in to_remove {
            gpu.remove_chunk(IVec3::new(coord.0, coord.1, coord.2));
            loaded.remove(&coord);
            requested.remove(&coord);
        }
    }

    // no super chunks
}
