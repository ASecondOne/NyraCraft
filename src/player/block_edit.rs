use glam::{IVec3, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use winit::event::MouseButton;

use crate::app::streaming::{
    DirtyChunks, EditedChunkRanges, SharedRequestQueue, request_chunk_remesh_decay,
    request_chunk_remesh_now,
};
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{BLOCK_LEAVES, BLOCK_LOG};
use crate::world::worldgen::WorldGen;

type CoordKey = (i32, i32, i32);
type ChunkKey = (i32, i32, i32);
const LEAF_DECAY_SEARCH_CAP: usize = 2048;
const LEAF_DECAY_BREAK_DELAY_TICKS: u64 = 4;
const FACE_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[derive(Clone, Copy)]
struct ScheduledLeafBreak {
    pos: IVec3,
    ready_tick: u64,
}

#[derive(Default)]
pub struct LeafDecayState {
    tick: u64,
    check_queue: VecDeque<IVec3>,
    check_pending: HashSet<CoordKey>,
    break_queue: VecDeque<ScheduledLeafBreak>,
    break_pending: HashSet<CoordKey>,
}

enum LeafDecayAction {
    None,
    Check(IVec3),
    Break(IVec3),
}

#[derive(Default)]
pub struct EditedBlockStore {
    by_block: HashMap<CoordKey, i8>,
    by_chunk: HashMap<ChunkKey, HashMap<CoordKey, i8>>,
}

pub type EditedBlocks = Arc<RwLock<EditedBlockStore>>;
pub type LeafDecayQueue = Arc<Mutex<LeafDecayState>>;

#[derive(Default, Clone, Copy)]
pub struct BlockEditResult {
    pub broke: Option<(IVec3, i8)>,
    pub placed: Option<(IVec3, i8)>,
}

pub fn new_edited_blocks() -> EditedBlocks {
    Arc::new(RwLock::new(EditedBlockStore::default()))
}

pub fn new_leaf_decay_queue() -> LeafDecayQueue {
    Arc::new(Mutex::new(LeafDecayState::default()))
}

impl EditedBlockStore {
    pub fn has_any_edits(&self) -> bool {
        !self.by_block.is_empty()
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> Option<i8> {
        self.by_block.get(&(x, y, z)).copied()
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, id: i8) {
        let key = (x, y, z);
        self.by_block.insert(key, id);
        let chunk = world_to_chunk_coord(IVec3::new(x, y, z));
        self.by_chunk
            .entry((chunk.x, chunk.y, chunk.z))
            .or_default()
            .insert(key, id);
    }

    pub fn collect_chunk_halo(&self, center: IVec3, halo: i32) -> HashMap<CoordKey, i8> {
        let mut out = HashMap::new();
        for dz in -halo..=halo {
            for dy in -halo..=halo {
                for dx in -halo..=halo {
                    let key = (center.x + dx, center.y + dy, center.z + dz);
                    let Some(chunk_map) = self.by_chunk.get(&key) else {
                        continue;
                    };
                    out.reserve(chunk_map.len());
                    for (coord, id) in chunk_map {
                        out.insert(*coord, *id);
                    }
                }
            }
        }
        out
    }

    pub fn chunk_override_y_range(&self, coord: IVec3) -> Option<(i32, i32)> {
        let chunk = self.by_chunk.get(&(coord.x, coord.y, coord.z))?;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        for &(.., y, _) in chunk.keys() {
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if min_y <= max_y {
            Some((min_y, max_y))
        } else {
            None
        }
    }
}

pub fn block_id_with_edits(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    x: i32,
    y: i32,
    z: i32,
) -> i8 {
    let store = edited_blocks.read().unwrap();
    block_id_with_store(world_gen, &store, x, y, z)
}

pub fn handle_block_mouse_input(
    button: MouseButton,
    looked_hit: Option<(IVec3, i8, IVec3)>,
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    leaf_decay_queue: &LeafDecayQueue,
    edited_chunk_ranges: &EditedChunkRanges,
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
    player_pos: Vec3,
    player_height: f32,
    player_radius: f32,
    selected_place_block: Option<i8>,
) -> BlockEditResult {
    let mut result = BlockEditResult::default();
    let Some((target_block, _target_id, place_block)) = looked_hit else {
        return result;
    };

    match button {
        MouseButton::Left => {
            let target_now = block_id_with_edits(
                world_gen,
                edited_blocks,
                target_block.x,
                target_block.y,
                target_block.z,
            );
            if target_now >= 0 {
                let tree_like = target_now == BLOCK_LOG as i8 || target_now == BLOCK_LEAVES as i8;
                edited_blocks.write().unwrap().set(
                    target_block.x,
                    target_block.y,
                    target_block.z,
                    -1,
                );
                mark_edited_chunk_range(edited_chunk_ranges, target_block);
                invalidate_block_edit(dirty_chunks, request_queue, target_block, tree_like, false);
                result.broke = Some((target_block, target_now));
                if tree_like {
                    queue_leaf_decay_neighbors(
                        world_gen,
                        edited_blocks,
                        leaf_decay_queue,
                        target_block,
                    );
                }
            }
        }
        MouseButton::Right => {
            let Some(place_block_id) = selected_place_block else {
                return result;
            };
            let place_id = block_id_with_edits(
                world_gen,
                edited_blocks,
                place_block.x,
                place_block.y,
                place_block.z,
            );
            if place_id < 0
                && world_gen.in_world_bounds(place_block.x, place_block.z)
                && !block_intersects_player(place_block, player_pos, player_height, player_radius)
            {
                edited_blocks.write().unwrap().set(
                    place_block.x,
                    place_block.y,
                    place_block.z,
                    place_block_id,
                );
                mark_edited_chunk_range(edited_chunk_ranges, place_block);
                invalidate_block_edit(dirty_chunks, request_queue, place_block, false, false);
                result.placed = Some((place_block, place_block_id));
                if place_block_id == BLOCK_LEAVES as i8
                    && !leaf_has_log_connection(world_gen, edited_blocks, place_block)
                {
                    edited_blocks.write().unwrap().set(
                        place_block.x,
                        place_block.y,
                        place_block.z,
                        -1,
                    );
                    mark_edited_chunk_range(edited_chunk_ranges, place_block);
                    invalidate_block_edit(dirty_chunks, request_queue, place_block, false, false);
                }
            }
        }
        _ => {}
    }
    result
}

pub fn tick_leaf_decay(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    leaf_decay_queue: &LeafDecayQueue,
    edited_chunk_ranges: &EditedChunkRanges,
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
) -> Option<IVec3> {
    match pop_leaf_decay_action(leaf_decay_queue) {
        LeafDecayAction::None => return None,
        LeafDecayAction::Check(candidate) => {
            if block_id_with_edits(
                world_gen,
                edited_blocks,
                candidate.x,
                candidate.y,
                candidate.z,
            ) != BLOCK_LEAVES as i8
            {
                return None;
            }
            if leaf_has_log_connection(world_gen, edited_blocks, candidate) {
                return None;
            }
            schedule_leaf_break(leaf_decay_queue, candidate);
            None
        }
        LeafDecayAction::Break(candidate) => {
            if block_id_with_edits(
                world_gen,
                edited_blocks,
                candidate.x,
                candidate.y,
                candidate.z,
            ) != BLOCK_LEAVES as i8
            {
                return None;
            }
            if leaf_has_log_connection(world_gen, edited_blocks, candidate) {
                return None;
            }

            edited_blocks
                .write()
                .unwrap()
                .set(candidate.x, candidate.y, candidate.z, -1);
            mark_edited_chunk_range(edited_chunk_ranges, candidate);
            invalidate_block_edit(dirty_chunks, request_queue, candidate, false, true);
            queue_leaf_decay_neighbors(world_gen, edited_blocks, leaf_decay_queue, candidate);
            Some(candidate)
        }
    }
}

fn block_intersects_player(
    block: IVec3,
    player_pos: Vec3,
    player_height: f32,
    player_radius: f32,
) -> bool {
    let block_min_x = block.x as f32;
    let block_max_x = block_min_x + 1.0;
    let block_min_y = block.y as f32;
    let block_max_y = block_min_y + 1.0;
    let block_min_z = block.z as f32;
    let block_max_z = block_min_z + 1.0;

    let player_min_x = player_pos.x - player_radius;
    let player_max_x = player_pos.x + player_radius;
    let player_min_y = player_pos.y;
    let player_max_y = player_pos.y + player_height;
    let player_min_z = player_pos.z - player_radius;
    let player_max_z = player_pos.z + player_radius;

    let overlaps_x = player_min_x < block_max_x && player_max_x > block_min_x;
    let overlaps_y = player_min_y < block_max_y && player_max_y > block_min_y;
    let overlaps_z = player_min_z < block_max_z && player_max_z > block_min_z;

    overlaps_x && overlaps_y && overlaps_z
}

fn world_to_chunk_coord(block: IVec3) -> IVec3 {
    let half = CHUNK_SIZE / 2;
    IVec3::new(
        ((block.x + half) as f32 / CHUNK_SIZE as f32).floor() as i32,
        ((block.y + half) as f32 / CHUNK_SIZE as f32).floor() as i32,
        ((block.z + half) as f32 / CHUNK_SIZE as f32).floor() as i32,
    )
}

fn mark_edited_chunk_range(edited_chunk_ranges: &EditedChunkRanges, block: IVec3) {
    mark_edited_chunk_ranges(edited_chunk_ranges, &[block]);
}

fn mark_edited_chunk_ranges(edited_chunk_ranges: &EditedChunkRanges, blocks: &[IVec3]) {
    if blocks.is_empty() {
        return;
    }
    let mut ranges = edited_chunk_ranges.lock().unwrap();
    for &block in blocks {
        let coord = world_to_chunk_coord(block);
        let key = (coord.x, coord.y, coord.z);
        let entry = ranges.entry(key).or_insert((block.y, block.y));
        entry.0 = entry.0.min(block.y);
        entry.1 = entry.1.max(block.y);
    }
}

fn invalidate_block_edit(
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
    block: IVec3,
    tree_like: bool,
    decay: bool,
) {
    invalidate_block_edits(dirty_chunks, request_queue, &[block], tree_like, decay);
}

fn invalidate_block_edits(
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
    blocks: &[IVec3],
    tree_like: bool,
    decay: bool,
) {
    if blocks.is_empty() {
        return;
    }

    let mut unique = HashSet::new();
    for &block in blocks {
        collect_affected_chunks(block, tree_like, &mut unique);
    }

    let mut dirty = dirty_chunks.lock().unwrap();
    for (x, y, z) in unique {
        let key = (x, y, z);
        let state = dirty.entry(key).or_insert(0);
        let next_rev = state.abs().saturating_add(1);
        *state = next_rev;
        let coord = IVec3::new(x, y, z);
        if decay {
            request_chunk_remesh_decay(request_queue, coord);
        } else {
            request_chunk_remesh_now(request_queue, coord);
        }
    }
}

fn collect_affected_chunks(block: IVec3, tree_like: bool, affected: &mut HashSet<ChunkKey>) {
    const TREE_XZ_MARGIN: i32 = 3;

    let coord = world_to_chunk_coord(block);
    let chunk_min = IVec3::new(
        coord.x * CHUNK_SIZE - CHUNK_SIZE / 2,
        coord.y * CHUNK_SIZE - CHUNK_SIZE / 2,
        coord.z * CHUNK_SIZE - CHUNK_SIZE / 2,
    );
    let local = block - chunk_min;

    affected.insert((coord.x, coord.y, coord.z));
    if local.x == 0 {
        let c = coord - IVec3::X;
        affected.insert((c.x, c.y, c.z));
    } else if local.x == CHUNK_SIZE - 1 {
        let c = coord + IVec3::X;
        affected.insert((c.x, c.y, c.z));
    }
    if local.y == 0 {
        let c = coord - IVec3::Y;
        affected.insert((c.x, c.y, c.z));
    } else if local.y == CHUNK_SIZE - 1 {
        let c = coord + IVec3::Y;
        affected.insert((c.x, c.y, c.z));
    }
    if local.z == 0 {
        let c = coord - IVec3::Z;
        affected.insert((c.x, c.y, c.z));
    } else if local.z == CHUNK_SIZE - 1 {
        let c = coord + IVec3::Z;
        affected.insert((c.x, c.y, c.z));
    }

    if tree_like {
        let mut x_offsets = vec![0];
        let mut z_offsets = vec![0];
        if local.x < TREE_XZ_MARGIN {
            x_offsets.push(-1);
        }
        if local.x >= CHUNK_SIZE - TREE_XZ_MARGIN {
            x_offsets.push(1);
        }
        if local.z < TREE_XZ_MARGIN {
            z_offsets.push(-1);
        }
        if local.z >= CHUNK_SIZE - TREE_XZ_MARGIN {
            z_offsets.push(1);
        }
        for ox in &x_offsets {
            for oz in &z_offsets {
                let c = coord + IVec3::new(*ox, 0, *oz);
                affected.insert((c.x, c.y, c.z));
            }
        }
    }
}

fn block_id_with_store(
    world_gen: &WorldGen,
    store: &EditedBlockStore,
    x: i32,
    y: i32,
    z: i32,
) -> i8 {
    if let Some(id) = store.get(x, y, z) {
        return id;
    }
    world_gen.block_id_full_at(x, y, z)
}

fn queue_leaf_decay_neighbors(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    leaf_decay_queue: &LeafDecayQueue,
    center: IVec3,
) {
    let store = edited_blocks.read().unwrap();
    for &(dx, dy, dz) in &FACE_NEIGHBORS {
        let candidate = IVec3::new(center.x + dx, center.y + dy, center.z + dz);
        if block_id_with_store(world_gen, &store, candidate.x, candidate.y, candidate.z)
            == BLOCK_LEAVES as i8
        {
            enqueue_leaf_decay_candidate(leaf_decay_queue, candidate);
        }
    }
}

fn enqueue_leaf_decay_candidate(leaf_decay_queue: &LeafDecayQueue, candidate: IVec3) {
    let mut state = leaf_decay_queue.lock().unwrap();
    let key = (candidate.x, candidate.y, candidate.z);
    if state.check_pending.insert(key) {
        state.check_queue.push_back(candidate);
    }
}

fn schedule_leaf_break(leaf_decay_queue: &LeafDecayQueue, candidate: IVec3) {
    let mut state = leaf_decay_queue.lock().unwrap();
    let key = (candidate.x, candidate.y, candidate.z);
    if !state.break_pending.insert(key) {
        return;
    }
    let ready_tick = state.tick.saturating_add(LEAF_DECAY_BREAK_DELAY_TICKS);
    state.break_queue.push_back(ScheduledLeafBreak {
        pos: candidate,
        ready_tick,
    });
}

fn pop_leaf_decay_action(leaf_decay_queue: &LeafDecayQueue) -> LeafDecayAction {
    let mut state = leaf_decay_queue.lock().unwrap();
    state.tick = state.tick.saturating_add(1);

    if let Some(front) = state.break_queue.front().copied()
        && front.ready_tick <= state.tick
    {
        state.break_queue.pop_front();
        state
            .break_pending
            .remove(&(front.pos.x, front.pos.y, front.pos.z));
        return LeafDecayAction::Break(front.pos);
    }

    let Some(next) = state.check_queue.pop_front() else {
        return LeafDecayAction::None;
    };
    state.check_pending.remove(&(next.x, next.y, next.z));
    LeafDecayAction::Check(next)
}

fn leaf_has_log_connection(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    start_leaf: IVec3,
) -> bool {
    let store = edited_blocks.read().unwrap();
    let mut visited = HashSet::<CoordKey>::new();
    let mut queue = VecDeque::from([start_leaf]);
    visited.insert((start_leaf.x, start_leaf.y, start_leaf.z));

    while let Some(pos) = queue.pop_front() {
        if visited.len() >= LEAF_DECAY_SEARCH_CAP {
            // Avoid long scans in dense canopies; keep leaves in this ambiguous case.
            return true;
        }

        for &(dx, dy, dz) in &FACE_NEIGHBORS {
            let next = IVec3::new(pos.x + dx, pos.y + dy, pos.z + dz);
            let next_id = block_id_with_store(world_gen, &store, next.x, next.y, next.z);
            if next_id == BLOCK_LOG as i8 {
                return true;
            }
            if next_id == BLOCK_LEAVES as i8 && visited.insert((next.x, next.y, next.z)) {
                queue.push_back(next);
            }
        }
    }

    false
}
