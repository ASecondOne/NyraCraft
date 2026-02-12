use glam::{IVec3, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use winit::event::MouseButton;

use crate::app::streaming::{
    DirtyChunks, EditedChunkRanges, SharedRequestQueue, request_chunk_remesh_now,
};
use crate::world::CHUNK_SIZE;
use crate::world::blocks::{BLOCK_LEAVES, BLOCK_LOG, BLOCK_STONE};
use crate::world::worldgen::WorldGen;

type CoordKey = (i32, i32, i32);
type ChunkKey = (i32, i32, i32);

#[derive(Default)]
pub struct EditedBlockStore {
    by_block: HashMap<CoordKey, i8>,
    by_chunk: HashMap<ChunkKey, HashMap<CoordKey, i8>>,
}

pub type EditedBlocks = Arc<RwLock<EditedBlockStore>>;

pub fn new_edited_blocks() -> EditedBlocks {
    Arc::new(RwLock::new(EditedBlockStore::default()))
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
    if let Some(id) = edited_blocks.read().unwrap().get(x, y, z) {
        return id;
    }
    world_gen.block_id_full_at(x, y, z)
}

pub fn handle_block_mouse_input(
    button: MouseButton,
    looked_hit: Option<(IVec3, i8, IVec3)>,
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    edited_chunk_ranges: &EditedChunkRanges,
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
    player_pos: Vec3,
    player_height: f32,
    player_radius: f32,
) {
    let Some((target_block, target_id, place_block)) = looked_hit else {
        return;
    };

    match button {
        MouseButton::Left => {
            if block_id_with_edits(
                world_gen,
                edited_blocks,
                target_block.x,
                target_block.y,
                target_block.z,
            ) >= 0
            {
                edited_blocks.write().unwrap().set(
                    target_block.x,
                    target_block.y,
                    target_block.z,
                    -1,
                );
                mark_edited_chunk_range(edited_chunk_ranges, target_block);
                let tree_like = target_id == BLOCK_LOG as i8 || target_id == BLOCK_LEAVES as i8;
                invalidate_block_edit(dirty_chunks, request_queue, target_block, tree_like);
            }
        }
        MouseButton::Right => {
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
                    BLOCK_STONE as i8,
                );
                mark_edited_chunk_range(edited_chunk_ranges, place_block);
                invalidate_block_edit(dirty_chunks, request_queue, place_block, false);
            }
        }
        _ => {}
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
    let coord = world_to_chunk_coord(block);
    let key = (coord.x, coord.y, coord.z);
    let mut ranges = edited_chunk_ranges.lock().unwrap();
    let entry = ranges.entry(key).or_insert((block.y, block.y));
    entry.0 = entry.0.min(block.y);
    entry.1 = entry.1.max(block.y);
}

fn invalidate_block_edit(
    dirty_chunks: &DirtyChunks,
    request_queue: &SharedRequestQueue,
    block: IVec3,
    tree_like: bool,
) {
    const TREE_XZ_MARGIN: i32 = 3;

    let coord = world_to_chunk_coord(block);
    let chunk_min = IVec3::new(
        coord.x * CHUNK_SIZE - CHUNK_SIZE / 2,
        coord.y * CHUNK_SIZE - CHUNK_SIZE / 2,
        coord.z * CHUNK_SIZE - CHUNK_SIZE / 2,
    );
    let local = block - chunk_min;

    let mut affected = vec![coord];
    if local.x == 0 {
        affected.push(coord - IVec3::X);
    } else if local.x == CHUNK_SIZE - 1 {
        affected.push(coord + IVec3::X);
    }
    if local.y == 0 {
        affected.push(coord - IVec3::Y);
    } else if local.y == CHUNK_SIZE - 1 {
        affected.push(coord + IVec3::Y);
    }
    if local.z == 0 {
        affected.push(coord - IVec3::Z);
    } else if local.z == CHUNK_SIZE - 1 {
        affected.push(coord + IVec3::Z);
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
                affected.push(coord + IVec3::new(*ox, 0, *oz));
            }
        }
    }

    let mut unique = HashSet::new();
    let mut dirty = dirty_chunks.lock().unwrap();
    for c in affected {
        if !unique.insert((c.x, c.y, c.z)) {
            continue;
        }
        let key = (c.x, c.y, c.z);
        let rev = dirty.entry(key).or_insert(0);
        *rev = rev.saturating_add(1);
        request_chunk_remesh_now(request_queue, c);
    }
}
