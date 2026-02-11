use crate::render::BlockTexture;

pub const DEFAULT_TILES_X: u32 = 16;
pub const BLOCK_STONE: usize = 0;
pub const BLOCK_DIRT: usize = 1;
pub const BLOCK_GRASS: usize = 2;
pub const BLOCK_LOG: usize = 3;
pub const BLOCK_LEAVES: usize = 4;
pub const BLOCK_COUNT: usize = 5;

#[derive(Clone, Copy)]
pub struct BlockIndexEntry {
    pub block_id: usize,
    pub item_id: u16,
    pub name: &'static str,
    pub texture_index: u32,
}

pub fn default_blocks(_tiles_x: u32) -> Vec<BlockTexture> {
    // Atlas slots are 1-based in filenames. We store zero-based tile indices.
    let tile_undefined = tile_from_slot(1);
    let tile_dirt = tile_from_slot(2);
    let tile_grass_top = tile_from_slot(3);
    let tile_grass_side = tile_from_slot(4);
    let tile_log_oak_top = tile_from_slot(5);
    let tile_log_oak = tile_from_slot(6);
    let tile_leave_oak = tile_from_slot(8);
    let tile_stone = tile_from_slot(9);

    let stone = BlockTexture {
        tiles: [tile_stone; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let dirt = BlockTexture {
        tiles: [tile_dirt; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let grass = BlockTexture {
        tiles: [
            tile_grass_side, // +X
            tile_grass_side, // -X
            tile_grass_top,  // +Y (top)
            tile_dirt,       // -Y (bottom)
            tile_grass_side, // +Z
            tile_grass_side, // -Z
        ],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [1, 1, 0, 0, 1, 1],
    };

    let log = BlockTexture {
        tiles: [
            tile_log_oak,      // +X
            tile_log_oak,      // -X
            tile_log_oak_top,  // +Y (top)
            tile_log_oak_top,  // -Y (bottom)
            tile_log_oak,      // +Z
            tile_log_oak,      // -Z
        ],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let leaves = BlockTexture {
        tiles: [tile_leave_oak; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [2, 2, 2, 2, 2, 2],
    };

    vec![stone, dirt, grass, log, leaves]
}

pub fn build_block_index(_tiles_x: u32) -> Vec<BlockIndexEntry> {
    let mut out = Vec::with_capacity(BLOCK_COUNT);
    out.extend_from_slice(&[
        BlockIndexEntry {
            block_id: BLOCK_STONE,
            item_id: 1,
            name: "block_stone",
            texture_index: tile_from_slot(1),
        },
        BlockIndexEntry {
            block_id: BLOCK_DIRT,
            item_id: 2,
            name: "block_dirt",
            texture_index: tile_from_slot(2),
        },
        BlockIndexEntry {
            block_id: BLOCK_GRASS,
            item_id: 3,
            name: "block_grass",
            texture_index: tile_from_slot(3),
        },
        BlockIndexEntry {
            block_id: BLOCK_LOG,
            item_id: 4,
            name: "block_log",
            texture_index: tile_from_slot(1),
        },
        BlockIndexEntry {
            block_id: BLOCK_LEAVES,
            item_id: 5,
            name: "block_leaves",
            texture_index: tile_from_slot(1),
        },
    ]);
    out
}

pub fn block_name_by_id(id: i8) -> &'static str {
    match id {
        x if x == BLOCK_STONE as i8 => "block_stone",
        x if x == BLOCK_DIRT as i8 => "block_dirt",
        x if x == BLOCK_GRASS as i8 => "block_grass",
        x if x == BLOCK_LOG as i8 => "block_log",
        x if x == BLOCK_LEAVES as i8 => "block_leaves",
        _ => "unknown",
    }
}

fn tile_from_slot(slot_1based: u32) -> u32 {
    slot_1based.saturating_sub(1)
}
