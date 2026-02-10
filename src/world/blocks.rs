use crate::colorthing::ColorThing;
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

pub fn default_blocks(tiles_x: u32) -> Vec<BlockTexture> {
    let no_tint = no_tint();
    let grass_side_tint = ColorThing::new(0.62, 0.82, 0.35);
    let grass_top_tint = ColorThing::new(23.0 / 255.0, 230.0 / 255.0, 37.0 / 255.0);
    let leaves_tint = ColorThing::new(0.42, 0.92, 0.38);

    let stone_tile = tile_index_1based(2, 1, tiles_x);
    let stone = BlockTexture {
        colors: [no_tint; 6],
        tiles: [stone_tile; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let dirt_tile = tile_index_1based(3, 1, tiles_x);
    let dirt = BlockTexture {
        colors: [no_tint; 6],
        tiles: [dirt_tile; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let grass = BlockTexture {
        colors: [
            grass_side_tint, // +X
            grass_side_tint, // -X
            grass_top_tint,  // +Y (top)
            no_tint,         // -Y (bottom)
            grass_side_tint, // +Z
            grass_side_tint, // -Z
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
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let log_side_tile = tile_index_1based(5, 2, tiles_x);
    let log_cap_tile = tile_index_1based(6, 2, tiles_x);
    let log = BlockTexture {
        colors: [no_tint; 6],
        tiles: [
            log_side_tile, // +X
            log_side_tile, // -X
            log_cap_tile,  // +Y
            log_cap_tile,  // -Y
            log_side_tile, // +Z
            log_side_tile, // -Z
        ],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let leaves_tile = tile_index_1based(5, 4, tiles_x);
    let leaves = BlockTexture {
        colors: [leaves_tint; 6],
        tiles: [leaves_tile; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [2, 2, 2, 2, 2, 2],
    };

    vec![stone, dirt, grass, log, leaves]
}

pub fn build_block_index(tiles_x: u32) -> Vec<BlockIndexEntry> {
    let mut out = Vec::with_capacity(BLOCK_COUNT);
    out.extend_from_slice(&[
        BlockIndexEntry {
            block_id: BLOCK_STONE,
            item_id: 1,
            name: "block_stone",
            texture_index: tile_index_1based(2, 1, tiles_x),
        },
        BlockIndexEntry {
            block_id: BLOCK_DIRT,
            item_id: 2,
            name: "block_dirt",
            texture_index: tile_index_1based(3, 1, tiles_x),
        },
        BlockIndexEntry {
            block_id: BLOCK_GRASS,
            item_id: 3,
            name: "block_grass",
            texture_index: tile_index_1based(9, 3, tiles_x),
        },
        BlockIndexEntry {
            block_id: BLOCK_LOG,
            item_id: 4,
            name: "block_log",
            texture_index: tile_index_1based(5, 2, tiles_x),
        },
        BlockIndexEntry {
            block_id: BLOCK_LEAVES,
            item_id: 5,
            name: "block_leaves",
            texture_index: tile_index_1based(5, 4, tiles_x),
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

pub fn block_item_id_by_id(id: i8) -> Option<u16> {
    match id {
        x if x == BLOCK_STONE as i8 => Some(1),
        x if x == BLOCK_DIRT as i8 => Some(2),
        x if x == BLOCK_GRASS as i8 => Some(3),
        x if x == BLOCK_LOG as i8 => Some(4),
        x if x == BLOCK_LEAVES as i8 => Some(5),
        _ => None,
    }
}

fn no_tint() -> ColorThing {
    ColorThing::new(1.0, 1.0, 1.0)
}

fn tile_index_1based(x: u32, y: u32, tiles_x: u32) -> u32 {
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    x0 + y0 * tiles_x
}
