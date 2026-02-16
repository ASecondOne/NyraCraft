use crate::render::BlockTexture;

pub const DEFAULT_TILES_X: u32 = 16;
pub const BLOCK_STONE: usize = 0;
pub const BLOCK_DIRT: usize = 1;
pub const BLOCK_GRASS: usize = 2;
pub const BLOCK_LOG: usize = 3;
pub const BLOCK_LEAVES: usize = 4;
pub const BLOCK_PLANKS_OAK: usize = 5;
pub const BLOCK_CRAFTING_TABLE: usize = 6;
pub const BLOCK_GRAVEL: usize = 7;
pub const BLOCK_SAND: usize = 8;
pub const BLOCK_COUNT: usize = 9;
pub const ITEM_APPLE: i8 = BLOCK_COUNT as i8;
pub const ITEM_STICK: i8 = BLOCK_COUNT as i8 + 1;
pub const HOTBAR_SLOTS: usize = 9;
pub const HOTBAR_LOADOUT: [Option<i8>; HOTBAR_SLOTS] = [None; HOTBAR_SLOTS];
pub const DESTROY_STAGE_TILE_START: u32 = 16; // slot 17 in source textures
pub const DESTROY_STAGE_COUNT: u32 = 10;

#[derive(Clone, Copy)]
pub struct BlockIndexEntry {
    pub block_id: usize,
    pub item_id: u16,
    pub name: &'static str,
    pub texture_index: u32,
}

#[derive(Clone, Copy)]
pub struct ItemDef {
    pub item_id: i8,
    pub name: &'static str,
    pub place_block_id: Option<i8>,
    pub item_atlas_slot_1based: Option<u32>,
}

pub const ITEM_DEFS: [ItemDef; BLOCK_COUNT + 2] = [
    ItemDef {
        item_id: BLOCK_STONE as i8,
        name: "block_stone",
        place_block_id: Some(BLOCK_STONE as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_DIRT as i8,
        name: "block_dirt",
        place_block_id: Some(BLOCK_DIRT as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_GRASS as i8,
        name: "block_grass",
        place_block_id: Some(BLOCK_GRASS as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_LOG as i8,
        name: "block_log",
        place_block_id: Some(BLOCK_LOG as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_LEAVES as i8,
        name: "block_leaves",
        place_block_id: Some(BLOCK_LEAVES as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_PLANKS_OAK as i8,
        name: "block_planks_oak",
        place_block_id: Some(BLOCK_PLANKS_OAK as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_CRAFTING_TABLE as i8,
        name: "block_crafting_table",
        place_block_id: Some(BLOCK_CRAFTING_TABLE as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_GRAVEL as i8,
        name: "block_gravel",
        place_block_id: Some(BLOCK_GRAVEL as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: BLOCK_SAND as i8,
        name: "block_sand",
        place_block_id: Some(BLOCK_SAND as i8),
        item_atlas_slot_1based: None,
    },
    ItemDef {
        item_id: ITEM_APPLE,
        name: "item_apple",
        place_block_id: None,
        item_atlas_slot_1based: Some(1),
    },
    ItemDef {
        item_id: ITEM_STICK,
        name: "item_stick",
        place_block_id: None,
        item_atlas_slot_1based: Some(2),
    },
];

pub fn item_def_by_id(item_id: i8) -> Option<&'static ItemDef> {
    ITEM_DEFS.iter().find(|def| def.item_id == item_id)
}

pub fn item_name_by_id(item_id: i8) -> &'static str {
    item_def_by_id(item_id)
        .map(|def| def.name)
        .unwrap_or("unknown_item")
}

pub fn placeable_block_id_for_item(item_id: i8) -> Option<i8> {
    item_def_by_id(item_id).and_then(|def| def.place_block_id)
}

pub fn item_icon_tile_index(item_id: i8) -> Option<u32> {
    item_def_by_id(item_id).and_then(|def| {
        def.item_atlas_slot_1based
            .map(|slot| slot.saturating_sub(1))
    })
}

pub fn default_blocks(_tiles_x: u32) -> Vec<BlockTexture> {
    // Atlas slots are 1-based in filenames. We store zero-based tile indices.
    let _tile_undefined = tile_from_slot(1);
    let tile_dirt = tile_from_slot(2);
    let tile_grass_top = tile_from_slot(3);
    let tile_grass_side = tile_from_slot(4);
    let tile_log_oak_top = tile_from_slot(5);
    let tile_log_oak = tile_from_slot(6);
    let tile_leave_oak = tile_from_slot(8);
    let tile_stone = tile_from_slot(9);
    let tile_planks_oak = tile_from_slot(10);
    let tile_crafting_table_top = tile_from_slot(11);
    let tile_crafting_table_side = tile_from_slot(12);
    let tile_crafting_table_front = tile_from_slot(13);
    let tile_gravel = tile_from_slot(14);
    let tile_sand = tile_from_slot(15);

    let stone = BlockTexture {
        tiles: [tile_stone; 6],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [3, 3, 0, 0, 0, 0],
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
            tile_log_oak,     // +X
            tile_log_oak,     // -X
            tile_log_oak_top, // +Y (top)
            tile_log_oak_top, // -Y (bottom)
            tile_log_oak,     // +Z
            tile_log_oak,     // -Z
        ],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let leaves = BlockTexture {
        tiles: [tile_leave_oak; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [2, 2, 2, 2, 2, 2],
    };

    let planks_oak = BlockTexture {
        tiles: [tile_planks_oak; 6],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let crafting_table = BlockTexture {
        tiles: [
            tile_crafting_table_side,  // +X
            tile_crafting_table_side,  // -X
            tile_crafting_table_top,   // +Y (top)
            tile_crafting_table_side,  // -Y (bottom)
            tile_crafting_table_front, // +Z
            tile_crafting_table_side,  // -Z
        ],
        rotations: [3, 3, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let gravel = BlockTexture {
        tiles: [tile_gravel; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    let sand = BlockTexture {
        tiles: [tile_sand; 6],
        rotations: [0, 0, 0, 0, 0, 0],
        transparent_mode: [0, 0, 0, 0, 0, 0],
    };

    vec![
        stone,
        dirt,
        grass,
        log,
        leaves,
        planks_oak,
        crafting_table,
        gravel,
        sand,
    ]
}

pub fn build_block_index(_tiles_x: u32) -> Vec<BlockIndexEntry> {
    let mut out = Vec::with_capacity(BLOCK_COUNT);
    out.extend_from_slice(&[
        BlockIndexEntry {
            block_id: BLOCK_STONE,
            item_id: BLOCK_STONE as u16,
            name: "block_stone",
            texture_index: tile_from_slot(1),
        },
        BlockIndexEntry {
            block_id: BLOCK_DIRT,
            item_id: BLOCK_DIRT as u16,
            name: "block_dirt",
            texture_index: tile_from_slot(2),
        },
        BlockIndexEntry {
            block_id: BLOCK_GRASS,
            item_id: BLOCK_GRASS as u16,
            name: "block_grass",
            texture_index: tile_from_slot(3),
        },
        BlockIndexEntry {
            block_id: BLOCK_LOG,
            item_id: BLOCK_LOG as u16,
            name: "block_log",
            texture_index: tile_from_slot(1),
        },
        BlockIndexEntry {
            block_id: BLOCK_LEAVES,
            item_id: BLOCK_LEAVES as u16,
            name: "block_leaves",
            texture_index: tile_from_slot(1),
        },
        BlockIndexEntry {
            block_id: BLOCK_PLANKS_OAK,
            item_id: BLOCK_PLANKS_OAK as u16,
            name: "block_planks_oak",
            texture_index: tile_from_slot(10),
        },
        BlockIndexEntry {
            block_id: BLOCK_CRAFTING_TABLE,
            item_id: BLOCK_CRAFTING_TABLE as u16,
            name: "block_crafting_table",
            texture_index: tile_from_slot(13),
        },
        BlockIndexEntry {
            block_id: BLOCK_GRAVEL,
            item_id: BLOCK_GRAVEL as u16,
            name: "block_gravel",
            texture_index: tile_from_slot(14),
        },
        BlockIndexEntry {
            block_id: BLOCK_SAND,
            item_id: BLOCK_SAND as u16,
            name: "block_sand",
            texture_index: tile_from_slot(15),
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
        x if x == BLOCK_PLANKS_OAK as i8 => "block_planks_oak",
        x if x == BLOCK_CRAFTING_TABLE as i8 => "block_crafting_table",
        x if x == BLOCK_GRAVEL as i8 => "block_gravel",
        x if x == BLOCK_SAND as i8 => "block_sand",
        _ => "unknown",
    }
}

pub fn block_hardness(id: i8) -> f32 {
    match id {
        x if x == BLOCK_STONE as i8 => 2.0,
        x if x == BLOCK_GRASS as i8 => 1.5,
        x if x == BLOCK_DIRT as i8 => 1.2,
        x if x == BLOCK_LEAVES as i8 => 0.2,
        x if x == BLOCK_LOG as i8 => 1.8,
        x if x == BLOCK_CRAFTING_TABLE as i8 => 1.6,
        x if x == BLOCK_PLANKS_OAK as i8 => 1.4,
        x if x == BLOCK_GRAVEL as i8 => 1.1,
        x if x == BLOCK_SAND as i8 => 1.0,
        _ => 1.0,
    }
}

pub fn block_break_time_seconds(id: i8) -> f32 {
    block_hardness(id)
}

pub fn can_break_block_with_strength(id: i8, break_strength: f32) -> bool {
    let strength = break_strength.max(0.05);
    let hardness = block_hardness(id).max(0.05);
    hardness < strength * 2.0
}

pub fn block_break_time_with_strength_seconds(id: i8, break_strength: f32) -> Option<f32> {
    if !can_break_block_with_strength(id, break_strength) {
        return None;
    }
    let strength = break_strength.max(0.05);
    Some(block_break_time_seconds(id).max(0.05) / strength)
}

pub fn block_break_stage(progress_seconds: f32, break_time_seconds: f32) -> u32 {
    let total = DESTROY_STAGE_COUNT.max(1);
    let ratio = (progress_seconds / break_time_seconds.max(0.001)).clamp(0.0, 0.999_9);
    ((ratio * total as f32).floor() as u32).min(total - 1)
}

pub fn parse_block_id(name_or_id: &str) -> Option<i8> {
    let raw = name_or_id.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(id) = raw.parse::<usize>()
        && id < BLOCK_COUNT
    {
        return Some(id as i8);
    }

    let key = raw.to_ascii_lowercase();
    match key.as_str() {
        "stone" | "block_stone" => Some(BLOCK_STONE as i8),
        "dirt" | "block_dirt" => Some(BLOCK_DIRT as i8),
        "grass" | "block_grass" | "grass_block" => Some(BLOCK_GRASS as i8),
        "log" | "oak_log" | "block_log" => Some(BLOCK_LOG as i8),
        "leaves" | "leaf" | "oak_leaves" | "block_leaves" => Some(BLOCK_LEAVES as i8),
        "planks" | "oak_planks" | "planks_oak" | "block_planks_oak" => Some(BLOCK_PLANKS_OAK as i8),
        "crafting_table" | "crafting" | "block_crafting_table" => Some(BLOCK_CRAFTING_TABLE as i8),
        "gravel" | "block_gravel" => Some(BLOCK_GRAVEL as i8),
        "sand" | "block_sand" => Some(BLOCK_SAND as i8),
        _ => None,
    }
}

pub fn parse_item_id(name_or_id: &str) -> Option<i8> {
    let raw = name_or_id.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(id) = raw.parse::<i16>() {
        if let Ok(id) = i8::try_from(id)
            && item_def_by_id(id).is_some()
        {
            return Some(id);
        }
    }

    let key = raw.to_ascii_lowercase();
    match key.as_str() {
        "apple" | "item_apple" => Some(ITEM_APPLE),
        "stick" | "item_stick" => Some(ITEM_STICK),
        _ => parse_block_id(raw),
    }
}

fn tile_from_slot(slot_1based: u32) -> u32 {
    slot_1based.saturating_sub(1)
}
