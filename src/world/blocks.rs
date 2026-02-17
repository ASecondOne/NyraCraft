use crate::render::BlockTexture;
use rand::Rng;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
pub const ITEM_FLINT_AXE: i8 = BLOCK_COUNT as i8 + 3;
pub const ITEM_FLINT_KNIFE: i8 = BLOCK_COUNT as i8 + 4;
pub const ITEM_FLINT_PICKAXE: i8 = BLOCK_COUNT as i8 + 5;
pub const ITEM_FLINT_SHOVEL: i8 = BLOCK_COUNT as i8 + 6;
pub const ITEM_FLINT_SWORD: i8 = BLOCK_COUNT as i8 + 7;
pub const HOTBAR_SLOTS: usize = 9;
pub const HOTBAR_LOADOUT: [Option<i8>; HOTBAR_SLOTS] = [
    Some(ITEM_FLINT_PICKAXE),
    Some(ITEM_FLINT_AXE),
    Some(ITEM_FLINT_SHOVEL),
    Some(ITEM_FLINT_SWORD),
    Some(ITEM_FLINT_KNIFE),
    None,
    None,
    None,
    None,
];
pub const DESTROY_STAGE_TILE_START: u32 = 16; // slot 17 in source textures
pub const DESTROY_STAGE_COUNT: u32 = 10;

const BLOCKS_JSON_DIR: &str = "src/content/blocks";
const ITEMS_JSON_DIR: &str = "src/content/items";
const BLOCK_ATLAS_PATH: &str = "src/texturing/atlas_output/atls_blocks.png";
const ITEM_ATLAS_PATH: &str = "src/texturing/atlas_output/atlas_items.png";
const ATLAS_TILE_SIZE: u32 = 16;
const BLOCK_ID_NAMESPACE: u32 = 1;
const ITEM_ID_NAMESPACE: u32 = 2;
const MISSING_TILE_INDEX: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolType {
    Hand,
    Pickaxe,
    Axe,
    Shovel,
    Knife,
    Sword,
}

#[derive(Clone, Debug)]
pub struct BlockIndexEntry {
    pub block_id: usize,
    pub item_id: u16,
    pub name: String,
    pub texture_index: u32,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ItemDef {
    pub item_id: i8,
    pub name: String,
    pub place_block_id: Option<i8>,
    pub item_atlas_slot_1based: Option<u32>,
    pub edible: bool,
    pub saturation: f32,
    pub effects: Vec<String>,
    pub breaktime: Option<f32>,
    pub durability: u16,
    pub max_stack_size: u8,
    pub tool_type: ToolType,
}

#[derive(Clone)]
struct BlockDef {
    id: usize,
    name: String,
    aliases: Vec<String>,
    hardness: f32,
    required_tool: Option<ToolType>,
    texture: BlockTexture,
    item_id: i8,
    item_atlas_slot_1based: Option<u32>,
    icon_tile_index: u32,
    drops: Vec<DropEntry>,
}

struct Registry {
    blocks: Vec<Option<BlockDef>>,
    block_textures: Vec<BlockTexture>,
    block_hardness: Vec<f32>,
    block_required_tools: Vec<Option<ToolType>>,
    block_index: Vec<BlockIndexEntry>,
    block_name_to_id: HashMap<String, i8>,
    items: Vec<ItemDef>,
    item_by_id: HashMap<i8, usize>,
    item_name_to_id: HashMap<String, i8>,
    placeable_block_to_item_id: HashMap<i8, i8>,
    block_namespace_to_id: HashMap<u32, i8>,
    item_namespace_to_id: HashMap<u32, i8>,
    max_item_tiles: u32,
}

#[derive(Clone)]
struct DropEntry {
    item_id: i8,
    min_count: u8,
    max_count: u8,
    chance: f32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum RawIdValue {
    Number(i64),
    Text(String),
}

#[derive(Debug, Deserialize, Clone)]
struct RawDropEntry {
    item_id: RawIdValue,
    #[serde(default)]
    count: Option<u8>,
    #[serde(default)]
    min_count: Option<u8>,
    #[serde(default)]
    max_count: Option<u8>,
    #[serde(default)]
    chance: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawBlockFile {
    id: RawIdValue,
    register_name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    hardness: Option<f32>,
    #[serde(default, alias = "preferred_tool", alias = "tool_type")]
    required_tool: Option<String>,
    #[serde(default)]
    texture_slot_1based: Option<u32>,
    #[serde(default)]
    textures_slot_1based: Option<[u32; 6]>,
    #[serde(default)]
    rotations: Option<[u32; 6]>,
    #[serde(default)]
    transparent_mode: Option<[u32; 6]>,
    #[serde(default)]
    item_id: Option<RawIdValue>,
    #[serde(default)]
    item_atlas_slot_1based: Option<u32>,
    #[serde(default)]
    icon_texture_slot_1based: Option<u32>,
    #[serde(default)]
    drop_item: Option<RawIdValue>,
    #[serde(default)]
    drop_items: Vec<RawDropEntry>,
}

#[derive(Debug, Deserialize)]
struct RawItemFile {
    id: RawIdValue,
    register_name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    place_block_id: Option<i8>,
    #[serde(default)]
    item_atlas_slot_1based: Option<u32>,
    #[serde(default)]
    edible: bool,
    #[serde(default)]
    saturation: f32,
    #[serde(default)]
    effects: Vec<String>,
    #[serde(default)]
    breaktime: Option<f32>,
    #[serde(default)]
    durability: Option<u16>,
    #[serde(default)]
    max_stack_size: Option<u8>,
    #[serde(default, alias = "tool")]
    tool_type: Option<String>,
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

fn registry_with_tiles_x(tiles_x: u32) -> &'static Registry {
    REGISTRY.get_or_init(|| load_registry(tiles_x.max(1)))
}

fn registry() -> &'static Registry {
    registry_with_tiles_x(DEFAULT_TILES_X)
}

fn sanitize_slot_1based(slot_1based: u32, max_tiles: u32) -> u32 {
    if slot_1based == 0 || slot_1based > max_tiles {
        eprintln!(
            "atlas slot {} out of bounds (valid 1..={}), using missing texture slot 1",
            slot_1based, max_tiles
        );
        MISSING_TILE_INDEX
    } else {
        slot_1based - 1
    }
}

fn sanitize_slot_opt(slot_1based: Option<u32>, max_tiles: u32) -> Option<u32> {
    slot_1based.map(|slot| sanitize_slot_1based(slot, max_tiles))
}

fn sanitize_slot_opt_1based(slot_1based: Option<u32>, max_tiles: u32) -> Option<u32> {
    sanitize_slot_opt(slot_1based, max_tiles).map(|slot_zero_based| slot_zero_based + 1)
}

fn missing_block_texture() -> BlockTexture {
    BlockTexture {
        tiles: [MISSING_TILE_INDEX; 6],
        rotations: [0; 6],
        transparent_mode: [0; 6],
    }
}

fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(OsStr::to_str)
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .collect();
    files.sort();
    files
}

fn load_json_defs<T: DeserializeOwned>(dir: &Path, label: &str) -> Vec<T> {
    let files = collect_json_files(dir);
    let mut out = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            eprintln!("failed to read {} def file {}", label, path.display());
            continue;
        };
        match serde_json::from_str::<T>(&text) {
            Ok(def) => out.push(def),
            Err(err) => eprintln!("failed to parse {} def {}: {}", label, path.display(), err),
        }
    }
    out
}

fn atlas_tile_capacity(path: &str, fallback_tiles_x: u32) -> u32 {
    let fallback = fallback_tiles_x.saturating_mul(fallback_tiles_x).max(1);
    let Ok((width, height)) = image::image_dimensions(path) else {
        return fallback;
    };
    if width < ATLAS_TILE_SIZE || height < ATLAS_TILE_SIZE {
        return fallback;
    }
    let tiles_x = width / ATLAS_TILE_SIZE;
    let tiles_y = height / ATLAS_TILE_SIZE;
    let total = tiles_x.saturating_mul(tiles_y);
    if total == 0 { fallback } else { total }
}

fn parse_scoped_id_text(raw: &str) -> Option<(u32, u32)> {
    let (ns_raw, local_raw) = raw.split_once(':')?;
    let namespace = ns_raw.trim().parse::<u32>().ok()?;
    let local = local_raw.trim().parse::<u32>().ok()?;
    if local == 0 {
        return None;
    }
    Some((namespace, local))
}

fn parse_tool_type(raw: &str) -> Option<ToolType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "hand" | "any" => Some(ToolType::Hand),
        "pickaxe" | "pick" => Some(ToolType::Pickaxe),
        "axe" | "hatchet" => Some(ToolType::Axe),
        "shovel" | "spade" => Some(ToolType::Shovel),
        "knife" => Some(ToolType::Knife),
        "sword" => Some(ToolType::Sword),
        _ => None,
    }
}

fn infer_item_tool_type(register_name: &str, aliases: &[String]) -> ToolType {
    let mut keys = Vec::with_capacity(aliases.len() + 1);
    keys.push(register_name.to_ascii_lowercase());
    keys.extend(aliases.iter().map(|a| a.to_ascii_lowercase()));

    if keys.iter().any(|k| k.contains("pickaxe") || k == "pickaxe") {
        ToolType::Pickaxe
    } else if keys
        .iter()
        .any(|k| k.contains("shovel") || k.contains("spade") || k == "shovel")
    {
        ToolType::Shovel
    } else if keys.iter().any(|k| k.contains("axe") || k == "axe") {
        ToolType::Axe
    } else if keys.iter().any(|k| k.contains("knife") || k == "knife") {
        ToolType::Knife
    } else if keys.iter().any(|k| k.contains("sword") || k == "sword") {
        ToolType::Sword
    } else {
        ToolType::Hand
    }
}

fn infer_block_required_tool(register_name: &str, aliases: &[String]) -> Option<ToolType> {
    let mut keys = Vec::with_capacity(aliases.len() + 1);
    keys.push(register_name.to_ascii_lowercase());
    keys.extend(aliases.iter().map(|a| a.to_ascii_lowercase()));

    if keys.iter().any(|k| {
        k.contains("stone")
            || k.contains("cobble")
            || k.contains("ore")
            || k == "rock"
            || k == "rocks"
    }) {
        Some(ToolType::Pickaxe)
    } else if keys.iter().any(|k| {
        k.contains("dirt")
            || k.contains("grass")
            || k.contains("sand")
            || k.contains("gravel")
            || k.contains("soil")
            || k.contains("clay")
    }) {
        Some(ToolType::Shovel)
    } else if keys.iter().any(|k| {
        k.contains("log")
            || k.contains("wood")
            || k.contains("planks")
            || k.contains("table")
            || k.contains("leaves")
            || k.contains("leaf")
    }) {
        Some(ToolType::Axe)
    } else {
        None
    }
}

fn parse_item_tool_type_or_default(raw: &RawItemFile) -> ToolType {
    if let Some(tool_raw) = raw.tool_type.as_deref() {
        if let Some(tool) = parse_tool_type(tool_raw) {
            return tool;
        }
        eprintln!(
            "item {} has invalid tool_type `{}`, using inferred/default",
            raw.register_name, tool_raw
        );
    }
    infer_item_tool_type(&raw.register_name, &raw.aliases)
}

fn parse_block_required_tool_or_default(raw: &RawBlockFile) -> Option<ToolType> {
    if let Some(tool_raw) = raw.required_tool.as_deref() {
        if let Some(tool) = parse_tool_type(tool_raw) {
            return (tool != ToolType::Hand).then_some(tool);
        }
        eprintln!(
            "block {} has invalid required_tool `{}`, using inferred/default",
            raw.register_name, tool_raw
        );
    }
    infer_block_required_tool(&raw.register_name, &raw.aliases)
}

fn parse_block_file_id(id: &RawIdValue, register_name: &str) -> Option<(usize, Option<u32>)> {
    match id {
        RawIdValue::Number(v) => {
            if *v < 0 {
                eprintln!("skipping block {}: id {} must be >= 0", register_name, v);
                return None;
            }
            Some((*v as usize, None))
        }
        RawIdValue::Text(s) => {
            if let Some((namespace, local)) = parse_scoped_id_text(s) {
                if namespace != BLOCK_ID_NAMESPACE {
                    eprintln!(
                        "skipping block {}: id {} uses unsupported namespace {}, expected {}",
                        register_name, s, namespace, BLOCK_ID_NAMESPACE
                    );
                    return None;
                }
                return Some(((local - 1) as usize, Some(local)));
            }

            if let Ok(raw_id) = s.trim().parse::<usize>() {
                return Some((raw_id, None));
            }

            eprintln!(
                "skipping block {}: id {} is not numeric or scoped like {}:n",
                register_name, s, BLOCK_ID_NAMESPACE
            );
            None
        }
    }
}

fn parse_item_file_id(id: &RawIdValue, register_name: &str) -> Option<(i8, Option<u32>)> {
    match id {
        RawIdValue::Number(v) => i8::try_from(*v).ok().map(|id| (id, None)).or_else(|| {
            eprintln!(
                "skipping item {}: id {} does not fit into i8",
                register_name, v
            );
            None
        }),
        RawIdValue::Text(s) => {
            if let Some((namespace, local)) = parse_scoped_id_text(s) {
                if namespace != ITEM_ID_NAMESPACE {
                    eprintln!(
                        "skipping item {}: id {} uses unsupported namespace {}, expected {}",
                        register_name, s, namespace, ITEM_ID_NAMESPACE
                    );
                    return None;
                }
                let raw = BLOCK_COUNT as i64 + local as i64 - 1;
                return i8::try_from(raw)
                    .ok()
                    .map(|id| (id, Some(local)))
                    .or_else(|| {
                        eprintln!(
                            "skipping item {}: scoped id {} exceeds i8 capacity",
                            register_name, s
                        );
                        None
                    });
            }

            if let Ok(raw_id) = s.trim().parse::<i64>() {
                return i8::try_from(raw_id).ok().map(|id| (id, None)).or_else(|| {
                    eprintln!(
                        "skipping item {}: id {} does not fit into i8",
                        register_name, raw_id
                    );
                    None
                });
            }

            eprintln!(
                "skipping item {}: id {} is not numeric or scoped like {}:n",
                register_name, s, ITEM_ID_NAMESPACE
            );
            None
        }
    }
}

fn resolve_drop_item_id(
    raw: &RawIdValue,
    block_namespace_to_id: &HashMap<u32, i8>,
    item_namespace_to_id: &HashMap<u32, i8>,
    placeable_block_to_item_id: &HashMap<i8, i8>,
    block_name_to_id: &HashMap<String, i8>,
    item_name_to_id: &HashMap<String, i8>,
) -> Option<i8> {
    match raw {
        RawIdValue::Number(v) => i8::try_from(*v).ok(),
        RawIdValue::Text(text) => {
            if let Some((namespace, local)) = parse_scoped_id_text(text) {
                return match namespace {
                    BLOCK_ID_NAMESPACE => block_namespace_to_id
                        .get(&local)
                        .and_then(|block_id| placeable_block_to_item_id.get(block_id).copied()),
                    ITEM_ID_NAMESPACE => item_namespace_to_id.get(&local).copied(),
                    _ => None,
                };
            }
            let key = text.trim().to_ascii_lowercase();
            if let Some(item_id) = item_name_to_id.get(&key).copied() {
                return Some(item_id);
            }
            if let Some(block_id) = block_name_to_id.get(&key).copied() {
                return placeable_block_to_item_id.get(&block_id).copied();
            }
            None
        }
    }
}

fn normalized_drop_entry(entry: &RawDropEntry, item_id: i8) -> Option<DropEntry> {
    let mut min_count = entry.min_count.unwrap_or(1);
    let mut max_count = entry.max_count.unwrap_or(min_count);
    if let Some(count) = entry.count {
        min_count = count;
        max_count = count;
    }
    if min_count == 0 && max_count == 0 {
        return None;
    }
    if max_count < min_count {
        std::mem::swap(&mut min_count, &mut max_count);
    }
    let chance = entry.chance.unwrap_or(1.0).clamp(0.0, 1.0);
    if chance <= 0.0 {
        return None;
    }
    Some(DropEntry {
        item_id,
        min_count,
        max_count,
        chance,
    })
}

fn insert_name_aliases(
    map: &mut HashMap<String, i8>,
    register_name: &str,
    aliases: &[String],
    id: i8,
) {
    map.insert(register_name.to_ascii_lowercase(), id);
    if let Some(short) = register_name.strip_prefix("block_") {
        map.entry(short.to_ascii_lowercase()).or_insert(id);
    }
    if let Some(short) = register_name.strip_prefix("item_") {
        map.entry(short.to_ascii_lowercase()).or_insert(id);
    }
    for alias in aliases {
        let key = alias.trim().to_ascii_lowercase();
        if !key.is_empty() {
            map.entry(key).or_insert(id);
        }
    }
}

fn build_face_tiles(raw: &RawBlockFile, max_tiles: u32) -> [u32; 6] {
    let default_slot = raw.texture_slot_1based.unwrap_or(1);
    let slots_1based = raw.textures_slot_1based.unwrap_or([default_slot; 6]);
    [
        sanitize_slot_1based(slots_1based[0], max_tiles),
        sanitize_slot_1based(slots_1based[1], max_tiles),
        sanitize_slot_1based(slots_1based[2], max_tiles),
        sanitize_slot_1based(slots_1based[3], max_tiles),
        sanitize_slot_1based(slots_1based[4], max_tiles),
        sanitize_slot_1based(slots_1based[5], max_tiles),
    ]
}

fn next_free_item_id(
    explicit_item_ids: &HashSet<i8>,
    used_block_item_ids: &HashSet<i8>,
) -> Option<i8> {
    for raw in 0..=i8::MAX as i16 {
        let id = raw as i8;
        if !explicit_item_ids.contains(&id) && !used_block_item_ids.contains(&id) {
            return Some(id);
        }
    }
    None
}

fn load_registry(tiles_x: u32) -> Registry {
    let max_block_tiles = atlas_tile_capacity(BLOCK_ATLAS_PATH, tiles_x.max(1));
    let max_item_tiles = atlas_tile_capacity(ITEM_ATLAS_PATH, tiles_x.max(1));

    let mut raw_items = load_json_defs::<RawItemFile>(Path::new(ITEMS_JSON_DIR), "item");
    let mut parsed_items: Vec<(RawItemFile, i8, Option<u32>)> = Vec::new();
    let mut explicit_item_ids = HashSet::<i8>::new();
    for raw in raw_items.drain(..) {
        let Some((item_id, namespace_local)) = parse_item_file_id(&raw.id, &raw.register_name)
        else {
            continue;
        };
        explicit_item_ids.insert(item_id);
        parsed_items.push((raw, item_id, namespace_local));
    }
    parsed_items.sort_by_key(|(_, item_id, _)| *item_id);

    let mut raw_blocks = load_json_defs::<RawBlockFile>(Path::new(BLOCKS_JSON_DIR), "block");
    if raw_blocks.is_empty() {
        panic!(
            "no valid block JSON defs found in {}. add block files before starting NyraCraft",
            BLOCKS_JSON_DIR
        );
    }
    let mut parsed_blocks: Vec<(RawBlockFile, usize, Option<u32>)> = Vec::new();
    for raw in raw_blocks.drain(..) {
        let Some((block_id, namespace_local)) = parse_block_file_id(&raw.id, &raw.register_name)
        else {
            continue;
        };
        parsed_blocks.push((raw, block_id, namespace_local));
    }
    if parsed_blocks.is_empty() {
        panic!(
            "no block entries in {} could be parsed, expected ids like {}:1",
            BLOCKS_JSON_DIR, BLOCK_ID_NAMESPACE
        );
    }
    parsed_blocks.sort_by_key(|(_, block_id, _)| *block_id);
    let max_block_id = parsed_blocks
        .iter()
        .map(|(_, block_id, _)| *block_id)
        .max()
        .unwrap_or(0);

    let mut blocks: Vec<Option<BlockDef>> = vec![None; max_block_id + 1];
    let mut pending_block_drops: Vec<Option<(Option<RawIdValue>, Vec<RawDropEntry>)>> =
        vec![None; max_block_id + 1];
    let mut used_block_item_ids = HashSet::<i8>::new();
    let mut block_namespace_to_id = HashMap::<u32, i8>::new();
    for (raw, runtime_block_id, namespace_local) in parsed_blocks {
        let Ok(item_id_default) = i8::try_from(runtime_block_id) else {
            eprintln!(
                "skipping block {}: runtime id {} does not fit into i8",
                raw.register_name, runtime_block_id
            );
            continue;
        };

        let texture = BlockTexture {
            tiles: build_face_tiles(&raw, max_block_tiles),
            rotations: raw.rotations.unwrap_or([0; 6]),
            transparent_mode: raw.transparent_mode.unwrap_or([0; 6]),
        };
        let icon_slot = raw
            .icon_texture_slot_1based
            .or(raw.texture_slot_1based)
            .or_else(|| raw.textures_slot_1based.map(|arr| arr[2]))
            .unwrap_or(1);
        let icon_tile_index = sanitize_slot_1based(icon_slot, max_block_tiles);
        let item_atlas_slot_1based =
            sanitize_slot_opt_1based(raw.item_atlas_slot_1based, max_item_tiles);
        let required_tool = parse_block_required_tool_or_default(&raw);
        let requested_item_id = raw
            .item_id
            .as_ref()
            .and_then(|value| parse_item_file_id(value, &raw.register_name).map(|(id, _)| id))
            .unwrap_or(item_id_default);
        let item_id = if explicit_item_ids.contains(&requested_item_id)
            || used_block_item_ids.contains(&requested_item_id)
        {
            let Some(remapped_id) = next_free_item_id(&explicit_item_ids, &used_block_item_ids)
            else {
                panic!(
                    "could not allocate free item id for block {} (requested {})",
                    raw.register_name, requested_item_id
                );
            };
            eprintln!(
                "block {} item_id {} conflicts with existing item id, remapped to {}",
                raw.register_name, requested_item_id, remapped_id
            );
            remapped_id
        } else {
            requested_item_id
        };
        used_block_item_ids.insert(item_id);
        let pending_drop_item = raw.drop_item;
        let pending_drop_items = raw.drop_items;

        let block = BlockDef {
            id: runtime_block_id,
            name: raw.register_name,
            aliases: raw.aliases,
            hardness: raw.hardness.unwrap_or(1.0),
            required_tool,
            texture,
            item_id,
            item_atlas_slot_1based,
            icon_tile_index,
            drops: Vec::new(),
        };

        let block_id = block.id;
        if blocks[block_id].is_some() {
            eprintln!(
                "duplicate block id {} encountered, keeping the last definition ({})",
                block.id, block.name
            );
        }
        if let Some(local) = namespace_local {
            if let Ok(block_id_i8) = i8::try_from(block_id) {
                block_namespace_to_id.insert(local, block_id_i8);
            }
        }
        pending_block_drops[block_id] = Some((pending_drop_item, pending_drop_items));
        blocks[block_id] = Some(block);
    }

    if block_namespace_to_id.is_empty() {
        for (idx, block) in blocks.iter().enumerate() {
            if block.is_some()
                && let Ok(block_id_i8) = i8::try_from(idx)
            {
                block_namespace_to_id.insert(idx as u32 + 1, block_id_i8);
            }
        }
    }

    let mut block_textures = vec![missing_block_texture(); blocks.len()];
    let mut block_hardness = vec![1.0; blocks.len()];
    let mut block_required_tools = vec![None; blocks.len()];
    let mut block_index = Vec::new();
    let mut block_name_to_id = HashMap::new();
    for (idx, block_opt) in blocks.iter().enumerate() {
        let Some(block) = block_opt else {
            continue;
        };
        block_textures[idx] = block.texture;
        block_hardness[idx] = block.hardness.max(0.05);
        block_required_tools[idx] = block.required_tool;
        if let Ok(block_id_i8) = i8::try_from(block.id) {
            insert_name_aliases(
                &mut block_name_to_id,
                &block.name,
                &block.aliases,
                block_id_i8,
            );
        }
        block_index.push(BlockIndexEntry {
            block_id: block.id,
            item_id: block.item_id.max(0) as u16,
            name: block.name.clone(),
            texture_index: block.icon_tile_index,
        });
    }
    block_index.sort_by_key(|entry| entry.block_id);

    let mut items = Vec::<ItemDef>::new();
    let mut item_by_id = HashMap::<i8, usize>::new();
    let mut item_aliases = HashMap::<i8, Vec<String>>::new();
    let mut item_namespace_to_id = HashMap::<u32, i8>::new();

    let mut upsert_item = |def: ItemDef, aliases: Vec<String>| {
        let item_id = def.item_id;
        if let Some(&idx) = item_by_id.get(&item_id) {
            items[idx] = def;
        } else {
            let idx = items.len();
            item_by_id.insert(item_id, idx);
            items.push(def);
        }
        item_aliases.insert(item_id, aliases);
    };

    for block in blocks.iter().flatten() {
        let item = ItemDef {
            item_id: block.item_id,
            name: block.name.clone(),
            place_block_id: i8::try_from(block.id).ok(),
            item_atlas_slot_1based: block.item_atlas_slot_1based,
            edible: false,
            saturation: 0.0,
            effects: Vec::new(),
            breaktime: None,
            durability: 0,
            max_stack_size: 64,
            tool_type: ToolType::Hand,
        };
        upsert_item(item, block.aliases.clone());
    }

    for (raw, item_id, namespace_local) in parsed_items {
        let breaktime = raw
            .breaktime
            .and_then(|value| (value.is_finite() && value > 0.0).then_some(value));
        let tool_type = parse_item_tool_type_or_default(&raw);
        let durability = raw.durability.unwrap_or_else(|| {
            if tool_type == ToolType::Hand {
                0
            } else {
                64
            }
        });
        let default_stack_size = if breaktime.is_some() || durability > 0 {
            1
        } else {
            64
        };
        let item = ItemDef {
            item_id,
            name: raw.register_name.clone(),
            place_block_id: raw.place_block_id,
            item_atlas_slot_1based: sanitize_slot_opt_1based(
                raw.item_atlas_slot_1based,
                max_item_tiles,
            ),
            edible: raw.edible,
            saturation: raw.saturation.max(0.0),
            effects: raw.effects,
            breaktime,
            durability,
            max_stack_size: raw.max_stack_size.unwrap_or(default_stack_size).clamp(1, 64),
            tool_type,
        };
        upsert_item(item, raw.aliases);
        if let Some(local) = namespace_local {
            item_namespace_to_id.insert(local, item_id);
        }
    }

    items.sort_by_key(|item| item.item_id);
    item_by_id.clear();
    for (idx, item) in items.iter().enumerate() {
        item_by_id.insert(item.item_id, idx);
    }

    let mut item_name_to_id = HashMap::new();
    for item in &items {
        insert_name_aliases(&mut item_name_to_id, &item.name, &[], item.item_id);
        if let Some(extra_aliases) = item_aliases.get(&item.item_id) {
            insert_name_aliases(
                &mut item_name_to_id,
                &item.name,
                extra_aliases,
                item.item_id,
            );
        }
    }
    let mut placeable_block_to_item_id = HashMap::<i8, i8>::new();
    for item in &items {
        if let Some(block_id) = item.place_block_id {
            placeable_block_to_item_id
                .entry(block_id)
                .or_insert(item.item_id);
        }
    }

    for (block_idx, block_opt) in blocks.iter_mut().enumerate() {
        let Some(block) = block_opt else {
            continue;
        };

        let mut resolved_drops = Vec::<DropEntry>::new();
        if let Some((drop_item, drop_items)) = pending_block_drops
            .get_mut(block_idx)
            .and_then(Option::take)
        {
            if let Some(raw_item) = drop_item {
                if let Some(item_id) = resolve_drop_item_id(
                    &raw_item,
                    &block_namespace_to_id,
                    &item_namespace_to_id,
                    &placeable_block_to_item_id,
                    &block_name_to_id,
                    &item_name_to_id,
                ) {
                    resolved_drops.push(DropEntry {
                        item_id,
                        min_count: 1,
                        max_count: 1,
                        chance: 1.0,
                    });
                } else {
                    eprintln!(
                        "block {} has invalid drop_item reference, ignoring it",
                        block.name
                    );
                }
            }

            for raw_entry in drop_items {
                let Some(item_id) = resolve_drop_item_id(
                    &raw_entry.item_id,
                    &block_namespace_to_id,
                    &item_namespace_to_id,
                    &placeable_block_to_item_id,
                    &block_name_to_id,
                    &item_name_to_id,
                ) else {
                    eprintln!(
                        "block {} has invalid drop_items entry, skipping one drop",
                        block.name
                    );
                    continue;
                };
                if let Some(entry) = normalized_drop_entry(&raw_entry, item_id) {
                    resolved_drops.push(entry);
                }
            }
        }

        if resolved_drops.is_empty() {
            // Default behavior if no JSON drop is provided.
            resolved_drops.push(DropEntry {
                item_id: block.item_id,
                min_count: 1,
                max_count: 1,
                chance: 1.0,
            });
        }
        block.drops = resolved_drops;
    }

    Registry {
        blocks,
        block_textures,
        block_hardness,
        block_required_tools,
        block_index,
        block_name_to_id,
        items,
        item_by_id,
        item_name_to_id,
        placeable_block_to_item_id,
        block_namespace_to_id,
        item_namespace_to_id,
        max_item_tiles,
    }
}

pub fn all_item_defs() -> &'static [ItemDef] {
    &registry().items
}

pub fn block_count() -> usize {
    registry().block_textures.len()
}

pub fn item_def_by_id(item_id: i8) -> Option<&'static ItemDef> {
    let reg = registry();
    let idx = *reg.item_by_id.get(&item_id)?;
    reg.items.get(idx)
}

pub fn item_name_by_id(item_id: i8) -> &'static str {
    item_def_by_id(item_id)
        .map(|def| def.name.as_str())
        .unwrap_or("unknown_item")
}

pub fn item_max_stack_size(item_id: i8) -> u8 {
    item_def_by_id(item_id)
        .map(|def| def.max_stack_size.clamp(1, 64))
        .unwrap_or(64)
}

pub fn item_tool_type(item_id: i8) -> ToolType {
    item_def_by_id(item_id)
        .map(|def| def.tool_type)
        .unwrap_or(ToolType::Hand)
}

pub fn item_breaktime(item_id: i8) -> Option<f32> {
    item_def_by_id(item_id)
        .and_then(|def| def.breaktime)
        .and_then(|value| (value.is_finite() && value > 0.0).then_some(value))
}

pub fn item_break_strength(item_id: i8) -> Option<f32> {
    item_breaktime(item_id).map(|time| 1.0 / time.max(0.05))
}

pub fn item_max_durability(item_id: i8) -> Option<u16> {
    item_def_by_id(item_id)
        .and_then(|def| (def.durability > 0).then_some(def.durability))
}

pub fn placeable_block_id_for_item(item_id: i8) -> Option<i8> {
    item_def_by_id(item_id).and_then(|def| def.place_block_id)
}

pub fn placeable_item_id_for_block(block_id: i8) -> Option<i8> {
    registry()
        .placeable_block_to_item_id
        .get(&block_id)
        .copied()
}

pub fn item_icon_tile_index(item_id: i8) -> Option<u32> {
    let reg = registry();
    item_def_by_id(item_id)
        .and_then(|def| def.item_atlas_slot_1based)
        .map(|slot| sanitize_slot_1based(slot, reg.max_item_tiles))
}

pub fn block_texture_by_id(id: i8) -> Option<&'static BlockTexture> {
    if id < 0 {
        return None;
    }
    let reg = registry();
    let idx = id as usize;
    if idx >= reg.blocks.len() || reg.blocks[idx].is_none() {
        return None;
    }
    reg.block_textures.get(idx)
}

pub fn default_blocks(tiles_x: u32) -> Vec<BlockTexture> {
    registry_with_tiles_x(tiles_x).block_textures.clone()
}

pub fn build_block_index(tiles_x: u32) -> Vec<BlockIndexEntry> {
    registry_with_tiles_x(tiles_x).block_index.clone()
}

pub fn block_name_by_id(id: i8) -> &'static str {
    if id < 0 {
        return "unknown";
    }
    registry()
        .blocks
        .get(id as usize)
        .and_then(|block| block.as_ref())
        .map(|block| block.name.as_str())
        .unwrap_or("unknown")
}

pub fn block_drop_rolls(id: i8) -> Vec<(i8, u8)> {
    if id < 0 {
        return Vec::new();
    }
    let reg = registry();
    let Some(block) = reg
        .blocks
        .get(id as usize)
        .and_then(|block_opt| block_opt.as_ref())
    else {
        return Vec::new();
    };

    let mut rng = rand::thread_rng();
    let mut out = Vec::with_capacity(block.drops.len());
    for drop in &block.drops {
        if rand::random::<f32>() > drop.chance {
            continue;
        }
        let count = if drop.min_count == drop.max_count {
            drop.min_count
        } else {
            rng.gen_range(drop.min_count..=drop.max_count)
        };
        if count > 0 {
            out.push((drop.item_id, count));
        }
    }
    out
}

pub fn block_hardness(id: i8) -> f32 {
    if id < 0 {
        return 1.0;
    }
    registry()
        .block_hardness
        .get(id as usize)
        .copied()
        .unwrap_or(1.0)
}

pub fn block_required_tool(id: i8) -> Option<ToolType> {
    if id < 0 {
        return None;
    }
    registry()
        .block_required_tools
        .get(id as usize)
        .copied()
        .flatten()
}

pub fn block_break_time_seconds(id: i8) -> f32 {
    block_hardness(id)
}

fn matching_tool_multiplier(tool: ToolType) -> f32 {
    match tool {
        ToolType::Hand => 1.0,
        ToolType::Pickaxe => 1.55,
        ToolType::Axe => 1.45,
        ToolType::Shovel => 1.45,
        ToolType::Knife => 1.30,
        ToolType::Sword => 1.20,
    }
}

fn off_tool_multiplier(tool: ToolType) -> f32 {
    match tool {
        ToolType::Hand => 1.0,
        ToolType::Pickaxe => 0.60,
        ToolType::Axe => 0.80,
        ToolType::Shovel => 0.0,
        ToolType::Knife => 0.0,
        ToolType::Sword => 0.0,
    }
}

fn adjusted_break_strength(id: i8, held_item_id: Option<i8>, break_strength: f32) -> Option<f32> {
    let mut strength = break_strength.max(0.05);
    let Some(required_tool) = block_required_tool(id) else {
        return Some(strength);
    };

    let held_tool = held_item_id.map(item_tool_type).unwrap_or(ToolType::Hand);
    if held_tool == ToolType::Hand {
        // Hand is universal: no tool-type lockout, only strength/hardness decides.
        return Some(strength);
    }
    if held_tool == required_tool {
        strength *= matching_tool_multiplier(held_tool);
        return Some(strength);
    }

    let multiplier = off_tool_multiplier(held_tool);
    if multiplier <= 0.0 {
        return None;
    }
    strength *= multiplier;
    (strength > 0.0).then_some(strength)
}

pub fn can_break_block_with_item(id: i8, held_item_id: Option<i8>, break_strength: f32) -> bool {
    let Some(strength) = adjusted_break_strength(id, held_item_id, break_strength) else {
        return false;
    };
    let hardness = block_hardness(id).max(0.05);
    hardness <= strength * 2.0
}

pub fn block_break_time_with_item_seconds(
    id: i8,
    held_item_id: Option<i8>,
    break_strength: f32,
) -> Option<f32> {
    let strength = adjusted_break_strength(id, held_item_id, break_strength)?;
    let hardness = block_hardness(id).max(0.05);
    if hardness > strength * 2.0 {
        return None;
    }
    Some(block_break_time_seconds(id).max(0.05) / strength)
}

pub fn block_break_stage(progress_seconds: f32, break_time_seconds: f32) -> u32 {
    let total = DESTROY_STAGE_COUNT.max(1);
    let ratio = (progress_seconds / break_time_seconds.max(0.001)).clamp(0.0, 0.999_9);
    ((ratio * total as f32).floor() as u32).min(total - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookupNamespace {
    Block,
    Item,
}

fn parse_lookup_namespace(raw: &str) -> Option<LookupNamespace> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "b" | "block" => Some(LookupNamespace::Block),
        "2" | "i" | "item" => Some(LookupNamespace::Item),
        _ => None,
    }
}

fn split_lookup_query(raw: &str) -> Option<(LookupNamespace, &str)> {
    let (prefix, body) = raw.split_once(':')?;
    let namespace = parse_lookup_namespace(prefix)?;
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    Some((namespace, body))
}

fn parse_namespace_local(body: &str) -> Option<u32> {
    let local = body.trim().parse::<u32>().ok()?;
    if local == 0 {
        return None;
    }
    Some(local)
}

fn parse_block_id_namespaced(body: &str) -> Option<i8> {
    if let Some(local) = parse_namespace_local(body) {
        return registry().block_namespace_to_id.get(&local).copied();
    }
    parse_block_id_plain(body)
}

fn parse_item_id_namespaced(body: &str) -> Option<i8> {
    if let Some(local) = parse_namespace_local(body) {
        return registry().item_namespace_to_id.get(&local).copied();
    }
    parse_item_id_plain(body)
}

fn parse_block_id_plain(name_or_id: &str) -> Option<i8> {
    let raw = name_or_id.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(id) = raw.parse::<usize>() {
        let reg = registry();
        if id < reg.blocks.len() && reg.blocks[id].is_some() {
            return Some(id as i8);
        }
    }

    let key = raw.to_ascii_lowercase();
    registry().block_name_to_id.get(&key).copied()
}

fn parse_item_id_plain(name_or_id: &str) -> Option<i8> {
    let raw = name_or_id.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(id) = raw.parse::<i16>()
        && let Ok(id) = i8::try_from(id)
        && item_def_by_id(id).is_some()
    {
        return Some(id);
    }

    let key = raw.to_ascii_lowercase();
    if let Some(id) = registry().item_name_to_id.get(&key).copied() {
        return Some(id);
    }

    None
}

pub fn parse_item_id(name_or_id: &str) -> Option<i8> {
    let raw = name_or_id.trim();
    if raw.is_empty() {
        return None;
    }

    if let Some((namespace, body)) = split_lookup_query(raw) {
        return match namespace {
            LookupNamespace::Item => parse_item_id_namespaced(body),
            LookupNamespace::Block => parse_block_id_namespaced(body)
                .and_then(placeable_item_id_for_block)
                .or_else(|| parse_block_id_namespaced(body)),
        };
    }

    if let Some(item_id) = parse_item_id_plain(raw) {
        return Some(item_id);
    }

    if let Some(block_id) = parse_block_id_plain(raw) {
        return placeable_item_id_for_block(block_id).or(Some(block_id));
    }

    None
}
