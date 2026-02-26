use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::player::inventory::InventorySnapshot;
use crate::player::EditedBlockEntry;
use crate::world::worldgen::WorldGen;

const SAVE_ROOT_DIR: &str = "save";
const META_FILE: &str = "meta.json";
const PLAYER_FILE: &str = "player.json";
const INVENTORY_FILE: &str = "inventory.json";
const EDITS_FILE: &str = "edits.json";
const SAVE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedPlayerState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub grounded: bool,
    pub camera_forward: [f32; 3],
    pub camera_up: [f32; 3],
    pub selected_hotbar_slot: u8,
    pub fly_mode: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SaveMeta {
    format_version: u32,
    world_id: u64,
    seed: u32,
    mode: String,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct SavedEditsFile {
    edits: Vec<EditedBlockEntry>,
}

#[derive(Default)]
pub struct LoadedRuntimeState {
    pub player: Option<SavedPlayerState>,
    pub inventory: Option<InventorySnapshot>,
    pub edited_blocks: Vec<EditedBlockEntry>,
    pub warnings: Vec<String>,
}

pub struct SaveIo {
    root_dir: PathBuf,
    meta: SaveMeta,
}

impl SaveIo {
    pub fn new(world_gen: &WorldGen) -> Self {
        let root_dir = PathBuf::from(SAVE_ROOT_DIR).join(format!("world_{:016x}", world_gen.world_id));
        let meta = SaveMeta {
            format_version: SAVE_FORMAT_VERSION,
            world_id: world_gen.world_id,
            seed: world_gen.seed,
            mode: world_gen.mode_name().to_string(),
        };
        Self { root_dir, meta }
    }

    pub fn save_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn load(&self) -> LoadedRuntimeState {
        let mut out = LoadedRuntimeState::default();
        if !self.root_dir.exists() {
            return out;
        }

        match read_json_optional::<SaveMeta>(&self.path(META_FILE)) {
            Ok(Some(meta)) => {
                if meta.format_version != SAVE_FORMAT_VERSION {
                    out.warnings.push(format!(
                        "save meta format {} differs from runtime format {}",
                        meta.format_version, SAVE_FORMAT_VERSION
                    ));
                }
                if meta.world_id != self.meta.world_id {
                    out.warnings.push(format!(
                        "save world id {} does not match runtime world {}",
                        meta.world_id, self.meta.world_id
                    ));
                }
            }
            Ok(None) => {}
            Err(err) => out.warnings.push(format!("failed to load meta.json: {err}")),
        }

        match read_json_optional::<SavedPlayerState>(&self.path(PLAYER_FILE)) {
            Ok(player) => out.player = player,
            Err(err) => out
                .warnings
                .push(format!("failed to load player.json: {err}")),
        }

        match read_json_optional::<InventorySnapshot>(&self.path(INVENTORY_FILE)) {
            Ok(inventory) => out.inventory = inventory,
            Err(err) => out
                .warnings
                .push(format!("failed to load inventory.json: {err}")),
        }

        match read_json_optional::<SavedEditsFile>(&self.path(EDITS_FILE)) {
            Ok(Some(edits)) => out.edited_blocks = edits.edits,
            Ok(None) => {}
            Err(err) => out
                .warnings
                .push(format!("failed to load edits.json: {err}")),
        }

        out
    }

    pub fn save_player(&self, player: &SavedPlayerState) -> io::Result<()> {
        self.ensure_dir()?;
        self.write_meta()?;
        write_json_atomic(&self.path(PLAYER_FILE), player)
    }

    pub fn save_inventory(&self, inventory: &InventorySnapshot) -> io::Result<()> {
        self.ensure_dir()?;
        self.write_meta()?;
        write_json_atomic(&self.path(INVENTORY_FILE), inventory)
    }

    pub fn save_edited_blocks(&self, edited_blocks: &[EditedBlockEntry]) -> io::Result<()> {
        self.ensure_dir()?;
        self.write_meta()?;
        write_json_atomic(
            &self.path(EDITS_FILE),
            &SavedEditsFile {
                edits: edited_blocks.to_vec(),
            },
        )
    }

    fn write_meta(&self) -> io::Result<()> {
        write_json_atomic(&self.path(META_FILE), &self.meta)
    }

    fn ensure_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root_dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root_dir.join(name)
    }
}

fn read_json_optional<T: DeserializeOwned>(path: &Path) -> io::Result<Option<T>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    let parsed = serde_json::from_slice::<T>(&bytes).map_err(|err| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid json in {}: {err}", path.display()),
        )
    })?;
    Ok(Some(parsed))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidInput,
            format!("path has no parent: {}", path.display()),
        )
    })?;
    fs::create_dir_all(parent)?;

    let data = serde_json::to_vec_pretty(value).map_err(|err| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("failed to serialize {}: {err}", path.display()),
        )
    })?;
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, data)?;

    if let Err(first_err) = fs::rename(&tmp_path, path) {
        if path.exists() {
            let _ = fs::remove_file(path);
            fs::rename(&tmp_path, path)?;
        } else {
            let _ = fs::remove_file(&tmp_path);
            return Err(first_err);
        }
    }
    Ok(())
}
