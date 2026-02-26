use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;

use crate::world::worldgen::WorldMode;

fn newest_atlas_input_mtime(script: &str, texture_dirs: &[&str]) -> Option<SystemTime> {
    let mut newest = fs::metadata(script).ok()?.modified().ok()?;
    for texture_dir in texture_dirs {
        let entries = fs::read_dir(texture_dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !ext.eq_ignore_ascii_case("png") && !ext.eq_ignore_ascii_case("tga") {
                continue;
            }
            let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            if modified > newest {
                newest = modified;
            }
        }
    }
    Some(newest)
}

fn atlases_are_up_to_date(script: &str, texture_dirs: &[&str], atlas_paths: &[&str]) -> bool {
    let Some(input_mtime) = newest_atlas_input_mtime(script, texture_dirs) else {
        return false;
    };
    atlas_paths.iter().all(|path| {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .map(|mtime| mtime >= input_mtime)
            .unwrap_or(false)
    })
}

pub fn generate_texture_atlases() {
    let script = "src/texturing/atlas_gen.py";
    let texture_dirs = [
        "src/texturing/textures_blocks",
        "src/texturing/textures_items",
    ];
    let atlas_paths = [
        "src/texturing/atlas_output/atls_blocks.png",
        "src/texturing/atlas_output/atlas_items.png",
    ];

    if atlases_are_up_to_date(script, &texture_dirs, &atlas_paths) {
        return;
    }

    let try_run = |exe: &str| -> Result<(), String> {
        let status = Command::new(exe)
            .arg(script)
            .status()
            .map_err(|e| format!("failed to run {exe}: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{exe} exited with status {status}"))
        }
    };

    if try_run("python3").is_err()
        && let Err(e) = try_run("python")
    {
        panic!("atlas generation failed: {e}");
    }

    for atlas_path in &atlas_paths {
        if !Path::new(atlas_path).exists() {
            panic!("atlas generation completed but atlas was not found at {atlas_path}");
        }
    }
}

fn detect_cpu_model_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(model) = cpuinfo
                .lines()
                .find_map(|line| line.strip_prefix("model name\t:"))
            {
                let model = model.trim();
                if !model.is_empty() {
                    return Some(model.to_string());
                }
            }
            if let Some(model) = cpuinfo
                .lines()
                .find_map(|line| line.strip_prefix("Hardware\t:"))
            {
                let model = model.trim();
                if !model.is_empty() {
                    return Some(model.to_string());
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                let brand = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !brand.is_empty() {
                    return Some(brand);
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(id) = std::env::var("PROCESSOR_IDENTIFIER") {
            let id = id.trim();
            if !id.is_empty() {
                return Some(id.to_string());
            }
        }
    }

    None
}

pub fn detect_cpu_label() -> String {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let base = detect_cpu_model_name().unwrap_or_else(|| std::env::consts::ARCH.to_string());
    if logical_cores > 0 {
        format!("{base} ({logical_cores} logical cores)")
    } else {
        base
    }
}

fn hash_seed_text(text: &str) -> u32 {
    let mut hash = 0x811C_9DC5_u32; // FNV-1a 32-bit offset basis
    for b in text.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash.max(1)
}

fn parse_seed_mode(seed_text: &str) -> (u32, WorldMode) {
    if seed_text.eq_ignore_ascii_case("FLAT") {
        return (0xF1A7_0001, WorldMode::Flat);
    }
    if seed_text.eq_ignore_ascii_case("GRID") {
        return (0x6A1D_0002, WorldMode::Grid);
    }
    if let Ok(seed) = seed_text.parse::<u32>() {
        return (seed, WorldMode::Normal);
    }
    (hash_seed_text(seed_text), WorldMode::Normal)
}

fn world_mode_name(mode: WorldMode) -> &'static str {
    match mode {
        WorldMode::Normal => "NORMAL",
        WorldMode::Flat => "FLAT",
        WorldMode::Grid => "GRID",
    }
}

fn parse_world_mode_name(name: &str) -> Option<WorldMode> {
    if name.eq_ignore_ascii_case("NORMAL") {
        return Some(WorldMode::Normal);
    }
    if name.eq_ignore_ascii_case("FLAT") {
        return Some(WorldMode::Flat);
    }
    if name.eq_ignore_ascii_case("GRID") {
        return Some(WorldMode::Grid);
    }
    None
}

fn read_prompt_line(prompt: &str) -> String {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        input.trim().to_string()
    } else {
        String::new()
    }
}

fn create_new_world_selection() -> (u32, WorldMode, String) {
    println!("Enter seed (number/text), FLAT, GRID, or blank for random:");
    let seed_text = read_prompt_line("seed> ");
    if seed_text.is_empty() {
        let seed: u32 = rand::random();
        return (seed, WorldMode::Normal, "<random>".to_string());
    }
    let (seed, mode) = parse_seed_mode(&seed_text);
    (seed, mode, seed_text)
}

#[derive(Deserialize)]
struct SaveMetaSummary {
    world_id: u64,
    seed: u32,
    mode: String,
}

#[derive(Clone)]
struct SavedWorldEntry {
    dir_name: String,
    world_id: u64,
    seed: u32,
    mode: WorldMode,
    modified: Option<SystemTime>,
}

fn discover_saved_worlds() -> Vec<SavedWorldEntry> {
    let mut worlds = Vec::new();
    let save_root = Path::new("save");
    let Ok(entries) = fs::read_dir(save_root) else {
        return worlds;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if !dir_name.starts_with("world_") {
            continue;
        }
        let dir_path = entry.path();
        let meta_path = dir_path.join("meta.json");
        let Ok(meta_raw) = fs::read_to_string(&meta_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<SaveMetaSummary>(&meta_raw) else {
            continue;
        };
        let Some(mode) = parse_world_mode_name(&meta.mode) else {
            continue;
        };
        let modified = fs::metadata(&meta_path).and_then(|m| m.modified()).ok();
        worlds.push(SavedWorldEntry {
            dir_name,
            world_id: meta.world_id,
            seed: meta.seed,
            mode,
            modified,
        });
    }
    worlds.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.dir_name.cmp(&b.dir_name))
    });
    worlds
}

fn choose_saved_world(worlds: &[SavedWorldEntry]) -> Option<(u32, WorldMode, String)> {
    if worlds.is_empty() {
        println!("No saved worlds found yet.");
        return None;
    }
    println!("Saved Worlds:");
    for (index, world) in worlds.iter().enumerate() {
        println!(
            "  {}) {} | mode={} | seed={} | world_id={:016x}",
            index + 1,
            world.dir_name,
            world_mode_name(world.mode),
            world.seed,
            world.world_id
        );
    }
    println!("Type a number to load, or B to go back.");
    loop {
        let raw = read_prompt_line("load> ");
        if raw.eq_ignore_ascii_case("b") {
            return None;
        }
        if let Ok(index) = raw.parse::<usize>()
            && (1..=worlds.len()).contains(&index)
        {
            let world = &worlds[index - 1];
            return Some((
                world.seed,
                world.mode,
                format!(
                    "load:{}:{}",
                    world.dir_name,
                    world_mode_name(world.mode).to_ascii_lowercase()
                ),
            ));
        }
        println!("Invalid selection. Enter a world number or B.");
    }
}

pub fn resolve_launch_seed() -> (u32, WorldMode, String) {
    let cli_seed = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if !cli_seed.trim().is_empty() {
        let seed_text = cli_seed.trim();
        let (seed, mode) = parse_seed_mode(seed_text);
        return (seed, mode, seed_text.to_string());
    }

    loop {
        println!();
        println!("=== NyraCraft Main Menu ===");
        let saved_worlds = discover_saved_worlds();
        if saved_worlds.is_empty() {
            println!("1) Create New World");
            println!("2) Quit");
            match read_prompt_line("menu> ").as_str() {
                "1" => return create_new_world_selection(),
                "2" => std::process::exit(0),
                _ => {
                    println!("Invalid menu option.");
                    continue;
                }
            }
        } else {
            println!("1) Load Saved World");
            println!("2) Create New World");
            println!("3) Quit");
            match read_prompt_line("menu> ").as_str() {
                "1" => {
                    if let Some(choice) = choose_saved_world(&saved_worlds) {
                        return choice;
                    }
                }
                "2" => return create_new_world_selection(),
                "3" => std::process::exit(0),
                _ => {
                    println!("Invalid menu option.");
                    continue;
                }
            }
        }
    }
}
