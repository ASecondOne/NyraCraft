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

pub fn resolve_launch_seed() -> (u32, WorldMode, String) {
    let cli_seed = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let seed_text = if cli_seed.trim().is_empty() {
        println!("Enter seed (number/text), FLAT, GRID, or blank for random:");
        print!("seed> ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            input
        } else {
            String::new()
        }
    } else {
        cli_seed
    };

    let seed_text = seed_text.trim();
    if seed_text.is_empty() {
        let seed: u32 = rand::random();
        return (seed, WorldMode::Normal, "<random>".to_string());
    }

    let (seed, mode) = parse_seed_mode(seed_text);
    (seed, mode, seed_text.to_string())
}
