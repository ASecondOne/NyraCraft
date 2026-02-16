use image::{Rgba, RgbaImage, imageops::{FilterType, resize, replace}};
use regex::Regex;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const TILE_SIZE: u32 = 16;
    const MAX_COLS: u32 = 16;
    let PREFIX_RE: Regex = Regex::new(r"(?i)^(\d+)(?:[_-].*)?\.(png|tga)$").expect("failed to create regex");

    let exe_path = env::current_exe()?;
    let script_dir = exe_path.parent().unwrap();

    let texture_dir = script_dir.join("textures");
    let output_dir = script_dir.join("atlas_output");
    let output_path = script_dir.join("atlas.png");

    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(&texture_dir)?;

    let mut indexed_files: Vec<(u32, PathBuf)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    let mut texture_files: Vec<PathBuf> = fs::read_dir(&texture_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.is_file())
        .filter(|p| match p.extension().and_then(|s| s.to_str()) {
            Some(ext) => {
                let ext = ext.to_lowercase();
                ext == "png" || ext == "tga"
            }
            None => false,
        })
        .collect();

    texture_files.sort();

    for file in texture_files {
        let file_name = file.file_name().and_then(|s| s.to_str()).unwrap_or("");

        if let Some(m) = PREFIX_RE.captures(file_name) {
            let idx = m[1].parse().unwrap_or(0);
            if idx <= 0 {
                skipped.push(file_name.to_string());
                continue;
            }
            indexed_files.push((idx, file));
        } else {
            skipped.push(file_name.to_string());
        }
    }

    if indexed_files.is_empty() {
        eprintln!(
            "[atlas_gen] no numbered textures (.png/.tga) found in {:?}",
            texture_dir
        );
        std::process::exit(1);
    }

    let mut seen: HashMap<u32, PathBuf> = HashMap::new();
    let mut duplicates: Vec<(u32, PathBuf, PathBuf)> = Vec::new();

    for (idx, file) in indexed_files {
        if let Some(first_file) = seen.get(&idx) {
            duplicates.push((idx, first_file.clone(), file));
        } else {
            seen.insert(idx, file);
        }
    }

    if !duplicates.is_empty() {
        eprintln!("[atlas_gen] duplicate numeric prefixes detected:");
        for (idx, first, second) in duplicates {
            eprintln!(
                "index: {}: {} and {}",
                idx,
                first.file_name().unwrap().to_str().unwrap(),
                second.file_name().unwrap().to_str().unwrap()
            );
        }

        std::process::exit(1);
    }

    let max_index = *seen.keys().max().unwrap();
    let rows = (max_index + MAX_COLS - 1) / MAX_COLS;
    let atlas_width = MAX_COLS * TILE_SIZE;
    let atlas_height = rows * TILE_SIZE;

    let mut atlas: RgbaImage = RgbaImage::from_pixel(
        u32::try_from(atlas_width).expect("atlas_width does not fit inside a u32"),
        u32::try_from(atlas_height).unwrap(),
        Rgba([0, 0, 0, 0]),
    );

    let mut entries: Vec<(&u32, &PathBuf)> = seen.iter().collect();

    entries.sort_by_key(|(idx, _)| *idx);

    for (idx, file) in entries {
        let img = image::open(file).expect("Error opening Image");

        let mut img = img.to_rgba8();

        if img.width() != TILE_SIZE || img.height() != TILE_SIZE {
            img = resize(&img, TILE_SIZE, TILE_SIZE, FilterType::Nearest);
        }

        let x = ((idx -1) % MAX_COLS) * TILE_SIZE;
        let y = ((idx -1) / MAX_COLS) * TILE_SIZE;

        replace(&mut atlas, &img, x as i64, y as i64);
    }

    atlas.save(&output_path)?;

    println!("[atlas_gen] generated {} ({}x{})", output_path.display(), atlas.width(), atlas_height);

    if !skipped.is_empty() {
        println!("[atlas_gen] skipped {} file(s) without valid numeric prefix or extension", skipped.len());
    }

    Ok(())
}
