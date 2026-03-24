use crate::render::atlas::TextureAtlas;
use image::RgbaImage;
use std::fs;
use std::path::{Path, PathBuf};

pub struct AtlasTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_x: u32,
    pub tile_uv_size: [f32; 2],
}

pub struct SampledTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

#[derive(Clone, Copy)]
pub struct UvRect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl UvRect {
    fn full() -> Self {
        Self {
            min: [0.0, 0.0],
            max: [1.0, 1.0],
        }
    }
}

pub struct CelestialTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub sun_uv: UvRect,
    pub moon_phase_uvs: [UvRect; 8],
}

fn aligned_bytes_per_row(width: u32) -> u32 {
    let raw = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    raw.div_ceil(align) * align
}

fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rgba: &RgbaImage,
    filter: wgpu::FilterMode,
) -> SampledTexture {
    let (width, height) = rgba.dimensions();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let src = rgba.as_raw();
    let raw_bpr = width * 4;
    let padded_bpr = aligned_bytes_per_row(width);
    let upload = if padded_bpr == raw_bpr {
        src.clone()
    } else {
        let mut data = vec![0u8; (padded_bpr * height) as usize];
        for row in 0..height as usize {
            let src_off = row * raw_bpr as usize;
            let dst_off = row * padded_bpr as usize;
            data[dst_off..dst_off + raw_bpr as usize]
                .copy_from_slice(&src[src_off..src_off + raw_bpr as usize]);
        }
        data
    };

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &upload,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded_bpr),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: filter,
        ..Default::default()
    });

    SampledTexture { view, sampler }
}

pub fn load_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &TextureAtlas,
) -> AtlasTexture {
    let image = image::open(&atlas.path)
        .unwrap_or_else(|e| panic!("failed to open atlas {}: {e}", atlas.path));
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let sampled = create_texture_from_rgba(
        device,
        queue,
        "atlas_texture",
        &rgba,
        wgpu::FilterMode::Nearest,
    );

    let tiles_x = width / atlas.tile_size;
    let tiles_y = height / atlas.tile_size;
    assert!(tiles_x > 0 && tiles_y > 0, "atlas tile size is invalid");
    let tile_uv_size = [
        atlas.tile_size as f32 / width as f32,
        atlas.tile_size as f32 / height as f32,
    ];

    AtlasTexture {
        view: sampled.view,
        sampler: sampled.sampler,
        tiles_x,
        tile_uv_size,
    }
}

fn first_png_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect();
    files.sort();
    files.into_iter().next()
}

pub fn load_grass_colormap_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> SampledTexture {
    let dir = Path::new("src/texturing/colormap");
    if let Some(path) = first_png_in_dir(dir) {
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                return create_texture_from_rgba(
                    device,
                    queue,
                    "grass_colormap_texture",
                    &rgba,
                    wgpu::FilterMode::Linear,
                );
            }
            Err(e) => {
                eprintln!("failed to open grass colormap {}: {e}", path.display());
            }
        }
    }

    let fallback = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
    create_texture_from_rgba(
        device,
        queue,
        "grass_colormap_fallback",
        &fallback,
        wgpu::FilterMode::Nearest,
    )
}

pub fn load_player_skin_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> SampledTexture {
    let candidate_paths = [
        PathBuf::from("src/texturing/generel_textures/steve.png"),
        PathBuf::from("src/texturing/player_skin.png"),
    ];
    for path in candidate_paths {
        if !path.exists() {
            continue;
        }
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                return create_texture_from_rgba(
                    device,
                    queue,
                    "player_skin_texture",
                    &rgba,
                    wgpu::FilterMode::Nearest,
                );
            }
            Err(e) => {
                eprintln!("failed to open player skin {}: {e}", path.display());
            }
        }
    }

    let fallback = image::RgbaImage::from_pixel(64, 64, image::Rgba([206, 163, 132, 255]));
    create_texture_from_rgba(
        device,
        queue,
        "player_skin_fallback",
        &fallback,
        wgpu::FilterMode::Nearest,
    )
}

pub fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> AtlasTexture {
    let fallback = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
    let sampled = create_texture_from_rgba(
        device,
        queue,
        "dummy_texture",
        &fallback,
        wgpu::FilterMode::Nearest,
    );

    AtlasTexture {
        view: sampled.view,
        sampler: sampled.sampler,
        tiles_x: 1,
        tile_uv_size: [1.0, 1.0],
    }
}

fn rgba_has_transparency(rgba: &RgbaImage) -> bool {
    rgba.pixels().any(|pixel| pixel[3] < 255)
}

fn punch_out_dark_background(rgba: &mut RgbaImage) {
    const FULL_CUT_THRESHOLD: u8 = 10;
    const SOFT_EDGE_THRESHOLD: u8 = 32;

    for pixel in rgba.pixels_mut() {
        if pixel[3] == 0 {
            continue;
        }
        let max_rgb = pixel[0].max(pixel[1]).max(pixel[2]);
        if max_rgb <= FULL_CUT_THRESHOLD {
            pixel[3] = 0;
            continue;
        }
        if max_rgb < SOFT_EDGE_THRESHOLD {
            let keep = (max_rgb - FULL_CUT_THRESHOLD) as f32
                / (SOFT_EDGE_THRESHOLD - FULL_CUT_THRESHOLD) as f32;
            pixel[3] = ((pixel[3] as f32) * keep.clamp(0.0, 1.0)).round() as u8;
        }
    }
}

fn load_best_rgba(paths: &[PathBuf], prefer_alpha: bool) -> Option<RgbaImage> {
    let mut fallback = None;
    for path in paths {
        if !path.exists() {
            continue;
        }
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                if prefer_alpha && rgba_has_transparency(&rgba) {
                    return Some(rgba);
                }
                if fallback.is_none() {
                    fallback = Some(rgba);
                }
            }
            Err(e) => eprintln!("failed to open texture {}: {e}", path.display()),
        }
    }
    fallback
}

pub fn load_celestial_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> CelestialTexture {
    let sun_paths = [
        PathBuf::from("src/texturing/generel_textures/sun.png"),
        PathBuf::from("src/texturing/generel_textures/sun_vv.png"),
        PathBuf::from("src/texturing/generel_textures/sun_vv.png.png"),
    ];
    let moon_paths = [PathBuf::from(
        "src/texturing/generel_textures/moon_phases.png",
    )];

    let mut sun = load_best_rgba(&sun_paths, true)
        .unwrap_or_else(|| image::RgbaImage::from_pixel(32, 32, image::Rgba([255, 255, 255, 255])));
    if !rgba_has_transparency(&sun) {
        punch_out_dark_background(&mut sun);
    }
    let mut moon = load_best_rgba(&moon_paths, true);
    if let Some(moon_img) = moon.as_mut()
        && !rgba_has_transparency(moon_img)
    {
        punch_out_dark_background(moon_img);
    }

    let mut sun_uv = UvRect::full();
    let mut moon_phase_uvs = [UvRect::full(); 8];

    let atlas = if let Some(moon_img) = moon {
        let sun_w = sun.width().max(1);
        let sun_h = sun.height().max(1);
        let moon_w = moon_img.width().max(1);
        let moon_h = moon_img.height().max(1);
        let atlas_w = sun_w + moon_w;
        let atlas_h = sun_h.max(moon_h);

        let mut atlas = image::RgbaImage::from_pixel(atlas_w, atlas_h, image::Rgba([0, 0, 0, 0]));
        image::imageops::overlay(&mut atlas, &sun, 0, 0);
        image::imageops::overlay(&mut atlas, &moon_img, sun_w as i64, 0);

        sun_uv = UvRect {
            min: [0.0, 0.0],
            max: [sun_w as f32 / atlas_w as f32, sun_h as f32 / atlas_h as f32],
        };

        let mut tile_size = 32_u32;
        if moon_w % 4 == 0 && moon_h % 2 == 0 {
            let tw = moon_w / 4;
            let th = moon_h / 2;
            if tw == th && tw > 0 {
                tile_size = tw;
            }
        }
        tile_size = tile_size.max(1).min(moon_w.max(1)).min(moon_h.max(1));
        let tiles_x = (moon_w / tile_size).max(1);
        let tiles_y = (moon_h / tile_size).max(1);
        let phase_tiles = (tiles_x * tiles_y).max(1) as usize;
        for phase in 0..8 {
            let tile_idx = phase % phase_tiles;
            let tx = (tile_idx as u32) % tiles_x;
            let ty = (tile_idx as u32) / tiles_x;
            let x0 = sun_w + tx * tile_size;
            let y0 = ty * tile_size;
            let x1 = sun_w + ((tx + 1) * tile_size).min(moon_w);
            let y1 = ((ty + 1) * tile_size).min(moon_h);
            moon_phase_uvs[phase] = UvRect {
                min: [x0 as f32 / atlas_w as f32, y0 as f32 / atlas_h as f32],
                max: [x1 as f32 / atlas_w as f32, y1 as f32 / atlas_h as f32],
            };
        }

        atlas
    } else {
        sun
    };

    let sampled = create_texture_from_rgba(
        device,
        queue,
        "celestial_texture",
        &atlas,
        wgpu::FilterMode::Linear,
    );

    CelestialTexture {
        view: sampled.view,
        sampler: sampled.sampler,
        sun_uv,
        moon_phase_uvs,
    }
}

#[cfg(test)]
mod tests {
    use super::{punch_out_dark_background, rgba_has_transparency};
    use image::{Rgba, RgbaImage};

    #[test]
    fn punch_out_dark_background_removes_black_matte() {
        let mut rgba = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 255]));
        rgba.put_pixel(1, 1, Rgba([255, 225, 96, 255]));
        rgba.put_pixel(2, 2, Rgba([20, 20, 20, 255]));

        punch_out_dark_background(&mut rgba);

        assert_eq!(rgba.get_pixel(0, 0)[3], 0);
        assert_eq!(rgba.get_pixel(1, 1)[3], 255);
        assert!(rgba.get_pixel(2, 2)[3] > 0);
        assert!(rgba.get_pixel(2, 2)[3] < 255);
    }

    #[test]
    fn transparency_detection_finds_existing_alpha() {
        let mut rgba = RgbaImage::from_pixel(2, 2, Rgba([255, 255, 255, 255]));
        assert!(!rgba_has_transparency(&rgba));
        rgba.put_pixel(1, 1, Rgba([255, 255, 255, 0]));
        assert!(rgba_has_transparency(&rgba));
    }
}
