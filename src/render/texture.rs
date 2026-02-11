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
    let sampled = create_texture_from_rgba(device, queue, "atlas_texture", &rgba, wgpu::FilterMode::Nearest);

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
        .filter(|p| p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("png")))
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

pub fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> AtlasTexture {
    let fallback = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
    let sampled = create_texture_from_rgba(device, queue, "dummy_texture", &fallback, wgpu::FilterMode::Nearest);

    AtlasTexture {
        view: sampled.view,
        sampler: sampled.sampler,
        tiles_x: 1,
        tile_uv_size: [1.0, 1.0],
    }
}
