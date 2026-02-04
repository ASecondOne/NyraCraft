use crate::render::atlas::TextureAtlas;

pub struct AtlasTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_x: u32,
    pub tile_uv_size: [f32; 2],
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

    let bytes_per_row = 4 * width;
    assert!(
        bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0,
        "atlas width must align to 256-byte rows; got {bytes_per_row}"
    );

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas_texture"),
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

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
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
        label: Some("atlas_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let tiles_x = width / atlas.tile_size;
    let tiles_y = height / atlas.tile_size;
    assert!(tiles_x > 0 && tiles_y > 0, "atlas tile size is invalid");
    let tile_uv_size = [
        atlas.tile_size as f32 / width as f32,
        atlas.tile_size as f32 / height as f32,
    ];

    AtlasTexture {
        view,
        sampler,
        tiles_x,
        tile_uv_size,
    }
}

pub fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> AtlasTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy_texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

    AtlasTexture {
        view,
        sampler,
        tiles_x: 1,
        tile_uv_size: [1.0, 1.0],
    }
}
