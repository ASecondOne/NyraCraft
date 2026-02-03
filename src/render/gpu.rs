use crate::camera::Camera;
use crate::render::atlas::TextureAtlas;
use crate::render::CubeStyle;
use crate::render::mesh::ChunkVertex;
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3};
use self_cell::self_cell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    mvp: [[f32; 4]; 4],
    use_texture: u32,
    tiles_x: u32,
    debug_faces: u32,
    _pad0: u32,
    tile_uv_size: [f32; 2],
    _pad1: [f32; 2],
}

struct ChunkGpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    center: Vec3,
    radius: f32,
}

#[allow(dead_code)]
struct GpuInner<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tiles_x: u32,
    tile_uv_size: [f32; 2],
    chunks: HashMap<IVec3, ChunkGpuMesh>,
}

self_cell! {
    struct GpuCell {
        owner: winit::window::Window,
        #[covariant]
        dependent: GpuInner,
    }
}

pub struct Gpu {
    cell: GpuCell,
    style: CubeStyle,
}

impl Gpu {
    pub fn new(window: winit::window::Window, style: CubeStyle, atlas: Option<TextureAtlas>) -> Self {
        let cell = GpuCell::new(window, |window| {
            let size = window.inner_size();

            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window).unwrap();

            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            }))
            .unwrap();

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                    .unwrap();

            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats[0];

            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

            surface.configure(&device, &config);

            let depth_view = create_depth_view(&device, &config);

            let (texture_view, sampler, tiles_x, tile_uv_size) = if style.use_texture {
                let atlas = atlas.expect("TextureAtlas required when use_texture is true");
                create_atlas_texture(&device, &queue, &atlas)
            } else {
                create_dummy_texture(&device, &queue)
            };

            let uniform = SceneUniform {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                use_texture: style.use_texture as u32,
                tiles_x,
                debug_faces: 0,
                _pad0: 0,
                tile_uv_size,
                _pad1: [0.0; 2],
            };

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene_uniform_buffer"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("scene_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cube.wgsl").into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cube_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cube_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[ChunkVertex::layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

            GpuInner {
                surface,
                device,
                queue,
                config,
                render_pipeline,
                depth_view,
                uniform_buffer,
                bind_group,
                tiles_x,
                tile_uv_size,
                chunks: HashMap::new(),
            }
        });

        Self { cell, style }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.config.width = new_size.width;
            gpu.config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.config);
            gpu.depth_view = create_depth_view(&gpu.device, &gpu.config);
        });
    }

    pub fn render(&self, camera: &Camera, debug_faces: bool) {
        self.cell.with_dependent(|_, gpu| {
            let mvp = build_mvp(gpu.config.width, gpu.config.height, camera);

            let uniform = SceneUniform {
                mvp: mvp.to_cols_array_2d(),
                use_texture: self.style.use_texture as u32,
                tiles_x: gpu.tiles_x,
                debug_faces: debug_faces as u32,
                _pad0: 0,
                tile_uv_size: gpu.tile_uv_size,
                _pad1: [0.0; 2],
            };

            gpu.queue
                .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

            let frame = match gpu.surface.get_current_texture() {
                Ok(frame) => frame,
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    return;
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    std::process::exit(1);
                }
                Err(_) => return,
            };
            let view = frame.texture.create_view(&Default::default());

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cube_encoder"),
                });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("cube_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.52,
                                g: 0.73,
                                b: 0.95,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gpu.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                rpass.set_pipeline(&gpu.render_pipeline);
                rpass.set_bind_group(0, &gpu.bind_group, &[]);
                for chunk in gpu.chunks.values() {
                    if !chunk_visible(camera.position, camera.forward, chunk) {
                        continue;
                    }
                    rpass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    rpass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }

            gpu.queue.submit([encoder.finish()]);
            frame.present();
        });
    }

    pub fn window(&self) -> &winit::window::Window {
        self.cell.borrow_owner()
    }

    pub fn upsert_chunk(
        &mut self,
        coord: IVec3,
        center: Vec3,
        radius: f32,
        vertices: &[ChunkVertex],
        indices: &[u32],
    ) {
        self.cell.with_dependent_mut(|_, gpu| {
            let vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk_vertex_buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk_index_buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            gpu.chunks.insert(
                coord,
                ChunkGpuMesh {
                    vertex_buffer,
                    index_buffer,
                    index_count: indices.len() as u32,
                    center,
                    radius,
                },
            );
        });
    }

    pub fn remove_chunk(&mut self, coord: IVec3) {
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.chunks.remove(&coord);
        });
    }

    pub fn clear_chunks(&mut self) {
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.chunks.clear();
        });
    }
}

fn build_mvp(width: u32, height: u32, camera: &Camera) -> Mat4 {
    let aspect = width as f32 / height as f32;
    let proj = Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 6000.0);
    let view = camera.view_matrix();
    let model = Mat4::IDENTITY;
    proj * view * model
}

fn create_depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &TextureAtlas,
) -> (wgpu::TextureView, wgpu::Sampler, u32, [f32; 2]) {
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

    (view, sampler, tiles_x, tile_uv_size)
}

fn create_dummy_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::TextureView, wgpu::Sampler, u32, [f32; 2]) {
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
    (view, sampler, 1, [1.0, 1.0])
}

fn chunk_visible(camera_pos: Vec3, camera_forward: Vec3, chunk: &ChunkGpuMesh) -> bool {
    let to_center = chunk.center - camera_pos;
    let dist_sq = to_center.length_squared();
    let height_chunks = (camera_pos.y / crate::world::CHUNK_SIZE as f32).max(0.0).floor();
    let extra_radius = height_chunks * 2.0 * crate::world::CHUNK_SIZE as f32;
    let radius = chunk.radius + extra_radius;
    if dist_sq <= radius * radius {
        return true;
    }
    let dist = dist_sq.sqrt();
    let forward = camera_forward.normalize();
    let dir = to_center / dist;
    let dot = forward.dot(dir);
    if dot <= 0.0 {
        return false;
    }
    let base_fov = 45.0_f32.to_radians();
    let expanded_fov = base_fov + 2.0_f32.to_radians();
    let half_fov = expanded_fov * 0.5;
    let margin = (radius / dist).asin();
    dot >= (half_fov + margin).cos()
}
