use crate::player::Camera;
use crate::render::atlas::TextureAtlas;
use crate::render::CubeStyle;
use crate::render::mesh::ChunkVertex;
use crate::render::texture::{AtlasTexture, create_dummy_texture, load_atlas_texture};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3};
use self_cell::self_cell;
use std::collections::{HashMap, HashSet, VecDeque};
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    mvp: [[f32; 4]; 4],
    use_texture: u32,
    tiles_x: u32,
    debug_faces: u32,
    debug_chunks: u32,
    tile_uv_size: [f32; 2],
    chunk_size: f32,
    _pad0: f32,
}

struct SuperChunkGpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    center: Vec3,
    radius: f32,
    line_buffer: wgpu::Buffer,
    line_count: u32,
    slots: HashMap<IVec3, ChunkSlot>,
    vertex_capacity: u32,
}

struct CpuChunkMesh {
    vertices: Vec<ChunkVertex>,
    indices: Vec<u32>,
}

struct PendingUpdate {
    vertices: Vec<ChunkVertex>,
    indices: Vec<u32>,
}

#[derive(Clone)]
struct ChunkSlot {
    vertex_offset: u32,
    vertex_capacity: u32,
    index_offset: u32,
    index_capacity: u32,
}

const SUPER_CHUNK_SIZE: i32 = 4;

#[allow(dead_code)]
struct GpuInner<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tiles_x: u32,
    tile_uv_size: [f32; 2],
    chunks: HashMap<IVec3, CpuChunkMesh>,
    super_chunks: HashMap<IVec3, SuperChunkGpuMesh>,
    dirty_supers: Vec<IVec3>,
    dirty_set: HashSet<IVec3>,
    visible_supers: Vec<IVec3>,
    pending_updates: HashMap<IVec3, PendingUpdate>,
    pending_queue: VecDeque<IVec3>,
    selection_buffer: wgpu::Buffer,
    selection_count: u32,
    selection_coord: Option<IVec3>,
    staged_indices: Vec<u32>,
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

pub struct GpuStats {
    pub chunks: usize,
    pub super_chunks: usize,
    pub dirty_supers: usize,
    pub visible_supers: usize,
    pub pending_updates: usize,
    pub pending_queue: usize,
    pub total_indices: u64,
    pub total_vertices_capacity: u64,
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

            let AtlasTexture {
                view: texture_view,
                sampler,
                tiles_x,
                tile_uv_size,
            } = if style.use_texture {
                let atlas = atlas.expect("TextureAtlas required when use_texture is true");
                load_atlas_texture(&device, &queue, &atlas)
            } else {
                create_dummy_texture(&device, &queue)
            };

            let uniform = SceneUniform {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                use_texture: style.use_texture as u32,
                tiles_x,
                debug_faces: 0,
                debug_chunks: 0,
                tile_uv_size,
                chunk_size: crate::world::CHUNK_SIZE as f32,
                _pad0: 0.0,
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

            let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("line_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/lines.wgsl").into()),
            });

            let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("line_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &line_shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &line_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

            let selection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("selection_line_buffer"),
                size: (std::mem::size_of::<[f32; 3]>() * 24) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            GpuInner {
                surface,
                device,
                queue,
                config,
                render_pipeline,
                line_pipeline,
                depth_view,
                uniform_buffer,
                bind_group,
                tiles_x,
                tile_uv_size,
                chunks: HashMap::new(),
                super_chunks: HashMap::new(),
                dirty_supers: Vec::new(),
                dirty_set: HashSet::new(),
                visible_supers: Vec::new(),
                pending_updates: HashMap::new(),
                pending_queue: VecDeque::new(),
                selection_buffer,
                selection_count: 0,
                selection_coord: None,
                staged_indices: Vec::new(),
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

    pub fn render(
        &mut self,
        camera: &Camera,
        debug_faces: bool,
        debug_chunks: bool,
        _draw_radius: i32,
    ) {
        self.cell.with_dependent_mut(|_, gpu| {
            apply_pending_uploads(gpu, 16);
            let mvp = build_mvp(gpu.config.width, gpu.config.height, camera);

            let uniform = SceneUniform {
                mvp: mvp.to_cols_array_2d(),
                use_texture: self.style.use_texture as u32,
                tiles_x: gpu.tiles_x,
                debug_faces: debug_faces as u32,
                debug_chunks: 0,
                tile_uv_size: gpu.tile_uv_size,
                chunk_size: crate::world::CHUNK_SIZE as f32,
                _pad0: 0.0,
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

                rpass.set_bind_group(0, &gpu.bind_group, &[]);
                rpass.set_pipeline(&gpu.render_pipeline);
                for coord in &gpu.visible_supers {
                    let Some(chunk) = gpu.super_chunks.get(coord) else {
                        continue;
                    };
                    rpass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    rpass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }

                if debug_chunks {
                    rpass.set_pipeline(&gpu.line_pipeline);
                    for chunk in gpu.super_chunks.values() {
                        rpass.set_vertex_buffer(0, chunk.line_buffer.slice(..));
                        rpass.draw(0..chunk.line_count, 0..1);
                    }
                }

                if gpu.selection_count > 0 {
                    rpass.set_pipeline(&gpu.line_pipeline);
                    rpass.set_vertex_buffer(0, gpu.selection_buffer.slice(..));
                    rpass.draw(0..gpu.selection_count, 0..1);
                }
            }

            gpu.queue.submit([encoder.finish()]);
            frame.present();
        });
    }

    pub fn set_selection(&mut self, coord: Option<IVec3>) {
        self.cell.with_dependent_mut(|_, gpu| {
            if coord == gpu.selection_coord {
                return;
            }
            gpu.selection_coord = coord;
            if let Some(coord) = coord {
                let vertices = block_outline_vertices(coord);
                gpu.selection_count = vertices.len() as u32;
                gpu.queue
                    .write_buffer(&gpu.selection_buffer, 0, bytemuck::cast_slice(&vertices));
            } else {
                gpu.selection_count = 0;
            }
        });
    }

    pub fn window(&self) -> &winit::window::Window {
        self.cell.borrow_owner()
    }

    pub fn stats(&self) -> GpuStats {
        self.cell.with_dependent(|_, gpu| {
            let mut total_indices = 0u64;
            let mut total_vertices_capacity = 0u64;
            for chunk in gpu.super_chunks.values() {
                total_indices += chunk.index_count as u64;
                total_vertices_capacity += chunk.vertex_capacity as u64;
            }
            GpuStats {
                chunks: gpu.chunks.len(),
                super_chunks: gpu.super_chunks.len(),
                dirty_supers: gpu.dirty_supers.len(),
                visible_supers: gpu.visible_supers.len(),
                pending_updates: gpu.pending_updates.len(),
                pending_queue: gpu.pending_queue.len(),
                total_indices,
                total_vertices_capacity,
            }
        })
    }

    pub fn upsert_chunk(
        &mut self,
        coord: IVec3,
        _center: Vec3,
        _radius: f32,
        vertices: Vec<ChunkVertex>,
        indices: Vec<u32>,
    ) {
        self.cell.with_dependent_mut(|_, gpu| {
            let schedule_vertices = vertices.clone();
            let schedule_indices = indices.clone();
            gpu.chunks.insert(
                coord,
                CpuChunkMesh {
                    vertices,
                    indices,
                },
            );
            if !schedule_superchunk_update(gpu, coord, schedule_vertices, schedule_indices) {
                mark_super_dirty(gpu, super_chunk_coord(coord));
            }
        });
    }

    pub fn remove_chunk(&mut self, coord: IVec3) {
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.chunks.remove(&coord);
            let super_coord = super_chunk_coord(coord);
            let slot = gpu
                .super_chunks
                .get(&super_coord)
                .and_then(|super_chunk| super_chunk.slots.get(&coord).cloned());
            if let Some(slot) = slot {
                if let Some(super_chunk) = gpu.super_chunks.get_mut(&super_coord) {
                    clear_chunk_slot(&gpu.queue, super_chunk, coord, &slot);
                } else {
                    mark_super_dirty(gpu, super_coord);
                }
            } else {
                mark_super_dirty(gpu, super_coord);
            }
        });
    }

    pub fn clear_chunks(&mut self) {
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.chunks.clear();
            gpu.super_chunks.clear();
            gpu.visible_supers.clear();
            gpu.dirty_supers.clear();
            gpu.dirty_set.clear();
            gpu.pending_updates.clear();
            gpu.pending_queue.clear();
        });
    }

    pub fn rebuild_dirty_superchunks(&mut self, camera_pos: Vec3, budget: usize) {
        self.cell.with_dependent_mut(|_, gpu| {
            for _ in 0..budget {
                if gpu.dirty_supers.is_empty() {
                    break;
                }
                let mut best_i = 0usize;
                let mut best_dist = f32::MAX;
                for (i, coord) in gpu.dirty_supers.iter().enumerate() {
                    let center = super_chunk_center(*coord);
                    let dist = (center - camera_pos).length_squared();
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                    }
                }
                let coord = gpu.dirty_supers.swap_remove(best_i);
                gpu.dirty_set.remove(&coord);
                rebuild_superchunk(gpu, coord);
            }
        });
    }

    pub fn update_visible(&mut self, camera: &Camera, draw_radius: i32) {
        self.cell.with_dependent_mut(|_, gpu| {
            let draw_radius = draw_radius as f32 * crate::world::CHUNK_SIZE as f32;
            let draw_radius_sq = draw_radius * draw_radius;
            gpu.visible_supers.clear();
            gpu.visible_supers.reserve(gpu.super_chunks.len());
            for (coord, chunk) in &gpu.super_chunks {
                let to_center = chunk.center - camera.position;
                if to_center.length_squared() > draw_radius_sq {
                    continue;
                }
                if !chunk_visible(camera.position, camera.forward, chunk) {
                    continue;
                }
                gpu.visible_supers.push(*coord);
            }
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

fn super_chunk_coord(chunk: IVec3) -> IVec3 {
    IVec3::new(
        div_floor(chunk.x, SUPER_CHUNK_SIZE),
        div_floor(chunk.y, SUPER_CHUNK_SIZE),
        div_floor(chunk.z, SUPER_CHUNK_SIZE),
    )
}

fn div_floor(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn mark_super_dirty(gpu: &mut GpuInner, coord: IVec3) {
    if gpu.dirty_set.insert(coord) {
        gpu.dirty_supers.push(coord);
    }
}

fn rebuild_superchunk(gpu: &mut GpuInner, coord: IVec3) {
    let mut vertices: Vec<ChunkVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut slots: HashMap<IVec3, ChunkSlot> = HashMap::new();
    let mut vertex_cursor = 0u32;
    let mut index_cursor = 0u32;
    let origin = IVec3::new(
        coord.x * SUPER_CHUNK_SIZE,
        coord.y * SUPER_CHUNK_SIZE,
        coord.z * SUPER_CHUNK_SIZE,
    );

    for z in 0..SUPER_CHUNK_SIZE {
        for y in 0..SUPER_CHUNK_SIZE {
            for x in 0..SUPER_CHUNK_SIZE {
                let c = IVec3::new(origin.x + x, origin.y + y, origin.z + z);
                let Some(chunk) = gpu.chunks.get(&c) else {
                    continue;
                };
                if chunk.vertices.is_empty() || chunk.indices.is_empty() {
                    continue;
                }
                let vertex_capacity = next_pow2_u32(chunk.vertices.len().max(1) as u32);
                let mut index_capacity = next_pow2_u32(chunk.indices.len().max(3) as u32);
                if index_capacity % 3 != 0 {
                    index_capacity += 3 - (index_capacity % 3);
                }
                let slot = ChunkSlot {
                    vertex_offset: vertex_cursor,
                    vertex_capacity,
                    index_offset: index_cursor,
                    index_capacity,
                };
                slots.insert(c, slot.clone());

                vertices.extend_from_slice(&chunk.vertices);
                vertices.resize((vertex_cursor + vertex_capacity) as usize, ChunkVertex::zeroed());

                indices.extend(chunk.indices.iter().map(|i| i + vertex_cursor));
                let pad = index_capacity.saturating_sub(chunk.indices.len() as u32);
                if pad > 0 {
                    let degenerate = vertex_cursor;
                    indices.extend(std::iter::repeat(degenerate).take(pad as usize));
                }

                vertex_cursor += vertex_capacity;
                index_cursor += index_capacity;
            }
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        gpu.super_chunks.remove(&coord);
        return;
    }

    let vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("super_chunk_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("super_chunk_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });

    let size = SUPER_CHUNK_SIZE as f32 * crate::world::CHUNK_SIZE as f32;
    let half = size * 0.5;
    let center = super_chunk_center(coord);
    let radius = half * (3.0f32).sqrt();

    let line_vertices = super_chunk_wireframe_vertices(coord);
    let line_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("super_chunk_line_buffer"),
        contents: bytemuck::cast_slice(&line_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    gpu.super_chunks.insert(
        coord,
        SuperChunkGpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            center,
            radius,
            line_buffer,
            line_count: line_vertices.len() as u32,
            slots,
            vertex_capacity: vertex_cursor,
        },
    );
}

fn super_chunk_center(coord: IVec3) -> Vec3 {
    let size = SUPER_CHUNK_SIZE as f32 * crate::world::CHUNK_SIZE as f32;
    let half = size * 0.5;
    Vec3::new(
        coord.x as f32 * size + half - crate::world::CHUNK_SIZE as f32 * 0.5,
        coord.y as f32 * size + half - crate::world::CHUNK_SIZE as f32 * 0.5,
        coord.z as f32 * size + half - crate::world::CHUNK_SIZE as f32 * 0.5,
    )
}
fn super_chunk_wireframe_vertices(coord: IVec3) -> Vec<[f32; 3]> {
    let size = SUPER_CHUNK_SIZE as f32 * crate::world::CHUNK_SIZE as f32;
    let half = size * 0.5;
    let min = Vec3::new(
        coord.x as f32 * size - half + crate::world::CHUNK_SIZE as f32 * 0.5,
        coord.y as f32 * size - half + crate::world::CHUNK_SIZE as f32 * 0.5,
        coord.z as f32 * size - half + crate::world::CHUNK_SIZE as f32 * 0.5,
    );
    let max = Vec3::new(min.x + size, min.y + size, min.z + size);

    let p000 = [min.x, min.y, min.z];
    let p001 = [min.x, min.y, max.z];
    let p010 = [min.x, max.y, min.z];
    let p011 = [min.x, max.y, max.z];
    let p100 = [max.x, min.y, min.z];
    let p101 = [max.x, min.y, max.z];
    let p110 = [max.x, max.y, min.z];
    let p111 = [max.x, max.y, max.z];

    vec![
        p000, p001, p000, p010, p000, p100,
        p111, p110, p111, p101, p111, p011,
        p001, p011, p001, p101,
        p010, p011, p010, p110,
        p100, p101, p100, p110,
    ]
}

fn block_outline_vertices(coord: IVec3) -> Vec<[f32; 3]> {
    let min = Vec3::new(
        coord.x as f32 - 0.002,
        coord.y as f32 - 0.002,
        coord.z as f32 - 0.002,
    );
    let max = Vec3::new(
        coord.x as f32 + 1.002,
        coord.y as f32 + 1.002,
        coord.z as f32 + 1.002,
    );

    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];

    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut vertices = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        vertices.push(corners[a].to_array());
        vertices.push(corners[b].to_array());
    }
    vertices
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

fn chunk_visible(camera_pos: Vec3, camera_forward: Vec3, chunk: &SuperChunkGpuMesh) -> bool {
    let to_center = chunk.center - camera_pos;
    let dist_sq = to_center.length_squared();
    if dist_sq <= chunk.radius * chunk.radius {
        return true;
    }
    let dist = dist_sq.sqrt();
    let forward = camera_forward.normalize();
    let dir = to_center / dist;
    let dot = forward.dot(dir);
    if dot <= 0.0 {
        return false;
    }
    let base_fov = 60.0_f32.to_radians();
    let half_fov = base_fov * 0.5;
    let margin = (chunk.radius / dist).asin();
    dot >= (half_fov + margin).cos()
}

fn next_pow2_u32(v: u32) -> u32 {
    v.next_power_of_two().max(1)
}

fn schedule_superchunk_update(
    gpu: &mut GpuInner,
    coord: IVec3,
    vertices: Vec<ChunkVertex>,
    indices: Vec<u32>,
) -> bool {
    let super_coord = super_chunk_coord(coord);
    let Some(slot) = gpu
        .super_chunks
        .get(&super_coord)
        .and_then(|super_chunk| super_chunk.slots.get(&coord).cloned())
    else {
        return false;
    };
    if vertices.is_empty() || indices.is_empty() {
        let Some(super_chunk) = gpu.super_chunks.get_mut(&super_coord) else {
            return false;
        };
        clear_chunk_slot(&gpu.queue, super_chunk, coord, &slot);
        return true;
    }
    if vertices.len() as u32 > slot.vertex_capacity {
        return false;
    }
    if indices.len() as u32 > slot.index_capacity {
        return false;
    }
    if gpu.pending_updates.insert(
        coord,
        PendingUpdate {
            vertices,
            indices,
        },
    ).is_none() {
        gpu.pending_queue.push_back(coord);
    }
    true
}

fn clear_chunk_slot(
    queue: &wgpu::Queue,
    super_chunk: &mut SuperChunkGpuMesh,
    coord: IVec3,
    slot: &ChunkSlot,
) {
    let zero_indices = vec![slot.vertex_offset; slot.index_capacity as usize];
    let i_start = slot.index_offset as wgpu::BufferAddress
        * std::mem::size_of::<u32>() as wgpu::BufferAddress;
    queue.write_buffer(
        &super_chunk.index_buffer,
        i_start,
        bytemuck::cast_slice(&zero_indices),
    );
    super_chunk.slots.remove(&coord);
}

fn apply_pending_uploads(gpu: &mut GpuInner, budget: usize) {
    for _ in 0..budget {
        let Some(coord) = gpu.pending_queue.pop_front() else {
            break;
        };
        let Some(update) = gpu.pending_updates.remove(&coord) else {
            continue;
        };
        let super_coord = super_chunk_coord(coord);
        let Some(super_chunk) = gpu.super_chunks.get(&super_coord) else {
            mark_super_dirty(gpu, super_coord);
            continue;
        };
        let Some(slot) = super_chunk.slots.get(&coord).cloned() else {
            mark_super_dirty(gpu, super_coord);
            continue;
        };
        if update.vertices.len() as u32 > slot.vertex_capacity
            || update.indices.len() as u32 > slot.index_capacity
        {
            mark_super_dirty(gpu, super_coord);
            continue;
        }

        let v_start = slot.vertex_offset as wgpu::BufferAddress
            * std::mem::size_of::<ChunkVertex>() as wgpu::BufferAddress;
        gpu.queue.write_buffer(
            &super_chunk.vertex_buffer,
            v_start,
            bytemuck::cast_slice(&update.vertices),
        );

        gpu.staged_indices.clear();
        if gpu.staged_indices.capacity() < slot.index_capacity as usize {
            gpu.staged_indices
                .reserve(slot.index_capacity as usize - gpu.staged_indices.capacity());
        }
        gpu.staged_indices
            .extend(update.indices.iter().map(|i| i + slot.vertex_offset));
        let pad = slot.index_capacity.saturating_sub(update.indices.len() as u32);
        if pad > 0 {
            gpu.staged_indices
                .extend(std::iter::repeat(slot.vertex_offset).take(pad as usize));
        }
        let i_start = slot.index_offset as wgpu::BufferAddress
            * std::mem::size_of::<u32>() as wgpu::BufferAddress;
        gpu.queue.write_buffer(
            &super_chunk.index_buffer,
            i_start,
            bytemuck::cast_slice(&gpu.staged_indices),
        );
    }
}
