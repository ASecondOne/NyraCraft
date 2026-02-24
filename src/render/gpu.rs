use crate::app::menu::{PauseMenuButton, compute_pause_menu_layout, hit_test_pause_menu_button};
use crate::player::Camera;
use crate::player::inventory::{
    CRAFT_GRID_SIDE, CRAFT_GRID_SLOTS, INVENTORY_STORAGE_SLOTS, InventorySlotRef, ItemStack,
    compute_inventory_layout, craft_input_slot_rect, craft_output_slot_rect, hotbar_slot_rect,
    storage_slot_rect,
};
use crate::render::CubeStyle;
use crate::render::atlas::TextureAtlas;
use crate::render::mesh::{ChunkVertex, PackedFarVertex};
use crate::render::texture::{
    AtlasTexture, create_dummy_texture, load_atlas_texture, load_grass_colormap_texture,
};
use crate::world::blocks::{
    BLOCK_CRAFTING_TABLE, BLOCK_DIRT, BLOCK_GRASS, BLOCK_GRAVEL, BLOCK_LEAVES, BLOCK_LOG,
    BLOCK_PLANKS_OAK, BLOCK_SAND, BLOCK_STONE, DESTROY_STAGE_COUNT, DESTROY_STAGE_TILE_START,
    HOTBAR_SLOTS, ITEM_APPLE, ITEM_STICK, block_count, block_texture_by_id, item_icon_tile_index,
    item_max_durability, item_max_stack_size, placeable_block_id_for_item,
};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat3, Mat4, Vec2, Vec3};
use self_cell::self_cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    mvp: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    tile_misc: [f32; 4],     // [tile_uv_x, tile_uv_y, chunk_size, colormap_scale]
    flags0: [u32; 4],        // [use_texture, tiles_x, debug_faces, debug_chunks]
    flags1: [u32; 4], // [occlusion_cull, grass_top_tile, grass_side_tile, grass_overlay_tile]
    colormap_misc: [f32; 4], // [colormap_strength, reserved, reserved, reserved]
    item_misc: [f32; 4], // [item_tile_uv_x, item_tile_uv_y, item_tiles_x, reserved]
    light_misc: [u32; 4], // [point_light_count, reserved, reserved, reserved]
    sun_dir: [f32; 4], // [dir_x, dir_y, dir_z, sun_strength]
    point_light_pos_radius: [[f32; 4]; MAX_POINT_LIGHTS], // [x, y, z, radius]
    point_light_color_intensity: [[f32; 4]; MAX_POINT_LIGHTS], // [r, g, b, intensity]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex {
    position: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
    use_texture: f32,
    atlas_select: f32,
}

impl UiVertex {
    fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    shader_location: 2,
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>())
                        as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    shader_location: 3,
                    offset: (std::mem::size_of::<[f32; 2]>()
                        + std::mem::size_of::<[f32; 4]>()
                        + std::mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    shader_location: 4,
                    offset: (std::mem::size_of::<[f32; 2]>()
                        + std::mem::size_of::<[f32; 4]>()
                        + std::mem::size_of::<[f32; 2]>()
                        + std::mem::size_of::<f32>())
                        as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

struct SuperChunkGpuMesh {
    raw_vertex_buffer: Option<wgpu::Buffer>,
    raw_index_buffer: Option<wgpu::Buffer>,
    raw_index_count: u32,
    packed_vertex_buffer: Option<wgpu::Buffer>,
    packed_index_buffer: Option<wgpu::Buffer>,
    packed_index_count: u32,
    center: Vec3,
    radius: f32,
    line_buffer: wgpu::Buffer,
    line_count: u32,
    raw_slots: HashMap<IVec3, ChunkSlot>,
    packed_slots: HashMap<IVec3, ChunkSlot>,
    raw_vertex_capacity: u32,
    packed_vertex_capacity: u32,
}

#[derive(Clone)]
enum ChunkVertices {
    Raw(Arc<[ChunkVertex]>),
    PackedFar(Arc<[PackedFarVertex]>),
}

struct CpuChunkMesh {
    vertices: ChunkVertices,
    indices: Arc<[u32]>,
}

struct PendingUpdate {
    vertices: ChunkVertices,
    indices: Arc<[u32]>,
}

#[derive(Clone, Copy)]
struct ChunkSlot {
    vertex_offset: u32,
    vertex_capacity: u32,
    index_offset: u32,
    index_capacity: u32,
}

const SUPER_CHUNK_SIZE: i32 = 4;
const HOTBAR_SLOT_COUNT: usize = HOTBAR_SLOTS;
const UI_ATLAS_ITEM: f32 = 0.0;
const UI_ATLAS_BLOCK: f32 = 1.0;
const SUN_TRANSPARENT_MODE: u32 = 15;
const DAY_CYCLE_SECONDS: f32 = 48.0 * 60.0; // 2 min per in-game hour, 48 min full day.
const MAX_POINT_LIGHTS: usize = 12;
const POINT_LIGHT_CULL_DISTANCE: f32 = 84.0;
const POINT_LIGHT_DROPPED_SAMPLE_CAP: usize = 192;

#[derive(Clone, Copy)]
struct DayCycleState {
    day_progress: f32,
    daylight: f32,
    sky_mix: f32,
    sun_height: f32,
}

#[derive(Clone, Copy)]
struct PointLight {
    position: Vec3,
    radius: f32,
    color: Vec3,
    intensity: f32,
}

struct PointLightCullResult {
    count: u32,
    pos_radius: [[f32; 4]; MAX_POINT_LIGHTS],
    color_intensity: [[f32; 4]; MAX_POINT_LIGHTS],
}

fn sample_day_cycle(elapsed_seconds: f32) -> DayCycleState {
    let day_progress = (elapsed_seconds / DAY_CYCLE_SECONDS).rem_euclid(1.0);
    // Noon at t=0, midnight at 0.5 cycle.
    let cycle_angle = day_progress * std::f32::consts::TAU;
    let sun_height = cycle_angle.cos();
    let day01 = ((sun_height + 1.0) * 0.5).clamp(0.0, 1.0);
    // Keep a small ambient floor so nights stay readable.
    let daylight = (0.08 + 0.92 * day01.powf(1.35)).clamp(0.08, 1.0);
    let sky_mix = day01.powf(0.8);
    DayCycleState {
        day_progress,
        daylight,
        sky_mix,
        sun_height,
    }
}

fn sun_direction(day_progress: f32, sun_height: f32) -> Vec3 {
    let orbit_angle = day_progress * std::f32::consts::TAU;
    let dir =
        Vec3::new(orbit_angle.sin(), sun_height, orbit_angle.cos() * 0.55).normalize_or_zero();
    if dir.length_squared() > 1.0e-8 {
        dir
    } else {
        Vec3::Y
    }
}

fn point_light_tint(block_id: i8) -> Vec3 {
    let (top, left, right) = block_icon_colors(block_id);
    let tint = Vec3::new(
        (top[0] + left[0] + right[0]) / 3.0,
        (top[1] + left[1] + right[1]) / 3.0,
        (top[2] + left[2] + right[2]) / 3.0,
    );
    tint.max(Vec3::splat(0.22)).min(Vec3::splat(1.0))
}

fn build_culled_point_lights(
    camera: &Camera,
    day_cycle: DayCycleState,
    dropped_items: &[DroppedItemRender],
    break_overlay: Option<(IVec3, u32)>,
) -> PointLightCullResult {
    let camera_forward = if camera.forward.length_squared() > 1.0e-8 {
        camera.forward.normalize()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    let night_factor = (1.0 - day_cycle.daylight).clamp(0.0, 1.0);
    let mut candidates: Vec<PointLight> = Vec::with_capacity(POINT_LIGHT_DROPPED_SAMPLE_CAP + 3);

    if let Some((coord, _stage)) = break_overlay {
        candidates.push(PointLight {
            position: Vec3::new(
                coord.x as f32 + 0.5,
                coord.y as f32 + 0.5,
                coord.z as f32 + 0.5,
            ),
            radius: 7.5 + 3.0 * night_factor,
            color: Vec3::new(1.0, 0.8, 0.42),
            intensity: 0.35 + 0.9 * night_factor,
        });
    }

    if night_factor > 0.2 {
        for item in dropped_items.iter().take(POINT_LIGHT_DROPPED_SAMPLE_CAP) {
            candidates.push(PointLight {
                position: item.position + Vec3::new(0.0, 0.12, 0.0),
                radius: 3.2 + 3.2 * night_factor,
                color: point_light_tint(item.block_id),
                intensity: 0.06 + 0.28 * night_factor,
            });
        }
    }

    let mut scored: Vec<(f32, PointLight)> = Vec::with_capacity(candidates.len());
    for light in candidates {
        let to_light = light.position - camera.position;
        let dist = to_light.length();
        if dist > POINT_LIGHT_CULL_DISTANCE + light.radius {
            continue;
        }
        let dir = if dist > 1.0e-5 {
            to_light / dist
        } else {
            camera_forward
        };
        let facing = ((camera_forward.dot(dir) + 0.35) * 0.74).clamp(0.0, 1.0);
        let attenuation = 1.0 / (1.0 + dist * 0.18 + dist * dist * 0.018);
        let score = light.intensity * (0.3 + 0.7 * facing) * attenuation;
        if score <= 0.001 {
            continue;
        }
        scored.push((score, light));
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut result = PointLightCullResult {
        count: 0,
        pos_radius: [[0.0; 4]; MAX_POINT_LIGHTS],
        color_intensity: [[0.0; 4]; MAX_POINT_LIGHTS],
    };

    for (_, light) in scored.into_iter().take(MAX_POINT_LIGHTS) {
        let idx = result.count as usize;
        result.pos_radius[idx] = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.radius.max(0.05),
        ];
        result.color_intensity[idx] = [
            light.color.x,
            light.color.y,
            light.color.z,
            light.intensity.max(0.0),
        ];
        result.count += 1;
    }

    result
}

struct GpuInner<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    packed_render_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    ui_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    adapter_summary: String,
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
    ui_vertex_buffer: wgpu::Buffer,
    ui_vertex_capacity: usize,
    ui_item_tiles_x: u32,
    ui_item_tile_uv_size: [f32; 2],
    dropped_mesh: DynamicMeshBuffer,
    break_overlay_mesh: DynamicMeshBuffer,
    sun_mesh: DynamicMeshBuffer,
    staged_indices: Vec<u32>,
}

#[derive(Default)]
struct DynamicMeshBuffer {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    vertex_capacity: usize,
    index_capacity: usize,
    index_count: u32,
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
    pub super_chunks: usize,
    pub dirty_supers: usize,
    pub visible_supers: usize,
    pub pending_updates: usize,
    pub pending_queue: usize,
    pub total_indices: u64,
    pub visible_indices: u64,
    pub visible_raw_indices: u64,
    pub visible_packed_indices: u64,
    pub total_draw_calls_est: u64,
    pub visible_draw_calls_est: u64,
    pub total_raw_vertices_capacity: u64,
    pub total_packed_vertices_capacity: u64,
}

#[derive(Clone, Copy)]
pub struct DroppedItemRender {
    pub position: Vec3,
    pub block_id: i8,
    pub spin_y: f32,
    pub tilt_z: f32,
}

impl Gpu {
    pub fn new(
        window: winit::window::Window,
        style: CubeStyle,
        atlas: Option<TextureAtlas>,
    ) -> Self {
        let cell = GpuCell::new(window, |window| {
            let size = window.inner_size();

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                dx12_shader_compiler: Default::default(),
                flags: wgpu::InstanceFlags::default(),
                gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
            });
            let surface = instance.create_surface(window).unwrap();

            let request_adapter =
                |power_preference: wgpu::PowerPreference,
                 force_fallback_adapter: bool,
                 compatible_surface: Option<&wgpu::Surface<'_>>| {
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        compatible_surface,
                        power_preference,
                        force_fallback_adapter,
                    }))
                };

            let attempts = [
                (wgpu::PowerPreference::HighPerformance, false),
                (wgpu::PowerPreference::LowPower, false),
                (wgpu::PowerPreference::HighPerformance, true),
                (wgpu::PowerPreference::LowPower, true),
            ];

            let mut adapter = None;
            for (power_preference, fallback) in attempts {
                adapter = request_adapter(power_preference, fallback, Some(&surface));
                if adapter.is_some() {
                    break;
                }
            }
            if adapter.is_none() {
                for (power_preference, fallback) in attempts {
                    adapter = request_adapter(power_preference, fallback, None)
                        .filter(|a| a.is_surface_supported(&surface));
                    if adapter.is_some() {
                        break;
                    }
                }
            }
            let adapter = adapter
                .unwrap_or_else(|| {
                    panic!(
                        "gpu: failed to acquire adapter for this surface after trying all backends/power prefs/fallbacks"
                    )
                });
            let adapter_info = adapter.get_info();
            let adapter_name = adapter_info.name.trim();
            let adapter_summary = format!(
                "{} ({:?}, {:?})",
                if adapter_name.is_empty() {
                    "Unknown GPU"
                } else {
                    adapter_name
                },
                adapter_info.device_type,
                adapter_info.backend
            );

            let (device, queue) = pollster::block_on(
                adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
            )
            .unwrap();

            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .or_else(|| caps.formats.first().copied())
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
            let alpha_mode = caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto);

            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

            surface.configure(&device, &config);

            let depth_view = create_depth_view(&device, &config);

            let AtlasTexture {
                view: texture_view,
                sampler: atlas_sampler,
                tiles_x,
                tile_uv_size,
            } = if style.use_texture {
                let atlas = atlas.expect("TextureAtlas required when use_texture is true");
                load_atlas_texture(&device, &queue, &atlas)
            } else {
                create_dummy_texture(&device, &queue)
            };

            let grass_colormap = load_grass_colormap_texture(&device, &queue);
            let AtlasTexture {
                view: ui_item_texture_view,
                sampler: ui_item_sampler,
                tiles_x: ui_item_tiles_x,
                tile_uv_size: ui_item_tile_uv_size,
            } = load_atlas_texture(
                &device,
                &queue,
                &TextureAtlas {
                    path: "src/texturing/atlas_output/atlas_items.png".to_string(),
                    tile_size: 16,
                },
            );
            let AtlasTexture {
                view: sun_texture_view,
                sampler: sun_sampler,
                ..
            } = if Path::new("src/texturing/sun_vv.png").exists() {
                load_atlas_texture(
                    &device,
                    &queue,
                    &TextureAtlas {
                        path: "src/texturing/sun_vv.png".to_string(),
                        tile_size: 32,
                    },
                )
            } else {
                create_dummy_texture(&device, &queue)
            };

            let uniform = SceneUniform {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 0.0, 0.0],
                tile_misc: [
                    tile_uv_size[0],
                    tile_uv_size[1],
                    crate::world::CHUNK_SIZE as f32,
                    0.0015,
                ],
                flags0: [style.use_texture as u32, tiles_x, 0, 0],
                flags1: [0, 2, 3, 6],
                colormap_misc: [0.72, 0.0, 0.0, 0.0],
                item_misc: [
                    ui_item_tile_uv_size[0],
                    ui_item_tile_uv_size[1],
                    ui_item_tiles_x as f32,
                    0.0,
                ],
                light_misc: [0, 0, 0, 0],
                sun_dir: [0.0, 1.0, 0.0, 0.0],
                point_light_pos_radius: [[0.0; 4]; MAX_POINT_LIGHTS],
                point_light_color_intensity: [[0.0; 4]; MAX_POINT_LIGHTS],
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
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
                        resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&grass_colormap.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&grass_colormap.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&ui_item_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&ui_item_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&sun_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(&sun_sampler),
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

            let packed_render_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("cube_packed_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main_packed",
                        buffers: &[PackedFarVertex::layout()],
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

            let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ui_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
            });
            let ui_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ui_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui_bind_group"),
                layout: &ui_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ui_item_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&ui_item_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                    },
                ],
            });
            let ui_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ui_pipeline_layout"),
                    bind_group_layouts: &[&ui_bind_group_layout],
                    push_constant_ranges: &[],
                });
            let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui_pipeline"),
                layout: Some(&ui_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ui_shader,
                    entry_point: "vs_main",
                    buffers: &[UiVertex::layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &ui_shader,
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
            let ui_vertex_capacity = 1024usize;
            let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_vertex_buffer"),
                size: (ui_vertex_capacity * std::mem::size_of::<UiVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            GpuInner {
                surface,
                device,
                queue,
                config,
                render_pipeline,
                packed_render_pipeline,
                line_pipeline,
                ui_pipeline,
                ui_bind_group,
                depth_view,
                uniform_buffer,
                bind_group,
                adapter_summary,
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
                ui_vertex_buffer,
                ui_vertex_capacity,
                ui_item_tiles_x,
                ui_item_tile_uv_size,
                dropped_mesh: DynamicMeshBuffer::default(),
                break_overlay_mesh: DynamicMeshBuffer::default(),
                sun_mesh: DynamicMeshBuffer::default(),
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
        elapsed_seconds: f32,
        draw_world: bool,
        debug_faces: bool,
        debug_chunks: bool,
        _draw_radius: i32,
        selected_hotbar_slot: u8,
        hotbar_slots: &[Option<ItemStack>; HOTBAR_SLOT_COUNT],
        inventory_open: bool,
        storage_slots: &[Option<ItemStack>; INVENTORY_STORAGE_SLOTS],
        craft_grid_side: usize,
        craft_input_slots: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
        craft_output: Option<ItemStack>,
        hovered_slot: Option<InventorySlotRef>,
        cursor_pos: Option<(f32, f32)>,
        dragged_item: Option<ItemStack>,
        break_overlay: Option<(IVec3, u32)>,
        chat_open: bool,
        chat_visible: bool,
        keybind_overlay_visible: bool,
        keybind_overlay_lines: &[&str],
        stats_overlay_visible: bool,
        stats_overlay_lines: &[String],
        chat_input: &str,
        chat_lines: &[String],
        dropped_items: &[DroppedItemRender],
        pause_menu_open: bool,
        pause_menu_cursor: Option<(f32, f32)>,
    ) {
        self.cell.with_dependent_mut(|_, gpu| {
            apply_pending_uploads(gpu, 8);
            let mvp = build_mvp(gpu.config.width, gpu.config.height, camera);
            let day_cycle = sample_day_cycle(elapsed_seconds);
            let sun_dir = sun_direction(day_cycle.day_progress, day_cycle.sun_height);
            let sun_strength = day_cycle.daylight.clamp(0.0, 1.0);
            let culled_lights =
                build_culled_point_lights(camera, day_cycle, dropped_items, break_overlay);

            let uniform = SceneUniform {
                mvp: mvp.to_cols_array_2d(),
                camera_pos: [camera.position.x, camera.position.y, camera.position.z, 0.0],
                tile_misc: [
                    gpu.tile_uv_size[0],
                    gpu.tile_uv_size[1],
                    crate::world::CHUNK_SIZE as f32,
                    0.0015,
                ],
                flags0: [
                    self.style.use_texture as u32,
                    gpu.tiles_x,
                    debug_faces as u32,
                    0,
                ],
                flags1: [0, 2, 3, 6],
                colormap_misc: [0.72, day_cycle.daylight, 0.0, 0.0],
                item_misc: [
                    gpu.ui_item_tile_uv_size[0],
                    gpu.ui_item_tile_uv_size[1],
                    gpu.ui_item_tiles_x as f32,
                    0.0,
                ],
                light_misc: [culled_lights.count, 0, 0, 0],
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, sun_strength],
                point_light_pos_radius: culled_lights.pos_radius,
                point_light_color_intensity: culled_lights.color_intensity,
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
            let mut ui_vertices = build_inventory_ui_vertices(
                gpu.config.width,
                gpu.config.height,
                elapsed_seconds,
                selected_hotbar_slot,
                hotbar_slots,
                inventory_open,
                storage_slots,
                craft_grid_side,
                craft_input_slots,
                craft_output,
                hovered_slot,
                cursor_pos,
                dragged_item,
                gpu.tiles_x,
                gpu.tile_uv_size,
                gpu.ui_item_tiles_x,
                gpu.ui_item_tile_uv_size,
            );
            push_chat_overlay(
                &mut ui_vertices,
                gpu.config.width,
                gpu.config.height,
                chat_open,
                chat_visible,
                chat_input,
                chat_lines,
            );
            push_stats_overlay(
                &mut ui_vertices,
                gpu.config.width,
                gpu.config.height,
                stats_overlay_visible,
                stats_overlay_lines,
            );
            push_keybind_overlay(
                &mut ui_vertices,
                gpu.config.width,
                gpu.config.height,
                keybind_overlay_visible,
                keybind_overlay_lines,
            );
            push_pause_menu_overlay(
                &mut ui_vertices,
                gpu.config.width,
                gpu.config.height,
                pause_menu_open,
                pause_menu_cursor,
            );
            if !ui_vertices.is_empty() {
                ensure_ui_vertex_capacity(gpu, ui_vertices.len());
                gpu.queue.write_buffer(
                    &gpu.ui_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&ui_vertices),
                );
            }

            if draw_world {
                let (drop_vertices, drop_indices) = build_dropped_item_mesh(dropped_items);
                upload_dynamic_chunk_mesh(
                    &gpu.device,
                    &gpu.queue,
                    &mut gpu.dropped_mesh,
                    &drop_vertices,
                    &drop_indices,
                    "dropped_item_vertex_buffer",
                    "dropped_item_index_buffer",
                );

                if let Some((coord, stage)) = break_overlay {
                    let (vertices, indices) = build_break_overlay_mesh(coord, stage);
                    upload_dynamic_chunk_mesh(
                        &gpu.device,
                        &gpu.queue,
                        &mut gpu.break_overlay_mesh,
                        &vertices,
                        &indices,
                        "break_overlay_vertex_buffer",
                        "break_overlay_index_buffer",
                    );
                } else {
                    gpu.break_overlay_mesh.index_count = 0;
                }

                let (sun_vertices, sun_indices) =
                    build_sun_mesh(camera, day_cycle.day_progress, day_cycle.sun_height);
                upload_dynamic_chunk_mesh(
                    &gpu.device,
                    &gpu.queue,
                    &mut gpu.sun_mesh,
                    &sun_vertices,
                    &sun_indices,
                    "sun_vertex_buffer",
                    "sun_index_buffer",
                );
            } else {
                gpu.dropped_mesh.index_count = 0;
                gpu.break_overlay_mesh.index_count = 0;
                gpu.sun_mesh.index_count = 0;
            }

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("cube_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02 + (0.52 - 0.02) * day_cycle.sky_mix as f64,
                                g: 0.04 + (0.73 - 0.04) * day_cycle.sky_mix as f64,
                                b: 0.09 + (0.95 - 0.09) * day_cycle.sky_mix as f64,
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

                if draw_world {
                    rpass.set_bind_group(0, &gpu.bind_group, &[]);
                    rpass.set_pipeline(&gpu.render_pipeline);
                    for coord in &gpu.visible_supers {
                        let Some(chunk) = gpu.super_chunks.get(coord) else {
                            continue;
                        };
                        if chunk.raw_index_count == 0 {
                            continue;
                        }
                        let (Some(vertex_buffer), Some(index_buffer)) = (
                            chunk.raw_vertex_buffer.as_ref(),
                            chunk.raw_index_buffer.as_ref(),
                        ) else {
                            continue;
                        };
                        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..chunk.raw_index_count, 0, 0..1);
                    }

                    rpass.set_pipeline(&gpu.packed_render_pipeline);
                    for coord in &gpu.visible_supers {
                        let Some(chunk) = gpu.super_chunks.get(coord) else {
                            continue;
                        };
                        if chunk.packed_index_count == 0 {
                            continue;
                        }
                        let (Some(vertex_buffer), Some(index_buffer)) = (
                            chunk.packed_vertex_buffer.as_ref(),
                            chunk.packed_index_buffer.as_ref(),
                        ) else {
                            continue;
                        };
                        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..chunk.packed_index_count, 0, 0..1);
                    }

                    if gpu.dropped_mesh.index_count > 0
                        && let (Some(vertex_buffer), Some(index_buffer)) = (
                            gpu.dropped_mesh.vertex_buffer.as_ref(),
                            gpu.dropped_mesh.index_buffer.as_ref(),
                        )
                    {
                        rpass.set_pipeline(&gpu.render_pipeline);
                        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..gpu.dropped_mesh.index_count, 0, 0..1);
                    }
                    if gpu.break_overlay_mesh.index_count > 0
                        && let (Some(vertex_buffer), Some(index_buffer)) = (
                            gpu.break_overlay_mesh.vertex_buffer.as_ref(),
                            gpu.break_overlay_mesh.index_buffer.as_ref(),
                        )
                    {
                        rpass.set_pipeline(&gpu.render_pipeline);
                        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..gpu.break_overlay_mesh.index_count, 0, 0..1);
                    }
                    if gpu.sun_mesh.index_count > 0
                        && let (Some(vertex_buffer), Some(index_buffer)) = (
                            gpu.sun_mesh.vertex_buffer.as_ref(),
                            gpu.sun_mesh.index_buffer.as_ref(),
                        )
                    {
                        rpass.set_pipeline(&gpu.render_pipeline);
                        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..gpu.sun_mesh.index_count, 0, 0..1);
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

                if !ui_vertices.is_empty() {
                    rpass.set_pipeline(&gpu.ui_pipeline);
                    rpass.set_bind_group(0, &gpu.ui_bind_group, &[]);
                    rpass.set_vertex_buffer(0, gpu.ui_vertex_buffer.slice(..));
                    rpass.draw(0..ui_vertices.len() as u32, 0..1);
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

    pub fn adapter_summary(&self) -> String {
        self.cell
            .with_dependent(|_, gpu| gpu.adapter_summary.clone())
    }

    pub fn stats(&self) -> GpuStats {
        self.cell.with_dependent(|_, gpu| {
            let mut total_indices = 0u64;
            let mut total_draw_calls_est = 0u64;
            let mut total_raw_vertices_capacity = 0u64;
            let mut total_packed_vertices_capacity = 0u64;
            for chunk in gpu.super_chunks.values() {
                total_indices += chunk.raw_index_count as u64 + chunk.packed_index_count as u64;
                if chunk.raw_index_count > 0 {
                    total_draw_calls_est += 1;
                }
                if chunk.packed_index_count > 0 {
                    total_draw_calls_est += 1;
                }
                total_raw_vertices_capacity += chunk.raw_vertex_capacity as u64;
                total_packed_vertices_capacity += chunk.packed_vertex_capacity as u64;
            }
            let mut visible_raw_indices = 0u64;
            let mut visible_packed_indices = 0u64;
            let mut visible_draw_calls_est = 0u64;
            for coord in &gpu.visible_supers {
                let Some(chunk) = gpu.super_chunks.get(coord) else {
                    continue;
                };
                if chunk.raw_index_count > 0 {
                    visible_raw_indices += chunk.raw_index_count as u64;
                    visible_draw_calls_est += 1;
                }
                if chunk.packed_index_count > 0 {
                    visible_packed_indices += chunk.packed_index_count as u64;
                    visible_draw_calls_est += 1;
                }
            }
            GpuStats {
                super_chunks: gpu.super_chunks.len(),
                dirty_supers: gpu.dirty_supers.len(),
                visible_supers: gpu.visible_supers.len(),
                pending_updates: gpu.pending_updates.len(),
                pending_queue: gpu.pending_queue.len(),
                total_indices,
                visible_indices: visible_raw_indices + visible_packed_indices,
                visible_raw_indices,
                visible_packed_indices,
                total_draw_calls_est,
                visible_draw_calls_est,
                total_raw_vertices_capacity,
                total_packed_vertices_capacity,
            }
        })
    }

    pub fn chunk_memory_bytes(&self, coord: IVec3) -> Option<u64> {
        self.cell.with_dependent(|_, gpu| {
            let super_coord = super_chunk_coord(coord);
            let super_chunk = gpu.super_chunks.get(&super_coord)?;
            if let Some(slot) = super_chunk.raw_slots.get(&coord) {
                return Some(raw_slot_memory_bytes(slot));
            }
            if let Some(slot) = super_chunk.packed_slots.get(&coord) {
                return Some(packed_slot_memory_bytes(slot));
            }
            None
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
        let vertices: Arc<[ChunkVertex]> = vertices.into();
        let indices: Arc<[u32]> = indices.into();
        self.cell.with_dependent_mut(|_, gpu| {
            upsert_chunk_impl(gpu, coord, ChunkVertices::Raw(vertices), indices);
        });
    }

    pub fn upsert_chunk_packed(
        &mut self,
        coord: IVec3,
        _center: Vec3,
        _radius: f32,
        vertices: Arc<[PackedFarVertex]>,
        indices: Arc<[u32]>,
    ) {
        self.cell.with_dependent_mut(|_, gpu| {
            upsert_chunk_impl(gpu, coord, ChunkVertices::PackedFar(vertices), indices);
        });
    }

    pub fn remove_chunk(&mut self, coord: IVec3) {
        self.cell.with_dependent_mut(|_, gpu| {
            gpu.chunks.remove(&coord);
            let super_coord = super_chunk_coord(coord);
            let slot = gpu.super_chunks.get(&super_coord).and_then(|super_chunk| {
                super_chunk
                    .raw_slots
                    .get(&coord)
                    .copied()
                    .map(|slot| (false, slot))
                    .or_else(|| {
                        super_chunk
                            .packed_slots
                            .get(&coord)
                            .copied()
                            .map(|slot| (true, slot))
                    })
            });
            if let Some((is_packed, slot)) = slot {
                if let Some(super_chunk) = gpu.super_chunks.get_mut(&super_coord) {
                    if is_packed {
                        if let Some(index_buffer) = super_chunk.packed_index_buffer.as_ref() {
                            clear_chunk_slot(
                                &gpu.queue,
                                index_buffer,
                                &mut super_chunk.packed_slots,
                                coord,
                                &slot,
                            );
                        } else {
                            mark_super_dirty(gpu, super_coord);
                        }
                    } else if let Some(index_buffer) = super_chunk.raw_index_buffer.as_ref() {
                        clear_chunk_slot(
                            &gpu.queue,
                            index_buffer,
                            &mut super_chunk.raw_slots,
                            coord,
                            &slot,
                        );
                    } else {
                        mark_super_dirty(gpu, super_coord);
                    }
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
            if gpu.dirty_supers.is_empty() || budget == 0 {
                return;
            }
            let len = gpu.dirty_supers.len();
            let take = budget.min(len);
            let mut rebuild_now: Vec<IVec3> = if take >= len {
                std::mem::take(&mut gpu.dirty_supers)
            } else {
                let split = len - take;
                // For large queues, skip nth-selection work and just drain a small tail slice.
                if len <= 256 {
                    // Keep far supers in-place and extract only the nearest slice this tick.
                    gpu.dirty_supers.select_nth_unstable_by(split, |a, b| {
                        let da = (super_chunk_center(*a) - camera_pos).length_squared();
                        let db = (super_chunk_center(*b) - camera_pos).length_squared();
                        db.total_cmp(&da)
                    });
                }
                gpu.dirty_supers.split_off(split)
            };
            for coord in rebuild_now.drain(..) {
                gpu.dirty_set.remove(&coord);
                rebuild_superchunk(gpu, coord);
            }
        });
    }

    pub fn update_visible(&mut self, camera: &Camera, draw_radius: i32) {
        self.cell.with_dependent_mut(|_, gpu| {
            let draw_radius = draw_radius as f32 * crate::world::CHUNK_SIZE as f32;
            let draw_radius_sq = draw_radius * draw_radius;
            let camera_forward = if camera.forward.length_squared() > 1.0e-8 {
                camera.forward.normalize()
            } else {
                Vec3::new(0.0, 0.0, -1.0)
            };
            gpu.visible_supers.clear();
            gpu.visible_supers.reserve(gpu.super_chunks.len());
            for (coord, chunk) in &gpu.super_chunks {
                let to_center = chunk.center - camera.position;
                if to_center.length_squared() > draw_radius_sq {
                    continue;
                }

                // Cheap horizon/underground cull for far terrain columns.
                let horiz_dist = Vec2::new(to_center.x, to_center.z).length();
                if horiz_dist > draw_radius * 0.35 && to_center.y < -(chunk.radius * 1.8) {
                    continue;
                }

                if !chunk_visible(camera.position, camera_forward, chunk) {
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

fn upsert_chunk_impl(
    gpu: &mut GpuInner,
    coord: IVec3,
    vertices: ChunkVertices,
    indices: Arc<[u32]>,
) {
    gpu.chunks.insert(
        coord,
        CpuChunkMesh {
            vertices: vertices.clone(),
            indices: Arc::clone(&indices),
        },
    );
    if !schedule_superchunk_update(gpu, coord, vertices, indices) {
        mark_super_dirty(gpu, super_chunk_coord(coord));
    }
}

fn rebuild_superchunk(gpu: &mut GpuInner, coord: IVec3) {
    let mut raw_vertices: Vec<ChunkVertex> = Vec::new();
    let mut raw_indices: Vec<u32> = Vec::new();
    let mut raw_slots: HashMap<IVec3, ChunkSlot> = HashMap::new();
    let mut raw_vertex_cursor = 0u32;
    let mut raw_index_cursor = 0u32;

    let mut packed_vertices: Vec<PackedFarVertex> = Vec::new();
    let mut packed_indices: Vec<u32> = Vec::new();
    let mut packed_slots: HashMap<IVec3, ChunkSlot> = HashMap::new();
    let mut packed_vertex_cursor = 0u32;
    let mut packed_index_cursor = 0u32;

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
                let chunk_vertex_len = match &chunk.vertices {
                    ChunkVertices::Raw(vertices) => vertices.len(),
                    ChunkVertices::PackedFar(vertices) => vertices.len(),
                };
                if chunk_vertex_len == 0 || chunk.indices.is_empty() {
                    continue;
                }
                let vertex_capacity = next_pow2_u32(chunk_vertex_len.max(1) as u32);
                let mut index_capacity = next_pow2_u32(chunk.indices.len().max(3) as u32);
                if !index_capacity.is_multiple_of(3) {
                    index_capacity += 3 - (index_capacity % 3);
                }
                match &chunk.vertices {
                    ChunkVertices::Raw(raw) => {
                        let slot = ChunkSlot {
                            vertex_offset: raw_vertex_cursor,
                            vertex_capacity,
                            index_offset: raw_index_cursor,
                            index_capacity,
                        };
                        raw_slots.insert(c, slot);

                        raw_vertices.extend_from_slice(raw);
                        raw_vertices.resize(
                            (raw_vertex_cursor + vertex_capacity) as usize,
                            ChunkVertex::zeroed(),
                        );

                        raw_indices.extend(chunk.indices.iter().map(|i| i + raw_vertex_cursor));
                        let pad = index_capacity.saturating_sub(chunk.indices.len() as u32);
                        if pad > 0 {
                            raw_indices
                                .extend(std::iter::repeat_n(raw_vertex_cursor, pad as usize));
                        }
                        raw_vertex_cursor += vertex_capacity;
                        raw_index_cursor += index_capacity;
                    }
                    ChunkVertices::PackedFar(packed) => {
                        let slot = ChunkSlot {
                            vertex_offset: packed_vertex_cursor,
                            vertex_capacity,
                            index_offset: packed_index_cursor,
                            index_capacity,
                        };
                        packed_slots.insert(c, slot);

                        packed_vertices.extend_from_slice(packed);
                        packed_vertices.resize(
                            (packed_vertex_cursor + vertex_capacity) as usize,
                            PackedFarVertex::zeroed(),
                        );

                        packed_indices
                            .extend(chunk.indices.iter().map(|i| i + packed_vertex_cursor));
                        let pad = index_capacity.saturating_sub(chunk.indices.len() as u32);
                        if pad > 0 {
                            packed_indices
                                .extend(std::iter::repeat_n(packed_vertex_cursor, pad as usize));
                        }
                        packed_vertex_cursor += vertex_capacity;
                        packed_index_cursor += index_capacity;
                    }
                }
            }
        }
    }

    if raw_indices.is_empty() && packed_indices.is_empty() {
        gpu.super_chunks.remove(&coord);
        return;
    }

    let raw_vertex_buffer = if raw_vertices.is_empty() {
        None
    } else {
        Some(
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("super_chunk_vertex_buffer_raw"),
                    contents: bytemuck::cast_slice(&raw_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
        )
    };
    let raw_index_buffer = if raw_indices.is_empty() {
        None
    } else {
        Some(
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("super_chunk_index_buffer_raw"),
                    contents: bytemuck::cast_slice(&raw_indices),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                }),
        )
    };

    let packed_vertex_buffer = if packed_vertices.is_empty() {
        None
    } else {
        Some(
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("super_chunk_vertex_buffer_packed"),
                    contents: bytemuck::cast_slice(&packed_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
        )
    };
    let packed_index_buffer = if packed_indices.is_empty() {
        None
    } else {
        Some(
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("super_chunk_index_buffer_packed"),
                    contents: bytemuck::cast_slice(&packed_indices),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                }),
        )
    };

    let size = SUPER_CHUNK_SIZE as f32 * crate::world::CHUNK_SIZE as f32;
    let half = size * 0.5;
    let center = super_chunk_center(coord);
    let radius = half * (3.0f32).sqrt();

    let line_vertices = super_chunk_wireframe_vertices(coord);
    let line_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("super_chunk_line_buffer"),
            contents: bytemuck::cast_slice(&line_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

    gpu.super_chunks.insert(
        coord,
        SuperChunkGpuMesh {
            raw_vertex_buffer,
            raw_index_buffer,
            raw_index_count: raw_indices.len() as u32,
            packed_vertex_buffer,
            packed_index_buffer,
            packed_index_count: packed_indices.len() as u32,
            center,
            radius,
            line_buffer,
            line_count: line_vertices.len() as u32,
            raw_slots,
            packed_slots,
            raw_vertex_capacity: raw_vertex_cursor,
            packed_vertex_capacity: packed_vertex_cursor,
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
        p000, p001, p000, p010, p000, p100, p111, p110, p111, p101, p111, p011, p001, p011, p001,
        p101, p010, p011, p010, p110, p100, p101, p100, p110,
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

fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
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

fn ensure_dynamic_chunk_mesh_capacity(
    device: &wgpu::Device,
    mesh: &mut DynamicMeshBuffer,
    required_vertices: usize,
    required_indices: usize,
    vertex_label: &'static str,
    index_label: &'static str,
) {
    if required_vertices > mesh.vertex_capacity || mesh.vertex_buffer.is_none() {
        let mut new_capacity = mesh.vertex_capacity.max(1);
        while new_capacity < required_vertices {
            new_capacity = new_capacity.saturating_mul(2);
        }
        mesh.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(vertex_label),
            size: (new_capacity * std::mem::size_of::<ChunkVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        mesh.vertex_capacity = new_capacity;
    }
    if required_indices > mesh.index_capacity || mesh.index_buffer.is_none() {
        let mut new_capacity = mesh.index_capacity.max(1);
        while new_capacity < required_indices {
            new_capacity = new_capacity.saturating_mul(2);
        }
        mesh.index_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(index_label),
            size: (new_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        mesh.index_capacity = new_capacity;
    }
}

fn upload_dynamic_chunk_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &mut DynamicMeshBuffer,
    vertices: &[ChunkVertex],
    indices: &[u32],
    vertex_label: &'static str,
    index_label: &'static str,
) {
    if vertices.is_empty() || indices.is_empty() {
        mesh.index_count = 0;
        return;
    }

    ensure_dynamic_chunk_mesh_capacity(
        device,
        mesh,
        vertices.len(),
        indices.len(),
        vertex_label,
        index_label,
    );

    if let (Some(vertex_buffer), Some(index_buffer)) =
        (mesh.vertex_buffer.as_ref(), mesh.index_buffer.as_ref())
    {
        queue.write_buffer(vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(index_buffer, 0, bytemuck::cast_slice(indices));
        mesh.index_count = indices.len() as u32;
    } else {
        mesh.index_count = 0;
    }
}

fn ensure_ui_vertex_capacity(gpu: &mut GpuInner, required_vertices: usize) {
    if required_vertices <= gpu.ui_vertex_capacity {
        return;
    }
    let mut new_capacity = gpu.ui_vertex_capacity.max(1);
    while new_capacity < required_vertices {
        new_capacity = new_capacity.saturating_mul(2);
    }
    gpu.ui_vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui_vertex_buffer"),
        size: (new_capacity * std::mem::size_of::<UiVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.ui_vertex_capacity = new_capacity;
}

#[allow(clippy::too_many_arguments)]
fn build_inventory_ui_vertices(
    width: u32,
    height: u32,
    elapsed_seconds: f32,
    selected_hotbar_slot: u8,
    hotbar_slots: &[Option<ItemStack>; HOTBAR_SLOT_COUNT],
    inventory_open: bool,
    storage_slots: &[Option<ItemStack>; INVENTORY_STORAGE_SLOTS],
    craft_grid_side: usize,
    craft_input_slots: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_output: Option<ItemStack>,
    hovered_slot: Option<InventorySlotRef>,
    cursor_pos: Option<(f32, f32)>,
    dragged_item: Option<ItemStack>,
    block_tiles_x: u32,
    block_tile_uv_size: [f32; 2],
    item_tiles_x: u32,
    item_tile_uv_size: [f32; 2],
) -> Vec<UiVertex> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let width_f = width as f32;
    let height_f = height as f32;
    let craft_grid_side = craft_grid_side.clamp(1, CRAFT_GRID_SIDE);
    let layout = compute_inventory_layout(width, height, craft_grid_side);
    let mut vertices = Vec::with_capacity(4096);
    let selected = (selected_hotbar_slot as usize).min(HOTBAR_SLOT_COUNT - 1);

    if inventory_open {
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            0.0,
            0.0,
            width_f,
            height_f,
            [0.0, 0.0, 0.0, 0.35],
        );
        let panel = layout.panel;
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            panel.x,
            panel.y,
            panel.w,
            panel.h,
            [0.08, 0.09, 0.11, 0.88],
        );
        let panel_border = (layout.slot * 0.08).clamp(2.0, 4.0);
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            panel.x - panel_border,
            panel.y - panel_border,
            panel.w + panel_border * 2.0,
            panel_border,
            [0.78, 0.80, 0.86, 0.24],
        );
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            panel.x - panel_border,
            panel.y + panel.h,
            panel.w + panel_border * 2.0,
            panel_border,
            [0.78, 0.80, 0.86, 0.24],
        );
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            panel.x - panel_border,
            panel.y,
            panel_border,
            panel.h,
            [0.78, 0.80, 0.86, 0.24],
        );
        push_ui_rect(
            &mut vertices,
            width_f,
            height_f,
            panel.x + panel.w,
            panel.y,
            panel_border,
            panel.h,
            [0.78, 0.80, 0.86, 0.24],
        );
    }

    for i in 0..HOTBAR_SLOT_COUNT {
        let rect = hotbar_slot_rect(&layout, i);
        let hovered = hovered_slot == Some(InventorySlotRef::Hotbar(i as u8));
        push_slot_box(
            &mut vertices,
            width_f,
            height_f,
            rect,
            i == selected,
            hovered,
        );
        if let Some(stack) = hotbar_slots.get(i).copied().flatten() {
            let icon_center_x = rect.x + rect.w * 0.5;
            let icon_center_y = rect.y + rect.h * 0.5;
            let icon_size = rect.w * 0.54;
            push_inventory_icon(
                &mut vertices,
                width_f,
                height_f,
                icon_center_x,
                icon_center_y,
                icon_size,
                stack.block_id,
                i == selected,
                rect.w,
                block_tiles_x,
                block_tile_uv_size,
                item_tiles_x,
                item_tile_uv_size,
            );
            push_stack_meter(
                &mut vertices,
                width_f,
                height_f,
                rect,
                stack.block_id,
                stack.count,
            );
            if stack.count > 1 {
                let count_text = stack.count.to_string();
                let pixel = (rect.w * 0.06).clamp(1.0, 2.4);
                let text_w = text_width_3x5(&count_text, pixel);
                let text_x = rect.x + rect.w - text_w - pixel * 1.3;
                let text_y = rect.y + rect.h - pixel * 6.0;
                push_text_3x5_shadow(
                    &mut vertices,
                    width_f,
                    height_f,
                    text_x,
                    text_y,
                    pixel,
                    &count_text,
                    [0.96, 0.96, 0.96, 0.95],
                );
            }
        }
    }

    if let Some(selected_stack) = hotbar_slots.get(selected).copied().flatten() {
        if let Some(max_durability) = item_max_durability(selected_stack.block_id) {
            let cur_durability = selected_stack.durability.max(1).min(max_durability.max(1));
            let text = format!("{}/{}", cur_durability, max_durability.max(1));
            let pixel = (layout.slot * 0.072).clamp(1.2, 2.8);
            let text_w = text_width_3x5(&text, pixel);
            let text_x = (width_f - text_w) * 0.5;
            let text_y = (layout.hotbar_y - pixel * 7.0).max(4.0);
            push_text_3x5_shadow(
                &mut vertices,
                width_f,
                height_f,
                text_x,
                text_y,
                pixel,
                &text,
                [0.90, 0.94, 0.98, 0.96],
            );
        }
    }

    if !inventory_open {
        push_held_item_viewmodel(
            &mut vertices,
            width_f,
            height_f,
            elapsed_seconds,
            hotbar_slots
                .get(selected)
                .copied()
                .flatten()
                .filter(|stack| stack.count > 0),
            block_tiles_x,
            block_tile_uv_size,
            item_tiles_x,
            item_tile_uv_size,
        );
    }

    if inventory_open {
        for i in 0..CRAFT_GRID_SLOTS {
            let row = i / CRAFT_GRID_SIDE;
            let col = i % CRAFT_GRID_SIDE;
            if row >= craft_grid_side || col >= craft_grid_side {
                continue;
            }
            let rect = craft_input_slot_rect(&layout, i);
            let hovered = hovered_slot == Some(InventorySlotRef::CraftInput(i as u8));
            push_slot_box(&mut vertices, width_f, height_f, rect, false, hovered);
            if let Some(stack) = craft_input_slots.get(i).copied().flatten() {
                let icon_center_x = rect.x + rect.w * 0.5;
                let icon_center_y = rect.y + rect.h * 0.5;
                let icon_size = rect.w * 0.54;
                push_inventory_icon(
                    &mut vertices,
                    width_f,
                    height_f,
                    icon_center_x,
                    icon_center_y,
                    icon_size,
                    stack.block_id,
                    false,
                    rect.w,
                    block_tiles_x,
                    block_tile_uv_size,
                    item_tiles_x,
                    item_tile_uv_size,
                );
                push_stack_meter(
                    &mut vertices,
                    width_f,
                    height_f,
                    rect,
                    stack.block_id,
                    stack.count,
                );
            }
        }

        let output_rect = craft_output_slot_rect(&layout);
        let output_hovered = hovered_slot == Some(InventorySlotRef::CraftOutput);
        push_slot_box(
            &mut vertices,
            width_f,
            height_f,
            output_rect,
            craft_output.is_some(),
            output_hovered,
        );
        if let Some(stack) = craft_output {
            let icon_center_x = output_rect.x + output_rect.w * 0.5;
            let icon_center_y = output_rect.y + output_rect.h * 0.5;
            let icon_size = output_rect.w * 0.54;
            push_inventory_icon(
                &mut vertices,
                width_f,
                height_f,
                icon_center_x,
                icon_center_y,
                icon_size,
                stack.block_id,
                output_hovered,
                output_rect.w,
                block_tiles_x,
                block_tile_uv_size,
                item_tiles_x,
                item_tile_uv_size,
            );
            push_stack_meter(
                &mut vertices,
                width_f,
                height_f,
                output_rect,
                stack.block_id,
                stack.count,
            );
        }

        let grid_right = layout.craft_input_start_x
            + craft_grid_side as f32 * (layout.slot + layout.gap)
            - layout.gap;
        let connector_start_x = grid_right + (output_rect.x - grid_right) * 0.24;
        let connector_end_x = output_rect.x - (output_rect.x - grid_right) * 0.20;
        let connector_y = output_rect.y + output_rect.h * 0.5;
        let connector_h = (layout.slot * 0.11).clamp(2.0, 4.0);
        let connector_color = if craft_output.is_some() {
            [0.86, 0.90, 0.64, 0.72]
        } else {
            [0.55, 0.58, 0.62, 0.38]
        };
        if connector_end_x > connector_start_x {
            push_ui_rect(
                &mut vertices,
                width_f,
                height_f,
                connector_start_x,
                connector_y - connector_h * 0.5,
                connector_end_x - connector_start_x,
                connector_h,
                connector_color,
            );
            let head_w = (layout.slot * 0.18).clamp(4.0, 10.0);
            push_ui_rect(
                &mut vertices,
                width_f,
                height_f,
                connector_end_x - head_w,
                connector_y - connector_h,
                head_w,
                connector_h,
                connector_color,
            );
            push_ui_rect(
                &mut vertices,
                width_f,
                height_f,
                connector_end_x - head_w,
                connector_y,
                head_w,
                connector_h,
                connector_color,
            );
        }

        for i in 0..INVENTORY_STORAGE_SLOTS {
            let rect = storage_slot_rect(&layout, i);
            let hovered = hovered_slot == Some(InventorySlotRef::Storage(i as u8));
            push_slot_box(&mut vertices, width_f, height_f, rect, false, hovered);
            if let Some(stack) = storage_slots.get(i).copied().flatten() {
                let icon_center_x = rect.x + rect.w * 0.5;
                let icon_center_y = rect.y + rect.h * 0.5;
                let icon_size = rect.w * 0.54;
                push_inventory_icon(
                    &mut vertices,
                    width_f,
                    height_f,
                    icon_center_x,
                    icon_center_y,
                    icon_size,
                    stack.block_id,
                    false,
                    rect.w,
                    block_tiles_x,
                    block_tile_uv_size,
                    item_tiles_x,
                    item_tile_uv_size,
                );
                push_stack_meter(
                    &mut vertices,
                    width_f,
                    height_f,
                    rect,
                    stack.block_id,
                    stack.count,
                );
            }
        }

        if let (Some(stack), Some((mx, my))) = (dragged_item, cursor_pos) {
            push_inventory_icon(
                &mut vertices,
                width_f,
                height_f,
                mx,
                my,
                layout.slot * 0.62,
                stack.block_id,
                true,
                layout.slot,
                block_tiles_x,
                block_tile_uv_size,
                item_tiles_x,
                item_tile_uv_size,
            );
        }
    }

    vertices
}

fn clipped_text(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    text.chars().take(max_chars).collect()
}

fn push_chat_overlay(
    out: &mut Vec<UiVertex>,
    width: u32,
    height: u32,
    chat_open: bool,
    chat_visible: bool,
    chat_input: &str,
    chat_lines: &[String],
) {
    if width == 0 || height == 0 {
        return;
    }
    if !chat_open && !chat_visible {
        return;
    }
    if !chat_open && chat_lines.is_empty() {
        return;
    }

    let width_f = width as f32;
    let height_f = height as f32;
    let pixel = (width_f.min(height_f) * 0.0032).clamp(1.5, 2.8);
    let line_h = pixel * 6.0;
    let panel_w = (width_f * 0.50).clamp(260.0, 620.0);
    let panel_x = (width_f * 0.018).clamp(8.0, 20.0);
    let panel_bottom_margin = (height_f * 0.16).clamp(78.0, 132.0);
    let pad = (pixel * 2.3).clamp(4.0, 9.0);
    let max_visible_lines = if chat_open { 8 } else { 5 };
    let shown_count = chat_lines.len().min(max_visible_lines);
    let line_count = shown_count + usize::from(chat_open);
    if line_count == 0 {
        return;
    }
    let panel_h =
        line_count as f32 * line_h + pad * 2.0 + if chat_open { pixel * 1.6 } else { 0.0 };
    let panel_y = (height_f - panel_h - panel_bottom_margin).max(8.0);
    let panel_alpha = if chat_open { 0.70 } else { 0.44 };
    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        [0.03, 0.04, 0.05, panel_alpha],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        (pixel * 0.9).clamp(1.0, 3.0),
        [0.80, 0.86, 0.92, 0.28],
    );

    let max_chars = (((panel_w - pad * 2.0) / (pixel * 4.0)).floor() as usize).max(1);
    let start_index = chat_lines.len().saturating_sub(shown_count);
    for (row, line) in chat_lines[start_index..].iter().enumerate() {
        let visible = clipped_text(line, max_chars);
        let y = panel_y + pad + row as f32 * line_h;
        push_text_3x5_shadow(
            out,
            width_f,
            height_f,
            panel_x + pad,
            y,
            pixel,
            &visible,
            [0.92, 0.94, 0.98, 0.96],
        );
    }

    if chat_open {
        let input_bg_y = panel_y + pad + shown_count as f32 * line_h + pixel * 0.4;
        push_ui_rect(
            out,
            width_f,
            height_f,
            panel_x + pad * 0.55,
            input_bg_y - pixel * 0.7,
            panel_w - pad * 1.1,
            line_h + pixel * 0.9,
            [0.0, 0.0, 0.0, 0.38],
        );
        let available_chars = max_chars.saturating_sub(2).max(1);
        let mut input_visible = clipped_text(chat_input, available_chars);
        if input_visible.is_empty() {
            input_visible.push(' ');
        }
        let input_text = format!("> {input_visible}_");
        push_text_3x5_shadow(
            out,
            width_f,
            height_f,
            panel_x + pad,
            input_bg_y,
            pixel,
            &input_text,
            [0.97, 0.98, 0.99, 0.98],
        );
    }
}

fn push_stats_overlay(
    out: &mut Vec<UiVertex>,
    width: u32,
    height: u32,
    visible: bool,
    lines: &[String],
) {
    if !visible || width == 0 || height == 0 || lines.is_empty() {
        return;
    }

    let width_f = width as f32;
    let height_f = height as f32;
    let pixel = (width_f.min(height_f) * 0.0029).clamp(1.3, 2.2);
    let line_h = pixel * 6.0;
    let panel_x = 8.0;
    let panel_y = 8.0;
    let panel_w = (width_f * 0.62).clamp(300.0, 900.0);
    let max_lines = (((height_f * 0.86) / line_h).floor() as usize).max(3);
    let shown_count = lines.len().min(max_lines);
    if shown_count == 0 {
        return;
    }
    let pad = (pixel * 2.1).clamp(3.0, 8.0);
    let panel_h = shown_count as f32 * line_h + pad * 2.0;

    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        [0.01, 0.01, 0.01, 0.62],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        (pixel * 0.9).clamp(1.0, 2.2),
        [0.76, 0.87, 0.96, 0.30],
    );

    let max_chars = (((panel_w - pad * 2.0) / (pixel * 4.0)).floor() as usize).max(1);
    for (idx, line) in lines.iter().take(shown_count).enumerate() {
        let text = clipped_text(line, max_chars);
        push_text_3x5_shadow(
            out,
            width_f,
            height_f,
            panel_x + pad,
            panel_y + pad + idx as f32 * line_h,
            pixel,
            &text,
            [0.92, 0.94, 0.98, 0.96],
        );
    }
}

fn push_keybind_overlay(
    out: &mut Vec<UiVertex>,
    width: u32,
    height: u32,
    visible: bool,
    lines: &[&str],
) {
    if !visible || width == 0 || height == 0 || lines.is_empty() {
        return;
    }

    let width_f = width as f32;
    let height_f = height as f32;
    let pixel = (width_f.min(height_f) * 0.0030).clamp(1.4, 2.4);
    let line_h = pixel * 6.0;
    let panel_w = (width_f * 0.76).clamp(360.0, 1040.0);
    let panel_h = (lines.len() as f32 * line_h + pixel * 6.0).clamp(120.0, height_f * 0.88);
    let panel_x = (width_f - panel_w) * 0.5;
    let panel_y = (height_f - panel_h) * 0.16;
    let pad = (pixel * 2.5).clamp(4.0, 9.0);

    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        [0.02, 0.02, 0.03, 0.82],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        panel_x,
        panel_y,
        panel_w,
        (pixel * 1.2).clamp(1.0, 3.0),
        [0.84, 0.91, 0.98, 0.35],
    );

    let max_chars = (((panel_w - pad * 2.0) / (pixel * 4.0)).floor() as usize).max(1);
    let max_lines = (((panel_h - pad * 2.0) / line_h).floor() as usize).max(1);
    for (idx, line) in lines.iter().take(max_lines).enumerate() {
        let text = clipped_text(line, max_chars);
        let color = if idx == 0 {
            [0.98, 0.98, 0.84, 0.99]
        } else {
            [0.94, 0.95, 0.98, 0.97]
        };
        push_text_3x5_shadow(
            out,
            width_f,
            height_f,
            panel_x + pad,
            panel_y + pad + idx as f32 * line_h,
            pixel,
            &text,
            color,
        );
    }
}

fn push_pause_menu_overlay(
    out: &mut Vec<UiVertex>,
    width: u32,
    height: u32,
    visible: bool,
    cursor_pos: Option<(f32, f32)>,
) {
    if !visible {
        return;
    }
    let Some(layout) = compute_pause_menu_layout(width, height) else {
        return;
    };

    let width_f = width as f32;
    let height_f = height as f32;
    let pixel = (width_f.min(height_f) * 0.0033).clamp(1.8, 3.2);
    let hovered_button =
        cursor_pos.and_then(|(mx, my)| hit_test_pause_menu_button(width, height, mx, my));

    push_ui_rect(
        out,
        width_f,
        height_f,
        0.0,
        0.0,
        width_f,
        height_f,
        [0.0, 0.0, 0.0, 1.0],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        layout.panel.x,
        layout.panel.y,
        layout.panel.w,
        layout.panel.h,
        [0.10, 0.11, 0.13, 0.96],
    );
    let border = (layout.panel.w * 0.012).clamp(2.0, 6.0);
    push_ui_rect(
        out,
        width_f,
        height_f,
        layout.panel.x - border,
        layout.panel.y - border,
        layout.panel.w + border * 2.0,
        border,
        [0.82, 0.86, 0.92, 0.38],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        layout.panel.x - border,
        layout.panel.y + layout.panel.h,
        layout.panel.w + border * 2.0,
        border,
        [0.82, 0.86, 0.92, 0.38],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        layout.panel.x - border,
        layout.panel.y,
        border,
        layout.panel.h,
        [0.82, 0.86, 0.92, 0.38],
    );
    push_ui_rect(
        out,
        width_f,
        height_f,
        layout.panel.x + layout.panel.w,
        layout.panel.y,
        border,
        layout.panel.h,
        [0.82, 0.86, 0.92, 0.38],
    );

    let title = "paused";
    let title_w = text_width_3x5(title, pixel * 1.45);
    let title_x = layout.panel.x + (layout.panel.w - title_w) * 0.5;
    let title_y = layout.panel.y + (layout.panel.h * 0.17).max(14.0);
    push_text_3x5_shadow(
        out,
        width_f,
        height_f,
        title_x,
        title_y,
        pixel * 1.45,
        title,
        [0.97, 0.97, 0.97, 0.98],
    );

    let return_hovered = hovered_button == Some(PauseMenuButton::ReturnToGame);
    let quit_hovered = hovered_button == Some(PauseMenuButton::Quit);
    push_pause_menu_button(
        out,
        width_f,
        height_f,
        layout.return_button.x,
        layout.return_button.y,
        layout.return_button.w,
        layout.return_button.h,
        return_hovered,
        "return to game",
        pixel,
    );
    push_pause_menu_button(
        out,
        width_f,
        height_f,
        layout.quit_button.x,
        layout.quit_button.y,
        layout.quit_button.w,
        layout.quit_button.h,
        quit_hovered,
        "quit",
        pixel,
    );
}

fn push_pause_menu_button(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    hovered: bool,
    text: &str,
    pixel: f32,
) {
    let bg = if hovered {
        [0.30, 0.34, 0.40, 0.98]
    } else {
        [0.18, 0.20, 0.25, 0.96]
    };
    push_ui_rect(out, width, height, x, y, w, h, bg);
    let inset = (h * 0.11).clamp(2.0, 6.0);
    push_ui_rect(
        out,
        width,
        height,
        x + inset,
        y + inset,
        (w - inset * 2.0).max(1.0),
        (h - inset * 2.0).max(1.0),
        if hovered {
            [0.36, 0.40, 0.48, 0.98]
        } else {
            [0.24, 0.26, 0.32, 0.94]
        },
    );

    let border = (h * 0.07).clamp(1.0, 3.0);
    let border_color = if hovered {
        [0.88, 0.94, 1.0, 0.72]
    } else {
        [0.78, 0.84, 0.94, 0.45]
    };
    push_ui_rect(out, width, height, x, y, w, border, border_color);
    push_ui_rect(
        out,
        width,
        height,
        x,
        y + h - border,
        w,
        border,
        border_color,
    );
    push_ui_rect(out, width, height, x, y, border, h, border_color);
    push_ui_rect(
        out,
        width,
        height,
        x + w - border,
        y,
        border,
        h,
        border_color,
    );

    let text_pixel = if text.len() > 8 {
        (pixel * 0.84).clamp(1.2, 2.1)
    } else {
        (pixel * 0.92).clamp(1.2, 2.3)
    };
    let text_w = text_width_3x5(text, text_pixel);
    let text_x = x + (w - text_w) * 0.5;
    let text_y = y + (h - text_pixel * 5.0) * 0.5;
    push_text_3x5_shadow(
        out,
        width,
        height,
        text_x,
        text_y,
        text_pixel,
        text,
        [0.96, 0.97, 0.99, 0.98],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_held_item_viewmodel(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    elapsed_seconds: f32,
    held_stack: Option<ItemStack>,
    block_tiles_x: u32,
    block_tile_uv_size: [f32; 2],
    item_tiles_x: u32,
    item_tile_uv_size: [f32; 2],
) {
    let Some(held) = held_stack else {
        return;
    };
    if held.count == 0 {
        return;
    }

    let base_size = (width.min(height) * 0.34).clamp(92.0, 220.0);
    let sway_x = (elapsed_seconds * 1.9).sin() * (base_size * 0.03);
    let sway_y = (elapsed_seconds * 1.35).sin() * (base_size * 0.02);
    let center_x = width * 0.84 + sway_x;
    let center_y = height * 0.79 + sway_y;

    let panel_w = base_size * 0.95;
    let panel_h = base_size * 0.48;
    let panel_x = center_x - panel_w * 0.5;
    let panel_y = center_y - panel_h * 0.16;
    push_ui_rect(
        out,
        width,
        height,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        [0.02, 0.02, 0.02, 0.22],
    );

    push_inventory_icon(
        out,
        width,
        height,
        center_x,
        center_y,
        base_size * 0.64,
        held.block_id,
        true,
        base_size,
        block_tiles_x,
        block_tile_uv_size,
        item_tiles_x,
        item_tile_uv_size,
    );
}

fn push_slot_box(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    rect: crate::player::inventory::UiRect,
    selected: bool,
    hovered: bool,
) {
    push_ui_rect(
        out,
        width,
        height,
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        [0.06, 0.06, 0.06, 0.78],
    );
    let inset = (rect.w * 0.07).clamp(2.0, 4.0);
    let inner_color = if hovered {
        [0.24, 0.25, 0.29, 0.88]
    } else if selected {
        [0.33, 0.33, 0.33, 0.85]
    } else {
        [0.18, 0.18, 0.18, 0.78]
    };
    push_ui_rect(
        out,
        width,
        height,
        rect.x + inset,
        rect.y + inset,
        (rect.w - inset * 2.0).max(1.0),
        (rect.h - inset * 2.0).max(1.0),
        inner_color,
    );

    if selected || hovered {
        let border = (rect.w * 0.08).clamp(1.5, 4.0);
        let border_color = if selected {
            [0.95, 0.95, 0.87, 0.98]
        } else {
            [0.76, 0.86, 0.98, 0.60]
        };
        push_ui_rect(
            out,
            width,
            height,
            rect.x - border,
            rect.y - border,
            rect.w + border * 2.0,
            border,
            border_color,
        );
        push_ui_rect(
            out,
            width,
            height,
            rect.x - border,
            rect.y + rect.h,
            rect.w + border * 2.0,
            border,
            border_color,
        );
        push_ui_rect(
            out,
            width,
            height,
            rect.x - border,
            rect.y,
            border,
            rect.h,
            border_color,
        );
        push_ui_rect(
            out,
            width,
            height,
            rect.x + rect.w,
            rect.y,
            border,
            rect.h,
            border_color,
        );
    }
}

fn push_stack_meter(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    rect: crate::player::inventory::UiRect,
    item_id: i8,
    count: u8,
) {
    if count == 0 {
        return;
    }
    let bar_margin = (rect.w * 0.12).clamp(2.0, 6.0);
    let bar_h = (rect.h * 0.09).clamp(2.0, 5.0);
    let bar_w = (rect.w - bar_margin * 2.0).max(2.0);
    let bar_x = rect.x + bar_margin;
    let bar_y = rect.y + rect.h - bar_h - (rect.h * 0.08).clamp(1.0, 4.0);
    let max_count = item_max_stack_size(item_id).max(1);
    let fill_ratio = (count as f32 / max_count as f32).clamp(0.0, 1.0);
    let fill_w = (bar_w * fill_ratio).max(1.0);

    push_ui_rect(
        out,
        width,
        height,
        bar_x,
        bar_y,
        bar_w,
        bar_h,
        [0.04, 0.04, 0.04, 0.92],
    );
    push_ui_rect(
        out,
        width,
        height,
        bar_x,
        bar_y,
        fill_w,
        bar_h,
        [0.80, 0.88, 0.36, 0.95],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_inventory_icon(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
    size: f32,
    item_id: i8,
    selected: bool,
    item_slot_size: f32,
    block_tiles_x: u32,
    block_tile_uv_size: [f32; 2],
    item_tiles_x: u32,
    item_tile_uv_size: [f32; 2],
) {
    if let Some(tile_index) = item_icon_tile_index(item_id) {
        push_item_icon_2d(
            out,
            width,
            height,
            center_x,
            center_y,
            item_slot_size * 0.94,
            tile_index,
            item_tiles_x,
            item_tile_uv_size,
            UI_ATLAS_ITEM,
            selected,
        );
    } else if let Some(block_id) = placeable_block_id_for_item(item_id) {
        push_block_icon_textured(
            out,
            width,
            height,
            center_x,
            center_y,
            size,
            block_id,
            block_tiles_x,
            block_tile_uv_size,
            selected,
        );
    } else if item_id >= 0 && (item_id as usize) < block_count() {
        push_block_icon_textured(
            out,
            width,
            height,
            center_x,
            center_y,
            size,
            item_id,
            block_tiles_x,
            block_tile_uv_size,
            selected,
        );
    } else {
        push_block_icon(
            out, width, height, center_x, center_y, size, item_id, selected,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_item_icon_2d(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
    size: f32,
    tile_index: u32,
    tiles_x: u32,
    tile_uv_size: [f32; 2],
    atlas_select: f32,
    selected: bool,
) {
    if tiles_x == 0 || tile_uv_size[0] <= 0.0 || tile_uv_size[1] <= 0.0 {
        return;
    }
    let icon_size = size;
    let half = icon_size * 0.5;
    let x = center_x - half;
    let y = center_y - half;
    let tint = if selected {
        [1.08, 1.08, 1.08, 1.0]
    } else {
        [1.0, 1.0, 1.0, 1.0]
    };

    let tile_x = tile_index % tiles_x;
    let tile_y = tile_index / tiles_x;
    let uv_w = tile_uv_size[0];
    let uv_h = tile_uv_size[1];
    // Pull UVs slightly inward to avoid bleeding from neighboring tiles.
    let pad_u = uv_w * 0.04;
    let pad_v = uv_h * 0.04;
    let u0 = tile_x as f32 * uv_w + pad_u;
    let v0 = tile_y as f32 * uv_h + pad_v;
    let u1 = (tile_x + 1) as f32 * uv_w - pad_u;
    let v1 = (tile_y + 1) as f32 * uv_h - pad_v;

    push_ui_textured_rect(
        out,
        width,
        height,
        x,
        y,
        icon_size,
        icon_size,
        [u0, v0],
        [u1, v1],
        tint,
        atlas_select,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_ui_textured_rect(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    color: [f32; 4],
    atlas_select: f32,
) {
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    let x0 = (x / width) * 2.0 - 1.0;
    let x1 = ((x + w) / width) * 2.0 - 1.0;
    let y0 = 1.0 - (y / height) * 2.0;
    let y1 = 1.0 - ((y + h) / height) * 2.0;

    let v0 = UiVertex {
        position: [x0, y0],
        color,
        uv: [uv_min[0], uv_min[1]],
        use_texture: 1.0,
        atlas_select,
    };
    let v1 = UiVertex {
        position: [x1, y0],
        color,
        uv: [uv_max[0], uv_min[1]],
        use_texture: 1.0,
        atlas_select,
    };
    let v2 = UiVertex {
        position: [x1, y1],
        color,
        uv: [uv_max[0], uv_max[1]],
        use_texture: 1.0,
        atlas_select,
    };
    let v3 = UiVertex {
        position: [x0, y1],
        color,
        uv: [uv_min[0], uv_max[1]],
        use_texture: 1.0,
        atlas_select,
    };
    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn tile_uv_bounds(
    tile_index: u32,
    tiles_x: u32,
    tile_uv_size: [f32; 2],
) -> Option<([f32; 2], [f32; 2])> {
    if tiles_x == 0 || tile_uv_size[0] <= 0.0 || tile_uv_size[1] <= 0.0 {
        return None;
    }
    let tile_x = tile_index % tiles_x;
    let tile_y = tile_index / tiles_x;
    let uv_w = tile_uv_size[0];
    let uv_h = tile_uv_size[1];
    // Pull UVs slightly inward to avoid bleeding from neighboring tiles.
    let pad_u = uv_w * 0.04;
    let pad_v = uv_h * 0.04;
    let u0 = tile_x as f32 * uv_w + pad_u;
    let v0 = tile_y as f32 * uv_h + pad_v;
    let u1 = (tile_x + 1) as f32 * uv_w - pad_u;
    let v1 = (tile_y + 1) as f32 * uv_h - pad_v;
    Some(([u0, v0], [u1, v1]))
}

#[allow(clippy::too_many_arguments)]
fn push_block_icon_textured(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
    size: f32,
    block_id: i8,
    tiles_x: u32,
    tile_uv_size: [f32; 2],
    selected: bool,
) {
    if tiles_x == 0 || tile_uv_size[0] <= 0.0 || tile_uv_size[1] <= 0.0 {
        push_block_icon(
            out, width, height, center_x, center_y, size, block_id, selected,
        );
        return;
    }

    let (tiles, _, _) = dropped_block_face_data(block_id);
    let (top_uv_min, top_uv_max) = match tile_uv_bounds(tiles[2], tiles_x, tile_uv_size) {
        Some(v) => v,
        None => {
            push_block_icon(
                out, width, height, center_x, center_y, size, block_id, selected,
            );
            return;
        }
    };
    let (left_uv_min, left_uv_max) = match tile_uv_bounds(tiles[1], tiles_x, tile_uv_size) {
        Some(v) => v,
        None => {
            push_block_icon(
                out, width, height, center_x, center_y, size, block_id, selected,
            );
            return;
        }
    };
    let (right_uv_min, right_uv_max) = match tile_uv_bounds(tiles[0], tiles_x, tile_uv_size) {
        Some(v) => v,
        None => {
            push_block_icon(
                out, width, height, center_x, center_y, size, block_id, selected,
            );
            return;
        }
    };

    // Isometric-ish mini cube for the hotbar icon.
    let half_w = size * 0.56;
    let half_h = size * 0.34;
    let depth = size * 0.56;

    let top = (center_x, center_y - half_h);
    let right = (center_x + half_w, center_y);
    let bottom = (center_x, center_y + half_h);
    let left = (center_x - half_w, center_y);
    let down = (center_x, center_y + half_h + depth);
    let down_left = (center_x - half_w, center_y + depth);
    let down_right = (center_x + half_w, center_y + depth);

    let select_boost = if selected { 1.08 } else { 1.0 };
    let top_tint = [select_boost, select_boost, select_boost, 1.0];
    let left_tint = [
        0.88 * select_boost,
        0.88 * select_boost,
        0.88 * select_boost,
        1.0,
    ];
    let right_tint = [
        0.74 * select_boost,
        0.74 * select_boost,
        0.74 * select_boost,
        1.0,
    ];

    let top_u_mid = (top_uv_min[0] + top_uv_max[0]) * 0.5;
    let top_v_mid = (top_uv_min[1] + top_uv_max[1]) * 0.5;
    push_ui_textured_quad(
        out,
        width,
        height,
        top,
        right,
        bottom,
        left,
        [top_u_mid, top_uv_min[1]],
        [top_uv_max[0], top_v_mid],
        [top_u_mid, top_uv_max[1]],
        [top_uv_min[0], top_v_mid],
        top_tint,
        UI_ATLAS_BLOCK,
    );
    push_ui_textured_quad(
        out,
        width,
        height,
        left,
        bottom,
        down,
        down_left,
        [left_uv_min[0], left_uv_min[1]],
        [left_uv_max[0], left_uv_min[1]],
        [left_uv_max[0], left_uv_max[1]],
        [left_uv_min[0], left_uv_max[1]],
        left_tint,
        UI_ATLAS_BLOCK,
    );
    push_ui_textured_quad(
        out,
        width,
        height,
        right,
        down_right,
        down,
        bottom,
        [right_uv_min[0], right_uv_min[1]],
        [right_uv_max[0], right_uv_min[1]],
        [right_uv_max[0], right_uv_max[1]],
        [right_uv_min[0], right_uv_max[1]],
        right_tint,
        UI_ATLAS_BLOCK,
    );
}

fn push_ui_rect(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
) {
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    let x0 = (x / width) * 2.0 - 1.0;
    let x1 = ((x + w) / width) * 2.0 - 1.0;
    let y0 = 1.0 - (y / height) * 2.0;
    let y1 = 1.0 - ((y + h) / height) * 2.0;

    let v0 = UiVertex {
        position: [x0, y0],
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let v1 = UiVertex {
        position: [x1, y0],
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let v2 = UiVertex {
        position: [x1, y1],
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let v3 = UiVertex {
        position: [x0, y1],
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn text_width_3x5(text: &str, pixel: f32) -> f32 {
    let count = text.chars().count() as f32;
    if count <= 0.0 {
        return 0.0;
    }
    // 3 px glyph width + 1 px spacing.
    count * (pixel * 3.0) + (count - 1.0) * pixel
}

fn push_text_3x5_shadow(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    x: f32,
    y: f32,
    pixel: f32,
    text: &str,
    color: [f32; 4],
) {
    if text.is_empty() || pixel <= 0.0 {
        return;
    }
    let shadow_offset = pixel.max(1.0);
    push_text_3x5(
        out,
        width,
        height,
        x + shadow_offset,
        y + shadow_offset,
        pixel,
        text,
        [0.0, 0.0, 0.0, color[3] * 0.75],
    );
    push_text_3x5(out, width, height, x, y, pixel, text, color);
}

fn push_text_3x5(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    mut x: f32,
    mut y: f32,
    pixel: f32,
    text: &str,
    color: [f32; 4],
) {
    if text.is_empty() || pixel <= 0.0 {
        return;
    }
    let line_start_x = x;
    for ch in text.chars() {
        if ch == '\n' {
            y += pixel * 6.0;
            x = line_start_x;
            continue;
        }
        let glyph = glyph_3x5(ch).unwrap_or([0b111, 0b101, 0b111, 0b101, 0b101]);
        for (row, bits) in glyph.iter().copied().enumerate() {
            for col in 0..3 {
                if (bits & (1 << (2 - col))) == 0 {
                    continue;
                }
                push_ui_rect(
                    out,
                    width,
                    height,
                    x + col as f32 * pixel,
                    y + row as f32 * pixel,
                    pixel,
                    pixel,
                    color,
                );
            }
        }
        x += pixel * 4.0;
    }
}

fn glyph_3x5(ch: char) -> Option<[u8; 5]> {
    match ch.to_ascii_lowercase() {
        '0' => Some([0b111, 0b101, 0b101, 0b101, 0b111]),
        '1' => Some([0b010, 0b110, 0b010, 0b010, 0b111]),
        '2' => Some([0b111, 0b001, 0b111, 0b100, 0b111]),
        '3' => Some([0b111, 0b001, 0b111, 0b001, 0b111]),
        '4' => Some([0b101, 0b101, 0b111, 0b001, 0b001]),
        '5' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
        '6' => Some([0b111, 0b100, 0b111, 0b101, 0b111]),
        '7' => Some([0b111, 0b001, 0b010, 0b010, 0b010]),
        '8' => Some([0b111, 0b101, 0b111, 0b101, 0b111]),
        '9' => Some([0b111, 0b101, 0b111, 0b001, 0b111]),
        'a' => Some([0b111, 0b001, 0b111, 0b101, 0b111]),
        'b' => Some([0b110, 0b101, 0b110, 0b101, 0b110]),
        'c' => Some([0b111, 0b100, 0b100, 0b100, 0b111]),
        'd' => Some([0b110, 0b101, 0b101, 0b101, 0b110]),
        'e' => Some([0b111, 0b100, 0b110, 0b100, 0b111]),
        'f' => Some([0b111, 0b100, 0b110, 0b100, 0b100]),
        'g' => Some([0b111, 0b100, 0b101, 0b101, 0b111]),
        'h' => Some([0b101, 0b101, 0b111, 0b101, 0b101]),
        'i' => Some([0b111, 0b010, 0b010, 0b010, 0b111]),
        'j' => Some([0b001, 0b001, 0b001, 0b101, 0b111]),
        'k' => Some([0b101, 0b101, 0b110, 0b101, 0b101]),
        'l' => Some([0b100, 0b100, 0b100, 0b100, 0b111]),
        'm' => Some([0b101, 0b111, 0b111, 0b101, 0b101]),
        'n' => Some([0b110, 0b101, 0b101, 0b101, 0b101]),
        'o' => Some([0b111, 0b101, 0b101, 0b101, 0b111]),
        'p' => Some([0b111, 0b101, 0b111, 0b100, 0b100]),
        'q' => Some([0b111, 0b101, 0b101, 0b111, 0b001]),
        'r' => Some([0b110, 0b101, 0b110, 0b101, 0b101]),
        's' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
        't' => Some([0b111, 0b010, 0b010, 0b010, 0b010]),
        'u' => Some([0b101, 0b101, 0b101, 0b101, 0b111]),
        'v' => Some([0b101, 0b101, 0b101, 0b101, 0b010]),
        'w' => Some([0b101, 0b101, 0b111, 0b111, 0b101]),
        'x' => Some([0b101, 0b101, 0b010, 0b101, 0b101]),
        'y' => Some([0b101, 0b101, 0b111, 0b001, 0b111]),
        'z' => Some([0b111, 0b001, 0b010, 0b100, 0b111]),
        '.' => Some([0b000, 0b000, 0b000, 0b000, 0b010]),
        ',' => Some([0b000, 0b000, 0b000, 0b010, 0b100]),
        ':' => Some([0b000, 0b010, 0b000, 0b010, 0b000]),
        ';' => Some([0b000, 0b010, 0b000, 0b010, 0b100]),
        '!' => Some([0b010, 0b010, 0b010, 0b000, 0b010]),
        '?' => Some([0b111, 0b001, 0b011, 0b000, 0b010]),
        '\'' => Some([0b010, 0b010, 0b000, 0b000, 0b000]),
        '"' => Some([0b101, 0b101, 0b000, 0b000, 0b000]),
        '/' => Some([0b001, 0b001, 0b010, 0b100, 0b100]),
        '\\' => Some([0b100, 0b100, 0b010, 0b001, 0b001]),
        '-' => Some([0b000, 0b000, 0b111, 0b000, 0b000]),
        '+' => Some([0b000, 0b010, 0b111, 0b010, 0b000]),
        '=' => Some([0b000, 0b111, 0b000, 0b111, 0b000]),
        '_' => Some([0b000, 0b000, 0b000, 0b000, 0b111]),
        '(' => Some([0b001, 0b010, 0b010, 0b010, 0b001]),
        ')' => Some([0b100, 0b010, 0b010, 0b010, 0b100]),
        '[' => Some([0b011, 0b010, 0b010, 0b010, 0b011]),
        ']' => Some([0b110, 0b010, 0b010, 0b010, 0b110]),
        '<' => Some([0b001, 0b010, 0b100, 0b010, 0b001]),
        '>' => Some([0b100, 0b010, 0b001, 0b010, 0b100]),
        '*' => Some([0b000, 0b101, 0b010, 0b101, 0b000]),
        '#' => Some([0b101, 0b111, 0b101, 0b111, 0b101]),
        '%' => Some([0b101, 0b001, 0b010, 0b100, 0b101]),
        '|' => Some([0b010, 0b010, 0b010, 0b010, 0b010]),
        '^' => Some([0b010, 0b101, 0b000, 0b000, 0b000]),
        '~' => Some([0b000, 0b010, 0b101, 0b000, 0b000]),
        '@' => Some([0b111, 0b101, 0b111, 0b100, 0b111]),
        ' ' => Some([0b000, 0b000, 0b000, 0b000, 0b000]),
        _ => None,
    }
}

fn push_block_icon(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
    size: f32,
    block_id: i8,
    selected: bool,
) {
    let (mut top_color, mut left_color, mut right_color) = block_icon_colors(block_id);
    if selected {
        top_color = scale_color(top_color, 1.12);
        left_color = scale_color(left_color, 1.12);
        right_color = scale_color(right_color, 1.12);
    }

    // Isometric-ish mini cube for the hotbar icon.
    let half_w = size * 0.56;
    let half_h = size * 0.34;
    let depth = size * 0.56;

    let top = (center_x, center_y - half_h);
    let right = (center_x + half_w, center_y);
    let bottom = (center_x, center_y + half_h);
    let left = (center_x - half_w, center_y);
    let down = (center_x, center_y + half_h + depth);
    let down_left = (center_x - half_w, center_y + depth);
    let down_right = (center_x + half_w, center_y + depth);

    push_ui_quad(out, width, height, top, right, bottom, left, top_color);
    push_ui_quad(
        out, width, height, left, bottom, down, down_left, left_color,
    );
    push_ui_quad(
        out,
        width,
        height,
        right,
        down_right,
        down,
        bottom,
        right_color,
    );
}

fn block_icon_colors(block_id: i8) -> ([f32; 4], [f32; 4], [f32; 4]) {
    match block_id {
        x if x == BLOCK_STONE as i8 => (
            [0.76, 0.76, 0.76, 1.0],
            [0.56, 0.56, 0.56, 1.0],
            [0.46, 0.46, 0.46, 1.0],
        ),
        x if x == BLOCK_DIRT as i8 => (
            [0.61, 0.45, 0.27, 1.0],
            [0.46, 0.33, 0.20, 1.0],
            [0.37, 0.26, 0.16, 1.0],
        ),
        x if x == BLOCK_GRASS as i8 => (
            [0.45, 0.73, 0.35, 1.0],
            [0.49, 0.38, 0.23, 1.0],
            [0.40, 0.30, 0.19, 1.0],
        ),
        x if x == BLOCK_LOG as i8 => (
            [0.72, 0.58, 0.38, 1.0],
            [0.55, 0.40, 0.24, 1.0],
            [0.44, 0.31, 0.18, 1.0],
        ),
        x if x == BLOCK_LEAVES as i8 => (
            [0.39, 0.63, 0.32, 0.95],
            [0.29, 0.48, 0.24, 0.95],
            [0.23, 0.39, 0.19, 0.95],
        ),
        x if x == BLOCK_PLANKS_OAK as i8 => (
            [0.72, 0.57, 0.36, 1.0],
            [0.54, 0.41, 0.25, 1.0],
            [0.44, 0.33, 0.20, 1.0],
        ),
        x if x == BLOCK_CRAFTING_TABLE as i8 => (
            [0.73, 0.59, 0.40, 1.0],
            [0.55, 0.42, 0.28, 1.0],
            [0.46, 0.34, 0.22, 1.0],
        ),
        x if x == BLOCK_GRAVEL as i8 => (
            [0.65, 0.65, 0.66, 1.0],
            [0.50, 0.50, 0.51, 1.0],
            [0.40, 0.40, 0.41, 1.0],
        ),
        x if x == BLOCK_SAND as i8 => (
            [0.88, 0.80, 0.57, 1.0],
            [0.69, 0.61, 0.43, 1.0],
            [0.56, 0.49, 0.34, 1.0],
        ),
        x if x == ITEM_APPLE => (
            [0.92, 0.20, 0.18, 1.0],
            [0.74, 0.13, 0.12, 1.0],
            [0.58, 0.10, 0.10, 1.0],
        ),
        x if x == ITEM_STICK => (
            [0.76, 0.62, 0.36, 1.0],
            [0.57, 0.44, 0.24, 1.0],
            [0.45, 0.33, 0.19, 1.0],
        ),
        _ => (
            [0.7, 0.7, 0.7, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.4, 0.4, 0.4, 1.0],
        ),
    }
}

fn scale_color(mut color: [f32; 4], scale: f32) -> [f32; 4] {
    color[0] = (color[0] * scale).clamp(0.0, 1.0);
    color[1] = (color[1] * scale).clamp(0.0, 1.0);
    color[2] = (color[2] * scale).clamp(0.0, 1.0);
    color
}

fn push_ui_quad(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    d: (f32, f32),
    color: [f32; 4],
) {
    fn to_ndc(width: f32, height: f32, p: (f32, f32)) -> [f32; 2] {
        let x = (p.0 / width) * 2.0 - 1.0;
        let y = 1.0 - (p.1 / height) * 2.0;
        [x, y]
    }

    let va = UiVertex {
        position: to_ndc(width, height, a),
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let vb = UiVertex {
        position: to_ndc(width, height, b),
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let vc = UiVertex {
        position: to_ndc(width, height, c),
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    let vd = UiVertex {
        position: to_ndc(width, height, d),
        color,
        uv: [0.0, 0.0],
        use_texture: 0.0,
        atlas_select: 0.0,
    };
    out.extend_from_slice(&[va, vb, vc, va, vc, vd]);
}

#[allow(clippy::too_many_arguments)]
fn push_ui_textured_quad(
    out: &mut Vec<UiVertex>,
    width: f32,
    height: f32,
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    d: (f32, f32),
    uv_a: [f32; 2],
    uv_b: [f32; 2],
    uv_c: [f32; 2],
    uv_d: [f32; 2],
    color: [f32; 4],
    atlas_select: f32,
) {
    fn to_ndc(width: f32, height: f32, p: (f32, f32)) -> [f32; 2] {
        let x = (p.0 / width) * 2.0 - 1.0;
        let y = 1.0 - (p.1 / height) * 2.0;
        [x, y]
    }

    let va = UiVertex {
        position: to_ndc(width, height, a),
        color,
        uv: uv_a,
        use_texture: 1.0,
        atlas_select,
    };
    let vb = UiVertex {
        position: to_ndc(width, height, b),
        color,
        uv: uv_b,
        use_texture: 1.0,
        atlas_select,
    };
    let vc = UiVertex {
        position: to_ndc(width, height, c),
        color,
        uv: uv_c,
        use_texture: 1.0,
        atlas_select,
    };
    let vd = UiVertex {
        position: to_ndc(width, height, d),
        color,
        uv: uv_d,
        use_texture: 1.0,
        atlas_select,
    };
    out.extend_from_slice(&[va, vb, vc, va, vc, vd]);
}

fn build_break_overlay_mesh(coord: IVec3, stage: u32) -> (Vec<ChunkVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let max_stage = DESTROY_STAGE_COUNT.saturating_sub(1);
    let stage = stage.min(max_stage);
    let tile = DESTROY_STAGE_TILE_START + stage;
    let color = [1.0, 1.0, 1.0, 0.9];
    let inflate = 0.0025f32;

    let x0 = coord.x as f32 - inflate;
    let y0 = coord.y as f32 - inflate;
    let z0 = coord.z as f32 - inflate;
    let x1 = coord.x as f32 + 1.0 + inflate;
    let y1 = coord.y as f32 + 1.0 + inflate;
    let z1 = coord.z as f32 + 1.0 + inflate;

    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        0,
        [x1, y0, z0],
        [x1, y1, z0],
        [x1, y1, z1],
        [x1, y0, z1],
        tile,
        0,
        1,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        1,
        [x0, y0, z1],
        [x0, y1, z1],
        [x0, y1, z0],
        [x0, y0, z0],
        tile,
        0,
        1,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        2,
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y1, z1],
        [x1, y1, z0],
        tile,
        0,
        1,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        3,
        [x0, y0, z1],
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y0, z1],
        tile,
        0,
        1,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        4,
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
        tile,
        0,
        1,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        5,
        [x1, y0, z0],
        [x0, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        tile,
        0,
        1,
        1,
        color,
    );

    (vertices, indices)
}

fn build_sun_mesh(
    camera: &Camera,
    day_progress: f32,
    sun_height: f32,
) -> (Vec<ChunkVertex>, Vec<u32>) {
    if sun_height < -0.22 {
        return (Vec::new(), Vec::new());
    }

    let mut vertices = Vec::with_capacity(8);
    let mut indices = Vec::with_capacity(12);

    let sun_dir = sun_direction(day_progress, sun_height);
    let center = camera.position + sun_dir * 900.0;
    let half = 76.0f32;
    let y = center.y;
    let x0 = center.x - half;
    let x1 = center.x + half;
    let z0 = center.z - half;
    let z1 = center.z + half;
    let glow = ((sun_height + 0.22) / 1.22).clamp(0.35, 1.0);
    let color = [glow, glow, glow, 1.0];

    // Draw both sides so the sun is visible regardless of camera height.
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        3,
        [x0, y, z1],
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        0,
        0,
        SUN_TRANSPARENT_MODE,
        1,
        color,
    );
    emit_dropped_item_face(
        &mut vertices,
        &mut indices,
        2,
        [x0, y + 0.01, z0],
        [x0, y + 0.01, z1],
        [x1, y + 0.01, z1],
        [x1, y + 0.01, z0],
        0,
        0,
        SUN_TRANSPARENT_MODE,
        1,
        color,
    );

    (vertices, indices)
}

fn build_dropped_item_mesh(items: &[DroppedItemRender]) -> (Vec<ChunkVertex>, Vec<u32>) {
    if items.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut vertices = Vec::with_capacity(items.len() * 24);
    let mut indices = Vec::with_capacity(items.len() * 36);
    for &item in items {
        if let Some(tile_index) = item_icon_tile_index(item.block_id) {
            emit_dropped_item_prism(&mut vertices, &mut indices, item, tile_index);
        } else {
            emit_dropped_item_cube(&mut vertices, &mut indices, item);
        }
    }
    (vertices, indices)
}

fn emit_dropped_item_cube(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    item: DroppedItemRender,
) {
    let half = 0.22f32;
    let rot = Mat3::from_rotation_y(item.spin_y) * Mat3::from_rotation_z(item.tilt_z);
    let to_world = |p: Vec3| -> [f32; 3] { (rot * p + item.position).to_array() };

    let p000 = to_world(Vec3::new(-half, -half, -half));
    let p001 = to_world(Vec3::new(-half, -half, half));
    let p010 = to_world(Vec3::new(-half, half, -half));
    let p011 = to_world(Vec3::new(-half, half, half));
    let p100 = to_world(Vec3::new(half, -half, -half));
    let p101 = to_world(Vec3::new(half, -half, half));
    let p110 = to_world(Vec3::new(half, half, -half));
    let p111 = to_world(Vec3::new(half, half, half));

    let (tiles, rotations, transparent_modes) = dropped_block_face_data(item.block_id);
    emit_dropped_item_face(
        vertices,
        indices,
        0,
        p100,
        p110,
        p111,
        p101,
        tiles[0],
        rotations[0],
        transparent_modes[0],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    emit_dropped_item_face(
        vertices,
        indices,
        1,
        p001,
        p011,
        p010,
        p000,
        tiles[1],
        rotations[1],
        transparent_modes[1],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    emit_dropped_item_face(
        vertices,
        indices,
        2,
        p010,
        p011,
        p111,
        p110,
        tiles[2],
        rotations[2],
        transparent_modes[2],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    emit_dropped_item_face(
        vertices,
        indices,
        3,
        p001,
        p000,
        p100,
        p101,
        tiles[3],
        rotations[3],
        transparent_modes[3],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    emit_dropped_item_face(
        vertices,
        indices,
        4,
        p001,
        p101,
        p111,
        p011,
        tiles[4],
        rotations[4],
        transparent_modes[4],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    emit_dropped_item_face(
        vertices,
        indices,
        5,
        p100,
        p000,
        p010,
        p110,
        tiles[5],
        rotations[5],
        transparent_modes[5],
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
}

const DROPPED_ITEM_TILE_PIXELS: u32 = 16;

struct CpuItemAtlas {
    width: u32,
    tiles_x: u32,
    tiles_y: u32,
    rgba: Vec<u8>,
}

fn cpu_item_atlas() -> Option<&'static CpuItemAtlas> {
    static ATLAS: OnceLock<Option<CpuItemAtlas>> = OnceLock::new();
    ATLAS.get_or_init(load_cpu_item_atlas).as_ref()
}

fn load_cpu_item_atlas() -> Option<CpuItemAtlas> {
    let atlas_rgba = image::open("src/texturing/atlas_output/atlas_items.png")
        .ok()?
        .to_rgba8();
    let width = atlas_rgba.width();
    let height = atlas_rgba.height();
    if width == 0
        || height == 0
        || width % DROPPED_ITEM_TILE_PIXELS != 0
        || height % DROPPED_ITEM_TILE_PIXELS != 0
    {
        return None;
    }
    Some(CpuItemAtlas {
        width,
        tiles_x: width / DROPPED_ITEM_TILE_PIXELS,
        tiles_y: height / DROPPED_ITEM_TILE_PIXELS,
        rgba: atlas_rgba.into_raw(),
    })
}

fn item_tile_pixel_rgba(tile_index: u32, px: u32, py: u32) -> Option<[u8; 4]> {
    if px >= DROPPED_ITEM_TILE_PIXELS || py >= DROPPED_ITEM_TILE_PIXELS {
        return None;
    }
    let atlas = cpu_item_atlas()?;
    if atlas.tiles_x == 0 {
        return None;
    }
    let tile_x = tile_index % atlas.tiles_x;
    let tile_y = tile_index / atlas.tiles_x;
    if tile_y >= atlas.tiles_y {
        return None;
    }
    let x = tile_x * DROPPED_ITEM_TILE_PIXELS + px;
    let y = tile_y * DROPPED_ITEM_TILE_PIXELS + py;
    let idx = ((y * atlas.width + x) * 4) as usize;
    if idx + 3 >= atlas.rgba.len() {
        return None;
    }
    Some([
        atlas.rgba[idx],
        atlas.rgba[idx + 1],
        atlas.rgba[idx + 2],
        atlas.rgba[idx + 3],
    ])
}

fn dropped_item_side_color(item_id: i8) -> [f32; 4] {
    match item_id {
        ITEM_APPLE => [0.82, 0.20, 0.18, 1.0],
        ITEM_STICK => [0.66, 0.50, 0.30, 1.0],
        _ => [0.70, 0.70, 0.70, 1.0],
    }
}

fn scale_rgba(mut c: [f32; 4], k: f32) -> [f32; 4] {
    c[0] = (c[0] * k).clamp(0.0, 1.0);
    c[1] = (c[1] * k).clamp(0.0, 1.0);
    c[2] = (c[2] * k).clamp(0.0, 1.0);
    c
}

fn emit_dropped_item_prism(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    item: DroppedItemRender,
    item_tile_index: u32,
) {
    // Thin cuboid so item drops still look like items, but with visible depth.
    let half_x = 0.18f32;
    let half_y = 0.20f32;
    let half_z = 0.055f32;
    let rot = Mat3::from_rotation_y(item.spin_y) * Mat3::from_rotation_z(item.tilt_z);
    let to_world = |p: Vec3| -> [f32; 3] { (rot * p + item.position).to_array() };

    let p000 = to_world(Vec3::new(-half_x, -half_y, -half_z));
    let p001 = to_world(Vec3::new(-half_x, -half_y, half_z));
    let p010 = to_world(Vec3::new(-half_x, half_y, -half_z));
    let p011 = to_world(Vec3::new(-half_x, half_y, half_z));
    let p100 = to_world(Vec3::new(half_x, -half_y, -half_z));
    let p101 = to_world(Vec3::new(half_x, -half_y, half_z));
    let p110 = to_world(Vec3::new(half_x, half_y, -half_z));
    let p111 = to_world(Vec3::new(half_x, half_y, half_z));

    let item_mode_alpha = 9u32; // 8+1 => sample item atlas, alpha-cutout mode.
    emit_dropped_item_face(
        vertices,
        indices,
        4,
        p001,
        p101,
        p111,
        p011,
        item_tile_index,
        0,
        item_mode_alpha,
        1,
        [1.0, 1.0, 1.0, 1.0],
    );
    let mirrored_side_rotation = if item.block_id == ITEM_APPLE { 2 } else { 0 };
    emit_dropped_item_face_mirror_y(
        vertices,
        indices,
        5,
        p100,
        p000,
        p010,
        p110,
        item_tile_index,
        mirrored_side_rotation,
        item_mode_alpha,
        1,
        [1.0, 1.0, 1.0, 1.0],
    );

    let mut opaque = [false; (DROPPED_ITEM_TILE_PIXELS * DROPPED_ITEM_TILE_PIXELS) as usize];
    let mut rgb = [[0.0f32; 3]; (DROPPED_ITEM_TILE_PIXELS * DROPPED_ITEM_TILE_PIXELS) as usize];
    let mut any_opaque = false;
    for py in 0..DROPPED_ITEM_TILE_PIXELS {
        for px in 0..DROPPED_ITEM_TILE_PIXELS {
            let idx = (py * DROPPED_ITEM_TILE_PIXELS + px) as usize;
            if let Some([r, g, b, a]) = item_tile_pixel_rgba(item_tile_index, px, py)
                && a >= 20
            {
                opaque[idx] = true;
                rgb[idx] = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                any_opaque = true;
            }
        }
    }

    if !any_opaque {
        // Fallback if atlas data isn't available.
        let base_side = dropped_item_side_color(item.block_id);
        let side_a = scale_rgba(base_side, 0.72);
        let side_b = scale_rgba(base_side, 0.82);
        let side_top = scale_rgba(base_side, 0.92);
        let side_bottom = scale_rgba(base_side, 0.58);
        emit_dropped_item_face(
            vertices, indices, 0, p100, p110, p111, p101, 0, 0, 0, 0, side_a,
        );
        emit_dropped_item_face(
            vertices, indices, 1, p001, p011, p010, p000, 0, 0, 0, 0, side_b,
        );
        emit_dropped_item_face(
            vertices, indices, 2, p010, p011, p111, p110, 0, 0, 0, 0, side_top,
        );
        emit_dropped_item_face(
            vertices,
            indices,
            3,
            p001,
            p000,
            p100,
            p101,
            0,
            0,
            0,
            0,
            side_bottom,
        );
        return;
    }

    let pixel_w = (half_x * 2.0) / DROPPED_ITEM_TILE_PIXELS as f32;
    let pixel_h = (half_y * 2.0) / DROPPED_ITEM_TILE_PIXELS as f32;
    let is_opaque = |x: i32, y: i32| -> bool {
        if x < 0
            || y < 0
            || x >= DROPPED_ITEM_TILE_PIXELS as i32
            || y >= DROPPED_ITEM_TILE_PIXELS as i32
        {
            return false;
        }
        opaque[(y as u32 * DROPPED_ITEM_TILE_PIXELS + x as u32) as usize]
    };

    for y in 0..DROPPED_ITEM_TILE_PIXELS as i32 {
        for x in 0..DROPPED_ITEM_TILE_PIXELS as i32 {
            if !is_opaque(x, y) {
                continue;
            }
            let idx = (y as u32 * DROPPED_ITEM_TILE_PIXELS + x as u32) as usize;
            let base = [rgb[idx][0], rgb[idx][1], rgb[idx][2], 1.0];

            let x0 = -half_x + x as f32 * pixel_w;
            let x1 = x0 + pixel_w;
            let y_top = half_y - y as f32 * pixel_h;
            let y_bot = y_top - pixel_h;

            if !is_opaque(x - 1, y) {
                emit_dropped_item_face(
                    vertices,
                    indices,
                    1,
                    to_world(Vec3::new(x0, y_bot, half_z)),
                    to_world(Vec3::new(x0, y_top, half_z)),
                    to_world(Vec3::new(x0, y_top, -half_z)),
                    to_world(Vec3::new(x0, y_bot, -half_z)),
                    0,
                    0,
                    0,
                    0,
                    scale_rgba(base, 0.84),
                );
            }
            if !is_opaque(x + 1, y) {
                emit_dropped_item_face(
                    vertices,
                    indices,
                    0,
                    to_world(Vec3::new(x1, y_bot, -half_z)),
                    to_world(Vec3::new(x1, y_top, -half_z)),
                    to_world(Vec3::new(x1, y_top, half_z)),
                    to_world(Vec3::new(x1, y_bot, half_z)),
                    0,
                    0,
                    0,
                    0,
                    scale_rgba(base, 0.74),
                );
            }
            if !is_opaque(x, y - 1) {
                emit_dropped_item_face(
                    vertices,
                    indices,
                    2,
                    to_world(Vec3::new(x0, y_top, -half_z)),
                    to_world(Vec3::new(x0, y_top, half_z)),
                    to_world(Vec3::new(x1, y_top, half_z)),
                    to_world(Vec3::new(x1, y_top, -half_z)),
                    0,
                    0,
                    0,
                    0,
                    scale_rgba(base, 0.95),
                );
            }
            if !is_opaque(x, y + 1) {
                emit_dropped_item_face(
                    vertices,
                    indices,
                    3,
                    to_world(Vec3::new(x0, y_bot, half_z)),
                    to_world(Vec3::new(x0, y_bot, -half_z)),
                    to_world(Vec3::new(x1, y_bot, -half_z)),
                    to_world(Vec3::new(x1, y_bot, half_z)),
                    0,
                    0,
                    0,
                    0,
                    scale_rgba(base, 0.60),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_dropped_item_face(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    face: u32,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    tile: u32,
    rotation: u32,
    transparent_mode: u32,
    use_texture: u32,
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    vertices.push(ChunkVertex {
        position: p0,
        uv: [0.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [1.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [1.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments)]
fn emit_dropped_item_face_mirror_y(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    face: u32,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    tile: u32,
    rotation: u32,
    transparent_mode: u32,
    use_texture: u32,
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    vertices.push(ChunkVertex {
        position: p0,
        uv: [0.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p1,
        uv: [1.0, 0.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p2,
        uv: [1.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    vertices.push(ChunkVertex {
        position: p3,
        uv: [0.0, 1.0],
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        color,
    });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn dropped_block_face_data(item_id: i8) -> ([u32; 6], [u32; 6], [u32; 6]) {
    let block_id = placeable_block_id_for_item(item_id).unwrap_or(item_id);
    if let Some(tex) = block_texture_by_id(block_id) {
        (tex.tiles, tex.rotations, tex.transparent_mode)
    } else {
        ([0; 6], [0; 6], [0; 6])
    }
}

fn chunk_visible(
    camera_pos: Vec3,
    camera_forward_normalized: Vec3,
    chunk: &SuperChunkGpuMesh,
) -> bool {
    let to_center = chunk.center - camera_pos;
    let dist_sq = to_center.length_squared();
    if dist_sq <= chunk.radius * chunk.radius {
        return true;
    }
    let dist = dist_sq.sqrt();
    let dir = to_center / dist;
    let dot = camera_forward_normalized.dot(dir);
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

fn raw_slot_memory_bytes(slot: &ChunkSlot) -> u64 {
    slot.vertex_capacity as u64 * std::mem::size_of::<ChunkVertex>() as u64
        + slot.index_capacity as u64 * std::mem::size_of::<u32>() as u64
}

fn packed_slot_memory_bytes(slot: &ChunkSlot) -> u64 {
    slot.vertex_capacity as u64 * std::mem::size_of::<PackedFarVertex>() as u64
        + slot.index_capacity as u64 * std::mem::size_of::<u32>() as u64
}

fn schedule_superchunk_update(
    gpu: &mut GpuInner,
    coord: IVec3,
    vertices: ChunkVertices,
    indices: Arc<[u32]>,
) -> bool {
    let super_coord = super_chunk_coord(coord);
    let Some((slot, is_packed)) =
        gpu.super_chunks
            .get(&super_coord)
            .and_then(|super_chunk| match &vertices {
                ChunkVertices::Raw(_) => super_chunk
                    .raw_slots
                    .get(&coord)
                    .copied()
                    .map(|slot| (slot, false)),
                ChunkVertices::PackedFar(_) => super_chunk
                    .packed_slots
                    .get(&coord)
                    .copied()
                    .map(|slot| (slot, true)),
            })
    else {
        return false;
    };
    let vertex_count = match &vertices {
        ChunkVertices::Raw(raw) => raw.len(),
        ChunkVertices::PackedFar(packed) => packed.len(),
    };
    if vertex_count == 0 || indices.is_empty() {
        let Some(super_chunk) = gpu.super_chunks.get_mut(&super_coord) else {
            return false;
        };
        if is_packed {
            if let Some(index_buffer) = super_chunk.packed_index_buffer.as_ref() {
                clear_chunk_slot(
                    &gpu.queue,
                    index_buffer,
                    &mut super_chunk.packed_slots,
                    coord,
                    &slot,
                );
            } else {
                return false;
            }
        } else if let Some(index_buffer) = super_chunk.raw_index_buffer.as_ref() {
            clear_chunk_slot(
                &gpu.queue,
                index_buffer,
                &mut super_chunk.raw_slots,
                coord,
                &slot,
            );
        } else {
            return false;
        }
        return true;
    }
    if vertex_count as u32 > slot.vertex_capacity {
        return false;
    }
    if indices.len() as u32 > slot.index_capacity {
        return false;
    }
    if gpu
        .pending_updates
        .insert(coord, PendingUpdate { vertices, indices })
        .is_none()
    {
        gpu.pending_queue.push_back(coord);
    }
    true
}

fn clear_chunk_slot(
    queue: &wgpu::Queue,
    index_buffer: &wgpu::Buffer,
    slots: &mut HashMap<IVec3, ChunkSlot>,
    coord: IVec3,
    slot: &ChunkSlot,
) {
    let zero_indices = vec![slot.vertex_offset; slot.index_capacity as usize];
    let i_start = slot.index_offset as wgpu::BufferAddress
        * std::mem::size_of::<u32>() as wgpu::BufferAddress;
    queue.write_buffer(index_buffer, i_start, bytemuck::cast_slice(&zero_indices));
    slots.remove(&coord);
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
        let Some((slot, is_packed)) =
            gpu.super_chunks
                .get(&super_coord)
                .and_then(|super_chunk| match &update.vertices {
                    ChunkVertices::Raw(_) => super_chunk
                        .raw_slots
                        .get(&coord)
                        .copied()
                        .map(|slot| (slot, false)),
                    ChunkVertices::PackedFar(_) => super_chunk
                        .packed_slots
                        .get(&coord)
                        .copied()
                        .map(|slot| (slot, true)),
                })
        else {
            mark_super_dirty(gpu, super_coord);
            continue;
        };
        let vertex_count = match &update.vertices {
            ChunkVertices::Raw(raw) => raw.len(),
            ChunkVertices::PackedFar(packed) => packed.len(),
        };
        if vertex_count as u32 > slot.vertex_capacity
            || update.indices.len() as u32 > slot.index_capacity
        {
            mark_super_dirty(gpu, super_coord);
            continue;
        }

        if is_packed {
            let Some(vertex_buffer) = gpu
                .super_chunks
                .get(&super_coord)
                .and_then(|super_chunk| super_chunk.packed_vertex_buffer.as_ref())
            else {
                mark_super_dirty(gpu, super_coord);
                continue;
            };
            let ChunkVertices::PackedFar(packed) = &update.vertices else {
                mark_super_dirty(gpu, super_coord);
                continue;
            };
            let v_start = slot.vertex_offset as wgpu::BufferAddress
                * std::mem::size_of::<PackedFarVertex>() as wgpu::BufferAddress;
            gpu.queue
                .write_buffer(vertex_buffer, v_start, bytemuck::cast_slice(packed));
        } else {
            let Some(vertex_buffer) = gpu
                .super_chunks
                .get(&super_coord)
                .and_then(|super_chunk| super_chunk.raw_vertex_buffer.as_ref())
            else {
                mark_super_dirty(gpu, super_coord);
                continue;
            };
            let ChunkVertices::Raw(raw) = &update.vertices else {
                mark_super_dirty(gpu, super_coord);
                continue;
            };
            let v_start = slot.vertex_offset as wgpu::BufferAddress
                * std::mem::size_of::<ChunkVertex>() as wgpu::BufferAddress;
            gpu.queue
                .write_buffer(vertex_buffer, v_start, bytemuck::cast_slice(raw));
        }

        gpu.staged_indices.clear();
        if gpu.staged_indices.capacity() < slot.index_capacity as usize {
            gpu.staged_indices
                .reserve(slot.index_capacity as usize - gpu.staged_indices.capacity());
        }
        gpu.staged_indices
            .extend(update.indices.iter().map(|i| i + slot.vertex_offset));
        let pad = slot
            .index_capacity
            .saturating_sub(update.indices.len() as u32);
        if pad > 0 {
            gpu.staged_indices
                .extend(std::iter::repeat_n(slot.vertex_offset, pad as usize));
        }
        let i_start = slot.index_offset as wgpu::BufferAddress
            * std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let Some(index_buffer) = gpu.super_chunks.get(&super_coord).and_then(|super_chunk| {
            if is_packed {
                super_chunk.packed_index_buffer.as_ref()
            } else {
                super_chunk.raw_index_buffer.as_ref()
            }
        }) else {
            mark_super_dirty(gpu, super_coord);
            continue;
        };
        gpu.queue.write_buffer(
            index_buffer,
            i_start,
            bytemuck::cast_slice(&gpu.staged_indices),
        );
    }
}
