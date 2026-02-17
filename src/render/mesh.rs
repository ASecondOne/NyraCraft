use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub tile: u32,
    pub face: u32,
    pub rotation: u32,
    pub use_texture: u32,
    pub transparent_mode: u32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PackedFarVertex {
    pub position: [i16; 4],
    pub uv: [i16; 2],
    pub meta: [u16; 2], // [tile, packed bits: face|rotation|use_texture|transparent_mode]
    pub color: [u8; 4],
}

impl ChunkVertex {
    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ChunkVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress
                        + mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress
                        + 2 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress
                        + 3 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress
                        + 4 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress
                        + 5 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

impl PackedFarVertex {
    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<PackedFarVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Sint16x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[i16; 4]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Sint16x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[i16; 4]>() as wgpu::BufferAddress
                        + mem::size_of::<[i16; 2]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint16x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[i16; 4]>() as wgpu::BufferAddress
                        + mem::size_of::<[i16; 2]>() as wgpu::BufferAddress
                        + mem::size_of::<[u16; 2]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        }
    }
}

fn quantize_unorm01_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

const PACKED_UV_SCALE: f32 = 256.0;
const PACKED_UV_LIMIT: f32 = i16::MAX as f32 / PACKED_UV_SCALE;

fn quantize_uv_to_i16(v: f32) -> i16 {
    (v.clamp(-PACKED_UV_LIMIT, PACKED_UV_LIMIT) * PACKED_UV_SCALE).round() as i16
}

pub fn pack_far_vertices(vertices: &[ChunkVertex]) -> Vec<PackedFarVertex> {
    let mut packed = Vec::with_capacity(vertices.len());
    for v in vertices {
        let face = v.face.min(0xF_u32);
        let rotation = v.rotation.min(0xF_u32);
        let use_texture = v.use_texture.min(0x1_u32);
        let transparent_mode = v.transparent_mode.min(0xF_u32);
        let flags = (face as u16)
            | ((rotation as u16) << 4)
            | ((use_texture as u16) << 8)
            | ((transparent_mode as u16) << 9);
        packed.push(PackedFarVertex {
            position: [
                v.position[0].round() as i16,
                v.position[1].round() as i16,
                v.position[2].round() as i16,
                0,
            ],
            uv: [quantize_uv_to_i16(v.uv[0]), quantize_uv_to_i16(v.uv[1])],
            meta: [v.tile.min(u16::MAX as u32) as u16, flags],
            color: [
                quantize_unorm01_to_u8(v.color[0]),
                quantize_unorm01_to_u8(v.color[1]),
                quantize_unorm01_to_u8(v.color[2]),
                quantize_unorm01_to_u8(v.color[3]),
            ],
        });
    }
    packed
}
