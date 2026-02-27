pub const FACE_COUNT: usize = 6;
pub const RENDER_SHAPE_CUBE: u8 = 0;
pub const RENDER_SHAPE_CROSS: u8 = 1;

#[derive(Clone, Copy)]
pub struct BlockTexture {
    pub tiles: [u32; FACE_COUNT],
    pub rotations: [u32; FACE_COUNT],
    pub transparent_mode: [u32; FACE_COUNT],
    pub light_emission: f32,
    pub render_shape: u8,
}
