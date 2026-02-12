pub const FACE_COUNT: usize = 6;

#[derive(Clone, Copy)]
pub struct BlockTexture {
    pub tiles: [u32; FACE_COUNT],
    pub rotations: [u32; FACE_COUNT],
    pub transparent_mode: [u32; FACE_COUNT],
}
