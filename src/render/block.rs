use crate::colorthing::ColorThing;
use bytemuck::{Pod, Zeroable};

pub const FACE_COUNT: usize = 6;

#[derive(Clone, Copy)]
pub struct BlockTexture {
    pub tiles: [u32; FACE_COUNT],
    pub colors: [ColorThing; FACE_COUNT],
    pub rotations: [u32; FACE_COUNT],
}

#[allow(dead_code)]
impl BlockTexture {
    pub fn solid(tile: u32, color: ColorThing) -> Self {
        Self {
            tiles: [tile; FACE_COUNT],
            colors: [color; FACE_COUNT],
            rotations: [0; FACE_COUNT],
        }
    }

    pub fn top_bottom_sides(
        top: u32,
        bottom: u32,
        side: u32,
        color: ColorThing,
    ) -> Self {
        Self {
            tiles: [side, side, top, bottom, side, side],
            colors: [color; FACE_COUNT],
            rotations: [0; FACE_COUNT],
        }
    }

    pub fn with_rotations(mut self, rotations: [u32; FACE_COUNT]) -> Self {
        self.rotations = rotations;
        self
    }

    pub fn to_raw(&self) -> BlockTextureRaw {
        BlockTextureRaw {
            colors: self
                .colors
                .iter()
                .map(|c| c.as_f32_rgba())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            tiles: self.tiles,
            rotations: self.rotations,
            _pad0: [0; 2],
            _pad1: [0; 2],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockTextureRaw {
    pub colors: [[f32; 4]; FACE_COUNT],
    pub tiles: [u32; FACE_COUNT],
    pub rotations: [u32; FACE_COUNT],
    pub _pad0: [u32; 2],
    pub _pad1: [u32; 2],
}
