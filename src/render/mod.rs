pub mod atlas;
pub mod block;
pub mod mesh;
pub mod gpu;
pub mod texture;

pub use atlas::TextureAtlas;
pub use block::BlockTexture;
pub use gpu::Gpu;
pub use gpu::GpuStats;
pub struct CubeStyle {
    pub use_texture: bool,
}
