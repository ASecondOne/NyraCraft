pub mod atlas;
pub mod block;
pub mod gpu;
pub mod mesh;
pub mod texture;

use glam::{IVec3, Vec3};
use std::sync::Arc;

use crate::render::mesh::{ChunkVertex, PackedFarVertex};

pub use atlas::TextureAtlas;
pub use block::BlockTexture;
pub use gpu::DroppedItemRender;
pub use gpu::Gpu;
pub use gpu::GpuStats;
pub struct CubeStyle {
    pub use_texture: bool,
}

pub trait MeshUploadBackend {
    fn stats(&self) -> GpuStats;
    fn chunk_memory_bytes(&self, coord: IVec3) -> Option<u64>;
    fn upsert_chunk(
        &mut self,
        coord: IVec3,
        center: Vec3,
        radius: f32,
        vertices: Vec<ChunkVertex>,
        indices: Vec<u32>,
    );
    fn upsert_chunk_packed(
        &mut self,
        coord: IVec3,
        center: Vec3,
        radius: f32,
        vertices: Arc<[PackedFarVertex]>,
        indices: Arc<[u32]>,
    );
    fn remove_chunk(&mut self, coord: IVec3);
    fn clear_chunks(&mut self);
    fn rebuild_dirty_superchunks(&mut self, camera_pos: Vec3, budget: usize);
}
