use crate::world::CHUNK_SIZE;
use crate::world::blocks::{
    BLOCK_DIRT, BLOCK_GRASS, BLOCK_GRAVEL, BLOCK_LEAVES, BLOCK_LOG, BLOCK_SAND, BLOCK_STONE,
    block_count,
};
use noise::{NoiseFn, Perlin};

pub const WORLD_SIZE_CHUNKS: i32 = 1000;
pub const WORLD_HALF_SIZE_CHUNKS: i32 = WORLD_SIZE_CHUNKS / 2;
pub const WORLD_HALF_SIZE_BLOCKS: i32 = WORLD_HALF_SIZE_CHUNKS * CHUNK_SIZE;
const TREE_CELL_SIZE: i32 = 6;
const TREE_CELL_MARGIN: i32 = 2;
const TREE_CELL_SPAWN_CHANCE_PCT: u32 = 92;
const FLAT_SURFACE_Y: i32 = 3;
const DESERT_HUMIDITY_MAX: f64 = 0.34;
const GRAVEL_DESERT_TEMPERATURE_MAX: f64 = 0.42;
const SAND_DESERT_TEMPERATURE_MIN: f64 = 0.62;
const CAVE_MIN_Y: i32 = -96;
const SURFACE_CAVE_DEPTH: i32 = 10;
const TREE_MIN_SUPPORT_LAYERS: i32 = 3;
const WORLDGEN_LAYOUT_VERSION: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceBiome {
    Temperate,
    GravelDesert,
    SandDesert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMode {
    Normal,
    Flat,
    Grid,
}

#[derive(Clone, Copy)]
pub struct TreeSpec {
    pub base_y: i32,
    pub trunk_h: i32,
    pub leaf_r: i32,
}

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    pub world_id: u64,
    pub mode: WorldMode,
    base_height: i32,
    plains_amp: f64,
    plains_freq: f64,
    perlin: Perlin,
    mountain_amp: f64,
    mountain_freq: f64,
    mountain_perlin: Perlin,
    cliffs_amp: f64,
    cliffs_freq: f64,
    cliffs_depth: f64,
    cliffs_perlin: Perlin,
    biome_freq: f64,
    biome_perlin: Perlin,
    temperature_freq: f64,
    temperature_perlin: Perlin,
    humidity_freq: f64,
    humidity_perlin: Perlin,
    cave_freq: f64,
    cave_freq_y: f64,
    cave_perlin: Perlin,
    cave_detail_freq: f64,
    cave_detail_freq_y: f64,
    cave_detail_perlin: Perlin,
    cave_entrance_freq: f64,
    cave_entrance_perlin: Perlin,
    cave_threshold: f64,
    surface_cave_boost: f64,
    offset_x: f64,
    offset_z: f64,
}

impl WorldGen {
    pub fn new(seed: u32, mode: WorldMode) -> Self {
        let perlin = Perlin::new(seed);
        let mountain_perlin = Perlin::new(seed.wrapping_add(1));
        let cliffs_perlin = Perlin::new(seed.wrapping_add(2));
        let biome_perlin = Perlin::new(seed.wrapping_add(3));
        let temperature_perlin = Perlin::new(seed.wrapping_add(4));
        let humidity_perlin = Perlin::new(seed.wrapping_add(5));
        let cave_perlin = Perlin::new(seed.wrapping_add(6));
        let cave_detail_perlin = Perlin::new(seed.wrapping_add(7));
        let cave_entrance_perlin = Perlin::new(seed.wrapping_add(8));
        let world_id = derive_world_id(seed, mode);
        let (offset_x, offset_z) = derive_offsets(seed);
        Self {
            seed,
            world_id,
            mode,
            base_height: 8,
            plains_amp: 5.0,
            plains_freq: 0.0065,
            perlin,
            mountain_amp: 34.0,
            mountain_freq: 0.018,
            mountain_perlin,
            cliffs_amp: 6.0,
            cliffs_freq: 0.02,
            cliffs_depth: 8.0,
            cliffs_perlin,
            biome_freq: 0.0012,
            biome_perlin,
            temperature_freq: 0.0016,
            temperature_perlin,
            humidity_freq: 0.0019,
            humidity_perlin,
            cave_freq: 0.029,
            cave_freq_y: 0.047,
            cave_perlin,
            cave_detail_freq: 0.071,
            cave_detail_freq_y: 0.098,
            cave_detail_perlin,
            cave_entrance_freq: 0.0125,
            cave_entrance_perlin,
            cave_threshold: 0.162,
            surface_cave_boost: 0.20,
            offset_x,
            offset_z,
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            WorldMode::Normal => "NORMAL",
            WorldMode::Flat => "FLAT",
            WorldMode::Grid => "GRID",
        }
    }

    pub fn in_world_bounds(&self, x: i32, z: i32) -> bool {
        x.abs() < WORLD_HALF_SIZE_BLOCKS && z.abs() < WORLD_HALF_SIZE_BLOCKS
    }

    pub fn height_at(&self, x: i32, z: i32) -> i32 {
        if !self.in_world_bounds(x, z) {
            return -1_000_000;
        }
        if self.mode != WorldMode::Normal {
            return FLAT_SURFACE_Y;
        }
        let fx = x as f64 + self.offset_x;
        let fz = z as f64 + self.offset_z;

        let biome_n = self
            .biome_perlin
            .get([fx * self.biome_freq, fz * self.biome_freq]);
        let mountain_w = smoothstep(0.20, 0.80, biome_n);
        let plains_w = 1.0 - mountain_w;

        let plains_low = self
            .perlin
            .get([fx * self.plains_freq, fz * self.plains_freq]);
        let plains_high = self
            .perlin
            .get([fx * self.plains_freq * 2.0, fz * self.plains_freq * 2.0]);
        let plains_h = self.base_height as f64
            + plains_low * self.plains_amp
            + plains_high * (self.plains_amp * 0.35);

        let mountain_n = self
            .mountain_perlin
            .get([fx * self.mountain_freq, fz * self.mountain_freq]);
        let ridged = (1.0 - mountain_n.abs()).powf(2.0);
        let mountain_h = self.base_height as f64 + 8.0 + ridged * self.mountain_amp;

        let cliffs_n = self
            .cliffs_perlin
            .get([fx * self.cliffs_freq, fz * self.cliffs_freq]);
        let cliffs_h = cliffs_n * self.cliffs_amp - cliffs_n.abs() * self.cliffs_depth;

        let mut height = plains_h * plains_w + mountain_h * mountain_w;
        height += cliffs_h * mountain_w * 0.35;
        height.round() as i32
    }

    pub fn highest_solid_y_at(&self, x: i32, z: i32) -> i32 {
        let height = self.height_at(x, z);
        if !self.in_world_bounds(x, z) {
            return height;
        }
        match self.mode {
            WorldMode::Normal | WorldMode::Flat => height,
            WorldMode::Grid => {
                if is_grid_marker_column(x, z) {
                    height + 2
                } else {
                    height
                }
            }
        }
    }

    fn climate_at(&self, x: i32, z: i32, height: i32) -> (f64, f64) {
        let fx = x as f64 + self.offset_x;
        let fz = z as f64 + self.offset_z;
        let temp_n = self
            .temperature_perlin
            .get([fx * self.temperature_freq, fz * self.temperature_freq]);
        let humidity_n = self
            .humidity_perlin
            .get([fx * self.humidity_freq, fz * self.humidity_freq]);

        let mut temperature = ((temp_n + 1.0) * 0.5).clamp(0.0, 1.0);
        let mut humidity = ((humidity_n + 1.0) * 0.5).clamp(0.0, 1.0);

        let height_norm = ((height - self.base_height) as f64 / 80.0).clamp(0.0, 1.0);
        temperature = (temperature - height_norm * 0.20).clamp(0.0, 1.0);
        humidity = (humidity * (1.0 - height_norm * 0.10)).clamp(0.0, 1.0);

        (temperature, humidity)
    }

    fn surface_biome_at(&self, x: i32, z: i32, height: i32) -> SurfaceBiome {
        if self.mode != WorldMode::Normal {
            return SurfaceBiome::Temperate;
        }
        let (temperature, humidity) = self.climate_at(x, z, height);
        if humidity <= DESERT_HUMIDITY_MAX && temperature <= GRAVEL_DESERT_TEMPERATURE_MAX {
            SurfaceBiome::GravelDesert
        } else if humidity <= DESERT_HUMIDITY_MAX && temperature >= SAND_DESERT_TEMPERATURE_MIN {
            SurfaceBiome::SandDesert
        } else {
            SurfaceBiome::Temperate
        }
    }

    fn cave_density_at(&self, x: i32, y: i32, z: i32) -> f64 {
        let fx = x as f64 + self.offset_x;
        let fz = z as f64 + self.offset_z;
        let fy = y as f64;
        let base = self
            .cave_perlin
            .get([
                fx * self.cave_freq,
                fy * self.cave_freq_y,
                fz * self.cave_freq,
            ])
            .abs();
        let detail = self
            .cave_detail_perlin
            .get([
                fx * self.cave_detail_freq,
                fy * self.cave_detail_freq_y,
                fz * self.cave_detail_freq,
            ])
            .abs();
        base + detail * 0.55
    }

    fn surface_cave_entrance_strength_at(&self, x: i32, z: i32) -> f64 {
        let fx = x as f64 + self.offset_x;
        let fz = z as f64 + self.offset_z;
        let n = self
            .cave_entrance_perlin
            .get([fx * self.cave_entrance_freq, fz * self.cave_entrance_freq]);
        smoothstep(0.34, 0.72, n)
    }

    fn should_carve_cave_at(&self, x: i32, y: i32, z: i32, height: i32) -> bool {
        if self.mode != WorldMode::Normal || y > height || y < CAVE_MIN_Y {
            return false;
        }

        let depth = height - y;
        let density = self.cave_density_at(x, y, z);

        if depth <= SURFACE_CAVE_DEPTH {
            let entrance = self.surface_cave_entrance_strength_at(x, z);
            if entrance <= 0.015 {
                return false;
            }
            if depth <= 1 && entrance < 0.36 {
                return false;
            }
            if depth <= 2 && entrance < 0.22 {
                return false;
            }
            let depth_t = 1.0 - (depth as f64 / SURFACE_CAVE_DEPTH as f64);
            let threshold = self.cave_threshold + entrance * self.surface_cave_boost * depth_t;
            return density < threshold;
        }

        let deep_t = ((depth - SURFACE_CAVE_DEPTH) as f64 / 48.0).clamp(0.0, 1.0);
        density < self.cave_threshold + deep_t * 0.03
    }

    fn has_tree_support_layers(&self, x: i32, z: i32, height: i32) -> bool {
        let min_y = height - (TREE_MIN_SUPPORT_LAYERS - 1);
        let mut y = min_y;
        while y <= height {
            if self.should_carve_cave_at(x, y, z, height) {
                return false;
            }
            y += 1;
        }
        true
    }

    pub fn block_id_at(&self, x: i32, y: i32, z: i32, height: i32) -> i8 {
        if !self.in_world_bounds(x, z) {
            return -1;
        }
        match self.mode {
            WorldMode::Normal => {
                if y > height {
                    -1
                } else {
                    let mut block_id = if y < height - 3 {
                        BLOCK_STONE as i8
                    } else {
                        match self.surface_biome_at(x, z, height) {
                            SurfaceBiome::Temperate => {
                                if y == height {
                                    BLOCK_GRASS as i8
                                } else {
                                    BLOCK_DIRT as i8
                                }
                            }
                            SurfaceBiome::GravelDesert => BLOCK_GRAVEL as i8,
                            SurfaceBiome::SandDesert => BLOCK_SAND as i8,
                        }
                    };
                    if self.should_carve_cave_at(x, y, z, height) {
                        block_id = -1;
                    }
                    if y == height && block_id < 0 && self.tree_at(x, z, height).is_some() {
                        return BLOCK_GRASS as i8;
                    }
                    block_id
                }
            }
            WorldMode::Flat | WorldMode::Grid => {
                if y > height {
                    -1
                } else if y == height {
                    BLOCK_GRASS as i8
                } else if y >= height - 2 {
                    BLOCK_DIRT as i8
                } else {
                    BLOCK_STONE as i8
                }
            }
        }
    }

    pub fn tree_at(&self, x: i32, z: i32, height: i32) -> Option<TreeSpec> {
        if self.mode != WorldMode::Normal {
            return None;
        }
        if !self.in_world_bounds(x, z) {
            return None;
        }
        if height < 0 {
            return None;
        }
        if !matches!(self.surface_biome_at(x, z, height), SurfaceBiome::Temperate) {
            return None;
        }
        if !self.has_tree_support_layers(x, z, height) {
            return None;
        }

        let cell_x = x.div_euclid(TREE_CELL_SIZE);
        let cell_z = z.div_euclid(TREE_CELL_SIZE);
        let cell_hash = hash2(self.seed ^ 0xB529_7A4D, cell_x, cell_z);
        let jitter_span = TREE_CELL_SIZE - TREE_CELL_MARGIN * 2;
        let local_x = TREE_CELL_MARGIN + (cell_hash % jitter_span as u32) as i32;
        let local_z = TREE_CELL_MARGIN + ((cell_hash >> 8) % jitter_span as u32) as i32;
        let tree_x = cell_x * TREE_CELL_SIZE + local_x;
        let tree_z = cell_z * TREE_CELL_SIZE + local_z;
        if x != tree_x || z != tree_z {
            return None;
        }

        let h = hash2(self.seed ^ 0x68E3_1DA4, x, z);
        if h % 100 >= TREE_CELL_SPAWN_CHANCE_PCT {
            return None;
        }
        let trunk_h = 4 + (h as i32 % 3);
        let leaf_r = 2 + ((h >> 3) as i32 & 1);
        Some(TreeSpec {
            base_y: height + 1,
            trunk_h,
            leaf_r,
        })
    }

    pub fn block_id_full_at(&self, x: i32, y: i32, z: i32) -> i8 {
        if !self.in_world_bounds(x, z) {
            return -1;
        }
        let height = self.height_at(x, z);
        let terrain = self.block_id_at(x, y, z, height);
        if terrain >= 0 {
            return terrain;
        }
        if self.mode == WorldMode::Grid
            && let Some(id) = grid_marker_block_id(x, y, z, height)
        {
            return id;
        }
        if self.mode != WorldMode::Normal {
            return -1;
        }

        // Trees are local structures with max leaf radius 3, so scan nearby centers.
        let max_leaf_r = 3;
        for tz in (z - max_leaf_r)..=(z + max_leaf_r) {
            for tx in (x - max_leaf_r)..=(x + max_leaf_r) {
                let th = self.height_at(tx, tz);
                let Some(tree) = self.tree_at(tx, tz, th) else {
                    continue;
                };

                let trunk_end = tree.base_y + tree.trunk_h;
                if x == tx && z == tz && y >= tree.base_y && y < trunk_end {
                    return BLOCK_LOG as i8;
                }

                let dy = y - trunk_end;
                if dy < -tree.leaf_r || dy > tree.leaf_r {
                    continue;
                }
                let dx = x - tx;
                let dz = z - tz;
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 <= tree.leaf_r * tree.leaf_r {
                    if dx == 0 && dz == 0 && y >= tree.base_y && y < trunk_end {
                        continue;
                    }
                    return BLOCK_LEAVES as i8;
                }
            }
        }
        -1
    }
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn derive_offsets(seed: u32) -> (f64, f64) {
    let mut x = seed as u64 ^ 0xA24BAED4963EE407;
    x ^= x >> 30;
    x = x.wrapping_mul(0x9E3779B97F4A7C15);
    let ox = ((x >> 11) & 0xFFFF) as f64;
    let mut y = seed as u64 ^ 0x9FB21C651E98DF25;
    y ^= y >> 27;
    y = y.wrapping_mul(0xBF58476D1CE4E5B9);
    let oz = ((y >> 11) & 0xFFFF) as f64;
    (ox, oz)
}

fn derive_world_id(seed: u32, mode: WorldMode) -> u64 {
    let mode_tag = match mode {
        WorldMode::Normal => 0_u64,
        WorldMode::Flat => 1_u64,
        WorldMode::Grid => 2_u64,
    };
    let mut x = (seed as u64
        ^ (mode_tag.wrapping_mul(0x9E3779B97F4A7C15))
        ^ WORLDGEN_LAYOUT_VERSION.wrapping_mul(0xD6E8FEB86659FD93))
    .wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn hash2(seed: u32, x: i32, z: i32) -> u32 {
    let mut h = seed ^ 0x9E3779B9;
    h = h.wrapping_mul(0x85EBCA6B) ^ (x as u32).wrapping_mul(0xC2B2AE35);
    h = h.rotate_left(13) ^ (z as u32).wrapping_mul(0x27D4EB2F);
    h
}

fn grid_marker_block_id(x: i32, y: i32, z: i32, height: i32) -> Option<i8> {
    if y != height + 2 {
        return None;
    }
    if !is_grid_marker_column(x, z) {
        return None;
    }

    let grid_x = x.div_euclid(4);
    let grid_z = z.div_euclid(4);
    let count = block_count().max(1) as i32;
    let idx = (grid_x * 31 + grid_z * 17).rem_euclid(count) as i8;
    Some(idx)
}

fn is_grid_marker_column(x: i32, z: i32) -> bool {
    x.rem_euclid(4) == 0 && z.rem_euclid(4) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trees_have_supported_base_layers_in_normal_mode() {
        let seeds = [1_u32, 42_u32, 777_u32];
        for &seed in &seeds {
            let world = WorldGen::new(seed, WorldMode::Normal);
            for z in -96..=96 {
                for x in -96..=96 {
                    let height = world.height_at(x, z);
                    if world.tree_at(x, z, height).is_some() {
                        let min_y = height - (TREE_MIN_SUPPORT_LAYERS - 1);
                        for y in min_y..=height {
                            assert!(
                                !world.should_carve_cave_at(x, y, z, height),
                                "floating tree support detected at seed={seed} x={x} z={z} y={y} height={height}"
                            );
                        }
                    }
                }
            }
        }
    }
}
