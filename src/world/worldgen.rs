use noise::{NoiseFn, Perlin};
use crate::world::blocks::{BLOCK_DIRT, BLOCK_GRASS, BLOCK_STONE};

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
    offset_x: f64,
    offset_z: f64,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let perlin = Perlin::new(seed);
        let mountain_perlin = Perlin::new(seed.wrapping_add(1));
        let cliffs_perlin = Perlin::new(seed.wrapping_add(2));
        let biome_perlin = Perlin::new(seed.wrapping_add(3));
        let world_id = derive_world_id(seed);
        let (offset_x, offset_z) = derive_offsets(seed);
        Self {
            seed,
            world_id,
            base_height: 0,
            plains_amp: 2.0,
            plains_freq: 0.008,
            perlin,
            mountain_amp: 28.0,
            mountain_freq: 0.05,
            mountain_perlin,
            cliffs_amp: 18.0,
            cliffs_freq: 0.025,
            cliffs_depth: 42.0,
            cliffs_perlin,
            biome_freq: 0.002,
            biome_perlin,
            offset_x,
            offset_z,
        }
    }

    pub fn height_at(&self, x: i32, z: i32) -> i32 {
        let fx = x as f64 + self.offset_x;
        let fz = z as f64 + self.offset_z;

        let biome_n = self
            .biome_perlin
            .get([fx * self.biome_freq, fz * self.biome_freq]);
        // Bias strongly toward plains: only extreme biome values become mountains/cliffs.
        let mountain_w = smoothstep(0.75, 0.95, biome_n);
        let cliffs_w = smoothstep(0.6, 0.85, -biome_n);
        let mut plains_w = 1.0 - mountain_w.max(cliffs_w);
        if plains_w < 0.0 {
            plains_w = 0.0;
        }
        let (mountain_w, cliffs_w, plains_w) = if plains_w > 0.5 {
            (0.0, 0.0, 1.0)
        } else {
            let sum = mountain_w + cliffs_w + plains_w;
            if sum > 0.0 {
                (mountain_w / sum, cliffs_w / sum, plains_w / sum)
            } else {
                (0.0, 0.0, 1.0)
            }
        };

        let plains_n = self
            .perlin
            .get([fx * self.plains_freq, fz * self.plains_freq]);
        let plains_h = self.base_height as f64 + plains_n.clamp(-1.0, 1.0) * self.plains_amp;
        let plains_h = plains_h.round();

        let mountain_n = self
            .mountain_perlin
            .get([fx * self.mountain_freq, fz * self.mountain_freq]);
        let ridged = (1.0 - mountain_n.abs()).powf(2.2);
        let mountain_h = self.base_height as f64 + ridged * self.mountain_amp;

        let cliffs_n = self
            .cliffs_perlin
            .get([fx * self.cliffs_freq, fz * self.cliffs_freq]);
        let cliffs_ridged = (1.0 - cliffs_n.abs()).powf(1.5);
        let cliffs_h = self.base_height as f64
            + cliffs_n * self.cliffs_amp
            - cliffs_ridged * self.cliffs_depth;

        let height = plains_h * plains_w + mountain_h * mountain_w + cliffs_h * cliffs_w;
        height as i32
    }

    pub fn block_id_at(&self, x: i32, y: i32, z: i32, height: i32) -> i8 {
        let _ = (x, z);
        if y > height {
            -1
        } else if y == height {
            BLOCK_GRASS as i8
        } else if y >= height - 3 {
            BLOCK_DIRT as i8
        } else {
            BLOCK_STONE as i8
        }
    }

    pub fn tree_at(&self, x: i32, z: i32, height: i32) -> Option<TreeSpec> {
        if height < 0 {
            return None;
        }
        let h = hash2(self.seed, x, z);
        if h % 100 >= 5 {
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

    pub fn is_cave(&self, x: i32, y: i32, z: i32, height: i32) -> bool {
        let _ = (x, y, z, height);
        false
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

fn derive_world_id(seed: u32) -> u64 {
    let mut x = seed as u64 + 0x9E3779B97F4A7C15;
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
