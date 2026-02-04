use noise::{NoiseFn, Perlin};

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    pub world_id: u64,
    base_height: i32,
    amplitude: f64,
    frequency: f64,
    perlin: Perlin,
    mountain_amp: f64,
    mountain_freq: f64,
    mountain_perlin: Perlin,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let perlin = Perlin::new(seed);
        let mountain_perlin = Perlin::new(seed.wrapping_add(1));
        let world_id = derive_world_id(seed);
        Self {
            seed,
            world_id,
            base_height: 0,
            amplitude: 12.0,
            frequency: 0.04,
            perlin,
            mountain_amp: 24.0,
            mountain_freq: 0.015,
            mountain_perlin,
        }
    }

    pub fn with_height(mut self, base: i32, amplitude: f64, frequency: f64) -> Self {
        self.base_height = base;
        self.amplitude = amplitude;
        self.frequency = frequency;
        self
    }

    pub fn with_mountains(mut self, amplitude: f64, frequency: f64) -> Self {
        self.mountain_amp = amplitude;
        self.mountain_freq = frequency;
        self
    }

    pub fn height_at(&self, x: i32, z: i32) -> i32 {
        let base = self.perlin.get([x as f64 * self.frequency, z as f64 * self.frequency]);
        let mountain = self
            .mountain_perlin
            .get([x as f64 * self.mountain_freq, z as f64 * self.mountain_freq]);
        let mountain = mountain.max(0.0) * self.mountain_amp;
        self.base_height + (base * self.amplitude + mountain) as i32
    }

    pub fn block_id_for_height(&self, y: i32, height: i32) -> i8 {
        if y > height {
            -1
        } else if y == height {
            2
        } else if y >= height - 3 {
            1
        } else {
            0
        }
    }

    pub fn is_cave(&self, x: i32, y: i32, z: i32, height: i32) -> bool {
        let _ = (x, y, z, height);
        false
    }

}

fn derive_world_id(seed: u32) -> u64 {
    let mut x = seed as u64 + 0x9E3779B97F4A7C15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
