#[derive(Clone, Copy)]
pub struct ColorThing {
    pub color_object: wgpu::Color,
}

impl ColorThing {
    pub fn new(r: f64, g: f64, b: f64) -> ColorThing {
        let a: f64 = 1.0;
        ColorThing {
            color_object: wgpu::Color { r, g, b, a },
        }
    }

    pub fn as_f32_rgba(&self) -> [f32; 4] {
        [
            self.color_object.r as f32,
            self.color_object.g as f32,
            self.color_object.b as f32,
            self.color_object.a as f32,
        ]
    }
}
