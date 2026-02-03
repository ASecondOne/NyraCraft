struct Uniforms {
    mvp: mat4x4<f32>,
    use_texture: u32,
    tiles_x: u32,
    debug_faces: u32,
    debug_chunks: u32,
    tile_uv_size: vec2<f32>,
    chunk_size: f32,
    _pad0: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
