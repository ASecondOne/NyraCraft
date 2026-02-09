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

@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tile: u32,
    @location(3) face: u32,
    @location(4) rotation: u32,
    @location(5) use_texture: u32,
    @location(6) transparent_mode: u32,
    @location(7) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tile: u32,
    @location(2) face: u32,
    @location(3) rotation: u32,
    @location(4) use_texture: u32,
    @location(5) transparent_mode: u32,
    @location(6) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
    out.uv = input.uv;
    out.tile = input.tile;
    out.face = input.face;
    out.rotation = input.rotation;
    out.use_texture = input.use_texture;
    out.transparent_mode = input.transparent_mode;
    out.color = input.color;
    return out;
}

fn rotate_uv(uv: vec2<f32>, rot: u32) -> vec2<f32> {
    switch(rot & 3u) {
        case 0u: { return uv; }
        case 1u: { return vec2<f32>(1.0 - uv.y, uv.x); }
        case 2u: { return vec2<f32>(1.0 - uv.x, 1.0 - uv.y); }
        default: { return vec2<f32>(uv.y, 1.0 - uv.x); }
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    if (uniforms.debug_faces == 1u) {
        switch(input.face) {
            case 0u: { color = vec4<f32>(1.0, 0.2, 0.2, 1.0); }
            case 1u: { color = vec4<f32>(0.2, 1.0, 0.2, 1.0); }
            case 2u: { color = vec4<f32>(0.2, 0.2, 1.0, 1.0); }
            case 3u: { color = vec4<f32>(1.0, 1.0, 0.2, 1.0); }
            case 4u: { color = vec4<f32>(1.0, 0.2, 1.0, 1.0); }
            default: { color = vec4<f32>(0.2, 1.0, 1.0, 1.0); }
        }
        return color;
    }
    if (uniforms.use_texture == 1u && input.use_texture == 1u) {
        let uv_rot = rotate_uv(input.uv, input.rotation);
        let tiles_x = uniforms.tiles_x;
        let tile_x = input.tile % tiles_x;
        let tile_y = input.tile / tiles_x;
        let uv_base = vec2<f32>(f32(tile_x), f32(tile_y)) * uniforms.tile_uv_size;
        let uv = uv_base + fract(uv_rot) * uniforms.tile_uv_size;
        let tex = textureSample(atlas_texture, atlas_sampler, uv);
        if (input.transparent_mode == 1u) {
            if (tex.a < 0.05) {
                discard;
            }
            let tinted_rgb = tex.rgb * color.rgb;
            let tint_mix = clamp(tex.a, 0.0, 1.0);
            color = vec4<f32>(
                mix(tex.rgb, tinted_rgb, tint_mix),
                tex.a * color.a,
            );
        } else if (input.transparent_mode == 2u) {
            // RGB atlas fallback: treat near-white as transparent background.
            if (tex.r > 0.92 && tex.g > 0.92 && tex.b > 0.92) {
                discard;
            }
            color = vec4<f32>(tex.rgb * color.rgb, color.a);
        } else {
            color = tex * color;
        }
    }
    return color;
}
