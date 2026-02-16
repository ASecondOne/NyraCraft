struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) use_texture: f32,
    @location(4) atlas_select: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) use_texture: f32,
    @location(3) atlas_select: f32,
};

@group(0) @binding(0) var item_tex: texture_2d<f32>;
@group(0) @binding(1) var item_sampler: sampler;
@group(0) @binding(2) var block_tex: texture_2d<f32>;
@group(0) @binding(3) var block_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(input.position, 0.0, 1.0);
    out.color = input.color;
    out.uv = input.uv;
    out.use_texture = input.use_texture;
    out.atlas_select = input.atlas_select;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.use_texture > 0.5 {
        var texel: vec4<f32>;
        if input.atlas_select > 0.5 {
            texel = textureSample(block_tex, block_sampler, input.uv);
        } else {
            texel = textureSample(item_tex, item_sampler, input.uv);
        }
        return texel * input.color;
    }
    return input.color;
}
