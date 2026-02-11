struct Uniforms {
    mvp: mat4x4<f32>,
    camera_pos: vec4<f32>,
    tile_misc: vec4<f32>,
    flags0: vec4<u32>,
    flags1: vec4<u32>,
    colormap_misc: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

@group(0) @binding(3)
var grass_colormap_texture: texture_2d<f32>;

@group(0) @binding(4)
var grass_colormap_sampler: sampler;

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
    @location(7) world_pos: vec3<f32>,
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
    out.world_pos = input.position;
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

fn face_normal(face: u32) -> vec3<f32> {
    switch(face) {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, -1.0); }
    }
}

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn biome_climate(world_pos: vec3<f32>) -> vec2<f32> {
    let xz = world_pos.xz;
    let height_n = clamp((world_pos.y - 12.0) / 72.0, 0.0, 1.0);

    let plains_forest = noise2(xz * 0.0025 + vec2<f32>(17.3, -4.7));
    let cliff_n = noise2(xz * 0.014 + vec2<f32>(-9.1, 22.6));

    var temperature: f32;
    var humidity: f32;

    // 4-biome style mapping:
    // plains (hot/wet), forest (mild/wet), mountains (cold/dry), cliffs (cold/wet)
    if (height_n > 0.68) {
        temperature = 0.22;
        humidity = 0.28;
    } else if (cliff_n > 0.82) {
        temperature = 0.30;
        humidity = 0.78;
    } else if (plains_forest < 0.45) {
        temperature = 0.80;
        humidity = 0.78;
    } else {
        temperature = 0.58;
        humidity = 0.86;
    }

    temperature = clamp(temperature - height_n * 0.22, 0.0, 1.0);
    humidity = clamp(humidity * (1.0 - height_n * 0.12), 0.0, 1.0);

    return vec2<f32>(temperature, humidity);
}

fn grass_colormap_uv(world_pos: vec3<f32>) -> vec2<f32> {
    let climate = biome_climate(world_pos);
    let temperature = climate.x;
    let humidity = climate.y;

    let h_eff = clamp(humidity * temperature, 0.0, 1.0);

    let u = clamp(temperature, 0.0, 1.0);
    let v = clamp(mix(u, 1.0, h_eff), 0.0, 1.0);
    return vec2<f32>(u, v);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (uniforms.flags1.x == 1u) {
        let view_dir = uniforms.camera_pos.xyz - input.world_pos;
        if (dot(face_normal(input.face), view_dir) <= 0.0) {
            discard;
        }
    }

    var color = input.color;
    if (uniforms.flags0.z == 1u) {
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
    if (uniforms.flags0.x == 1u && input.use_texture == 1u) {
        let uv_rot = rotate_uv(input.uv, input.rotation);
        let tiles_x = uniforms.flags0.y;
        let tile_x = input.tile % tiles_x;
        let tile_y = input.tile / tiles_x;
        let uv_base = vec2<f32>(f32(tile_x), f32(tile_y)) * uniforms.tile_misc.xy;
        let uv = uv_base + fract(uv_rot) * uniforms.tile_misc.xy;
        let tex = textureSample(atlas_texture, atlas_sampler, uv);
        var tex_rgb = tex.rgb;
        var tex_a = tex.a;

        // Apply biome colormap on grass top and grass side (including hanging side overlay).
        let is_grass_top = input.face == 2u && input.tile == uniforms.flags1.y;
        let is_side_face = input.face == 0u || input.face == 1u || input.face == 4u || input.face == 5u;
        let is_grass_side = is_side_face && input.tile == uniforms.flags1.z;
        if (is_grass_side) {
            let overlay_tile = uniforms.flags1.w;
            let overlay_x = overlay_tile % tiles_x;
            let overlay_y = overlay_tile / tiles_x;
            let overlay_uv_base = vec2<f32>(f32(overlay_x), f32(overlay_y)) * uniforms.tile_misc.xy;
            let overlay_uv = overlay_uv_base + fract(uv_rot) * uniforms.tile_misc.xy;
            let overlay = textureSample(atlas_texture, atlas_sampler, overlay_uv);
            tex_rgb = mix(tex_rgb, overlay.rgb, overlay.a);
            tex_a = max(tex_a, overlay.a);
        }

        if (is_grass_top || is_grass_side) {
            let cm_uv = grass_colormap_uv(input.world_pos);
            let cm = textureSample(grass_colormap_texture, grass_colormap_sampler, cm_uv).rgb;
            // Keep texture identity while applying biome tint.
            tex_rgb = mix(tex_rgb, tex_rgb * cm, uniforms.colormap_misc.x);
        }

        if (input.transparent_mode == 1u) {
            if (tex_a < 0.05) {
                discard;
            }
            let tinted_rgb = tex_rgb * color.rgb;
            let tint_mix = clamp(tex_a, 0.0, 1.0);
            color = vec4<f32>(
                mix(tex_rgb, tinted_rgb, tint_mix),
                tex_a * color.a,
            );
        } else if (input.transparent_mode == 2u) {
            // RGB atlas fallback: treat near-white as transparent background.
            if (tex_rgb.r > 0.92 && tex_rgb.g > 0.92 && tex_rgb.b > 0.92) {
                discard;
            }
            color = vec4<f32>(tex_rgb * color.rgb, color.a);
        } else {
            color = vec4<f32>(tex_rgb, tex_a) * color;
        }
    }
    return color;
}
