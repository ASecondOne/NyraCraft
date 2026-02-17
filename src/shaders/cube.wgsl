struct Uniforms {
    mvp: mat4x4<f32>,
    camera_pos: vec4<f32>,
    tile_misc: vec4<f32>,
    flags0: vec4<u32>,
    flags1: vec4<u32>,
    colormap_misc: vec4<f32>,
    item_misc: vec4<f32>,
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

@group(0) @binding(5)
var item_atlas_texture: texture_2d<f32>;

@group(0) @binding(6)
var item_atlas_sampler: sampler;

@group(0) @binding(7)
var sun_texture: texture_2d<f32>;

@group(0) @binding(8)
var sun_sampler: sampler;

struct VertexInputRaw {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tile: u32,
    @location(3) face: u32,
    @location(4) rotation: u32,
    @location(5) use_texture: u32,
    @location(6) transparent_mode: u32,
    @location(7) color: vec4<f32>,
};

struct VertexInputPacked {
    @location(0) position_packed: vec4<i32>,
    @location(1) uv_packed: vec2<i32>,
    @location(2) packed_info: vec2<u32>,
    @location(3) color: vec4<f32>,
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
    @location(8) grass_cm_uv: vec2<f32>,
};

fn emit_vertex_output(
    position: vec3<f32>,
    uv: vec2<f32>,
    tile: u32,
    face: u32,
    rotation: u32,
    use_texture: u32,
    transparent_mode: u32,
    color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.tile = tile;
    out.face = face;
    out.rotation = rotation;
    out.use_texture = use_texture;
    out.transparent_mode = transparent_mode;
    out.color = color;
    out.world_pos = position;
    out.grass_cm_uv = grass_colormap_uv(position);
    return out;
}

@vertex
fn vs_main(input: VertexInputRaw) -> VertexOutput {
    return emit_vertex_output(
        input.position,
        input.uv,
        input.tile,
        input.face,
        input.rotation,
        input.use_texture,
        input.transparent_mode,
        input.color,
    );
}

@vertex
fn vs_main_packed(input: VertexInputPacked) -> VertexOutput {
    let pos = vec3<f32>(
        f32(input.position_packed.x),
        f32(input.position_packed.y),
        f32(input.position_packed.z),
    );
    let uv = vec2<f32>(
        f32(input.uv_packed.x) / 256.0,
        f32(input.uv_packed.y) / 256.0,
    );
    let tile = input.packed_info.x;
    let flags = input.packed_info.y;
    let face = flags & 0xFu;
    let rotation = (flags >> 4u) & 0xFu;
    let use_texture = (flags >> 8u) & 0x1u;
    let transparent_mode = (flags >> 9u) & 0xFu;
    return emit_vertex_output(
        pos,
        uv,
        tile,
        face,
        rotation,
        use_texture,
        transparent_mode,
        input.color,
    );
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
    let sun_mode = 15u;
    let day_factor = uniforms.colormap_misc.y;

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
        let uv_tile_rot = rotate_uv(fract(input.uv), input.rotation);
        let block_tiles_x = uniforms.flags0.y;
        let block_tile_x = input.tile % block_tiles_x;
        let block_tile_y = input.tile / block_tiles_x;
        let block_uv_base = vec2<f32>(f32(block_tile_x), f32(block_tile_y)) * uniforms.tile_misc.xy;
        let block_uv = block_uv_base + uv_tile_rot * uniforms.tile_misc.xy;
        let item_atlas_flag = 8u;
        let sample_sun = input.transparent_mode == sun_mode;
        let sample_item_atlas = !sample_sun && input.transparent_mode >= item_atlas_flag;
        var transparent_mode = input.transparent_mode;
        var tex: vec4<f32>;
        if (sample_sun) {
            // Sun texture is not tiled in the main atlas.
            transparent_mode = 1u;
            tex = textureSample(sun_texture, sun_sampler, clamp(uv_rot, vec2<f32>(0.0), vec2<f32>(1.0)));
        } else if (sample_item_atlas) {
            transparent_mode = input.transparent_mode - item_atlas_flag;
            let item_tiles_x = max(u32(uniforms.item_misc.z), 1u);
            let item_tile_x = input.tile % item_tiles_x;
            let item_tile_y = input.tile / item_tiles_x;
            let item_uv_base = vec2<f32>(f32(item_tile_x), f32(item_tile_y)) * uniforms.item_misc.xy;
            let item_uv = item_uv_base + uv_tile_rot * uniforms.item_misc.xy;
            tex = textureSample(item_atlas_texture, item_atlas_sampler, item_uv);
        } else {
            tex = textureSample(atlas_texture, atlas_sampler, block_uv);
        }
        var tex_rgb = tex.rgb;
        var tex_a = tex.a;

        // Apply biome colormap on grass top and grass side (including hanging side overlay).
        let is_grass_top = input.face == 2u && input.tile == uniforms.flags1.y;
        let is_side_face = input.face == 0u || input.face == 1u || input.face == 4u || input.face == 5u;
        let is_grass_side = is_side_face && input.tile == uniforms.flags1.z;
        if (!sample_item_atlas && is_grass_side) {
            let overlay_tile = uniforms.flags1.w;
            let overlay_x = overlay_tile % block_tiles_x;
            let overlay_y = overlay_tile / block_tiles_x;
            let overlay_uv_base = vec2<f32>(f32(overlay_x), f32(overlay_y)) * uniforms.tile_misc.xy;
            let overlay_uv = overlay_uv_base + uv_tile_rot * uniforms.tile_misc.xy;
            let overlay = textureSample(atlas_texture, atlas_sampler, overlay_uv);
            tex_rgb = mix(tex_rgb, overlay.rgb, overlay.a);
            tex_a = max(tex_a, overlay.a);
        }

        if (!sample_item_atlas && !sample_sun && (is_grass_top || is_grass_side)) {
            let cm_uv = input.grass_cm_uv;
            let cm = textureSample(grass_colormap_texture, grass_colormap_sampler, cm_uv).rgb;
            // Keep texture identity while applying biome tint.
            tex_rgb = mix(tex_rgb, tex_rgb * cm, uniforms.colormap_misc.x);
        }

        if (transparent_mode == 1u) {
            if (tex_a < 0.05) {
                discard;
            }
            let tinted_rgb = tex_rgb * color.rgb;
            let tint_mix = clamp(tex_a, 0.0, 1.0);
            color = vec4<f32>(
                mix(tex_rgb, tinted_rgb, tint_mix),
                tex_a * color.a,
            );
        } else if (transparent_mode == 2u) {
            // RGB atlas fallback: treat near-white as transparent background.
            if (tex_rgb.r > 0.92 && tex_rgb.g > 0.92 && tex_rgb.b > 0.92) {
                discard;
            }
            color = vec4<f32>(tex_rgb * color.rgb, color.a);
        } else {
            color = vec4<f32>(tex_rgb, tex_a) * color;
        }
    }
    if (input.transparent_mode != sun_mode) {
        color = vec4<f32>(color.rgb * day_factor, color.a);
    }
    return color;
}
