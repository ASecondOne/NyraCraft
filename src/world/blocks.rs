use crate::colorthing::ColorThing;
use crate::render::BlockTexture;

pub const DEFAULT_TILES_X: u32 = 16;

pub fn default_blocks(tiles_x: u32) -> Vec<BlockTexture> {
    let stone = BlockTexture::solid(
        tile_index_1based(2, 1, tiles_x),
        ColorThing::new(1.0, 1.0, 1.0),
    );
    let dirt = BlockTexture::solid(
        tile_index_1based(3, 1, tiles_x),
        ColorThing::new(1.0, 1.0, 1.0),
    );
    let grass = BlockTexture {
        colors: [
            ColorThing::new(0.62, 0.82, 0.35), // +X
            ColorThing::new(0.62, 0.82, 0.35), // -X
            ColorThing::new(0.62, 0.82, 0.35), // +Y (top)
            ColorThing::new(1.0, 1.0, 1.0),    // -Y (bottom)
            ColorThing::new(0.62, 0.82, 0.35), // +Z
            ColorThing::new(0.62, 0.82, 0.35), // -Z
        ],
        tiles: [
            tile_index_1based(4, 1, tiles_x), // sides
            tile_index_1based(4, 1, tiles_x),
            tile_index_1based(9, 3, tiles_x), // top (tinted)
            tile_index_1based(3, 1, tiles_x), // bottom
            tile_index_1based(4, 1, tiles_x),
            tile_index_1based(4, 1, tiles_x),
        ],
        rotations: [3, 3, 0, 0, 0, 0],
    };

    vec![stone, dirt, grass]
}

fn tile_index_1based(x: u32, y: u32, tiles_x: u32) -> u32 {
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    x0 + y0 * tiles_x
}
