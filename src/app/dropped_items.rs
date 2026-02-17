use glam::{IVec3, Vec3};

use crate::player::inventory::{InventoryState, ItemStack};
use crate::player::{EditedBlocks, block_id_with_edits};
use crate::render::gpu::DroppedItemRender;
use crate::world::blocks::{block_drop_rolls, item_max_stack_size};
use crate::world::worldgen::WorldGen;

const DROPPED_ITEM_DESPAWN_SECS: f32 = 5.0 * 60.0;

#[derive(Clone, Copy)]
pub struct DroppedItem {
    position: Vec3,
    velocity: Vec3,
    stack: ItemStack,
    pickup_delay: f32,
    spin_phase: f32,
    despawn_timer: f32,
}

fn spawn_dropped_item(
    dropped_items: &mut Vec<DroppedItem>,
    position: Vec3,
    velocity: Vec3,
    stack: ItemStack,
    pickup_delay: f32,
) {
    dropped_items.push(DroppedItem {
        position,
        velocity,
        stack,
        pickup_delay,
        spin_phase: rand::random::<f32>() * std::f32::consts::TAU,
        despawn_timer: DROPPED_ITEM_DESPAWN_SECS,
    });
}

fn drop_cell(position: Vec3) -> IVec3 {
    IVec3::new(
        position.x.floor() as i32,
        (position.y - 0.28).floor() as i32,
        position.z.floor() as i32,
    )
}

fn drop_center_cell(position: Vec3) -> IVec3 {
    IVec3::new(
        position.x.floor() as i32,
        position.y.floor() as i32,
        position.z.floor() as i32,
    )
}

fn is_solid(world_gen: &WorldGen, edited_blocks: &EditedBlocks, cell: IVec3) -> bool {
    block_id_with_edits(world_gen, edited_blocks, cell.x, cell.y, cell.z) >= 0
}

fn first_air_y(
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    x: i32,
    mut y: i32,
    z: i32,
) -> Option<i32> {
    if !world_gen.in_world_bounds(x, z) {
        return None;
    }
    y = y.max(world_gen.highest_solid_y_at(x, z) + 1);
    for _ in 0..512 {
        if block_id_with_edits(world_gen, edited_blocks, x, y, z) < 0 {
            return Some(y);
        }
        y += 1;
    }
    None
}

fn relocate_drop_away_from_obstacle(
    drop: &mut DroppedItem,
    obstacle: IVec3,
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
) -> bool {
    let candidates = [
        (obstacle.x, obstacle.z),     // top of placed/obstructing block
        (obstacle.x + 1, obstacle.z), // right
        (obstacle.x - 1, obstacle.z), // left
    ];

    for (x, z) in candidates {
        let start_y = if x == obstacle.x && z == obstacle.z {
            obstacle.y + 1
        } else {
            obstacle.y
        };
        let Some(target_y) = first_air_y(world_gen, edited_blocks, x, start_y, z) else {
            continue;
        };
        drop.position = Vec3::new(x as f32 + 0.5, target_y as f32 + 0.66, z as f32 + 0.5);
        drop.velocity.y = drop.velocity.y.max(0.0);
        drop.velocity.x *= 0.5;
        drop.velocity.z *= 0.5;
        return true;
    }
    false
}

pub fn spawn_block_drop(dropped_items: &mut Vec<DroppedItem>, block: IVec3, block_id: i8) {
    for (item_id, count) in block_drop_rolls(block_id) {
        spawn_item_drop(dropped_items, block, item_id, count, 1.0);
    }
}

fn spawn_item_drop(
    dropped_items: &mut Vec<DroppedItem>,
    block: IVec3,
    item_id: i8,
    count: u8,
    speed_scale: f32,
) {
    if count == 0 {
        return;
    }
    let jitter_x = (rand::random::<f32>() - 0.5) * 0.32;
    let jitter_z = (rand::random::<f32>() - 0.5) * 0.32;
    spawn_dropped_item(
        dropped_items,
        Vec3::new(
            block.x as f32 + 0.5 + jitter_x,
            block.y as f32 + 0.66,
            block.z as f32 + 0.5 + jitter_z,
        ),
        Vec3::new(
            jitter_x * (2.2 * speed_scale),
            1.9 + rand::random::<f32>() * 1.0 * speed_scale,
            jitter_z * (2.2 * speed_scale),
        ),
        ItemStack::new(item_id, count),
        0.14,
    );
}

pub fn throw_hotbar_item(
    dropped_items: &mut Vec<DroppedItem>,
    player_position: Vec3,
    player_height: f32,
    forward: Vec3,
    stack: ItemStack,
) {
    if stack.count == 0 {
        return;
    }
    let dir = if forward.length_squared() > 0.0001 {
        forward.normalize()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    spawn_dropped_item(
        dropped_items,
        player_position + Vec3::new(0.0, player_height * 0.72, 0.0) + dir * 0.55,
        dir * 7.4 + Vec3::new(0.0, 2.4, 0.0),
        stack,
        0.24,
    );
}

pub fn nudge_items_from_placed_block(
    dropped_items: &mut [DroppedItem],
    placed_block: IVec3,
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
) {
    for drop in dropped_items {
        let cell = drop_cell(drop.position);
        if cell == placed_block {
            let _ = relocate_drop_away_from_obstacle(drop, placed_block, world_gen, edited_blocks);
        }
    }
}

fn merge_stacks_in_same_block(dropped_items: &mut Vec<DroppedItem>) {
    let mut i = 0usize;
    while i < dropped_items.len() {
        let cell_i = drop_cell(dropped_items[i].position);
        let block_i = dropped_items[i].stack.block_id;
        let mut j = i + 1;
        while j < dropped_items.len() {
            let same_kind = dropped_items[j].stack.block_id == block_i;
            let same_durability = dropped_items[j].stack.durability == dropped_items[i].stack.durability;
            let same_cell = drop_cell(dropped_items[j].position) == cell_i;
            if !same_kind || !same_durability || !same_cell {
                j += 1;
                continue;
            }

            let free = item_max_stack_size(block_i).saturating_sub(dropped_items[i].stack.count);
            if free > 0 {
                let moved = free.min(dropped_items[j].stack.count);
                dropped_items[i].stack.count += moved;
                dropped_items[j].stack.count -= moved;
                dropped_items[i].pickup_delay = dropped_items[i]
                    .pickup_delay
                    .min(dropped_items[j].pickup_delay);
                dropped_items[i].despawn_timer = dropped_items[i]
                    .despawn_timer
                    .min(dropped_items[j].despawn_timer);
            }

            if dropped_items[j].stack.count == 0 {
                dropped_items.swap_remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

pub fn update_dropped_items(
    dropped_items: &mut Vec<DroppedItem>,
    dt: f32,
    player_position: Vec3,
    player_height: f32,
    world_gen: &WorldGen,
    edited_blocks: &EditedBlocks,
    inventory: &mut InventoryState,
) {
    let player_pickup_center = player_position + Vec3::new(0.0, player_height * 0.5, 0.0);
    let mut i = 0usize;
    while i < dropped_items.len() {
        let mut remove = false;
        {
            let drop = &mut dropped_items[i];
            drop.despawn_timer -= dt;
            if drop.despawn_timer <= 0.0 {
                remove = true;
            }
            if remove {
                // Let expired drops be removed without doing extra work.
                continue;
            }
            drop.pickup_delay = (drop.pickup_delay - dt).max(0.0);
            drop.velocity.y -= 18.0 * dt;
            let drag = (1.0 - dt * 3.2).clamp(0.0, 1.0);
            drop.velocity.x *= drag;
            drop.velocity.z *= drag;

            let mut next = drop.position + drop.velocity * dt;
            let below_x = next.x.floor() as i32;
            let below_y = (next.y - 0.28).floor() as i32;
            let below_z = next.z.floor() as i32;
            if block_id_with_edits(world_gen, edited_blocks, below_x, below_y, below_z) >= 0
                && drop.velocity.y <= 0.0
            {
                next.y = below_y as f32 + 1.28;
                drop.velocity.y = 0.0;
            }
            drop.position = next;
            if drop.position.y < -128.0 {
                remove = true;
            }

            if !remove {
                let obstacle = drop_center_cell(drop.position);
                if is_solid(world_gen, edited_blocks, obstacle) {
                    let _ =
                        relocate_drop_away_from_obstacle(drop, obstacle, world_gen, edited_blocks);
                }
            }

            if !remove
                && drop.pickup_delay <= 0.0
                && (drop.position - player_pickup_center).length_squared() <= 3.24
            {
                if let Some(remaining) = inventory.add_item_stack(drop.stack) {
                    drop.stack = remaining;
                    drop.pickup_delay = 0.12;
                } else {
                    remove = true;
                }
            }
        }
        if remove {
            dropped_items.swap_remove(i);
        } else {
            i += 1;
        }
    }

    if dropped_items.len() > 1 {
        merge_stacks_in_same_block(dropped_items);
    }
}

pub fn build_dropped_item_render_data(
    dropped_items: &[DroppedItem],
    render_time: f32,
) -> Vec<DroppedItemRender> {
    dropped_items
        .iter()
        .map(|drop| {
            let spin = render_time * 2.4 + drop.spin_phase;
            let bob = (render_time * 3.2 + drop.spin_phase).sin() * 0.06;
            DroppedItemRender {
                position: drop.position + Vec3::new(0.0, bob, 0.0),
                block_id: drop.stack.block_id,
                spin_y: spin,
                tilt_z: 0.28,
            }
        })
        .collect()
}
