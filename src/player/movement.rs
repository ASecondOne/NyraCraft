use glam::Vec3;

use crate::physics::collision::collides;

pub struct PlayerState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
}

pub struct PlayerConfig {
    pub height: f32,
    pub radius: f32,
    pub eye_height: f32,
    pub move_speed: f32,
    pub jump_speed: f32,
    pub gravity: f32,
}

pub struct PlayerInput {
    pub move_forward: bool,
    pub move_back: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub move_up: bool,
    pub move_down: bool,
    pub jump: bool,
    pub fly_mode: bool,
}

pub fn update_player<F>(
    state: &mut PlayerState,
    input: &mut PlayerInput,
    camera_forward: Vec3,
    camera_up: Vec3,
    dt: f32,
    config: &PlayerConfig,
    is_solid: &F,
) -> Vec3
where
    F: Fn(i32, i32, i32) -> bool,
{
    let forward = camera_forward.normalize();
    let forward_flat = Vec3::new(forward.x, 0.0, forward.z);
    let forward_flat = if forward_flat.length_squared() > 0.0 {
        forward_flat.normalize()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    let right = forward_flat.cross(camera_up).normalize();

    let mut move_dir = Vec3::ZERO;
    if input.move_forward {
        move_dir += forward_flat;
    }
    if input.move_back {
        move_dir -= forward_flat;
    }
    if input.move_right {
        move_dir += right;
    }
    if input.move_left {
        move_dir -= right;
    }
    if input.fly_mode {
        if input.move_up {
            move_dir += camera_up;
        }
        if input.move_down {
            move_dir -= camera_up;
        }
    }

    let move_dir = if move_dir.length_squared() > 0.0 {
        move_dir.normalize()
    } else {
        Vec3::ZERO
    };

    state.velocity.x = move_dir.x * config.move_speed;
    state.velocity.z = move_dir.z * config.move_speed;

    if input.fly_mode {
        state.velocity.y = move_dir.y * config.move_speed;
        state.grounded = false;
        input.jump = false;
    } else {
        if input.jump && state.grounded {
            state.velocity.y = config.jump_speed;
            state.grounded = false;
        }
        input.jump = false;
        state.velocity.y += config.gravity * dt;
    }

    let mut new_pos = state.position;
    // X axis
    new_pos.x += state.velocity.x * dt;
    if collides(new_pos, config.height, config.radius, is_solid) {
        if state.velocity.x > 0.0 {
            let max_x = new_pos.x + config.radius;
            let block_x = max_x.floor() as i32;
            new_pos.x = block_x as f32 - config.radius - 0.001;
        } else if state.velocity.x < 0.0 {
            let min_x = new_pos.x - config.radius;
            let block_x = min_x.floor() as i32;
            new_pos.x = block_x as f32 + 1.0 + config.radius + 0.001;
        }
        state.velocity.x = 0.0;
    }
    // Z axis
    new_pos.z += state.velocity.z * dt;
    if collides(new_pos, config.height, config.radius, is_solid) {
        if state.velocity.z > 0.0 {
            let max_z = new_pos.z + config.radius;
            let block_z = max_z.floor() as i32;
            new_pos.z = block_z as f32 - config.radius - 0.001;
        } else if state.velocity.z < 0.0 {
            let min_z = new_pos.z - config.radius;
            let block_z = min_z.floor() as i32;
            new_pos.z = block_z as f32 + 1.0 + config.radius + 0.001;
        }
        state.velocity.z = 0.0;
    }
    // Y axis
    new_pos.y += state.velocity.y * dt;
    if collides(new_pos, config.height, config.radius, is_solid) {
        if state.velocity.y < 0.0 {
            let min_y = new_pos.y;
            let block_y = min_y.floor() as i32;
            new_pos.y = block_y as f32 + 1.0 + 0.001;
            state.grounded = true;
        } else if state.velocity.y > 0.0 {
            let max_y = new_pos.y + config.height;
            let block_y = max_y.floor() as i32;
            new_pos.y = block_y as f32 - config.height - 0.001;
        }
        state.velocity.y = 0.0;
    } else {
        state.grounded = false;
    }

    state.position = new_pos;
    state.position + Vec3::new(0.0, config.eye_height, 0.0)
}
