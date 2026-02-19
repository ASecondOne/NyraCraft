use glam::Vec3;
use winit::window::{CursorGrabMode, Window};

use crate::player::{Camera, PlayerInput};

const KEYBIND_LINES: [&str; 17] = [
    "--- NyraCraft Keybinds ---",
    "Movement: W/A/S/D | Jump: Space",
    "Sprint: Shift | Sneak: C | Descend (fly): C",
    "Pause menu: Escape (return to game / quit)",
    "Sneak edge-guard: C prevents walking off ledges",
    "Mouse Left: Break block | Mouse Right: Place block / use crafting table",
    "Mine blocks: Hold Left Mouse (block hardness enabled)",
    "Hotbar: 1-9 select slot | Mouse Wheel cycles slots",
    "Drop held item: Q",
    "Inventory: E to toggle, shift-click quick move, right-drag split stacks",
    "Chat: / or T to open | Enter to send command/text",
    "F1: Toggle this keybind panel",
    "F3: Toggle debug stats panel",
    "F1 + F : Toggle face debug colors",
    "F1 + W : Toggle chunk wireframe",
    "F1 + P/V/R/M/X : Stream pause / render pause / reload / fly / fullscreen",
    "F1 + D : Toggle debug stats panel",
];

pub fn keybind_lines() -> &'static [&'static str] {
    &KEYBIND_LINES
}

pub fn try_enable_mouse_look(
    window: &Window,
    mouse_enabled: &mut bool,
    last_cursor_pos: &mut Option<(f64, f64)>,
) {
    if window.set_cursor_grab(CursorGrabMode::Locked).is_ok()
        || window.set_cursor_grab(CursorGrabMode::Confined).is_ok()
    {
        window.set_cursor_visible(false);
        *mouse_enabled = true;
        *last_cursor_pos = None;
    }
}

pub fn disable_mouse_look(
    window: &Window,
    mouse_enabled: &mut bool,
    last_cursor_pos: &mut Option<(f64, f64)>,
) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
    *mouse_enabled = false;
    *last_cursor_pos = None;
}

pub fn clear_movement_input(input: &mut PlayerInput) {
    input.move_forward = false;
    input.move_back = false;
    input.move_left = false;
    input.move_right = false;
    input.move_up = false;
    input.move_down = false;
    input.jump = false;
    input.sprint = false;
    input.sneak = false;
}

pub fn apply_look_delta(
    camera: &mut Camera,
    yaw: &mut f32,
    pitch: &mut f32,
    dx: f32,
    dy: f32,
    mouse_sensitivity: f32,
) {
    let dx = dx.clamp(-100.0, 100.0);
    let dy = dy.clamp(-100.0, 100.0);
    *yaw += dx * mouse_sensitivity;
    *pitch -= dy * mouse_sensitivity;

    let max_pitch = 1.5533f32;
    *pitch = (*pitch).clamp(-max_pitch, max_pitch);

    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    camera.forward = Vec3::new(cy * cp, sp, sy * cp).normalize();
}
