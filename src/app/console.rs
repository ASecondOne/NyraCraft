use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::app::controls::try_enable_mouse_look;
use crate::player::inventory::InventoryState;
use crate::world::blocks::{ITEM_DEFS, item_name_by_id, parse_item_id};

#[derive(Default)]
pub struct CommandConsoleState {
    pub open: bool,
    pub input: String,
}

fn sanitize_console_input(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_control()).collect()
}

pub fn append_console_input(console: &mut CommandConsoleState, text: &str) {
    let commit = sanitize_console_input(text);
    if commit.is_empty() {
        return;
    }
    console.input.push_str(&commit);
    if console.input.len() > 180 {
        console.input.truncate(180);
    }
}

pub fn is_console_open_shortcut(event: &KeyEvent) -> bool {
    if !matches!(event.state, ElementState::Pressed) {
        return false;
    }
    if let Some(text) = event.text.as_ref()
        && text.as_str() == "/"
    {
        return true;
    }
    matches!(
        event.physical_key,
        PhysicalKey::Code(KeyCode::Slash) | PhysicalKey::Code(KeyCode::KeyT)
    )
}

pub fn close_command_console(
    console: &mut CommandConsoleState,
    inventory_open: bool,
    window: &Window,
    mouse_enabled: &mut bool,
    last_cursor_pos: &mut Option<(f64, f64)>,
) {
    console.open = false;
    console.input.clear();
    window.set_ime_allowed(false);
    if !inventory_open {
        try_enable_mouse_look(window, mouse_enabled, last_cursor_pos);
    }
}

pub fn execute_and_close_command_console(
    console: &mut CommandConsoleState,
    inventory: &mut InventoryState,
    inventory_open: bool,
    window: &Window,
    mouse_enabled: &mut bool,
    last_cursor_pos: &mut Option<(f64, f64)>,
) {
    let command_text = console.input.trim().to_string();
    if !command_text.is_empty() {
        let feedback = execute_console_command(&command_text, inventory);
        println!("/{} -> {}", command_text.trim_start_matches('/'), feedback);
    }
    close_command_console(
        console,
        inventory_open,
        window,
        mouse_enabled,
        last_cursor_pos,
    );
}

pub fn execute_console_command(raw: &str, inventory: &mut InventoryState) -> String {
    let line = raw.trim().trim_start_matches('/');
    if line.is_empty() {
        return "empty command".to_string();
    }

    let mut parts = line.split_whitespace();
    let command = parts.next().unwrap_or_default().to_ascii_lowercase();
    match command.as_str() {
        "help" => "commands: give <item> [count], items".to_string(),
        "items" => ITEM_DEFS
            .iter()
            .map(|def| format!("{}:{}", def.item_id, def.name))
            .collect::<Vec<_>>()
            .join(", "),
        "give" => {
            let Some(block_text) = parts.next() else {
                return "usage: give <item> [count]".to_string();
            };
            let Some(item_id) = parse_item_id(block_text) else {
                return format!(
                    "unknown item `{block_text}` (stone, dirt, grass, log, leaves, apple, stick)"
                );
            };
            let count = if let Some(count_text) = parts.next() {
                let Ok(parsed) = count_text.parse::<u16>() else {
                    return format!("invalid count `{count_text}`");
                };
                parsed.clamp(1, 4096)
            } else {
                64
            };
            let remaining = inventory.add_item(item_id, count);
            let added = count.saturating_sub(remaining);
            if added == 0 {
                format!(
                    "inventory full, could not give {}",
                    item_name_by_id(item_id)
                )
            } else if remaining > 0 {
                format!(
                    "gave {} x{} ({} could not fit)",
                    item_name_by_id(item_id),
                    added,
                    remaining
                )
            } else {
                format!("gave {} x{}", item_name_by_id(item_id), added)
            }
        }
        _ => format!("unknown command `{command}` (try `help`)"),
    }
}
