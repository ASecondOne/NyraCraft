use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::app::controls::try_enable_mouse_look;
use crate::app::logger::log_info;
use crate::player::inventory::InventoryState;
use crate::world::blocks::{all_item_defs, item_name_by_id, parse_item_id};

const MAX_INPUT_CHARS: usize = 180;
const MAX_CHAT_LINES: usize = 10;
const MAX_CHAT_LINE_CHARS: usize = 180;

#[derive(Default)]
pub struct CommandConsoleState {
    pub open: bool,
    pub input: String,
    pub chat_lines: Vec<String>,
}

fn sanitize_console_input(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_control()).collect()
}

fn truncate_to_char_limit(text: &mut String, max_chars: usize) {
    if text.chars().count() <= max_chars {
        return;
    }
    *text = text.chars().take(max_chars).collect();
}

pub fn append_console_input(console: &mut CommandConsoleState, text: &str) {
    let commit = sanitize_console_input(text);
    if commit.is_empty() {
        return;
    }
    console.input.push_str(&commit);
    truncate_to_char_limit(&mut console.input, MAX_INPUT_CHARS);
}

pub fn push_chat_line(console: &mut CommandConsoleState, line: &str) {
    let mut sanitized = sanitize_console_input(line);
    sanitized = sanitized.trim().to_string();
    if sanitized.is_empty() {
        return;
    }
    truncate_to_char_limit(&mut sanitized, MAX_CHAT_LINE_CHARS);
    console.chat_lines.push(sanitized);
    if console.chat_lines.len() > MAX_CHAT_LINES {
        let overflow = console.chat_lines.len() - MAX_CHAT_LINES;
        console.chat_lines.drain(0..overflow);
    }
}

fn execute_chat_line<F>(
    raw_line: &str,
    inventory: &mut InventoryState,
    command_handler: &mut F,
) -> Option<String>
where
    F: FnMut(&str) -> Option<String>,
{
    let mut line = sanitize_console_input(raw_line);
    line = line.trim().to_string();
    if line.is_empty() {
        return None;
    }
    truncate_to_char_limit(&mut line, MAX_CHAT_LINE_CHARS);
    if line.starts_with('/') {
        let feedback = execute_console_command_with_handler(&line, inventory, command_handler);
        let command = line.trim_start_matches('/').trim();
        log_info(format!("command /{} -> {}", command, feedback));
        Some(format!("/{command} -> {feedback}"))
    } else {
        Some(line)
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

pub fn execute_and_close_command_console<F>(
    console: &mut CommandConsoleState,
    inventory: &mut InventoryState,
    inventory_open: bool,
    window: &Window,
    mouse_enabled: &mut bool,
    last_cursor_pos: &mut Option<(f64, f64)>,
    mut command_handler: F,
) -> bool
where
    F: FnMut(&str) -> Option<String>,
{
    let mut submitted = false;
    if let Some(message) = execute_chat_line(&console.input, inventory, &mut command_handler) {
        push_chat_line(console, &message);
        submitted = true;
    }
    close_command_console(
        console,
        inventory_open,
        window,
        mouse_enabled,
        last_cursor_pos,
    );
    submitted
}

#[allow(dead_code)]
pub fn execute_console_command(raw: &str, inventory: &mut InventoryState) -> String {
    fn no_external_command(_: &str) -> Option<String> {
        None
    }
    let mut no_handler = no_external_command;
    execute_console_command_with_handler(raw, inventory, &mut no_handler)
}

fn execute_console_command_with_handler<F>(
    raw: &str,
    inventory: &mut InventoryState,
    command_handler: &mut F,
) -> String
where
    F: FnMut(&str) -> Option<String>,
{
    let line = raw.trim().trim_start_matches('/');
    if line.is_empty() {
        return "empty command".to_string();
    }

    let mut parts = line.split_whitespace();
    let command = parts.next().unwrap_or_default().to_ascii_lowercase();
    match command.as_str() {
        "help" => {
            "commands: give <item> [count], items, time set <day|night|morning|none>, tp <x> <y> <z> | ids: 1:<block_id> (block), 2:<item_id> (item), tp supports ~ relative coords".to_string()
        }
        "items" => all_item_defs()
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
                    "unknown item `{block_text}` (stone, dirt, grass, log, leaves, apple, stick, 1:<block_id>, 2:<item_id>)"
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
        _ => command_handler(line)
            .unwrap_or_else(|| format!("unknown command `{command}` (try `help`)")),
    }
}
