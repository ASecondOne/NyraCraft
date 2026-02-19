use crate::player::inventory::UiRect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauseMenuButton {
    ReturnToGame,
    Quit,
}

#[derive(Clone, Copy, Debug)]
pub struct PauseMenuLayout {
    pub panel: UiRect,
    pub return_button: UiRect,
    pub quit_button: UiRect,
}

pub fn compute_pause_menu_layout(width: u32, height: u32) -> Option<PauseMenuLayout> {
    if width == 0 || height == 0 {
        return None;
    }

    let width_f = width as f32;
    let height_f = height as f32;
    let panel_w = (width_f * 0.34).clamp(280.0, 460.0);
    let panel_h = (height_f * 0.46).clamp(260.0, 360.0);
    let panel_x = (width_f - panel_w) * 0.5;
    let panel_y = (height_f - panel_h) * 0.30;

    let button_w = (panel_w * 0.82).clamp(220.0, panel_w - 24.0);
    let button_h = (panel_h * 0.16).clamp(34.0, 52.0);
    let button_x = panel_x + (panel_w - button_w) * 0.5;
    let button_gap = (button_h * 0.34).clamp(10.0, 18.0);
    let first_button_y = panel_y + panel_h * 0.33;

    Some(PauseMenuLayout {
        panel: UiRect {
            x: panel_x,
            y: panel_y,
            w: panel_w,
            h: panel_h,
        },
        return_button: UiRect {
            x: button_x,
            y: first_button_y,
            w: button_w,
            h: button_h,
        },
        quit_button: UiRect {
            x: button_x,
            y: first_button_y + button_h + button_gap,
            w: button_w,
            h: button_h,
        },
    })
}

pub fn hit_test_pause_menu_button(
    width: u32,
    height: u32,
    x: f32,
    y: f32,
) -> Option<PauseMenuButton> {
    let layout = compute_pause_menu_layout(width, height)?;
    if layout.return_button.contains(x, y) {
        Some(PauseMenuButton::ReturnToGame)
    } else if layout.quit_button.contains(x, y) {
        Some(PauseMenuButton::Quit)
    } else {
        None
    }
}
