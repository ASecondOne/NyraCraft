use crate::player::crafting::{
    craft_once, craft_output_preview, craft_slot_in_side, normalize_craft_side,
};
use crate::world::blocks::{
    HOTBAR_LOADOUT, HOTBAR_SLOTS, item_break_strength, item_max_durability, item_max_stack_size,
    placeable_block_id_for_item,
};

pub const INVENTORY_ROWS: usize = 4;
pub const INVENTORY_COLS: usize = HOTBAR_SLOTS;
pub const INVENTORY_STORAGE_SLOTS: usize = INVENTORY_ROWS * INVENTORY_COLS;
pub const MAX_STACK_SIZE: u8 = 64;
pub use crate::player::crafting::{CRAFT_GRID_SIDE, CRAFT_GRID_SLOTS, CraftGridMode};

fn item_stack_limit(item_id: i8) -> u8 {
    item_max_stack_size(item_id).clamp(1, MAX_STACK_SIZE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemStack {
    pub block_id: i8,
    pub count: u8,
    pub durability: u16,
}

impl ItemStack {
    pub fn new(block_id: i8, count: u8) -> Self {
        let limit = item_stack_limit(block_id);
        Self {
            block_id,
            count: count.min(limit),
            durability: item_max_durability(block_id).unwrap_or(0),
        }
    }

    pub const fn with_durability(block_id: i8, count: u8, durability: u16) -> Self {
        Self {
            block_id,
            count,
            durability,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventorySlotRef {
    Hotbar(u8),
    Storage(u8),
    CraftInput(u8),
    CraftOutput,
}

#[derive(Clone, Copy, Debug)]
pub struct UiRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl UiRect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px <= self.x + self.w && py <= self.y + self.h
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InventoryLayout {
    pub slot: f32,
    pub gap: f32,
    pub hotbar_start_x: f32,
    pub hotbar_y: f32,
    pub panel: UiRect,
    pub storage_start_x: f32,
    pub storage_start_y: f32,
    pub craft_input_start_x: f32,
    pub craft_input_start_y: f32,
    pub craft_output_x: f32,
    pub craft_output_y: f32,
}

pub struct InventoryState {
    pub open: bool,
    craft_mode: CraftGridMode,
    pub hotbar: [Option<ItemStack>; HOTBAR_SLOTS],
    pub storage: [Option<ItemStack>; INVENTORY_STORAGE_SLOTS],
    pub craft_input: [Option<ItemStack>; CRAFT_GRID_SLOTS],
    pub dragged_item: Option<ItemStack>,
}

impl InventoryState {
    pub fn new() -> Self {
        Self {
            open: false,
            craft_mode: CraftGridMode::Player2x2,
            hotbar: std::array::from_fn(|i| {
                HOTBAR_LOADOUT[i].map(|item_id| ItemStack::new(item_id, item_stack_limit(item_id)))
            }),
            storage: [None; INVENTORY_STORAGE_SLOTS],
            craft_input: [None; CRAFT_GRID_SLOTS],
            dragged_item: None,
        }
    }

    pub fn selected_hotbar_block(&self, slot: u8) -> Option<i8> {
        self.hotbar
            .get(slot as usize)
            .copied()
            .flatten()
            .and_then(|stack| {
                if stack.count > 0 {
                    placeable_block_id_for_item(stack.block_id)
                } else {
                    None
                }
            })
    }

    pub fn selected_hotbar_item_id(&self, slot: u8) -> Option<i8> {
        self.hotbar
            .get(slot as usize)
            .copied()
            .flatten()
            .and_then(|stack| (stack.count > 0).then_some(stack.block_id))
    }

    pub fn selected_hotbar_break_strength(&self, slot: u8) -> Option<f32> {
        self.hotbar
            .get(slot as usize)
            .copied()
            .flatten()
            .and_then(|stack| {
                if stack.count > 0 {
                    item_break_strength(stack.block_id)
                } else {
                    None
                }
            })
    }

    pub fn apply_selected_hotbar_durability_loss(&mut self, slot: u8, amount: u16) -> bool {
        if amount == 0 {
            return false;
        }
        let Some(slot_ref) = self.hotbar.get_mut(slot as usize) else {
            return false;
        };
        let Some(mut stack) = *slot_ref else {
            return false;
        };
        if stack.count == 0 {
            *slot_ref = None;
            return false;
        }
        let Some(max_durability) = item_max_durability(stack.block_id) else {
            return false;
        };
        if stack.durability == 0 {
            stack.durability = max_durability;
        }
        if amount >= stack.durability {
            *slot_ref = None;
            true
        } else {
            stack.durability -= amount;
            *slot_ref = Some(stack);
            false
        }
    }

    pub fn consume_selected_hotbar(&mut self, slot: u8, amount: u8) -> bool {
        if amount == 0 {
            return true;
        }
        let Some(slot_ref) = self.hotbar.get_mut(slot as usize) else {
            return false;
        };
        let Some(mut stack) = *slot_ref else {
            return false;
        };
        if stack.count < amount {
            return false;
        }
        stack.count -= amount;
        if stack.count == 0 {
            *slot_ref = None;
        } else {
            *slot_ref = Some(stack);
        }
        true
    }

    pub fn take_one_selected_hotbar(&mut self, slot: u8) -> Option<ItemStack> {
        let slot_ref = self.hotbar.get_mut(slot as usize)?;
        let mut stack = slot_ref.take()?;
        if stack.count == 0 {
            return None;
        }
        stack.count -= 1;
        let dropped = ItemStack::with_durability(stack.block_id, 1, stack.durability);
        if stack.count > 0 {
            *slot_ref = Some(stack);
        }
        Some(dropped)
    }

    pub fn add_item(&mut self, item_id: i8, count: u16) -> u16 {
        let mut remaining = count;
        if remaining == 0 {
            return 0;
        }
        let limit = item_stack_limit(item_id) as u16;
        while remaining > 0 {
            let chunk = remaining.min(limit);
            let stack = ItemStack::new(item_id, chunk as u8);
            let inserted = match self.add_item_stack(stack) {
                Some(leftover) => chunk.saturating_sub(leftover.count as u16),
                None => chunk,
            };
            if inserted == 0 {
                break;
            }
            remaining = remaining.saturating_sub(inserted);
        }
        remaining
    }

    pub fn add_item_stack(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut stack = normalize_item_stack(Some(stack))?;
        let limit = item_stack_limit(stack.block_id);

        for slot in self.hotbar.iter_mut().chain(self.storage.iter_mut()) {
            if stack.count == 0 {
                break;
            }
            let Some(existing) = slot.as_mut() else {
                continue;
            };
            if existing.block_id != stack.block_id
                || existing.durability != stack.durability
                || existing.count >= limit
            {
                continue;
            }
            let free = limit - existing.count;
            let moved = free.min(stack.count);
            existing.count += moved;
            stack.count -= moved;
        }

        for slot in self.hotbar.iter_mut().chain(self.storage.iter_mut()) {
            if stack.count == 0 {
                break;
            }
            if slot.is_some() {
                continue;
            }
            let moved = stack.count.min(limit);
            *slot = Some(ItemStack::with_durability(
                stack.block_id,
                moved,
                stack.durability,
            ));
            stack.count -= moved;
        }

        (stack.count > 0).then_some(stack)
    }

    pub fn toggle_open(&mut self) {
        if self.open {
            self.close();
        } else {
            self.open_player_inventory();
        }
    }

    pub fn open_player_inventory(&mut self) {
        self.open = true;
        self.craft_mode = CraftGridMode::Player2x2;
        self.stow_craft_slots_outside_active_grid();
    }

    pub fn open_crafting_table(&mut self) {
        self.open = true;
        self.craft_mode = CraftGridMode::Table3x3;
    }

    pub fn craft_grid_side(&self) -> usize {
        self.craft_mode.side()
    }

    pub fn close(&mut self) {
        self.open = false;
        self.stow_dragged_item();
    }

    pub fn craft_output_preview(&self) -> Option<ItemStack> {
        craft_output_preview(&self.craft_input, self.craft_grid_side())
    }

    pub fn left_click_slot(&mut self, slot_ref: Option<InventorySlotRef>) {
        let Some(slot_ref) = slot_ref else {
            return;
        };
        if matches!(slot_ref, InventorySlotRef::CraftOutput) {
            self.take_craft_output();
            return;
        }

        let slot_item = self.slot_item(slot_ref);
        match self.dragged_item {
            None => {
                if let Some(item) = slot_item {
                    self.dragged_item = Some(item);
                    self.set_slot_item(slot_ref, None);
                }
            }
            Some(mut held) => {
                if let Some(mut slot_stack) = slot_item {
                    let limit = item_stack_limit(slot_stack.block_id);
                    if slot_stack.block_id == held.block_id
                        && slot_stack.durability == held.durability
                        && slot_stack.count < limit
                    {
                        let free = limit - slot_stack.count;
                        let moved = free.min(held.count);
                        slot_stack.count += moved;
                        held.count -= moved;
                        self.set_slot_item(slot_ref, Some(slot_stack));
                        self.dragged_item = (held.count > 0).then_some(held);
                    } else {
                        self.set_slot_item(slot_ref, Some(held));
                        self.dragged_item = Some(slot_stack);
                    }
                } else {
                    self.set_slot_item(slot_ref, Some(held));
                    self.dragged_item = None;
                }
            }
        }
    }

    pub fn right_click_slot(&mut self, slot_ref: Option<InventorySlotRef>) {
        let Some(slot_ref) = slot_ref else {
            return;
        };
        if matches!(slot_ref, InventorySlotRef::CraftOutput) {
            self.take_craft_output();
            return;
        }

        let slot_item = self.slot_item(slot_ref);
        match self.dragged_item {
            None => {
                if let Some(mut item) = slot_item {
                    let take = ((item.count as u16).saturating_add(1) / 2) as u8;
                    let remain = item.count.saturating_sub(take);
                    self.dragged_item = Some(ItemStack::with_durability(
                        item.block_id,
                        take,
                        item.durability,
                    ));
                    if remain == 0 {
                        self.set_slot_item(slot_ref, None);
                    } else {
                        item.count = remain;
                        self.set_slot_item(slot_ref, Some(item));
                    }
                }
            }
            Some(mut held) => {
                if held.count == 0 {
                    self.dragged_item = None;
                    return;
                }
                if let Some(mut slot_stack) = slot_item {
                    let limit = item_stack_limit(slot_stack.block_id);
                    if slot_stack.block_id == held.block_id
                        && slot_stack.durability == held.durability
                        && slot_stack.count < limit
                    {
                        slot_stack.count = slot_stack.count.saturating_add(1);
                        held.count = held.count.saturating_sub(1);
                        self.set_slot_item(slot_ref, Some(slot_stack));
                    }
                } else {
                    self.set_slot_item(
                        slot_ref,
                        Some(ItemStack::with_durability(
                            held.block_id,
                            1,
                            held.durability,
                        )),
                    );
                    held.count = held.count.saturating_sub(1);
                }
                self.dragged_item = (held.count > 0).then_some(held);
            }
        }
    }

    pub fn quick_move_slot(&mut self, slot_ref: Option<InventorySlotRef>) {
        if self.dragged_item.is_some() {
            return;
        }
        let Some(slot_ref) = slot_ref else {
            return;
        };
        if matches!(slot_ref, InventorySlotRef::CraftOutput) {
            return;
        }
        let Some(mut stack) = self.slot_item(slot_ref) else {
            return;
        };

        match slot_ref {
            InventorySlotRef::Hotbar(_) => move_stack_into_slots(&mut stack, &mut self.storage),
            InventorySlotRef::Storage(_) => move_stack_into_slots(&mut stack, &mut self.hotbar),
            InventorySlotRef::CraftInput(_) => {
                move_stack_into_slots(&mut stack, &mut self.hotbar);
                if stack.count > 0 {
                    move_stack_into_slots(&mut stack, &mut self.storage);
                }
            }
            InventorySlotRef::CraftOutput => {}
        }

        if stack.count == 0 {
            self.set_slot_item(slot_ref, None);
        } else {
            self.set_slot_item(slot_ref, Some(stack));
        }
    }

    fn take_craft_output(&mut self) {
        let grid_side = self.craft_grid_side();
        let Some(output) = craft_output_preview(&self.craft_input, grid_side) else {
            return;
        };

        match self.dragged_item {
            None => {
                if let Some(crafted) = craft_once(&mut self.craft_input, grid_side) {
                    self.dragged_item = Some(crafted);
                }
            }
            Some(mut held) => {
                let limit = item_stack_limit(held.block_id);
                if held.block_id != output.block_id
                    || held.durability != output.durability
                    || held.count >= limit
                {
                    return;
                }
                let free = limit - held.count;
                if free < output.count {
                    return;
                }
                if let Some(crafted) = craft_once(&mut self.craft_input, grid_side) {
                    held.count = held.count.saturating_add(crafted.count);
                    self.dragged_item = Some(held);
                }
            }
        }
    }

    fn stow_dragged_item(&mut self) {
        let Some(stack) = self.dragged_item.take() else {
            return;
        };
        self.stow_item_stack(stack);
    }

    fn stow_item_stack(&mut self, stack: ItemStack) {
        if let Some(remaining) = self.add_item_stack(stack) {
            self.hotbar[0] = Some(remaining);
        }
    }

    fn stow_craft_slots_outside_active_grid(&mut self) {
        let side = self.craft_grid_side();
        for index in 0..CRAFT_GRID_SLOTS {
            if craft_slot_in_side(index, side) {
                continue;
            }
            let Some(stack) = self.craft_input[index].take() else {
                continue;
            };
            if let Some(remaining) = self.add_item_stack(stack) {
                self.craft_input[index] = Some(remaining);
            }
        }
    }

    fn slot_item(&self, slot_ref: InventorySlotRef) -> Option<ItemStack> {
        match slot_ref {
            InventorySlotRef::Hotbar(idx) => self.hotbar.get(idx as usize).copied().flatten(),
            InventorySlotRef::Storage(idx) => self.storage.get(idx as usize).copied().flatten(),
            InventorySlotRef::CraftInput(idx) => {
                let index = idx as usize;
                if !craft_slot_in_side(index, self.craft_grid_side()) {
                    return None;
                }
                self.craft_input.get(index).copied().flatten()
            }
            InventorySlotRef::CraftOutput => self.craft_output_preview(),
        }
    }

    fn set_slot_item(&mut self, slot_ref: InventorySlotRef, item: Option<ItemStack>) {
        let item = normalize_item_stack(item);
        match slot_ref {
            InventorySlotRef::Hotbar(idx) => {
                if let Some(slot) = self.hotbar.get_mut(idx as usize) {
                    *slot = item;
                }
            }
            InventorySlotRef::Storage(idx) => {
                if let Some(slot) = self.storage.get_mut(idx as usize) {
                    *slot = item;
                }
            }
            InventorySlotRef::CraftInput(idx) => {
                let index = idx as usize;
                if !craft_slot_in_side(index, self.craft_grid_side()) {
                    return;
                }
                if let Some(slot) = self.craft_input.get_mut(index) {
                    *slot = item;
                }
            }
            InventorySlotRef::CraftOutput => {}
        }
    }
}

fn normalize_item_stack(item: Option<ItemStack>) -> Option<ItemStack> {
    let mut item = item?;
    if item.count == 0 {
        return None;
    }
    let limit = item_stack_limit(item.block_id);
    if item.count > limit {
        item.count = limit;
    }
    item.durability = item_max_durability(item.block_id)
        .map(|max| {
            if item.durability == 0 {
                max
            } else {
                item.durability.min(max)
            }
        })
        .unwrap_or(0);
    Some(item)
}

fn move_stack_into_slots(stack: &mut ItemStack, slots: &mut [Option<ItemStack>]) {
    if stack.count == 0 {
        return;
    }
    let limit = item_stack_limit(stack.block_id);

    for slot in slots.iter_mut() {
        if stack.count == 0 {
            return;
        }
        let Some(existing) = slot.as_mut() else {
            continue;
        };
        if existing.block_id != stack.block_id
            || existing.durability != stack.durability
            || existing.count >= limit
        {
            continue;
        }
        let free = limit - existing.count;
        let moved = free.min(stack.count);
        existing.count += moved;
        stack.count -= moved;
    }

    for slot in slots.iter_mut() {
        if stack.count == 0 {
            return;
        }
        if slot.is_some() {
            continue;
        }
        let moved = stack.count.min(limit);
        *slot = Some(ItemStack::with_durability(
            stack.block_id,
            moved,
            stack.durability,
        ));
        stack.count -= moved;
    }
}

pub fn compute_inventory_layout(
    width: u32,
    height: u32,
    craft_grid_side: usize,
) -> InventoryLayout {
    let craft_grid_side = normalize_craft_side(craft_grid_side);
    let width_f = width as f32;
    let height_f = height as f32;
    let mut slot = (height_f * 0.062).clamp(20.0, 44.0);
    let mut gap = (slot * 0.18).clamp(4.0, 8.0);
    let max_total = width_f * 0.92;
    let mut total = HOTBAR_SLOTS as f32 * slot + (HOTBAR_SLOTS as f32 - 1.0) * gap;
    if total > max_total {
        let scale = max_total / total;
        slot *= scale;
        gap *= scale;
        total = HOTBAR_SLOTS as f32 * slot + (HOTBAR_SLOTS as f32 - 1.0) * gap;
    }

    let margin_bottom = (height_f * 0.03).clamp(12.0, 28.0);
    let hotbar_start_x = (width_f - total) * 0.5;
    let hotbar_y = (height_f - margin_bottom - slot).max(2.0);

    let storage_w = total;
    let storage_h = INVENTORY_ROWS as f32 * slot + (INVENTORY_ROWS as f32 - 1.0) * gap;
    let craft_side_f = craft_grid_side as f32;
    let craft_h = craft_side_f * slot + (craft_side_f - 1.0) * gap;
    let craft_to_storage_gap = (slot * 0.42).clamp(8.0, 18.0);
    let panel_pad = (slot * 0.34).clamp(8.0, 16.0);
    let panel = UiRect {
        x: hotbar_start_x - panel_pad,
        y: (hotbar_y - (storage_h + craft_h + craft_to_storage_gap + panel_pad * 2.25) - gap * 1.6)
            .max(8.0),
        w: storage_w + panel_pad * 2.0,
        h: storage_h + craft_h + craft_to_storage_gap + panel_pad * 2.25,
    };

    let craft_input_start_x = panel.x + panel_pad;
    let craft_input_start_y = panel.y + panel_pad * 1.05;
    let output_gap = (slot * 0.34).clamp(8.0, 16.0);
    let craft_output_x =
        craft_input_start_x + craft_side_f * slot + (craft_side_f - 1.0) * gap + output_gap;
    let craft_output_y = craft_input_start_y + (craft_h - slot) * 0.5;

    let storage_start_x = panel.x + panel_pad;
    let storage_start_y = craft_input_start_y + craft_h + craft_to_storage_gap;

    InventoryLayout {
        slot,
        gap,
        hotbar_start_x,
        hotbar_y,
        panel,
        storage_start_x,
        storage_start_y,
        craft_input_start_x,
        craft_input_start_y,
        craft_output_x,
        craft_output_y,
    }
}

pub fn hotbar_slot_rect(layout: &InventoryLayout, index: usize) -> UiRect {
    UiRect {
        x: layout.hotbar_start_x + index as f32 * (layout.slot + layout.gap),
        y: layout.hotbar_y,
        w: layout.slot,
        h: layout.slot,
    }
}

pub fn storage_slot_rect(layout: &InventoryLayout, index: usize) -> UiRect {
    let row = index / INVENTORY_COLS;
    let col = index % INVENTORY_COLS;
    UiRect {
        x: layout.storage_start_x + col as f32 * (layout.slot + layout.gap),
        y: layout.storage_start_y + row as f32 * (layout.slot + layout.gap),
        w: layout.slot,
        h: layout.slot,
    }
}

pub fn craft_input_slot_rect(layout: &InventoryLayout, index: usize) -> UiRect {
    let row = index / CRAFT_GRID_SIDE;
    let col = index % CRAFT_GRID_SIDE;
    UiRect {
        x: layout.craft_input_start_x + col as f32 * (layout.slot + layout.gap),
        y: layout.craft_input_start_y + row as f32 * (layout.slot + layout.gap),
        w: layout.slot,
        h: layout.slot,
    }
}

pub fn craft_output_slot_rect(layout: &InventoryLayout) -> UiRect {
    UiRect {
        x: layout.craft_output_x,
        y: layout.craft_output_y,
        w: layout.slot,
        h: layout.slot,
    }
}

pub fn hit_test_slot(
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    inventory_open: bool,
    craft_grid_side: usize,
) -> Option<InventorySlotRef> {
    let craft_grid_side = normalize_craft_side(craft_grid_side);
    let layout = compute_inventory_layout(width, height, craft_grid_side);

    if inventory_open {
        if craft_output_slot_rect(&layout).contains(x, y) {
            return Some(InventorySlotRef::CraftOutput);
        }
        for i in 0..CRAFT_GRID_SLOTS {
            if !craft_slot_in_side(i, craft_grid_side) {
                continue;
            }
            if craft_input_slot_rect(&layout, i).contains(x, y) {
                return Some(InventorySlotRef::CraftInput(i as u8));
            }
        }
        for i in 0..INVENTORY_STORAGE_SLOTS {
            if storage_slot_rect(&layout, i).contains(x, y) {
                return Some(InventorySlotRef::Storage(i as u8));
            }
        }
    }

    for i in 0..HOTBAR_SLOTS {
        if hotbar_slot_rect(&layout, i).contains(x, y) {
            return Some(InventorySlotRef::Hotbar(i as u8));
        }
    }
    None
}
