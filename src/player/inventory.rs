use crate::world::blocks::{
    BLOCK_CRAFTING_TABLE, BLOCK_LOG, BLOCK_PLANKS_OAK, HOTBAR_LOADOUT, HOTBAR_SLOTS, ITEM_STICK,
    placeable_block_id_for_item,
};

pub const INVENTORY_ROWS: usize = 4;
pub const INVENTORY_COLS: usize = HOTBAR_SLOTS;
pub const INVENTORY_STORAGE_SLOTS: usize = INVENTORY_ROWS * INVENTORY_COLS;
pub const PLAYER_CRAFT_GRID_SIDE: usize = 2;
pub const TABLE_CRAFT_GRID_SIDE: usize = 3;
pub const CRAFT_GRID_SIDE: usize = TABLE_CRAFT_GRID_SIDE;
pub const CRAFT_GRID_SLOTS: usize = CRAFT_GRID_SIDE * CRAFT_GRID_SIDE;
pub const MAX_STACK_SIZE: u8 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemStack {
    pub block_id: i8,
    pub count: u8,
}

impl ItemStack {
    pub const fn new(block_id: i8, count: u8) -> Self {
        Self { block_id, count }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventorySlotRef {
    Hotbar(u8),
    Storage(u8),
    CraftInput(u8),
    CraftOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CraftGridMode {
    Player2x2,
    Table3x3,
}

impl CraftGridMode {
    pub const fn side(self) -> usize {
        match self {
            Self::Player2x2 => PLAYER_CRAFT_GRID_SIDE,
            Self::Table3x3 => TABLE_CRAFT_GRID_SIDE,
        }
    }
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

#[derive(Clone, Copy)]
struct CraftRecipe {
    width: u8,
    height: u8,
    input: [Option<i8>; CRAFT_GRID_SLOTS],
    output: ItemStack,
}

#[derive(Clone, Copy)]
struct NormalizedCraftGrid {
    origin_row: usize,
    origin_col: usize,
    width: usize,
    height: usize,
    cells: [Option<ItemStack>; CRAFT_GRID_SLOTS],
}

#[derive(Clone, Copy)]
struct CraftMatch {
    recipe: &'static CraftRecipe,
    grid: NormalizedCraftGrid,
}

const CRAFT_RECIPES: [CraftRecipe; 3] = [
    CraftRecipe {
        width: 1,
        height: 1,
        input: [
            Some(BLOCK_LOG as i8),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        output: ItemStack::new(BLOCK_PLANKS_OAK as i8, 4),
    },
    CraftRecipe {
        width: 1,
        height: 2,
        input: [
            Some(BLOCK_PLANKS_OAK as i8),
            None,
            None,
            Some(BLOCK_PLANKS_OAK as i8),
            None,
            None,
            None,
            None,
            None,
        ],
        output: ItemStack::new(ITEM_STICK, 4),
    },
    CraftRecipe {
        width: 2,
        height: 2,
        input: [
            Some(BLOCK_PLANKS_OAK as i8),
            Some(BLOCK_PLANKS_OAK as i8),
            None,
            Some(BLOCK_PLANKS_OAK as i8),
            Some(BLOCK_PLANKS_OAK as i8),
            None,
            None,
            None,
            None,
        ],
        output: ItemStack::new(BLOCK_CRAFTING_TABLE as i8, 1),
    },
];

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
                HOTBAR_LOADOUT[i].map(|block_id| ItemStack::new(block_id, MAX_STACK_SIZE))
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
        let dropped = ItemStack::new(stack.block_id, 1);
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

        for slot in self.hotbar.iter_mut().chain(self.storage.iter_mut()) {
            if remaining == 0 {
                break;
            }
            let Some(stack) = slot.as_mut() else {
                continue;
            };
            if stack.block_id != item_id || stack.count >= MAX_STACK_SIZE {
                continue;
            }
            let free = (MAX_STACK_SIZE - stack.count) as u16;
            let moved = free.min(remaining);
            stack.count += moved as u8;
            remaining -= moved;
        }

        for slot in self.hotbar.iter_mut().chain(self.storage.iter_mut()) {
            if remaining == 0 {
                break;
            }
            if slot.is_some() {
                continue;
            }
            let moved = remaining.min(MAX_STACK_SIZE as u16);
            *slot = Some(ItemStack::new(item_id, moved as u8));
            remaining -= moved;
        }

        remaining
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
        self.craft_match().map(|matched| matched.recipe.output)
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
                    if slot_stack.block_id == held.block_id && slot_stack.count < MAX_STACK_SIZE {
                        let free = MAX_STACK_SIZE - slot_stack.count;
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
                    self.dragged_item = Some(ItemStack::new(item.block_id, take));
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
                    if slot_stack.block_id == held.block_id && slot_stack.count < MAX_STACK_SIZE {
                        slot_stack.count = slot_stack.count.saturating_add(1);
                        held.count = held.count.saturating_sub(1);
                        self.set_slot_item(slot_ref, Some(slot_stack));
                    }
                } else {
                    self.set_slot_item(slot_ref, Some(ItemStack::new(held.block_id, 1)));
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
        let Some(matched) = self.craft_match() else {
            return;
        };
        let output = matched.recipe.output;

        match self.dragged_item {
            None => {
                if self.consume_craft_ingredients(&matched) {
                    self.dragged_item = Some(output);
                }
            }
            Some(mut held) => {
                if held.block_id != output.block_id || held.count >= MAX_STACK_SIZE {
                    return;
                }
                let free = MAX_STACK_SIZE - held.count;
                if free < output.count {
                    return;
                }
                if self.consume_craft_ingredients(&matched) {
                    held.count = held.count.saturating_add(output.count);
                    self.dragged_item = Some(held);
                }
            }
        }
    }

    fn consume_craft_ingredients(&mut self, matched: &CraftMatch) -> bool {
        for row in 0..matched.grid.height {
            for col in 0..matched.grid.width {
                let pattern_index = row * CRAFT_GRID_SIDE + col;
                let Some(required_id) = matched.recipe.input[pattern_index] else {
                    continue;
                };
                let slot_index = (matched.grid.origin_row + row) * CRAFT_GRID_SIDE
                    + (matched.grid.origin_col + col);
                let Some(mut stack) = self.craft_input.get(slot_index).copied().flatten() else {
                    return false;
                };
                if stack.block_id != required_id || stack.count == 0 {
                    return false;
                }
                stack.count -= 1;
                self.craft_input[slot_index] = (stack.count > 0).then_some(stack);
            }
        }
        true
    }

    fn craft_match(&self) -> Option<CraftMatch> {
        let normalized = normalize_craft_input(&self.craft_input, self.craft_grid_side())?;
        let recipe = CRAFT_RECIPES
            .iter()
            .find(|recipe| recipe_matches_grid(recipe, &normalized))?;
        Some(CraftMatch {
            recipe,
            grid: normalized,
        })
    }

    fn stow_dragged_item(&mut self) {
        let Some(stack) = self.dragged_item.take() else {
            return;
        };
        self.stow_item_stack(stack);
    }

    fn stow_item_stack(&mut self, stack: ItemStack) {
        let remaining = self.add_item(stack.block_id, stack.count as u16);
        if remaining > 0 {
            self.hotbar[0] = Some(ItemStack::new(stack.block_id, remaining as u8));
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
            let remaining = self.add_item(stack.block_id, stack.count as u16);
            if remaining > 0 {
                self.craft_input[index] = Some(ItemStack::new(stack.block_id, remaining as u8));
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
    if item.count > MAX_STACK_SIZE {
        item.count = MAX_STACK_SIZE;
    }
    Some(item)
}

fn move_stack_into_slots(stack: &mut ItemStack, slots: &mut [Option<ItemStack>]) {
    if stack.count == 0 {
        return;
    }

    for slot in slots.iter_mut() {
        if stack.count == 0 {
            return;
        }
        let Some(existing) = slot.as_mut() else {
            continue;
        };
        if existing.block_id != stack.block_id || existing.count >= MAX_STACK_SIZE {
            continue;
        }
        let free = MAX_STACK_SIZE - existing.count;
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
        let moved = stack.count.min(MAX_STACK_SIZE);
        *slot = Some(ItemStack::new(stack.block_id, moved));
        stack.count -= moved;
    }
}

fn normalize_craft_input(
    input: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
) -> Option<NormalizedCraftGrid> {
    let craft_grid_side = normalize_craft_side(craft_grid_side);
    let mut min_row = craft_grid_side;
    let mut min_col = craft_grid_side;
    let mut max_row = 0usize;
    let mut max_col = 0usize;
    let mut has_any = false;

    for (index, entry) in input.iter().enumerate() {
        let Some(stack) = entry else {
            continue;
        };
        if stack.count == 0 {
            continue;
        }
        let row = index / CRAFT_GRID_SIDE;
        let col = index % CRAFT_GRID_SIDE;
        if row >= craft_grid_side || col >= craft_grid_side {
            continue;
        }
        min_row = min_row.min(row);
        min_col = min_col.min(col);
        max_row = max_row.max(row);
        max_col = max_col.max(col);
        has_any = true;
    }

    if !has_any {
        return None;
    }

    let width = max_col - min_col + 1;
    let height = max_row - min_row + 1;
    let mut cells = [None; CRAFT_GRID_SLOTS];
    for row in 0..height {
        for col in 0..width {
            let src_index = (min_row + row) * CRAFT_GRID_SIDE + (min_col + col);
            let dst_index = row * CRAFT_GRID_SIDE + col;
            cells[dst_index] = input[src_index];
        }
    }

    Some(NormalizedCraftGrid {
        origin_row: min_row,
        origin_col: min_col,
        width,
        height,
        cells,
    })
}

fn recipe_matches_grid(recipe: &CraftRecipe, grid: &NormalizedCraftGrid) -> bool {
    if recipe.width as usize != grid.width || recipe.height as usize != grid.height {
        return false;
    }

    for row in 0..grid.height {
        for col in 0..grid.width {
            let index = row * CRAFT_GRID_SIDE + col;
            let expected = recipe.input[index];
            let got = grid.cells[index].map(|stack| stack.block_id);
            if expected != got {
                return false;
            }
            if expected.is_some() && grid.cells[index].map(|stack| stack.count).unwrap_or(0) == 0 {
                return false;
            }
        }
    }

    true
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

fn normalize_craft_side(side: usize) -> usize {
    side.clamp(1, CRAFT_GRID_SIDE)
}

fn craft_slot_in_side(index: usize, craft_grid_side: usize) -> bool {
    let row = index / CRAFT_GRID_SIDE;
    let col = index % CRAFT_GRID_SIDE;
    row < craft_grid_side && col < craft_grid_side
}
