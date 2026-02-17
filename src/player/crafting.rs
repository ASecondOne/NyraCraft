use crate::player::inventory::ItemStack;
use crate::world::blocks::parse_item_id;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const PLAYER_CRAFT_GRID_SIDE: usize = 2;
pub const TABLE_CRAFT_GRID_SIDE: usize = 3;
pub const CRAFT_GRID_SIDE: usize = TABLE_CRAFT_GRID_SIDE;
pub const CRAFT_GRID_SLOTS: usize = CRAFT_GRID_SIDE * CRAFT_GRID_SIDE;

const RECIPES_JSON_DIR: &str = "src/content/recipes";

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

#[derive(Clone)]
struct CraftRecipe {
    output: ItemStack,
    kind: RecipeKind,
}

#[derive(Clone)]
enum RecipeKind {
    Shaped(ShapedRecipe),
    Shapeless(ShapelessRecipe),
}

#[derive(Clone)]
struct ShapedRecipe {
    width: usize,
    height: usize,
    cells: [Option<i8>; CRAFT_GRID_SLOTS],
}

#[derive(Clone)]
struct ShapelessRecipe {
    required: BTreeMap<i8, u16>,
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
enum CraftMatchKind {
    Shaped(NormalizedCraftGrid),
    Shapeless,
}

#[derive(Clone, Copy)]
struct CraftMatch {
    recipe_index: usize,
    kind: CraftMatchKind,
}

struct Registry {
    recipes: Vec<CraftRecipe>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawRecipeFile {
    One(RawRecipeDef),
    Many(Vec<RawRecipeDef>),
}

#[derive(Debug, Deserialize, Clone)]
struct RawRecipeDef {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    recipe_type: String,
    output: RawRecipeOutput,
    #[serde(default)]
    pattern: Vec<String>,
    #[serde(default)]
    key: HashMap<String, RawIngredient>,
    #[serde(default)]
    ingredients: Vec<RawIngredient>,
    #[serde(default)]
    aliases: Vec<RawRecipeAlias>,
}

#[derive(Debug, Deserialize, Clone)]
struct RawRecipeAlias {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    recipe_type: Option<String>,
    #[serde(default)]
    pattern: Option<Vec<String>>,
    #[serde(default)]
    key: Option<HashMap<String, RawIngredient>>,
    #[serde(default)]
    ingredients: Option<Vec<RawIngredient>>,
}

#[derive(Debug, Deserialize, Clone)]
struct RawRecipeOutput {
    item: String,
    #[serde(default = "default_one_u8")]
    count: u8,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum RawIngredient {
    Text(String),
    Entry {
        item: String,
        #[serde(default = "default_one_u8")]
        count: u8,
    },
}

impl RawIngredient {
    fn item_name(&self) -> &str {
        match self {
            Self::Text(name) => name,
            Self::Entry { item, .. } => item,
        }
    }

    fn count(&self) -> u8 {
        match self {
            Self::Text(_) => 1,
            Self::Entry { count, .. } => (*count).max(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedRecipeKind {
    Shaped,
    Shapeless,
}

static RECIPES: OnceLock<Registry> = OnceLock::new();

pub fn normalize_craft_side(side: usize) -> usize {
    side.clamp(1, CRAFT_GRID_SIDE)
}

pub fn craft_slot_in_side(index: usize, craft_grid_side: usize) -> bool {
    let row = index / CRAFT_GRID_SIDE;
    let col = index % CRAFT_GRID_SIDE;
    row < craft_grid_side && col < craft_grid_side
}

pub fn craft_output_preview(
    input: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
) -> Option<ItemStack> {
    let matched = craft_match(input, craft_grid_side)?;
    registry().recipes.get(matched.recipe_index).map(|r| r.output)
}

pub fn craft_once(
    input: &mut [Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
) -> Option<ItemStack> {
    let matched = craft_match(input, craft_grid_side)?;
    let recipe = registry().recipes.get(matched.recipe_index)?;
    let output = recipe.output;
    let consumed = match (matched.kind, &recipe.kind) {
        (CraftMatchKind::Shaped(grid), RecipeKind::Shaped(shaped)) => {
            consume_shaped_ingredients(input, &grid, shaped)
        }
        (CraftMatchKind::Shapeless, RecipeKind::Shapeless(shapeless)) => {
            consume_shapeless_ingredients(input, craft_grid_side, shapeless)
        }
        _ => false,
    };
    consumed.then_some(output)
}

fn registry() -> &'static Registry {
    RECIPES.get_or_init(load_registry)
}

fn default_one_u8() -> u8 {
    1
}

fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(OsStr::to_str)
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .collect();
    files.sort();
    files
}

fn load_registry() -> Registry {
    let mut recipes = Vec::<CraftRecipe>::new();
    let files = collect_json_files(Path::new(RECIPES_JSON_DIR));

    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            eprintln!("failed to read crafting recipe file {}", path.display());
            continue;
        };

        let raw_defs = match serde_json::from_str::<RawRecipeFile>(&text) {
            Ok(RawRecipeFile::One(def)) => vec![def],
            Ok(RawRecipeFile::Many(defs)) => defs,
            Err(err) => {
                eprintln!(
                    "failed to parse crafting recipe file {}: {}",
                    path.display(),
                    err
                );
                continue;
            }
        };

        for (recipe_idx, raw) in raw_defs.iter().enumerate() {
            let stem = path
                .file_stem()
                .and_then(OsStr::to_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("recipe_file_{}", recipes.len() + 1));
            let base_id = raw
                .id
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}#{}", stem, recipe_idx + 1));

            if let Some(recipe) = build_recipe_from_parts(
                &base_id,
                &raw.recipe_type,
                &raw.output,
                &raw.pattern,
                &raw.key,
                &raw.ingredients,
            ) {
                recipes.push(recipe);
            }

            for (alias_idx, alias) in raw.aliases.iter().enumerate() {
                let alias_id = alias
                    .id
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{}#alias{}", base_id, alias_idx + 1));

                let alias_type = alias
                    .recipe_type
                    .as_deref()
                    .unwrap_or(raw.recipe_type.as_str());
                let alias_pattern = alias.pattern.as_deref().unwrap_or(raw.pattern.as_slice());
                let alias_key = alias.key.as_ref().unwrap_or(&raw.key);
                let alias_ingredients = alias
                    .ingredients
                    .as_deref()
                    .unwrap_or(raw.ingredients.as_slice());

                if let Some(recipe) = build_recipe_from_parts(
                    &alias_id,
                    alias_type,
                    &raw.output,
                    alias_pattern,
                    alias_key,
                    alias_ingredients,
                ) {
                    recipes.push(recipe);
                }
            }
        }
    }

    if recipes.is_empty() {
        eprintln!(
            "no valid crafting recipes loaded from {}, crafting output will stay empty",
            RECIPES_JSON_DIR
        );
    }

    Registry { recipes }
}

fn parse_recipe_kind(raw: &str) -> Option<ParsedRecipeKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "shaped" => Some(ParsedRecipeKind::Shaped),
        "shapeless" => Some(ParsedRecipeKind::Shapeless),
        _ => None,
    }
}

fn resolve_recipe_item_id(item_name: &str, recipe_id: &str, field: &str) -> Option<i8> {
    let Some(item_id) = parse_item_id(item_name) else {
        eprintln!(
            "recipe {} has unknown {} item `{}`",
            recipe_id, field, item_name
        );
        return None;
    };
    Some(item_id)
}

fn build_recipe_from_parts(
    recipe_id: &str,
    recipe_type: &str,
    output: &RawRecipeOutput,
    pattern: &[String],
    key: &HashMap<String, RawIngredient>,
    ingredients: &[RawIngredient],
) -> Option<CraftRecipe> {
    let kind = parse_recipe_kind(recipe_type)?;
    let output_item_id = resolve_recipe_item_id(output.item.as_str(), recipe_id, "output")?;
    let output_count = output.count.max(1);
    let output_stack = ItemStack::new(output_item_id, output_count);

    let recipe_kind = match kind {
        ParsedRecipeKind::Shaped => {
            let shaped = build_shaped_recipe(recipe_id, pattern, key)?;
            RecipeKind::Shaped(shaped)
        }
        ParsedRecipeKind::Shapeless => {
            let shapeless = build_shapeless_recipe(recipe_id, ingredients)?;
            RecipeKind::Shapeless(shapeless)
        }
    };

    Some(CraftRecipe {
        output: output_stack,
        kind: recipe_kind,
    })
}

fn is_pattern_empty_symbol(ch: char) -> bool {
    matches!(ch, ' ' | '.' | '_')
}

fn build_shaped_recipe(
    recipe_id: &str,
    pattern: &[String],
    key: &HashMap<String, RawIngredient>,
) -> Option<ShapedRecipe> {
    if pattern.is_empty() {
        eprintln!("shaped recipe {} has empty pattern", recipe_id);
        return None;
    }

    if pattern.len() > CRAFT_GRID_SIDE {
        eprintln!(
            "shaped recipe {} is too tall ({} > {})",
            recipe_id,
            pattern.len(),
            CRAFT_GRID_SIDE
        );
        return None;
    }

    let width = pattern
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    if width == 0 {
        eprintln!("shaped recipe {} has no pattern symbols", recipe_id);
        return None;
    }
    if width > CRAFT_GRID_SIDE {
        eprintln!(
            "shaped recipe {} is too wide ({} > {})",
            recipe_id, width, CRAFT_GRID_SIDE
        );
        return None;
    }

    let mut cells = [None; CRAFT_GRID_SLOTS];
    let mut has_any = false;

    for (row_idx, row_text) in pattern.iter().enumerate() {
        let row_chars: Vec<char> = row_text.chars().collect();
        for col_idx in 0..width {
            let symbol = row_chars.get(col_idx).copied().unwrap_or(' ');
            if is_pattern_empty_symbol(symbol) {
                continue;
            }
            let symbol_key = symbol.to_string();
            let Some(spec) = key.get(&symbol_key) else {
                eprintln!(
                    "shaped recipe {} references unknown key symbol `{}`",
                    recipe_id, symbol
                );
                return None;
            };
            let item_id = resolve_recipe_item_id(spec.item_name(), recipe_id, "key")?;
            cells[row_idx * CRAFT_GRID_SIDE + col_idx] = Some(item_id);
            has_any = true;
        }
    }

    if !has_any {
        eprintln!("shaped recipe {} has no ingredients", recipe_id);
        return None;
    }

    Some(ShapedRecipe {
        width,
        height: pattern.len(),
        cells,
    })
}

fn build_shapeless_recipe(recipe_id: &str, ingredients: &[RawIngredient]) -> Option<ShapelessRecipe> {
    if ingredients.is_empty() {
        eprintln!("shapeless recipe {} has no ingredients", recipe_id);
        return None;
    }

    let mut required = BTreeMap::<i8, u16>::new();
    for spec in ingredients {
        let item_id = resolve_recipe_item_id(spec.item_name(), recipe_id, "ingredient")?;
        let count = spec.count().max(1) as u16;
        *required.entry(item_id).or_insert(0) += count;
    }

    if required.is_empty() {
        eprintln!("shapeless recipe {} has no valid ingredients", recipe_id);
        return None;
    }

    Some(ShapelessRecipe { required })
}

fn craft_match(
    input: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
) -> Option<CraftMatch> {
    let reg = registry();
    if reg.recipes.is_empty() {
        return None;
    }

    let normalized = normalize_craft_input(input, craft_grid_side);
    let counts = active_grid_counts(input, craft_grid_side);

    for (recipe_index, recipe) in reg.recipes.iter().enumerate() {
        match &recipe.kind {
            RecipeKind::Shaped(shaped) => {
                let Some(grid) = normalized else {
                    continue;
                };
                if shaped_recipe_matches_grid(shaped, &grid) {
                    return Some(CraftMatch {
                        recipe_index,
                        kind: CraftMatchKind::Shaped(grid),
                    });
                }
            }
            RecipeKind::Shapeless(shapeless) => {
                if shapeless_recipe_matches_counts(shapeless, &counts) {
                    return Some(CraftMatch {
                        recipe_index,
                        kind: CraftMatchKind::Shapeless,
                    });
                }
            }
        }
    }

    None
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

fn shaped_recipe_matches_grid(recipe: &ShapedRecipe, grid: &NormalizedCraftGrid) -> bool {
    if recipe.width != grid.width || recipe.height != grid.height {
        return false;
    }

    for row in 0..grid.height {
        for col in 0..grid.width {
            let index = row * CRAFT_GRID_SIDE + col;
            let expected = recipe.cells[index];
            let got = grid.cells[index].and_then(|stack| (stack.count > 0).then_some(stack.block_id));
            if expected != got {
                return false;
            }
        }
    }

    true
}

fn active_grid_counts(
    input: &[Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
) -> BTreeMap<i8, u16> {
    let side = normalize_craft_side(craft_grid_side);
    let mut counts = BTreeMap::<i8, u16>::new();
    for index in 0..CRAFT_GRID_SLOTS {
        if !craft_slot_in_side(index, side) {
            continue;
        }
        let Some(stack) = input[index] else {
            continue;
        };
        if stack.count == 0 {
            continue;
        }
        *counts.entry(stack.block_id).or_insert(0) += stack.count as u16;
    }
    counts
}

fn shapeless_recipe_matches_counts(recipe: &ShapelessRecipe, counts: &BTreeMap<i8, u16>) -> bool {
    if counts.is_empty() {
        return false;
    }

    // Only recipe ingredient item types may be present.
    for item_id in counts.keys() {
        if !recipe.required.contains_key(item_id) {
            return false;
        }
    }

    // Required counts must be available, order doesn't matter.
    for (item_id, required_count) in &recipe.required {
        let available = counts.get(item_id).copied().unwrap_or(0);
        if available < *required_count {
            return false;
        }
    }

    true
}

fn consume_shaped_ingredients(
    input: &mut [Option<ItemStack>; CRAFT_GRID_SLOTS],
    grid: &NormalizedCraftGrid,
    recipe: &ShapedRecipe,
) -> bool {
    for row in 0..grid.height {
        for col in 0..grid.width {
            let pattern_index = row * CRAFT_GRID_SIDE + col;
            if recipe.cells[pattern_index].is_none() {
                continue;
            }

            let slot_index =
                (grid.origin_row + row) * CRAFT_GRID_SIDE + (grid.origin_col + col);
            let Some(mut stack) = input.get(slot_index).copied().flatten() else {
                return false;
            };
            if stack.count == 0 {
                return false;
            }
            stack.count -= 1;
            input[slot_index] = (stack.count > 0).then_some(stack);
        }
    }
    true
}

fn consume_shapeless_ingredients(
    input: &mut [Option<ItemStack>; CRAFT_GRID_SLOTS],
    craft_grid_side: usize,
    recipe: &ShapelessRecipe,
) -> bool {
    let side = normalize_craft_side(craft_grid_side);
    let mut needed = recipe.required.clone();

    for index in 0..CRAFT_GRID_SLOTS {
        if !craft_slot_in_side(index, side) {
            continue;
        }
        let Some(mut stack) = input[index] else {
            continue;
        };
        if stack.count == 0 {
            continue;
        }

        let Some(need) = needed.get_mut(&stack.block_id) else {
            continue;
        };
        if *need == 0 {
            continue;
        }

        let take = (*need).min(stack.count as u16) as u8;
        stack.count = stack.count.saturating_sub(take);
        *need = need.saturating_sub(take as u16);
        input[index] = (stack.count > 0).then_some(stack);
    }

    needed.values().all(|v| *v == 0)
}
