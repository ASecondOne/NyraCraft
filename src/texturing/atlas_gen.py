from __future__ import annotations

from pathlib import Path
import re
import sys
from PIL import Image

TILE_SIZE = 16
MAX_COLS = 16
PREFIX_RE = re.compile(r"^(\d+)(?:[_-].*)?\.(png|tga)$", re.IGNORECASE)

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    texture_dir = script_dir / "textures"
    output_dir = script_dir / "atlas_output"
    output_path = output_dir / "atlas.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    indexed_files: list[tuple[int, Path]] = []
    skipped: list[str] = []

    texture_files = sorted([p for p in texture_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".tga"}])
    for file in texture_files:
        m = PREFIX_RE.match(file.name)
        if not m:
            skipped.append(file.name)
            continue
        idx = int(m.group(1))
        if idx <= 0:
            skipped.append(file.name)
            continue
        indexed_files.append((idx, file))

    if not indexed_files:
        print(f"[atlas_gen] no numbered textures (.png/.tga) found in {texture_dir}", file=sys.stderr)
        return 1

    seen: dict[int, Path] = {}
    duplicates: list[tuple[int, Path, Path]] = []
    for idx, file in indexed_files:
        if idx in seen:
            duplicates.append((idx, seen[idx], file))
        else:
            seen[idx] = file

    if duplicates:
        print("[atlas_gen] duplicate numeric prefixes detected:", file=sys.stderr)
        for idx, first, second in duplicates:
            print(f"  index {idx}: {first.name} and {second.name}", file=sys.stderr)
        return 1

    max_index = max(seen.keys())
    rows = (max_index + MAX_COLS - 1) // MAX_COLS
    atlas = Image.new("RGBA", (MAX_COLS * TILE_SIZE, rows * TILE_SIZE), (0, 0, 0, 0))

    for idx, file in sorted(seen.items()):
        with Image.open(file) as img:
            img = img.convert("RGBA")
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
            x = ((idx - 1) % MAX_COLS) * TILE_SIZE
            y = ((idx - 1) // MAX_COLS) * TILE_SIZE
            atlas.paste(img, (x, y))

    atlas.save(output_path)

    print(f"[atlas_gen] generated {output_path} ({atlas.width}x{atlas.height})")
    if skipped:
        print(f"[atlas_gen] skipped {len(skipped)} file(s) without valid numeric prefix or extension")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
