"""Reorganize data/santos/ from per-year folders into the flat
images/ + labels/ layout that src/data/dataset.py and configs/data/*.yaml expect.

The year metadata is preserved in each filename's suffix
(e.g. ``0001_2010.jpg``), so the year folders are redundant.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent / "data" / "santos"
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
SKIP = {"desktop.ini", ".DS_Store", "README.md"}


def main() -> int:
    if not ROOT.is_dir():
        print(f"error: {ROOT} not found", file=sys.stderr)
        return 1

    year_dirs = sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name.isdigit())
    if not year_dirs:
        print("nothing to do: no year subdirectories found")
        return 0

    IMAGES.mkdir(exist_ok=True)
    LABELS.mkdir(exist_ok=True)

    moved_img = moved_lbl = skipped = 0
    for ydir in year_dirs:
        for f in ydir.iterdir():
            if f.name in SKIP:
                f.unlink()
                skipped += 1
                continue
            ext = f.suffix.lower()
            if ext == ".jpg":
                dst = IMAGES / f.name
                moved_img += 1
            elif ext == ".txt":
                dst = LABELS / f.name
                moved_lbl += 1
            else:
                print(f"warn: unexpected file {f}", file=sys.stderr)
                continue
            if dst.exists():
                print(f"error: destination already exists: {dst}", file=sys.stderr)
                return 1
            shutil.move(str(f), str(dst))

        leftovers = list(ydir.iterdir())
        if leftovers:
            print(f"warn: {ydir} still has files: {leftovers}", file=sys.stderr)
        else:
            ydir.rmdir()

    stems_img = {p.stem for p in IMAGES.iterdir()}
    stems_lbl = {p.stem for p in LABELS.iterdir()}
    orphan_img = stems_img - stems_lbl
    orphan_lbl = stems_lbl - stems_img

    print(f"moved {moved_img} images -> {IMAGES.relative_to(ROOT.parent.parent)}")
    print(f"moved {moved_lbl} labels -> {LABELS.relative_to(ROOT.parent.parent)}")
    print(f"removed {skipped} junk files (desktop.ini / .DS_Store)")
    print(f"images without label: {len(orphan_img)}")
    print(f"labels without image: {len(orphan_lbl)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
