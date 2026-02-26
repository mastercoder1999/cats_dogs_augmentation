from __future__ import annotations

import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# ---------- CONFIG ----------
PROJECT_ROOT = Path(".")

RAW_BATCH = PROJECT_ROOT / "dataset" / "raw" / "new_batch_01"
CLEAN_DIR = PROJECT_ROOT / "dataset" / "clean"
SPLIT_DIR = PROJECT_ROOT / "dataset" / "split"

CLASSES = ["cats", "dogs"]

# splits (doivent sommer à 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42

# extensions acceptées
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Stats:
    scanned: int = 0
    ok: int = 0
    corrupted: int = 0
    duplicates: int = 0
    copied: int = 0


def is_allowed_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_image(path: Path) -> bool:
    """
    Vérifie que Pillow peut ouvrir l'image (détecte beaucoup de corruptions).
    """
    try:
        with Image.open(path) as im:
            im.verify()
        # re-open pour éviter certains faux positifs
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def ensure_dirs() -> None:
    # clean
    for c in CLASSES:
        (CLEAN_DIR / c).mkdir(parents=True, exist_ok=True)
    # split
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            (SPLIT_DIR / split / c).mkdir(parents=True, exist_ok=True)


def collect_raw_images() -> Dict[str, List[Path]]:
    data: Dict[str, List[Path]] = {}
    for c in CLASSES:
        src = RAW_BATCH / c
        if not src.exists():
            raise FileNotFoundError(f"Missing folder: {src}")
        files = [p for p in src.rglob("*") if is_allowed_image(p)]
        data[c] = sorted(files)
    return data


def copy_clean_dedup(raw_by_class: Dict[str, List[Path]]) -> Tuple[Dict[str, List[Path]], Dict[str, Stats]]:
    """
    Copie vers CLEAN_DIR avec:
      - vérification corruption
      - dédoublonnage exact (sha1 global)
      - renommage stable
    Retourne la liste des fichiers "clean" par classe.
    """
    seen_hashes: Dict[str, Path] = {}
    clean_files: Dict[str, List[Path]] = {c: [] for c in CLASSES}
    stats: Dict[str, Stats] = {c: Stats() for c in CLASSES}

    for c in CLASSES:
        idx = 0
        for p in raw_by_class[c]:
            st = stats[c]
            st.scanned += 1

            if not verify_image(p):
                st.corrupted += 1
                continue

            h = sha1_file(p)
            if h in seen_hashes:
                st.duplicates += 1
                continue

            seen_hashes[h] = p
            st.ok += 1

            idx += 1
            dst_name = f"{c[:-1]}_{idx:06d}{p.suffix.lower()}"  # cat_000001.jpg / dog_000001.png
            dst = CLEAN_DIR / c / dst_name

            # copy2 garde metadata de base
            shutil.copy2(p, dst)
            st.copied += 1
            clean_files[c].append(dst)

    return clean_files, stats


def split_files(clean_by_class: Dict[str, List[Path]]) -> Dict[str, Dict[str, int]]:
    """
    Split en train/val/test en copiant depuis CLEAN_DIR vers SPLIT_DIR.
    (On copie plutôt que déplacer: tu gardes clean comme source fiable.)
    """
    random.seed(SEED)
    report: Dict[str, Dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in ["train", "val", "test"]}

    for c, files in clean_by_class.items():
        files = files[:]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val  # reste

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files_list in splits.items():
            out_dir = SPLIT_DIR / split_name / c
            for src in split_files_list:
                dst = out_dir / src.name
                # si déjà présent, on écrase (re-run friendly)
                shutil.copy2(src, dst)
            report[split_name][c] = len(split_files_list)

    return report


def main() -> None:
    # sanity check ratios
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    ensure_dirs()

    raw = collect_raw_images()
    clean, stats = copy_clean_dedup(raw)
    split_report = split_files(clean)

    # --- print report ---
    print("\n=== CLEANING REPORT ===")
    for c in CLASSES:
        st = stats[c]
        print(
            f"{c}: scanned={st.scanned} ok={st.ok} corrupted={st.corrupted} "
            f"duplicates={st.duplicates} copied={st.copied}"
        )

    print("\n=== SPLIT REPORT ===")
    for split_name in ["train", "val", "test"]:
        cats_n = split_report[split_name]["cats"]
        dogs_n = split_report[split_name]["dogs"]
        print(f"{split_name}: cats={cats_n} dogs={dogs_n} total={cats_n + dogs_n}")

    # quick balance check
    print("\n=== CLASS BALANCE (clean) ===")
    print(f"cats={len(clean['cats'])} dogs={len(clean['dogs'])}")


if __name__ == "__main__":
    main()
