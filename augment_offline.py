from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageOps, ImageFilter


# ---------------- CONFIG ----------------
SEED = 42

SRC = Path("dataset/original")      # <-- set to your original folder (cats/ dogs inside)
DST = Path("dataset/augmented")     # output (cats/ dogs inside)
CLASSES = ["cats", "dogs"]

AUG_PER_IMAGE = 2   # 2 => total ~3x (orig + aug1 + aug2)

# Probabilities (tune here)
P_FLIP = 0.50
P_ROTATE = 0.15      # reduced (was effectively higher)
P_AFFINE = 0.30
P_COLOR = 0.80       # increased
P_BLUR = 0.12        # moderate but not crazy
P_SHARPEN = 0.10
P_NOISE = 0.08

# Strength ranges (tune here)
ROTATE_DEG = 7       # max absolute rotation degrees (smaller)
TRANSLATE_FRAC = 0.05  # max translate fraction (5%)
ZOOM_RANGE = (0.92, 1.06)

BRIGHTNESS = (0.80, 1.20)
CONTRAST   = (0.80, 1.20)
SATURATION = (0.80, 1.25)

# Hue shift implemented by shifting HSV hue (small range)
HUE_SHIFT = (-6, 6)  # degrees in [0..360] mapped to [0..255] hue channel later

BLUR_RADIUS = (0.6, 1.6)
NOISE_STD = (2, 8)  # per-channel std dev (0-255 scale)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------- HELPERS ----------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def ensure_dirs() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (DST / c).mkdir(parents=True, exist_ok=True)


def shift_hue_rgb(img: Image.Image, shift_deg: int) -> Image.Image:
    """
    Shift hue by +/- a few degrees. Works on RGB PIL image.
    """
    if shift_deg == 0:
        return img

    hsv = img.convert("HSV")
    h, s, v = hsv.split()

    # Hue in PIL HSV is 0..255 representing 0..360 degrees.
    shift = int((shift_deg / 360.0) * 255)  # can be negative

    # apply shift with wrap-around
    h = h.point(lambda x: (x + shift) % 256)

    hsv2 = Image.merge("HSV", (h, s, v))
    return hsv2.convert("RGB")


def add_noise(img: Image.Image) -> Image.Image:
    """
    Add light gaussian-like noise using PIL point operations.
    (Fast enough for offline augmentation; keeps it simple.)
    """
    import math

    std = random.uniform(NOISE_STD[0], NOISE_STD[1])

    def jitter(x: int) -> int:
        # Box-Muller approx with two uniforms -> gaussian-ish
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2.0 * math.pi * u2)
        y = int(round(x + z * std))
        return 0 if y < 0 else (255 if y > 255 else y)

    r, g, b = img.split()
    r = r.point(jitter)
    g = g.point(jitter)
    b = b.point(jitter)
    return Image.merge("RGB", (r, g, b))


def random_affine(img: Image.Image) -> Image.Image:
    """
    Small translate + zoom using crop/pad approach (simple and stable).
    """
    w, h = img.size
    # Zoom
    scale = random.uniform(ZOOM_RANGE[0], ZOOM_RANGE[1])
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # If zoomed in -> crop center-ish with small random offset
    if scale > 1.0:
        max_dx = int((new_w - w) * 0.5)
        max_dy = int((new_h - h) * 0.5)
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        left = (new_w - w) // 2 + dx
        top  = (new_h - h) // 2 + dy
        return resized.crop((left, top, left + w, top + h))

    # If zoomed out -> pad to original size with small random offset
    else:
        canvas = Image.new("RGB", (w, h))
        max_dx = int((w - new_w) * 0.5)
        max_dy = int((h - new_h) * 0.5)
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        left = (w - new_w) // 2 + dx
        top  = (h - new_h) // 2 + dy
        canvas.paste(resized, (left, top))
        return canvas


def augment(img: Image.Image) -> Image.Image:
    # Flip
    if random.random() < P_FLIP:
        img = ImageOps.mirror(img)

    # Rotate (less often)
    if random.random() < P_ROTATE:
        angle = random.uniform(-ROTATE_DEG, ROTATE_DEG)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Affine-like (zoom/translate)
    if random.random() < P_AFFINE:
        img = random_affine(img)

    # Color / Tone (more often)
    if random.random() < P_COLOR:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST))
        img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION))

        hue = random.randint(HUE_SHIFT[0], HUE_SHIFT[1])
        img = shift_hue_rgb(img, hue)

    # Blur vs Sharpen (rare-ish, and not both usually)
    r = random.random()
    if r < P_BLUR:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*BLUR_RADIUS)))
    elif r < (P_BLUR + P_SHARPEN):
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    # Noise (rare)
    if random.random() < P_NOISE:
        img = add_noise(img)

    return img


def main() -> None:
    random.seed(SEED)
    ensure_dirs()

    print("Using source:", SRC.resolve())
    total_in = 0
    for c in CLASSES:
        found = [p for p in (SRC / c).rglob("*") if is_image(p)]
        print(f"{c}: found {len(found)} images in {SRC / c}")
        total_in += len(found)
    if total_in == 0:
        print("[ERROR] No input images found. Check SRC path and cats/dogs folders.")
        return

    for c in CLASSES:
        files = sorted([p for p in (SRC / c).rglob("*") if is_image(p)])
        for i, p in enumerate(files, start=1):
            with Image.open(p) as im:
                im = im.convert("RGB")

                # always save original copy into augmented set
                out_orig = DST / c / f"{c[:-1]}_{i:06d}_orig.jpg"
                im.save(out_orig, quality=95)

                # generate N augmentations
                for k in range(1, AUG_PER_IMAGE + 1):
                    aug = augment(im)
                    out_aug = DST / c / f"{c[:-1]}_{i:06d}_aug{k}.jpg"
                    aug.save(out_aug, quality=95)

    # quick output count
    for c in CLASSES:
        out_n = len([p for p in (DST / c).iterdir() if p.is_file()])
        print(f"{c}: wrote {out_n} files into {DST / c}")
    print("Done:", DST)


if __name__ == "__main__":
    main()
