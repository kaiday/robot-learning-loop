"""Grid images: one tile per simulation, green border = success, red = failure, with a caption."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN, RED, INK, PAPER = (27, 150, 90), (214, 60, 60), (11, 11, 11), (252, 252, 251)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(frame: np.ndarray, ok: bool, caption: str, border: int = 8) -> Image.Image:
    img = Image.fromarray(frame)
    w, h = img.size
    cap_h = 46
    out = Image.new("RGB", (w + 2 * border, h + 2 * border + cap_h), GREEN if ok else RED)
    out.paste(img, (border, border))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, h + 2 * border, w + 2 * border, h + 2 * border + cap_h], fill=PAPER)
    for i, line in enumerate(caption.split("\n")[:2]):
        draw.text((border + 2, h + 2 * border + 4 + i * 20), line, fill=INK, font=_font(17))
    return out


def grid(tiles: list[Image.Image], cols: int, title: str = "") -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    gap, head = 6, (40 if title else 0)
    out = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap + head), PAPER)
    if title:
        ImageDraw.Draw(out).text((gap + 4, 9), title, fill=INK, font=_font(22))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out.paste(t, (gap + c * (tw + gap), head + gap + r * (th + gap)))
    return out


def save_grid(items: list[tuple[np.ndarray, bool, str]], path: Path, cols: int = 4, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid([tile(f, ok, cap) for f, ok, cap in items], cols, title).save(path)
    return path
