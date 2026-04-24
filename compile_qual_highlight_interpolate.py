from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).parent / "exp-qual-highlight-interpolate"
OUT_PATH = "stacked.png"

ROWS = [
    [
        "nerf_lego_000_0_all_src.png",
        "nerf_lego_001_0_all_dst.png",
        "nerf_lego_002_0_all_dst.png",
        "nerf_lego_003_0_all_dst.png",
        "nerf_lego_004_0_all_dst.png",
    ],
    [
        "nerf_lego_000_0_all_og.png",
        "nerf_lego_001_0_all_deformed.png",
        "nerf_lego_002_0_all_deformed.png",
        "nerf_lego_003_0_all_deformed.png",
        "nerf_lego_004_0_all_deformed.png",
    ],
]


def main() -> None:
    rows = [[Image.open(SRC_DIR / name) for name in row] for row in ROWS]

    w, h = rows[0][0].size
    for row in rows:
        for im in row:
            assert im.size == (w, h), f"size mismatch: {im.filename} {im.size} vs {(w, h)}"

    cols = len(rows[0])
    out = Image.new(rows[0][0].mode, (w * cols, h * len(rows)))
    for y, row in enumerate(rows):
        for x, im in enumerate(row):
            out.paste(im, (x * w, y * h))

    out.save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    main()
