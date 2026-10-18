#!/usr/bin/env python3
"""Cuts the per-community ground textures in Sources/Flora/Assets/ground from CC0 ambientCG
materials (docs/flora.md §9).

    python3 tools/ground_textures.py MATERIALS_DIR Sources/Flora/Assets/ground

MATERIALS_DIR holds the unzipped 1K-JPG downloads (ID_1K-JPG_Color.jpg, _NormalGL.jpg and
_Displacement.jpg for each ID in LAYERS), from https://ambientcg.com/get?file=ID_1K-JPG.zip.
Needs Pillow and NumPy.

Each layer becomes two 512² PNGs, tileable as the sources are (resized with wrap-around):
  ground-NAME.png         sRGB colour, its mean set to the target in linear light;
  ground-NAME-normal.png  the OpenGL-convention normal's x and y (x right, y up the texture;
                          z follows from them) in red and green, and the displacement
                          (0 low … 255 high) in blue, for height blending. No alpha: the
                          app decodes through premultiplied CoreGraphics, which would black
                          out the colour of texels with zero alpha.
Layer order is the bundle's biome channel order (Look.communities), then bare rock.
"""
import os
import sys

import numpy as np
from PIL import Image

SIZE = 512

# Layer name, ambientCG material, target mean colour (sRGB, applied in linear light).
# Targets are ground as seen up close, kept near the luminance of the community's canopy
# colour in Look.communities so the fade to it far away doesn't jump in brightness.
LAYERS = [
    ('strand', 'Ground033', (160, 150, 124)),        # pale calcareous sand
    ('rocky-shore', 'Gravel024', (62, 60, 56)),      # black basalt clinker (ʻaʻā)
    ('marsh', 'Ground051', (72, 68, 52)),            # wet mud
    ('dry-forest', 'Ground067', (118, 84, 60)),      # dry red-brown soil, litter, stones
    ('mesic-forest', 'Ground023', (84, 70, 52)),     # brown leaf litter over soil
    ('wet-forest', 'Ground024', (52, 58, 42)),       # dark wet litter with moss
    ('riparian', 'Ground038', (50, 50, 40)),         # dark wet litter, twigs, fern debris
    ('cloud-shrubland', 'Ground037', (62, 78, 46)),  # mossy ground
    ('bog', 'Moss003', (72, 86, 50)),                # wet moss over peat
    ('cliff', 'Rock029', (116, 72, 50)),             # weathered red basalt, lateritic soil
    ('montane', 'Ground068', (64, 72, 48)),          # moss over red soil and rock
    ('rock', 'Rock031', (62, 60, 58)),               # bare basalt for steep slopes (Look.rock)
]

AUTHOR = 'Lennart Demes (ambientCG)'
LICENCE = '[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)'


def decode(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def encode(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def resize_tiled(img, size=SIZE):
    """Resizes a tileable image so it still tiles: the filter sees the wrapped neighbours."""
    w, h = img.size
    big = Image.new(img.mode, (3 * w, 3 * h))
    for i in range(3):
        for j in range(3):
            big.paste(img, (i * w, j * h))
    big = big.resize((3 * size, 3 * size), Image.LANCZOS)
    return big.crop((size, size, 2 * size, 2 * size))


def main(src, out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for name, material, target in LAYERS:
        def path(kind):
            return os.path.join(src, f'{material}_1K-JPG_{kind}.jpg')
        colour = np.asarray(resize_tiled(Image.open(path('Color')).convert('RGB')), dtype=np.float64)
        linear = decode(colour)
        mean = linear.reshape(-1, 3).mean(axis=0)
        linear *= decode(np.array(target, dtype=np.float64)) / np.maximum(mean, 1e-4)
        # Crevices photographed black: nothing on the ground reflects much under 1%.
        linear = np.maximum(linear, 0.008)
        height = np.asarray(resize_tiled(Image.open(path('Displacement')).convert('L')), dtype=np.float64)
        lo, hi = np.percentile(height, [1, 99])
        height = np.clip((height - lo) / max(hi - lo, 1e-3), 0, 1) * 255.0
        Image.fromarray(encode(linear).round().astype(np.uint8), 'RGB').save(
            os.path.join(out, f'ground-{name}.png'), optimize=True)

        normal = np.asarray(resize_tiled(Image.open(path('NormalGL')).convert('RGB')), dtype=np.float64) / 127.5 - 1
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
        packed = np.dstack([(normal[..., :2] + 1) * 127.5, height]).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(packed, 'RGB').save(os.path.join(out, f'ground-{name}-normal.png'), optimize=True)
        url = f'https://ambientcg.com/view?id={material}'
        rows.append(f'| `ground-{name}.png`, `ground-{name}-normal.png` | [{material}]({url}) | {AUTHOR} | {LICENCE} |'
                    f' 512², mean set to sRGB {target}; displacement in the normal map\'s blue. |')
        print(name, material, 'source mean sRGB', tuple(int(v) for v in encode(mean).round()))
    with open(os.path.join(out, 'CREDITS.md'), 'w') as f:
        f.write('# Ground texture credits\n\n'
                'Cut by `tools/ground_textures.py` from the ambientCG materials below (1K-JPG\n'
                'downloads): resized to 512² with wrap-around, colour mean set in linear light,\n'
                'displacement packed into the normal map\'s blue channel. All CC0.\n\n'
                '| Texture | Source | Author | Licence | Notes |\n|---|---|---|---|---|\n')
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
