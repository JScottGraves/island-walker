#!/usr/bin/env python3
"""Cuts the native-tree textures in Sources/Flora/Assets from CC-licensed photos.

    python3 tools/flora_textures.py PHOTOS_DIR Sources/Flora/Assets

PHOTOS_DIR holds the downloaded Wikimedia Commons files named as in SOURCES below
(credits.json beside them records URL, author and licence; Assets/CREDITS.md is written
from it). Needs Pillow, NumPy and SciPy.

Leaf, flower and frond cards are keyed off plain backgrounds (white herbarium vouchers,
dark shade) and colour-corrected towards the live plant; bark is cropped, flattened
(lighting gradients removed) and made tileable. Where no photo cuts cleanly the texture
is drawn here instead (loulu fan, hāpuʻu trunk, ʻaʻaliʻi capsules), and CREDITS.md says so.
All output is 512² sRGB PNG; alpha cards keep their colour bled into the clear texels so
filtering and mips don't fringe.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SIZE = 512
rng = np.random.default_rng(42)

# The photo behind each texture: file in PHOTOS_DIR and crop box (fractions x0, y0, x1, y1).
SOURCES = {
    'ohia-leaves': ('Starr-110817-8234-Metrosideros_polymorpha-voucher-Kulamanu_Kula-Maui_24985192002.jpg', (0.14, 0.43, 0.78, 0.99)),
    'ohia-flowers': ('Starr-030329-0001-Metrosideros_polymorpha-red_flower-Auwahi-Maui_24547066961_.jpg', (0.17, 0.0, 0.97, 1.0)),
    'koa-phyllodes': ('Acacia_koa_with_phyllode_between_the_branch_and_the_compound_leaves.jpg', (0.075, 0.085, 0.56, 0.62)),
    'naupaka-leaves': ('Starr-110209-0697-Scaevola_taccada-leaves-Resort_Management_Group_Nursery_Kihei-.jpg', None),
    'aalii-leaves': ('Starr-041227-9001-Dodonaea_viscosa-plant_voucher-Moaulanui-Kahoolawe_24722924315.jpg', (0.12, 0.08, 0.72, 0.56)),
    'hapuu-frond': ('Cibotium_chamissoi_4969582483_.jpg', (0.0, 0.12, 0.885, 0.96)),
    'loulu-fan': ('Starr-130411-4021-Pritchardia_martii-frond_underside_with_silvery_fur-Pololei_Ha.jpg', (0.30, 0.55, 0.75, 0.85)),
    'ohia-bark': ('Starr-110713-7131-Metrosideros_polymorpha-bark-Koolau_Gap-Maui_24468453334_.jpg', (0.015, 0.66, 0.19, 0.99)),
    'koa-bark': ('Starr-180312-2513-Acacia_koa-trunk_bark-Kahikinui-Maui_40526664794_.jpg', (0.0, 0.02, 0.28, 0.98)),
    'loulu-bark': ('Starr-130411-3984-Pritchardia_munroi-trunk-Pololei_Haiku-Maui_25091753622_.jpg', (0.29, 0.12, 0.36, 0.9)),
    'aalii-bark': ('Dodonaea_viscosa_5188013052_.jpg', (0.27, 0.08, 0.68, 0.9)),
}

# Mean linear albedo and hue each texture is corrected to (sRGB swatch of the live plant).
TARGETS = {
    'ohia-leaves': (48, 74, 40),      # dark glossy green; the voucher is dried olive
    'ohia-flowers': None,             # live photo: kept as shot
    'koa-phyllodes': (82, 98, 64),    # grey-green
    'naupaka-leaves': (84, 122, 56),  # bright, glossy yellow-green
    'aalii-leaves': (66, 92, 44),
    'hapuu-frond': (88, 132, 52),     # light green
    'ohia-bark': (112, 100, 86),      # grey-brown, flaky
    'koa-bark': (124, 110, 94),        # rough grey-brown
    'loulu-bark': (128, 124, 114),    # smooth grey with ring scars
    'aalii-bark': (86, 70, 58),       # stringy dark brown
}

# Fan leaf layout shared with Flora's palm builder: hastula at (0.5, 0.16) in the card,
# segments spread ±100° from the card's up axis; a point at radius ρ (0…1) and angle θ
# sits at u = 0.5 + 0.49 ρ sin θ, v = 0.16 + 0.83 ρ cos θ (v measured from the bottom).
FAN_HALF_ANGLE = math.radians(100)
FAN_ORIGIN = (0.5, 0.16)
FAN_SCALE = (0.49, 0.83)


def decode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def encode(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def load(photos, name):
    file, box = SOURCES[name]
    im = Image.open(os.path.join(photos, file)).convert('RGB')
    if box:
        w, h = im.size
        im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    return decode(np.asarray(im) / 255.0)


def edge_fade(alpha, width=0.04):
    """Fades a key out towards the crop's edges, where the photo frame cuts leaves off."""
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum(np.minimum(xx, w - 1 - xx) / w, np.minimum(yy, h - 1 - yy) / h)
    return alpha * smoothstep(0, width, d)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def luminance(c):
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


def fit(rgb, alpha, size=SIZE, pad=0.02):
    """Scales an RGBA cut-out to fill a size² card (keeping aspect), centred, with a margin."""
    ys, xs = np.nonzero(alpha > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb, alpha = rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    h, w = alpha.shape
    s = (1 - 2 * pad) * size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out_rgb = np.zeros((size, size, 3))
    out_a = np.zeros((size, size))
    # Resize premultiplied so edges don't pick up the background colour.
    pre = np.concatenate([rgb * alpha[..., None], alpha[..., None]], axis=2)
    pre = np.stack([np.asarray(Image.fromarray(pre[..., k].astype(np.float32), 'F').resize((nw, nh), Image.LANCZOS))
                    for k in range(4)], axis=2)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    a = np.clip(pre[..., 3], 0, 1)
    out_a[oy:oy + nh, ox:ox + nw] = a
    out_rgb[oy:oy + nh, ox:ox + nw] = np.clip(pre[..., :3], 0, None) / np.maximum(a, 1e-4)[..., None]
    return out_rgb, out_a


def bleed(rgb, alpha):
    """Fills clear texels with the colour of the nearest opaque ones, so linear filtering and
    mips never average in black."""
    solid = alpha > 0.5
    if not solid.any():
        return rgb
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    filled = rgb[iy, ix]
    w = smoothstep(0.0, 0.5, alpha)[..., None]
    return filled * (1 - w) + rgb * w


def recolour(rgb, alpha, target, keep=0.35, contrast=1.0):
    """Moves a card's mean colour to `target` (sRGB swatch), keeping its light and dark detail
    (raised to `contrast`) and `keep` of its own hue variation."""
    if target is None:
        return rgb
    t = decode(np.array(target) / 255.0)
    w = alpha[..., None]
    mean = (rgb * w).sum(axis=(0, 1)) / max(w.sum(), 1)
    lum = luminance(rgb)
    mean_lum = max(luminance(mean[None, None, :])[0, 0], 1e-4)
    detail = (np.maximum(lum, 1e-5) / mean_lum)[..., None] ** contrast
    tinted = t * detail
    own = rgb * (luminance(t[None, None, :])[0, 0] / mean_lum) * (detail / np.maximum(lum / mean_lum, 1e-5)[..., None])
    return np.clip(tinted * (1 - keep) + own * keep, 0, 1)


def key_white(rgb, lo=0.06, hi=0.16):
    """Alpha off a white or pale-grey background (herbarium vouchers): distance from the
    border's median colour, with soft shadows (grey, low chroma) left out."""
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    darker = luminance(bg[None, None, :]) - luminance(rgb)
    d = np.maximum(chroma * 1.6, darker * 0.7)
    a = smoothstep(lo, hi, d)
    a = ndimage.grey_opening(a, size=(3, 3))
    return a


def write(rgb, alpha, path):
    if alpha is not None:
        rgb = bleed(rgb, alpha)
        img = np.concatenate([encode(rgb), np.clip(alpha, 0, 1)[..., None]], axis=2)
        Image.fromarray((img * 255 + 0.5).astype(np.uint8), 'RGBA').save(path, optimize=True)
    else:
        Image.fromarray((encode(rgb) * 255 + 0.5).astype(np.uint8), 'RGB').save(path, optimize=True)
    print('wrote', path)


# ---- Cards ----

def ohia_leaves(photos):
    rgb = load(photos, 'ohia-leaves')
    a = edge_fade(key_white(rgb))
    rgb, a = fit(rgb, a)
    return recolour(rgb, a, TARGETS['ohia-leaves']), a


def ohia_flowers(photos):
    rgb = load(photos, 'ohia-flowers')
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    redness = (r - np.maximum(g, b)) / np.maximum(r, 1e-3)
    a = smoothstep(0.35, 0.6, redness) * smoothstep(0.02, 0.06, r)
    a = ndimage.grey_opening(a, size=(2, 2))
    # Keep the pompom, not stray red specks in the background.
    labels, n = ndimage.label(a > 0.3)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(a), labels, range(1, n + 1))
        a *= ndimage.binary_dilation(labels == (np.argmax(sizes) + 1), iterations=6)
    rgb, a = fit(rgb, a)
    return rgb, a


def koa_phyllodes(photos):
    rgb = load(photos, 'koa-phyllodes')
    a = edge_fade(key_white(rgb, 0.05, 0.13))
    rgb, a = fit(rgb, a)
    return recolour(rgb, a, TARGETS['koa-phyllodes'], keep=0.5), a


def naupaka_leaves(photos):
    file, _ = SOURCES['naupaka-leaves']
    im = Image.open(os.path.join(photos, file)).convert('RGB')
    w, h = im.size
    cx, cy, r = 0.455 * w, 0.565 * h, 0.47 * h
    im = im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((SIZE, SIZE), Image.LANCZOS)
    rgb = decode(np.asarray(im) / 255.0)
    hsv = np.asarray(im.convert('HSV')) / 255.0
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    green = smoothstep(40, 60, hue) * (1 - smoothstep(150, 175, hue))
    a = green * smoothstep(0.15, 0.3, sat) * smoothstep(0.3, 0.45, val)
    a = ndimage.grey_closing(ndimage.grey_opening(a, size=(3, 3)), size=(3, 3))
    # A round rosette: fade out before the photo's edge cuts the leaves off.
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1) - 0.5
    rr = np.hypot(xx, yy) * 2
    a *= 1 - smoothstep(0.8, 0.98, rr + 0.06 * np.sin(np.arctan2(yy, xx) * 9))
    return recolour(rgb, a, TARGETS['naupaka-leaves'], keep=0.5), a


def capsule_cluster(draw, cx, cy, s, colour, rng):
    """ʻAʻaliʻi capsules: papery, two- or three-winged, a few per stalk."""
    for _ in range(rng.integers(3, 7)):
        ang = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.3, 1.0) * s * 1.6
        x, y = cx + d * math.cos(ang), cy + d * math.sin(ang)
        draw.line([(cx, cy), (x, y)], fill=(70, 60, 40, 255), width=max(1, int(s * 0.12)))
        tint = tuple(int(np.clip(c * rng.uniform(0.75, 1.15), 0, 255)) for c in colour)
        rot = rng.uniform(0, math.pi)
        for k in range(rng.integers(2, 4)):
            a = rot + k * math.pi
            wx, wy = x + 0.42 * s * math.cos(a), y + 0.42 * s * math.sin(a)
            draw.ellipse([wx - 0.45 * s, wy - 0.45 * s, wx + 0.45 * s, wy + 0.45 * s], fill=tint + (235,))
        draw.ellipse([x - 0.16 * s, y - 0.16 * s, x + 0.16 * s, y + 0.16 * s], fill=(60, 40, 30, 255))


def aalii_leaves(photos):
    rgb = load(photos, 'aalii-leaves')
    a = key_white(rgb)
    rgb, a = fit(rgb, a, pad=0.08)
    rgb = recolour(rgb, a, TARGETS['aalii-leaves'], keep=0.45)
    # Drawn capsule clusters at the twig ends (no photo cuts cleanly): red-brown, papery.
    over = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    edge = ndimage.binary_erosion(a > 0.5, iterations=2) ^ (a > 0.5)
    ys, xs = np.nonzero(edge)
    for _ in range(9):
        k = rng.integers(len(xs))
        capsule_cluster(d, xs[k], ys[k], 9, (150, 58, 42), rng)
    o = np.asarray(over) / 255.0
    oa = o[..., 3]
    rgb = rgb * (1 - oa[..., None]) + decode(o[..., :3]) * oa[..., None]
    return rgb, np.maximum(a, oa)


def hapuu_frond(photos):
    """A whole Cibotium frond, base at the bottom of the card and tip at the top: pinnae keyed
    off dark shade (one side of the rachis, mirrored), stacked so the frond carries ~20
    pairs, and cut to a long, narrow lanceolate outline (widest a third of the way up,
    pinnae shortening to the tip) with soft ends, so no straight row edges show."""
    rgb = load(photos, 'hapuu-frond')
    g = rgb[..., 1]
    a = smoothstep(0.02, 0.06, g) * smoothstep(0.0, 0.03, g - rgb[..., 2])
    a = ndimage.grey_opening(a, size=(2, 2))
    half = SIZE // 2
    def resize(x, w, h):
        return np.asarray(Image.fromarray(x.astype(np.float32), 'F').resize((w, h), Image.LANCZOS))
    # One strip of pinnae, half the card wide, a third of it tall, feathered at its ends.
    sh = SIZE // 3 + 8
    strip_a = np.clip(resize(a, half - 5, sh), 0, 1)
    strip = np.stack([resize(rgb[..., k] * a, half - 5, sh) for k in range(3)], axis=2) / np.maximum(strip_a, 1e-4)[..., None]
    feather = np.minimum(np.arange(sh), np.arange(sh)[::-1])[:, None]
    strip_a *= smoothstep(0, 10, feather)
    left = np.zeros((SIZE, half - 5, 3)); left_a = np.zeros((SIZE, half - 5))
    for k, y in enumerate(range(-6, SIZE, sh - 8)):
        y0, y1 = max(y, 0), min(y + sh, SIZE)
        if y1 <= y0:
            continue
        src = slice(y0 - y, y1 - y)
        take = strip_a[src] > left_a[y0:y1]
        left[y0:y1] = np.where(take[..., None], strip[src], left[y0:y1])
        left_a[y0:y1] = np.maximum(left_a[y0:y1], strip_a[src])
    out = np.zeros((SIZE, SIZE, 3)); out_a = np.zeros((SIZE, SIZE))
    out[:, :half - 5], out_a[:, :half - 5] = left, left_a
    out[:, half + 5:], out_a[:, half + 5:] = left[:, ::-1], left_a[:, ::-1]
    # Lanceolate outline: pinna reach by height (v = 0 at the base, row 0 is the top).
    v = 1 - (np.arange(SIZE) + 0.5) / SIZE
    reach = np.where(v < 0.33, 0.55 + 0.45 * smoothstep(0.0, 0.33, v), 1 - 0.94 * smoothstep(0.33, 1.0, v) ** 0.9)
    reach *= smoothstep(0.0, 0.05, v)  # bare stipe at the very base
    across = np.abs((np.arange(SIZE) + 0.5) / SIZE - 0.5) * 2
    edge = reach[:, None] - across[None, :]
    out_a *= smoothstep(-0.02, 0.04, edge)
    # Rachis: a pale stripe down the middle, thinning to the tip.
    rw = (6 * (1 - 0.6 * v)).astype(int)
    rachis = decode(np.array([150, 160, 96]) / 255.0)
    for y in range(SIZE):
        out[y, half - rw[y]:half + rw[y]] = rachis
        out_a[y, half - rw[y]:half + rw[y]] = 1.0 if v[y] < 0.985 else 0.0
    return recolour(out, out_a, TARGETS['hapuu-frond'], keep=0.5), out_a


def loulu_fan(photos):
    """Drawn: Pritchardia fans are a stiff disc of pleated segments split at the tips, which
    no field photo isolates cleanly. Colours are sampled from a P. martii leaf photo."""
    sample = load(photos, 'loulu-fan')
    upper = sample.reshape(-1, 3).mean(axis=0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    u, v = (xx + 0.5) / SIZE, 1 - (yy + 0.5) / SIZE
    px, py = (u - FAN_ORIGIN[0]) / FAN_SCALE[0], (v - FAN_ORIGIN[1]) / FAN_SCALE[1]
    rho, theta = np.hypot(px, py), np.arctan2(px, py)
    segments = 56
    s = (theta + FAN_HALF_ANGLE) / (2 * FAN_HALF_ANGLE) * segments
    k, f = np.floor(s), s - np.floor(s)
    # Per-segment length and split depth vary a little; tips are split and ragged.
    jitter = np.sin(k * 12.9898) * 43758.5453
    jitter = jitter - np.floor(jitter)
    length = 0.93 + 0.07 * jitter
    split = 0.62 + 0.12 * np.sin(k * 7.13 + 1.0)
    tip = np.where(rho > split * length, np.abs(f - 0.5) * 2 < 1 - smoothstep(split * length, length, rho) * 0.9, True)
    inside = (np.abs(theta) < FAN_HALF_ANGLE) & (rho < length) & (rho > 0.03) & tip
    a = inside.astype(np.float64)
    a = ndimage.gaussian_filter(a, 0.6)
    # Pleats: each segment folds along its midrib (light ridge, darker valleys).
    fold = 1 - np.abs(f - 0.5) * 2
    shade = 0.78 + 0.32 * fold ** 1.5 - 0.12 * (np.abs(f - 0.5) > 0.46)
    veins = 1 - 0.06 * (np.sin(f * math.pi * 10) > 0.8)
    grain = 1 + 0.06 * ndimage.gaussian_filter(rng.standard_normal((SIZE, SIZE)), 1.2) * 4
    rgb = upper[None, None, :] * (shade * veins * grain)[..., None]
    # Paler and yellower towards the hastula; tips a little browner.
    rgb *= (1 + 0.25 * (1 - smoothstep(0.0, 0.35, rho)))[..., None]
    rgb = rgb * (1 - 0.25 * smoothstep(0.8, 1.0, rho))[..., None] + 0.25 * smoothstep(0.8, 1.0, rho)[..., None] * decode(np.array([130, 120, 80]) / 255.0)
    rgb = recolour(rgb, a, (84, 108, 70), keep=0.6)
    return rgb, a


# ---- Bark ----

def flatten(rgb, sigma):
    """Removes lighting gradients (trunk curvature, sun patches): divides out a heavy blur."""
    lum = luminance(rgb)
    low = ndimage.gaussian_filter(lum, sigma)
    return rgb * (lum.mean() / np.maximum(low, 1e-4))[..., None]


def make_tileable(rgb, blend=0.18):
    """Classic offset-and-blend: the image shifted by half its size is crossfaded in along
    the seams, so the texture wraps in both directions."""
    h, w, _ = rgb.shape
    shifted = np.roll(np.roll(rgb, h // 2, axis=0), w // 2, axis=1)
    yy, xx = np.mgrid[0:h, 0:w]
    dx = np.minimum(xx, w - 1 - xx) / (w * blend)
    dy = np.minimum(yy, h - 1 - yy) / (h * blend)
    m = np.clip(np.minimum(dx, dy), 0, 1)
    m = (m * m * (3 - 2 * m))[..., None]
    return rgb * m + shifted * (1 - m)


def bark(photos, name):
    rgb = load(photos, name)
    rgb = flatten(rgb, min(rgb.shape[:2]) / 3)
    # Keep the photo's aspect: width to the card, then the middle SIZE rows (u runs around
    # the trunk, v along it).
    h, w, _ = rgb.shape
    im = Image.fromarray((encode(np.clip(rgb, 0, 1)) * 255).astype(np.uint8))
    nh = max(SIZE, round(h * SIZE / w))
    im = im.resize((SIZE, nh), Image.LANCZOS)
    top = (nh - SIZE) // 2
    im = im.crop((0, top, SIZE, top + SIZE))
    rgb = decode(np.asarray(im) / 255.0)
    rgb = make_tileable(rgb)
    # Recolour with the full alpha (every texel counts).
    return recolour(rgb, np.ones(rgb.shape[:2]), TARGETS[name], keep=0.2, contrast=0.6)


def periodic_noise(shape, scale_u, scale_v, seed):
    """Tileable noise: white noise shaped in the Fourier domain (anisotropic Gaussian), so
    it wraps at the edges by construction."""
    r = np.random.default_rng(seed)
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    spectrum = np.exp(-((fx * scale_u) ** 2 + (fy * scale_v) ** 2))
    n = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal(shape)) * spectrum))
    return (n - n.mean()) / (n.std() + 1e-9)


def hapuu_bark():
    """Drawn: hāpuʻu trunks are a mat of dark, wiry roots and fibres, with golden pulu
    wool near the crown (the shader tints the top with the pulu colour)."""
    shape = (SIZE, SIZE)
    fibres = 0.6 * periodic_noise(shape, 14, 260, 1) + 0.4 * periodic_noise(shape, 6, 90, 2)
    strands = np.abs(periodic_noise(shape, 30, 400, 3))
    dark = decode(np.array([30, 23, 18]) / 255.0)
    mid = decode(np.array([72, 52, 36]) / 255.0)
    t = smoothstep(-1.0, 1.4, fibres + 0.6 * (strands < 0.25))
    rgb = dark * (1 - t[..., None]) + mid * t[..., None]
    rgb *= (0.8 + 0.4 * smoothstep(-2, 2, periodic_noise(shape, 30, 30, 4)))[..., None]
    return rgb


def credits_md(photos, out):
    credits = json.load(open(os.path.join(photos, 'credits.json')))
    lines = ['# Flora texture credits', '',
             'Textures in this folder are cut from the photos below by `tools/flora_textures.py`',
             '(cropped, keyed, colour-corrected, resized to 512² and made tileable). Each keeps its',
             'source licence; the CC BY-SA ones are shared alike under the same licence.', '',
             '| Texture | Source | Author | Licence | Notes |', '|---|---|---|---|---|']
    notes = {
        'ohia-leaves': 'Dried voucher keyed off white; recoloured to the live leaf.',
        'ohia-flowers': 'Keyed on redness.',
        'koa-phyllodes': 'Keyed off white (phyllodes only, not the juvenile compound leaf).',
        'naupaka-leaves': 'Rosette keyed off the ground; round fade drawn.',
        'aalii-leaves': 'Voucher keyed off white. **Capsule clusters drawn procedurally.**',
        'hapuu-frond': 'Pinnae keyed off dark shade, mirrored about a drawn rachis (C. chamissoi).',
        'loulu-fan': '**Drawn procedurally**; only the colour is sampled from this photo of P. martii.',
        'ohia-bark': 'Crop, flattened, tileable.',
        'koa-bark': 'Crop, flattened, tileable.',
        'loulu-bark': 'Crop, flattened, tileable. Trunk of P. munroi (the Oʻahu loulu trunks are alike).',
        'aalii-bark': 'Crop, flattened, tileable.',
    }
    for name, (file, _) in SOURCES.items():
        c = credits[file]
        licence_url = c['licenseUrl'] or c['page'] + '#Licensing'
        lines.append(f"| `{name}.png` | [{c['title'][5:]}]({c['page']}) | {c['artist']} | "
                     f"[{c['license']}]({licence_url}) | {notes[name]} |")
    lines.append('| `hapuu-bark.png` | — | — | — | **Drawn procedurally** (no photo of the fibrous trunk cut cleanly). |')
    lines.append('')
    open(os.path.join(out, 'CREDITS.md'), 'w').write('\n'.join(lines))
    print('wrote', os.path.join(out, 'CREDITS.md'))


def main():
    photos, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    cards = {'ohia-leaves': ohia_leaves, 'ohia-flowers': ohia_flowers, 'koa-phyllodes': koa_phyllodes,
             'naupaka-leaves': naupaka_leaves, 'aalii-leaves': aalii_leaves, 'hapuu-frond': hapuu_frond,
             'loulu-fan': loulu_fan}
    only = sys.argv[3:]
    for name, f in cards.items():
        if only and name not in only:
            continue
        rgb, a = f(photos)
        write(rgb, a, os.path.join(out, name + '.png'))
    for name in ('ohia-bark', 'koa-bark', 'loulu-bark', 'aalii-bark'):
        if only and name not in only:
            continue
        write(bark(photos, name), None, os.path.join(out, name + '.png'))
    if not only or 'hapuu-bark' in only:
        write(hapuu_bark(), None, os.path.join(out, 'hapuu-bark.png'))
    credits_md(photos, out)


if __name__ == '__main__':
    main()
