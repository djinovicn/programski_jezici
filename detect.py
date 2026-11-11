"""Detekcija feromagnetnih cestica na mikroskopskim slikama (OpenCV).

Pokretanje:
    python detect.py <folder_sa_slikama> <izlazni_folder>

Za svaku sliku X.jpg pravi X.csv sa kolonama:
    centroid_x, centroid_y, area_px, major_axis_px, minor_axis_px, orientation_deg

Postupak:
    1. mapa       - Lab boje: cestice su svijetle i crvenkaste, izolacija tamna
    2. maska      - Otsu prag + morfologija
    3. jezgra     - strozi prag, iz kojeg se izrezu tamne pukotine (black-hat)
    4. watershed  - jezgra se sire nazad do ivica maske
    5. podjela    - spojene cestice se razdvajaju na najuzem mjestu
    6. mjerenje   - povrsina i orijentisani pravougaonik (cv2.minAreaRect)
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# 1. mapa
W_A = 1.0            # koliko se racuna crvenkastost (Lab a) uz svjetlinu (Lab L)
BLUR = 2.0           # sigma Gausovog zamucenja

# 2. maska
THR_OFFSET = 15      # pomjeraj Otsu praga
OPEN_K = 3           # otvaranje (sitan sum)
CLOSE_K = 5          # zatvaranje (sitne rupe)

# 3. jezgra
HI_OFFSET = 10       # jezgra traze jos ovoliko svjetlije piksele
CORE_BLUR = 1.5      # dodatno zamucenje da ogrebotine ne smetaju
CRACK_K = 21         # kernel za black-hat, mora biti siri od pukotine
CRACK_T = 28         # koliko pukotina mora biti tamnija od okoline
CORE_OPEN = 5
CORE_MIN = 250       # manja jezgra se zanemaruju

# 5. podjela
SOLIDITY = 0.8       # ispod ovoga regija vjerovatno sadrzi vise cestica
SPLIT_MIN_AREA = 300
PIECE_MIN = 300      # najmanji dio koji se racuna kao posebna cestica
MAX_DEPTH = 6

# 6. mjerenje
MIN_AREA = 80


def ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# ---------------------------------------------------------------- 1. mapa
def feature_map(img):
    """Siva slika 0..255: cestice svijetle, izolacija tamna."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    f = lab[..., 0] + W_A * (lab[..., 1] - 128)
    f = cv2.GaussianBlur(f, (0, 0), BLUR)
    lo, hi = np.percentile(f, (1, 99))
    return np.clip((f - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 2. maska
def fill_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = mask.copy()
    if hierarchy is not None:
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] >= 0:          # ima roditelja -> rupa
                cv2.drawContours(out, [c], -1, 255, -1)
    return out


def segment(f):
    """Vraca masku svih cestica i Otsu prag."""
    t, _ = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (f > t + THR_OFFSET).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(OPEN_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(CLOSE_K))
    return fill_holes(mask), t


# ---------------------------------------------------------------- 3. jezgra
def find_cores(f, mask, t):
    """Po jedno jezgro za svaku cesticu (mapa oznaka)."""
    fc = cv2.GaussianBlur(f, (0, 0), CORE_BLUR)
    cores = (fc > t + HI_OFFSET).astype(np.uint8) * 255
    cores &= mask
    # black-hat = koliko je piksel tamniji od svoje okoline -> pukotine
    blackhat = cv2.morphologyEx(fc, cv2.MORPH_BLACKHAT, ellipse(CRACK_K))
    cores[blackhat > CRACK_T] = 0
    cores = cv2.morphologyEx(cores, cv2.MORPH_OPEN, ellipse(CORE_OPEN))

    n, markers, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=4)
    small = stats[:, cv2.CC_STAT_AREA] < CORE_MIN
    small[0] = True
    markers[small[markers]] = 0

    # dio maske koji je ostao bez jezgra dobija svoje (sitne cestice)
    n_comp, comps = cv2.connectedComponents(mask)
    has_core = np.zeros(n_comp, bool)
    has_core[comps[markers > 0]] = True
    missing = (~has_core[comps]) & (mask > 0)
    markers[missing] = comps[missing] + n
    return markers.astype(np.int32)


# ---------------------------------------------------------------- 4. watershed
def relabel(m):
    """Nakon sjecenja maskom ista oznaka moze ostati u vise komada - razdvoji ih."""
    edge = np.zeros(m.shape, bool)
    edge[:, 1:] |= (m[:, 1:] != m[:, :-1]) & (m[:, 1:] > 0) & (m[:, :-1] > 0)
    edge[1:, :] |= (m[1:, :] != m[:-1, :]) & (m[1:, :] > 0) & (m[:-1, :] > 0)
    solid = ((m > 0) & ~edge).astype(np.uint8)
    _, lab = cv2.connectedComponents(solid, connectivity=4)
    grown = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    fix = (m > 0) & (lab == 0)
    lab[fix] = grown[fix]
    return lab


def grow_cores(f, mask, markers):
    inv = 255 - f
    ws = cv2.watershed(cv2.merge([inv, inv, inv]), markers)
    ws[(mask == 0) | (ws < 0)] = 0
    return relabel(ws)


# ---------------------------------------------------------------- 5. podjela
def solidity(comp):
    """Povrsina / povrsina konveksnog omotaca (1 = potpuno konveksno)."""
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    return comp.sum() / max(cv2.contourArea(cv2.convexHull(c)), 1.0)


def flood(markers, dist, comp, r):
    """Siri markere nazad do pune regije, sloj po sloj od sredine ka ivici.

    Na svakom nivou markeri rastu samo kroz piksele koji su od ivice dalji
    od tog nivoa, pa se dva dijela sretnu bas na najuzem mjestu.
    """
    lab = markers.astype(np.float32)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    level = float(r)
    while True:
        allowed = (dist > level) & (comp > 0)
        while True:
            grown = cv2.dilate(lab, cross)
            new = (lab == 0) & allowed & (grown > 0)
            if not new.any():
                break
            lab[new] = grown[new]
        if level <= 0:
            break
        level = max(level - 1.0, 0.0)
    return lab.astype(np.int32)


def split_component(comp, depth=0):
    """Dijeli regiju (0/1) dok dijelovi ne postanu kompaktni.

    Distance transform kaze koliko je svaki piksel daleko od ivice. Regija
    se "ljusti" sve dublje; kad se raspadne na vise dovoljno velikih dijelova,
    oni postaju markeri i sire se nazad funkcijom flood().
    """
    if depth > MAX_DEPTH or comp.sum() < SPLIT_MIN_AREA or solidity(comp) >= SOLIDITY:
        return [comp]
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    r = 2.0
    while r < dist.max():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (dist > r).astype(np.uint8), connectivity=4)
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= PIECE_MIN]
        if len(big) >= 2:
            markers = np.zeros(comp.shape, np.int32)
            for j, i in enumerate(big, 1):
                markers[lab == i] = j
            parts_lab = flood(markers, dist, comp, r)
            parts = []
            for j in range(1, len(big) + 1):
                piece = ((parts_lab == j) & (comp > 0)).astype(np.uint8)
                if piece.any():
                    parts += split_component(piece, depth + 1)
            return parts
        r += max(1.0, r / 8)
    return [comp]


def pixels_by_label(labels):
    """Za svaku oznaku vraca (x, y) njenih piksela, bez prolaska kroz sliku za svaku posebno."""
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat)
    starts = np.concatenate([[0], np.cumsum(counts)])
    w = labels.shape[1]
    for lab in range(1, counts.size):
        if counts[lab]:
            idx = order[starts[lab]:starts[lab + 1]]
            yield idx % w, idx // w


def shape_split(labels):
    out = np.zeros_like(labels)
    h, w = labels.shape
    nxt = 1
    for xs, ys in pixels_by_label(labels):
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, h)
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, w)
        comp = np.zeros((y1 - y0, x1 - x0), np.uint8)
        comp[ys - y0, xs - x0] = 1
        for piece in split_component(comp):
            out[y0:y1, x0:x1][piece > 0] = nxt
            nxt += 1
    return out


# ---------------------------------------------------------------- 6. mjerenje
def measure(labels):
    rows = []
    for xs, ys in pixels_by_label(labels):
        area = xs.size
        if area < MIN_AREA:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        w += 1                       # minAreaRect mjeri izmedju centara piksela
        h += 1
        if w < h:                    # major je uvijek duza strana
            w, h = h, w
            angle += 90
        rows.append((cx, cy, area, w, h, angle % 180))
    return rows


def process(img):
    f = feature_map(img)
    mask, t = segment(f)
    markers = find_cores(f, mask, t)
    labels = grow_cores(f, mask, markers)
    labels = shape_split(labels)
    return measure(labels)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["centroid_x", "centroid_y", "area_px",
                         "major_axis_px", "minor_axis_px", "orientation_deg"])
        for cx, cy, area, w, h, angle in rows:
            writer.writerow([f"{cx:.2f}", f"{cy:.2f}", area,
                             f"{w:.2f}", f"{h:.2f}", f"{angle:.2f}"])


def main(in_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(in_dir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    for p in images:
        rows = process(cv2.imread(str(p)))
        write_csv(out / f"{p.stem}.csv", rows)
        print(f"{p.name}: {len(rows)} cestica")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Upotreba: python detect.py <folder_sa_slikama> <izlazni_folder>")
    main(sys.argv[1], sys.argv[2])
