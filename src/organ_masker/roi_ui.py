"""
roi_ui.py
---------
Contour-only interactive ROI tools with ENTER-to-confirm and retry.
Brush size control for freehand drawing. On confirm, freehand masks are
binary-fill-holed for foreground only (no hollow interior for FG).
Background mask is left exactly as drawn.

Also contains ROI reuse helpers:
- load_reused_prompts(): reuse previously exported ROIs (JSON + optional *_pos/neg.png)
  across different downsample + HOA levels, resized to current seed frame size.
"""

import cv2
import numpy as np
from pathlib import Path
from screeninfo import get_monitors
import json
from scipy.ndimage import binary_fill_holes  # used only on confirm to solidify FG loops

GREEN = (0, 255, 0)
RED   = (0, 0, 255)
THK   = 3  # contour thickness


def fit_to_screen(img_bgr, margin=0.7):
    mon = get_monitors()[0]
    H, W = img_bgr.shape[:2]
    scale = min((mon.width * margin) / W, (mon.height * margin) / H, 1.0)
    if scale < 1.0:
        img = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    else:
        img = img_bgr
    return img, scale


def select_rects_overlay(img_bgr, multi=False):
    name = "Rectangles: ENTER=confirm, r=reset, ESC=cancel"
    rects, cur, drawing = [], None, False
    start = None

    def redraw():
        out = img_bgr.copy()
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, THK)
        if cur is not None:
            x1, y1, x2, y2 = cur
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, THK)
        return out

    def cb(event, x, y, flags, param):
        nonlocal drawing, cur, rects, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; start = (x, y); cur = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cur = (start[0], start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1, y1, x2, y2 = cur
            x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                rects.append((x1, y1, x2, y2))
            cur = None

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, cb)
    while True:
        cv2.imshow(name, redraw())
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            rects = []
            break
        if k == ord('r'):
            rects = []
            cur = None
        if k == 13:
            if rects or not multi:
                break
    cv2.destroyWindow(name)
    return rects


def select_points_overlay(img_bgr):
    name = "Points: L=FG(green), R=BG(red), ENTER=confirm, r=reset, ESC=cancel"
    pts, labs = [], []

    def cb(event, x, y, flags, param):
        nonlocal pts, labs
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); labs.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts.append((x, y)); labs.append(0)

    def redraw():
        out = img_bgr.copy()
        for (x, y), l in zip(pts, labs):
            cv2.circle(out, (x, y), 4, GREEN if l == 1 else RED, -1)
        return out

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, cb)
    while True:
        cv2.imshow(name, redraw())
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            pts = []
            labs = []
            break
        if k == ord('r'):
            pts = []
            labs = []
        if k == 13 and len(pts) > 0:
            break
    cv2.destroyWindow(name)
    return pts, labs


def select_circles_overlay(img_bgr):
    name = "Circles: drag radius; b=toggle FG/BG, ENTER=confirm, r=reset, ESC=cancel"
    circles = []  # (x, y, r, label)
    mode = 1
    start = None
    temp = None

    def cb(event, x, y, flags, param):
        nonlocal start, temp, circles
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y); temp = None
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            r = int(np.hypot(x - start[0], y - start[1])); temp = (start[0], start[1], r)
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            r = int(np.hypot(x - start[0], y - start[1])); circles.append((start[0], start[1], r, mode)); start = None; temp = None

    def redraw():
        out = img_bgr.copy()
        for (x, y, r, l) in circles:
            cv2.circle(out, (x, y), r, GREEN if l == 1 else RED, THK)
        if temp is not None:
            x, y, r = temp
            cv2.circle(out, (x, y), r, GREEN if mode == 1 else RED, THK)
        return out

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, cb)
    while True:
        cv2.imshow(name, redraw())
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            circles = []
            break
        if k == ord('r'):
            circles = []
        if k == ord('b'):
            mode = 1 - mode
        if k == 13 and circles:
            break
    cv2.destroyWindow(name)
    return circles


def draw_mask_overlay(img_bgr):
    """
    Brush UI without cv2.displayOverlay (Qt-free).
    LMB = FG (green), RMB = BG (red)
    ENTER = confirm, c = clear, [ / ] or - / + = resize brush

    On ENTER: fills holes inside connected FG regions so closed loops
    become solid (no hollow interior). BG is NOT hole-filled.
    """
    name = "Brush: L=FG(green), R=BG(red), ENTER=confirm, c=clear, [ ] / - +=size"
    H, W = img_bgr.shape[:2]
    mpos = np.zeros((H, W), np.uint8)
    mneg = np.zeros((H, W), np.uint8)
    drawing = False
    cur_fg = True
    radius = 8

    def cb(event, x, y, flags, param):
        nonlocal drawing, cur_fg
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; cur_fg = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True; cur_fg = False
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if cur_fg:
                cv2.circle(mpos, (x, y), radius, 1, -1)
            else:
                cv2.circle(mneg, (x, y), radius, 1, -1)

    def redraw():
        out = img_bgr.copy()
        # draw FG contours
        cnts, _ = cv2.findContours((mpos > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, GREEN, THK)
        # draw BG contours
        cnts, _ = cv2.findContours((mneg > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, RED, THK)
        # HUD
        hud1 = f"Brush radius: {radius}"
        hud2 = "L=FG  R=BG   ENTER=confirm (fills FG loops)   c=clear   [ ] / - +=size"
        cv2.putText(out, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.putText(out, hud2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        return out

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, cb)

    confirmed = False
    while True:
        cv2.imshow(name, redraw())
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            mpos[:] = 0; mneg[:] = 0
            break
        if k == ord('c'):
            mpos[:] = 0; mneg[:] = 0
        if k in (ord('['), ord('-')):
            radius = max(1, radius - 1)
        if k in (ord(']'), ord('+')):
            radius = min(256, radius + 1)
        if k == 13 and (mpos.any() or mneg.any()):
            confirmed = True
            break

    cv2.destroyWindow(name)

    if confirmed and mpos.any():
        mpos = binary_fill_holes(mpos.astype(bool)).astype(np.uint8)

    return mpos, mneg, {"note": "interactive brush (FG filled on confirm)"}


# ===== ROI reuse helpers below =====

def _effective_divisor(ds_local: int, hoa_level: int) -> int:
    ds_local = ds_local if ds_local else 1
    return (2 ** int(hoa_level)) * int(ds_local)


def _scale_boxes(boxes, scale_xy):
    if not boxes:
        return []
    sx, sy = scale_xy
    out = []
    for (x1, y1, x2, y2) in boxes:
        out.append([
            int(round(x1 * sx)), int(round(y1 * sy)),
            int(round(x2 * sx)), int(round(y2 * sy))
        ])
    return out


def _scale_points(points, scale_xy):
    if points is None:
        return None
    sx, sy = scale_xy
    pts = []
    for (x, y) in points:
        pts.append((int(round(x * sx)), int(round(y * sy))))
    return np.array(pts, dtype=np.int32)


def _load_mask_if_exists(folder: Path, plane: str, sli: int, target_hw):
    Ht, Wt = target_hw
    pos_path = folder / f"roi_{plane}_{sli:06d}_pos.png"
    neg_path = folder / f"roi_{plane}_{sli:06d}_neg.png"
    pos = neg = None
    if pos_path.exists():
        m = cv2.imread(str(pos_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            pos = (cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    if neg_path.exists():
        m = cv2.imread(str(neg_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            neg = (cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return pos, neg


def load_reused_prompts(
    roi_folder: str | Path,
    target_plane: str,
    target_downsample: int,
    target_hoa_level: int,
    target_hw: tuple[int, int],
):
    roi_dir = Path(roi_folder)
    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI folder not found: {roi_dir}")

    Ht, Wt = target_hw
    curr_div = _effective_divisor(target_downsample if target_downsample else 1, target_hoa_level)

    out = []
    for jf in sorted(roi_dir.glob("roi_*.json")):
        if jf.name.endswith("_settings.json"):
            continue
        meta = json.loads(jf.read_text())
        plane = meta.get("plane")
        if plane != target_plane:
            continue

        sli = int(meta["slice"])
        prev_ds = int(meta.get("downsample", 1))
        prev_hoa = int(meta.get("hoa_downsample_level", 0))
        prev_div = _effective_divisor(prev_ds, prev_hoa)

        sx = float(prev_div) / float(curr_div)
        sy = float(prev_div) / float(curr_div)

        boxes = meta.get("boxes")
        neg_boxes = meta.get("neg_boxes", [])
        points = meta.get("points")
        point_labels = meta.get("point_labels")

        boxes_scaled = _scale_boxes(boxes, (sx, sy)) if boxes else []
        neg_boxes_scaled = _scale_boxes(neg_boxes, (sx, sy)) if neg_boxes else []
        points_scaled = _scale_points(points, (sx, sy)) if points is not None else None
        labels_np = np.array(point_labels, dtype=np.int32) if point_labels is not None else None

        pos_mask, neg_mask = _load_mask_if_exists(roi_dir, plane, sli, (Ht, Wt))

        out.append({
            "slice_idx": sli,
            "boxes": boxes_scaled,
            "neg_boxes": neg_boxes_scaled,
            "points": points_scaled,
            "point_labels": labels_np,
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
        })

    return out
