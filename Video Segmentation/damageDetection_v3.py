"""
fracture_detect.py
------------------
Detect particle fractures, debonding, and matrix cracking in optical
light-microscopy tensile-test videos (.avi / .mp4) of particle-reinforced
composite plastics.

Detection strategy
------------------
  1. Segment particles from reference frame (Otsu threshold + morphology)
  2. Estimate per-frame bulk motion via phase correlation on particle mask
  3. Apply damage-specific filters to motion-compensated diff image,
     restricted to physically meaningful regions:
       - Fracture  : thin dark lines (1-5px wide) ON particle regions
       - Debonding : rapid intensity change AT particle boundaries
       - Matrix crack : elongated dark features that grow over time,
                        in matrix regions and along particle edges
  4. Confirm events by persistence (fracture/debonding) or growth (cracks)
  5. Track confirmed events by re-detecting within a search window each frame

Usage examples
--------------
  python fracture_detect.py video.avi
  python fracture_detect.py video.avi --half-res
  python fracture_detect.py video.avi --confirm-fracture 3 --confirm-debond 4
  python fracture_detect.py video.avi --fracture-darkness 15 --min-crack-length 40
  python fracture_detect.py video.avi --output results/ --verbose
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_COLORS = {
    "fracture":  (0,   80,  255),   # blue-red
    "debonding": (0,   165, 255),   # orange
    "crack":     (0,   0,   255),   # red
}

SEVERITY_AREA = (100, 800)          # (low->medium, medium->high) px^2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    det_id: int
    frame_index: int
    timestamp_s: float
    event_type: str             # fracture | debonding | crack
    bbox: tuple                 # (x, y, w, h) full-res
    centroid: tuple             # (cx, cy) full-res
    area_px: float
    length_px: float            # major axis length
    severity: str               # low | medium | high


@dataclass
class Candidate:
    """Unconfirmed detection accumulating hits before promotion."""
    bbox: tuple                 # full-res
    event_type: str
    first_frame: int
    hits: int = 1
    last_seen: int = 0
    prev_area: float = 0.0      # for crack growth tracking


@dataclass
class ActiveEvent:
    """Confirmed, currently tracked event."""
    det_id: int
    event_type: str
    bbox: tuple                 # full-res, updated each frame
    severity: str
    first_frame: int
    last_area: float = 0.0
    active: bool = True


@dataclass
class FrameStats:
    frame_index: int
    timestamp_s: float
    n_new: int = 0
    n_active: int = 0
    n_candidates: int = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def severity(area: float) -> str:
    if area < SEVERITY_AREA[0]:
        return "low"
    if area < SEVERITY_AREA[1]:
        return "medium"
    return "high"


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_bbox(bbox: tuple, W: int, H: int) -> Optional[tuple]:
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def contour_stats(contour: np.ndarray) -> tuple:
    """Return (bbox, centroid, area, length) for a contour."""
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    rect = cv2.minAreaRect(contour)
    length = max(rect[1]) if min(rect[1]) > 0 else max(w, h)
    return (x, y, w, h), (cx, cy), area, length


def get_proc_size(W: int, H: int, half_res: bool) -> tuple:
    return (W // 2, H // 2) if half_res else (W, H)


def expand_bbox(bbox: tuple, margin: int, W: int, H: int) -> Optional[tuple]:
    x, y, w, h = bbox
    return clamp_bbox((x - margin, y - margin,
                       w + 2 * margin, h + 2 * margin), W, H)


# ---------------------------------------------------------------------------
# Reference frame and particle segmentation
# ---------------------------------------------------------------------------

def extract_reference_frame(video_path: str, out_dir: Path) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("[ERROR] Cannot read first frame for reference")
    path = str(out_dir / "reference_frame.png")
    cv2.imwrite(path, frame)
    print(f"[INFO] Reference frame saved : {path}")
    return frame


def segment_particles(
    ref_frame: np.ndarray,
    blur_ksize: int = 5,
    morph_close: int = 10,
    morph_open: int = 5,
) -> np.ndarray:
    """
    Segment particles by local texture variance.
    Particles have smoother interiors than the granular matrix,
    so regions of LOW local variance correspond to particle interiors.
    """
    gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Local mean and variance using a moderate window
    ksize = 25
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq_mean = cv2.filter2D(gray ** 2, -1, kernel)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)

    # Normalize variance to 0-255
    var_norm = cv2.normalize(local_var, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    # LOW variance = smooth = particle interior
    # Invert so particles are bright
    var_inv = cv2.bitwise_not(var_norm)

    # Threshold — particles are the smoothest regions
    _, mask = cv2.threshold(var_inv, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup — close gaps within particles,
    # remove small matrix speckle
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open,  morph_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=2)

    return mask


def build_region_masks(particle_mask: np.ndarray) -> tuple:
    """
    Build three region masks from the particle mask:
      - particle_mask  : interior of particles (fracture zone)
      - boundary_mask  : dilated particle edges (debonding zone)
      - matrix_mask    : everything outside particles + boundary (crack zone)
    """
    # Boundary = dilated edge ring around particles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated   = cv2.dilate(particle_mask, kernel, iterations=2)
    eroded    = cv2.erode( particle_mask, kernel, iterations=2)
    boundary  = cv2.subtract(dilated, eroded)

    # Matrix = everything not particle and not boundary
    matrix = cv2.bitwise_not(dilated)

    return particle_mask, boundary, matrix


# ---------------------------------------------------------------------------
# Motion compensation via phase correlation
# ---------------------------------------------------------------------------

def estimate_shift_phase(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> tuple:
    """
    Estimate (dx, dy) translation between frames using phase correlation.
    More stable than sparse optical flow for uniform rigid-body motion.
    Returns (dx, dy) in pixels.
    """
    prev_f = np.float32(prev_gray)
    curr_f = np.float32(curr_gray)
    (dx, dy), _ = cv2.phaseCorrelate(prev_f, curr_f)
    return dx, dy


def translate_frame(
    frame: np.ndarray,
    dx: float,
    dy: float,
    size: tuple,
) -> np.ndarray:
    """Translate frame by (dx, dy) using an affine warp."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, size,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def accumulate_shift(
    total_dx: float,
    total_dy: float,
    dx: float,
    dy: float,
) -> tuple:
    """Accumulate frame-to-frame shifts for ref-frame compensation."""
    return total_dx + dx, total_dy + dy


# ---------------------------------------------------------------------------
# Damage-specific detection filters
# ---------------------------------------------------------------------------

def detect_fractures(
    diff: np.ndarray,
    particle_mask: np.ndarray,
    darkness_threshold: int,
    min_length: float,
    max_width: float = 5.0,
) -> list:
    """
    Detect thin dark lines on particle surfaces.
    Uses a bank of oriented matched filters tuned for 1-5px wide lines.
    Returns list of contours (in diff/proc coordinates).
    """
    # Invert diff so dark lines become bright — we want decreases in intensity
    # (new dark lines appearing on particles)
    inverted = cv2.bitwise_not(diff)

    # Apply particle mask — fractures only occur on particles
    masked = cv2.bitwise_and(inverted, inverted, mask=particle_mask)

    # Bank of oriented line filters (0, 30, 60, 90, 120, 150 degrees)
    response = np.zeros_like(masked, dtype=np.float32)
    for angle in range(0, 180, 30):
        rad = np.deg2rad(angle)
        # Oriented kernel: thin line detector at this angle
        ksize = 15
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=1.0,
            theta=rad,
            lambd=4.0,
            gamma=0.3,
            psi=0,
            ktype=cv2.CV_32F,
        )
        r = cv2.filter2D(masked.astype(np.float32), cv2.CV_32F, kernel)
        response = np.maximum(response, r)

    # Threshold response
    resp_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(resp_norm, darkness_threshold, 255, cv2.THRESH_BINARY)

    # Morphological thinning / cleanup
    k_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_thin, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        _, _, area, length = contour_stats(c)
        rect = cv2.minAreaRect(c)
        dims = sorted(rect[1])
        width = dims[0] if dims[0] > 0 else 1
        if length >= min_length and width <= max_width:
            results.append(c)
    return results


def detect_debonding(
    diff: np.ndarray,
    boundary_mask: np.ndarray,
    threshold: int,
    min_area: int,
) -> list:
    """
    Detect rapid intensity changes at particle boundaries.
    Returns list of contours.
    """
    masked = cv2.bitwise_and(diff, diff, mask=boundary_mask)
    _, binary = cv2.threshold(masked, threshold, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def detect_cracks(
    diff: np.ndarray,
    matrix_mask: np.ndarray,
    particle_mask: np.ndarray,
    threshold: int,
    min_length: float,
    min_aspect_ratio: float = 2.5,
) -> list:
    """
    Detect elongated dark features in matrix and along particle edges.
    Matrix cracks are elongated (high aspect ratio) and dark.
    Returns list of contours.
    """
    # Cracks can appear in matrix AND along particle edges
    crack_zone = cv2.bitwise_or(matrix_mask, particle_mask)
    masked = cv2.bitwise_and(diff, diff, mask=crack_zone)

    # Enhance elongated dark features
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,  21))
    bth = cv2.add(
        cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, k_h),
        cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, k_v),
    )
    enhanced = cv2.addWeighted(
        cv2.normalize(bth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        0.6, masked, 0.4, 0,
    )

    _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        _, _, area, length = contour_stats(c)
        rect = cv2.minAreaRect(c)
        dims = rect[1]
        if min(dims) == 0:
            continue
        ar = max(dims) / min(dims)
        if length >= min_length and ar >= min_aspect_ratio:
            results.append(c)
    return results


# ---------------------------------------------------------------------------
# Candidate buffer and confirmation
# ---------------------------------------------------------------------------

def match_to_candidates(
    candidates: list,
    new_bbox: tuple,
    frame_index: int,
    iou_thr: float = 0.2,
) -> int:
    """Return index of best matching candidate, or -1 if none."""
    best_iou = iou_thr
    best_idx = -1
    for i, cand in enumerate(candidates):
        overlap = iou(cand.bbox, new_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_idx = i
    return best_idx


def update_candidate_buffer(
    candidates: list,
    new_contours: list,
    event_type: str,
    frame_index: int,
    confirm_frames: int,
    confirm_window: int,
    scale: float,
    full_W: int,
    full_H: int,
) -> tuple:
    """
    Match new contours to existing candidates.
    Confirmed candidates (hits >= confirm_frames within confirm_window)
    are returned as a list of (bbox_full, event_type, area, length).
    """
    # Expire old candidates
    candidates = [c for c in candidates
                  if (frame_index - c.last_seen) <= confirm_window]

    confirmed = []
    for c in new_contours:
        bbox_proc, _, area, length = contour_stats(c)
        # Scale to full resolution
        bbox_full = clamp_bbox(
            tuple(int(v / scale) for v in bbox_proc),
            full_W, full_H,
        )
        if bbox_full is None:
            continue

        idx = match_to_candidates(candidates, bbox_full, frame_index)
        if idx >= 0:
            cand = candidates[idx]
            cand.hits += 1
            cand.last_seen = frame_index
            # Blend bbox
            cand.bbox = tuple(
                int(0.6 * a + 0.4 * b)
                for a, b in zip(cand.bbox, bbox_full)
            )
            cand.prev_area = area / (scale * scale)
            if cand.hits >= confirm_frames:
                confirmed.append((cand.bbox, event_type,
                                  area / (scale * scale), length / scale))
                candidates.pop(idx)
        else:
            candidates.append(Candidate(
                bbox=bbox_full,
                event_type=event_type,
                first_frame=frame_index,
                hits=1,
                last_seen=frame_index,
                prev_area=area / (scale * scale),
            ))

    return confirmed, candidates


# ---------------------------------------------------------------------------
# Active event tracking (lightweight centroid / bbox re-detection)
# ---------------------------------------------------------------------------

def update_active_events(
    events: list,
    curr_gray_full: np.ndarray,
    search_margin: int,
    full_W: int,
    full_H: int,
) -> None:
    """
    For each active event, re-detect the feature within a search window
    around the last known bbox by finding the strongest response in that
    region. Updates event.bbox in place.
    """
    for ev in events:
        if not ev.active:
            continue
        search = expand_bbox(ev.bbox, search_margin, full_W, full_H)
        if search is None:
            ev.active = False
            continue
        sx, sy, sw, sh = search
        roi = curr_gray_full[sy:sy+sh, sx:sx+sw]
        if roi.size == 0:
            continue
        # Use local Laplacian response to find strongest edge feature
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        lap_abs = np.abs(lap).astype(np.uint8)
        _, local_mask = cv2.threshold(lap_abs, 10, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        best = max(cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(best)
        new_bbox = clamp_bbox(
            (sx + bx, sy + by, bw, bh), full_W, full_H)
        if new_bbox is not None:
            ev.bbox = new_bbox


def overlaps_active(bbox: tuple, events: list, iou_thr: float = 0.2) -> bool:
    return any(ev.active and iou(bbox, ev.bbox) > iou_thr for ev in events)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def open_video_writer(path: Path, fps: float, W: int, H: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open video writer: {path}")
    return writer


def open_csv(path: Path) -> tuple:
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "det_id", "frame_index", "timestamp_s", "event_type",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "centroid_x", "centroid_y", "area_px", "length_px", "severity",
    ])
    return f, w


def write_csv_row(writer, det: Detection) -> None:
    x, y, w, h = det.bbox
    cx, cy = det.centroid
    writer.writerow([
        det.det_id, det.frame_index, f"{det.timestamp_s:.4f}",
        det.event_type, x, y, w, h, cx, cy,
        f"{det.area_px:.1f}", f"{det.length_px:.1f}", det.severity,
    ])


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    new_dets: list,
    active_events: list,
    frame_index: int,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    new_ids = {d.det_id for d in new_dets}

    # Persistent active events (thin border)
    for ev in active_events:
        if not ev.active or ev.det_id in new_ids:
            continue
        color = EVENT_COLORS.get(ev.event_type, (255, 255, 255))
        x, y, w, h = [int(v) for v in ev.bbox]
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
        cv2.putText(out, f"#{ev.det_id} {ev.event_type}",
                    (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # New confirmations (thicker border + NEW label)
    for det in new_dets:
        color = EVENT_COLORS.get(det.event_type, (255, 255, 255))
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        lbl = f"NEW #{det.det_id} {det.event_type} [{det.severity}]"
        cv2.putText(out, lbl, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cx, cy = det.centroid
        cv2.circle(out, (cx, cy), 3, color, -1)

    # HUD
    ts   = frame_index / fps if fps > 0 else 0
    n_act = sum(1 for ev in active_events if ev.active)
    hud  = (f"Frame {frame_index:05d}  |  {ts:.2f}s  |  "
            f"new: {len(new_dets)}  active: {n_act}")
    cv2.putText(out, hud, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def draw_legend(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    H = out.shape[0]
    for i, (label, color) in enumerate(EVENT_COLORS.items()):
        y = H - 10 - i * 18
        cv2.rectangle(out, (6, y-10), (18, y+2), color, -1)
        cv2.putText(out, label, (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    all_dets: list,
    frame_stats: list,
    video_path: str,
    elapsed: float,
    fps: float,
    n_frames: int,
) -> None:
    print("\n" + "=" * 60)
    print("  DAMAGE DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Video        : {video_path}")
    print(f"  Frames       : {n_frames}  ({n_frames/fps:.2f}s @ {fps:.2f} fps)")
    print(f"  Process time : {elapsed:.1f}s")
    print(f"  Total events : {len(all_dets)}")
    print()
    if not all_dets:
        print("  No damage events detected.")
        print("=" * 60)
        return
    from collections import Counter
    by_type = Counter(d.event_type for d in all_dets)
    by_sev  = Counter(d.severity   for d in all_dets)
    print("  By type:")
    for t, n in sorted(by_type.items()):
        print(f"    {t:<14s} {n:>5d}")
    print()
    print("  By severity:")
    for s in ("high", "medium", "low"):
        print(f"    {s:<14s} {by_sev[s]:>5d}")
    if frame_stats:
        peak = max(frame_stats, key=lambda fs: fs.n_new)
        print()
        print(f"  Peak frame   : {peak.frame_index} "
              f"({peak.timestamp_s:.2f}s) — {peak.n_new} new events")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_video(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {args.video}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem         = Path(args.video).stem
    pw, ph       = get_proc_size(full_W, full_H, args.half_res)
    scale        = pw / full_W
    cap.release()

    print(f"[INFO] Video          : {args.video}")
    print(f"[INFO] Resolution     : {full_W}x{full_H}  "
          f"|  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"[INFO] Proc res       : {pw}x{ph}"
          f"{'  (half-res)' if args.half_res else ''}")
    print(f"[INFO] Confirm        : fracture={args.confirm_fracture}f  "
          f"debond={args.confirm_debond}f  crack={args.confirm_crack}f  "
          f"(window={args.confirm_window}f)")
    print(f"[INFO] Thresholds     : fracture darkness={args.fracture_darkness}  "
          f"debond={args.debond_threshold}  crack={args.crack_threshold}")
    print(f"[INFO] Min lengths    : fracture={args.min_fracture_length}px  "
          f"crack={args.min_crack_length}px  (proc-res)")
    print(f"[INFO] Output dir     : {out_dir}")

    # --- Reference frame and particle segmentation ---
    cap = cv2.VideoCapture(args.video)
    ret, ref_frame_full = cap.read()
    cap.release()
    if not ret:
        sys.exit("[ERROR] Cannot read reference frame")

    ref_path = str(out_dir / "reference_frame.png")
    cv2.imwrite(ref_path, ref_frame_full)
    print(f"[INFO] Reference frame: {ref_path}")

    ref_proc       = cv2.resize(ref_frame_full, (pw, ph))
    ref_gray_proc  = cv2.cvtColor(ref_proc, cv2.COLOR_BGR2GRAY)
    ref_gray_full  = cv2.cvtColor(ref_frame_full, cv2.COLOR_BGR2GRAY)

    # Segment particles at proc resolution
    particle_mask_proc = segment_particles(ref_proc)
    part_mask_p, boundary_mask_p, matrix_mask_p = build_region_masks(
        particle_mask_proc)

    # Save mask for inspection
    cv2.imwrite(str(out_dir / "particle_mask.png"), particle_mask_proc)
    print(f"[INFO] Particle mask  : {out_dir / 'particle_mask.png'}")

    # --- Output writers ---
    video_out = open_video_writer(
        out_dir / f"{stem}_annotated.mp4", fps, full_W, full_H)
    csv_path  = out_dir / f"{stem}_detections.csv"
    csv_file, csv_writer = open_csv(csv_path)

    # --- State ---
    all_dets:        list = []
    active_events:   list = []
    cands_fracture:  list = []
    cands_debond:    list = []
    cands_crack:     list = []
    frame_stats_list: list = []
    prev_gray_proc: Optional[np.ndarray] = None
    total_dx = 0.0
    total_dy = 0.0
    det_id   = 0
    frame_idx = 0
    t_start   = time.time()

    print("\n[INFO] Processing ...\n")
    pbar = tqdm(total=total_frames, unit="frame", dynamic_ncols=True)

    cap = cv2.VideoCapture(args.video)

    while True:
        ret, frame_full = cap.read()
        if not ret:
            break

        timestamp_s   = frame_idx / fps
        proc_frame    = cv2.resize(frame_full, (pw, ph))
        curr_gray_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2GRAY)
        new_dets: list = []

        fs = FrameStats(frame_index=frame_idx, timestamp_s=timestamp_s)

        if prev_gray_proc is not None:

            # ----------------------------------------------------------------
            # Step 1 — Estimate frame-to-frame shift (phase correlation)
            # ----------------------------------------------------------------
            dx, dy = estimate_shift_phase(prev_gray_proc, curr_gray_proc)
            total_dx, total_dy = accumulate_shift(total_dx, total_dy, dx, dy)

            # ----------------------------------------------------------------
            # Step 2 — Compensated frame-to-frame diff
            # ----------------------------------------------------------------
            warped = translate_frame(curr_gray_proc, dx, dy, (pw, ph))
            diff_fd = cv2.absdiff(prev_gray_proc, warped)

            # ----------------------------------------------------------------
            # Step 3 — Compensated reference diff
            # ----------------------------------------------------------------
            ref_warped = translate_frame(
                ref_gray_proc, -total_dx, -total_dy, (pw, ph))
            diff_rd = cv2.absdiff(ref_warped, curr_gray_proc)

            # Use whichever diff channel is stronger per pixel
            diff = cv2.max(diff_fd, diff_rd)

            # ----------------------------------------------------------------
            # Step 4 — Per-type detection
            # ----------------------------------------------------------------
            frac_contours = detect_fractures(
                diff, part_mask_p,
                darkness_threshold=args.fracture_darkness,
                min_length=args.min_fracture_length,
            )

            deb_contours = detect_debonding(
                diff, boundary_mask_p,
                threshold=args.debond_threshold,
                min_area=args.min_debond_area,
            )

            crack_contours = detect_cracks(
                diff, matrix_mask_p, part_mask_p,
                threshold=args.crack_threshold,
                min_length=args.min_crack_length,
            )

            fs.n_candidates = (len(frac_contours) +
                               len(deb_contours) +
                               len(crack_contours))

            # ----------------------------------------------------------------
            # Step 5 — Temporal confirmation per type
            # ----------------------------------------------------------------
            def confirm_and_promote(contours, etype, cands, n_confirm):
                nonlocal det_id
                confirmed, updated = update_candidate_buffer(
                    cands, contours, etype, frame_idx,
                    n_confirm, args.confirm_window,
                    scale, full_W, full_H,
                )
                promoted = []
                for bbox, et, area, length in confirmed:
                    if overlaps_active(bbox, active_events):
                        continue
                    safe = clamp_bbox(bbox, full_W, full_H)
                    if safe is None:
                        continue
                    x, y, w, h = safe
                    cx, cy = x + w // 2, y + h // 2
                    det = Detection(
                        det_id=det_id,
                        frame_index=frame_idx,
                        timestamp_s=timestamp_s,
                        event_type=et,
                        bbox=safe,
                        centroid=(cx, cy),
                        area_px=area,
                        length_px=length,
                        severity=severity(area),
                    )
                    det_id += 1
                    promoted.append(det)
                    all_dets.append(det)
                    write_csv_row(csv_writer, det)
                    active_events.append(ActiveEvent(
                        det_id=det.det_id,
                        event_type=et,
                        bbox=safe,
                        severity=det.severity,
                        first_frame=frame_idx,
                        last_area=area,
                    ))
                return promoted, updated

            new_frac,  cands_fracture = confirm_and_promote(
                frac_contours,  "fracture",  cands_fracture,
                args.confirm_fracture)
            new_deb,   cands_debond  = confirm_and_promote(
                deb_contours,   "debonding", cands_debond,
                args.confirm_debond)
            new_crack, cands_crack   = confirm_and_promote(
                crack_contours, "crack",     cands_crack,
                args.confirm_crack)

            new_dets = new_frac + new_deb + new_crack
            fs.n_new = len(new_dets)

            # ----------------------------------------------------------------
            # Step 6 — Update active event positions
            # ----------------------------------------------------------------
            update_active_events(
                active_events, curr_gray_full,
                search_margin=args.search_margin,
                full_W=full_W, full_H=full_H,
            )

        fs.n_active = sum(1 for ev in active_events if ev.active)
        frame_stats_list.append(fs)

        # --- Console output ---
        if args.verbose or new_dets:
            print(f"  Frame {frame_idx:05d} | {timestamp_s:7.2f}s | "
                  f"new: {fs.n_new}  "
                  f"active: {fs.n_active}  "
                  f"cands: {fs.n_candidates}")
            if args.verbose and new_dets:
                for det in new_dets:
                    print(f"    #{det.det_id} {det.event_type:<12s} "
                          f"bbox={det.bbox}  "
                          f"len={det.length_px:.1f}px  "
                          f"sev={det.severity}")

        # --- Annotate and write ---
        annotated = annotate_frame(
            frame_full, new_dets, active_events, frame_idx, fps)
        video_out.write(draw_legend(annotated))

        prev_gray_proc = curr_gray_proc
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    video_out.release()
    csv_file.close()

    elapsed = time.time() - t_start
    print(f"\n[INFO] Annotated video : {out_dir / f'{stem}_annotated.mp4'}")
    print(f"[INFO] CSV log         : {csv_path}")
    print(f"[INFO] Particle mask   : {out_dir / 'particle_mask.png'}")
    print_summary(all_dets, frame_stats_list, args.video, elapsed, fps, frame_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Detect particle fractures, debonding, and matrix cracking "
            "in optical-microscopy tensile-test videos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("video", help="Path to input video (.avi or .mp4)")

    # Output
    p.add_argument("--output", default="output",
                   help="Output directory (default: ./output/)")
    p.add_argument("--half-res", action="store_true",
                   help="Process at half resolution for ~4x speedup "
                        "(output video is still full resolution)")

    # Confirmation windows — primary sensitivity controls
    p.add_argument("--confirm-fracture", type=int, default=2,
                   metavar="N",
                   help="Frames a fracture candidate must persist to be "
                        "confirmed (default: 2)")
    p.add_argument("--confirm-debond", type=int, default=3,
                   metavar="N",
                   help="Frames a debonding candidate must persist to be "
                        "confirmed (default: 3)")
    p.add_argument("--confirm-crack", type=int, default=2,
                   metavar="N",
                   help="Frames a crack candidate must persist to be "
                        "confirmed (default: 2)")
    p.add_argument("--confirm-window", type=int, default=8,
                   metavar="N",
                   help="Frame window over which confirmation hits must "
                        "occur (default: 8)")

    # Detection thresholds
    p.add_argument("--fracture-darkness", type=int, default=20,
                   metavar="N",
                   help="Minimum Gabor filter response to detect a fracture "
                        "line; lower = more sensitive (default: 20)")
    p.add_argument("--debond-threshold", type=int, default=25,
                   metavar="N",
                   help="Pixel intensity change threshold for debonding "
                        "detection (default: 25)")
    p.add_argument("--crack-threshold", type=int, default=20,
                   metavar="N",
                   help="Pixel intensity change threshold for crack "
                        "detection (default: 20)")

    # Size filters
    p.add_argument("--min-fracture-length", type=float, default=5.0,
                   metavar="PX",
                   help="Minimum fracture line length in proc-res pixels "
                        "(default: 5.0)")
    p.add_argument("--min-crack-length", type=float, default=20.0,
                   metavar="PX",
                   help="Minimum crack length in proc-res pixels "
                        "(default: 20.0)")
    p.add_argument("--min-debond-area", type=int, default=5,
                   metavar="PX",
                   help="Minimum debonding contour area in proc-res pixels "
                        "(default: 5)")

    # Tracking
    p.add_argument("--search-margin", type=int, default=20,
                   metavar="PX",
                   help="Full-res pixel margin around each active event "
                        "bbox to search for the feature each frame "
                        "(default: 20)")

    # Misc
    p.add_argument("--verbose", action="store_true",
                   help="Print per-detection detail each frame")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not os.path.isfile(args.video):
        parser.error(f"Video file not found: {args.video}")
    if Path(args.video).suffix.lower() not in (".avi", ".mp4"):
        parser.error("Unsupported file type. Use .avi or .mp4")

    process_video(args)


if __name__ == "__main__":
    main()