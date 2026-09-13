"""
Cellpose + SAM Hybrid Particle Detection Script (with fracture detection)
=========================================================================
Replaces SamAutomaticMaskGenerator with Cellpose for per-frame particle
detection. SAM's SamPredictor is retained only for fracture splitting.

Cellpose is well-suited for low-contrast brightfield microscopy and handles
variable particle sizes via automatic diameter estimation.

Usage
-----
    python detect_particles_cellpose_fracture.py <video> --sam-checkpoint sam_vit_l_0b3195.pth --model-type vit_l --cellpose-model cyto3 --diameter 0 --redetect-interval 10 --output ./results/

    --diameter 0  →  auto-estimate per detection frame (recommended when
                     particle size is unknown or variable)

Requirements
------------
    pip install cellpose segment-anything opencv-python numpy tqdm torch
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Tunable parameters ───────────────────────────────────────────────────────

MIN_AREA_PX2             = 400
MAX_AREA_PX2             = 250_000
MIN_SOLIDITY             = 0.25

# Cellpose flow/mask thresholds.
# flow_threshold  — higher = stricter shape reconstruction (0.1–0.9)
# cellprob_threshold — lower = detect fainter/lower-contrast particles (-6 to 6)
#   If you're getting too much noise, raise it (e.g. 1 or 2).
CELLPOSE_FLOW_THRESH     = 0.7
CELLPOSE_PROB_THRESH     = 1.0

# Cellpose channel config for brightfield:
#   [0, 0] = grayscale (recommended for brightfield with no fluorescence)
#   [2, 1] = if you have a cytoplasm + nucleus channel
CELLPOSE_CHANNELS        = [0, 0]

FRACTURE_SOLIDITY_DROP   = 0.15
FRACTURE_MIN_HALF_AREA   = 200
FRACTURE_MIN_SPLIT_RATIO = 0.30

EDGE_MARGIN              = 2

CONTOUR_COLOUR           = (0, 220,   0)
FRACTURE_COLOUR          = (0, 165, 255)
CENTROID_COLOUR          = (0,   0, 255)
CENTROID_RADIUS          = 5
OUTPUT_CODEC             = "mp4v"

# ──────────────────────────────────────────────────────────────────────────────


def load_cellpose(model_name: str, device: str):
    try:
        from cellpose import models
    except ImportError:
        print("ERROR: cellpose not installed.")
        print("       Run:  pip install cellpose")
        sys.exit(1)

    use_gpu = (device == "cuda")
    print(f"Loading Cellpose model '{model_name}' (gpu={use_gpu}) …")
    # Cellpose 4.x uses CellposeModel instead of Cellpose
    model = models.CellposeModel(model_type=model_name, gpu=use_gpu)
    return model


def load_sam_predictor(checkpoint: str, model_type: str, device: str):
    """Load only SAM's point predictor — used solely for fracture splitting."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("ERROR: segment-anything not installed.")
        sys.exit(1)
    print(f"Loading SAM predictor ({model_type}) from {checkpoint} …")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def shape_stats(contour):
    area        = cv2.contourArea(contour)
    perimeter   = cv2.arcLength(contour, True)
    hull_area   = cv2.contourArea(cv2.convexHull(contour))
    solidity    = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
    equiv_diam  = np.sqrt(4 * area / np.pi)
    return area, solidity, circularity, equiv_diam


def mask_to_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else None


def make_particle(mask: np.ndarray, fractured: bool = False):
    contour = mask_to_contour(mask)
    if contour is None:
        return None
    M  = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
    _, solidity, _, _ = shape_stats(contour)
    return {
        "mask"             : mask.astype(bool),
        "contour"          : contour,
        "centroid"         : (cx, cy),
        "baseline_solidity": solidity,
        "fractured"        : fractured,
    }


def mask_touches_edge(mask, h, w, margin=EDGE_MARGIN):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return False
    return (rows.min() < margin or rows.max() >= h - margin or
            cols.min() < margin or cols.max() >= w - margin)


# ─── Cellpose detection ───────────────────────────────────────────────────────

def run_cellpose_on_frame(frame_bgr: np.ndarray, model,
                          diameter: float, frame_idx: int = 0) -> list:
    t0 = time.time()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    gray = clahe.apply(gray)

    # Cellpose 4.x: eval() returns (masks, flows, styles) — no diams
    # Pass diameter=0 to trigger auto-estimation
    masks_cp, flows, styles = model.eval(
        gray,
        diameter          = diameter if diameter > 0 else None,
        channels          = CELLPOSE_CHANNELS,
        flow_threshold    = CELLPOSE_FLOW_THRESH,
        cellprob_threshold= CELLPOSE_PROB_THRESH,
        do_3D             = False,
    )

    n_raw     = masks_cp.max()
    particles = []

    for label_id in range(1, n_raw + 1):
        binary_mask = (masks_cp == label_id)
        area = binary_mask.sum()

        if not (MIN_AREA_PX2 <= area <= MAX_AREA_PX2):
            continue

        contour = mask_to_contour(binary_mask)
        if contour is None:
            continue
        _, solidity, _, _ = shape_stats(contour)
        if solidity < MIN_SOLIDITY:
            continue

        p = make_particle(binary_mask)
        if p is not None:
            particles.append(p)

    elapsed = time.time() - t0
    print(f"  [frame {frame_idx:>5}] Cellpose: {n_raw} segments → "
          f"{len(particles)} particles  ({elapsed:.1f}s)")
    return particles


# ─── Fracture detection (unchanged from original) ────────────────────────────

def get_connected_components(mask: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= FRACTURE_MIN_HALF_AREA:
            components.append((area, labels == i))
    components.sort(key=lambda x: -x[0])
    return [m for _, m in components]


def check_fracture_by_components(mask, h, w):
    components = get_connected_components(mask)
    if len(components) < 2:
        return False
    rows, cols = np.where(mask)
    if not len(rows):
        return False
    bbox_h = rows.max() - rows.min() + 1
    bbox_w = cols.max() - cols.min() + 1
    r1, c1 = np.where(components[0])
    r2, c2 = np.where(components[1])
    dist = np.sqrt((r1.mean() - r2.mean())**2 + (c1.mean() - c2.mean())**2)
    diag = np.sqrt(bbox_h**2 + bbox_w**2)
    return (dist / diag if diag > 0 else 0) >= FRACTURE_MIN_SPLIT_RATIO


def check_fracture_by_solidity(mask, baseline_solidity):
    contour = mask_to_contour(mask)
    if contour is None:
        return False
    _, solidity, _, _ = shape_stats(contour)
    return (baseline_solidity - solidity) >= FRACTURE_SOLIDITY_DROP


def try_split_with_sam(frame_rgb, particle, predictor):
    mask       = particle["mask"]
    components = get_connected_components(mask)

    if len(components) < 2:
        rows, cols = np.where(mask)
        if not len(rows):
            return []
        cy, cx  = int(rows.mean()), int(cols.mean())
        bbox_h  = rows.max() - rows.min()
        bbox_w  = cols.max() - cols.min()
        offset  = max(bbox_h, bbox_w) // 4
        prompt_points = (
            [(cx, cy - offset), (cx, cy + offset)] if bbox_h >= bbox_w
            else [(cx - offset, cy), (cx + offset, cy)]
        )
    else:
        r1, c1 = np.where(components[0])
        r2, c2 = np.where(components[1])
        prompt_points = [(int(c1.mean()), int(r1.mean())),
                         (int(c2.mean()), int(r2.mean()))]

    predictor.set_image(frame_rgb)
    new_particles = []

    for px, py in prompt_points:
        masks, scores, _ = predictor.predict(
            point_coords     = np.array([[px, py]]),
            point_labels     = np.array([1]),
            multimask_output = True,
        )
        best = masks[np.argmax(scores)].astype(bool)
        if best.sum() < FRACTURE_MIN_HALF_AREA:
            continue
        p = make_particle(best, fractured=True)
        if p is not None:
            new_particles.append(p)

    if len(new_particles) == 2:
        overlap = (new_particles[0]["mask"] & new_particles[1]["mask"]).sum()
        total   = new_particles[0]["mask"].sum() + new_particles[1]["mask"].sum()
        if (overlap / total if total > 0 else 1.0) < 0.3:
            return new_particles
    return []


def check_and_resolve_fractures(particles, frame_rgb, predictor,
                                 frame_idx, h, w):
    updated     = []
    n_fractures = 0
    for p in particles:
        split    = check_fracture_by_components(p["mask"], h, w)
        sol_drop = check_fracture_by_solidity(p["mask"], p["baseline_solidity"])
        if split or sol_drop:
            halves = try_split_with_sam(frame_rgb, p, predictor)
            if len(halves) == 2:
                updated.extend(halves)
                n_fractures += 1
                print(f"  [frame {frame_idx:>5}] Fracture split at "
                      f"{p['centroid']} "
                      f"({'components' if split else 'solidity drop'})")
            else:
                p["fractured"] = True
                updated.append(p)
        else:
            updated.append(p)
    return updated, n_fractures


# ─── Annotation ───────────────────────────────────────────────────────────────

def annotate_frame(frame, particles):
    vis = frame.copy()
    for p in particles:
        colour = FRACTURE_COLOUR if p.get("fractured") else CONTOUR_COLOUR
        cv2.drawContours(vis, [p["contour"]], -1, colour, 2)
        cv2.circle(vis, p["centroid"], CENTROID_RADIUS, CENTROID_COLOUR, -1)
    return vis


# ─── Main loop ────────────────────────────────────────────────────────────────

def process_video(video_path, cellpose_model, predictor,
                  diameter, redetect_interval, output_dir):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo      : {video_path}")
    print(f"Resolution : {width} × {height} px")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Diameter   : {'auto' if diameter == 0 else f'{diameter}px'}")
    print(f"Re-detect  : every {redetect_interval} frames"
          if redetect_interval > 0 else "Re-detect  : first frame only")

    stem     = Path(video_path).stem
    out_path = output_dir / f"{stem}_cellpose_fracture.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*OUTPUT_CODEC),
        fps, (width, height)
    )

    particles      = []
    frame_idx      = 0
    dropped_total  = 0
    redetect_count = 0
    fracture_total = 0

    print(f"\nProcessing {total_frames} frames …\n")

    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            is_detection_frame = (
                frame_idx == 0 or
                (redetect_interval > 0 and frame_idx % redetect_interval == 0)
            )

            if is_detection_frame:
                particles = run_cellpose_on_frame(
                    frame, cellpose_model, diameter, frame_idx
                )
                redetect_count += 1
                particles, n_frac = check_and_resolve_fractures(
                    particles, frame_rgb, predictor, frame_idx, height, width
                )
                fracture_total += n_frac

            before    = len(particles)
            particles = [p for p in particles
                         if not mask_touches_edge(p["mask"], height, width)]
            dropped_total += before - len(particles)

            writer.write(annotate_frame(frame, particles))
            pbar.set_postfix(
                active=len(particles), dropped=dropped_total,
                fractures=fracture_total, redetect=redetect_count,
            )
            pbar.update(1)
            frame_idx += 1

    cap.release()
    writer.release()

    print(f"\n✓ Done → {out_path}")
    print(f"  Cellpose ran       : {redetect_count} time(s)")
    print(f"  Fractures detected : {fracture_total}")
    print(f"  Final particles    : {len(particles)}")
    print(f"  Total dropped      : {dropped_total}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cellpose + SAM hybrid particle/fracture detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video")
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--model-type",     default="vit_l",
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--cellpose-model", default="cyto3",
                   choices=["cyto3", "cyto2", "nuclei"],
                   help="Cellpose model (default: cyto3)")
    p.add_argument("--diameter",       type=float, default=0,
                   help="Expected particle diameter in pixels. "
                        "0 = auto-estimate (default)")
    p.add_argument("--redetect-interval", type=int, default=10)
    p.add_argument("--output",         default=".")
    p.add_argument("--device",         default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cellpose_model = load_cellpose(args.cellpose_model, device)
    predictor      = load_sam_predictor(
        args.sam_checkpoint, args.model_type, device
    )

    process_video(
        video_path        = args.video,
        cellpose_model    = cellpose_model,
        predictor         = predictor,
        diameter          = args.diameter,
        redetect_interval = args.redetect_interval,
        output_dir        = output_dir,
    )


if __name__ == "__main__":
    main()