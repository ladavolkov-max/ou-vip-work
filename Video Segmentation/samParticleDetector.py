"""
SAM-Based Particle Detection Script
=====================================
Uses Meta's Segment Anything Model (SAM) to detect particles in microscopy
images. SAM is free and open source (Apache 2.0 license).

Three modes:
  1. AUTO mode    — SAM scans the whole image and proposes all segments.
                    No clicks needed. Good first pass.
  2. PROMPT mode  — You supply point coordinates (particle centres) and SAM
                    segments exactly those particles. More accurate for large,
                    subtle particles that auto-detection may miss.
  3. MANUAL mode  — An interactive window opens. Click the centre of each
                    particle and SAM segments it in real time. Press Z to undo
                    the last click, S to save and exit, Q to quit without saving.

Setup (one-time)
----------------
1. Install dependencies:
       pip install segment-anything opencv-python numpy pandas torch torchvision

   NOTE: use opencv-python (not opencv-python-headless) for MANUAL mode,
   as it needs GUI window support.

2. Download a SAM model checkpoint from Meta's GitHub:
       https://github.com/facebookresearch/segment-anything#model-checkpoints

   Recommended: vit_b  (fastest, ~375 MB)
       https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

   Higher accuracy: vit_h  (~2.5 GB)
       https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

3. Run:
       # Auto mode:
       python detect_particles_sam.py reference_frame.png --checkpoint sam_vit_h_4b8939.pth --model-type vit_h

       # Prompt mode (x,y coordinates of particle centres):
       python detect_particles_sam.py reference_frame.png --checkpoint sam_vit_h_4b8939.pth --model-type vit_h ^
           --points 120,450 340,820 780,230

       # Manual mode (interactive clicking):
       python detect_particles_sam.py reference_frame.png --checkpoint sam_vit_h_4b8939.pth --model-type vit_h ^
           --manual

GPU note
--------
SAM runs on CPU but is much faster with a CUDA GPU.
The script auto-detects CUDA if available, otherwise falls back to CPU.

Output
------
  <stem>_sam_detected.png   — annotated image (green contours + red centroids)
  <stem>_sam_particles.csv  — one row per particle with shape statistics
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="segment_anything")


'''
----Mask Generation Times----
vit_b (375MB) : default settings - 24.9s. 48 points per side - 70.2s.
vit_l (1.16GB): default settings - 45.7s. 48 points per side - 90.7s. 
vit_h (2.39GB): default settings - 52.9s.'''




# ─── Tunable parameters ───────────────────────────────────────────────────────

# Filters applied to detected segments
MIN_AREA_PX2     =   800    # ignore tiny segments (noise) (default 800)
MAX_AREA_PX2     = 250_000  # ignore huge segments (background) (default 250_000)
MIN_SOLIDITY     =   0.35   # area / convex-hull area — rejects spindly shapes (default 0.35)

# AUTO mode: SAM mask-generation settings
POINTS_PER_SIDE  = 48       # grid density — increase for finer coverage (slower) (default 32)
PRED_IOU_THRESH  = 0.86     # lower → more (noisier) proposals (default 0.86)
STABILITY_THRESH = 0.80     # SAM's own confidence score (0–1) (default 0.80)

# MANUAL mode display
DISPLAY_SCALE    = 0.4      # scale factor for the interactive window (fits screen) (default 0.4)
CLICK_COLOUR     = (0,   0, 255)   # red dot at each clicked point (default (0, 0, 255))
CONTOUR_COLOUR   = (0, 220,   0)   # green contour for accepted segments (default (0, 220,   0))
PENDING_COLOUR   = (0, 200, 255)   # yellow contour while SAM is processing (default (0, 200, 255))

# ──────────────────────────────────────────────────────────────────────────────


def load_sam(checkpoint: str, model_type: str, device: str):
    """Load SAM model onto the specified device."""
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("ERROR: segment-anything not installed.")
        print("       Run:  pip install segment-anything")
        sys.exit(1)

    print(f"Loading SAM ({model_type}) from {checkpoint} on {device} …")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return sam


def shape_stats(contour):
    """Return (area, solidity, circularity, equiv_diameter) for a contour."""
    area          = cv2.contourArea(contour)
    perimeter     = cv2.arcLength(contour, True)
    hull_area     = cv2.contourArea(cv2.convexHull(contour))
    solidity      = area / hull_area if hull_area > 0 else 0
    circularity   = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
    equiv_diameter = np.sqrt(4 * area / np.pi)
    return area, solidity, circularity, equiv_diameter


def mask_to_contour(mask: np.ndarray):
    """Return the largest external contour from a binary mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def predict_at_point(predictor, x: int, y: int):
    """
    Run SAM prediction at a single (x, y) point.
    Returns (contour, mask) or (None, None) if rejected.
    """
    masks, scores, _ = predictor.predict(
        point_coords     = np.array([[x, y]]),
        point_labels     = np.array([1]),   # 1 = foreground
        multimask_output = True,
    )
    best_mask = masks[np.argmax(scores)]
    contour   = mask_to_contour(best_mask)
    if contour is None:
        return None, None

    area, solidity, _, _ = shape_stats(contour)
    if not (MIN_AREA_PX2 <= area <= MAX_AREA_PX2):
        return None, None

    return contour, best_mask


# ─── AUTO mode ────────────────────────────────────────────────────────────────

def run_auto_mode(img_rgb: np.ndarray, sam) -> list:
    """SAM scans the whole image and proposes all segments; we filter them."""
    from segment_anything import SamAutomaticMaskGenerator

    print("Running SAM in AUTO mode …")
    generator = SamAutomaticMaskGenerator(
        model                  = sam,
        points_per_side        = POINTS_PER_SIDE,
        pred_iou_thresh        = PRED_IOU_THRESH,
        stability_score_thresh = STABILITY_THRESH,
    )

    print("Running SAM mask generation …")
    t0 = time.time()
    img_rgb_cpu = np.ascontiguousarray(img_rgb)
    masks = generator.generate(img_rgb_cpu)
    print(f"  Done in {time.time() - t0:.1f}s — {len(masks)} segments proposed")
    print(f"  SAM proposed {len(masks)} segments — filtering …")

    particles = []
    for m in tqdm(masks, desc="Filtering segments", unit="mask"):
        if not (MIN_AREA_PX2 <= float(m["area"]) <= MAX_AREA_PX2):
            continue
        if float(m["stability_score"]) < STABILITY_THRESH:
            continue
        contour = mask_to_contour(m["segmentation"])
        if contour is None:
            continue
        _, solidity, _, _ = shape_stats(contour)
        if solidity < MIN_SOLIDITY:
            continue
        particles.append((contour, m["segmentation"]))

    print(f"  Kept {len(particles)} particles after filtering.")
    return particles


# ─── PROMPT mode ──────────────────────────────────────────────────────────────

def run_prompt_mode(img_rgb: np.ndarray, sam, points: list) -> list:
    """Segment particles at each supplied (x, y) coordinate."""
    from segment_anything import SamPredictor

    print(f"Running SAM in PROMPT mode with {len(points)} point(s) …")
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    particles = []
    for x, y in points:
        contour, mask = predict_at_point(predictor, x, y)
        if contour is None:
            print(f"  Skipped ({x}, {y}) — no valid segment found")
            continue
        area, solidity, _, _ = shape_stats(contour)
        particles.append((contour, mask))
        print(f"  ({x:4d}, {y:4d})  →  area={area:.0f} px²  solidity={solidity:.2f}")

    print(f"  Kept {len(particles)} particles.")
    return particles


# ─── MANUAL mode ──────────────────────────────────────────────────────────────

def run_manual_mode(img_bgr: np.ndarray, img_rgb: np.ndarray, sam) -> list:
    """
    Interactive clicking mode.

    Controls
    --------
    Left-click  — click the centre of a particle; SAM segments it immediately
    Z           — undo the last accepted segmentation
    S           — save results and exit
    Q           — quit without saving
    """
    from segment_anything import SamPredictor

    print("\nMANUAL mode — interactive window opening …")
    print("  Left-click  : segment particle at clicked location")
    print("  Z           : undo last segmentation")
    print("  S           : save and exit")
    print("  Q           : quit without saving")
    print()

    predictor = SamPredictor(sam)
    print("Encoding image (this may take a few seconds) …")
    predictor.set_image(img_rgb)
    print("Ready — click on particles in the window.\n")

    particles  = []   # list of (contour, mask)
    click_pts  = []   # list of (x_orig, y_orig) for display

    # Working display canvas (scaled down so it fits on screen)
    def make_display(base, particles, click_pts):
        vis = base.copy()
        for c, _ in particles:
            cv2.drawContours(vis, [c], -1, CONTOUR_COLOUR, 2)
        for px, py in click_pts:
            cv2.circle(vis, (px, py), 6, CLICK_COLOUR, -1)
        h, w = vis.shape[:2]
        disp = cv2.resize(vis, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        # Overlay instructions
        instructions = [
            "Left-click: segment particle",
            "Z: undo last   S: save & exit   Q: quit",
            f"Particles accepted: {len(particles)}",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(disp, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(disp, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return disp

    win_name = "SAM Manual Mode"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    result = {"save": False}

    def on_click(event, x_disp, y_disp, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Convert display coords back to original image coords
        x_orig = int(x_disp / DISPLAY_SCALE)
        y_orig = int(y_disp / DISPLAY_SCALE)

        print(f"  Clicked ({x_orig}, {y_orig}) — running SAM …", end=" ", flush=True)

        contour, mask = predict_at_point(predictor, x_orig, y_orig)
        if contour is None:
            print("no valid segment found (try clicking closer to the particle centre)")
            return

        area, solidity, _, equiv_diam = shape_stats(contour)
        particles.append((contour, mask))
        click_pts.append((x_orig, y_orig))
        print(f"accepted  area={area:.0f} px²  Ø={equiv_diam:.1f} px  solidity={solidity:.2f}")

        disp = make_display(img_bgr, particles, click_pts)
        cv2.imshow(win_name, disp)

    cv2.setMouseCallback(win_name, on_click)

    # Initial display
    disp = make_display(img_bgr, particles, click_pts)
    cv2.imshow(win_name, disp)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("Quit — no results saved.")
            result["save"] = False
            break

        elif key == ord("s") or key == ord("S"):
            print(f"Saving {len(particles)} particle(s) …")
            result["save"] = True
            break

        elif key == ord("z") or key == ord("Z"):
            if particles:
                particles.pop()
                click_pts.pop()
                print("  Undid last segmentation.")
                disp = make_display(img_bgr, particles, click_pts)
                cv2.imshow(win_name, disp)
            else:
                print("  Nothing to undo.")

        # Handle window close button
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed — no results saved.")
            result["save"] = False
            break

    cv2.destroyAllWindows()

    if not result["save"]:
        return []
    return particles


# ─── Output helpers ───────────────────────────────────────────────────────────

def build_dataframe(particles: list) -> pd.DataFrame:
    rows = []
    for pid, (c, _) in enumerate(particles, start=1):
        area, solidity, circularity, equiv_diam = shape_stats(c)
        perimeter      = cv2.arcLength(c, True)
        M              = cv2.moments(c)
        cx             = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy             = int(M["m01"] / M["m00"]) if M["m00"] else 0
        bx, by, bw, bh = cv2.boundingRect(c)
        rows.append(dict(
            particle_id       = pid,
            centroid_x        = cx,
            centroid_y        = cy,
            area_px2          = round(area,          2),
            perimeter_px      = round(perimeter,     2),
            circularity       = round(circularity,   4),
            solidity          = round(solidity,      4),
            bbox_x            = bx,
            bbox_y            = by,
            bbox_w            = bw,
            bbox_h            = bh,
            equiv_diameter_px = round(equiv_diam,    2),
        ))
    return pd.DataFrame(rows)


def save_outputs(img_bgr, particles, df, image_path, output_dir):
    vis = img_bgr.copy()
    cv2.drawContours(vis, [c for c, _ in particles], -1, CONTOUR_COLOUR, 2)
    for row in df.itertuples():
        cv2.circle(vis, (row.centroid_x, row.centroid_y), 5, CLICK_COLOUR, -1)

    stem    = Path(image_path).stem
    out_img = output_dir / f"{stem}_sam_detected.png"
    out_csv = output_dir / f"{stem}_sam_particles.csv"

    cv2.imwrite(str(out_img), vis)
    df.to_csv(str(out_csv), index=False)

    print(f"\n✓ Saved {len(particles)} particles")
    print(f"  Annotated image → {out_img}")
    print(f"  Statistics CSV  → {out_csv}")
    if not df.empty:
        print(f"\n  Area (px²):    min={df.area_px2.min():.0f}  "
              f"median={df.area_px2.median():.0f}  "
              f"max={df.area_px2.max():.0f}")
        print(f"  Equiv Ø (px):  min={df.equiv_diameter_px.min():.1f}  "
              f"median={df.equiv_diameter_px.median():.1f}  "
              f"max={df.equiv_diameter_px.max():.1f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect particles using SAM (Segment Anything Model).\n"
                    "Modes: auto (default), prompt (--points), manual (--manual).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image",
                   help="Path to input microscopy image")
    p.add_argument("--checkpoint", required=True,
                   help="Path to SAM model checkpoint (.pth file)")
    p.add_argument("--model-type", default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"],
                   help="SAM model type matching the checkpoint (default: vit_b)")
    p.add_argument("--points", nargs="*", default=None, metavar="X,Y",
                   help="PROMPT mode: particle centre coordinates, e.g. --points 120,450 340,820")
    p.add_argument("--manual", action="store_true",
                   help="MANUAL mode: interactive clicking window")
    p.add_argument("--output", default=".",
                   help="Output directory (default: current dir)")
    p.add_argument("--device", default=None,
                   help="Device override: 'cuda' or 'cpu' (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"ERROR: Could not read image: {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image: {args.image}  ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)")

    # Load SAM
    sam = load_sam(args.checkpoint, args.model_type, device)

    # Run selected mode
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manual:
        particles = run_manual_mode(img_bgr, img_rgb, sam)
    elif args.points:
        try:
            points = [tuple(int(v) for v in pt.split(",")) for pt in args.points]
        except ValueError:
            print("ERROR: --points must be x,y pairs, e.g. --points 120,450 340,820")
            sys.exit(1)
        particles = run_prompt_mode(img_rgb, sam, points)
    else:
        particles = run_auto_mode(img_rgb, sam)

    if not particles:
        print("No particles to save.")
        sys.exit(0)

    df = build_dataframe(particles)
    save_outputs(img_bgr, particles, df, args.image, output_dir)
    print()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()