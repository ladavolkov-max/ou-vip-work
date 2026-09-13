"""
SAM 2.1 Particle Detection Script
===================================
Uses Meta's Segment Anything Model 2.1 (SAM 2.1) to detect particles in
microscopy images. SAM 2.1 is free and open source (Apache 2.0 license)
and is significantly more accurate than SAM 1.

Three modes:
  1. AUTO mode    — SAM scans the whole image and proposes all segments.
                    No clicks needed. Good first pass.
  2. PROMPT mode  — You supply point coordinates (particle centres) and SAM
                    segments exactly those particles.
  3. MANUAL mode  — An interactive window opens. Click the centre of each
                    particle and SAM segments it in real time.
                    Z = undo, S = save and exit, Q = quit without saving.

Setup (one-time)
----------------
1. Install dependencies:
       pip install sam-2 opencv-python numpy pandas torch torchvision tqdm

2. Download a SAM 2.1 checkpoint (paste URL into browser to download):
       Tiny    (~150 MB, fastest):
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
       Small   (~180 MB):
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
       Base+   (~320 MB):
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
       Large   (~900 MB, most accurate):
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

   Place the .pt file in the same folder as this script.

3. Run:
       # Auto mode:
       python detect_particles_sam2.py reference_frame.png --checkpoint sam2.1_hiera_large.pt --model-type large

       # Prompt mode:
       python detect_particles_sam2.py reference_frame.png --checkpoint sam2.1_hiera_large.pt --model-type large ^
           --points 120,450 340,820

       # Manual mode:
       python detect_particles_sam2.py reference_frame.png --checkpoint sam2.1_hiera_large.pt --model-type large ^
           --manual

   Model type must match the checkpoint:
       sam2.1_hiera_tiny.pt       → --model-type tiny
       sam2.1_hiera_small.pt      → --model-type small
       sam2.1_hiera_base_plus.pt  → --model-type base_plus
       sam2.1_hiera_large.pt      → --model-type large

GPU note
--------
SAM 2.1 runs on CPU but is much faster with a CUDA GPU.
The script auto-detects CUDA if available, otherwise falls back to CPU.

Output
------
  <stem>_sam2_detected.png   — annotated image (green contours + red centroids)
  <stem>_sam2_particles.csv  — one row per particle with shape statistics
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="sam2")
warnings.filterwarnings("ignore", category=UserWarning,   module="sam2")


# ─── Tunable parameters ───────────────────────────────────────────────────────

# Filters applied to detected segments
MIN_AREA_PX2     =   800    # ignore tiny segments (noise)
MAX_AREA_PX2     = 250_000  # ignore huge segments (background)
MIN_SOLIDITY     =   0.35   # area / convex-hull area — rejects spindly shapes

# AUTO mode: SAM 2 mask-generation settings
POINTS_PER_SIDE  = 48       # grid density — increase for finer coverage (slower)
PRED_IOU_THRESH  = 0.70     # lower → more (noisier) proposals
STABILITY_THRESH = 0.70     # SAM's own confidence score (0–1)

# MANUAL mode display
DISPLAY_SCALE    = 0.4      # scale factor for the interactive window
CLICK_COLOUR     = (0,   0, 255)   # red dot at each clicked point
CONTOUR_COLOUR   = (0, 220,   0)   # green contour for accepted segments

# ──────────────────────────────────────────────────────────────────────────────

# SAM 2.1 model config names (used internally by sam-2 package)
MODEL_CONFIG_MAP = {
    "tiny"      : "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small"     : "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus" : "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large"     : "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def load_sam2(checkpoint: str, model_type: str, device: str):
    """Load SAM 2.1 model onto the specified device."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("ERROR: sam-2 not installed.")
        print("       Run:  pip install sam-2")
        sys.exit(1)

    if model_type not in MODEL_CONFIG_MAP:
        print(f"ERROR: Unknown model type '{model_type}'.")
        print(f"       Choose from: {', '.join(MODEL_CONFIG_MAP.keys())}")
        sys.exit(1)

    config = MODEL_CONFIG_MAP[model_type]
    print(f"Loading SAM 2.1 ({model_type}) from {checkpoint} on {device} …")

    sam2 = build_sam2(config, checkpoint, device=device)
    return sam2


def shape_stats(contour):
    """Return (area, solidity, circularity, equiv_diameter) for a contour."""
    area           = cv2.contourArea(contour)
    perimeter      = cv2.arcLength(contour, True)
    hull_area      = cv2.contourArea(cv2.convexHull(contour))
    solidity       = area / hull_area if hull_area > 0 else 0
    circularity    = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
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
    Run SAM 2 prediction at a single (x, y) point.
    Returns (contour, mask) or (None, None) if rejected.
    """
    masks, scores, _ = predictor.predict(
        point_coords     = np.array([[x, y]]),
        point_labels     = np.array([1]),
        multimask_output = True,
    )
    best_mask = masks[np.argmax(scores)]
    contour   = mask_to_contour(best_mask)
    if contour is None:
        return None, None

    area, _, _, _ = shape_stats(contour)
    if not (MIN_AREA_PX2 <= area <= MAX_AREA_PX2):
        return None, None

    return contour, best_mask


# ─── AUTO mode ────────────────────────────────────────────────────────────────

def run_auto_mode(img_rgb: np.ndarray, sam2) -> list:
    """SAM 2 scans the whole image and proposes all segments; we filter them."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print("Running SAM 2.1 in AUTO mode …")
    generator = SAM2AutomaticMaskGenerator(
        model                  = sam2,
        points_per_side        = POINTS_PER_SIDE,
        pred_iou_thresh        = PRED_IOU_THRESH,
        stability_score_thresh = STABILITY_THRESH,
    )

    import time
    print("Generating masks (this may take a while) …")
    t0    = time.time()
    masks = generator.generate(img_rgb)
    print(f"  Done in {time.time() - t0:.1f}s — {len(masks)} segments proposed")

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

def run_prompt_mode(img_rgb: np.ndarray, sam2, points: list) -> list:
    """Segment particles at each supplied (x, y) coordinate."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"Running SAM 2.1 in PROMPT mode with {len(points)} point(s) …")
    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(img_rgb)

    particles = []
    for x, y in points:
        contour, mask = predict_at_point(predictor, x, y)
        if contour is None:
            print(f"  Skipped ({x}, {y}) — no valid segment found")
            continue
        area, solidity, _, equiv_diam = shape_stats(contour)
        particles.append((contour, mask))
        print(f"  ({x:4d}, {y:4d})  →  area={area:.0f} px²  "
              f"Ø={equiv_diam:.1f} px  solidity={solidity:.2f}")

    print(f"  Kept {len(particles)} particles.")
    return particles


# ─── MANUAL mode ──────────────────────────────────────────────────────────────

def run_manual_mode(img_bgr: np.ndarray, img_rgb: np.ndarray, sam2) -> list:
    """
    Interactive clicking mode.
    Left-click = segment particle, Z = undo, S = save, Q = quit.
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("\nMANUAL mode — interactive window opening …")
    print("  Left-click  : segment particle at clicked location")
    print("  Z           : undo last segmentation")
    print("  S           : save and exit")
    print("  Q           : quit without saving")
    print()

    predictor = SAM2ImagePredictor(sam2)
    print("Encoding image (this may take a few seconds) …")
    predictor.set_image(img_rgb)
    print("Ready — click on particles in the window.\n")

    particles = []
    click_pts = []

    def make_display(base, particles, click_pts):
        vis = base.copy()
        for c, _ in particles:
            cv2.drawContours(vis, [c], -1, CONTOUR_COLOUR, 2)
        for px, py in click_pts:
            cv2.circle(vis, (px, py), 6, CLICK_COLOUR, -1)
        h, w   = vis.shape[:2]
        disp   = cv2.resize(vis, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        lines  = [
            "Left-click: segment particle",
            "Z: undo last   S: save & exit   Q: quit",
            f"Particles accepted: {len(particles)}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(disp, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(disp, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return disp

    win_name = "SAM 2.1 Manual Mode"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    result = {"save": False}

    def on_click(event, x_disp, y_disp, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x_orig = int(x_disp / DISPLAY_SCALE)
        y_orig = int(y_disp / DISPLAY_SCALE)
        print(f"  Clicked ({x_orig}, {y_orig}) — running SAM 2.1 …",
              end=" ", flush=True)
        contour, mask = predict_at_point(predictor, x_orig, y_orig)
        if contour is None:
            print("no valid segment found (try clicking closer to the centre)")
            return
        area, solidity, _, equiv_diam = shape_stats(contour)
        particles.append((contour, mask))
        click_pts.append((x_orig, y_orig))
        print(f"accepted  area={area:.0f} px²  Ø={equiv_diam:.1f} px  "
              f"solidity={solidity:.2f}")
        cv2.imshow(win_name, make_display(img_bgr, particles, click_pts))

    cv2.setMouseCallback(win_name, on_click)
    cv2.imshow(win_name, make_display(img_bgr, particles, click_pts))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("Quit — no results saved.")
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
                cv2.imshow(win_name, make_display(img_bgr, particles, click_pts))
            else:
                print("  Nothing to undo.")
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed — no results saved.")
            break

    cv2.destroyAllWindows()
    return particles if result["save"] else []


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


def save_outputs(img_bgr, particles, df, image_path, output_dir, model_type):
    vis = img_bgr.copy()
    cv2.drawContours(vis, [c for c, _ in particles], -1, CONTOUR_COLOUR, 2)
    for row in df.itertuples():
        cv2.circle(vis, (row.centroid_x, row.centroid_y), 5, CLICK_COLOUR, -1)

    stem    = Path(image_path).stem
    out_img = output_dir / f"{stem}_sam2_{model_type}_minArea{MIN_AREA_PX2}_maxArea{MAX_AREA_PX2}_minSol{MIN_SOLIDITY}_pts{POINTS_PER_SIDE}_iou{str(PRED_IOU_THRESH).replace('.','')}_stab{str(STABILITY_THRESH).replace('.','')}.png"
    out_csv = output_dir / f"{stem}_sam2_particles_{model_type}_minArea{MIN_AREA_PX2}_maxArea{MAX_AREA_PX2}_minSol{MIN_SOLIDITY}_pts{POINTS_PER_SIDE}_iou{str(PRED_IOU_THRESH).replace('.','')}_stab{str(STABILITY_THRESH).replace('.','')}.csv"

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
        description="Detect particles using SAM 2.1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image",
                   help="Path to input microscopy image")
    p.add_argument("--checkpoint", required=True,
                   help="Path to SAM 2.1 checkpoint (.pt file)")
    p.add_argument("--model-type", default="large",
                   choices=["tiny", "small", "base_plus", "large"],
                   help="Model size matching the checkpoint (default: large)")
    p.add_argument("--points", nargs="*", default=None, metavar="X,Y",
                   help="PROMPT mode: particle centre coordinates, "
                        "e.g. --points 120,450 340,820")
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

    # Load SAM 2.1
    sam2 = load_sam2(args.checkpoint, args.model_type, args.device or device)

    # Run selected mode
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manual:
        particles = run_manual_mode(img_bgr, img_rgb, sam2)
    elif args.points:
        try:
            points = [tuple(int(v) for v in pt.split(",")) for pt in args.points]
        except ValueError:
            print("ERROR: --points must be x,y pairs, e.g. --points 120,450 340,820")
            sys.exit(1)
        particles = run_prompt_mode(img_rgb, sam2, points)
    else:
        particles = run_auto_mode(img_rgb, sam2)

    if not particles:
        print("No particles to save.")
        sys.exit(0)

    df = build_dataframe(particles)
    save_outputs(img_bgr, particles, df, args.image, output_dir, args.model_type)
    print()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()