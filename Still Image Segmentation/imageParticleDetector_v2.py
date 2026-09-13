"""
Particle Detection Script  —  Best Current Result
==================================================
Detects dark particles in microscopy images with uneven illumination
using a Difference of Gaussians (DoG) approach.

Detected ~685 particles on the reference frame, covering small dark
particles well and tracing the dark edges of larger particles.
Large particles with subtle contrast remain a known limitation —
see notes at the bottom for next steps.

Usage
-----
    python detect_particles.py <image_path> [--output <dir>]

    # Example:
    python detect_particles.py reference_frame.png --output ./results/

Requirements
------------
    pip install opencv-python-headless numpy pandas

Output
------
  <stem>_detected.png   — original image with green contours + red centroids
  <stem>_particles.csv  — one row per particle with shape statistics
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Tunable parameters ───────────────────────────────────────────────────────

# DoG scales (pixels).
#   SIGMA_FINE   ≈ half the smallest particle radius you care about.
#   SIGMA_COARSE ≈ radius of the largest particle you care about.
#   Increasing SIGMA_COARSE captures larger dark features.
SIGMA_FINE   = 5
SIGMA_COARSE = 25

# Threshold on the DoG response map.
#   = mean  +  DOG_THRESHOLD_SIGMA × std  of the whole DoG image.
#   Lower  → more detections (more noise).
#   Higher → fewer, higher-confidence detections.
DOG_THRESHOLD_SIGMA = 0.5

# Morphological cleanup kernel size (pixels, elliptical).
MORPH_KERNEL_SIZE = 5
MORPH_OPEN_ITER   = 2   # open:  removes thin speckle noise
MORPH_CLOSE_ITER  = 3   # close: fills holes, connects nearby fragments

# Particle size filter (pixels²).
MIN_AREA =    400       # smaller blobs are noise
MAX_AREA = 200_000      # larger blobs are likely merged aggregates

# Solidity = area / convex-hull area.  Rejects very spindly / jagged noise.
MIN_SOLIDITY = 0.30

# ──────────────────────────────────────────────────────────────────────────────


def detect_particles(image_path: str, output_dir: str = ".") -> pd.DataFrame:
    """
    Detect particles in a microscopy image and save annotated outputs.

    Parameters
    ----------
    image_path : str
        Path to the input image (any OpenCV-supported format).
    output_dir : str
        Directory to write the annotated image and CSV into.

    Returns
    -------
    pd.DataFrame
        One row per detected particle with columns:
          particle_id, centroid_x, centroid_y, area_px2, perimeter_px,
          circularity, solidity, bbox_x, bbox_y, bbox_w, bbox_h,
          equiv_diameter_px
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Green channel — typically the highest signal in bright-field microscopy
    green = img[:, :, 1].astype(np.float32)

    # ── 2. Difference of Gaussians ────────────────────────────────────────────
    # DoG isolates features at a specific spatial scale and is robust to the
    # uneven illumination (vignetting) present in the reference image.
    #
    #   g_fine   = image smoothed at fine scale   (retains particle detail)
    #   g_coarse = image smoothed at coarse scale  (estimates local background)
    #   DoG      = g_coarse − g_fine
    #
    # DoG > 0  where a pixel is DARKER than its local neighbourhood
    # → strong positive response at dark particle cores/edges
    g_fine   = cv2.GaussianBlur(green, (0, 0), SIGMA_FINE)
    g_coarse = cv2.GaussianBlur(green, (0, 0), SIGMA_COARSE)
    dog      = g_coarse - g_fine

    # ── 3. Threshold ──────────────────────────────────────────────────────────
    threshold = dog.mean() + DOG_THRESHOLD_SIGMA * dog.std()
    binary    = (dog > threshold).astype(np.uint8) * 255

    # ── 4. Morphological cleanup ──────────────────────────────────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=MORPH_OPEN_ITER)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)

    # ── 5. Find & filter contours ─────────────────────────────────────────────
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    particles = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0 and (area / hull_area) >= MIN_SOLIDITY:
            particles.append(c)

    # ── 6. Per-particle statistics ────────────────────────────────────────────
    rows = []
    for pid, c in enumerate(particles, start=1):
        area      = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        M         = cv2.moments(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))

        circularity    = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
        solidity       = area / hull_area if hull_area > 0 else 0
        equiv_diameter = np.sqrt(4 * area / np.pi)
        cx             = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy             = int(M["m01"] / M["m00"]) if M["m00"] else 0
        bx, by, bw, bh = cv2.boundingRect(c)

        rows.append(dict(
            particle_id       = pid,
            centroid_x        = cx,
            centroid_y        = cy,
            area_px2          = round(area,           2),
            perimeter_px      = round(perimeter,      2),
            circularity       = round(circularity,    4),
            solidity          = round(solidity,       4),
            bbox_x            = bx,
            bbox_y            = by,
            bbox_w            = bw,
            bbox_h            = bh,
            equiv_diameter_px = round(equiv_diameter, 2),
        ))

    df = pd.DataFrame(rows)

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    vis = img.copy()
    cv2.drawContours(vis, particles, -1, (0, 220, 0), 2)
    for row in rows:
        cv2.circle(vis, (row["centroid_x"], row["centroid_y"]), 4, (0, 0, 255), -1)

    stem    = Path(image_path).stem
    out_img = output_dir / f"{stem}_detected.png"
    out_csv = output_dir / f"{stem}_particles.csv"

    cv2.imwrite(str(out_img), vis)
    df.to_csv(str(out_csv), index=False)

    print(f"✓ Detected {len(particles)} particles")
    print(f"  Annotated image → {out_img}")
    print(f"  Statistics CSV  → {out_csv}")
    if not df.empty:
        print(f"\n  Area (px²) :  min={df.area_px2.min():.0f}  "
              f"median={df.area_px2.median():.0f}  "
              f"max={df.area_px2.max():.0f}")
        print(f"  Equiv Ø (px):  min={df.equiv_diameter_px.min():.1f}  "
              f"median={df.equiv_diameter_px.median():.1f}  "
              f"max={df.equiv_diameter_px.max():.1f}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect particles in a microscopy image."
    )
    parser.add_argument("image",    help="Path to input image")
    parser.add_argument("--output", default=".",
                        help="Output directory (default: current dir)")
    args = parser.parse_args()

    df = detect_particles(args.image, output_dir=args.output)
    print()
    print(df.head(10).to_string(index=False))


# ─── Known limitations & suggested next steps ─────────────────────────────────
#
# WHAT WORKS WELL
#   • Small, distinctly dark particles throughout the image
#   • Uniform detection across bright centre and darker edges (DoG handles
#     the vignetting without any explicit correction step)
#
# KNOWN LIMITATION: LARGE PARTICLES
#   The large oval particles (50–250 px equivalent diameter) have a subtle
#   dark boundary ring but a mid-grey interior that is nearly identical in
#   mean intensity to the surrounding matrix (< 5 grey-level difference).
#   The DoG currently traces only their darkest edges rather than their
#   full body.
#
# SUGGESTED NEXT STEPS (in order of effort)
#   1. Tune parameters:  increase SIGMA_COARSE (e.g. 40–60) and lower
#      DOG_THRESHOLD_SIGMA (e.g. 0.2–0.3) to make the DoG more sensitive
#      to larger, lower-contrast features — at the cost of more false positives.
#
#   2. Structure-tensor texture:  compute local gradient orientation coherence.
#      The parallel striations inside large particles produce a high
#      "orientation coherence" score vs. the isotropic granular matrix.
#
#   3. Lightweight ML (e.g. scikit-learn Random Forest or a small U-Net):
#      annotate ~10–20 particles with precise outlines using a tool like
#      LabelMe or QuPath, then train on local patch features (intensity,
#      gradient magnitude, orientation coherence, DoG at multiple scales).
#      Even a simple pixel classifier trained on a single image generalises
#      well to other frames from the same acquisition.