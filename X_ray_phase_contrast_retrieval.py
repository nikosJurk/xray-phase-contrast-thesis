
# =============================================================================
# SECTION 1 -- Imports and GPU/CPU setup
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import h5py
from scipy import ndimage
from scipy.ndimage import map_coordinates, shift, gaussian_filter
from types import SimpleNamespace
import pandas as pd
from datetime import datetime
import io
import re
import contextlib
from usefull_functions import (
    check_data, view_stack_jupyter, circular_profile_angle, modulation_lstsq,
    modregger_resolution, compute_mtf_siemens_clean_new, resolution_summary,
)
from PBI_phase_retrieval_functions import *
# Try to import the GPU stack (cupy/pyfftw) and project-specific GPU utilities.
# If anything fails (no GPU, missing packages, etc.), fall back to CPU-only mode.
try:
    import cupy as cp
    import pyfftw
    from utils import *
    from utils import remove_outliers, mshow, mshow_polar, mshow_complex
    from rec import Rec
    GPU_AVAILABLE = True
    try:
        # Actually test that the GPU is usable, not just that cupy imported fine
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1)
    except Exception as exc:
        GPU_AVAILABLE = False
        print(f"GPU not usable ({type(exc).__name__}: {exc}) -> using CPU fallback for remove_outliers")
except Exception as _e:
    GPU_AVAILABLE = False
    print(f"[imports] GPU stack not available here ({_e}) -> using CPU fallback")

# CPU fallback implementation, used only when the GPU version couldn't be imported/used
if not GPU_AVAILABLE:
    def remove_outliers(data, dezinger=3, dezinger_threshold=0.1):
        """CPU version of utils.remove_outliers (median-filter dezinger)."""
        if dezinger <= 0:
            return data
        out = np.array(data, dtype="float32", copy=True)
        for k in range(out.shape[0]):
            frame = out[k]
            # Compare each pixel to a local median; replace outlier ("zinger") pixels
            fdata = ndimage.median_filter(frame, size=(dezinger, dezinger))
            mask = np.abs(frame - fdata) > np.abs(fdata) * dezinger_threshold
            frame[mask] = fdata[mask]
            out[k] = frame
        return out

# Project-specific helper functions used later in the pipeline
from usefull_functions import (
    check_data, view_stack_jupyter, circular_profile_angle, modulation_lstsq,
    modregger_resolution, compute_mtf_siemens_clean_new, resolution_summary,
)
from PBI_phase_retrieval_functions import *


def _sustained_mtf_crossing(freqs, mtf, level=0.1, n_sustain=15):
    """Find the frequency at which the MTF drops below `level` and STAYS
    below it for `n_sustain` consecutive points.

    More robust than a naive "first point below threshold" crossing finder,
    since it avoids false triggers from noise dips near the low-frequency
    normalization band.
    """
    freqs = np.asarray(freqs, dtype=float)
    mtf = np.asarray(mtf, dtype=float)
    order = np.argsort(freqs)
    f, m = freqs[order], mtf[order]
    n = len(m)
    for i in range(n - 1):
        if m[i] < level:
            continue
        remaining = m[i + 1:]
        check_len = min(n_sustain, len(remaining))
        if check_len == 0:
            continue
        if np.all(remaining[:check_len] < level):
            f_a, m_a = f[i], m[i]
            f_b, m_b = f[i + 1], m[i + 1]
            if m_a == m_b:
                return f_b
            return f_a + (level - m_a) * (f_b - f_a) / (m_b - m_a)
    return np.nan


# =============================================================================
# SECTION 2 -- CONFIG (edit these paths/settings before running)
# =============================================================================
# NOTE: the original script pointed at DTU-cluster paths that included a
# personal student ID (e.g. "/zhome/.../s242871/..."). Those have been
# replaced with generic placeholders below -- set them to wherever you keep
# the data on your own system before running.

# Root output directory for saved arrays/figures
OUT_DIR = os.environ.get("MCTF_OUT_DIR", "./outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Directory containing the real, measured HDF5 scans (dark/flat/5 distances)
RAW_DATA_DIR = os.environ.get("MCTF_RAW_DATA_DIR", "./data/raw_scans")

# Directory containing pre-generated simulated projections (.npy files),
# used only when USE_SIMULATED_DATA = True
SIM_RAW_DIR = os.environ.get("MCTF_SIM_DATA_DIR", "./data/simulated")

# Toggle: True = load simulated .npy projections, False = load real h5 scans
USE_SIMULATED_DATA = False

# ---- Figure-sizing convention, used by most plots in this script ----
FIGSIZE_SINGLE = (7, 7)
PANEL_SIZE = (10, 10)


# =============================================================================
# SECTION 3 -- Load data (simulated or real)
# =============================================================================
if USE_SIMULATED_DATA:
    d78 = np.load(os.path.join(SIM_RAW_DIR, "sim_d1.npy"))
    d79 = np.load(os.path.join(SIM_RAW_DIR, "sim_d2.npy"))
    d80 = np.load(os.path.join(SIM_RAW_DIR, "sim_d3.npy"))
    d81 = np.load(os.path.join(SIM_RAW_DIR, "sim_d4.npy"))
    d82 = np.load(os.path.join(SIM_RAW_DIR, "sim_d5.npy"))

    print("Loaded SIMULATED data (already flat-field normalised -- "
          "no dark/flat correction applied).")
    for name, arr in [("d78 (sim D1)", d78), ("d79 (sim D2)", d79),
                       ("d80 (sim D3)", d80), ("d81 (sim D4)", d81),
                       ("d82 (sim D5)", d82)]:
        check_data(arr, name)

    data = np.array([d78, d79, d80, d81, d82], dtype=np.float32)
    print("Final data shape:", data.shape)

else:
    dataset_path = "entry/instrument/orca/data"

    def _load_scan(filename, description):
        with h5py.File(os.path.join(RAW_DATA_DIR, filename), "r") as f:
            arr = f[dataset_path][:].astype("float32")
        print(f"Loaded {filename}. {description}")
        check_data(arr, filename)
        print(" ")
        return arr

    # NOTE: scan numbers below are specific to the DanMAX April 2026 beamtime
    # dataset used for this thesis. Update filenames/descriptions if reusing
    # this script for a different measurement session.
    s76 = _load_scan("scan-0076_orca.h5", "Dark field (100 frames).")
    s77 = _load_scan("scan-0077_orca.h5", "Flat field (100 frames).")
    s78 = _load_scan("scan-0078_orca.h5", "Image closest to focus.")
    s79 = _load_scan("scan-0079_orca.h5", "Sample moved 2 mm.")
    s80 = _load_scan("scan-0080_orca.h5", "Sample moved +5 mm (reference distance image).")
    s81 = _load_scan("scan-0081_orca.h5", "Sample moved +6 mm.")
    s82 = _load_scan("scan-0082_orca.h5", "Sample moved +6 mm further.")

    for name, arr in [("s76", s76), ("s77", s77), ("s78", s78),
                       ("s79", s79), ("s80", s80), ("s81", s81), ("s82", s82)]:
        print(f"{name}: shape={arr.shape}, frames={arr.shape[0]}")


# =============================================================================
# SECTION 4 -- Experimental geometry (measured z1/z2/magnification per distance)
# =============================================================================
z1 = np.array([119.69, 121.69, 126.69, 132.69, 138.69]) * 1e-3  # focus -> sample, meters
z2 = np.array([1437.0, 1435.0, 1430.0, 1424.0, 1418.0]) * 1e-3  # sample -> detector, meters
magnifications = np.array(
    [13.00601554, 12.79225902, 12.28739443, 11.73178084, 11.22424111],
    dtype=np.float32,
)
energy = 19.55  # keV
print("z1 [m] =", z1)
print("z2 [m] =", z2)
print("M =", magnifications)


# =============================================================================
# SECTION 5 -- Beam/detector stability check (real data only)
# =============================================================================
if USE_SIMULATED_DATA:
    print("[SKIPPED] Beam/detector stability check requires multi-frame stacks "
          "(s76-s82), which don't exist for simulated data (single image per "
          "distance, no frame-to-frame statistics to check).")
else:
    detector_gain_e_per_adu = 1.0   # ORCA conversion gain [photo-electrons / ADU]
    dark_offset_adu = float(np.mean(s76))  # average dark-field level, used as background offset

    def compute_means(stack):
        # Average intensity per frame (mean over the two spatial axes, keep the frame axis)
        return stack.mean(axis=(1, 2))

    stacks = {
        "s76 (dark)": s76, "s77 (flat)": s77, "s78": s78,
        "s79": s79, "s80": s80, "s81": s81, "s82": s82,
    }
    colors = {
        "s76 (dark)": "black", "s77 (flat)": "red", "s78": "tab:blue",
        "s79": "tab:blue", "s80": "tab:blue", "s81": "tab:blue", "s82": "tab:blue",
    }

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    axs = axs.flatten()
    for ax, (name, stack) in zip(axs, stacks.items()):
        means = compute_means(stack)
        n_pixels = stack.shape[1] * stack.shape[2]
        ax.plot(means, marker='o', markersize=3, color=colors[name])
        ax.set_title(f"{name}: mean intensity per frame")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Mean intensity")
        ax.grid(alpha=0.3)
        # Relative frame-to-frame variation, as a percentage of the mean
        rel_std = np.std(means) / np.mean(means) * 100
        if name.startswith("s76"):
            # Dark field has no photon signal, so shot-noise comparison doesn't apply
            print(f"{name}: relative frame-to-frame variation = {rel_std:.4f}% "
                  f"(dark: shot-noise comparison not applicable)")
        else:
            # Convert mean signal (above dark offset) to electrons, then compute the
            # theoretical shot-noise-limited relative fluctuation (Poisson statistics)
            signal_e = max(np.mean(means) - dark_offset_adu, 0.0) * detector_gain_e_per_adu
            expected_rel_std = 1 / np.sqrt(signal_e * n_pixels) * 100
            print(f"{name}: relative frame-to-frame variation = {rel_std:.4f}%, "
                  f"expected shot-noise-limited = {expected_rel_std:.6f}%  "
                  f"(ratio = {rel_std / expected_rel_std:.1f}x)")
    axs[-1].axis('off')  # unused subplot (7 datasets in an 8-slot grid)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 6 -- Flat-field correction (real data) / pass-through (simulated data)
# =============================================================================
if USE_SIMULATED_DATA:
    print("[SKIPPED] Flat-field correction not applied -- simulated data is "
          "already flat-field normalised (background ~1, generated directly "
          "by the forward model). `data` was already set in the loading cell.")
    print("Final data shape:", data.shape)
    print(f"min={data.min():.4f}  max={data.max():.4f}  mean={data.mean():.4f}")

    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count or inf_count:
        print(f"WARNING: NaN={nan_count}, Inf={inf_count} in the data!")
    else:
        print("OK: No NaN/Inf in the data.")

else:
    SKIP_FRAMES = 0

    def stack_mean_clean(stack, dezinger=3, dezinger_threshold=0.1):
        # Drop the first SKIP_FRAMES frames (e.g. if the beam/detector needs to settle)
        stack = stack[SKIP_FRAMES:]
        if GPU_AVAILABLE:
            # GPU path: remove outliers frame-by-frame first, then average
            return np.mean(remove_outliers(stack, dezinger=dezinger,
                                            dezinger_threshold=dezinger_threshold), axis=0)
        # CPU path: average first, then dezinger the resulting mean image
        mean_img = stack.mean(axis=0)
        fdata = ndimage.median_filter(mean_img, size=(dezinger, dezinger))
        mask = np.abs(mean_img - fdata) > np.abs(fdata) * dezinger_threshold
        mean_img = np.where(mask, fdata, mean_img)
        return mean_img.astype(np.float32)

    # Clean, averaged dark and flat field images
    dark = stack_mean_clean(s76)
    flat = stack_mean_clean(s77)

    eps = 1e-6
    denom = flat - dark  # flat-field response above dark level

    # Guard against division by (near) zero in dead/low-response pixels
    bad_pixels = denom < eps
    print(f"Pixels with denom<eps: {bad_pixels.sum()} out of {denom.size} "
          f"({100 * bad_pixels.sum() / denom.size:.3f}%)")
    denom = np.where(bad_pixels, eps, denom)

    def correct(scan):
        # Standard flat-field correction: (raw - dark) / (flat - dark)
        scan_mean = stack_mean_clean(scan)
        return (scan_mean - dark) / denom

    # Apply flat-field correction to each of the 5 sample-distance datasets
    d78 = correct(s78)
    d79 = correct(s79)
    d80 = correct(s80)
    d81 = correct(s81)
    d82 = correct(s82)

    data = np.array([d78, d79, d80, d81, d82], dtype=np.float32)
    print("Final data shape:", data.shape)
    print(f"min={data.min():.4f}  max={data.max():.4f}  mean={data.mean():.4f}")

    # Sanity check: make sure the correction didn't introduce NaN/Inf values
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count or inf_count:
        print(f"WARNING: NaN={nan_count}, Inf={inf_count} in the data!")
    else:
        print("OK: No NaN/Inf in the data.")


# =============================================================================
# SECTION 7 -- Derived geometry quantities (wavelength, effective distances, voxel size)
# =============================================================================
# Physical constants (SI units, energy will be converted from keV to J implicitly via eV)
PLANCK_CONSTANT = 4.135667696e-18  # eV*s (note: pre-scaled for the keV energy below)
SPEED_OF_LIGHT = 299792458  # m/s

# X-ray wavelength from energy: lambda = h*c / E
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

detector_pixelsize = 0.55e-6  # meters, physical detector pixel size
focusToDetectorDistance = z1 + z2  # total source-to-detector distance for each position
ndist = len(z1)  # number of propagation distances

# Magnifications normalized to the first (reference) distance
norm_magnifications = magnifications / magnifications[0]

# Effective free-space propagation distance for each configuration (Fresnel-scaling geometry),
# rescaled by the squared normalized magnification so all distances are referenced
# to the same effective sample-plane pixel size
distances = (z1 * z2) / (z1 + z2)
distances = distances * norm_magnifications ** 2

# Effective (sample-plane) voxel size, referenced to the first/reference magnification
voxelsize = detector_pixelsize / magnifications[0]

print("focusToDetectorDistance [m] =", focusToDetectorDistance)
print("effective distances [m] =", distances)
print("voxelsize [m] =", voxelsize)
print("wavelength [m] =", wavelength)


# =============================================================================
# SECTION 8 -- CTF transfer-function plot (combined, all 5 distances)
# =============================================================================
n = data.shape[1]
fx = np.fft.fftfreq(n, d=voxelsize)  # spatial frequency axis, in 1/m
fx_pos = fx[:n // 2]  # keep only positive frequencies

distances_rec = distances / norm_magnifications ** 2  # "unwind" the magnification scaling to get true propagation distances

# Compute the sin^2 CTF term for each distance
taylorExp = []
for k in range(ndist):
    tf = np.sin(np.pi * wavelength * distances_rec[k] * fx_pos ** 2) ** 2
    taylorExp.append(tf)
taylorExp = np.array(taylorExp)

# Sum of all individual transfer functions, normalized to its max (combined sensitivity)
tf_sum = np.sum(taylorExp, axis=0)
tf_sum_norm = tf_sum / np.max(tf_sum)

# Key frequency limits:
fmin = 1 / np.sqrt(2 * wavelength * np.max(distances_rec))  # lowest reliably recoverable frequency
fmax_detector = 1 / (2 * detector_pixelsize)  # Nyquist limit of the raw (unmagnified) detector pixel
fmax_voxel = 1 / (2 * voxelsize)  # Nyquist limit of the effective (magnified) voxel size

print(f"fmin = {fmin:.6e} m^-1")
print(f"fmax detector = {fmax_detector:.6e} m^-1")
print(f"fmax voxel = {fmax_voxel:.6e} m^-1")

plt.figure(figsize=(7, 5))
for k in range(ndist):
    plt.plot(fx_pos, taylorExp[k], "--", linewidth=1, label=rf"$D_{k + 1}$")
plt.plot(fx_pos, tf_sum_norm, linewidth=3, color="black", label=rf"$\sum D_1-D_{ndist}$")
plt.axvline(fmin, color="blue", label=r"$f_{min}$")
plt.axvline(fmax_detector, color="red", label=r"$f_{max,det}$ (raw pixel, unmagnified)")
plt.axvline(fmax_voxel, color="green", label=r"$f_{max,vox}$ (effective, after magnification)")
plt.ylabel(r"$\sin^2(\pi \lambda D f^2)$", fontsize=15)
plt.xlabel(r"$f$ (m$^{-1}$)", fontsize=15)
plt.legend()
plt.ylim(-0.02, 1.02)
plt.xlim(0, 1.0e6)
plt.grid()
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 9 -- Rescale each projection to a common magnification
# =============================================================================
ntheta = 1

scaled_list = []
for k in range(ndist):
    a = ndimage.zoom(data[k], 1 / norm_magnifications[k], order=1)
    scaled_list.append(a.astype(data.dtype))
    print(f"k={k}, scaled shape = {a.shape}")

# All rescaled images may have slightly different shapes -> crop to a common size
min_y = min(a.shape[0] for a in scaled_list)
min_x = min(a.shape[1] for a in scaled_list)
print("Common output shape:", min_y, min_x)

data_scaled = np.zeros((ndist, min_y, min_x), dtype=data.dtype)
for k, a in enumerate(scaled_list):
    # Center-crop each rescaled image to the common (min_y, min_x) size
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    data_scaled[k] = a[
        cy - min_y // 2: cy - min_y // 2 + min_y,
        cx - min_x // 2: cx - min_x // 2 + min_x
    ]
print("data_scaled shape:", data_scaled.shape)

# --- Compare each rescaled distance against the reference (D1): visual diff + sub-pixel shift ---
plot_magnification = True
if plot_magnification:
    for k in range(ndist):
        # Estimate residual sub-pixel shift relative to the reference via cross-correlation
        shift_est = registration_shift(
            data_scaled[k][None, ...], data_scaled[0][None, ...],
            upsample_factor=10, space="real"
        )
        print(f"k={k}: estimated shift relative to reference (D1) = "
              f"dy={shift_est[0, 0]:.3f}px, dx={shift_est[0, 1]:.3f}px")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im = axs[0].imshow(data_scaled[0], cmap="gray", vmin=0.9, vmax=1.2)
        axs[0].set_title("data_scaled ref")
        fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(data_scaled[k], cmap="gray", vmin=0.9, vmax=1.2)
        axs[1].set_title(f"data_scaled dist {k}")
        fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(data_scaled[k] - data_scaled[0], cmap="gray", vmin=-0.1, vmax=0.1)
        axs[2].set_title(f"difference (shift: {shift_est[0, 0]:.2f}, {shift_est[0, 1]:.2f} px)")
        fig.colorbar(im, ax=axs[2])
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 10 -- Crop around the Siemens star and align (sub-pixel shift correction)
# =============================================================================
if USE_SIMULATED_DATA:
    # The simulated star is built centered at (ny-1)/2, (nx-1)/2 -- use that
    # instead of the real-data-specific pixel coordinates.
    _ny_full, _nx_full = data_scaled.shape[1], data_scaled.shape[2]
    cy_star = int((_ny_full - 1) / 2.0)
    cx_star = int((_nx_full - 1) / 2.0)
    print(f"[SIM] Using simulated star center: cy_star={cy_star}, cx_star={cx_star}")
else:
    # NOTE: these pixel coordinates were located manually for this specific
    # dataset/detector crop. If you re-run on a different measurement, find
    # the star center visually first (e.g. plt.imshow(data_scaled[0])) and
    # update these two numbers.
    cy_star = 1250
    cx_star = 2150
    print(f"[REAL] Using real-data star center: cy_star={cy_star}, cx_star={cx_star}")

crop_size = 2500

# Crop a region around a well-defined feature (star target) to use for alignment
star_stack = center_crop_stack(data_scaled, cy_star, cx_star, crop_size)

# High-pass filter to remove slowly varying background and emphasize sharp features,
# which gives a more reliable cross-correlation for shift estimation
star_hp = np.array([highpass(im, sigma=20) for im in star_stack], dtype=np.float32)

# Estimate sub-pixel shift of each distance relative to the reference (k=0)
better_shifts = np.zeros((ndist, 1, 2), dtype=np.float32)
for k in range(ndist):
    better_shifts[k] = registration_shift(
        star_hp[k:k + 1], star_hp[0:1], upsample_factor=100, space="real"
    )
    print(f"{k}: better shift [dy, dx] = {better_shifts[k, 0]}")

# Apply the estimated shifts to the full (uncropped, rescaled) stack
data_shifted_better = data_scaled.copy()
for k in range(ndist):
    data_shifted_better[k:k + 1] = apply_shift_rect(
        data_scaled[k:k + 1], -better_shifts[k:k + 1, 0]
    )

# Re-crop the same region from the now-aligned stack, for verification
data_crop_better = center_crop_stack(data_shifted_better, cy_star, cx_star, crop_size)
print(data_crop_better.shape)

### Verify residual shift after first correction
verify_hp = np.array([highpass(im, sigma=20) for im in data_crop_better], dtype=np.float32)
verify_shifts = np.zeros((ndist, 1, 2), dtype=np.float32)
for k in range(ndist):
    verify_shifts[k] = registration_shift(
        verify_hp[k:k + 1], verify_hp[0:1], upsample_factor=100, space="real"
    )
    print(f"{k}: residual shift [dy, dx] = {verify_shifts[k, 0]}")

# Apply a second, finer correction based on the residual shift just measured
data_crop_better = apply_shift_rect(data_crop_better, -verify_shifts[:, 0])

# Second verification pass, to confirm the alignment has converged
verify_hp_final = np.array([highpass(im, sigma=20) for im in data_crop_better], dtype=np.float32)
verify_shifts_final = np.zeros((ndist, 1, 2), dtype=np.float32)
for k in range(ndist):
    verify_shifts_final[k] = registration_shift(
        verify_hp_final[k:k + 1], verify_hp_final[0:1], upsample_factor=100, space="real"
    )
    print(f"{k}: residual shift after 2nd pass [dy, dx] = {verify_shifts_final[k, 0]}")


# =============================================================================
# SECTION 11 -- Background/edge-level normalization
# =============================================================================
# Correct for per-distance flat-field baseline mismatch introduced by the
# magnification rescale (larger rescale factors, e.g. D4/D5, show more
# edge-region background drift).
_n = data_crop_better.shape[1]
_border = int(_n * 0.04)  # use the outermost ~4% of pixels as background estimate

_bg_levels = np.zeros(ndist, dtype=np.float64)
_last_edge_mask = None  # kept explicit so the post-loop print below doesn't
                         # silently rely on Python's loop-variable leakage
for k in range(ndist):
    _img = data_crop_better[k]
    _edge_mask = np.zeros_like(_img, dtype=bool)
    _edge_mask[:_border, :] = True
    _edge_mask[-_border:, :] = True
    _edge_mask[:, :_border] = True
    _edge_mask[:, -_border:] = True
    _bg_levels[k] = np.median(_img[_edge_mask])
    _last_edge_mask = _edge_mask
    print(f"D{k + 1}: background level (outer {_border}px border) = {_bg_levels[k]:.4f}")

# Normalize every distance to the SAME reference background level (use D1's
# background as the common target, since D1 has the smallest rescale factor
# and hence the cleanest edges)
_ref_bg = _bg_levels[0]
print(f"\nNormalizing all distances to reference background = {_ref_bg:.4f}")
for k in range(ndist):
    if abs(_bg_levels[k]) > 1e-8:
        data_crop_better[k] = data_crop_better[k] * (_ref_bg / _bg_levels[k])
    # NOTE: reports the background level using each distance's OWN edge mask
    # (all masks are geometrically identical since `_border` doesn't change
    # across distances, so re-using `_last_edge_mask` here is safe).
    print(f"D{k + 1}: corrected background level = "
          f"{np.median(data_crop_better[k][_last_edge_mask]):.4f}")


# =============================================================================
# SECTION 12 -- Save the aligned/corrected D1 frame (checkpoint)
# =============================================================================
MTF_MODR_PARAMS = dict(
    r_min=10, r_max=950, n_radii=1000, n_arms=36,
    exclude_radius_ranges=((53, 60), (95, 145), (200, 280), (430, 550)),
    exclude_angle_ranges=((125, 150),),
    norm_radius_range=(575, 950),
)

MODREGGER_HALF = 900
MODREGGER_NBLFAC = 2.0
MODREGGER_HIGHFRQ = 2.0
PANEL_SIZE = (6, 6)  # overrides the larger SECTION 2 default for the small checkpoint plot below

_tag = "sim" if USE_SIMULATED_DATA else "measured"
measured_img_aligned = data_crop_better[0]
np.save(os.path.join(OUT_DIR, f"{_tag}_D1_aligned.npy"), measured_img_aligned)

fig_save, ax_save = plt.subplots(figsize=PANEL_SIZE)
ax_save.imshow(measured_img_aligned, cmap="gray",
               vmin=np.percentile(measured_img_aligned, 2),
               vmax=np.percentile(measured_img_aligned, 98))
ax_save.set_title(f"{_tag.capitalize()} D1 (aligned, cropped) -- saved", fontsize=11)
ax_save.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{_tag}_D1_aligned.png"), dpi=130)
plt.show()
print(f"Saved: {os.path.join(OUT_DIR, f'{_tag}_D1_aligned.npy')}  "
      f"shape={measured_img_aligned.shape}")
print(f"Saved: {os.path.join(OUT_DIR, f'{_tag}_D1_aligned.png')}")


# =============================================================================
# SECTION 13 -- Visual check of final alignment quality across all distances
# =============================================================================
med = np.median(data_crop_better[0])  # use reference frame's median as the display center point

for k in range(ndist):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(data_crop_better[0], cmap="gray", vmin=med - 0.1, vmax=med + 0.1)
    axs[0].set_title("Reference")
    axs[1].imshow(data_crop_better[k], cmap="gray", vmin=med - 0.1, vmax=med + 0.1)
    axs[1].set_title(f"Better aligned dist {k}")
    axs[2].imshow(data_crop_better[k] - data_crop_better[0], cmap="gray", vmin=-0.1, vmax=0.1)
    axs[2].set_title("Difference")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 14 -- Residual magnification (scale) error estimation and correction
# =============================================================================
def _zoom_and_center(img, scale, out_shape):
    """Rescale `img` by `scale` and place it centered on a zero-padded canvas of `out_shape`."""
    zoomed = ndimage.zoom(img, scale, order=1)
    zy, zx = zoomed.shape
    ny, nx = out_shape
    cy, cx = ny // 2, nx // 2
    cy2, cx2 = zy // 2, zx // 2
    canvas = np.zeros(out_shape, dtype=img.dtype)
    y0 = max(0, cy2 - cy)
    x0 = max(0, cx2 - cx)
    oy0 = max(0, cy - cy2)
    ox0 = max(0, cx - cx2)
    h = min(ny - oy0, zy - y0)
    w = min(nx - ox0, zx - x0)
    canvas[oy0:oy0 + h, ox0:ox0 + w] = zoomed[y0:y0 + h, x0:x0 + w]
    return canvas


def _ncc(a, b):
    """Normalized cross-correlation between two images (single scalar similarity score)."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.sum(a * b) / denom)


def estimate_scale_error(img_k, img_ref, coarse_range=(0.97, 1.03), coarse_n=61,
                          fine_half_width=0.003, fine_n=41, downsample=4):
    """Estimate the residual scale mismatch between `img_k` and `img_ref` via
    a two-stage (coarse, then fine) normalized-cross-correlation search."""
    ref_small = img_ref[::downsample, ::downsample]
    k_small = img_k[::downsample, ::downsample]
    scales = np.linspace(*coarse_range, coarse_n)
    scores = np.zeros_like(scales)
    for i, s in enumerate(scales):
        cand = _zoom_and_center(k_small, s, ref_small.shape)
        scores[i] = _ncc(cand, ref_small)
    coarse_best = scales[np.argmax(scores)]
    fine_scales = np.linspace(coarse_best - fine_half_width,
                               coarse_best + fine_half_width, fine_n)
    fine_scores = np.zeros_like(fine_scales)
    for i, s in enumerate(fine_scales):
        cand = _zoom_and_center(img_k, s, img_ref.shape)
        fine_scores[i] = _ncc(cand, img_ref)
    best_idx = np.argmax(fine_scores)
    best_scale = fine_scales[best_idx]
    best_score = fine_scores[best_idx]
    return best_scale, best_score


def _pad_apodize(img, pad_y, pad_x):
    """Reflect-pad and apodize the border so the periodic FFT does not wrap.
    Handles non-square images with independent y/x padding."""
    out = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    ny, nx = out.shape
    wy = np.ones(ny, dtype=np.float32)
    wx = np.ones(nx, dtype=np.float32)
    if pad_y > 0:
        ramp_y = 0.5 * (1 - np.cos(np.pi * np.arange(pad_y) / pad_y))
        wy[:pad_y] = ramp_y
        wy[-pad_y:] = ramp_y[::-1]
    if pad_x > 0:
        ramp_x = 0.5 * (1 - np.cos(np.pi * np.arange(pad_x) / pad_x))
        wx[:pad_x] = ramp_x
        wx[-pad_x:] = ramp_x[::-1]
    return (out * wy[:, None] * wx[None, :]).astype(np.float32)


def CTFPurePhase(rads, wlen, dists, fx, fy, alpha=1e-3, pad=None,
                  smooth_denominator=0.0, fmax_cyc_per_m=None, half=False):
    """Multi-distance CTF pure-phase inversion.

    `half=False`: the returned phase is on the same absolute scale as the
    simulation script's reconstruction (no extra 0.5 factor). Set
    `half=True` to restore the legacy halved-phase behaviour.

    Handles non-square inputs safely (pad_y/pad_x computed independently),
    even though in this pipeline `rads` is normally square after cropping.
    """
    rads = np.asarray(rads)
    dists = np.asarray(dists)
    ny0, nx0 = rads.shape[-2], rads.shape[-1]

    # Default padding: 1/8 of each dimension, computed independently
    if pad is None:
        pad_y = max(ny0 // 8, 1)
        pad_x = max(nx0 // 8, 1)
    else:
        pad_y = pad_x = pad

    if pad_y > 0 or pad_x > 0:
        # Pad + apodize each radiograph, then rebuild the frequency grid for the new size
        dpix_y = 1.0 / (fy.shape[0] * (fy[1, 0] - fy[0, 0]))
        dpix_x = 1.0 / (fx.shape[1] * (fx[0, 1] - fx[0, 0]))
        rads = np.array([_pad_apodize(r, pad_y, pad_x) for r in rads], dtype=np.float32)
        npad_y, npad_x = rads.shape[-2], rads.shape[-1]
        fy1 = np.fft.fftfreq(npad_y, d=dpix_y)
        fx1 = np.fft.fftfreq(npad_x, d=dpix_x)
        fy_pad, fx_pad = np.meshgrid(fy1, fx1, indexing="ij")
        fx, fy = fx_pad, fy_pad

    numerator = np.zeros_like(np.fft.fft2(rads[0]), dtype=np.complex64)
    denominator = np.zeros_like(rads[0], dtype=np.float32)
    q2 = fx ** 2 + fy ** 2  # squared spatial frequency magnitude

    # Accumulate the CTF-weighted numerator/denominator across all distances (least-squares combination)
    for j in range(len(dists)):
        rad_freq = np.fft.fft2(rads[j])
        ctf = np.sin(np.pi * wlen * dists[j] * q2)
        numerator += ctf * rad_freq
        denominator += 2 * ctf ** 2
    numerator /= len(dists)
    denominator = denominator / len(dists)

    # Optional smoothing of the denominator (in frequency space) to reduce noise near CTF zero-crossings
    if smooth_denominator:
        denominator = ndimage.gaussian_filter(
            np.fft.fftshift(denominator), sigma=smooth_denominator)
        denominator = np.fft.ifftshift(denominator)

    # Regularization term to avoid division blow-up where the CTF is near zero
    denominator = denominator + alpha
    ratio = numerator / denominator
    if half:
        ratio = ratio * 0.5  # legacy scaling option, kept for backward compatibility

    # Optional low-pass cutoff (soft Gaussian roll-off) at a chosen max frequency
    if fmax_cyc_per_m is not None:
        ratio = ratio * np.exp(-(q2 / fmax_cyc_per_m ** 2) ** 2)

    phase = np.real(np.fft.ifft2(ratio))

    # Remove the padding added earlier, back to the original image size
    if pad_y > 0 or pad_x > 0:
        phase = phase[pad_y:-pad_y if pad_y > 0 else None,
                      pad_x:-pad_x if pad_x > 0 else None]
    return phase.astype(np.float32)


ndist = data_crop_better.shape[0]
n = data_crop_better.shape[1]

REFINE_SCALE = True

# --- Estimate any residual magnification mismatch (beyond the nominal geometry) per distance ---
scale_hp = np.array([highpass(im, sigma=20) for im in data_crop_better], dtype=np.float32)
residual_scales = np.ones(ndist, dtype=np.float64)
for k in range(1, ndist):
    s_best, s_score = estimate_scale_error(scale_hp[k], scale_hp[0])
    residual_scales[k] = s_best
    print(f"k={k}: residual scale vs D1 = {s_best:.5f} (NCC={s_score:.4f}) "
          f"-> implied magnification {magnifications[k] / s_best:.3f} "
          f"instead of {magnifications[k]:.3f}")

# --- If enabled, correct each distance for its residual scale error, then re-align shifts ---
if REFINE_SCALE:
    for k in range(1, ndist):
        if abs(residual_scales[k] - 1.0) > 1e-4:
            data_crop_better[k] = _zoom_and_center(
                data_crop_better[k], residual_scales[k], data_crop_better[0].shape)

    # Re-check (and correct) sub-pixel shifts after the scale correction, since rescaling can shift content
    refine_hp = np.array([highpass(im, sigma=20) for im in data_crop_better], dtype=np.float32)
    refine_shifts = np.zeros((ndist, 1, 2), dtype=np.float32)
    for k in range(ndist):
        refine_shifts[k] = registration_shift(refine_hp[k:k + 1], refine_hp[0:1],
                                               upsample_factor=100, space="real")
        print(f"k={k}: shift after scale refinement = {refine_shifts[k, 0]}")
    data_crop_better = apply_shift_rect(data_crop_better, -refine_shifts[:, 0])


# =============================================================================
# SECTION 15 -- Run multi-distance CTF phase retrieval for several distance combos
# =============================================================================
rdata_ctf = data_crop_better - 1.0  # subtract the flat-field baseline (background = 0)
fx = np.fft.fftfreq(n, d=voxelsize)
fx, fy = np.meshgrid(fx, fx)
wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

# Choose which distance convention to feed into the reconstruction:
# "scaled" = magnification-rescaled distances, "raw" = true physical distances
DISTANCE_CONVENTION = "raw"
distances_for_rec = {
    "scaled": distances,
    "raw": distances / norm_magnifications ** 2,
}[DISTANCE_CONVENTION]
print(f"DISTANCE_CONVENTION={DISTANCE_CONVENTION}, distances [m] = {distances_for_rec}")

distance_labels = ["D1", "D2", "D3", "D4", "D5"]

# Different combinations of distances to test in the multi-distance CTF reconstruction
combinations = {
    "Case 01: D1": [0],
    "Case 06: D1+D2": [0, 1],
    "Case 09: D1+D2+D3": [0, 1, 2],
    "Case 10: D1+D2+D3+D4": [0, 1, 2, 3],
    "Case 11: D1+D2+D3+D4+D5": [0, 1, 2, 3, 4],
    "Case 07: D1+D5": [0, 4],
    "Case 12: D1+D3+D4+D5": [0, 2, 3, 4],
}

alpha_val = 1e-2  # Tikhonov-style regularization strength for the CTF denominator

CTF_DENOM_SMOOTH = 0.0
CTF_FMAX = None
CTF_PAD = None

# --- Run CTF phase retrieval for every distance combination ---
CTFrec_dict = {}
for label, idx in combinations.items():
    idx = np.array(idx)
    rads = rdata_ctf[idx]
    # Local variable name, does not shadow the module-level `distances_rec`
    # used later for the transfer-function grid plots.
    distances_case = distances_for_rec[idx]
    recCTFPurePhase = CTFPurePhase(
        rads, wlen, distances_case, fx, fy, alpha_val,
        pad=CTF_PAD, smooth_denominator=CTF_DENOM_SMOOTH,
        fmax_cyc_per_m=CTF_FMAX)
    CTFrec_dict[label] = recCTFPurePhase
    print(f"{label}: used distances {idx}, reconstruction shape {recCTFPurePhase.shape}")

# Use the full 5-distance reconstruction to set a consistent display range for all cases
_ref_rec = CTFrec_dict["Case 11: D1+D2+D3+D4+D5"]
CLIM = (float(np.percentile(_ref_rec, 1)), float(np.percentile(_ref_rec, 99)))
print(f"display range (1-99 percentile of Case 11) = {CLIM}")

# --- Plot all reconstructions in a grid for visual comparison ---
ncols = 4
nrows = int(np.ceil(len(combinations) / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
axs = axs.flatten()
for ax, (label, rec) in zip(axs, CTFrec_dict.items()):
    im = ax.imshow(rec, cmap="gray", vmin=CLIM[0], vmax=CLIM[1])
    ax.set_title(label, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax in axs[len(combinations):]:
    ax.axis('off')
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 16 -- CTF transfer-function plots (one per distance combination)
# =============================================================================
n = data_crop_better.shape[1]
fx = np.fft.fftfreq(n, d=voxelsize)
fx_pos = fx[:n // 2]
distances_rec = distances_for_rec
fmin = 1 / np.sqrt(2 * wavelength * np.max(distances_rec))
fmax_detector = 1 / (2 * detector_pixelsize)
fmax_voxel = 1 / (2 * voxelsize)

for label, idx_list in combinations.items():
    idx_array = np.array(idx_list)
    sub_distances = distances_rec[idx_array]

    # Individual sin^2 CTF term for each distance used in this combination
    taylorExp = []
    for d_val in sub_distances:
        tf = np.sin(np.pi * wavelength * d_val * fx_pos ** 2) ** 2
        taylorExp.append(tf)
    taylorExp = np.array(taylorExp)

    # Combined (summed, normalized) transfer function for this combination
    tf_sum = np.sum(taylorExp, axis=0)
    tf_sum_norm = tf_sum / np.max(tf_sum) if np.max(tf_sum) > 0 else tf_sum

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, orig_idx in enumerate(idx_array):
        ax.plot(fx_pos, taylorExp[i], "--", linewidth=0.8, alpha=0.7,
                label=rf"$D_{orig_idx + 1}$")
    ax.plot(fx_pos, tf_sum_norm, linewidth=2, color="black", label=label)

    ax.axvline(fmin, color="blue", linestyle=":", alpha=0.5, label=r"$f_{min}$")
    ax.axvline(fmax_voxel, color="green", linestyle=":", alpha=0.5, label=r"$f_{max}$ (voxel)")

    ax.set_title(label, fontsize=11)
    ax.set_xlabel(r"$f$ (m$^{-1}$)", fontsize=10)
    ax.set_ylabel(r"$\sin^2(\pi\lambda D f^2)$", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, min(1.0e6, fx_pos[-1]))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 17 -- Multi-method comparison for two representative materials (Ta, Si3N4)
# =============================================================================
psize = voxelsize  # alias, this section uses "psize" instead of "voxelsize"

ndist, ny, nx = rdata_ctf.shape
rads_intensity = rdata_ctf + 1.0

# -----------------------------------------------------------------------
# `fy` was already a correct 2D meshgrid (ny, nx), but `fx` had been left
# as a 1D vector (nx,) -- it never went through np.meshgrid. Any function
# that does fx.shape[1] or broadcasts fx**2 + fy**2 with mismatched shapes
# (e.g. CTFPurePhase) would either crash or silently produce wrong results.
#
# We take the 1D profile from the already-correct `fy` (fy[:, 0]) so that
# `fy_grid` stays bit-for-bit identical to before, and only rebuild
# `fx_grid` as a proper 2D meshgrid with the same indexing convention.
# -----------------------------------------------------------------------
assert fx.ndim == 1, f"expected fx to be 1D, got shape {fx.shape}"
assert fy.ndim == 2, f"expected fy to be 2D, got shape {fy.shape}"
assert fx.shape[0] == nx, f"fx length {fx.shape[0]} != nx {nx}"
assert fy.shape == (ny, nx), f"fy shape {fy.shape} != (ny, nx)=({ny},{nx})"

fy_1d = fy[:, 0]
fy_grid, fx_grid = np.meshgrid(fy_1d, fx, indexing='ij')

print(f"[fix] fx was {fx.shape} (1D) -> fx_grid now {fx_grid.shape} (2D)")
print(f"[fix] fy_grid unchanged, shape {fy_grid.shape}")
assert fx_grid.shape == (ny, nx) and fy_grid.shape == (ny, nx), \
    f"grid shape mismatch after fix: fx_grid={fx_grid.shape}, fy_grid={fy_grid.shape}"

distances_rec = distances_for_rec

_Rm_full_sim = np.ones((ny, nx, ndist), dtype=np.float32)
print(f"Shapes check - rads: {rdata_ctf.shape}, Rm: {_Rm_full_sim.shape}, fx_grid: {fx_grid.shape}")

# Dynamically locate the "full" combination (all distances together), so
# downstream code doesn't rely on a hardcoded label that could break if
# `combinations` changes.
all_indices = list(range(ndist))
full_label = None
if 'combinations' in globals():
    for label, idx in combinations.items():
        if sorted(list(idx)) == all_indices:
            full_label = label
            break
    if full_label is None:
        full_label = list(combinations.keys())[-1]
    idx_all = np.array(combinations[full_label])
else:
    full_label = "all"
    idx_all = np.arange(ndist)

print(f"full_label = '{full_label}'")

results_to_plot = {}

# --- CTF (with Rm weighting) -- multi-distance, all combinations ---
ny_data, nx_data = ny, nx
_Rm_full = np.ones((ny_data, nx_data, ndist), dtype=np.float32)

CTF_dict = {}
for label, idx in combinations.items():
    idx = np.array(idx)
    rads = rdata_ctf[idx]
    distances_case = distances_rec[idx]
    _Rm_case = _Rm_full[:, :, idx]
    rec = CTF(rads, wlen, distances_case, fx_grid, fy_grid, _Rm_case, alpha_val)
    CTF_dict[label] = rec.astype(np.float32)
    print(f"[CTF] {label}: reconstruction shape {rec.shape}")

results_to_plot[f"CTF (Rm) ({full_label})"] = CTF_dict[full_label]

# --- multiCTF (material-independent) ---
rads_ctf_all = rdata_ctf[idx_all]
dist_case_ctf = distances_rec[idx_all]
Rm_case_ctf = _Rm_full_sim[:, :, idx_all]
rec_multictf = CTF(rads_ctf_all, wlen, dist_case_ctf, fx_grid, fy_grid, Rm_case_ctf, alpha_val)
results_to_plot[f"multiCTF ({full_label})"] = rec_multictf.astype(np.float32)

# --- CTFPurePhase (material-independent) ---
rec_pure_phase = CTFPurePhase(rads_ctf_all, wlen, dist_case_ctf, fx_grid, fy_grid, alpha_val)
results_to_plot[f"CTFPurePhase ({full_label})"] = rec_pure_phase.astype(np.float32)

rads_data_all = rads_intensity[idx_all]

# Representative delta/beta values for the two materials used in the
# experimental Siemens star target (values at the beamtime's photon energy).
materials = {
    "Ta": {"delta": 7.272642e-06, "beta": 5.660273e-07},
    "Si3N4": {"delta": 1.745160e-06, "beta": 5.058573e-09},
}

for mat_name, props in materials.items():
    delta = props["delta"]
    beta = props["beta"]

    rec_multihomo = homoCTF(rads_data_all, wlen, dist_case_ctf, delta, beta,
                             fx_grid, fy_grid, Rm_case_ctf, alpha_val)
    results_to_plot[f"multihomoCTF [{mat_name}] ({full_label})"] = rec_multihomo.astype(np.float32)

    rec_multipag = multiPaganin(rads_data_all, wlen, dist_case_ctf, delta, beta,
                                 fx_grid, fy_grid, Rm_case_ctf, alpha_val)
    results_to_plot[f"multiPaganin [{mat_name}] ({full_label})"] = rec_multipag.astype(np.float32)

    rec_single_pag_d1 = Paganin(rads_intensity[0], wlen, distances_rec[0], delta, beta,
                                 fx_grid, fy_grid, _Rm_full_sim[:, :, 0])
    results_to_plot[f"Single Paganin D1 [{mat_name}]"] = rec_single_pag_d1.astype(np.float32)

    rec_sgldst_d1 = sglDstCTF(rads_intensity[0], wlen, distances_rec[0], delta, beta,
                               fx_grid, fy_grid, _Rm_full_sim[:, :, 0], alpha_val)
    results_to_plot[f"sglDstCTF D1 [{mat_name}]"] = rec_sgldst_d1.astype(np.float32)

# --- Vertical comparison plot of every method computed above ---
n_plots = len(results_to_plot)
fig, axes = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots))
if n_plots == 1:
    axes = [axes]

for ax, (title, img) in zip(axes, results_to_plot.items()):
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 18 -- MTF analysis (direct, circular profiles on the real-valued reconstruction)
# =============================================================================
# NOTE: xcenter/ycenter below were located manually for this specific dataset
# (Siemens star center in the cropped/aligned frame). Re-measure and update
# if reusing this script on different data.
xcenter, ycenter = 1273, 1278

r_min, r_max, n_radii = 50, 950, 1000
n_arms = 36
N_SAMPLES_PROFILE = 4096
NOISE_CORRECT_MODULATION = True
NORM_RADIUS_RANGE = (575, 950)
NOISE_FREQ_OFFSETS = (-3, -2, +2, +3, +5, +7)
voxelsize_um = voxelsize * 1e6
nyquist = 1 / (2 * voxelsize_um)

_exclude_radius_ranges = ((110, 150), (230, 290), (470, 575))
_exclude_angle_ranges = ((125, 150),)

# Dynamically locate the "full" combination (all distances together) again
# here, since downstream code (e.g. the combined MTF+Modregger table)
# should use `full_label` rather than a hardcoded string.
_all_indices = list(range(ndist))
full_label = None
for label, idx in combinations.items():
    if sorted(list(idx)) == _all_indices:
        full_label = label
        break
if full_label is None:
    full_label = list(combinations.keys())[-1]
    print(f"[Section 18] WARNING: no case with all distances found -- "
          f"using the last case as fallback: '{full_label}'")
else:
    print(f"[Section 18] full_label = '{full_label}'")


def _freq_of_radius(r_px):
    return n_arms / (2 * np.pi * r_px * voxelsize_um)


_radii_all = np.linspace(r_min, r_max, n_radii)


def _bad_radius(r):
    return any(r0 <= r <= r1 for r0, r1 in _exclude_radius_ranges)


radii_used = np.array([r for r in _radii_all if not _bad_radius(r)])
freqs_used = _freq_of_radius(radii_used)
noise_freqs = [n_arms + off for off in NOISE_FREQ_OFFSETS if (n_arms + off) > 0]


def extract_profiles(img):
    """Circular profiles taken DIRECTLY on the real-valued reconstruction."""
    return [circular_profile_angle(img, xcenter, ycenter, r,
                                    n_samples=N_SAMPLES_PROFILE)
            for r in radii_used]


def modulation_from_profiles(profiles, exclude_angle_ranges=()):
    sig, noi = [], []
    for angle_deg, prof in profiles:
        amp, noise_amp = modulation_lstsq(
            angle_deg, prof, n_arms=n_arms,
            exclude_angle_ranges=exclude_angle_ranges,
            noise_freqs=noise_freqs)
        sig.append(amp)
        noi.append(noise_amp)
    return np.asarray(sig, float), np.asarray(noi, float)


def denoise_modulation(sig, noi):
    sig = np.asarray(sig, float)
    if not NOISE_CORRECT_MODULATION:
        return sig
    noi = np.nan_to_num(np.asarray(noi, float), nan=0.0)
    return np.sqrt(np.clip(sig ** 2 - noi ** 2, 0.0, None))


def normalise_mtf(sig, noi=None):
    eps = 1e-12
    sig_raw = np.asarray(sig, float)
    sig = denoise_modulation(sig_raw, noi) if noi is not None else sig_raw
    sel = (radii_used >= NORM_RADIUS_RANGE[0]) & (radii_used <= NORM_RADIUS_RANGE[1])
    if not np.any(sel):
        raise ValueError(f"NORM_RADIUS_RANGE={NORM_RADIUS_RANGE} has no "
                          f"samples inside [{r_min}, {r_max}].")
    return sig / (np.nanmean(sig[sel]) + eps)


def _sorted_by_freq(mtf):
    order = np.argsort(freqs_used)
    return freqs_used[order], np.asarray(mtf)[order], radii_used[order]


# ---- compute MTF per case, directly on the real-valued reconstruction ----
mtf_results_combos = {}
for label, img in CTFrec_dict.items():
    profiles = extract_profiles(img)
    sig, noi = modulation_from_profiles(profiles, _exclude_angle_ranges)
    mtf = normalise_mtf(sig, noi)
    f_s, mtf_s, r_s = _sorted_by_freq(mtf)
    mtf_results_combos[label] = dict(freqs=f_s, mtf=mtf_s, radii=r_s)


def _fmt(v):
    return f"{v:.1f}" if np.isfinite(v) else "N/A"


print(f"{'Case':<28s} {'MTF10 [nm]':>15s}")
print("-" * 50)
mtf10_summary = {}
for label, res in mtf_results_combos.items():
    f10 = _sustained_mtf_crossing(res["freqs"], res["mtf"])
    res_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan
    mtf10_summary[label] = res_nm
    print(f"{label:<28s} {_fmt(res_nm):>15s}")

# ---- one MTF plot per case ----
_colors = plt.cm.tab10(np.linspace(0, 1, len(mtf_results_combos)))
for (label, res), c in zip(mtf_results_combos.items(), _colors):
    plt.figure(figsize=(7, 5))
    plt.plot(res["freqs"], res["mtf"], "-o", markersize=3, linewidth=1.3,
              color=c, label=label)
    plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
    if nyquist < res["freqs"].max() * 1.2:
        plt.axvline(nyquist, linestyle="--", color="gray", alpha=0.6, label="Nyquist")
    plt.xlabel("Spatial frequency (cycles/µm)")
    plt.ylabel("Normalized MTF (direct)")
    plt.title(f"MTF (direct) -- {label}")
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

# ---- one combined plot with all MTF curves together ----
plt.figure(figsize=FIGSIZE_SINGLE)
_colors = plt.cm.tab10(np.linspace(0, 1, len(mtf_results_combos)))
for (label, res), c in zip(mtf_results_combos.items(), _colors):
    plt.plot(res["freqs"], res["mtf"], "-o", markersize=3, linewidth=1.3,
              color=c, label=label)
plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
if nyquist < max(res["freqs"].max() for res in mtf_results_combos.values()) * 1.2:
    plt.axvline(nyquist, linestyle="--", color="gray", alpha=0.6, label="Nyquist")
plt.xlabel("Spatial frequency (cycles/µm)")
plt.ylabel("Normalized MTF (direct)")
plt.title("MTF (direct) -- all cases")
plt.grid(True)
plt.legend(fontsize=8, loc="upper right")
plt.ylim(0, 1.2)
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 19 -- Visual check: reconstruction with circular profiles / exclusion zones overlaid
# =============================================================================
case_label_to_plot = full_label  # "full" case: all distances together
img_case5 = CTFrec_dict[case_label_to_plot]
_exclude_radius_ranges = ((110, 150), (230, 290), (470, 575))
_exclude_angle_ranges = ((125, 150),)
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(img_case5, cmap="gray")
ax.set_title(f"Circular profiles + excluded regions -- {case_label_to_plot}")

# --- all circular profiles (radii_used), thin cyan lines ---
for r in radii_used:
    circle = plt.Circle((xcenter, ycenter), r, fill=False,
                         edgecolor="cyan", linewidth=0.3, alpha=0.35)
    ax.add_patch(circle)

# --- excluded radius ranges (_exclude_radius_ranges), red annulus ---
for r0, r1 in _exclude_radius_ranges:
    wedge = patches.Wedge((xcenter, ycenter), r1, 0, 360,
                           width=(r1 - r0),
                           facecolor="red", edgecolor="none",
                           alpha=0.30)
    ax.add_patch(wedge)

# --- excluded angular range (_exclude_angle_ranges), yellow wedge ---
for a0, a1 in _exclude_angle_ranges:
    wedge = patches.Wedge((xcenter, ycenter), r_max, a0, a1,
                           width=(r_max - r_min),
                           facecolor="yellow", edgecolor="none",
                           alpha=0.30)
    ax.add_patch(wedge)

# --- center marker ---
ax.plot(xcenter, ycenter, "+", color="lime", markersize=14, markeredgewidth=2)

legend_elems = [
    Line2D([0], [0], color="cyan", linewidth=1.5, label="circular profiles"),
    Patch(facecolor="red", alpha=0.30, label="excluded radius ranges"),
    Patch(facecolor="yellow", alpha=0.30, label="excluded angle range"),
    Line2D([0], [0], marker="+", color="lime", linestyle="None",
           markersize=10, label="center"),
]
ax.legend(handles=legend_elems, fontsize=9, loc="upper right")

ax.set_xlim(0, img_case5.shape[1])
ax.set_ylim(img_case5.shape[0], 0)  # image coordinates (y inverted)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 20 -- Fourier-domain (Modregger) resolution + CNR + run summary
# =============================================================================
EXCLUDE_RADIUS_RANGES2 = ((53, 60), (95, 145), (200, 280), (430, 550))
EXCLUDE_ANGLE_RANGES2 = ((125, 150),)

_script_start = datetime.now()

# ---- headline result: CTFPurePhase, full distance combination ----
_summary_key = f"CTFPurePhase ({full_label})"
_img = results_to_plot[_summary_key]
q_wrapped = np.exp(1j * _img).astype(np.complex64)

# ---- image center (fallback if not defined earlier in this run) ----
if "xcenter" not in globals() or "ycenter" not in globals():
    xcenter, ycenter = (nx - 1) / 2.0, (ny - 1) / 2.0
    print(f"[Section 20] xcenter/ycenter not set -- using grid center: "
          f"({xcenter:.1f}, {ycenter:.1f})")

# ---- r_in / r_out (from sample_meta if available, else fallback) ----
if "sample_meta" in globals() and "r_out" in sample_meta and "r_in" in sample_meta:
    _r_out_px = sample_meta["r_out"] / psize
    _r_in_px = sample_meta["r_in"] / psize
else:
    _r_out_px = 0.45 * min(ny, nx)
    _r_in_px = 0.02 * min(ny, nx)
    print(f"[Section 20] sample_meta not found -- using fallback radii: "
          f"r_in={_r_in_px:.1f}px, r_out={_r_out_px:.1f}px "
          f"(VERIFY these match the actual star in your data)")

result = plot_abs_phase_ring_profiles(
    q_wrapped,
    xcenter=xcenter,
    ycenter=ycenter,
    radius_px=min(300, 0.9 * _r_out_px),
    voxelsize_um=psize * 1e6,
    normalize=True,
    phase_vmin=-0.8,
    phase_vmax=0.8,
)
print(f"key = {_summary_key}")
print(f"ndist = {ndist}")
print(f"alpha = {alpha_val}")
if "z1" in globals() and "z2" in globals():
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
else:
    print("z1/z2: not found in the global namespace")
print("  ----  ")
_script_end = datetime.now()
try:
    _elapsed = _script_end - _script_start
    _hours, _remainder = divmod(int(_elapsed.total_seconds()), 3600)
    _minutes = _remainder // 60
    print(f"Script ended {_script_end.strftime('%d/%m at %H:%M')}")
    print(f"Total run time {_hours:02d}:{_minutes:02d}")
except NameError:
    print(f"Script ended {_script_end.strftime('%d/%m at %H:%M')}")
    print("Total run time: N/A")

results = resolution_summary(
    q=q_wrapped,
    voxelsize_um=psize * 1e6,
    rec_params={
        "key": _summary_key,
        "ndist": ndist,
        "alpha": alpha_val,
        "z1": z1 if "z1" in globals() else None,
        "z2": z2 if "z2" in globals() else None,
    },
    xcenter=xcenter, ycenter=ycenter,
    r_min=max(50, 1.2 * _r_in_px),
    r_max=0.95 * _r_out_px,
    n_radii=500,
    n_arms=arm_pairs if "arm_pairs" in globals() else 36,
    exclude_radius_ranges=EXCLUDE_RADIUS_RANGES2 if "EXCLUDE_RADIUS_RANGES2" in globals() else (),
    exclude_angle_ranges=EXCLUDE_ANGLE_RANGES2 if "EXCLUDE_ANGLE_RANGES2" in globals() else (),
    norm_radius_range=(575, 950),
    modregger_roi=(slice(int(ycenter - 450), int(ycenter + 450)),
                   slice(int(xcenter - 500), int(xcenter + 500))),
    modregger_nblfac=2.0,
    show_plots=True,
)


# =============================================================================
# SECTION 21 -- CNR (Contrast-to-Noise Ratio) via matched signal/background ROI pairs
# =============================================================================
n_arms = 36
_period_deg = 360.0 / n_arms
_half_period_deg = 0.5 * _period_deg
_roi_radius_px = 900


def _roi_center(theta_deg, r_px):
    th = np.radians(theta_deg)
    return xcenter + r_px * np.cos(th), ycenter + r_px * np.sin(th)


def extract_rotated_roi(img, cx, cy, width, height, angle_deg, order=1):
    theta = np.deg2rad(angle_deg)
    u = np.arange(width) - (width - 1) / 2.0
    v = np.arange(height) - (height - 1) / 2.0
    uu, vv = np.meshgrid(u, v)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xx = cx + uu * cos_t - vv * sin_t
    yy = cy + uu * sin_t + vv * cos_t
    coords = np.vstack([yy.ravel(), xx.ravel()])
    patch = map_coordinates(img, coords, order=order, mode="reflect")
    return patch.reshape(height, width)


def compute_cnr(img, signal_roi, bg_roi):
    sig_patch = extract_rotated_roi(img, **signal_roi)
    bg_patch = extract_rotated_roi(img, **bg_roi)
    mu_sig, mu_bg = np.mean(sig_patch), np.mean(bg_patch)
    sigma_bg = np.std(bg_patch)
    cnr = np.abs(mu_sig - mu_bg) / (sigma_bg + 1e-12)
    return cnr, mu_sig, mu_bg, sigma_bg


def _roi_corners(cx, cy, width, height, angle_deg):
    theta = np.deg2rad(angle_deg)
    hw, hh = width / 2.0, height / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world = local @ R.T + np.array([cx, cy])
    return world


def plot_rois_multi(img, roi_pairs, vmin=None, vmax=None, title=""):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    for sig_roi, bg_roi in roi_pairs:
        sig_corners = _roi_corners(**sig_roi)
        bg_corners = _roi_corners(**bg_roi)
        ax.add_patch(patches.Polygon(sig_corners, closed=True, linewidth=1.2,
                                      edgecolor="lime", facecolor="none"))
        ax.add_patch(patches.Polygon(bg_corners, closed=True, linewidth=1.2,
                                      edgecolor="red", facecolor="none"))
    ax.set_title(f"{title} -- {len(roi_pairs)} ROI pairs")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


_exclude_angle_ranges = MTF_MODR_PARAMS["exclude_angle_ranges"]


def _in_excluded(angle_deg):
    a = angle_deg % 360
    return any(a0 <= a <= a1 for a0, a1 in _exclude_angle_ranges)


# --- Grid-search base_angle using the full-distance case (cleanest signal) ---
_probe_img = CTFrec_dict["Case 11: D1+D2+D3+D4+D5"]


def _mean_cnr_for_base_angle(_ba):
    # Filter FOR THIS `_ba`, checking both the signal AND background angle
    _angles = [a for a in range(0, 360, int(_period_deg))
               if not _in_excluded(_ba + a)
               and not _in_excluded(_ba + a + _half_period_deg)]
    _vals = []
    for ang in _angles:
        _st = _ba + ang
        _bt = _ba + ang + _half_period_deg
        _cxs, _cys = _roi_center(_st, _roi_radius_px)
        _cxb, _cyb = _roi_center(_bt, _roi_radius_px)
        _s = dict(cx=_cxs, cy=_cys, width=30, height=80, angle_deg=_st + 90)
        _b = dict(cx=_cxb, cy=_cyb, width=30, height=80, angle_deg=_bt + 90)
        _vals.append(compute_cnr(_probe_img, _s, _b)[0])
    return np.mean(_vals) if _vals else -np.inf


_candidate_angles = np.linspace(0, _period_deg, 101, endpoint=False)
_cnr_scores = [_mean_cnr_for_base_angle(a) for a in _candidate_angles]
base_angle = float(_candidate_angles[np.argmax(_cnr_scores)])
print(f"[CNR] Grid-search (on full case) base_angle = {base_angle:.3f} deg "
      f"(max mean CNR = {max(_cnr_scores):.2f})")

# --- final angles_list, filtered WITH the correct base_angle ---
angles_list = [a for a in range(0, 360, int(_period_deg))
               if not _in_excluded(base_angle + a)
               and not _in_excluded(base_angle + a + _half_period_deg)]

multi_roi_pairs = []
for ang in angles_list:
    _sig_theta = base_angle + ang
    _bg_theta = base_angle + ang + _half_period_deg
    _th_s, _th_b = np.radians(_sig_theta), np.radians(_bg_theta)
    s_roi = dict(cx=xcenter + _roi_radius_px * np.cos(_th_s),
                 cy=ycenter + _roi_radius_px * np.sin(_th_s),
                 width=30, height=80, angle_deg=_sig_theta + 90)
    b_roi = dict(cx=xcenter + _roi_radius_px * np.cos(_th_b),
                 cy=ycenter + _roi_radius_px * np.sin(_th_b),
                 width=30, height=80, angle_deg=_bg_theta + 90)
    multi_roi_pairs.append((s_roi, b_roi))

cnr_results_combos = {}
for label, img in CTFrec_dict.items():
    cnr_values = [compute_cnr(img, s_roi, b_roi)[0]
                  for s_roi, b_roi in multi_roi_pairs]
    cnr_results_combos[label] = float(np.mean(cnr_values))

# Plot the ROI overlay on the full-distance case
_plot_label = "Case 11: D1+D2+D3+D4+D5"
_plot_img = CTFrec_dict[_plot_label]
plot_rois_multi(_plot_img, multi_roi_pairs,
                 vmin=np.percentile(_plot_img, 2), vmax=np.percentile(_plot_img, 98),
                 title=_plot_label)


# =============================================================================
# SECTION 22 -- Combined resolution table: MTF10 + Modregger (phase/abs, x/y)
#               for the measured data (CTFPurePhase, all distance combinations)
# =============================================================================
_pat = re.compile(
    r"^\s*(phase|absorption)\s+([xy])\s+([\d.]+)\s+([\d.]+)", re.MULTILINE
)

# r_in/r_out fallback (no sample_meta available for real data)
if "_r_out_px" not in globals() or "_r_in_px" not in globals():
    _r_out_px = 0.45 * min(ny, nx)
    _r_in_px = 0.02 * min(ny, nx)
    print(f"[resolution_table] fallback radii: r_in={_r_in_px:.1f}px, r_out={_r_out_px:.1f}px")

rows = []
for key, img in CTFrec_dict.items():
    # ---- MTF10 (same pipeline as Section 18) ----
    profiles = extract_profiles(img)
    sig, noi = modulation_from_profiles(profiles, _exclude_angle_ranges)
    mtf = normalise_mtf(sig, noi)
    f_s, mtf_s, r_s = _sorted_by_freq(mtf)
    f10 = _sustained_mtf_crossing(f_s, mtf_s)
    mtf10_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan

    # ---- Modregger phase/abs x/y ----
    q = np.exp(1j * img).astype(np.complex64)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        resolution_summary(
            q=q,
            voxelsize_um=voxelsize * 1e6,
            rec_params={"key": key},
            xcenter=xcenter, ycenter=ycenter,
            r_min=max(50, 1.2 * _r_in_px),
            r_max=0.95 * _r_out_px,
            n_radii=500,
            n_arms=arm_pairs if "arm_pairs" in globals() else 36,
            exclude_radius_ranges=EXCLUDE_RADIUS_RANGES2 if "EXCLUDE_RADIUS_RANGES2" in globals() else (),
            exclude_angle_ranges=EXCLUDE_ANGLE_RANGES2 if "EXCLUDE_ANGLE_RANGES2" in globals() else (),
            norm_radius_range=(575, 950),
            modregger_roi=(slice(int(ycenter - 450), int(ycenter + 450)),
                           slice(int(xcenter - 500), int(xcenter + 500))),
            modregger_nblfac=2.0,
            show_plots=False,
        )
    matches = _pat.findall(buf.getvalue())[-4:]
    # keep BOTH the value (r) and its uncertainty (u)
    vals = {f"{ch}_{d}": (float(r), float(u)) for ch, d, r, u in matches}  # (mean_um, std_um)

    def _mean_nm(k):
        return vals[k][0] * 1000 if k in vals else np.nan

    def _std_nm(k):
        return vals[k][1] * 1000 if k in vals else np.nan

    rows.append({
        "case": key,
        "MTF10_nm": mtf10_nm,
        "phase_x_nm": _mean_nm("phase_x"),
        "phase_x_std_nm": _std_nm("phase_x"),
        "phase_y_nm": _mean_nm("phase_y"),
        "phase_y_std_nm": _std_nm("phase_y"),
        "abs_x_nm": _mean_nm("absorption_x"),
        "abs_x_std_nm": _std_nm("absorption_x"),
        "abs_y_nm": _mean_nm("absorption_y"),
        "abs_y_std_nm": _std_nm("absorption_y"),
    })

resolution_table_measured = pd.DataFrame(rows)
print(resolution_table_measured.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# =============================================================================
# SECTION 23 -- Build reconstructions for the remaining methods (homoCTF,
#               multiPaganin, single-distance Paganin, sglDstCTF) using
#               Ta as the representative material
# =============================================================================
if "psize" not in globals():
    psize = voxelsize

if "_r_out_px" not in globals() or "_r_in_px" not in globals():
    _r_out_px = 0.45 * min(ny, nx)
    _r_in_px = 0.02 * min(ny, nx)
    print(f"[prereq] fallback radii: r_in={_r_in_px:.1f}px, r_out={_r_out_px:.1f}px")

if "materials" not in globals():
    materials = {
        "Ta": {"delta": 7.272642e-06, "beta": 5.660273e-07},
        "Si3N4": {"delta": 1.745160e-06, "beta": 5.058573e-09},
    }

_delta_ta, _beta_ta = materials["Ta"]["delta"], materials["Ta"]["beta"]

_Rm_full_sim = np.ones((ny, nx, ndist), dtype=np.float32)
rads_intensity = rdata_ctf + 1.0

homoCTF_dict = {}
multiPaganin_dict = {}
for label, idx in combinations.items():
    idx = np.array(idx)
    rads_int = rads_intensity[idx]
    distances_case = distances_rec[idx]
    _Rm_case = _Rm_full_sim[:, :, idx]

    rec_h = homoCTF(rads_int, wlen, distances_case, _delta_ta, _beta_ta,
                     fx_grid, fy_grid, _Rm_case, alpha_val)
    homoCTF_dict[label] = rec_h.astype(np.float32)

    rec_p = multiPaganin(rads_int, wlen, distances_case, _delta_ta, _beta_ta,
                          fx_grid, fy_grid, _Rm_case, alpha_val)
    multiPaganin_dict[label] = rec_p.astype(np.float32)

Paganin_single_dict = {
    "D1": Paganin(rads_intensity[0], wlen, distances_rec[0], _delta_ta, _beta_ta,
                  fx_grid, fy_grid, _Rm_full_sim[:, :, 0]).astype(np.float32)
}

sglDstCTF_dict = {
    "D1": sglDstCTF(rads_intensity[0], wlen, distances_rec[0], _delta_ta, _beta_ta,
                     fx_grid, fy_grid, _Rm_full_sim[:, :, 0], alpha_val).astype(np.float32)
}

print("[prereq] homoCTF_dict, multiPaganin_dict, Paganin_single_dict, sglDstCTF_dict ready")


# =============================================================================
# SECTION 24 -- Combined resolution table across ALL methods and distance combos
# =============================================================================
_methods_progression = {
    "CTFPurePhase": CTFrec_dict,
    "CTF (Rm)": CTF_dict,
    "homoCTF": homoCTF_dict,
    "multiPaganin": multiPaganin_dict,
    "Paganin (D1)": Paganin_single_dict,
    "sglDstCTF (D1)": sglDstCTF_dict,
}

rows = []
for method_name, method_dict in _methods_progression.items():
    for label, img in method_dict.items():
        # ---- MTF10 (same pipeline as Section 18) ----
        profiles = extract_profiles(img)
        sig, noi = modulation_from_profiles(profiles, _exclude_angle_ranges)
        mtf = normalise_mtf(sig, noi)
        f_s, mtf_s, r_s = _sorted_by_freq(mtf)
        f10 = _sustained_mtf_crossing(f_s, mtf_s)
        mtf10_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan

        # ---- Modregger phase/abs x/y ----
        q = np.exp(1j * img).astype(np.complex64) if method_name == "CTFPurePhase" \
            else img.astype(np.complex64)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resolution_summary(
                q=q,
                voxelsize_um=psize * 1e6,
                rec_params={"key": f"{method_name} -- {label}"},
                xcenter=xcenter, ycenter=ycenter,
                r_min=max(50, 1.2 * _r_in_px),
                r_max=0.95 * _r_out_px,
                n_radii=500,
                n_arms=arm_pairs if "arm_pairs" in globals() else 36,
                exclude_radius_ranges=EXCLUDE_RADIUS_RANGES2 if "EXCLUDE_RADIUS_RANGES2" in globals() else (),
                exclude_angle_ranges=EXCLUDE_ANGLE_RANGES2 if "EXCLUDE_ANGLE_RANGES2" in globals() else (),
                norm_radius_range=(575, 950),
                modregger_roi=(slice(int(ycenter - 450), int(ycenter + 450)),
                               slice(int(xcenter - 500), int(xcenter + 500))),
                modregger_nblfac=2.0,
                show_plots=False,
            )

        matches = _pat.findall(buf.getvalue())[-4:]
        vals = {f"{ch}_{d}": float(r) for ch, d, r, u in matches}  # um

        rows.append({
            "method": method_name,
            "case": label,
            "MTF10_nm": mtf10_nm,
            "phase_x_nm": vals.get("phase_x", np.nan) * 1000,
            "phase_y_nm": vals.get("phase_y", np.nan) * 1000,
            "abs_x_nm": vals.get("absorption_x", np.nan) * 1000,
            "abs_y_nm": vals.get("absorption_y", np.nan) * 1000,
        })

resolution_table_all_methods_progression = pd.DataFrame(rows)
print(resolution_table_all_methods_progression.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# =============================================================================
# SECTION 25 -- Grid plot of every reconstructed image (all methods x all cases)
# =============================================================================
_all_imgs = []
for method_name, method_dict in _methods_progression.items():
    for label, img in method_dict.items():
        _all_imgs.append((f"{method_name}\n{label}", img))

n_imgs = len(_all_imgs)
n_cols = 4
n_rows = int(np.ceil(n_imgs / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.atleast_2d(axes).reshape(-1)  # flatten, works even if n_rows == 1
for ax, (title, img) in zip(axes, _all_imgs):
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax in axes[n_imgs:]:
    ax.axis('off')
plt.tight_layout()
plt.show()
