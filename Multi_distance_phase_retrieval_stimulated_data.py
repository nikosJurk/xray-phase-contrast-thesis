
# =============================================================================
# SECTION 1 -- Imports and CONFIG
# =============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from scipy import ndimage
from scipy.ndimage import map_coordinates
from types import SimpleNamespace
import h5py

from PBI_phase_retrieval_functions import *
from usefull_functions import (
    circular_profile_angle, modulation_lstsq, modregger_resolution,
    compute_mtf_siemens_clean_new, resolution_summary,
)

try:
    import pandas as pd
except Exception as _e:
    raise RuntimeError(f"pandas is required for the summary tables ({_e})")

# Optional GPU stack -- this script itself runs fully on CPU, but some
# environments still expose a GPU engine we might reuse later.
try:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    import warnings
    warnings.filterwarnings("ignore", message=".*peer.*")
    from utils_sim import *
    _HAVE_GPU = True
except Exception as _e:
    _HAVE_GPU = False
    print(f"[imports] GPU/engine stack not available here ({_e}).")
    print("          Not a problem: this script is CPU-only anyway "
          "(no CA, no probe, no Rec engine).")

# Xraylib, used as a fallback source of optical constants (delta/beta)
try:
    import xraylib as _xrl
    _HAVE_XRAYLIB = True
except Exception:
    _HAVE_XRAYLIB = False

# ---- CONFIG: paths -- edit these (or set the environment variables) before
# running. The original script used DTU-cluster paths containing a personal
# student ID; those have been replaced with generic placeholders. ----

# Directory containing the NIST optical-constants pipeline (preferred
# source of delta/beta). If unavailable, xraylib is used as a fallback.
OC_SOURCE_DIR = os.environ.get(
    "MCTF_SIM_OC_SOURCE_DIR", "./oc_source"
)
OC_NIST_DIR = os.path.join(OC_SOURCE_DIR, "NIST/")

# Output directory for saved figures/arrays from this simulation script
OUT_DIR = os.environ.get("MCTF_SIM_OUT_DIR", "./outputs_simulation")
os.makedirs(OUT_DIR, exist_ok=True)

# Directory containing the REAL measured dark/flat HDF5 scans (used only if
# USE_REAL_FLAT_DARK = True in SECTION 3b, to build a realistic flat/dark
# model instead of an idealized flat=1 one)
REAL_DATA_DIR = os.environ.get("MCTF_RAW_DATA_DIR", "./data/raw_scans")
REAL_DATASET_PATH = "entry/instrument/orca/data"

_OC = None
try:
    if OC_SOURCE_DIR not in sys.path:
        sys.path.insert(0, OC_SOURCE_DIR)
    from optical_constants import oc as _OC
    print(f"[oc] NIST optical-constants pipeline available ({OC_SOURCE_DIR})")
except Exception as _e:
    print(f"[oc] NIST pipeline not importable ({_e}); will use xraylib if present.")


def get_delta_beta(material, density, E_keV, elements=None, atoms=None):
    """
    Return (delta, beta) with n = 1 - delta + i*beta.
    Priority: NIST oc() -> xraylib -> error.
    """
    if material == "vacuum":
        return 0.0, 0.0
    if _OC is not None and elements is not None and atoms is not None:
        lam_nm = np.atleast_1d(1.239841984 / E_keV)
        n = _OC(lam_nm, density, atoms, elements, OC_NIST_DIR)[0]
        return float(1.0 - n.real), float(n.imag)
    if _HAVE_XRAYLIB:
        d = 1.0 - _xrl.Refractive_Index_Re(material, E_keV, density)
        b = _xrl.Refractive_Index_Im(material, E_keV, density)
        return float(d), float(b)
    raise RuntimeError("No optical-constants source available (NIST or xraylib).")


def _sustained_mtf_crossing(freqs, mtf, threshold=0.1, n_points=20):
    """Find the first MTF point below `threshold` that stays below it for
    the following `n_points` points (a sustained crossing, more robust than
    a naive "first point below threshold" finder, which can trigger on a
    noise dip near the low-frequency normalization band).

    NOTE: the original script defined this function TWICE, with two
    incompatible signatures/algorithms (an earlier `level`/`n_sustain`
    version, and this `threshold`/`n_points` version). Since Python keeps
    only the LAST definition, the second version is what actually ran at
    runtime throughout the script; the first has been removed here to avoid
    the duplicate-definition confusion.
    """
    freqs = np.asarray(freqs, dtype=float)
    mtf = np.asarray(mtf, dtype=float)

    valid = np.isfinite(freqs) & np.isfinite(mtf)
    freqs = freqs[valid]
    mtf = mtf[valid]

    for i in range(len(mtf) - n_points):
        window = mtf[i:i + n_points + 1]
        if np.all(window < threshold):
            return freqs[i]
    return np.nan


# ---- Figure-sizing convention, used by EVERY plot in this script ----
FIGSIZE_SINGLE = (7, 7)
PANEL_SIZE = (10, 10)


# =============================================================================
# SECTION 2 -- GEOMETRY & args (5 distances, NO coded aperture, NO probe)
# =============================================================================
energy = 19.55                          # [keV] photon energy
PLANCK_CONSTANT = 4.135667696e-18       # keV*s
SPEED_OF_LIGHT = 299792458              # m/s
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy   # [m]
k_wave = 2 * np.pi / wavelength                          # [1/m]

detector_pixelsize = 0.55e-6            # [m] (ORCA 10x -> 0.55 um effective)

# Measured geometry -- z1 (focus->sample), z2 (sample->detector)
z1 = np.array([119.69, 121.69, 126.69, 132.69, 138.69]) * 1e-3
z2 = np.array([1437.0, 1435.0, 1430.0, 1424.0, 1418.0]) * 1e-3
magnifications = (z1 + z2) / z1
ndist = len(z1)

focusToDetectorDistance = z1 + z2
norm_magnifications = magnifications / magnifications[0]

# Effective (Fresnel-scaled) propagation distance on the COMMON grid of D1.
# Fresnel scaling theorem: D/dx^2 is the invariant, and the natural pixel of
# distance k is det_px/M_k = voxelsize * M_0/M_k, hence the norm_M^2 factor.
distances = (z1 * z2) / (z1 + z2) * norm_magnifications**2
voxelsize = detector_pixelsize / magnifications[0]          # common-grid pixel

args = SimpleNamespace()
args.ndist = ndist
args.energy = energy
args.wavelength = wavelength
args.detector_pixelsize = detector_pixelsize
args.voxelsize = voxelsize
args.z1 = z1
args.z2 = z2
args.distances = distances
args.magnifications = magnifications
args.has_ca = False
args.has_probe = False

print("=" * 62)
print("GEOMETRY (multi-distance, no CA, no probe)")
print("=" * 62)
print(f"  Energy               : {energy} keV  (lambda = {wavelength*1e12:.4f} pm)")
print(f"  z1 [mm]              : {z1*1e3}")
print(f"  z2 [mm]              : {z2*1e3}")
print(f"  focusToDetectorDist  : {focusToDetectorDistance*1e3}")
print(f"  Magnification M      : {magnifications}")
print(f"  norm. magnification  : {norm_magnifications}")
print(f"  effective distances  : {distances}")
print(f"  reference voxelsize  : {voxelsize*1e9:.2f} nm")
print("=" * 62)


# =============================================================================
# SECTION 2b -- GEOMETRIC CALIBRATION MISMATCH
# =============================================================================
GEOM_OFFSET_MM = 0.00       # non-zero -> forward and reconstruction disagree on z1
REC_USES_RAW_DISTANCES = False

_z1_offset_m = GEOM_OFFSET_MM * 1e-3
z1_true = z1 + _z1_offset_m
magnifications_true = (z1_true + z2) / z1_true
norm_magnifications_true = magnifications_true / magnifications_true[0]
distances_true = (z1_true * z2) / (z1_true + z2) * norm_magnifications_true**2

# distances used by the FORWARD model (truth) and by the RECONSTRUCTION
distances_fwd = distances_true
distances_rec = (distances / norm_magnifications**2) if REC_USES_RAW_DISTANCES else distances

print(f"  Geometric calibration error : fixed offset = {GEOM_OFFSET_MM:+.3f} mm "
      f"(SHARED, on the absolute z1 reference)")
print(f"  forward  distances [mm]     : {distances_fwd*1e3}")
print(f"  rec.     distances [mm]     : {distances_rec*1e3}")

# When GEOM_OFFSET_MM = 0, the reconstruction "knows" the true distances
# exactly -- a self-consistent (circular) test, giving best-case resolution.
_self_consistent = np.allclose(distances_fwd, distances_rec)
if _self_consistent:
    print("  [NOTE] forward == reconstruction distances: this is a SELF-CONSISTENT")
    print("         (circular) test. The resolution numbers below are best-case;")
    print("         set GEOM_OFFSET_MM != 0 for a realistic mismatch.")
print("=" * 62)

print("distances_rec (used in reconstruction) [mm]:", distances_rec * 1e3)
print("distances_fwd (used in forward propagation) [mm]:", distances_fwd * 1e3)
print("Ratio fwd/rec:", distances_fwd / distances_rec)


# =============================================================================
# SECTION 3 -- SAMPLE: NTT-AT XRESO-50HC datasheet model
#              Ta absorber (500 nm) on a SiC membrane (200 nm)
# =============================================================================
SAMPLE_MATERIAL_LIBRARY = {
    "Au":     dict(density=19.30, elements=["Au"],          atoms=[1]),
    "Si":     dict(density=2.33,  elements=["Si"],          atoms=[1]),
    "Ta":     dict(density=16.65, elements=["Ta"],          atoms=[1]),
    "SiC":    dict(density=3.21,  elements=["Si", "C"],     atoms=[1, 1]),
    "water":  dict(density=1.00,  elements=["H", "O"],      atoms=[2, 1]),
    "fat":    dict(density=0.92,  elements=["C", "H", "O"], atoms=[57, 104, 6]),
    "vacuum": dict(density=0.0,   elements=None,            atoms=None),
}


def material_transmission(name, thickness_m, E_keV, k):
    """Complex transmission (amplitude * phase) of a slab of given material/thickness."""
    if name == "vacuum":
        return np.complex64(1.0), 0.0, 0.0
    spec = SAMPLE_MATERIAL_LIBRARY[name]
    delta, beta = get_delta_beta(name, spec["density"], E_keV,
                                 elements=spec.get("elements"),
                                 atoms=spec.get("atoms"))
    amp = np.exp(-k * beta * thickness_m)
    ph = -k * delta * thickness_m
    return (amp * np.exp(1j * ph)).astype(np.complex64), delta, beta


def make_siemens_star_physical(
    ny, nx, voxelsize_m, arm_pairs,
    feat_min_m, feat_max_m,
    absorber_material, absorber_thickness_m,
    membrane_material, membrane_thickness_m,
    k, E_keV,
    supersample=3,
    exclude_radius_ranges=(),
    exclude_angle_ranges=(),
):
    """Build a physically-modeled Siemens star: a SiC membrane everywhere,
    with Ta absorber wedges alternating with open (membrane-only) wedges."""
    T_membrane, d_mem, b_mem = material_transmission(
        membrane_material, membrane_thickness_m, E_keV, k)
    T_ta, d_ta, b_ta = material_transmission(
        absorber_material, absorber_thickness_m, E_keV, k)

    dphi = np.pi / arm_pairs
    r_in = feat_min_m / dphi
    r_out = feat_max_m / dphi

    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    # Anti-alias the wedge pattern via sub-pixel supersampling
    offs = (np.arange(supersample, dtype=np.float32) + 0.5) / supersample - 0.5
    frac = np.zeros((ny, nx), dtype=np.float32)
    for dy in offs:
        for dx in offs:
            x = (xx - cx + dx) * np.float32(voxelsize_m)
            y = (yy - cy + dy) * np.float32(voxelsize_m)
            r = np.hypot(x, y)
            th = np.mod(np.arctan2(y, x), 2 * np.pi)
            wedge_idx = np.floor(th / dphi).astype(np.int32)
            frac += ((r >= r_in) & (r <= r_out) & (wedge_idx % 2 == 0))
    frac /= supersample * supersample

    # Zero out any defect/exclusion zones, matching the real target
    r_px = np.hypot(xx - cx, yy - cy)
    theta_deg = np.rad2deg(np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi))
    exclude_mask = np.zeros((ny, nx), dtype=bool)
    for r0, r1 in exclude_radius_ranges:
        exclude_mask |= (r_px >= r0) & (r_px <= r1)
    for a0, a1 in exclude_angle_ranges:
        if a0 > a1:
            exclude_mask |= (theta_deg >= a0) | (theta_deg <= a1)
        else:
            exclude_mask |= (theta_deg >= a0) & (theta_deg <= a1)
    frac[exclude_mask] = 0.0

    # Complex transmission function: membrane everywhere, Ta only on wedges
    psi = (T_membrane * ((1.0 - frac) + frac * T_ta)).astype(np.complex64)

    meta = dict(r_in=r_in, r_out=r_out, dphi=dphi, arm_pairs=arm_pairs,
                membrane=(membrane_material, d_mem, b_mem),
                ta=(absorber_material, d_ta, b_ta),
                T_membrane=T_membrane, T_ta=T_ta,
                ta_fraction=frac, exclude_mask=exclude_mask,
                supersample=supersample)
    return psi, meta


# ---- sample parameters ----
ny = 2592                      # [px] matches the real data.shape[1]
nx = 3712                      # [px] matches the real data.shape[2]
feat_min = 50e-9               # [m] innermost feature (50 nm, datasheet)
arm_pairs = 36                 # real Siemens-star target: 36 arm pairs
SUPERSAMPLE = 3

TA_THICKNESS = 500e-9         # [m] absorber pattern
SIC_THICKNESS = 200e-9        # [m] membrane

SIM_STAR_OUTER_RADIUS_UM = 46.8
dphi = np.pi / arm_pairs
feat_max = SIM_STAR_OUTER_RADIUS_UM * 1e-6 * dphi

# Use the same defect exclusion zones as the real measured target
USE_REAL_DEFECT_EXCLUSIONS = True
REAL_EXCLUDE_RADIUS_RANGES = ((55, 60), (125, 140), (250, 270), (505, 550))
REAL_EXCLUDE_ANGLE_RANGES = ((126, 146),)
if USE_REAL_DEFECT_EXCLUSIONS:
    EXCLUDE_RADIUS_RANGES = REAL_EXCLUDE_RADIUS_RANGES
    EXCLUDE_ANGLE_RANGES = REAL_EXCLUDE_ANGLE_RANGES
else:
    EXCLUDE_RADIUS_RANGES = ()
    EXCLUDE_ANGLE_RANGES = ()

# NOTE: exclusions are passed as the REAL target's zones here -- the
# simulated object itself gets these same defect zones "cut out" of its
# absorber pattern, so it matches the physical target used in the real
# measurements. EXCLUDE_RADIUS_RANGES / EXCLUDE_ANGLE_RANGES (the "no-2"
# names) are the canonical values used for the sample geometry throughout
# this script; EXCLUDE_RADIUS_RANGES2 (used later, in Sections 7/8) is a
# SEPARATE, analysis-only set of exclusion ranges for the MTF/CNR masks --
# see the NOTE in Section 7 for why these are kept distinct.
psi, sample_meta = make_siemens_star_physical(
    ny, nx, voxelsize, arm_pairs, feat_min, feat_max,
    absorber_material="Ta", absorber_thickness_m=TA_THICKNESS,
    membrane_material="SiC", membrane_thickness_m=SIC_THICKNESS,
    k=k_wave, E_keV=energy, supersample=SUPERSAMPLE,
    exclude_radius_ranges=REAL_EXCLUDE_RADIUS_RANGES, exclude_angle_ranges=REAL_EXCLUDE_ANGLE_RANGES,
)

args.ny, args.nx = ny, nx
mem_name, mem_d, mem_b = sample_meta["membrane"]
ta_name, ta_d, ta_b = sample_meta["ta"]
phi_ta = k_wave * ta_d * TA_THICKNESS
mu_t_ta = 2 * k_wave * ta_b * TA_THICKNESS

print("=" * 62)
print("SAMPLE (XRESO-50HC datasheet model)")
print("=" * 62)
print(f"  Array/pixel : {ny} x {nx}, {voxelsize*1e9:.2f} nm/px, "
      f"supersample = {SUPERSAMPLE}x{SUPERSAMPLE}")
print(f"  Arm pairs   : {arm_pairs}")
print(f"  Outer radius: {sample_meta['r_out']*1e6:.3f} um "
      f"({sample_meta['r_out']/voxelsize:.1f} px)")
print(f"  Membrane ({mem_name}, {SIC_THICKNESS*1e9:.0f} nm, everywhere): "
      f"delta = {mem_d:.3e}, beta = {mem_b:.3e}")
print(f"  Absorber ({ta_name}, {TA_THICKNESS*1e9:.0f} nm, Ta arms only): "
      f"delta = {ta_d:.3e}, beta = {ta_b:.3e}")
print(f"  Ta phase shift          : {phi_ta:.4f} rad "
      f"({'weak-object OK' if phi_ta < 0.3 else 'MARGINAL for the weak-object/CTF assumption'})")
print(f"  Ta absorption (1 - T^2) : {1 - np.exp(-mu_t_ta):.4f}")
print(f"  Exclusions used         : radius {EXCLUDE_RADIUS_RANGES}, "
      f"angle {EXCLUDE_ANGLE_RANGES}")

_nyq_feature_m = 2 * voxelsize
print(f"  Nyquist feature size    : {_nyq_feature_m*1e9:.1f} nm (= 2 x voxelsize)")
if feat_min < _nyq_feature_m:
    print(f"  [NOTE] feat_min = {feat_min*1e9:.1f} nm < {_nyq_feature_m*1e9:.1f} nm: "
          f"the innermost arms are below Nyquist. They are now area-averaged "
          f"(not aliased), but they carry no recoverable information.")
print("=" * 62)


# =============================================================================
# SECTION 3b -- REALISTIC FLAT/DARK FIELD, built from the REAL measured
# dark (scan-0076) and flat (scan-0077) scans (REALFLAT).
# =============================================================================
USE_REAL_FLAT_DARK = False   # False -> falls back to the idealized flat=1 model

real_gain_map_sim = None
real_dark_std_sim = None


def _crop_or_pad_to(img, target_shape):
    """Center-crop or pad an image to a target shape, filling any pad
    with the image's own median value."""
    ty, tx = target_shape
    sy, sx = img.shape
    out = np.full(target_shape, np.nanmedian(img), dtype=img.dtype)
    cy0, cx0 = max(0, (sy - ty)//2), max(0, (sx - tx)//2)
    oy0, ox0 = max(0, (ty - sy)//2), max(0, (tx - sx)//2)
    h, w = min(ty, sy), min(tx, sx)
    out[oy0:oy0+h, ox0:ox0+w] = img[cy0:cy0+h, cx0:cx0+w]
    return out


if USE_REAL_FLAT_DARK:
    try:
        with h5py.File(os.path.join(REAL_DATA_DIR, "scan-0076_orca.h5"), "r") as f:
            real_dark_stack = f[REAL_DATASET_PATH][:].astype("float32")
        with h5py.File(os.path.join(REAL_DATA_DIR, "scan-0077_orca.h5"), "r") as f:
            real_flat_stack = f[REAL_DATASET_PATH][:].astype("float32")

        real_dark_mean = real_dark_stack.mean(axis=0)
        real_dark_std_temporal = real_dark_stack.std(axis=0)
        real_flat_mean = real_flat_stack.mean(axis=0)

        # Pixel-to-pixel gain non-uniformity, normalized to a median of 1
        real_gain_map = real_flat_mean - real_dark_mean
        real_gain_map = real_gain_map / np.nanmedian(real_gain_map)

        real_gain_map_sim = _crop_or_pad_to(real_gain_map, (ny, nx))
        real_dark_std_sim = _crop_or_pad_to(real_dark_std_temporal, (ny, nx))

        print("=" * 62)
        print("REALISTIC FLAT/DARK (from scan-0076 dark, scan-0077 flat)")
        print("=" * 62)
        print(f"  Real dark stack: {real_dark_stack.shape}, "
              f"real flat stack: {real_flat_stack.shape}")
        print(f"  Real gain map  : mean={real_gain_map.mean():.4f}, "
              f"std={real_gain_map.std():.4f} "
              f"({100*real_gain_map.std():.2f}% pixel-to-pixel non-uniformity)")
        print(f"  Real dark offset     : mean={real_dark_mean.mean():.2f} ADU")
        print(f"  Real read/dark noise : mean std={real_dark_std_temporal.mean():.3f} ADU")
        print(f"  Cropped/padded to simulation grid: {real_gain_map_sim.shape}")
        print("=" * 62)
    except Exception as _e:
        print(f"[REALFLAT] Could not load real dark/flat scans ({_e}); "
              f"falling back to the idealized flat=1 model.")
        USE_REAL_FLAT_DARK = False
        real_gain_map_sim = None
        real_dark_std_sim = None


# =============================================================================
# SECTION 4 -- FORWARD SIMULATION (Fresnel propagation + partial coherence)
# =============================================================================
PAD_PX = None            # None -> auto (max(0, (lambda*D/dx^2 - N)/2) per axis)
PAD_PX_MAX = 1024        # safety cap so a huge D cannot blow up memory
BAND_LIMIT_ASM = True


def fresnel_propagate(psi, wavelength, distance, voxelsize,
                      pad_px=PAD_PX, band_limit=BAND_LIMIT_ASM, verbose=True):
    """Propagate a complex field psi by `distance` using the angular-spectrum
    method, with automatic edge-padding and optional band-limiting to
    suppress aliasing at large propagation distances."""
    ny0, nx0 = psi.shape
    support_px = wavelength * abs(distance) / voxelsize**2
    if pad_px is None:
        pad_needed = int(np.ceil(max(0.0, (support_px - min(ny0, nx0)) / 2.0)))
        pad_px = min(pad_needed, PAD_PX_MAX)
        if verbose and pad_needed > PAD_PX_MAX:
            print(f"  [WARNING] required padding {pad_needed} px > PAD_PX_MAX "
                  f"({PAD_PX_MAX} px): the propagation support "
                  f"({support_px:.0f} px) does not fit -> expect wrap-around "
                  f"artefacts. Increase PAD_PX_MAX.")
    if pad_px:
        psi = np.pad(psi, ((pad_px, pad_px), (pad_px, pad_px)), mode="edge")
    ny, nx = psi.shape

    fx = np.fft.fftfreq(nx, d=voxelsize)
    fy = np.fft.fftfreq(ny, d=voxelsize)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    evanescent = arg <= 0
    arg = np.clip(arg, 0, None)

    # Angular-spectrum transfer function
    H = np.exp(1j * k * distance * np.sqrt(arg))
    H[evanescent] = 0.0

    if band_limit:
        # Antialiasing frequency limit (Matsushima band-limited ASM)
        dfx = 1.0 / (nx * voxelsize)
        dfy = 1.0 / (ny * voxelsize)
        fx_lim = 1.0 / (wavelength * np.sqrt((2.0 * dfx * distance)**2 + 1.0))
        fy_lim = 1.0 / (wavelength * np.sqrt((2.0 * dfy * distance)**2 + 1.0))
        H[(np.abs(FX) > fx_lim) | (np.abs(FY) > fy_lim)] = 0.0

    field = np.fft.ifft2(np.fft.fft2(psi) * H)
    if pad_px:
        field = field[pad_px:pad_px + ny0, pad_px:pad_px + nx0]
    return field


ADD_SOURCE_BLUR = True
ADD_DETECTOR_PSF = True    # REALFLAT: restored to True (was False)
ADD_NOISE = True

# =============================================================================
# PHOTON FLUX -- derived explicitly from the measured beam flux, the
# exposure time, and the MLL aperture geometry, instead of a hand-picked
# PHOTONS_PER_PIXEL constant.
#
# The opening angle set by the MLL aperture and focal length determines the
# illuminated field of view (FOV) at the sample plane; the illuminated real
# detector array is assumed to receive the full measured flux over the
# exposure time, spread over the ny x nx real detector pixels.
# =============================================================================
MLL_APERTURE_UM = 105.0                 # [um] MLL aperture (full diameter)
MLL_FOCAL_LENGTH_MM_PER_KEV = 3.75      # [mm/keV] MLL focal-length scaling
FLUX_PHOTONS_PER_S = 2.6e12             # [photons/s] measured flux on the detector
EXPOSURE_TIME_S = 0.2                   # [s] exposure time per frame

mll_focal_length_mm = MLL_FOCAL_LENGTH_MM_PER_KEV * energy
mll_opening_angle_rad = (MLL_APERTURE_UM * 1e-6) / (mll_focal_length_mm * 1e-3)

# FOV (diameter) at the sample plane, using the reference (D1) source-to-
# sample distance z1[0]; a = half-FOV = the characteristic beam radius there.
fov_sample_m = mll_opening_angle_rad * z1[0]
a_half_fov_m = fov_sample_m / 2.0

total_photons = FLUX_PHOTONS_PER_S * EXPOSURE_TIME_S
n_real_pixels_illuminated = ny * nx   # full real detector array, ny x nx
PHOTONS_PER_PIXEL = total_photons / n_real_pixels_illuminated   # per REAL detector pixel

print("=" * 62)
print("PHOTON FLUX -- derived from measured flux, exposure time, and MLL geometry")
print("=" * 62)
print(f"  MLL aperture             : {MLL_APERTURE_UM:.1f} um")
print(f"  MLL focal length         : {mll_focal_length_mm:.4f} mm "
      f"({MLL_FOCAL_LENGTH_MM_PER_KEV} mm/keV x {energy} keV)")
print(f"  MLL opening angle        : {mll_opening_angle_rad:.6e} rad")
print(f"  FOV at sample (D1, z1[0]): {fov_sample_m*1e6:.2f} um "
      f"(a = half-FOV = {a_half_fov_m*1e6:.2f} um)")
print(f"  Measured flux            : {FLUX_PHOTONS_PER_S:.3e} photons/s")
print(f"  Exposure time            : {EXPOSURE_TIME_S} s")
print(f"  Total photons/frame      : {total_photons:.4e}")
print(f"  Illuminated real pixels  : {n_real_pixels_illuminated} ({ny} x {nx})")
print(f"  --> PHOTONS_PER_PIXEL    : {PHOTONS_PER_PIXEL:.1f}  "
      f"(per REAL detector pixel)")
print("=" * 62)

DETECTOR_PSF_SIGMA_REAL_PX = 2.20   # in REAL detector pixels
NOISE_SEED = 0
# SIMULATE_FLAT_FIELD is the OLD idealized (Poisson-only, no real gain
# structure) path; it only runs if USE_REAL_FLAT_DARK is False.
SIMULATE_FLAT_FIELD = False
N_FLAT = 20
rng = np.random.default_rng(NOISE_SEED)

# ---- effective (demagnified) source = MLL focal spot ----
SOURCE_OBJECT_DISTANCE_M = 42.2
SOURCE_IMAGE_DISTANCE_M = 0.07331
SOURCE_MAGNIFICATION = SOURCE_IMAGE_DISTANCE_M / (SOURCE_OBJECT_DISTANCE_M + SOURCE_IMAGE_DISTANCE_M)

SOURCE_SIZE_ORIGINAL_H_UM = 122.0
SOURCE_SIZE_ORIGINAL_V_UM = 28.0
SOURCE_SIZE_CORRECTION_FACTOR = 4.8  # start from pure geometry, re-fit after
                                       # correcting PHOTONS_PER_PIXEL above

SOURCE_SIZE_H = (SOURCE_SIZE_ORIGINAL_H_UM * 1e-6 * SOURCE_MAGNIFICATION
                 * SOURCE_SIZE_CORRECTION_FACTOR)
SOURCE_SIZE_V = (SOURCE_SIZE_ORIGINAL_V_UM * 1e-6 * SOURCE_MAGNIFICATION
                 * SOURCE_SIZE_CORRECTION_FACTOR)

args.source_size_h = SOURCE_SIZE_H
args.source_size_v = SOURCE_SIZE_V
args.photons_per_pixel = PHOTONS_PER_PIXEL

print("=" * 62)
print("PARTIAL COHERENCE -- anisotropic source-blur forward model")
print("=" * 62)
print(f"  Source demagnification  : {SOURCE_MAGNIFICATION:.4e}")
print(f"  Correction factor       : {SOURCE_SIZE_CORRECTION_FACTOR:.2f} "
      f"({(SOURCE_SIZE_CORRECTION_FACTOR-1)*100:+.0f}% vs pure geometry)")
print(f"  Effective focal spot H  : {SOURCE_SIZE_H*1e9:.2f} nm FWHM")
print(f"  Effective focal spot V  : {SOURCE_SIZE_V*1e9:.2f} nm FWHM")
print(f"  Flat/dark model         : "
      f"{'REALISTIC (measured gain map + dark noise)' if USE_REAL_FLAT_DARK else 'idealized (flat=1)'}")

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))     # = 1/2.3548

data = np.zeros((ndist, ny, nx), dtype=np.float32)

for kdist in range(ndist):
    # 1) Fresnel-propagate the sample's exit wave to the detector
    field_det = fresnel_propagate(psi, wavelength, distances_fwd[kdist], voxelsize)
    intensity = np.abs(field_det)**2
    flat = np.ones_like(intensity)

    # 2) Partial coherence: blur from the finite source size, referenced to
    #    the object plane (penumbra = source_size * z2/z1, divided by M)
    blur_object_h = blur_object_v = 0.0
    sigma_px_h = sigma_px_v = 0.0
    if ADD_SOURCE_BLUR:
        blur_object_h = SOURCE_SIZE_H * z2[kdist] / (z1_true[kdist] + z2[kdist])
        blur_object_v = SOURCE_SIZE_V * z2[kdist] / (z1_true[kdist] + z2[kdist])
        sigma_px_h = (blur_object_h * FWHM_TO_SIGMA) / voxelsize
        sigma_px_v = (blur_object_v * FWHM_TO_SIGMA) / voxelsize
        if sigma_px_h > 0 or sigma_px_v > 0:
            intensity = ndimage.gaussian_filter(intensity,
                                                sigma=(sigma_px_v, sigma_px_h))

    # 3) Detector PSF blur, scaled to this distance's effective pixel size
    zoom_factor = magnifications[kdist] / magnifications[0]

    sigma_px_det = 0.0
    if ADD_DETECTOR_PSF:
        sigma_px_det = DETECTOR_PSF_SIGMA_REAL_PX / zoom_factor
        intensity = ndimage.gaussian_filter(intensity, sigma=sigma_px_det)
        # a flat field is uniform, so the PSF leaves it unchanged

    photons_eff = PHOTONS_PER_PIXEL * zoom_factor**2   # per common-grid px

    if USE_REAL_FLAT_DARK and real_gain_map_sim is not None:
        # 4a) REALFLAT path: raw/flat/dark capture built from the REAL
        # measured gain map and REAL measured dark/read noise, mirroring
        # exactly what the measurement script's flat-field correction does:
        #   data = (raw - dark_avg) / (flat_avg - dark_avg)
        signal_counts = intensity * real_gain_map_sim * photons_eff
        raw_counts = (rng.poisson(np.clip(signal_counts, 0, None)).astype(np.float32)
                     + rng.normal(0, real_dark_std_sim))

        flat_signal = real_gain_map_sim * photons_eff
        flat_avg = (rng.poisson(np.clip(flat_signal, 0, None) * N_FLAT)
                   .astype(np.float32) / N_FLAT
                   + rng.normal(0, real_dark_std_sim / np.sqrt(N_FLAT)))

        dark_avg = rng.normal(0, real_dark_std_sim / np.sqrt(N_FLAT))

        denom = flat_avg - dark_avg
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        data[kdist] = ((raw_counts - dark_avg) / denom).astype(np.float32)

    else:
        # 4b) Old idealized path: flat=1, optional pure-Poisson synthetic
        # flat via SIMULATE_FLAT_FIELD (no real gain structure)
        if ADD_NOISE:
            intensity_noisy = (rng.poisson(np.clip(intensity, 0, None) * photons_eff)
                               .astype(np.float32) / photons_eff)
            if SIMULATE_FLAT_FIELD:
                flat = (rng.poisson(photons_eff * N_FLAT, size=intensity.shape)
                        .astype(np.float32) / (photons_eff * N_FLAT))
        else:
            intensity_noisy = intensity
        data[kdist] = (intensity_noisy / flat).astype(np.float32)

    print(f"  D{kdist+1}: D_eff = {distances_fwd[kdist]*1e3:8.4f} mm, "
          f"M = {magnifications[kdist]:.4f}, zoom = {zoom_factor:.4f}, "
          f"blur H = {blur_object_h*1e9:6.1f} nm ({sigma_px_h:.2f} px), "
          f"V = {blur_object_v*1e9:5.1f} nm ({sigma_px_v:.2f} px), "
          f"det PSF = {sigma_px_det:.2f} px "
          f"(real-px equiv {sigma_px_det*zoom_factor:.1f})")
print("=" * 62)


# =============================================================================
# SECTION 5 -- CTF TRANSFER-FUNCTION PLOT (combined, all 5 distances)
# =============================================================================
fx_1d_pos = np.fft.fftfreq(nx, d=voxelsize)[:nx // 2]

taylorExp = np.array([np.sin(np.pi * wavelength * distances_rec[k] * fx_1d_pos**2)**2
                      for k in range(ndist)])
tf_sum = taylorExp.sum(axis=0)

fmin = 1 / np.sqrt(2 * wavelength * np.max(distances_rec))
fmax_voxel = 1 / (2 * voxelsize)          # the only real sampling limit
print(f"fmin (first CTF maximum)      = {fmin:.6e} m^-1 = {fmin/1e6:.4f} cyc/um")
print(f"fmax (voxel Nyquist)          = {fmax_voxel:.6e} m^-1 = {fmax_voxel/1e6:.4f} cyc/um")
for k in range(ndist):
    f0 = np.sqrt(1.0 / (wavelength * distances_rec[k]))
    print(f"  D{k+1}: first CTF zero at {f0/1e6:.4f} cyc/um")

plt.figure(figsize=(7, 5))
for k in range(ndist):
    plt.plot(fx_1d_pos, taylorExp[k], "--", linewidth=1, label=rf"$D_{k+1}$")
plt.plot(fx_1d_pos, tf_sum / ndist, linewidth=3, color="black",
         label=rf"mean $D_1-D_{ndist}$")
plt.axvline(fmin, color="blue", label=r"$f_{min}$")
plt.axvline(fmax_voxel, color="green", label=r"$f_{max}$ (voxel Nyquist)")
plt.ylabel(r"$\sin^2(\pi \lambda D f^2)$", fontsize=15)
plt.xlabel(r"$f$ (m$^{-1}$)", fontsize=15)
plt.legend()
plt.ylim(-0.02, 1.02)
plt.xlim(0, 1.0e6)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ctf_transfer_functions.png"), dpi=130)
plt.show()


# =============================================================================
# SECTION 6 -- CTF PURE-PHASE RECONSTRUCTION (5 distance combinations)
# =============================================================================
def CTFPurePhase(rads, wlen, dists, fx, fy, alpha=1e-3):
    """
    Multi-distance pure-phase CTF retrieval.

    Parameters
    ----------
    rads : ndarray
        Stack of projections, shape (ndist, ny, nx).
    wlen : float
        X-ray wavelength in meters.
    dists : ndarray
        Effective propagation distances in meters.
    fx, fy : ndarray
        Spatial frequency grids in 1/m.
    alpha : float
        Regularization factor.

    Returns
    -------
    phase : ndarray
        Retrieved phase projection, shape (ny, nx).
    """
    rads = np.asarray(rads)
    dists = np.asarray(dists)

    numerator = np.zeros_like(np.fft.fft2(rads[0]), dtype=np.complex64)
    denominator = np.zeros_like(rads[0], dtype=np.float32)

    q2 = fx**2 + fy**2

    for j in range(len(dists)):
        rad_freq = np.fft.fft2(rads[j])
        ctf = np.sin(np.pi * wlen * dists[j] * q2)

        numerator += ctf * rad_freq
        denominator += 2 * ctf**2

    numerator /= len(dists)
    denominator = denominator / len(dists) + alpha

    phase = np.real(np.fft.ifft2(numerator / denominator))
    phase *= 0.5

    return phase.astype(np.float32)


# ---- normalised contrast: the flat field is vacuum (no sample) ----
rdata_ctf = data - 1.0
ny_data, nx_data = data.shape[1], data.shape[2]
fy_grid, fx_grid = np.meshgrid(np.fft.fftfreq(ny_data, d=voxelsize),
                               np.fft.fftfreq(nx_data, d=voxelsize),
                               indexing="ij")
wlen = wavelength
alpha_val = 1e-2

combinations = {
    "Case 01: D1":               [0],
    "Case 02: D1+D2":            [0, 1],
    "Case 03: D1+D2+D3":         [0, 1, 2],
    "Case 04: D1+D2+D3+D4":      [0, 1, 2, 3],
    "Case 05: D1+D2+D3+D4+D5":   [0, 1, 2, 3, 4],
}
CTFrec_dict = {}
for label, idx in combinations.items():
    idx = np.array(idx)
    rads = rdata_ctf[idx]
    distances_case = distances_rec[idx]
    recCTFPurePhase = CTFPurePhase(rads, wlen, distances_case, fx_grid, fy_grid, alpha_val)
    CTFrec_dict[label] = recCTFPurePhase

# ---- ground truth: the CTF cannot recover DC -> mean-subtract both ----
gt_phase = np.angle(psi).astype(np.float32)
gt_phase_ac = gt_phase - gt_phase.mean()
for label in CTFrec_dict:
    CTFrec_dict[label] = CTFrec_dict[label] - CTFrec_dict[label].mean()


# =============================================================================
# SECTION 6b -- CTF transfer function, per (cumulative) D-combination
# The sum is divided by the number of distances (mean CTF^2) so that
# the black curves are comparable BETWEEN cases.
# =============================================================================
for _label, _idx in combinations.items():
    plt.figure(figsize=FIGSIZE_SINGLE)
    _tf_case = []
    for _j in _idx:
        _tf = np.sin(np.pi * wavelength * distances_rec[_j] * fx_1d_pos**2)**2
        _tf_case.append(_tf)
        plt.plot(fx_1d_pos, _tf, "--", linewidth=1, label=rf"$D_{_j+1}$")
    _tf_case = np.array(_tf_case)
    plt.plot(fx_1d_pos, _tf_case.mean(axis=0), linewidth=3, color="black",
             label="mean (comparable across cases)")
    plt.axvline(fmin, color="blue", label=r"$f_{min}$")
    plt.axvline(fmax_voxel, color="green", label=r"$f_{max,vox}$")
    plt.ylabel(r"$\sin^2(\pi \lambda D f^2)$", fontsize=15)
    plt.xlabel(r"$f$ (m$^{-1}$)", fontsize=15)
    plt.title(_label, fontsize=11)
    plt.legend(fontsize=7, loc="upper right", framealpha=0.85)
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, 1.0e6)
    plt.grid()
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 6c -- MULTI-METHOD RECONSTRUCTION: CTF (Rm), multiCTF, CTFPurePhase,
# homoCTF / multiPaganin / single-Paganin / sglDstCTF for Ta and Si3N4,
# all on the FULL (5-distance) combination.
# =============================================================================
psize = voxelsize

rads_ctf = rdata_ctf.astype(np.float32)    # shape: (ndist, ny, nx)
ndist, ny, nx = rads_ctf.shape
rads_intensity = rads_ctf + 1.0

fx = np.fft.fftfreq(nx, d=psize)
fy = np.fft.fftfreq(ny, d=psize)
fx_grid, fy_grid = np.meshgrid(fx, fy)
_Rm_full_sim = np.ones((ny, nx, ndist), dtype=np.float32)

print(f"Shapes check - rads: {rads_ctf.shape}, Rm: {_Rm_full_sim.shape}, fx_grid: {fx_grid.shape}")

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

alpha_val = 1e-2
alpha_val_other = alpha_val

results_to_plot = {}

# ---- CTF (with Rm weighting) -- multi-distance, all combinations ----
ny_data, nx_data = ny, nx
_Rm_full = np.ones((ny_data, nx_data, ndist), dtype=np.float32)

CTF_dict = {}
for label, idx in combinations.items():
    idx = np.array(idx)
    rads = rdata_ctf[idx]
    distances_case = distances_rec[idx]
    _Rm_case = _Rm_full[:, :, idx]
    rec = CTF(rads, wlen, distances_case, fx_grid, fy_grid, _Rm_case, alpha_val_other)
    CTF_dict[label] = rec.astype(np.float32)
    print(f"[CTF] {label}: reconstruction shape {rec.shape}")

results_to_plot[f"CTF (Rm) ({full_label})"] = CTF_dict[full_label]

# ---- multiCTF (material-independent) ----
rads_ctf_all = rdata_ctf[idx_all]
dist_case_ctf = distances_rec[idx_all]
Rm_case_ctf = _Rm_full_sim[:, :, idx_all]
rec_multictf = CTF(rads_ctf_all, wlen, dist_case_ctf, fx_grid, fy_grid, Rm_case_ctf, alpha_val)
results_to_plot[f"multiCTF ({full_label})"] = rec_multictf.astype(np.float32)

# ---- CTFPurePhase (material-independent) ----
rec_pure_phase = CTFPurePhase(rads_ctf_all, wlen, dist_case_ctf, fx_grid, fy_grid, alpha_val)
results_to_plot[f"CTFPurePhase ({full_label})"] = rec_pure_phase.astype(np.float32)

rads_data_all = rads_intensity[idx_all]

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

# ---- vertical comparison plot ----
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
# SECTION 7 -- MTF ANALYSIS (direct, on the real-valued reconstruction, with
# circular profiles) + diagnostic overlay of the excluded zones
# =============================================================================
# NOTE on EXCLUDE_RADIUS_RANGES2 / EXCLUDE_ANGLE_RANGES2: the original
# script reassigned these two names three times across this MTF section,
# the diagnostic-plot section, and the CNR section (Section 8), each time
# with DIFFERENT ranges. Since they're plain globals, whichever assignment
# ran most recently silently overrode the earlier ones for any code that
# read them afterwards. To avoid that ambiguity, each analysis below now
# uses its OWN explicitly named exclusion-range variables instead of
# sharing one mutable global.
MTF_EXCLUDE_RADIUS_RANGES = ((53, 60), (95, 145), (200, 280), (430, 550))
MTF_EXCLUDE_ANGLE_RANGES = ((125, 150),)

xcenter, ycenter = (nx - 1) / 2.0, (ny - 1) / 2.0   # simulated star is centered on the full grid
_exclude_radius_ranges = MTF_EXCLUDE_RADIUS_RANGES
_exclude_angle_ranges = MTF_EXCLUDE_ANGLE_RANGES

r_min, r_max, n_radii = 10, 950, 1000
n_arms = 36
N_SAMPLES_PROFILE = 4096
NOISE_CORRECT_MODULATION = True
NORM_RADIUS_RANGE = (550, 950)
NOISE_FREQ_OFFSETS = (-3, -2, +2, +3, +5, +7)
voxelsize_um = voxelsize * 1e6
nyquist = 1 / (2 * voxelsize_um)


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
    return np.sqrt(np.clip(sig**2 - noi**2, 0.0, None))


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

# ---- MTF10 (sustained crossing) per case ----
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
    plt.xlabel("Spatial frequency (cycles/µm)")
    plt.ylabel("Normalized MTF")
    plt.title(f"MTF -- {label}")
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.ylim(0, 1.2)
    plt.xlim(0, 3)
    plt.tight_layout()
    plt.show()

# ---- MTF all cases in a single plot ----
plt.figure(figsize=FIGSIZE_SINGLE)
_colors = plt.cm.tab10(np.linspace(0, 1, len(mtf_results_combos)))
for (label, res), c in zip(mtf_results_combos.items(), _colors):
    plt.plot(res["freqs"], res["mtf"], "-o", markersize=3, linewidth=1.3,
             color=c, label=label)
plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
plt.xlabel("Spatial frequency (cycles/µm)")
plt.ylabel("Normalized MTF")
plt.title("MTF -- all cases")
plt.grid(True)
plt.legend(fontsize=8, loc="upper right")
plt.ylim(0, 1.2)
plt.xlim(0, 3)
plt.tight_layout()
plt.show()

# =============================================================================
# DIAGNOSTIC PLOT: the star image with the excluded radius/angle zones
# overlaid, AND the actual circular sampling radii (radii_used) as thin
# green rings, so you can visually confirm both what is excluded and
# exactly where the MTF profiles are actually being sampled.
# =============================================================================
def plot_star_with_exclusions(img, xcenter, ycenter, r_max_plot,
                              exclude_radius_ranges, exclude_angle_ranges,
                              sample_radii=None,
                              n_sample_rings_to_draw=60,
                              title="Reconstruction with excluded zones",
                              vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    if vmin is None:
        vmin = np.percentile(img, 2)
    if vmax is None:
        vmax = np.percentile(img, 98)
    im = ax.imshow(img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)

    # radius exclusions -> red annuli
    for r0, r1 in exclude_radius_ranges:
        ax.add_patch(Wedge((xcenter, ycenter), r1, 0, 360,
                           width=(r1 - r0), facecolor="red",
                           edgecolor="none", alpha=0.35, zorder=3))

    # angle exclusions -> orange wedge sector, spanning the plotted radius
    for a0, a1 in exclude_angle_ranges:
        if a0 <= a1:
            spans = [(a0, a1)]
        else:
            spans = [(a0, 360), (0, a1)]
        for t1, t2 in spans:
            ax.add_patch(Wedge((xcenter, ycenter), r_max_plot, t1, t2,
                               width=r_max_plot, facecolor="orange",
                               edgecolor="none", alpha=0.30, zorder=3))

    # actual MTF sampling radii, drawn as thin green circles (subsampled to
    # n_sample_rings_to_draw evenly spaced rings, since radii_used typically
    # has hundreds of entries and drawing every one would paint a solid disk)
    if sample_radii is not None and len(sample_radii) > 0:
        idx = np.linspace(0, len(sample_radii) - 1, n_sample_rings_to_draw).astype(int)
        idx = np.unique(idx)
        theta = np.linspace(0, 2 * np.pi, 500)
        for r in sample_radii[idx]:
            ax.plot(xcenter + r * np.cos(theta), ycenter + r * np.sin(theta),
                    color="lime", linewidth=0.6, alpha=0.6, zorder=2)

    ax.plot(xcenter, ycenter, "r+", markersize=12, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("x pixels")
    ax.set_ylabel("y pixels")
    plt.colorbar(im, ax=ax, label="value")

    legend_handles = [
        Line2D([0], [0], color="none", marker="s", markerfacecolor="red",
              alpha=0.35, markersize=15, label="excl. radius"),
        Line2D([0], [0], color="none", marker="s", markerfacecolor="orange",
              alpha=0.30, markersize=15, label="excl. angle"),
    ]
    if sample_radii is not None and len(sample_radii) > 0:
        legend_handles.append(
            Line2D([0], [0], color="lime", linewidth=1.5, label="sampled radii")
        )
    ax.legend(handles=legend_handles, loc="upper right", facecolor="black",
             edgecolor="none", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.show()


_SHOW_ALL_CASES_WITH_EXCLUSIONS = True   # set False to only show the first case

_cases_to_plot = (CTFrec_dict.items() if _SHOW_ALL_CASES_WITH_EXCLUSIONS
                 else [next(iter(CTFrec_dict.items()))])

for _label, _img in _cases_to_plot:
    plot_star_with_exclusions(
        _img, xcenter, ycenter,
        r_max_plot=r_max,
        exclude_radius_ranges=MTF_EXCLUDE_RADIUS_RANGES,
        exclude_angle_ranges=MTF_EXCLUDE_ANGLE_RANGES,
        sample_radii=radii_used,
        n_sample_rings_to_draw=60,
        title=f"{_label} -- excluded zones + sampled radii",
    )


# =============================================================================
# SECTION 8 -- CNR ANALYSIS (multi-ROI over angle)
# =============================================================================
# NOTE: this section originally reused the shared EXCLUDE_RADIUS_RANGES2 /
# EXCLUDE_ANGLE_RANGES2 names with YET a different set of ranges than
# Section 7's MTF analysis -- see the NOTE at the top of Section 7. It now
# uses its own explicitly named ranges instead.
CNR_EXCLUDE_RADIUS_RANGES = ((100, 160), (240, 290), (480, 580))
CNR_EXCLUDE_ANGLE_RANGES = ((125, 150),)


def extract_rotated_roi(img, cx, cy, width, height, angle_deg, order=1):
    """Extract a rectangular ROI patch, centered at (cx, cy) and rotated by angle_deg."""
    theta = np.deg2rad(angle_deg)
    u = np.arange(width) - (width - 1) / 2.0
    v = np.arange(height) - (height - 1) / 2.0
    uu, vv = np.meshgrid(u, v)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xx = cx + uu * cos_t - vv * sin_t
    yy = cy + uu * sin_t + vv * cos_t
    patch = map_coordinates(img, np.vstack([yy.ravel(), xx.ravel()]),
                            order=order, mode="reflect")
    return patch.reshape(height, width)


def compute_cnr(img, signal_roi, bg_roi):
    """Contrast-to-noise ratio between a signal ROI and a background ROI."""
    sig_patch = extract_rotated_roi(img, **signal_roi)
    bg_patch = extract_rotated_roi(img, **bg_roi)
    mu_sig, mu_bg = np.mean(sig_patch), np.mean(bg_patch)
    sigma_bg = np.std(bg_patch)
    return np.abs(mu_sig - mu_bg) / (sigma_bg + 1e-12), mu_sig, mu_bg, sigma_bg


def _roi_corners(cx, cy, width, height, angle_deg):
    """4 corner coordinates of a rotated rectangle, for plotting."""
    theta = np.deg2rad(angle_deg)
    hw, hh = width / 2.0, height / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return local @ R.T + np.array([cx, cy])


def plot_rois_multi(img, roi_pairs, vmin=None, vmax=None, title="",
                    exclude_radius_ranges=(), exclude_angle_ranges=(),
                    r_max_plot=None):
    """Overlay all signal/background ROI rectangles and exclusion zones on the image."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    im = ax.imshow(img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    for r0, r1 in exclude_radius_ranges:
        ax.add_patch(patches.Wedge((xcenter, ycenter), r1, 0, 360,
                                   width=(r1 - r0), facecolor="red",
                                   edgecolor="none", alpha=0.30, zorder=3))
    _r_max_plot = r_max_plot if r_max_plot is not None else 1100
    for a0, a1 in exclude_angle_ranges:
        spans = [(a0, a1)] if a0 <= a1 else [(a0, 360), (0, a1)]
        for t1, t2 in spans:
            ax.add_patch(patches.Wedge((xcenter, ycenter), _r_max_plot, t1, t2,
                                       width=_r_max_plot, facecolor="orange",
                                       edgecolor="none", alpha=0.30, zorder=3))
    for sig_roi, bg_roi in roi_pairs:
        ax.add_patch(patches.Polygon(_roi_corners(**sig_roi), closed=True,
                                     linewidth=1.2, edgecolor="lime",
                                     facecolor="none", zorder=4))
        ax.add_patch(patches.Polygon(_roi_corners(**bg_roi), closed=True,
                                     linewidth=1.2, edgecolor="red",
                                     facecolor="none", zorder=4))

    ax.set_title(f"{title} -- CNR ROIs")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cnr_rois.png"), dpi=130)
    plt.show()


_period_deg = 360.0 / arm_pairs
_half_period_deg = 0.5 * _period_deg
base_angle = np.rad2deg(sample_meta["dphi"]) / 2.0
_roi_radius_px = 900
assert _roi_radius_px < sample_meta["r_out"] / voxelsize, \
    "CNR ROI radius is outside the star!"


def _wedge_index_of(angle_deg):
    return int(np.floor((np.deg2rad(angle_deg) % (2 * np.pi)) / sample_meta["dphi"]))


assert _wedge_index_of(base_angle) % 2 == 0, "signal ROI is not on a Ta wedge!"
assert _wedge_index_of(base_angle + _half_period_deg) % 2 == 1, \
    "background ROI is not in a gap!"

print(f"[CNR] base_angle = {base_angle:.2f} deg -> signal on Ta wedge, "
      f"background at +{_half_period_deg:.1f} deg in the SiC gap")


def _in_excluded(angle_deg):
    a = angle_deg % 360
    return any(a0 <= a <= a1 for a0, a1 in CNR_EXCLUDE_ANGLE_RANGES)


angles_list = [
    a for a in range(0, 360, int(_period_deg))
    if not _in_excluded(a + base_angle)
    and not _in_excluded(a + base_angle + _half_period_deg)
]

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

# Plot the ROI overlay only for the full 5-distance case
_plot_label = "Case 05: D1+D2+D3+D4+D5"
_plot_img = CTFrec_dict[_plot_label]
plot_rois_multi(_plot_img, multi_roi_pairs,
                vmin=np.percentile(_plot_img, 2), vmax=np.percentile(_plot_img, 98),
                title=_plot_label,
                exclude_radius_ranges=CNR_EXCLUDE_RADIUS_RANGES,
                exclude_angle_ranges=CNR_EXCLUDE_ANGLE_RANGES,
                r_max_plot=_roi_radius_px * 1.1)


# =============================================================================
# Figure 1: ground truth + 5 raw images on the detector (D1-D5)
# =============================================================================
fig1, axs1 = plt.subplots(2, 3, figsize=(3 * PANEL_SIZE[0], 2 * PANEL_SIZE[1]))
axs1 = axs1.flatten()
axs1[0].imshow(gt_phase_ac, cmap="gray")
axs1[0].set_title("Ground truth phase (mean-subtracted)", fontsize=10,
                  fontweight="bold")
axs1[0].axis("off")
for k in range(ndist):
    ax = axs1[k + 1]
    ax.imshow(data[k], cmap="gray",
              vmin=np.percentile(data[k], 2), vmax=np.percentile(data[k], 98))
    ax.set_title(f"Raw detector D{k+1}", fontsize=10)
    ax.axis("off")
plt.suptitle("Raw images (flat-field normalised)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gt_plus_raw_grid.png"), dpi=130)
plt.show()


# =============================================================================
# Figure 2: ground truth + 5 CTF reconstructions (Case 01-05)
# =============================================================================
_case_labels_list = list(combinations.keys())
fig2, axs2 = plt.subplots(2, 3, figsize=(3 * PANEL_SIZE[0], 2 * PANEL_SIZE[1]))
axs2 = axs2.flatten()
axs2[0].imshow(gt_phase_ac, cmap="gray",
               vmin=np.percentile(gt_phase_ac, 2), vmax=np.percentile(gt_phase_ac, 98))
axs2[0].set_title("Ground truth phase (mean-subtracted)", fontsize=10,
                  fontweight="bold")
axs2[0].axis("off")
for k in range(ndist):
    ax = axs2[k + 1]
    img = CTFrec_dict[_case_labels_list[k]]
    # Per-image autoscaling (rather than GT-referenced scaling), to check
    # whether any halo artifact is just hidden by the shared scale above.
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(img, 2), vmax=np.percentile(img, 98))
    ax.set_title(_case_labels_list[k], fontsize=9)
    ax.axis("off")
plt.suptitle("CTF reconstruction (per-image autoscaled)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gt_plus_recon_grid.png"), dpi=130)
plt.show()


# =============================================================================
# SANITY CHECK (supersampled for a smooth, clearly-shaped blob): source blur
# ALONE (full + zoomed) vs source blur + detector PSF (full + zoomed)
# =============================================================================
SUPERSAMPLE_FACTOR = 10   # render at 10x finer resolution for a smooth shape

test_size = 50 * SUPERSAMPLE_FACTOR
test_img = np.zeros((test_size, test_size), dtype=np.float32)
test_img[test_size//2, test_size//2] = 1.0

# Scale the sigma values up by the same supersample factor (since we're
# working on a finer grid, the same PHYSICAL blur corresponds to a larger
# sigma in these finer pixel units)
test_sigma_h = (SOURCE_SIZE_H * FWHM_TO_SIGMA * z2[0] / (z1_true[0] + z2[0])
                / voxelsize) * SUPERSAMPLE_FACTOR
test_sigma_v = (SOURCE_SIZE_V * FWHM_TO_SIGMA * z2[0] / (z1_true[0] + z2[0])
                / voxelsize) * SUPERSAMPLE_FACTOR

print(f"Using actual D1 source blur (supersampled x{SUPERSAMPLE_FACTOR}): "
      f"sigma_h={test_sigma_h:.2f}px, sigma_v={test_sigma_v:.2f}px")

test_blurred = ndimage.gaussian_filter(test_img, sigma=(test_sigma_v, test_sigma_h))
_detector_sigma_supersampled = DETECTOR_PSF_SIGMA_REAL_PX * SUPERSAMPLE_FACTOR
test_blurred_with_detector = ndimage.gaussian_filter(
    test_blurred, sigma=_detector_sigma_supersampled)

_cy, _cx = test_size // 2, test_size // 2
_zoom_half = int(max(test_sigma_h, test_sigma_v, _detector_sigma_supersampled) * 4)
_zoom_half = max(_zoom_half, 10)

fig, axs = plt.subplots(1, 4, figsize=(24, 6))

im0 = axs[0].imshow(test_blurred, cmap="gray")
axs[0].set_title(f"Full view: Source blur ONLY\n"
                 f"sigma=(v={test_sigma_v:.1f}, h={test_sigma_h:.1f})")
axs[0].set_xlabel("x (columns)")
axs[0].set_ylabel("y (rows)")
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(test_blurred, cmap="gray")
axs[1].set_xlim(_cx - _zoom_half, _cx + _zoom_half)
axs[1].set_ylim(_cy + _zoom_half, _cy - _zoom_half)
axs[1].set_title(f"Zoomed: Source blur ONLY\n"
                 f"H={SOURCE_SIZE_H*1e9:.0f}nm, V={SOURCE_SIZE_V*1e9:.0f}nm")
axs[1].set_xlabel("x (columns)")
axs[1].set_ylabel("y (rows)")
plt.colorbar(im1, ax=axs[1], fraction=0.046)

im2 = axs[2].imshow(test_blurred_with_detector, cmap="gray")
axs[2].set_title(f"Full view: Source + Detector PSF\n"
                 f"(detector = {DETECTOR_PSF_SIGMA_REAL_PX}px)")
axs[2].set_xlabel("x (columns)")
axs[2].set_ylabel("y (rows)")
plt.colorbar(im2, ax=axs[2], fraction=0.046)

im3 = axs[3].imshow(test_blurred_with_detector, cmap="gray")
axs[3].set_xlim(_cx - _zoom_half, _cx + _zoom_half)
axs[3].set_ylim(_cy + _zoom_half, _cy - _zoom_half)
axs[3].set_title(f"Zoomed: Source + Detector PSF\n"
                 f"Note: more isotropic (rounder) shape")
axs[3].set_xlabel("x (columns)")
axs[3].set_ylabel("y (rows)")
plt.colorbar(im3, ax=axs[3], fraction=0.046)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 9 -- RESOLUTION / RING-PROFILE ANALYSIS, for every distance
# combination, with a combined MTF plot at the end.
#
# NOTE: the original script had this section three times, with the first
# version (single "full_label" case only) additionally fixing a real bug --
# the Modregger ROI needs a fixed PHYSICAL size (in um), not a fixed pixel
# count, otherwise it silently covers only the unresolved, aliased center
# of the star whenever the voxel size shrinks (e.g. closer-to-focus
# geometries). That physical-size fix is preserved below and applied to
# every case in the loop.
# =============================================================================
from datetime import datetime
from usefull_functions import plot_abs_phase_ring_profiles, resolution_summary

_script_start = datetime.now()

if "xcenter" not in globals() or "ycenter" not in globals():
    xcenter, ycenter = (nx - 1) / 2.0, (ny - 1) / 2.0
    print(f"[Section 9] xcenter/ycenter not set -- using grid center: "
          f"({xcenter:.1f}, {ycenter:.1f})")

if "sample_meta" in globals() and "r_out" in sample_meta and "r_in" in sample_meta:
    _r_out_px = sample_meta["r_out"] / psize
    _r_in_px = sample_meta["r_in"] / psize
else:
    _r_out_px = 0.45 * min(ny, nx)
    _r_in_px = 0.02 * min(ny, nx)
    print(f"[Section 9] sample_meta not found -- using fallback radii: "
          f"r_in={_r_in_px:.1f}px, r_out={_r_out_px:.1f}px "
          f"(VERIFY these match the actual star in your data)")

if "z1" in globals() and "z2" in globals():
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
print("  ----  ")

# ---- Modregger ROI with a fixed PHYSICAL size (um), converted to pixels
# using the CURRENT psize, instead of a fixed pixel count that silently
# breaks whenever psize changes (see the note above). The physical size
# used here matches the DanMAX reference-voxel ROI (900x1000 px at
# ~42.29 nm/px). ----
_REFERENCE_VOXEL_UM = 0.04229          # DanMAX reference voxel (42.29 nm)
_ROI_HALF_HEIGHT_UM = 450 * _REFERENCE_VOXEL_UM   # = 19.03 um
_ROI_HALF_WIDTH_UM = 500 * _REFERENCE_VOXEL_UM    # = 21.15 um

_psize_um = psize * 1e6
_roi_half_h_px = int(round(_ROI_HALF_HEIGHT_UM / _psize_um))
_roi_half_w_px = int(round(_ROI_HALF_WIDTH_UM / _psize_um))

# Clamp so the ROI never runs outside the image, regardless of psize.
_roi_half_h_px = min(_roi_half_h_px, int(ycenter) - 1, ny - int(ycenter) - 1)
_roi_half_w_px = min(_roi_half_w_px, int(xcenter) - 1, nx - int(xcenter) - 1)

print(f"[Section 9] Modregger ROI: physical size = "
      f"{2*_ROI_HALF_HEIGHT_UM:.2f} x {2*_ROI_HALF_WIDTH_UM:.2f} um "
      f"-> {2*_roi_half_h_px} x {2*_roi_half_w_px} px "
      f"(psize = {_psize_um*1e3:.2f} nm)")

# ---- MTF parameters shared with resolution_summary below ----
_r_min = max(50, 1.2 * _r_in_px)
_r_max = 0.95 * _r_out_px
_n_radii = 500
_n_arms = arm_pairs if "arm_pairs" in globals() else 36
_excl_r = EXCLUDE_RADIUS_RANGES if "EXCLUDE_RADIUS_RANGES" in globals() else ()
_excl_a = EXCLUDE_ANGLE_RANGES if "EXCLUDE_ANGLE_RANGES" in globals() else ()
_norm_range = (0.6 * _r_out_px, 0.8 * _r_out_px)

_radii_all_s9 = np.linspace(_r_min, _r_max, _n_radii)


def _bad_radius_s9(r):
    return any(r0 <= r <= r1 for r0, r1 in _excl_r)


radii_used_s9 = np.array([r for r in _radii_all_s9 if not _bad_radius_s9(r)])
freqs_used_s9 = _n_arms / (2 * np.pi * radii_used_s9 * voxelsize_um)
noise_freqs_s9 = [_n_arms + off for off in (-3, -2, 2, 3, 5, 7)
                  if (_n_arms + off) > 0]


def _mtf_for_image(img):
    """MTF via circular profiles, using Section 9's r_min/r_max/norm_radius_range."""
    sig, noi = [], []
    for r in radii_used_s9:
        angle_deg, prof = circular_profile_angle(
            img, xcenter, ycenter, r, n_samples=4096)
        amp, noise_amp = modulation_lstsq(
            angle_deg, prof, n_arms=_n_arms,
            exclude_angle_ranges=_excl_a, noise_freqs=noise_freqs_s9)
        sig.append(amp)
        noi.append(noise_amp)
    sig = np.asarray(sig, float)
    noi = np.nan_to_num(np.asarray(noi, float), nan=0.0)
    sig_dn = np.sqrt(np.clip(sig**2 - noi**2, 0.0, None))
    sel = (radii_used_s9 >= _norm_range[0]) & (radii_used_s9 <= _norm_range[1])
    mtf = sig_dn / (np.nanmean(sig_dn[sel]) + 1e-12)
    order = np.argsort(freqs_used_s9)
    return freqs_used_s9[order], mtf[order]


mtf_results_s9 = {}
all_case_results = {}
for label, idx in combinations.items():
    _img = CTFrec_dict[label]
    # CTFPurePhase's output is a real-valued phase map -> wrap into complex
    # (unit amplitude) so resolution_summary/plot_abs_phase_ring_profiles
    # can work with the abs/phase split they expect.
    q_wrapped = np.exp(1j * _img).astype(np.complex64)

    print("=" * 62)
    print(f"key = {label}   ndist = {len(idx)}   alpha = {alpha_val}")
    print("=" * 62)

    result = plot_abs_phase_ring_profiles(
        q_wrapped, xcenter=xcenter, ycenter=ycenter,
        radius_px=min(300, 0.9 * _r_out_px),
        voxelsize_um=voxelsize_um, normalize=True,
        phase_vmin=-0.8, phase_vmax=0.8,
    )

    res = resolution_summary(
        q=q_wrapped, voxelsize_um=voxelsize_um,
        rec_params={"key": label, "ndist": len(idx), "alpha": alpha_val,
                    "z1": z1 if "z1" in globals() else None,
                    "z2": z2 if "z2" in globals() else None},
        xcenter=xcenter, ycenter=ycenter,
        r_min=_r_min, r_max=_r_max, n_radii=_n_radii, n_arms=_n_arms,
        exclude_radius_ranges=_excl_r, exclude_angle_ranges=_excl_a,
        norm_radius_range=(550, 950),
        modregger_roi=(slice(int(ycenter - _roi_half_h_px), int(ycenter + _roi_half_h_px)),
                       slice(int(xcenter - _roi_half_w_px), int(xcenter + _roi_half_w_px))),
        modregger_nblfac=2.0,
        show_plots=True,
    )
    all_case_results[label] = res

    f_s, mtf_s = _mtf_for_image(_img)
    mtf_results_s9[label] = dict(freqs=f_s, mtf=mtf_s)

if "gt_phase_ac" in globals():
    f_gt, mtf_gt = _mtf_for_image(gt_phase_ac)

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

# ---- combined MTF plot, all cases together ----
plt.figure(figsize=FIGSIZE_SINGLE)
_colors = plt.cm.tab10(np.linspace(0, 1, len(mtf_results_s9)))
for (label, res), c in zip(mtf_results_s9.items(), _colors):
    plt.plot(res["freqs"], res["mtf"], "-o", markersize=3, linewidth=1.3,
              color=c, label=label)
plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
plt.xlabel("Spatial frequency (cycles/µm)")
plt.ylabel("Normalized MTF")
plt.title("MTF")
plt.grid(True)
plt.legend(fontsize=8, loc="upper right")
plt.ylim(0, 1.2)
plt.xlim(0, 2.8)
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 9b -- Combined table: MTF10 + Modregger (phase x/y, abs x/y), for
# the material-comparison reconstructions built in Section 6c
# (results_to_plot: CTF (Rm), multiCTF, CTFPurePhase, homoCTF/multiPaganin/
# Paganin/sglDstCTF for Ta and Si3N4), all on the FULL distance combination.
# =============================================================================
import io
import re
import contextlib

_pat = re.compile(
    r"^\s*(phase|absorption)\s+([xy])\s+([\d.]+)\s+([\d.]+)", re.MULTILINE
)

rows = []
for key, img in results_to_plot.items():
    profiles = extract_profiles(img)
    sig, noi = modulation_from_profiles(profiles, _exclude_angle_ranges)
    mtf = normalise_mtf(sig, noi)
    f_s, mtf_s, r_s = _sorted_by_freq(mtf)
    f10 = _sustained_mtf_crossing(f_s, mtf_s)
    mtf10_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan

    q = np.exp(1j * img).astype(np.complex64) if "PurePhase" in key \
        else img.astype(np.complex64)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        resolution_summary(
            q=q,
            voxelsize_um=psize * 1e6,
            rec_params={"key": key},
            xcenter=xcenter, ycenter=ycenter,
            r_min=max(50, 1.2 * _r_in_px),
            r_max=0.95 * _r_out_px,
            n_radii=500,
            n_arms=arm_pairs if "arm_pairs" in globals() else 36,
            exclude_radius_ranges=EXCLUDE_RADIUS_RANGES if "EXCLUDE_RADIUS_RANGES" in globals() else (),
            exclude_angle_ranges=EXCLUDE_ANGLE_RANGES if "EXCLUDE_ANGLE_RANGES" in globals() else (),
            norm_radius_range=(550, 950),
            modregger_roi=(slice(int(ycenter - 450), int(ycenter + 450)),
                           slice(int(xcenter - 500), int(xcenter + 500))),
            modregger_nblfac=2.0,
            show_plots=False,
        )

    matches = _pat.findall(buf.getvalue())[-4:]
    vals = {f"{ch}_{d}": float(r) for ch, d, r, u in matches}   # um

    rows.append({
        "method": key,
        "MTF10_nm": mtf10_nm,
        "phase_x_nm": vals.get("phase_x", np.nan) * 1000,
        "phase_y_nm": vals.get("phase_y", np.nan) * 1000,
        "abs_x_nm": vals.get("absorption_x", np.nan) * 1000,
        "abs_y_nm": vals.get("absorption_y", np.nan) * 1000,
    })

resolution_table = pd.DataFrame(rows)
print(resolution_table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# =============================================================================
# SECTION 10 -- Progression dicts for EVERY method (D1 -> D1+D2+...+D5),
# a combined resolution table across all methods x all cases, and a grid
# plot of every reconstructed image.
# =============================================================================
_delta_ta, _beta_ta = materials["Ta"]["delta"], materials["Ta"]["beta"]

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

# Single-distance methods only make sense for D1 (the reference distance)
Paganin_single_dict = {
    "D1": Paganin(rads_intensity[0], wlen, distances_rec[0], _delta_ta, _beta_ta,
                  fx_grid, fy_grid, _Rm_full_sim[:, :, 0]).astype(np.float32)
}

sglDstCTF_dict = {
    "D1": sglDstCTF(rads_intensity[0], wlen, distances_rec[0], _delta_ta, _beta_ta,
                     fx_grid, fy_grid, _Rm_full_sim[:, :, 0], alpha_val).astype(np.float32)
}

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
        profiles = extract_profiles(img)
        sig, noi = modulation_from_profiles(profiles, _exclude_angle_ranges)
        mtf = normalise_mtf(sig, noi)
        f_s, mtf_s, r_s = _sorted_by_freq(mtf)
        f10 = _sustained_mtf_crossing(f_s, mtf_s)
        mtf10_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan

        # CTFPurePhase -> real-valued (phase only) -> wrap into complex
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
                exclude_radius_ranges=EXCLUDE_RADIUS_RANGES if "EXCLUDE_RADIUS_RANGES" in globals() else (),
                exclude_angle_ranges=EXCLUDE_ANGLE_RANGES if "EXCLUDE_ANGLE_RANGES" in globals() else (),
                norm_radius_range=(0.6 * _r_out_px, 0.8 * _r_out_px),
                modregger_roi=(slice(int(ycenter - 450), int(ycenter + 450)),
                               slice(int(xcenter - 500), int(xcenter + 500))),
                modregger_nblfac=2.0,
                show_plots=False,
            )

        matches = _pat.findall(buf.getvalue())[-4:]
        vals = {f"{ch}_{d}": float(r) for ch, d, r, u in matches}   # um

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

# ---- grid plot of every reconstructed image (all methods x all cases) ----
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


# =============================================================================
# SECTION 11 -- Arc alignment v3: register a REAL measured reconstruction
# (loaded from disk) to this simulation's matching case, and compare them
# quantitatively along several concentric arcs. This validates the forward
# model against the real DanMAX measurement.
# =============================================================================
# Why this is fast: earlier versions warped the ENTIRE 2500x2500 image
# (~6 MP) on every optimizer iteration -> hundreds of full-resolution
# interpolations. This version does all alignment work on a crop around the
# star target plus binning (ALIGN_BIN), i.e. ~0.1 MP per evaluation
# (~60x faster), with a SINGLE full-resolution warp at the end.
#
# Alignment strategy: band-pass filter -> rotation from 1D angular
# cross-correlation -> translation from phase correlation -> bounded refine
# with a guard rail (the refine step is only kept if it improves the NCC).
#
# The final alignment used from here on is the SCALE-CORRECTED simulation
# (the simulation shrunk by the free scale estimate found during the scale
# scan), rather than the raw/unscaled simulation compared against a
# scale=1 measured warp. `data0_cropped` is reassigned to this
# scale-corrected simulation right after the full-resolution warp.
#
# The fit summary reports, per concentric radius, Pearson r (shape
# agreement), RMSE (overall residual), and an amplitude ratio
# (measured/sim, contrast/blur agreement) -- letting you distinguish
# genuine misalignment (low Pearson r) from a correct shape but wrong
# contrast/blur amplitude (high Pearson r, amplitude ratio far from 1.0).
# =============================================================================
import time
from scipy.optimize import minimize

_REQUIRED_SIM_GLOBALS = (
    "CTFrec_dict", "combinations", "voxelsize", "magnifications",
    "arm_pairs", "PANEL_SIZE",
)
_missing_sim_globals = [name for name in _REQUIRED_SIM_GLOBALS
                        if name not in globals()]
if _missing_sim_globals:
    raise RuntimeError(
        "SECTION 11 (arc alignment) requires globals already computed "
        f"earlier in this script: {', '.join(_missing_sim_globals)}."
    )

for _v in (
    "SIM_C", "MEAS_C",
    "SIM_CENTER_X_PX", "SIM_CENTER_Y_PX",
    "MEAS_CENTER_X_PX", "MEAS_CENTER_Y_PX",
    "COMMON_CENTER_X_PX", "COMMON_CENTER_Y_PX",
    "CONCENTRIC_PERCENTS", "CONTRAST_SIGN",
    "MEAS_ALIGN_DX_PX", "MEAS_ALIGN_DY_PX",
    "MEAS_ALIGN_ROT_DEG", "MEAS_ALIGN_SCALE",
    "SCALE_BOUNDS", "SHIFT_BOUND_PX", "ROT_BOUND_DEG",
    "MEASURED_ALIGNED_PATH", "SIM_CASE_KEY",
    "SIM_CENTER_MODE", "MEAS_CENTER_MODE",
    "measured_img", "measured_img_signed",
    "measured_raw", "data0_cropped",
):
    if _v in globals():
        del globals()[_v]

try:
    get_ipython().run_line_magic("matplotlib", "widget")
except Exception:
    pass  # not running inside Jupyter -- fine, just skip the widget backend

# ---- CONFIG ----
# NOTE: the original path pointed at a DTU-cluster location containing a
# personal student ID. Set MCTF_ALIGN_MEASURED_PATH to your own saved
# aligned-measurement .npy file (produced by the real-data pipeline) before
# running this section.
MEASURED_ALIGNED_PATH = os.environ.get(
    "MCTF_ALIGN_MEASURED_PATH",
    "./outputs/all_cases_recon/Case_11_D1-D2-D3-D4-D5.npy",
)

MEASURED_PHYSICAL_PIXEL_M = 0.55e-6
MEASURED_MAGNIFICATION = magnifications[0]
voxelsize_meas_raw = MEASURED_PHYSICAL_PIXEL_M / MEASURED_MAGNIFICATION

SIM_CASE_KEY = "Case 05: D1+D2+D3+D4+D5"
SIM_CENTER_MODE = "grid"  # "grid" | "auto"
MEAS_CENTER_MODE = "fixed"  # "fixed" | "auto"
SIM_CENTER_X_PX, SIM_CENTER_Y_PX = 1250.0, 1250.0
MEAS_CENTER_X_PX, MEAS_CENTER_Y_PX = 1273.0, 1278.0
CONCENTRIC_PERCENTS = [9, 15, 28]
SCALE_BOUNDS = (1, 1)
SHIFT_BOUND_PX = 32.0
ROT_SEARCH_DEG = 2.0
ROT_BOUND_DEG = ROT_SEARCH_DEG

_ARC_ALIGN_CONFIG_NAMES = {
    "MEASURED_ALIGNED_PATH", "SIM_CASE_KEY", "SIM_CENTER_MODE", "MEAS_CENTER_MODE",
    "SIM_CENTER_X_PX", "SIM_CENTER_Y_PX", "MEAS_CENTER_X_PX", "MEAS_CENTER_Y_PX",
    "CONCENTRIC_PERCENTS", "SCALE_BOUNDS", "SHIFT_BOUND_PX", "ROT_BOUND_DEG",
}
_arc_align_overrides = globals().get("ARC_ALIGN_OVERRIDES", {})
if not isinstance(_arc_align_overrides, dict):
    raise TypeError("ARC_ALIGN_OVERRIDES must be a dict when provided")
_unknown_arc_align_overrides = set(_arc_align_overrides) - _ARC_ALIGN_CONFIG_NAMES
if _unknown_arc_align_overrides:
    raise KeyError(
        "Unknown ARC_ALIGN_OVERRIDES key(s): "
        + ", ".join(sorted(_unknown_arc_align_overrides))
    )
for _override_key, _override_value in _arc_align_overrides.items():
    globals()[_override_key] = _override_value
    print(f"[override] {_override_key}={_override_value!r}")

measured_raw = np.load(MEASURED_ALIGNED_PATH).astype(np.float64)
sim_raw = CTFrec_dict[SIM_CASE_KEY].astype(np.float64)

_measured_distance_indices = [
    int(token) for token in re.findall(
        r"D(\d+)", os.path.basename(MEASURED_ALIGNED_PATH)
    )
]
_sim_distance_indices = [int(index) + 1 for index in combinations[SIM_CASE_KEY]]
if _measured_distance_indices != _sim_distance_indices:
    print(
        "[WARN] comparing different distance combinations: measured "
        f"{_measured_distance_indices or 'none'} vs simulation "
        f"{_sim_distance_indices or 'none'} ({SIM_CASE_KEY})."
    )
else:
    print(f"[distance] measured and simulation both use D{',D'.join(map(str, _sim_distance_indices))}")

if SIM_CENTER_MODE not in ("grid", "auto"):
    raise ValueError("SIM_CENTER_MODE must be 'grid' or 'auto'")
if MEAS_CENTER_MODE not in ("fixed", "auto"):
    raise ValueError("MEAS_CENTER_MODE must be 'fixed' or 'auto'")

if SIM_CENTER_MODE == "grid":
    SIM_CENTER_X_PX = (sim_raw.shape[1] - 1) / 2.0
    SIM_CENTER_Y_PX = (sim_raw.shape[0] - 1) / 2.0
SIM_EDGE_RADIUS_UM = 46.8
LINE_ANGLE_DEG = 53

ALIGN_BIN = 8          # binning ONLY for alignment (8 -> fast, 4 -> more accurate)
ROI_FACTOR = 1.15      # crop half-size = ROI_FACTOR * star radius
REFINE = True

SIM_C = (SIM_CENTER_X_PX, SIM_CENTER_Y_PX)
MEAS_C = (MEAS_CENTER_X_PX, MEAS_CENTER_Y_PX)
_sim_grid_center = ((sim_raw.shape[1] - 1) / 2.0, (sim_raw.shape[0] - 1) / 2.0)
_sim_grid_delta = (SIM_C[0] - _sim_grid_center[0], SIM_C[1] - _sim_grid_center[1])
print(
    f"[check] SIM_C vs grid center: SIM_C=({SIM_C[0]:.1f}, {SIM_C[1]:.1f}) "
    f"grid=({_sim_grid_center[0]:.1f}, {_sim_grid_center[1]:.1f}) "
    f"delta=({_sim_grid_delta[0]:+.2f}, {_sim_grid_delta[1]:+.2f}) px"
)
if SIM_CENTER_MODE == "grid" and max(abs(_sim_grid_delta[0]), abs(_sim_grid_delta[1])) > 0.5:
    raise RuntimeError(
        "SIM_CENTER_MODE='grid' produced a SIM_C inconsistent with the "
        "simulated grid center; stale alignment state or configuration leaked."
    )
assert (0.0 <= SIM_C[0] < sim_raw.shape[1] and 0.0 <= SIM_C[1] < sim_raw.shape[0]), \
    f"SIM_C={SIM_C} lies outside sim_raw shape {sim_raw.shape}"
assert (0.0 <= MEAS_C[0] < measured_raw.shape[1] and 0.0 <= MEAS_C[1] < measured_raw.shape[0]), \
    f"MEAS_C={MEAS_C} lies outside measured_raw shape {measured_raw.shape}"
target_shape = sim_raw.shape
data0_cropped = sim_raw

SCALE_FROM_VOXELS = voxelsize_meas_raw / voxelsize
_r_edge_px = SIM_EDGE_RADIUS_UM * 1e-6 / voxelsize
print(f"[scale] sim={voxelsize*1e9:.2f} nm/px, meas={voxelsize_meas_raw*1e9:.2f} nm/px, "
      f"theoretical scale={SCALE_FROM_VOXELS:.5f}")
if np.isclose(voxelsize_meas_raw, voxelsize):
    print(
        "[scale] SCALE_FROM_VOXELS is identically 1.00000 when "
        "MEASURED_PHYSICAL_PIXEL_M/MEASURED_MAGNIFICATION == voxelsize; "
        "this is not an independent check. The actual scale is determined "
        f"only by refine within bounds {SCALE_BOUNDS[0]:.3f}-{SCALE_BOUNDS[1]:.3f}."
    )
print(f"[geom] star radius = {_r_edge_px:.1f} px, alignment bin = {ALIGN_BIN}")


def crop_bin(img, c_xy, half_px, bin_f):
    """Crop around c_xy and mean-bin. Returns (binned_img, center in binned px)."""
    half = int(np.ceil(half_px / bin_f)) * bin_f
    x0 = int(round(c_xy[0])) - half
    y0 = int(round(c_xy[1])) - half
    n = 2 * half
    out = np.full((n, n), np.nan)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + n, img.shape[1]), min(y0 + n, img.shape[0])
    out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    out = np.nan_to_num(out, nan=float(np.nanmedian(out)))
    nb = n // bin_f
    binned = out.reshape(nb, bin_f, nb, bin_f).mean(axis=(1, 3))
    cb = ((c_xy[0] - x0 + 0.5) / bin_f - 0.5, (c_xy[1] - y0 + 0.5) / bin_f - 0.5)
    return binned, cb


def bandpass(img, sigma_low, sigma_high=1.0, eps=1e-9):
    a = np.nan_to_num(img.astype(np.float64), nan=0.0)
    a = ndimage.gaussian_filter(a, sigma_high) - ndimage.gaussian_filter(a, sigma_low)
    return a / (np.sqrt(ndimage.gaussian_filter(a * a, sigma_low)) + eps)


def warp(img, meas_c, sim_c, rot_deg, scale, dx, dy, out_shape, order=1, cval=np.nan,
         _grid_cache={}):
    ny_, nx_ = out_shape
    key = (ny_, nx_)
    if key not in _grid_cache:
        _grid_cache[key] = np.mgrid[0:ny_, 0:nx_].astype(np.float64)
    yy, xx = _grid_cache[key]
    ux = xx - (sim_c[0] + dx)
    uy = yy - (sim_c[1] + dy)
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    src_x = (c * ux + s * uy) / scale + meas_c[0]
    src_y = (-s * ux + c * uy) / scale + meas_c[1]
    return map_coordinates(img, [src_y, src_x], order=order, mode="constant", cval=cval)


def _parabolic_peak(c, i):
    n = len(c)
    ym1, y0, yp1 = c[(i - 1) % n], c[i], c[(i + 1) % n]
    d = ym1 - 2 * y0 + yp1
    return 0.0 if d == 0 else 0.5 * (ym1 - yp1) / d


def detect_object_center(img, guess_xy, r_edge_px, bin_f=8, n_iter=3, verbose=True):
    c = list(guess_xy)
    _padding_warned = False
    for _ in range(n_iter):
        _requested_half = 1.2 * r_edge_px
        _requested_half_binned = int(np.ceil(_requested_half / bin_f)) * bin_f
        _cx0 = int(round(c[0]))
        _cy0 = int(round(c[1]))
        _x0_req = _cx0 - _requested_half_binned
        _x1_req = _cx0 + _requested_half_binned
        _y0_req = _cy0 - _requested_half_binned
        _y1_req = _cy0 + _requested_half_binned
        _missing_left = max(0, -_x0_req)
        _missing_right = max(0, _x1_req - img.shape[1])
        _missing_top = max(0, -_y0_req)
        _missing_bottom = max(0, _y1_req - img.shape[0])
        if not _padding_warned and max(
            _missing_left, _missing_right, _missing_top, _missing_bottom
        ) > 0:
            print(
                "[WARN] center detection is unreliable because the requested "
                "crop requires padding: "
                f"left={_missing_left}px right={_missing_right}px "
                f"top={_missing_top}px bottom={_missing_bottom}px."
            )
            _padding_warned = True
        _max_half = min(
            float(_cx0), float(img.shape[1] - _cx0),
            float(_cy0), float(img.shape[0] - _cy0),
        )
        half = np.floor(min(_requested_half, _max_half) / bin_f) * bin_f
        half = max(float(bin_f), half)
        b, cb = crop_bin(img, c, half, bin_f)
        x0 = int(round(c[0])) - int(np.ceil(half / bin_f)) * bin_f
        y0 = int(round(c[1])) - int(np.ceil(half / bin_f)) * bin_f
        bp = bandpass(b, sigma_low=max(3.0, r_edge_px / bin_f / 6))
        yy, xx = np.mgrid[0:b.shape[0], 0:b.shape[1]].astype(np.float64)
        r_b = r_edge_px / bin_f
        w = np.clip((0.98 * r_b - np.hypot(xx - cb[0], yy - cb[1])) / (0.1 * r_b), 0, 1)
        a = bp * w
        F = np.fft.fft2(a) * np.conj(np.fft.fft2(a[::-1, ::-1]))
        cc = np.fft.ifft2(F / (np.abs(F) + 1e-12)).real
        iy, ix = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
        ny_, nx_ = cc.shape
        sx = (ix - nx_ if ix > nx_ // 2 else ix) + _parabolic_peak(np.abs(cc[iy, :]), ix)
        sy = (iy - ny_ if iy > ny_ // 2 else iy) + _parabolic_peak(np.abs(cc[:, ix]), iy)
        c = [((sx + nx_ - 1) / 2.0 + 0.5) * bin_f - 0.5 + x0,
             ((sy + ny_ - 1) / 2.0 + 0.5) * bin_f - 0.5 + y0]
    out = (c[0], c[1])
    if verbose:
        print(f"[center] guess ({guess_xy[0]:.1f}, {guess_xy[1]:.1f}) -> detected "
              f"({out[0]:.1f}, {out[1]:.1f})  [delta = {out[0]-guess_xy[0]:+.1f}, "
              f"{out[1]-guess_xy[1]:+.1f} px]")
    return out


_t0 = time.time()
if SIM_CENTER_MODE == "auto":
    SIM_C = detect_object_center(sim_raw, SIM_C, _r_edge_px, ALIGN_BIN)
else:
    print(f"[center] SIM_CENTER_MODE='grid': using simulated grid center "
          f"({SIM_C[0]:.1f}, {SIM_C[1]:.1f})")
SIM_CENTER_X_PX, SIM_CENTER_Y_PX = SIM_C
if MEAS_CENTER_MODE == "auto":
    MEAS_C = detect_object_center(measured_raw, MEAS_C, _r_edge_px, ALIGN_BIN)
    MEAS_CENTER_X_PX, MEAS_CENTER_Y_PX = MEAS_C
else:
    _meas_detected_c = detect_object_center(measured_raw, MEAS_C, _r_edge_px, ALIGN_BIN)
    _meas_delta = (_meas_detected_c[0] - MEAS_C[0], _meas_detected_c[1] - MEAS_C[1])
    print(
        f"[center] MEAS_CENTER_MODE='fixed': using "
        f"({MEAS_C[0]:.1f}, {MEAS_C[1]:.1f}); diagnostic detected "
        f"({_meas_detected_c[0]:.1f}, {_meas_detected_c[1]:.1f}), "
        f"delta=({_meas_delta[0]:+.1f}, {_meas_delta[1]:+.1f}) px"
    )
    if max(abs(_meas_delta[0]), abs(_meas_delta[1])) > 2.0:
        print(
            "[WARN] measured-center delta exceeds 2 px: correct this same "
            "center in the real pipeline, otherwise MTF10/Modregger are "
            "computed around the wrong center."
        )
    SIM_CENTER_X_PX, SIM_CENTER_Y_PX = SIM_C
assert (0.0 <= SIM_C[0] < sim_raw.shape[1] and 0.0 <= SIM_C[1] < sim_raw.shape[0]), \
    f"SIM_C={SIM_C} lies outside sim_raw shape {sim_raw.shape}"
assert (0.0 <= MEAS_C[0] < measured_raw.shape[1] and 0.0 <= MEAS_C[1] < measured_raw.shape[0]), \
    f"MEAS_C={MEAS_C} lies outside measured_raw shape {measured_raw.shape}"

_half = ROI_FACTOR * _r_edge_px
sim_b, SIM_Cb = crop_bin(sim_raw, SIM_C, _half, ALIGN_BIN)
meas_b, MEAS_Cb = crop_bin(measured_raw, MEAS_C, _half, ALIGN_BIN)
_r_edge_b = _r_edge_px / ALIGN_BIN
sim_bp = bandpass(sim_b, sigma_low=max(3.0, 25.0 / ALIGN_BIN * 4))
meas_bp = bandpass(meas_b, sigma_low=max(3.0, 25.0 / ALIGN_BIN * 4))
print(f"[prep] align shape = {sim_b.shape} ({time.time()-_t0:.1f}s)")

SPOKE_PERIOD_DEG = 360.0 / arm_pairs


def angular_profile(img_bp, c_xy, r_lo, r_hi, n_ang=1440, n_rad=24):
    th = np.deg2rad(np.linspace(0, 360, n_ang, endpoint=False))
    rr = np.linspace(r_lo, r_hi, n_rad)
    TH, RR = np.meshgrid(th, rr)
    vals = map_coordinates(img_bp, [c_xy[1] + RR * np.sin(TH), c_xy[0] + RR * np.cos(TH)],
                           order=1, mode="constant", cval=0.0)
    p = vals.mean(axis=0)
    return p - p.mean()


p_sim = angular_profile(sim_bp, SIM_Cb, 0.55 * _r_edge_b, 0.95 * _r_edge_b)
p_meas = angular_profile(meas_bp, MEAS_Cb, 0.55 * _r_edge_b, 0.95 * _r_edge_b)
_cc = np.fft.irfft(np.fft.rfft(p_sim) * np.conj(np.fft.rfft(p_meas)), n=p_sim.size)
_lags = np.arange(p_sim.size) * 360.0 / p_sim.size
_lags = np.where(_lags > 180, _lags - 360, _lags)
_win = min(ROT_SEARCH_DEG, SPOKE_PERIOD_DEG / 2)
_ok = np.abs(_lags) <= _win
_rot_pos = float(_lags[_ok][np.argmax(_cc[_ok])])
_rot_neg = float(_lags[_ok][np.argmin(_cc[_ok])])
print(f"[rot] spoke period={SPOKE_PERIOD_DEG:.2f}deg, window=+-{_win:.2f}deg -> "
      f"candidates {_rot_pos:+.3f}deg (same contrast) / {_rot_neg:+.3f}deg (inverted)")

_yyb, _xxb = np.mgrid[0:sim_b.shape[0], 0:sim_b.shape[1]].astype(np.float64)
_rb = np.hypot(_xxb - SIM_Cb[0], _yyb - SIM_Cb[1])
mask_star = (_rb > 0.12 * _r_edge_b) & (_rb < 1.02 * _r_edge_b)
_w = np.clip((1.05 * _r_edge_b - _rb) / (0.05 * _r_edge_b), 0, 1) * \
     np.clip((_rb - 0.08 * _r_edge_b) / (0.06 * _r_edge_b), 0, 1)

_sim_vec = (sim_bp * _w)[mask_star]
_sim_vec = _sim_vec - _sim_vec.mean()
_sim_nrm = np.linalg.norm(_sim_vec)


def ncc(p, img_bp=None):
    dx, dy, rot, scale = p
    src = meas_bp if img_bp is None else img_bp
    w = warp(src, MEAS_Cb, SIM_Cb, rot, scale, dx, dy, sim_b.shape, order=1, cval=0.0)
    v = (w * _w)[mask_star]
    v = v - v.mean()
    d = np.linalg.norm(v) * _sim_nrm
    return 0.0 if d == 0 else float(np.dot(v, _sim_vec) / d)


def phase_shift_px(img_bp, rot_deg):
    m = warp(img_bp, MEAS_Cb, SIM_Cb, rot_deg, SCALE_FROM_VOXELS, 0.0, 0.0,
             sim_b.shape, order=1, cval=0.0)
    F = np.fft.fft2(sim_bp * _w) * np.conj(np.fft.fft2(m * _w))
    cc2 = np.fft.ifft2(F / (np.abs(F) + 1e-12)).real
    iy, ix = np.unravel_index(np.argmax(cc2), cc2.shape)
    ny_, nx_ = cc2.shape
    return (float(ix - nx_ if ix > nx_ // 2 else ix),
            float(iy - ny_ if iy > ny_ // 2 else iy))


_cands = []
for _sign in (1.0, -1.0):
    _bp = _sign * meas_bp
    for _rot_c in sorted({round(_rot_pos, 4), round(_rot_neg, 4)}):
        _dxc, _dyc = phase_shift_px(_bp, _rot_c)
        _cands.append((ncc([_dxc, _dyc, _rot_c, SCALE_FROM_VOXELS], _bp),
                       _sign, _rot_c, _dxc, _dyc))
_cands.sort(key=lambda t: -t[0])
for _n, _s, _r, _dxc, _dyc in _cands:
    print(f"   cand: sign={_s:+.0f} rot={_r:+.3f}deg dx={_dxc*ALIGN_BIN:+.0f} "
          f"dy={_dyc*ALIGN_BIN:+.0f} -> NCC={_n:+.4f}")
_ncc_c, CONTRAST_SIGN, _rot_est, _dx_est, _dy_est = _cands[0]
meas_bp = CONTRAST_SIGN * meas_bp
print(f"[pick] contrast sign={CONTRAST_SIGN:+.0f}, rot={_rot_est:+.3f}deg, "
      f"dx={_dx_est*ALIGN_BIN:+.1f}px, dy={_dy_est*ALIGN_BIN:+.1f}px (full-res)")
if CONTRAST_SIGN < 0:
    print(
        "[note] CONTRAST_SIGN=-1 means the measured reconstruction has an "
        "inverted phase sign relative to the corrected +D simulation; inspect "
        "the phase/sign convention in the real pipeline. MEAS_DISPLAY_INVERT "
        "is retained only for display."
    )

_p_init = np.array([_dx_est, _dy_est, _rot_est, SCALE_FROM_VOXELS])
_ncc_init = ncc(_p_init)
print(f"[ncc] geometric init: {_ncc_init:.4f}  ({time.time()-_t0:.1f}s)")

_p_init[3] = float(np.clip(_p_init[3], SCALE_BOUNDS[0], SCALE_BOUNDS[1]))

_scale_scan_values = np.arange(0.94, 1.0601, 0.005)
_scale_scan_ncc = []
print("[scan] scale  NCC")
for _scale_scan in _scale_scan_values:
    _scan_ncc = ncc([_dx_est, _dy_est, _rot_est, float(_scale_scan)])
    _scale_scan_ncc.append(_scan_ncc)
    print(f"[scan] {float(_scale_scan):.3f}  {_scan_ncc:+.5f}")
_scale_scan_best_i = int(np.argmax(_scale_scan_ncc))
_scale_scan_best = float(_scale_scan_values[_scale_scan_best_i])
_scale_scan_best_ncc = float(_scale_scan_ncc[_scale_scan_best_i])
print(f"[scan] best scale={_scale_scan_best:.5f} NCC={_scale_scan_best_ncc:+.5f}")

_scale_scan_best_clamped = float(np.clip(_scale_scan_best, SCALE_BOUNDS[0], SCALE_BOUNDS[1]))
if _scale_scan_best_clamped != _scale_scan_best:
    print(
        f"[scan] scan-best scale={_scale_scan_best:.5f} lies outside "
        f"SCALE_BOUNDS={SCALE_BOUNDS} -> clamped to "
        f"{_scale_scan_best_clamped:.5f} before use."
    )
_scale_scan_best_clamped_ncc = ncc([_dx_est, _dy_est, _rot_est, _scale_scan_best_clamped])

if _scale_scan_best_clamped_ncc > _ncc_init + 0.005:
    _p_init[3] = _scale_scan_best_clamped
    _ncc_init = _scale_scan_best_clamped_ncc
    print(
        f"[scan] starting refine from (clamped) scan scale="
        f"{_scale_scan_best_clamped:.5f} (NCC improvement="
        f"{_scale_scan_best_clamped_ncc - ncc([_dx_est, _dy_est, _rot_est, SCALE_FROM_VOXELS]):+.5f})"
    )

if REFINE:
    _shift_bound_binned = SHIFT_BOUND_PX / ALIGN_BIN
    _bounds = [
        (_dx_est - _shift_bound_binned, _dx_est + _shift_bound_binned),
        (_dy_est - _shift_bound_binned, _dy_est + _shift_bound_binned),
        (_rot_est - ROT_BOUND_DEG, _rot_est + ROT_BOUND_DEG),
        tuple(SCALE_BOUNDS),
    ]
    _res = minimize(lambda p: -ncc(p), _p_init, method="Powell", bounds=_bounds,
                    options=dict(xtol=1e-3, ftol=1e-4, maxiter=60, maxfev=600))
    _p_best, _ncc_best = (_res.x, -_res.fun) if -_res.fun > _ncc_init else (_p_init, _ncc_init)
else:
    _p_best, _ncc_best = _p_init, _ncc_init

_bound_param_names = ("dx_binned", "dy_binned", "rot_deg", "scale")
for _param_name, _param_value, (_bound_lo, _bound_hi) in zip(
    _bound_param_names, _p_best, _bounds if REFINE else (
        (float("-inf"), float("inf")), (float("-inf"), float("inf")),
        (float("-inf"), float("inf")), tuple(SCALE_BOUNDS),
    )
):
    for _bound_side, _bound_value in (("lower", _bound_lo), ("upper", _bound_hi)):
        if np.isfinite(_bound_value) and abs(_param_value - _bound_value) < (
            1e-3 * max(1.0, abs(_bound_value))
        ):
            print(
                f"[WARN] refine parameter {_param_name}={_param_value:.6g} "
                f"is constrained by its {_bound_side} bound "
                f"{_bound_value:.6g}; relax the bounds or check the measured "
                "nominal pixel size / magnification."
            )

MEAS_ALIGN_DX_PX = float(_p_best[0]) * ALIGN_BIN
MEAS_ALIGN_DY_PX = float(_p_best[1]) * ALIGN_BIN
MEAS_ALIGN_ROT_DEG = float(_p_best[2])
MEAS_ALIGN_SCALE = float(_p_best[3])
print(f"[align] dx={MEAS_ALIGN_DX_PX:+.2f}px dy={MEAS_ALIGN_DY_PX:+.2f}px "
      f"rot={MEAS_ALIGN_ROT_DEG:+.4f}deg scale={MEAS_ALIGN_SCALE:.5f} | "
      f"NCC={_ncc_best:.4f} | {time.time()-_t0:.1f}s")
if _ncc_best < 0.3:
    print("[warn] low NCC -> check mirror/flip, the phase contrast sign, or the centers.")

measured_img = warp(measured_raw, MEAS_C, SIM_C, MEAS_ALIGN_ROT_DEG, MEAS_ALIGN_SCALE,
                    MEAS_ALIGN_DX_PX, MEAS_ALIGN_DY_PX, target_shape, order=3)
measured_img_signed = CONTRAST_SIGN * measured_img
_yy_full, _xx_full = np.mgrid[0:target_shape[0], 0:target_shape[1]].astype(np.float64)
_r_full = np.hypot(_xx_full - SIM_CENTER_X_PX, _yy_full - SIM_CENTER_Y_PX)
_full_star_mask = (
    (_r_full >= 0.12 * _r_edge_px) & (_r_full <= 1.02 * _r_edge_px)
    & np.isfinite(data0_cropped) & np.isfinite(measured_img_signed)
)
_full_sim_vec = data0_cropped[_full_star_mask].astype(np.float64)
_full_meas_vec = measured_img_signed[_full_star_mask].astype(np.float64)
if _full_sim_vec.size and _full_meas_vec.size:
    _full_sim_vec -= _full_sim_vec.mean()
    _full_meas_vec -= _full_meas_vec.mean()
    _full_ncc = float(
        np.dot(_full_sim_vec, _full_meas_vec)
        / (np.linalg.norm(_full_sim_vec) * np.linalg.norm(_full_meas_vec) + 1e-12)
    )
else:
    _full_ncc = float("nan")
print(
    f"[ncc] full-resolution star-mask NCC (annulus 0.12-1.02 r_edge, "
    f"UNSCALED sim, diagnostic only) = {_full_ncc:+.5f}"
)
COMMON_CENTER_X_PX = SIM_CENTER_X_PX
COMMON_CENTER_Y_PX = SIM_CENTER_Y_PX
voxelsize_meas = voxelsize        # now a common grid
print(f"[arcs] center of semicircles/arcs (object center on the output grid) = "
      f"({COMMON_CENTER_X_PX:.1f}, {COMMON_CENTER_Y_PX:.1f})")

_th_r = np.deg2rad(-MEAS_ALIGN_ROT_DEG)
MEAS_CENTER_TRUE_X = MEAS_CENTER_X_PX - (np.cos(_th_r) * MEAS_ALIGN_DX_PX +
                                         np.sin(_th_r) * MEAS_ALIGN_DY_PX) / MEAS_ALIGN_SCALE
MEAS_CENTER_TRUE_Y = MEAS_CENTER_Y_PX - (-np.sin(_th_r) * MEAS_ALIGN_DX_PX +
                                         np.cos(_th_r) * MEAS_ALIGN_DY_PX) / MEAS_ALIGN_SCALE
if abs(MEAS_ALIGN_DX_PX) > 20 or abs(MEAS_ALIGN_DY_PX) > 20:
    print(f"[warn] large dx/dy -> the MEAS_CENTER you provided is wrong. The star "
          f"center in measured_raw is approximately ({MEAS_CENTER_TRUE_X:.1f}, "
          f"{MEAS_CENTER_TRUE_Y:.1f}) rather than ({MEAS_CENTER_X_PX:.1f}, {MEAS_CENTER_Y_PX:.1f}).")
print(f"[done] total {time.time()-_t0:.1f}s")

inv_scale = 1.0 / _scale_scan_best
sim_shrunk = warp(sim_raw, SIM_C, SIM_C, 0.0, inv_scale, 0.0, 0.0, target_shape, order=3)
data0_cropped = sim_shrunk
print(f"[scale-fix] free scale mismatch (scan estimate, before SCALE_BOUNDS lock) = "
      f"{_scale_scan_best:.5f}")
print(f"[scale-fix] inv_scale (sim shrink applied) = "
      f"{inv_scale:.5f}  ({(1-inv_scale)*100:+.2f}%)")
print("[scale-fix] data0_cropped -> sim_shrunk (this is now used for the "
      "panels and the circular/arc profiles below)")

_full_sim_vec2 = data0_cropped[_full_star_mask].astype(np.float64)
if _full_sim_vec2.size and _full_meas_vec.size:
    _full_sim_vec2 = _full_sim_vec2 - _full_sim_vec2.mean()
    _full_ncc = float(
        np.dot(_full_sim_vec2, _full_meas_vec)
        / (np.linalg.norm(_full_sim_vec2) * np.linalg.norm(_full_meas_vec) + 1e-12)
    )
    print(
        f"[ncc] full-resolution star-mask NCC (annulus 0.12-1.02 r_edge, "
        f"SCALE-CORRECTED sim) = {_full_ncc:+.5f}"
    )

ARC_MARGIN_DEG = 10.0
_n_samples = 2000
_theta_deg = np.linspace(LINE_ANGLE_DEG - ARC_MARGIN_DEG, LINE_ANGLE_DEG + ARC_MARGIN_DEG, _n_samples)
_theta_rad = np.deg2rad(_theta_deg)
_theta_rel_deg = _theta_deg - LINE_ANGLE_DEG


def arc_px_from_center(cx, cy, voxelsize_m, radius_um, theta_rad):
    r_px = radius_um * 1e-6 / voxelsize_m
    return cx + r_px * np.cos(theta_rad), cy + r_px * np.sin(theta_rad)


_meas_filled = np.nan_to_num(measured_img)


def profile_norm(p, mode="zscore"):
    p = np.asarray(p, dtype=np.float64)
    if mode == "zscore":
        return (p - np.nanmean(p)) / (np.nanstd(p) + 1e-12)
    if mode == "minmax":
        lo, hi = np.nanmin(p), np.nanmax(p)
        return (p - lo) / (hi - lo + 1e-12)
    return p / np.nanmean(p)


PROFILE_NORM = "zscore"      # "zscore" | "minmax" | "mean"
_sgn = CONTRAST_SIGN


def _normalize_for_overlay(img, p_lo=2, p_hi=98):
    lo, hi = np.nanpercentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


MEAS_DISPLAY_INVERT = CONTRAST_SIGN < 0
_meas_disp = -measured_img if MEAS_DISPLAY_INVERT else measured_img

_sim_norm_img = _normalize_for_overlay(data0_cropped)
_meas_norm_img = np.nan_to_num(_normalize_for_overlay(_meas_disp))

_overlay_rgb = np.zeros((*target_shape, 3), dtype=np.float32)
_overlay_rgb[..., 0] = _sim_norm_img
_overlay_rgb[..., 1] = _meas_norm_img

CONCENTRIC_FRACTIONS = np.array(CONCENTRIC_PERCENTS) / 100.0
N_CONCENTRIC = len(CONCENTRIC_FRACTIONS)

_mtf_params = globals().get("MTF_PARAMS", {})
_arc_exclusion_ranges = _mtf_params.get(
    "exclude_radius_ranges", MTF_EXCLUDE_RADIUS_RANGES if "MTF_EXCLUDE_RADIUS_RANGES" in globals() else ()
)
print("[arcs] radius checks:")
print("  percent    r_um       r_px  in_exclusion")
for _percent in CONCENTRIC_PERCENTS:
    _r_um_check = SIM_EDGE_RADIUS_UM * float(_percent) / 100.0
    _r_px_check = _r_um_check * 1e-6 / voxelsize
    _in_exclusion = any(
        float(_lo) <= _r_px_check <= float(_hi) for _lo, _hi in _arc_exclusion_ranges
    )
    print(
        f"  {_percent:7.1f}  {_r_um_check:7.2f}  {_r_px_check:8.1f}  "
        f"{'yes' if _in_exclusion else 'no':>12s}"
    )
    if _in_exclusion:
        print(
            f"[WARN] arc radius {_percent:g}% ({_r_px_check:.1f} px) "
            f"falls inside MTF exclusion range(s) {_arc_exclusion_ranges}."
        )

CIRCLE_LW = 3.0
CIRCLE_ALPHA = 0.9

_theta_half_deg = np.linspace(180, 360, 500)
_theta_half_rad = np.deg2rad(_theta_half_deg)

fig_panels, axs_panels = plt.subplots(1, 3, figsize=(3 * PANEL_SIZE[0], PANEL_SIZE[1]))

_panel_imgs = [
    (_overlay_rgb, "Overlay (red=Sim scale-corrected, green=Meas)", True),
    (_sim_norm_img, "Simulation (scale-corrected)", False),
    (_meas_norm_img, "Measured", False),
]

for _ax, (_img, _title, _is_rgb) in zip(axs_panels, _panel_imgs):
    _ax.imshow(_img, origin="upper", cmap=None if _is_rgb else "gray")
    for _i, _frac in enumerate(CONCENTRIC_FRACTIONS):
        _r_um = SIM_EDGE_RADIUS_UM * _frac
        _x_s, _y_s = arc_px_from_center(SIM_CENTER_X_PX, SIM_CENTER_Y_PX, voxelsize, _r_um, _theta_half_rad)
        _ax.plot(_x_s, _y_s, color="red", lw=CIRCLE_LW, alpha=CIRCLE_ALPHA, ls="--",
                label="Simulation" if _i == 0 else "_nolegend_")
        _x_m, _y_m = arc_px_from_center(COMMON_CENTER_X_PX, COMMON_CENTER_Y_PX, voxelsize, _r_um, _theta_half_rad)
        _ax.plot(_x_m, _y_m, color="lime", lw=CIRCLE_LW, alpha=CIRCLE_ALPHA, ls=":",
                label="Measured" if _i == 0 else "_nolegend_")
        _ax.text(_x_s[len(_x_s) // 2], _y_s[len(_y_s) // 2],
                 f" {CONCENTRIC_PERCENTS[_i]}%", color="white", fontsize=8,
                 ha="center", va="bottom")
    _ax.plot(SIM_CENTER_X_PX, SIM_CENTER_Y_PX, "+", color="white",
            markersize=10, markeredgewidth=2)
    _ax.set_title(_title, fontsize=10)
    _ax.axis("off")

axs_panels[0].legend(loc="upper right", fontsize=8, facecolor="black",
                     edgecolor="none", labelcolor="white")
fig_panels.suptitle(f"Concentric semicircles (0-180 deg), % of edge radius = "
                    f"{CONCENTRIC_PERCENTS} | rot={MEAS_ALIGN_ROT_DEG:+.3f} deg, "
                    f"meas_scale={MEAS_ALIGN_SCALE:.4f}, sim inv_scale={inv_scale:.4f}",
                    fontsize=11)
plt.tight_layout()
plt.show()

fig_prof, axs_prof = plt.subplots(N_CONCENTRIC, 1,
                                  figsize=(PANEL_SIZE[0], PANEL_SIZE[1] * N_CONCENTRIC),
                                  sharex=True)
if N_CONCENTRIC == 1:
    axs_prof = [axs_prof]

_fit_rows = []

for _i, _frac in enumerate(CONCENTRIC_FRACTIONS):
    _r_um = SIM_EDGE_RADIUS_UM * _frac
    _x_s, _y_s = arc_px_from_center(SIM_CENTER_X_PX, SIM_CENTER_Y_PX, voxelsize, _r_um, _theta_half_rad)
    _ps = map_coordinates(data0_cropped, [_y_s, _x_s], order=1, mode="nearest")
    _x_m, _y_m = arc_px_from_center(COMMON_CENTER_X_PX, COMMON_CENTER_Y_PX, voxelsize, _r_um, _theta_half_rad)
    _pm = map_coordinates(_meas_filled, [_y_m, _x_m], order=1, mode="nearest")
    _pm_signed = _sgn * _pm

    _ps_z = profile_norm(_ps, "zscore")
    _pm_z = profile_norm(_pm_signed, "zscore")
    _arc_finite = np.isfinite(_ps_z) & np.isfinite(_pm_z)
    _n_finite = int(np.count_nonzero(_arc_finite))

    if (
        _n_finite > 1
        and np.std(_ps_z[_arc_finite]) > 1e-12
        and np.std(_pm_z[_arc_finite]) > 1e-12
    ):
        _ps_f, _pm_f = _ps_z[_arc_finite], _pm_z[_arc_finite]
        _arc_corr = float(
            np.dot(_ps_f, _pm_f) / np.sqrt(np.dot(_ps_f, _ps_f) * np.dot(_pm_f, _pm_f))
        )
        _arc_r2 = _arc_corr ** 2
        _arc_rmse = float(np.sqrt(np.mean((_ps_f - _pm_f) ** 2)))
        _ps_raw_std = float(np.std(_ps[_arc_finite]))
        _pm_raw_std = float(np.std(_pm_signed[_arc_finite]))
        _amp_ratio = (_pm_raw_std / _ps_raw_std) if _ps_raw_std > 1e-12 else float("nan")
    else:
        _arc_corr = _arc_r2 = _arc_rmse = _amp_ratio = float("nan")

    print(
        f"[fit] {CONCENTRIC_PERCENTS[_i]:g}% radius (r={_r_um:.1f}um): "
        f"Pearson r={_arc_corr:+.4f}  R^2={_arc_r2:.4f}  "
        f"RMSE(z)={_arc_rmse:.4f}  amp.ratio(meas/sim)={_amp_ratio:.3f}"
    )
    _fit_rows.append(dict(
        percent=CONCENTRIC_PERCENTS[_i], r_um=_r_um, n_samples=_n_finite,
        pearson_r=_arc_corr, r2=_arc_r2, rmse_z=_arc_rmse, amp_ratio=_amp_ratio,
    ))

    axs_prof[_i].plot(_theta_half_deg, profile_norm(_ps, PROFILE_NORM),
                      color="tab:blue", lw=2.0, label="Simulation (scale-corrected)")
    axs_prof[_i].plot(_theta_half_deg, profile_norm(_pm_signed, PROFILE_NORM),
                      color="tab:red", lw=2.0, label="Measured")
    axs_prof[_i].set_ylabel(f"{PROFILE_NORM} intensity")
    axs_prof[_i].set_title(
        f"r={_r_um:.1f}um ({CONCENTRIC_PERCENTS[_i]}% of edge radius) -- "
        f"Pearson r={_arc_corr:+.4f} (R^2={_arc_r2:.3f})  "
        f"RMSE(z)={_arc_rmse:.3f}  amp.ratio={_amp_ratio:.3f}",
        fontsize=10,
    )
    axs_prof[_i].grid(True, alpha=0.4)
    axs_prof[_i].legend(fontsize=8)
axs_prof[-1].set_xlabel("angle [deg] (0-180)")
plt.tight_layout()
plt.show()

print()
print("=" * 78)
print("FIT SUMMARY (simulation vs measured, per concentric radius)")
print("=" * 78)
print(f"{'%':>5s} {'r[um]':>8s} {'n':>6s} {'Pearson r':>11s} {'R^2':>7s} "
      f"{'RMSE(z)':>9s} {'amp.ratio':>10s}")
for _row in _fit_rows:
    print(f"{_row['percent']:5g} {_row['r_um']:8.1f} {_row['n_samples']:6d} "
          f"{_row['pearson_r']:+11.4f} {_row['r2']:7.4f} "
          f"{_row['rmse_z']:9.4f} {_row['amp_ratio']:10.3f}")
_valid_r = [r["pearson_r"] for r in _fit_rows if np.isfinite(r["pearson_r"])]
_valid_rmse = [r["rmse_z"] for r in _fit_rows if np.isfinite(r["rmse_z"])]
_valid_amp = [r["amp_ratio"] for r in _fit_rows if np.isfinite(r["amp_ratio"])]
if _valid_r:
    print("-" * 78)
    print(f"{'mean':>5s} {'':>8s} {'':>6s} "
          f"{np.mean(_valid_r):+11.4f} {np.mean(_valid_r)**2:7.4f} "
          f"{np.mean(_valid_rmse):9.4f} {np.mean(_valid_amp):10.3f}")
print("=" * 78)
print(
    "[fit] interpretation: Pearson r/R^2 measure SHAPE agreement only "
    "(scale-free). RMSE(z) measures overall residual after both curves "
    "are z-score normalised (0=identical). amp.ratio=std(measured)/"
    "std(sim) in RAW units -- amp.ratio<1 means the measured curve has "
    "LESS contrast (more blur) than the simulation at that radius; "
    "amp.ratio>1 means the opposite. A high Pearson r together with an "
    "amp.ratio far from 1.0 indicates a blur/contrast mismatch rather "
    "than a genuine misalignment."
)

print(
    f"[summary] case={SIM_CASE_KEY} "
    f"sim_center=({SIM_C[0]:.1f},{SIM_C[1]:.1f}) "
    f"meas_center=({MEAS_C[0]:.1f},{MEAS_C[1]:.1f}) "
    f"dx={MEAS_ALIGN_DX_PX:+.2f}px dy={MEAS_ALIGN_DY_PX:+.2f}px "
    f"rot={MEAS_ALIGN_ROT_DEG:+.4f}deg meas_scale={MEAS_ALIGN_SCALE:.5f} "
    f"sim_inv_scale={inv_scale:.5f} contrast_sign={CONTRAST_SIGN:+.0f} "
    f"binned_ncc={_ncc_best:+.5f} full_ncc={_full_ncc:+.5f}"
)


# =============================================================================
# SECTION 12 -- CNR SUMMARY across ALL reconstruction methods and cases,
# merged into the combined resolution table.
# =============================================================================
_REQUIRED_CNR_ALL_GLOBALS = ("compute_cnr", "multi_roi_pairs", "_methods_progression")
_missing_cnr_all = [n for n in _REQUIRED_CNR_ALL_GLOBALS if n not in globals()]
if _missing_cnr_all:
    raise RuntimeError(
        "This section requires compute_cnr, multi_roi_pairs and "
        "_methods_progression to already be defined; missing: "
        f"{', '.join(_missing_cnr_all)}"
    )

cnr_rows_all_methods = []
for method_name, method_dict in _methods_progression.items():
    for label, img in method_dict.items():
        cnr_values = [compute_cnr(img, s_roi, b_roi)[0]
                      for s_roi, b_roi in multi_roi_pairs]
        cnr_rows_all_methods.append({
            "method": method_name,
            "case": label,
            "CNR_mean": float(np.mean(cnr_values)),
            "CNR_std": float(np.std(cnr_values)),
            "n_wedges": len(cnr_values),
        })

cnr_table_all_methods = pd.DataFrame(cnr_rows_all_methods)
print("=" * 78)
print("CNR SUMMARY -- all reconstruction methods, all cases")
print("=" * 78)
print(cnr_table_all_methods.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

resolution_table_all_methods_progression = resolution_table_all_methods_progression.merge(
    cnr_table_all_methods[["method", "case", "CNR_mean", "CNR_std"]],
    on=["method", "case"], how="left",
)
print()
print("=" * 78)
print("COMBINED TABLE: MTF10 + Modregger + CNR -- all methods, all cases")
print("=" * 78)
print(resolution_table_all_methods_progression.to_string(
    index=False, float_format=lambda x: f"{x:.2f}"))

_pivot = cnr_table_all_methods.pivot(index="case", columns="method", values="CNR_mean")
_pivot = _pivot.reindex(columns=list(_methods_progression.keys()))

fig, ax = plt.subplots(figsize=(10, 6))
_pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("CNR (mean over all wedges)")
ax.set_title("CNR by reconstruction method and case")
ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.4, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnr_all_methods.png"), dpi=130)
plt.show()
