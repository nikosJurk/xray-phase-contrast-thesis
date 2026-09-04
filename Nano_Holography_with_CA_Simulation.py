"""
===============================================================================
Single-distance coded-aperture nano-holography  --  MAIN SCRIPT
===============================================================================
Uses Viktor's reconstruction engine UNCHANGED:
    rec_sim.py, utils_sim.py, cuda_kernels_sim.py, chunking_sim.py
This script only builds the INPUTS (sample psi, coded aperture `code`, probe q,
scan positions) in physical units and feeds them to that engine.

Optical chain (sign of z1c selects CA position):
    [CA?]  ->  Focus  ->  [CA?]  ->  Sample  ->  Detector
    z1c < 0      : CA upstream of focus   (Viktor's default, ID16A-like)
    0 < z1c < z1 : CA between focus and sample
The propagation distances (distance, distancec) follow from z1, z1c, z2 and
are passed to the engine, which handles a signed distancec correctly.

Build/test order: Section 1 (imports) -> 2 (geometry/args) -> 3 (sample) ->
4 (coded aperture) -> 5 (probe) -> 6 (positions) -> 7 (forward sim) ->
8 (reconstruction) -> 9 (plots / resolution).  THIS FILE: sections 1-3.

Credits: All credit for the reconstruction engine (rec_sim.py, utils_sim.py,
cuda_kernels_sim.py, chunking_sim.py) and the underlying method goes to
Viktor Nikitin.
===============================================================================
"""

# =============================================================================
# SECTION 1 -- IMPORTS
# =============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- GPU stack (Viktor's engine). On the HPC these import; off-cluster they may
#     not. We import defensively so Sections 2-3 (CPU) can be checked anywhere. ---
try:
    import cupy as cp
    import pandas as pd
    import xraylib
    import cupyx.scipy.ndimage as gpu_ndimage
    from types import SimpleNamespace
    import warnings
    warnings.filterwarnings("ignore", message=".*peer.*")
    from utils_sim import *           # mshow, mshow_polar, mplot_positions, ...
    from rec_sim import Rec
    _HAVE_GPU = True
except Exception as _e:
    from types import SimpleNamespace
    _HAVE_GPU = False
    print(f"[imports] GPU/engine stack not available here ({_e}).")
    print("          Sections 2-3 (CPU) still run; Sections 7-8 need the HPC.")

# xraylib is also used on CPU for optical constants; provide a flag.
try:
    import xraylib as _xrl
    _HAVE_XRAYLIB = True
except Exception:
    _HAVE_XRAYLIB = False

# --- Optional: your NIST optical-constants pipeline (preferred source) ---
OC_SOURCE_DIR = "/dtu/3d-imaging-center/projects/2026_DANFIX_XHIST/analysis/Nis/oc_source"
OC_NIST_DIR = os.path.join(OC_SOURCE_DIR, "NIST/")
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
    Return (delta, beta). Priority: your NIST oc() -> xraylib -> error.
    For NIST, elements+atoms+density are used (your tester.ipynb convention).
    For xraylib, the compound string `material` + density are used.
    """
    if material == "vacuum":
        return 0.0, 0.0
    if _OC is not None and elements is not None and atoms is not None:
        lam_nm = 1.239841984 / E_keV
        n = _OC(lam_nm, density, atoms, elements, OC_NIST_DIR)[0]
        return float(1.0 - n.real), float(n.imag)
    if _HAVE_XRAYLIB:
        d = 1.0 - _xrl.Refractive_Index_Re(material, E_keV, density)
        b = _xrl.Refractive_Index_Im(material, E_keV, density)
        return float(d), float(b)
    raise RuntimeError("No optical-constants source available (NIST or xraylib).")


# =============================================================================
# SECTION 2 -- GEOMETRY & args
# =============================================================================
# ---- X-ray / detector ----
energy = 19.55                     # [keV] photon energy
wavelength = 1.239841984e-9 / energy   # [m]
detector_pixelsize = 0.55e-6    # [m] detector pixel pitch

# ---- grid sizes (test at Viktor's defaults first) ----
# ---- grid sizes ----
n      = 2688     # [px] detector image size (n x n) -- increased to fit
                   # the full Siemens star (arm_pairs=36, feat_max=5um)
ncode  = 4096      # [px] coded-aperture canvas (must exceed object FOV so it can shift)
pad    = 0        # [px] optional object padding
ex     = 0        # [px] extra border when extracting CA patches

# ---- distances (point-projection geometry) ----
focusToDetectorDistance = 1.555     # [m] focus -> detector
z1  =     0.125              # [m] focus -> sample 

# ---------------------------------------------------------------------------
# CA POSITION SWITCH -- set z1c to move the coded aperture:
#   z1c < 0        -> CA UPSTREAM of focus (Viktor default / ID16A); e.g. -17.5e-3
#   0 < z1c < z1   -> CA BETWEEN focus and sample;                    e.g.   5.0e-3
# The reconstruction engine handles the resulting signed distancec correctly.
z1c = 0.109                    # [m] focus -> CA (signed) [z1c = z1 - CA_distance]
# ---------------------------------------------------------------------------
z2  = focusToDetectorDistance - z1  # [m] sample -> detector

# sanity checks on the CA position
if z1c == 0:
    raise ValueError("z1c == 0 puts the CA exactly at the focus (degenerate).")
if 0 < z1c and z1c >= z1:
    raise ValueError(f"z1c ({z1c*1e3:.1f} mm) >= z1 ({z1*1e3:.1f} mm): CA would be "
                     f"at or behind the sample. Use 0 < z1c < z1.")

# effective propagation distances (thin-lens / Fresnel-scaling relations)
distance  = (z1 * z2) / focusToDetectorDistance      # sample <-> detector
distancec = (z1 - z1c) / (z1c / z1)                  # sample <-> CA (signed)

# magnifications and sample-plane pixel
magnification   = focusToDetectorDistance / z1       # sample -> detector
magnification_c = z1 / z1c                            # CA -> sample (signed)
voxelsize = abs(detector_pixelsize / magnification)  # [m/px] object-plane pixel

ca_position = "upstream of focus" if z1c < 0 else "between focus and sample"
print("=" * 62)
print("GEOMETRY")
print("=" * 62)
print(f"  Energy              : {energy} keV (lambda = {wavelength*1e12:.3f} pm)")
print(f"  CA position         : {ca_position} (z1c = {z1c*1e3:.2f} mm)")
print(f"  focus->sample z1    : {z1*1e3:.3f} mm")
print(f"  sample->detector z2 : {z2*1e3:.3f} mm")
print(f"  Magnification (det) : {magnification:.2f}")
print(f"  Magnification (CA)  : {magnification_c:.3f}")
print(f"  Sample pixel        : {voxelsize*1e9:.2f} nm")
print(f"  distance  (smp<->det): {distance*1e3:.4f} mm")
print(f"  distancec (smp<->CA) : {distancec*1e3:.4f} mm")
print("=" * 62)

# ---- pack args for Viktor's Rec engine ----
args = SimpleNamespace()
args.ngpus = 1
args.nchunk = 4
args.n = n
args.ncode = ncode
args.pad = pad
args.npsi = args.n + 2 * args.pad
args.ex = ex
args.npatch = args.npsi + 2 * args.ex
args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.distancec = distancec
args.energy = energy
# args.npos is set in Section 6 (after we build the position grid).

# Instantiate the reconstruction engine (GPU only).
if _HAVE_GPU:
    cl_rec = Rec(args)
else:
    cl_rec = None
    print("[rec] Rec engine not instantiated (no GPU here).")


# =============================================================================
# SECTION 3 -- SAMPLE: NTT-AT XRESO-50HC datasheet model
#              Ta absorber (500 nm) on a SiC membrane (200 nm)
# =============================================================================
# Extend the material library already defined in Section 1 with SiC
# (needed for the membrane). Keeps a single source of truth for materials.


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
    npsi, voxelsize_m, arm_pairs,
    feat_min_m, feat_max_m,
    absorber_material, absorber_thickness_m,
    membrane_material, membrane_thickness_m,
    k, E_keV,
    supersample=3,
    exclude_radius_ranges=(),
    exclude_angle_ranges=(),
):
    """Build a physically-modeled Siemens star: a SiC membrane everywhere,
    with Ta absorber wedges alternating with open (membrane-only) wedges.
    Square grid (npsi x npsi), matching the engine's convention."""
    ny = nx = npsi
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
feat_min = 50e-9               # [m] innermost feature (50 nm, datasheet)
ARM_PAIRS = 36                 # real Siemens-star target: 36 arm pairs
SUPERSAMPLE = 3
TA_THICKNESS = 500e-9          # [m] absorber pattern
SIC_THICKNESS = 200e-9         # [m] SiC membrane
SIM_STAR_OUTER_RADIUS_UM = 46.8

arm_pairs = ARM_PAIRS
dphi = np.pi / arm_pairs
feat_max = SIM_STAR_OUTER_RADIUS_UM * 1e-6 * dphi

# sanity check: star must fit inside the square npsi grid
_grid_half = args.npsi * voxelsize / 2.0
if SIM_STAR_OUTER_RADIUS_UM * 1e-6 > _grid_half:
    print(f"  [WARNING] outer radius {SIM_STAR_OUTER_RADIUS_UM:.2f} um exceeds "
          f"grid half-size {_grid_half*1e6:.2f} um -- star will be cropped. "
          f"Increase npsi or reduce SIM_STAR_OUTER_RADIUS_UM.")

USE_REAL_DEFECT_EXCLUSIONS = True
REAL_EXCLUDE_RADIUS_RANGES = ((55, 60), (120, 145), (250, 270), (505, 550))
REAL_EXCLUDE_ANGLE_RANGES = ((126, 146),)

if USE_REAL_DEFECT_EXCLUSIONS:
    EXCLUDE_RADIUS_RANGES = REAL_EXCLUDE_RADIUS_RANGES
    EXCLUDE_ANGLE_RANGES = REAL_EXCLUDE_ANGLE_RANGES
else:
    EXCLUDE_RADIUS_RANGES = ()
    EXCLUDE_ANGLE_RANGES = ()

k_wave = 2 * np.pi / wavelength

# NOTE: exclusions are passed as empty here -- the simulated object itself
# has no defects; EXCLUDE_* is only used later, for MTF/CNR analysis masks.
psi_np, sample_meta = make_siemens_star_physical(
    args.npsi, voxelsize, arm_pairs, feat_min, feat_max,
    absorber_material="Ta", absorber_thickness_m=TA_THICKNESS,
    membrane_material="SiC", membrane_thickness_m=SIC_THICKNESS,
    k=k_wave, E_keV=energy, supersample=SUPERSAMPLE,
    exclude_radius_ranges=REAL_EXCLUDE_RADIUS_RANGES, exclude_angle_ranges=EXCLUDE_ANGLE_RANGES,
)

mem_name, mem_d, mem_b = sample_meta["membrane"]
ta_name, ta_d, ta_b = sample_meta["ta"]
phi_ta = k_wave * ta_d * TA_THICKNESS
mu_t_ta = 2 * k_wave * ta_b * TA_THICKNESS

print("=" * 62)
print("SAMPLE (XRESO-50HC datasheet model)")
print("=" * 62)
print(f"  Array/pixel : {args.npsi} x {args.npsi}, {voxelsize*1e9:.2f} nm/px, "
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

# psi for the engine must be complex64 on GPU; keep a numpy copy here.
if _HAVE_GPU:
    psi = cp.asarray(psi_np)
else:
    psi = psi_np

# ---- quick ground-truth plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(np.abs(psi_np), cmap="gray")
ax[0].set_title(f"|psi| amplitude  (dark = {ta_name} absorber)")
ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]")
im = ax[1].imshow(np.angle(psi_np), cmap="twilight")
ax[1].set_title("arg(psi) phase [rad]")
ax[1].set_xlabel("x [px]"); ax[1].set_ylabel("y [px]")
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
plt.suptitle("Section 3: ground-truth Siemens star", fontweight="bold")
plt.tight_layout()
plt.show()
# =============================================================================
# SECTION 4 -- CODED APERTURE  (random binary pattern, your parameters)
# =============================================================================
# The CA is a random binary mask of `material` of thickness CA_THICKNESS, on a
# canvas of ncode x ncode pixels (the canvas must exceed the object FOV so the
# CA can be shifted across it). Its complex transmission is built from the
# material's delta/beta (NIST/xraylib), then optionally rotated and edge-smoothed
# to mimic a real fabricated aperture.
#
# Pixel convention: the CA lives in the SAME object-plane sampling as the sample
# (voxelsize m/px), which is what Viktor's engine assumes for S()/Dc().
#
# Parameters you tune:
CA_MATERIAL    = "Au"        # CA material: "Au", "Si", "Ta", ... (from library)
CA_THICKNESS   = 550e-9      # [m] CA path length (independent of the sample)
CA_BIT_SIZE    = 12e-6        # [m] size of one random square (feature pitch)
CA_FILL_FRAC   = 0.55        # open fraction of the random binary pattern (0..1)
CA_SEED        = 10          # RNG seed for reproducible pattern

CA_ROTATE      = True        # apply a rotation to the pattern?
CA_ROTATE_DEG  = 45.0        # [deg] rotation angle (Viktor uses 45)

CA_SMOOTH      = True        # apply gentle Fourier low-pass to smooth edges?
CA_SMOOTH_STR  = 10.0        # smoothing strength (larger = smoother edges)


def build_coded_aperture(ncode, voxelsize_m, bit_size_m, material, thickness_m,
                         E_keV, wavelength_m, fill_frac=0.55, seed=10,
                         rotate=True, rotate_deg=45.0,
                         smooth=True, smooth_strength=10.0):
    """
    Build a complex coded-aperture transmission `code` (ncode x ncode, complex64).

    Steps: random binary pattern at the requested bit size -> upsample to the
    canvas -> material complex factor (delta/beta * thickness) -> optional
    rotation -> optional Fourier low-pass -> exp(i*k*OPD).
    Returns (code, info_dict).
    """
    if _HAVE_GPU:
        import numpy as _np
        from scipy import ndimage as _nd      # CPU scipy for build; fine pre-GPU
    else:
        import numpy as _np
        from scipy import ndimage as _nd

    rng = _np.random.default_rng(seed)

    # 1) random binary pattern at 2x canvas, then crop -- gives tiling variety
    nbig = 2 * ncode
    big = rng.random((nbig, nbig)) < fill_frac

    # 2) how many canvas pixels per bit, and the small pattern size
    #    bit_size is physical; the CA pixel is voxelsize, so bit = bit_size/voxelsize px
    bit_px = max(1.0, bit_size_m / voxelsize_m)
    nsmall = int(max(2, round(ncode / bit_px)))
    c0 = nbig // 2
    half = nsmall // 2
    small = big[c0 - half:c0 + half, c0 - half:c0 + half].astype(_np.float32)

    # 3) upsample (nearest, tiled) to the full canvas
    scale = ncode / max(1, small.shape[0])
    patt = _nd.zoom(small, zoom=scale, order=0, grid_mode=True,
                    mode="grid-wrap").astype(bool)
    patt = patt[:ncode, :ncode]
    if patt.shape != (ncode, ncode):   # pad if off-by-one
        pp = _np.zeros((ncode, ncode), bool)
        pp[:patt.shape[0], :patt.shape[1]] = patt
        patt = pp

    # 4) material complex factor; thickness in object-plane pixels
    if material == "vacuum":
        delta, beta = 0.0, 0.0
    else:
        spec = SAMPLE_MATERIAL_LIBRARY[material]
        delta, beta = get_delta_beta(material, spec["density"], E_keV,
                                     spec["elements"], spec["atoms"])
    thickness_px = thickness_m / voxelsize_m
    # complex refractive factor where the pattern is "on" (path length present)
    Rill = patt.astype(_np.float32) * (-delta + 1j * beta) * thickness_px

    # 5) optional rotation
    if rotate:
        Rill = _nd.rotate(Rill, rotate_deg, axes=(1, 0), reshape=False,
                          order=3, mode="reflect", prefilter=True)

    # 6) optional Fourier low-pass to smooth edges
    if smooth:
        fy = (_np.arange(-ncode // 2, ncode // 2) / (2.0 * ncode)).astype(_np.float32)
        vy, vx = _np.meshgrid(fy, fy, indexing="ij")
        gauss = _np.exp(-smooth_strength * (vx**2 + vy**2)).astype(_np.float32)
        F = _np.fft.fftshift(_np.fft.fftn(_np.fft.fftshift(Rill)))
        Rill = _np.fft.ifftshift(_np.fft.ifftn(_np.fft.ifftshift(F * gauss)))
        Rill = Rill.astype(_np.complex64)

    # 7) complex transmission: code = exp(i * k0 * OPD), OPD = Re(Rill)*voxelsize
    k0 = 2.0 * _np.pi / wavelength_m
    code = _np.exp(1j * Rill * voxelsize_m * k0).astype(_np.complex64)

    info = dict(material=material, delta=delta, beta=beta,
                bit_size_m=bit_size_m, bit_px=bit_px, nsmall=nsmall,
                fill_frac=float(patt.mean()), thickness_m=thickness_m,
                rotate=rotate, rotate_deg=rotate_deg,
                smooth=smooth, amp_range=(float(_np.abs(code).min()),
                                          float(_np.abs(code).max())),
                phase_range=(float(_np.angle(code).min()),
                             float(_np.angle(code).max())))
    return code, info


code_np, ca_info = build_coded_aperture(
    args.ncode, voxelsize, CA_BIT_SIZE, CA_MATERIAL, CA_THICKNESS,
    energy, wavelength, fill_frac=CA_FILL_FRAC, seed=CA_SEED,
    rotate=CA_ROTATE, rotate_deg=CA_ROTATE_DEG,
    smooth=CA_SMOOTH, smooth_strength=CA_SMOOTH_STR)

print("=" * 62)
print("CODED APERTURE")
print("=" * 62)
print(f"  Material            : {CA_MATERIAL} "
      f"(delta={ca_info['delta']:.3e}, beta={ca_info['beta']:.3e})")
print(f"  Thickness           : {CA_THICKNESS*1e9:.0f} nm")
print(f"  Canvas              : {args.ncode} x {args.ncode} px "
      f"(pixel = {voxelsize*1e9:.2f} nm)")
print(f"  Bit size            : {CA_BIT_SIZE*1e6:.2f} um "
      f"(= {ca_info['bit_px']:.1f} px)")
print(f"  Fill fraction       : {ca_info['fill_frac']:.3f}")
print(f"  Rotation            : {'ON %g deg' % CA_ROTATE_DEG if CA_ROTATE else 'OFF'}")
print(f"  Edge smoothing      : {'ON (str=%g)' % CA_SMOOTH_STR if CA_SMOOTH else 'OFF'}")
print(f"  Amplitude range     : {ca_info['amp_range'][0]:.4f} -> "
      f"{ca_info['amp_range'][1]:.4f}  (~1 for a phase CA)")
print(f"  Phase range [rad]   : {ca_info['phase_range'][0]:.4f} -> "
      f"{ca_info['phase_range'][1]:.4f}")
print("=" * 62)

# code for the engine (complex64). Keep a numpy copy for plotting.
if _HAVE_GPU:
    code = cp.asarray(code_np)
else:
    code = code_np

# ---- plot CA amplitude & phase ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(np.abs(code_np), cmap="gray")
ax[0].set_title("|code| amplitude")
ax[0].set_xlabel("x [CA px]"); ax[0].set_ylabel("y [CA px]")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.angle(code_np), cmap="gray")
ax[1].set_title("arg(code) phase [rad]")
ax[1].set_xlabel("x [CA px]"); ax[1].set_ylabel("y [CA px]")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.suptitle(f"Section 4: coded aperture ({CA_MATERIAL}, "
             f"{CA_THICKNESS*1e9:.0f} nm)", fontweight="bold")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 5 -- PROBE  (synthetic focused illumination, or flat)
# =============================================================================
# The probe q is the illumination wavefield at the object plane (npsi x npsi).
# In the forward model D(Dc(S(code)*q)*psi) it multiplies the shifted CA, so it
# (together with the CA) sets the structured illumination diversity that makes
# coded-aperture holography work in the hard far-field regime.
#
# USE_PROBE switch:
#   True  -> synthetic focused probe: smooth Gaussian-ish amplitude envelope
#            (beam brightest in the center) plus a smooth structured phase
#            (wavefront curvature + mild aberration). Mimics a real focused beam.
#   False -> flat illumination, q = 1 everywhere. Tests reconstruction with the
#            CA providing ALL the structured-illumination diversity (no probe).
#
# Note: the reconstruction also RE-ESTIMATES the probe (vars["q"] is optimized),
# so the exact synthetic shape is not critical -- this is a fair comparison of
# "structured focused probe + CA" vs "flat illumination + CA".
USE_PROBE      = False
PROBE_WAIST_FR = 0.35      # Gaussian amplitude waist as fraction of array (if USE_PROBE)
PROBE_CURV     = 6.0       # parabolic phase curvature strength (wavefront curvature)
PROBE_ABERR    = 0.6       # amplitude of smooth random aberration in the phase [rad]
PROBE_SEED     = 7


def make_synthetic_probe(npsi, use_probe=True, waist_frac=0.35,
                         curvature=6.0, aberr=0.6, seed=7, smooth_strength=200.0):
    """
    Synthetic complex probe q (npsi x npsi, complex64).

    use_probe=False -> returns ones (flat illumination).
    use_probe=True  -> Gaussian amplitude envelope * exp(i*phase), where phase =
                       parabolic curvature + smooth low-passed random aberration.
    """
    if not use_probe:
        return np.ones((npsi, npsi), dtype=np.complex64)

    yy, xx = np.indices((npsi, npsi), dtype=np.float64)
    c = (npsi - 1) / 2.0
    x = (xx - c) / npsi
    y = (yy - c) / npsi
    rr2 = x * x + y * y

    # amplitude: Gaussian envelope (focused beam, bright center)
    w = waist_frac
    amp = np.exp(-rr2 / (2.0 * w * w))

    # phase: parabolic wavefront curvature + smooth random aberration
    phase = curvature * rr2 * (2 * np.pi)   # defocus-like curvature
    rng = np.random.default_rng(seed)
    rough = rng.standard_normal((npsi, npsi)).astype(np.float32)
    # low-pass the random field so the aberration is smooth, not pixel noise
    fy = (np.arange(-npsi // 2, npsi // 2) / npsi).astype(np.float32)
    vy, vx = np.meshgrid(fy, fy, indexing="ij")
    gauss = np.exp(-smooth_strength * (vx**2 + vy**2)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(rough)))
    smooth_rough = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(F * gauss))))
    smooth_rough /= (np.abs(smooth_rough).max() + 1e-12)
    phase = phase + aberr * smooth_rough

    q = (amp * np.exp(1j * phase)).astype(np.complex64)
    return q


q_np = make_synthetic_probe(args.npsi, use_probe=USE_PROBE,
                            waist_frac=PROBE_WAIST_FR, curvature=PROBE_CURV,
                            aberr=PROBE_ABERR, seed=PROBE_SEED)

print("=" * 62)
print("PROBE")
print("=" * 62)
if USE_PROBE:
    print(f"  Mode                : synthetic focused probe (USE_PROBE=True)")
    print(f"  Amplitude waist     : {PROBE_WAIST_FR:.2f} x array")
    print(f"  Curvature / aberr   : {PROBE_CURV:.1f} / {PROBE_ABERR:.2f} rad")
else:
    print(f"  Mode                : FLAT illumination, q=1 (USE_PROBE=False)")
print(f"  Array               : {args.npsi} x {args.npsi}")
print(f"  Amplitude range     : {np.abs(q_np).min():.4f} -> {np.abs(q_np).max():.4f}")
print(f"  Phase range [rad]   : {np.angle(q_np).min():.4f} -> {np.angle(q_np).max():.4f}")
print("=" * 62)

# probe for the engine (complex64). Keep numpy copy for plotting.
if _HAVE_GPU:
    q = cp.asarray(q_np)
else:
    q = q_np

# ---- plot probe amplitude & phase ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(np.abs(q_np), cmap="gray")
ax[0].set_title("|q| amplitude")
ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.angle(q_np), cmap="gray")
ax[1].set_title("arg(q) phase [rad]")
ax[1].set_xlabel("x [px]"); ax[1].set_ylabel("y [px]")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.suptitle(f"Section 5: probe ({'synthetic' if USE_PROBE else 'flat q=1'})",
             fontweight="bold")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 6 -- SCAN POSITIONS  (regular grid, optional jitter)
# =============================================================================
# The coded aperture is stepped across a regular CA_NPOS x CA_NPOS grid of
# lateral positions (replacing Viktor's random scatter). Positions are in CA
# pixels, centered at 0. Each position is one recorded hologram.
#
# Parameters:
#   CA_NPOS         : grid is CA_NPOS x CA_NPOS  -> total npos = CA_NPOS**2
#   CA_GRID_SPACING : [CA px] spacing between adjacent grid points
#   CA_JITTER       : [CA px] std-dev of a small random shift added to each
#                     point (0 = perfectly regular grid). A little jitter breaks
#                     the grid periodicity and reduces reconstruction artifacts.
#   CA_JITTER_SEED  : RNG seed for the jitter
#
# Engine split (as in Viktor's notebook):
#   ri = integer part of pos (fast patch extraction)
#   r  = fractional part of pos (subpixel, handled by Fourier phase ramps)
CA_NPOS         = 3         # 3 -> 3x3 = 9 positions; 7 -> 49; 15 -> 225
CA_GRID_SPACING = 600.0      # [CA px] spacing between grid points
CA_JITTER       = 20.0       # [CA px] random jitter std-dev (0 = exact grid)
CA_JITTER_SEED  = 3


def make_grid_positions(npos_side, spacing_px, jitter_px=0.0, seed=3):
    """
    Regular npos_side x npos_side grid of CA positions, centered at 0, in CA px.
    Optional Gaussian jitter. Returns pos (npos, 2) as [y, x], float32.
    """
    offs = (np.arange(npos_side) - (npos_side - 1) / 2.0) * spacing_px
    gy, gx = np.meshgrid(offs, offs, indexing="ij")
    pos = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float32)
    if jitter_px > 0:
        rng = np.random.default_rng(seed)
        pos = pos + rng.normal(0.0, jitter_px, size=pos.shape).astype(np.float32)
    return pos


pos = make_grid_positions(CA_NPOS, CA_GRID_SPACING, CA_JITTER, CA_JITTER_SEED)
npos = pos.shape[0]
args.npos = npos     # the engine needs this

# integer / fractional split for the engine
ri = np.rint(pos).astype(np.int32)        # (npos, 2)
r  = (pos - ri).astype(np.float32)        # (npos, 2)

# guard: make sure the grid stays inside the CA canvas (so shifts are valid)
max_abs = np.abs(pos).max()
canvas_half = args.ncode / 2.0 - args.npsi / 2.0
print("=" * 62)
print("SCAN POSITIONS (coded aperture)")
print("=" * 62)
print(f"  Grid                : {CA_NPOS} x {CA_NPOS} = {npos} positions")
print(f"  Spacing             : {CA_GRID_SPACING:.1f} CA px "
      f"(= {CA_GRID_SPACING*voxelsize*1e6:.3f} um)")
print(f"  Jitter (std)        : {CA_JITTER:.1f} CA px"
      f"{' (regular grid)' if CA_JITTER == 0 else ''}")
print(f"  Max |shift|         : {max_abs:.1f} CA px")
if max_abs > canvas_half:
    print(f"  WARNING: max shift {max_abs:.0f} px exceeds safe canvas half-range "
          f"{canvas_half:.0f} px. The CA may run out of canvas at extreme "
          f"positions -- increase ncode or reduce spacing/jitter.")
else:
    print(f"  Canvas headroom     : OK (safe half-range {canvas_half:.0f} px)")
print("=" * 62)

# ---- save the positions (npy + csv) ----
pos_save_dir = "outputs"
os.makedirs(pos_save_dir, exist_ok=True)
np.save(os.path.join(pos_save_dir, "ca_positions.npy"), pos)
np.savetxt(os.path.join(pos_save_dir, "ca_positions.csv"), pos,
           delimiter=",", header="y_shift_px,x_shift_px", comments="")
print(f"  Saved positions     : {pos_save_dir}/ca_positions.npy (+ .csv)")

# ---- plot the scan grid ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(pos[:, 1], pos[:, 0], "o-", ms=6, lw=0.5, color="tab:red")
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_aspect("equal", "box")
ax[0].set_xlabel("x shift [CA px]"); ax[0].set_ylabel("y shift [CA px]")
ax[0].set_title(f"Scan grid ({CA_NPOS}x{CA_NPOS} = {npos} positions)")
ax[0].grid(True, alpha=0.3)
# overlay positions on the CA phase
cy = args.ncode // 2; cx = args.ncode // 2
im = ax[1].imshow(np.angle(code_np), cmap="gray")
ax[1].scatter(cx + pos[:, 1], cy + pos[:, 0], s=40, c="red",
              edgecolors="yellow", linewidths=0.5)
ax[1].set_title("positions over CA phase\n(CA center = origin)")
ax[1].set_xlabel("x [CA px]"); ax[1].set_ylabel("y [CA px]")
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
plt.suptitle("Section 6: coded-aperture scan positions", fontweight="bold")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 7 -- FORWARD SIMULATION  (Viktor's engine: GPU)
# =============================================================================
# Generate the coded holograms and the flat-field reference using the engine's
# forward model:   field = D( Dc( S(ri,r,code) * q ) * psi )
#   S   : shift+extract the CA patch at each position
#   *q  : multiply by the probe
#   Dc  : propagate CA -> sample (distancec)
#   *psi: multiply by the sample
#   D   : propagate sample -> detector (distance)
# data[k] = |field[k]|^2 is the intensity recorded at position k.
#
# The flat-field reference uses an open aperture (code=1) and flat object
# (psi=1) with the same probe -- this is what the reconstruction uses to
# initialize the probe estimate.
#
# Requires the GPU engine (cl_rec). Run this on the HPC.
if cl_rec is None:
    raise RuntimeError("Section 7 needs the GPU engine (cl_rec). Run on the HPC.")

# coded measurements: complex field stack (npos, n, n) -> intensity
field = cl_rec.fwd(ri, r, code, psi, q)          # complex64 (npos, n, n)
data = np.abs(field) ** 2                         # intensity stack (float32)

# flat-field reference: open CA (code*0+1), flat object (psi*0+1), same probe
ref_field = cl_rec.fwd(ri, r, code * 0 + 1, psi * 0 + 1, q)
ref = (np.abs(ref_field) ** 2)[0]                 # (n, n)

# move to numpy for plotting/saving if on GPU
def _to_np(a):
    try:
        return a.get()
    except AttributeError:
        return np.asarray(a)

data_np = _to_np(data)
ref_np = _to_np(ref)

print("=" * 62)
print("FORWARD SIMULATION")
print("=" * 62)
print(f"  Probe mode          : {'synthetic' if USE_PROBE else 'flat (q=1)'}")
print(f"  Holograms (data)    : {data_np.shape}  (npos, n, n)")
print(f"  Flat reference (ref): {ref_np.shape}")
print(f"  Data intensity range: {data_np.min():.4f} -> {data_np.max():.4f}")
print("=" * 62)

# ---- save the simulated data + reference ----
sim_dir = "outputs"
os.makedirs(sim_dir, exist_ok=True)
np.save(os.path.join(sim_dir, "ca_data.npy"), data_np)
np.save(os.path.join(sim_dir, "ca_ref.npy"), ref_np)
print(f"  Saved               : {sim_dir}/ca_data.npy, ca_ref.npy")

# ---- plot: first hologram, flat reference, normalized ----
eps = 1e-6
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
im0 = ax[0].imshow(data_np[0], cmap="gray")
ax[0].set_title("coded hologram (position 0)")
ax[0].set_xlabel("x [det px]"); ax[0].set_ylabel("y [det px]")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(ref_np, cmap="gray")
ax[1].set_title("flat-field reference")
ax[1].set_xlabel("x [det px]"); ax[1].set_ylabel("y [det px]")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
im2 = ax[2].imshow(data_np[0] / (ref_np + eps), cmap="gray")
ax[2].set_title("normalized  data/ref  (position 0)")
ax[2].set_xlabel("x [det px]"); ax[2].set_ylabel("y [det px]")
plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
plt.suptitle("Section 7: forward simulation", fontweight="bold")
plt.tight_layout()
plt.show()

# ---- plot: a montage of several holograms across the scan ----
nshow = min(npos, 9)
ncols = 3
nrows = int(np.ceil(nshow / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axs = np.atleast_1d(axs).ravel()
for j in range(nshow):
    im = axs[j].imshow(data_np[j], cmap="gray")
    axs[j].set_title(f"pos {j}  (dy={ri[j,0]}, dx={ri[j,1]})")
    axs[j].axis("off")
for j in range(nshow, len(axs)):
    axs[j].axis("off")
plt.suptitle(f"Section 7: coded holograms across the {CA_NPOS}x{CA_NPOS} scan",
             fontweight="bold")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 8 -- RECONSTRUCTION  (Viktor's BH optimizer: GPU)
# =============================================================================
# Recover the sample psi (and refine the probe q and positions r) from the npos
# coded holograms, using the engine's Barzilai-Hestenes optimizer.
#
# Steps:
#   1) initialize the probe estimate q_init by back-propagating the flat field
#      (detector -> CA -> sample) via the engine adjoints DcT(DT(.))
#   2) pack the initial state `vars` (psi flat, q from q_init, code known,
#      positions ri/r with a small perturbation to avoid local minima)
#   3) set solver controls and run cl_rec.BH(data, vars)
#
# Requires the GPU engine. Run on the HPC.
if cl_rec is None:
    raise RuntimeError("Section 8 needs the GPU engine (cl_rec). Run on the HPC.")

import pandas as pd

# ---- 1) initialize probe from the flat-field reference ----
# DcT(DT(sqrt(ref))) backpropagates detector intensity amplitude to the object
# plane, giving a starting probe estimate.
q_init = cl_rec.DcT(cl_rec.DT(np.sqrt(ref[np.newaxis])))[0]
q_init = cp.array(q_init)
# crop central valid region and symmetric-pad back (removes edge artifacts)
ppad = 3 * args.pad // 2
if ppad > 0:
    q_init_c = q_init[ppad:args.npsi - ppad, ppad:args.npsi - ppad]
    q_init = cp.array(np.pad(q_init_c.get(), ((ppad, ppad), (ppad, ppad)),
                             mode="symmetric"))

# ---- 2) pack initial reconstruction state ----
vars = {}
vars["psi"]  = np.ones((args.npsi, args.npsi), dtype="complex64")  # flat start
vars["q"]    = cp.array(q_init, copy=True)                         # probe guess
vars["code"] = code                                                # known CA
vars["ri"]   = np.rint(pos).astype("int32")
vars["r_init"] = (pos - vars["ri"]).astype("float32")
# small random perturbation on the subpixel positions to avoid local minima
vars["r"]    = vars["r_init"] + (np.random.random((npos, 2)) - 0.5).astype("float32")
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])

# ---- 3) solver controls ----
cl_rec.rho      = [1, 2, 0.1]     # robust-loss weights (psi, q, r) -- Viktor's values
cl_rec.niter    = 1050             # total iterations
cl_rec.vis_step = 50              # visualize every N iters
cl_rec.err_step = 50               # compute error every N iters
cl_rec.eps      = 0.0             # division stabilizer (try 1e-6 if unstable)
cl_rec.lam      = 0.25             # regularization weight (try 1e-3 if noisy)
cl_rec.path_out = "./outputs/ca_rec/"
os.makedirs(cl_rec.path_out, exist_ok=True)
cl_rec.show     = True            # live plotting during reconstruction

print("=" * 62)
print("RECONSTRUCTION (Barzilai-Hestenes)")
print("=" * 62)
print(f"  Probe mode (fwd)    : {'synthetic' if USE_PROBE else 'flat (q=1)'}")
print(f"  Positions           : {npos} ({CA_NPOS}x{CA_NPOS})")
print(f"  Iterations          : {cl_rec.niter}")
print(f"  rho (psi,q,r)       : {cl_rec.rho}")
print(f"  Output snapshots    : {cl_rec.path_out}")
print("=" * 62)

# ---- run reconstruction ----
vars = cl_rec.BH(data, vars)

# ---- recovered fields ----
psi_rec = vars["psi"]
q_rec   = vars["q"]
psi_rec_np = psi_rec.get() if hasattr(psi_rec, "get") else np.asarray(psi_rec)
q_rec_np   = q_rec.get() if hasattr(q_rec, "get") else np.asarray(q_rec)

# ---- plot recovered sample (amplitude + phase) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(np.abs(psi_rec_np), cmap="gray")
ax[0].set_title("recovered |psi| amplitude")
ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.angle(psi_rec_np), cmap="gray")
ax[1].set_title("recovered arg(psi) phase [rad]")
ax[1].set_xlabel("x [px]"); ax[1].set_ylabel("y [px]")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.suptitle(f"Section 8: reconstructed sample "
             f"({'with probe' if USE_PROBE else 'flat illumination'})",
             fontweight="bold")
plt.tight_layout()
plt.show()
# =============================================================================
# SECTION 9 -- RESOLUTION / RING-PROFILE ANALYSIS + RUN SUMMARY
# =============================================================================
from datetime import datetime
from usefull_functions import plot_abs_phase_ring_profiles, resolution_summary
from datetime import datetime
_script_start = datetime.now()
%matplotlib widget
# NOTE: in your real-data script, the analysis runs on the reconstructed
# SAMPLE (there, a bit confusingly, called "q"). In THIS simulation script,
# the reconstructed sample is vars["psi"] -- vars["q"] is the PROBE, not the
# sample. So we use psi_rec_np here, not vars["q"].
psi_rec_np = vars["psi"]
if hasattr(psi_rec_np, "get"):
    psi_rec_np = psi_rec_np.get()

# the star is centered on the square npsi grid, by construction (Section 3)
_xcenter = (args.npsi - 1) / 2.0
_ycenter = (args.npsi - 1) / 2.0
_r_out_px = sample_meta["r_out"] / voxelsize
_r_in_px = sample_meta["r_in"] / voxelsize

result = plot_abs_phase_ring_profiles(
    psi_rec_np,
    xcenter=_xcenter,
    ycenter=_ycenter,
    radius_px=min(150, 0.9 * _r_out_px),
    voxelsize_um=voxelsize * 1e6,
    normalize=True,
    phase_vmin=-0.8,
    phase_vmax=0.8,
)

print(f"npos = {npos}")
print(f"lam = {cl_rec.lam}")
print(f"niter = {cl_rec.niter}")
print(f"z1 = {z1}")
print(f"z2 = {z2}")
print("  ----  ")
_script_end = datetime.now()
_elapsed = _script_end - _script_start
_hours, _remainder = divmod(int(_elapsed.total_seconds()), 3600)
_minutes = _remainder // 60
print(f"Script ended {_script_end.strftime('%d/%m at %H:%M')}")
print(f"Total run time {_hours:02d}:{_minutes:02d}")


results = resolution_summary(
    q=psi_rec_np,
    voxelsize_um=voxelsize * 1e6,
    rec_params={
        "npos":  npos,
        "lam":   cl_rec.lam,
        "niter": cl_rec.niter,
        "z1":    z1,
        "z2":    z2,
    },
    xcenter=_xcenter, ycenter=_ycenter,
    r_min=max(50, 1.2 * _r_in_px),
    r_max=0.95 * _r_out_px,
    n_radii=500,
    n_arms=arm_pairs,
    exclude_radius_ranges=[(50, 65),(100, 160), (240, 290), (480, 580)],
    exclude_angle_ranges=[(125, 150)],
    norm_radius_range=(0.6 * _r_out_px, 0.8 * _r_out_px),
    norm_radius_range_abs=(0.15 * _r_out_px, 0.3 * _r_out_px),
    modregger_roi=(slice(int(_ycenter - 450), int(_ycenter + 450)),
                   slice(int(_xcenter - 500), int(_xcenter + 500))),
    modregger_nblfac=2.0,
    show_plots=True,
)
# =============================================================================
# SECTION 0 -- common target/ROI parameters + convert q to NumPy
# =============================================================================
q_np = q.get() if hasattr(q, "get") else np.asarray(q)

_xcenter = (args.npsi - 1) / 2.0
_ycenter = (args.npsi - 1) / 2.0
r_in_px  = 10       # [px] inner radius of the star
r_out_px = 950      # [px] outer radius of the star
arm_pairs = 36      # number of arm pairs of the target

EXCLUDE_RADIUS_RANGES = [(50, 65),(100, 160), (240, 290), (480, 580)]
EXCLUDE_ANGLE_RANGES  = [(125, 150)]

voxelsize_um = voxelsize * 1e6

OUT_DIR = globals().get("OUT_DIR", "./outputs/")
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_SIZE = globals().get("PANEL_SIZE", (7, 7))

case_label = f"npos={npos}, niter={cl_rec.niter}, lam={cl_rec.lam}"
# =============================================================================
# SECTION 7 -- MTF ANALYSIS (amplitude + phase, on the reconstructed sample)
# =============================================================================
import matplotlib.patches as patches
from usefull_functions import circular_profile_angle, modulation_lstsq, _mtf_crossing

# same conventions as in Section 9 above
_xcenter = (args.npsi - 1) / 2.0
_ycenter = (args.npsi - 1) / 2.0
_r_out_px = sample_meta["r_out"] / voxelsize
_r_in_px  = sample_meta["r_in"] / voxelsize

r_min, r_max, n_radii = max(10, 1.2 * _r_in_px), 0.95 * _r_out_px, 1000
n_arms = arm_pairs
N_SAMPLES_PROFILE = 4096
NOISE_CORRECT_MODULATION = True
NORM_RADIUS_RANGE = (0.6 * _r_out_px, 0.8 * _r_out_px)
NOISE_FREQ_OFFSETS = (-3, -2, +2, +3, +5, +7)

voxelsize_um = voxelsize * 1e6
nyquist = 1 / (2 * voxelsize_um)

_exclude_radius_ranges = EXCLUDE_RADIUS_RANGES
_exclude_angle_ranges  = EXCLUDE_ANGLE_RANGES


def _freq_of_radius(r_px):
    return n_arms / (2 * np.pi * r_px * voxelsize_um)


_radii_all = np.linspace(r_min, r_max, n_radii)


def _bad_radius(r):
    return any(r0 <= r <= r1 for r0, r1 in _exclude_radius_ranges)


radii_used = np.array([r for r in _radii_all if not _bad_radius(r)])
freqs_used = _freq_of_radius(radii_used)
noise_freqs = [n_arms + off for off in NOISE_FREQ_OFFSETS if (n_arms + off) > 0]


def extract_profiles(img):
    """Circular profiles taken DIRECTLY on the given real-valued image."""
    return [circular_profile_angle(img, _xcenter, _ycenter, r,
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


def _fmt(v):
    return f"{v:.1f}" if np.isfinite(v) else "N/A"


def compute_mtf_for_image(img, exclude_angle_ranges=()):
    """Runs the full MTF pipeline on a given real-valued image."""
    profiles = extract_profiles(img)
    sig, noi = modulation_from_profiles(profiles, exclude_angle_ranges)
    mtf = normalise_mtf(sig, noi)
    f_s, mtf_s, r_s = _sorted_by_freq(mtf)
    f10 = _mtf_crossing(f_s, mtf_s, level=0.1)
    res_nm = 1000.0 / (2 * f10) if np.isfinite(f10) and f10 > 0 else np.nan
    return dict(freqs=f_s, mtf=mtf_s, radii=r_s, mtf10_nm=res_nm)


def plot_image_with_rings(img, xcenter, ycenter, radii, n_rings_show=12,
                           vmin=None, vmax=None, title="",
                           exclude_radius_ranges=(), exclude_angle_ranges=(),
                           r_max_plot=None, savepath=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)

    _r_max_plot = r_max_plot if r_max_plot is not None else radii.max() * 1.1

    for r0, r1 in exclude_radius_ranges:
        ax.add_patch(patches.Wedge((xcenter, ycenter), r1, 0, 360,
                                    width=(r1 - r0), facecolor="red",
                                    edgecolor="none", alpha=0.30, zorder=3))
    for a0, a1 in exclude_angle_ranges:
        spans = [(a0, a1)] if a0 <= a1 else [(a0, 360), (0, a1)]
        for t1, t2 in spans:
            ax.add_patch(patches.Wedge((xcenter, ycenter), _r_max_plot, t1, t2,
                                        width=_r_max_plot, facecolor="orange",
                                        edgecolor="none", alpha=0.30, zorder=3))

    _cross_half = 0.05 * max(img.shape)
    ax.plot([xcenter - _cross_half, xcenter + _cross_half], [ycenter, ycenter],
            color="red", lw=1.2, zorder=5)
    ax.plot([xcenter, xcenter], [ycenter - _cross_half, ycenter + _cross_half],
            color="red", lw=1.2, zorder=5)

    idx = np.linspace(0, len(radii) - 1, min(n_rings_show, len(radii))).astype(int)
    for r in radii[idx]:
        ax.add_patch(plt.Circle((xcenter, ycenter), r, fill=False,
                                 edgecolor="lime", linewidth=0.8, alpha=0.8, zorder=4))

    ax.set_title(f"{title} -- center + MTF rings")
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=130)
    plt.show()


case_label = f"npos={npos}, niter={cl_rec.niter}, lam={cl_rec.lam}"

psi_rec_np = vars["psi"]
if hasattr(psi_rec_np, "get"):
    psi_rec_np = psi_rec_np.get()

img_abs = np.abs(psi_rec_np)
img_phase = np.angle(psi_rec_np)

# ---- visual check: center + rings + excluded zones, amplitude ----
plot_image_with_rings(img_abs, _xcenter, _ycenter, radii_used,
                       n_rings_show=12,
                       vmin=np.percentile(img_abs, 2),
                       vmax=np.percentile(img_abs, 98),
                       title=f"{case_label} (amplitude)",
                       exclude_radius_ranges=EXCLUDE_RADIUS_RANGES,
                       exclude_angle_ranges=EXCLUDE_ANGLE_RANGES,
                       r_max_plot=_r_out_px * 1.1,
                       savepath=os.path.join(cl_rec.path_out, "mtf_rings_overlay_abs.png"))

# ---- visual check: center + rings + excluded zones, phase ----
plot_image_with_rings(img_phase, _xcenter, _ycenter, radii_used,
                       n_rings_show=12,
                       vmin=-0.8, vmax=0.8,
                       title=f"{case_label} (phase)",
                       exclude_radius_ranges=EXCLUDE_RADIUS_RANGES,
                       exclude_angle_ranges=EXCLUDE_ANGLE_RANGES,
                       r_max_plot=_r_out_px * 1.1,
                       savepath=os.path.join(cl_rec.path_out, "mtf_rings_overlay_phase.png"))

# ---- MTF computation: amplitude & phase ----
mtf_abs = compute_mtf_for_image(img_abs, _exclude_angle_ranges)
mtf_phase = compute_mtf_for_image(img_phase, _exclude_angle_ranges)

mtf10_nm_abs = mtf_abs["mtf10_nm"]
mtf10_nm_phase = mtf_phase["mtf10_nm"]

print(f"{'Case':<28s} {'MTF10 abs [nm]':>16s} {'MTF10 phase [nm]':>18s}")
print("-" * 66)
print(f"{case_label:<28s} {_fmt(mtf10_nm_abs):>16s} {_fmt(mtf10_nm_phase):>18s}")

# ---- MTF plot: amplitude vs phase together ----
plt.figure(figsize=(7, 5))
plt.plot(mtf_abs["freqs"], mtf_abs["mtf"], "-o", markersize=3, linewidth=1.3,
          color="tab:blue", label="amplitude")
plt.plot(mtf_phase["freqs"], mtf_phase["mtf"], "-o", markersize=3, linewidth=1.3,
          color="tab:orange", label="phase")
plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
if nyquist < max(mtf_abs["freqs"].max(), mtf_phase["freqs"].max()) * 1.2:
    plt.axvline(nyquist, linestyle="--", color="gray", alpha=0.6, label="Nyquist")
plt.xlabel("Spatial frequency (cycles/µm)")
plt.ylabel("Normalized MTF (direct)")
plt.title(f"MTF (direct) -- {case_label}")
plt.grid(True)
plt.legend(fontsize=8, loc="upper right")
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig(os.path.join(cl_rec.path_out, "mtf_abs_phase.png"), dpi=130)
plt.show()

# ---- variable needed by Section 10 (summary table) ----
mtf10_nm = mtf10_nm_abs   # keep amplitude as the "primary" MTF10_nm, for consistency with the summary table
# =============================================================================
# SECTION 8 -- CNR ANALYSIS (amplitude + phase, single reconstruction)
# =============================================================================
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

OUT_DIR = globals().get("OUT_DIR", cl_rec.path_out)
os.makedirs(OUT_DIR, exist_ok=True)
PANEL_SIZE = globals().get("PANEL_SIZE", (7, 7))

# same center as Section 7 (MTF)
xcenter, ycenter = _xcenter, _ycenter


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
                     r_max_plot=None, savepath=None):
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
    if savepath:
        plt.savefig(savepath, dpi=130)
    plt.show()


def compute_cnr_mean(img, roi_pairs):
    """Mean CNR over a given real-valued image, across all ROI pairs."""
    vals = [compute_cnr(img, s_roi, b_roi)[0] for s_roi, b_roi in roi_pairs]
    return float(np.mean(vals)), vals


# ---- images on which CNR is measured: amplitude & phase ----
img_cnr_abs = np.abs(psi_rec_np)
img_cnr_phase = np.angle(psi_rec_np)

_period_deg = 360.0 / arm_pairs
_half_period_deg = 0.5 * _period_deg
base_angle = np.rad2deg(sample_meta["dphi"]) / 2.0
_r_out_px = sample_meta["r_out"] / voxelsize
_roi_radius_px = min(900, 0.9 * _r_out_px)
assert _roi_radius_px < _r_out_px, "CNR ROI radius is outside the star!"


def _wedge_index_of(angle_deg):
    return int(np.floor((np.deg2rad(angle_deg) % (2 * np.pi)) / sample_meta["dphi"]))


assert _wedge_index_of(base_angle) % 2 == 0, "signal ROI is not on a Ta wedge!"
assert _wedge_index_of(base_angle + _half_period_deg) % 2 == 1, \
    "background ROI is not in a gap!"
print(f"[CNR] base_angle = {base_angle:.2f} deg -> signal on a Ta wedge, "
      f"background at +{_half_period_deg:.1f} deg in the SiC gap")


def _in_excluded(angle_deg):
    a = angle_deg % 360
    return any(a0 <= a <= a1 for a0, a1 in EXCLUDE_ANGLE_RANGES)


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

case_label = f"npos={npos}, niter={cl_rec.niter}, lam={cl_rec.lam}"

cnr_result_abs, cnr_values_abs = compute_cnr_mean(img_cnr_abs, multi_roi_pairs)
cnr_result_phase, cnr_values_phase = compute_cnr_mean(img_cnr_phase, multi_roi_pairs)

print(f"[CNR] {case_label}")
print(f"  amplitude : CNR = {cnr_result_abs:.3f}  (N={len(cnr_values_abs)} ROI pairs)")
print(f"  phase     : CNR = {cnr_result_phase:.3f}  (N={len(cnr_values_phase)} ROI pairs)")

plot_rois_multi(img_cnr_abs, multi_roi_pairs,
                 vmin=np.percentile(img_cnr_abs, 2), vmax=np.percentile(img_cnr_abs, 98),
                 title=f"{case_label} (amplitude)",
                 
                 r_max_plot=_roi_radius_px * 1.1,
                 savepath=os.path.join(OUT_DIR, "cnr_rois_abs.png"))

plot_rois_multi(img_cnr_phase, multi_roi_pairs,
                 vmin=-0.8, vmax=0.8,
                 title=f"{case_label} (phase)",
                 
                 r_max_plot=_roi_radius_px * 1.1,
                 savepath=os.path.join(OUT_DIR, "cnr_rois_phase.png"))

# ---- variable needed by Section 10 (summary table) ----
cnr_result = cnr_result_abs   # keep amplitude as the "primary" CNR, for consistency with the summary table