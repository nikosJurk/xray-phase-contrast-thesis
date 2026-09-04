"""
Nano-Holography Reconstruction — X-ray CA (coded-aperture) holography
=======================================================================

Experimental data: DanMAX (April 2026), 10x10-position scan of a Siemens
star resolution target (XRESO-50HC), coded aperture (CA) 550 nm, 10x ORCA
detector, pixel size 45 nm.

Pipeline:
  1. Load raw ORCA frames (sample scan + flat + dark)
  2. Flat/dark field correction
  3. Crop to region of interest
  4. Estimate relative CA shifts between scan positions via phase
     correlation on high-pass-filtered residuals
  5. Reconstruct object + probe with ptychography-style optimization (Rec/BH)
  6. Evaluate resolution (ring profiles, MTF) and export results

Credits: All credit for this reconstruction pipeline and analysis code
goes to Viktor Nikitin.
"""

%matplotlib inline
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import shift, gaussian_filter
from types import SimpleNamespace
from utils import *
from utils import remove_outliers, mshow, mshow_polar, mshow_complex

from rec import Rec
import pandas as pd

from datetime import datetime
from scipy import ndimage as ndi
from skimage.filters import gaussian, sobel
from skimage.registration import phase_cross_correlation
from concurrent.futures import ThreadPoolExecutor

from usefull_functions import check_data, view_stack_jupyter, check_gpu_memory, plot_abs_phase_ring_profiles, circular_profile

# ---------------------------------------------------------------------------
# Setup / GPU check
# ---------------------------------------------------------------------------
_script_start = datetime.now()
print(f"Script started {_script_start.strftime('%d/%m at %H:%M')}")
check_gpu_memory()  # Run it

# ---------------------------------------------------------------------------
# Select a sub-grid of scan positions from the full 10x10 grid
# ---------------------------------------------------------------------------
npos = 3       # CONFIG: size of the sub-grid to extract (e.g. 10x10 or 3x3)
full_n = 10    # full grid size (the raw data is 10x10)

ids0 = np.arange(full_n * full_n)
ids = np.zeros(npos * npos, dtype=np.int32)

center = full_n // 2
ss = center - npos // 2

for ky in range(npos):
    start = (ss + ky) * full_n + ss
    ids[ky * npos:(ky + 1) * npos] = ids0[start:start + npos]

print(ids.reshape(npos, npos))

# Data directory
outdir = "/zhome/5b/3/97851/Desktop/2026_DANFIX_XHIST/raw_data_3DIM/DanMAX April 2026/code_fixed_550nm/"

# HDF5 dataset path inside the ORCA files
dataset_path = "entry/instrument/orca/data"

# ---------------------------------------------------------------------------
# Load raw scan, flat field, and dark field
# ---------------------------------------------------------------------------
with h5py.File(os.path.join(outdir, "scan-0110_orca.h5"), "r") as f:
    s110 = f[dataset_path][ids].astype("float32")
    print("Loaded scan-0110_orca.h5. 10x10 grid, CA (coded aperture) moving, sample fixed.")
    print("Sample: XRESO-50HC.  https://keytech.ntt-at.com/en/xray/prd_0024.html")
    check_data(s110, "s110")
    print(" ")

with h5py.File(os.path.join(outdir, "scan-0114_orca.h5"), "r") as f:
    s114 = f[dataset_path][:].astype("float32")
    print("Loaded scan-0114_orca.h5. Flat field.")
    check_data(s114, "s114")
    print(" ")

with h5py.File(os.path.join(outdir, "scan-0115_orca.h5"), "r") as f:
    s115 = f[dataset_path][:].astype("float32")
    print("Loaded scan-0115_orca.h5. Dark field.")
    check_data(s115, "s115")
    print(" ")

# ---------------------------------------------------------------------------
# Geometry parameters (first pass, refined further below)
# ---------------------------------------------------------------------------
# Sanity relation: (1430+200-73.31)/(200-73.31) = 12.287.
# TODO: could loop over z1, and for each z1 derive z2 from the equation.
z1 = 126.69     # mm CONFIG:
z2 = 1430       # mm CONFIG:
M = (z1 + z2) / z1
zca = 16        # mm, CA -> sample
energy = 19.55  # keV
# CRL = 12, Si(111) mono, 10X ORCA objective, pixel size 45 nm, CA type 550 nm
print(z1, z2, M)
# view_stack_jupyter(s110)

# ---------------------------------------------------------------------------
# Flat / dark field correction
# ---------------------------------------------------------------------------
FlatDark = True  # CONFIG: apply flat/dark correction to the data

ref0 = np.mean(s114, axis=0).astype(np.float32)  # mean flat field
dark = np.mean(s115, axis=0).astype(np.float32)  # mean dark field
data = s110.astype(np.float32)                   # scan frames

denom = ref0 - dark

eps = 1e-6
denom = np.where(denom <= eps, eps, denom)  # avoid division by ~0

data_corr = (data - dark[None, :, :]) / denom[None, :, :]

print("Shape:", data_corr.shape)
if FlatDark:
    data = np.copy(data_corr)
    print("Data IS flat/dark corrected")
    check_data(data, "data_corr")
else:
    print("Data NOT flat/dark corrected")  # check_data(data, "Original data")

mshow(dark, True)
mshow(ref0, vmin=0, vmax=20000, show=True)
mshow(data[0], True)

# ---------------------------------------------------------------------------
# Crop region of interest: x=850:3000, y=150:2300
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = 850, 3000
y_min, y_max = 150, 2300

data_crop = data[:, y_min:y_max, x_min:x_max]
data = np.copy(data_crop)

plt.imshow(data[0])
plt.title("Cropped x=850:3000, y=150:2300")
plt.show()
data = np.copy(data_crop)  # CONFIG:
check_data(data, "data crop")
view_stack_jupyter(data)

# ---------------------------------------------------------------------------
# Utility functions: shift estimation between scan positions
# ---------------------------------------------------------------------------
def my_phase_corr(d1, d2):
    """Subpixel-free phase correlation: returns integer (dy, dx)."""
    # Hann window to suppress edge effects (optional but helps a lot)
    win = np.outer(np.hanning(d1.shape[0]), np.hanning(d1.shape[1])).astype(np.float32)
    F = np.fft.fft2(d1 * win) * np.conj(np.fft.fft2(d2 * win))
    cc = np.fft.fftshift(np.fft.ifft2(F)).real
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    H, W = d1.shape
    return np.array([iy - H // 2, ix - W // 2], np.int32)


def accumulate_to_middle(rel_shifts):
    """
    rel_shifts[k] = shift between frame k and k+1 (dy, dx).
    Returns absolute shifts w.r.t. the middle frame in the 1D sequence.
    Works for any number of positions (>= 1).
    """
    npos = rel_shifts.shape[0]
    mid = npos // 2

    abs_sh = np.zeros_like(rel_shifts, np.int32)

    # below middle
    for k in range(mid):
        abs_sh[k] = np.sum(rel_shifts[k:mid], axis=0)

    # above middle
    for k in range(mid + 1, npos):
        abs_sh[k] = -np.sum(rel_shifts[mid:k], axis=0)

    # middle stays (0,0)
    return abs_sh


# ---------------------------------------------------------------------------
# Isolate the CA (coded aperture) modulation, then measure shifts
# ---------------------------------------------------------------------------
def estimate_sample(stack, eps=1e-6, use_geomean=True):
    """Sample is static -> average across frames. Geometric mean is more robust."""
    stack = stack.astype(np.float32, copy=False)
    if use_geomean:
        return np.exp(np.mean(np.log(stack + eps), axis=0)).astype(np.float32)
    else:
        return np.mean(stack, axis=0).astype(np.float32)


def highpass_fft(img, frac=0.04, mask=None):
    """Simple circular high-pass filter in Fourier domain; frac ~ cutoff radius as fraction of Nyquist."""
    H, W = img.shape

    # reuse precomputed mask if provided
    if mask is None:
        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)
        mask = (R > frac).astype(np.float32)
        mask = np.fft.fftshift(mask)

    F = np.fft.fft2(img)
    F *= mask
    return np.fft.ifft2(F).real.astype(np.float32)


def stack_to_CA_residuals(data, hp_frac=0.04, log_domain=True):
    """
    Remove the fixed sample, leaving only the CA (coded aperture) modulation.

    input  : data  - shape (N, H, W)
    output : rdata - residuals (same shape)
             sample_est - estimated static sample
    """
    data = data.astype(np.float32, copy=False)
    sample_est = estimate_sample(data, use_geomean=True)

    if log_domain:
        res = np.log(data + 1e-6) - np.log(sample_est + 1e-6)   # additive separation
    else:
        res = data / (sample_est + 1e-6)                        # multiplicative separation
        res -= 1.0

    N, H, W = res.shape
    rdata = np.empty_like(res)

    # precompute high-pass mask once
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    mask = (R > hp_frac).astype(np.float32)
    mask = np.fft.fftshift(mask)

    # parallel high-pass over frames
    def _hp_one(k):
        return highpass_fft(res[k], frac=hp_frac, mask=mask)

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(_hp_one, range(N)))

    for k in range(N):
        rdata[k] = results[k]

    return rdata, sample_est


def compute_shifts(data, hp_frac=0.04, log_domain=True):
    """
    Compute relative and absolute shifts for a stack of frames.

    input:
        data - np.ndarray, shape (N, H, W), any N (e.g. 25 for 5x5, 225 for 15x15)

    returns:
        abs_shifts - absolute shifts w.r.t. middle frame (N, 2) (dy, dx)
        rel_shifts - neighbor shifts between k and k+1  (N, 2) (last row = 0)
        rdata      - residual stack (N, H, W)
        sample_est - estimated static sample (H, W)
    """
    # 1) remove the fixed Siemens star -> leave CA residuals
    rdata, sample_est = stack_to_CA_residuals(data, hp_frac=hp_frac, log_domain=log_domain)

    # 2) neighbor correlation on the residuals
    npos = rdata.shape[0]
    rel_shifts = np.zeros((npos, 2), np.int32)

    def _pair_corr(k):
        return my_phase_corr(rdata[k], rdata[k + 1])

    # parallel phase-correlation on neighbor pairs
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(_pair_corr, range(npos - 1)))

    for k in range(npos - 1):
        rel_shifts[k] = results[k]
    rel_shifts[-1] = 0

    # 3) accumulate to the middle frame
    abs_shifts = accumulate_to_middle(rel_shifts)

    return abs_shifts, rel_shifts, rdata, sample_est


# ---------------------------------------------------------------------------
# Run shift estimation
# ---------------------------------------------------------------------------
abs_shifts, rel_shifts, rdata, sample_est = compute_shifts(data)

print("Absolute shifts wrt middle (dy, dx) — first/last 5:")
print(abs_shifts[:5], " ... ", abs_shifts[-5:])
print("Middle index:", abs_shifts.shape[0] // 2,
      "->", abs_shifts[abs_shifts.shape[0] // 2])

plt.figure(figsize=(5, 5))
plt.title("Estimated sample (geometric mean)")
plt.imshow(sample_est, cmap='gray')
plt.colorbar()
plt.show()

plt.figure(figsize=(5, 5))
plt.title("Residual (frame 0) ~ CA only")
plt.imshow(rdata[0], cmap='gray')
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------------
# Reorder frames into a serpentine (boustrophedon) scan path and re-estimate
# shifts with a coarser (gaussian-smoothed) phase correlation
# ---------------------------------------------------------------------------
ids = np.arange(npos * npos)
for ky in range(npos):
    if ky % 2 == 1:
        ids[ky * npos:(ky + 1) * npos] = ids[ky * npos:(ky + 1) * npos][::-1]
print(ids)
data = data[ids]
rdata = rdata[ids]
shifts_relative = np.zeros([data.shape[0], 2], dtype='float32')


def my_phase_corr(d1, d2):
    image_product = np.fft.fft2(d1) * np.fft.fft2(d2).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ind = np.unravel_index(np.argmax(cc_image.real, axis=None), cc_image.real.shape)
    # shifts = np.subtract(ind, d1.shape[-1]//2)
    shifts = np.array(ind)
    shifts[0] = ind[0] - d1.shape[0] // 2
    shifts[1] = ind[1] - d1.shape[1] // 2
    return shifts


ids_res = []
for k in range(0, data.shape[0] - 1):
    tmp1 = rdata[k].copy()
    tmp2 = rdata[k + 1].copy()
    tmp1 = gaussian_filter(tmp1, 4)
    tmp2 = gaussian_filter(tmp2, 4)

    shifts_relative[k] = my_phase_corr(tmp1, tmp2)

    # sanity check block (kept for reference, disabled)
    # h,w = tmp1.shape
    # nn = 512
    # offset = 3200
    # tmp = rdata[k,h//2-nn//2:h//2+nn//2,w//2-nn//2+offset:w//2+nn//2+offset]
    # tmp1 = rdata[k+1,h//2-nn//2:h//2+nn//2,w//2-nn//2+offset:w//2+nn//2+offset]
    # tmp2 = shift(tmp,-shifts_relative[k])
    # dif = tmp1-tmp2
    # nn = np.linalg.norm(dif[nn//2-nn//8:nn//2+nn//8,nn//2-nn//8:nn//2+nn//8])
    # if nn>8:
    #     print('WARNING')
    #     mshow_complex(tmp1+1j*tmp,True)
    #     mshow_complex(rdata[k]+1j*rdata[k+1],True)
    #     mshow(dif,True,vmax=0.3,vmin=-0.3)
    print(k, shifts_relative[k])

ipos = npos * npos // 2 - npos // 2 * (npos % 2 == 0)  # align w.r.t. the middle position
shifts = shifts_relative * 0
for k in range(ipos):
    shifts[k] = np.sum(shifts_relative[k:ipos], axis=0)
shifts[ipos] = 0  # shifts[ipos]
for k in range(ipos, npos * npos):
    shifts[k] = np.sum(-shifts_relative[ipos:k], axis=0)

plt.plot(shifts[..., 0], shifts[..., 1], '.')
plt.plot(shifts[ipos, 0], shifts[ipos, 1], 'rx')
plt.grid()

# ---------------------------------------------------------------------------
# Refined imaging geometry (meters)
# ---------------------------------------------------------------------------
import numpy as np

n = data.shape[1]

energy = 19.55  # keV
wavelength = 1.2398419840550367e-09 / energy  # m

# --- Geometry (meters) ---
pixelsize = 0.55e-6   # m  (converted from um)
z1 = 0.125            # m   0.12669   CONFIG:
z2 = 1.430            # m   CONFIG:

focusToDetectorDistance = z1 + z2
z_eff = (z1 * z2) / focusToDetectorDistance
magnification = focusToDetectorDistance / z1
voxelsize = pixelsize / magnification  # m

print("----- Geometry Parameters -----")
print(f"z1 (focus -> sample):           {z1:.6f} m")
print(f"z2 (sample -> detector):        {z2:.6f} m")
print(f"Source -> detector distance:    {focusToDetectorDistance:.6f} m")
print("")

print("----- Imaging Properties -----")
print(f"Wavelength:                    {wavelength:.6e} m")
print(f"Effective propagation (z_eff): {z_eff:.6e} m")
print(f"Magnification (M):             {magnification:.3f}")
print(f"Detector pixel size:           {pixelsize:.6e} m")
print(f"Object voxel size:             {voxelsize:.6e} m")
print("")

print("----- Sanity Check -----")
print(f"M ~ (z1+z2)/z1 -> {(z1+z2)/z1:.3f}")
print(f"z_eff ~ (z1*z2)/(z1+z2) -> {(z1*z2)/(z1+z2):.6e}")

# ---------------------------------------------------------------------------
# Free up memory before running the reconstruction
# ---------------------------------------------------------------------------
to_delete = ["s110", "s114", "s115", "data_corr", "data_crop", "rdata", "dark", "ref0", "denom", "tmp1", "tmp2", "sample_est"]
for var in to_delete:
    if var in globals():
        del globals()[var]
# gc.collect()
# %whos

# ---------------------------------------------------------------------------
# Reconstruction settings
# ---------------------------------------------------------------------------
args = SimpleNamespace()

args.ngpus = 1   # number of GPUs
args.lam = 0.25  # CONFIG: regularization strength if results look noisy (for a small number of positions)

args.n = n
args.ex = 16  # extra padding for shifts
args.npsi = int(np.ceil((n + 2 * np.amax(np.abs(shifts)) + 2 * args.ex) / 32) * 32)  # reconstruction size after padding
args.npatch = args.n + 2 * args.ex
args.npos = npos * npos
args.nchunk = 4  # CONFIG: number of frames per GPU

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = z1  # CONFIG: propagation distance — this value seems more accurate, gives better quality

args.rho = [1, 2, 0.1]  # weights for object, probe, and position correction

args.niter = 1050    # number of iterations CONFIG:
args.err_step = 50   # error display step CONFIG:
args.vis_step = 50   # visualization display step CONFIG:

args.path_out = "./data/tmp/"  # Note: output path
os.makedirs(args.path_out, exist_ok=True)  # Note: ensure output dir exists
args.show = True
print(args.n, args.ex, args.npsi, args.npatch, args.npos, args.nchunk, args.distance)

# ---------------------------------------------------------------------------
# Run the reconstruction
# ---------------------------------------------------------------------------
cl_rec = Rec(args)
vars = {}
vars["psi"] = cp.ones([args.npsi, args.npsi], dtype='complex64')
vars["q"] = cp.ones([args.n, args.n], dtype='complex64')
vars["ri"] = np.round(shifts).astype("int32")                       # integer part of the shift
vars["r"] = np.array(shifts - vars["ri"]).astype("float32")         # fractional part of the shift
vars["r_init"] = np.array(shifts - vars["ri"]).astype("float32")
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])
print(np.shape(data))
plt.imshow(data[0])

# reconstruction
vars = cl_rec.BH(data, vars)
q = vars["q"]
print(f"npos = {npos}")
print(f"lam = {args.lam}")
print(f"niter = {args.niter}")
print(f"z1 = {z1}")
print(f"z2 = {z2}")
print("  ----  ")

# ---------------------------------------------------------------------------
# Display probe amplitude / phase (auto-cropped to signal region)
# ---------------------------------------------------------------------------
P = cp.asnumpy(vars["psi"])
if P.ndim == 3:
    P = P[0]

A = np.abs(P)
row_std = A.std(axis=1)
col_std = A.std(axis=0)
thr = 0.05 * max(row_std.max(), col_std.max())
rows = np.where(row_std > thr)[0]
cols = np.where(col_std > thr)[0]
P = P[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(np.abs(P), cmap='gray')
ax[0].set_title("Probe amplitude  |psi|")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(np.angle(P), cmap='gray')
ax[1].set_title("Probe phase  arg(psi)")
fig.colorbar(im1, ax=ax[1], fraction=0.046)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Cell 3: ring profiles — no crop applied here
# ---------------------------------------------------------------------------
import numpy as np

Q = cp.asnumpy(q) if hasattr(q, 'get') else np.asarray(q)
if Q.ndim == 3:
    Q = Q[0]

# automatically crop the padding (here it doesn't crop anything, Q is already 2150x2150)
A = np.abs(Q)
row_std, col_std = A.std(axis=1), A.std(axis=0)
thr = 0.05 * max(row_std.max(), col_std.max())
rows, cols = np.where(row_std > thr)[0], np.where(col_std > thr)[0]
y_off, x_off = rows[0], cols[0]
Q = Q[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

# ---- center ----
# q comes out on the same grid as the cropped data (2150x2150), so no additional padding
padx = 0
pady = 0
xc_full, yc_full = 1082, 1060   # star center in the FULL (uncropped) frame
xcenter = xc_full                # = 1875 - 850 = 1025
ycenter = yc_full                # = 1170 - 150 = 1020
print("Q.shape =", Q.shape, "| center =", xcenter, ycenter)

result = plot_abs_phase_ring_profiles(
    Q,
    xcenter=xcenter,
    ycenter=ycenter,
    radius_px=157,
    voxelsize_um=voxelsize * 1e6,
    normalize=True,
    phase_vmin=-0.8,
    phase_vmax=0.8,
)

# ---------------------------------------------------------------------------
# Resolution summary (radial profile / MTF-style analysis)
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from usefull_functions import resolution_summary

# same as the crop applied to the raw data
x_min, y_min = 850, 150

_q = vars["q"]
Q = cp.asnumpy(_q) if hasattr(_q, 'get') else np.asarray(_q)
if Q.ndim == 3:
    Q = Q[0]

# ---- auto-crop of the padding (here it doesn't crop anything, Q is already 2150x2150) ----
A = np.abs(Q)
row_std, col_std = A.std(axis=1), A.std(axis=0)
thr = 0.05 * max(row_std.max(), col_std.max())
rows, cols = np.where(row_std > thr)[0], np.where(col_std > thr)[0]
y_off, x_off = rows[0], cols[0]
Q = Q[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

# ---- center: star center in the FULL frame -> minus crop -> minus auto-crop ----
padx = 0
pady = 0
xc_full, yc_full = 1082, 1060      # star center in the FULL (uncropped) frame
xcenter = xc_full
ycenter = yc_full

A = np.abs(Q)
plt.imshow(A, cmap='gray')
plt.plot(xcenter, ycenter, 'r+', ms=12, mew=2)
plt.title("check center")
plt.show()

# ---- maximum radius that fits ----
H, Wd = A.shape
r_max = min(850, xcenter, ycenter, Wd - 1 - xcenter, H - 1 - ycenter)
print("Q.shape:", A.shape, "| center:", xcenter, ycenter, "| r_max:", r_max)

# ---- ROI: 900x1000 px box around the center (as in the previous 800:1700 / 900:1900) ----
hy, hx = 450, 500
ys = slice(max(0, int(ycenter - hy)), min(H, int(ycenter + hy)))
xs = slice(max(0, int(xcenter - hx)), min(Wd, int(xcenter + hx)))
print("ROI:", ys, xs)

results = resolution_summary(
    q=Q,
    voxelsize_um=voxelsize * 1e6,
    rec_params={"npos": 10, "lam": 0.05, "niter": 1050, "z1": 0.12669, "z2": 1.43},
    xcenter=xcenter, ycenter=ycenter,
    r_min=50, r_max=r_max, n_radii=500,
    exclude_radius_ranges=[(450, 520), (220, 260), (100, 135)],
    norm_radius_range=(600, 800),
    norm_radius_range_abs=(150, 300),
    modregger_roi=(ys, xs),
    modregger_nblfac=2.0,
    show_plots=True,
)

# ---------------------------------------------------------------------------
# Export reconstructed abs / phase as TIFF and PNG
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff  # pip install tifffile if not already installed
import os

# ---- load reconstruction data ----
Q = cp.asnumpy(vars["q"]) if hasattr(vars["q"], 'get') else np.asarray(vars["q"])
if Q.ndim == 3:
    Q = Q[0]

# ---- auto-crop the padding ----
A = np.abs(Q)
row_std = A.std(axis=1)
col_std = A.std(axis=0)
thr = 0.05 * max(row_std.max(), col_std.max())
rows = np.where(row_std > thr)[0]
cols = np.where(col_std > thr)[0]
Q = Q[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

abs_img = np.abs(Q).astype(np.float32)
phase_img = np.angle(Q).astype(np.float32)

# ---- preview plot (always shown, not saved here) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(abs_img, cmap='gray')
ax[0].set_title("Reconstructed abs")
fig.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(phase_img, cmap='gray')
ax[1].set_title("Reconstructed phase")
fig.colorbar(im1, ax=ax[1], fraction=0.046)

fig.tight_layout()
plt.show()

# ---- save outputs ----
out_dir = os.path.join(os.getcwd(), "holo_images")
os.makedirs(out_dir, exist_ok=True)

base_name = "Nano_recon_npos_10X10"


def save_if_missing(path, save_fn):
    """Calls save_fn(path) only if the file doesn't already exist."""
    if os.path.exists(path):
        print(f"  SKIPPED (already exists): {path}")
    else:
        save_fn(path)
        print(f"  Saved: {path}")


print(f"Saving to: {out_dir}\n")

# --- TIFF (32-bit float, raw data) ---
save_if_missing(
    os.path.join(out_dir, f"{base_name}_abs.tiff"),
    lambda p: tiff.imwrite(p, abs_img)
)
save_if_missing(
    os.path.join(out_dir, f"{base_name}_phase.tiff"),
    lambda p: tiff.imwrite(p, phase_img)
)

# --- PNG (single images, no colorbar/title, clean render) ---
save_if_missing(
    os.path.join(out_dir, f"{base_name}_abs.png"),
    lambda p: plt.imsave(p, abs_img, cmap='gray')
)
save_if_missing(
    os.path.join(out_dir, f"{base_name}_phase.png"),
    lambda p: plt.imsave(p, phase_img, cmap='gray')
)

print("\nDone.")

# ---------------------------------------------------------------------------
# Batch-save every figure produced by resolution_summary()
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from usefull_functions import resolution_summary
import os
from contextlib import contextmanager

out_dir = os.path.join(os.getcwd(), "holo_images")
os.makedirs(out_dir, exist_ok=True)

_save_counter = {"n": 0}


@contextmanager
def capture_all_shows(prefix, out_dir=out_dir, dpi=300):
    """
    While inside this block, every plt.show() call — whether ours or
    triggered internally inside another function — saves the figure
    to disk instead of just displaying/closing it.
    """
    original_show = plt.show

    def custom_show(*args, **kwargs):
        fig = plt.gcf()
        _save_counter["n"] += 1
        fpath = os.path.join(out_dir, f"{prefix}_{_save_counter['n']}.png")
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {fpath}")
        plt.close(fig)

    plt.show = custom_show
    try:
        yield
    finally:
        plt.show = original_show
        # anything left open (didn't go through show()) gets saved too
        for num in plt.get_fignums():
            fig = plt.figure(num)
            _save_counter["n"] += 1
            fpath = os.path.join(out_dir, f"{prefix}_{_save_counter['n']}.png")
            fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
            print(f"  Saved: {fpath}")
        plt.close('all')


# same crop as applied to the raw data
x_min, y_min = 850, 150
_q = vars["q"]
Q = cp.asnumpy(_q) if hasattr(_q, 'get') else np.asarray(_q)
if Q.ndim == 3:
    Q = Q[0]

# ---- auto-crop the padding ----
A = np.abs(Q)
row_std, col_std = A.std(axis=1), A.std(axis=0)
thr = 0.05 * max(row_std.max(), col_std.max())
rows, cols = np.where(row_std > thr)[0], np.where(col_std > thr)[0]
y_off, x_off = rows[0], cols[0]
Q = Q[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

# ---- center ----
padx = 0
pady = 0
xc_full, yc_full = 1082, 1060
xcenter = xc_full
ycenter = yc_full

# ---- max radius ----
H, Wd = A.shape
r_max = min(850, xcenter, ycenter, Wd - 1 - xcenter, H - 1 - ycenter)
print("Q.shape:", A.shape, "| center:", xcenter, ycenter, "| r_max:", r_max)

# ---- ROI ----
hy, hx = 450, 500
ys = slice(max(0, int(ycenter - hy)), min(H, int(ycenter + hy)))
xs = slice(max(0, int(xcenter - hx)), min(Wd, int(xcenter + hx)))
print("ROI:", ys, xs)

# ---- save full abs & phase image (before cropping to ROI) ----
with capture_all_shows("phase_abs"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(np.abs(Q), cmap='gray')
    axes[0].set_title("Amplitude (abs)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.angle(Q), cmap='gray')
    axes[1].set_title("Phase")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

with capture_all_shows("resolution_summary"):
    results = resolution_summary(
        q=Q,
        voxelsize_um=voxelsize * 1e6,
        rec_params={"npos": 10, "lam": 0.25, "niter": 1050, "z1": 0.12669, "z2": 1.43},
        xcenter=xcenter, ycenter=ycenter,
        r_min=50, r_max=r_max, n_radii=500,
        exclude_radius_ranges=[(450, 520), (220, 260), (100, 135)],
        norm_radius_range=(600, 800),
        norm_radius_range_abs=(150, 300),
        modregger_roi=(ys, xs),
        modregger_nblfac=2.0,
        show_plots=True,
    )

"""
Circular (radial) MTF profile from Siemens star reconstructions
============================================================================

Reads:
    Nano_recon_npos_<N>X<N>_phase.tiff
    Nano_recon_npos_<N>X<N>_abs.tiff
for N = 3, 4, 10 (or whatever folders exist), computes the MTF(f) around the
center of the Siemens star target, and finds the radius/frequency where the
MTF drops to 10% (and optionally 50%).

Usage inside Jupyter:
    from mtf_circular_profile import analyze_star
    res = analyze_star("holo_images/10X10/Nano_recon_npos_10X10_phase.tiff",
                        voxel_size_um=0.0442, n_spokes=36)
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d


def load_tiff(path):
    img = tifffile.imread(path)
    if img.ndim == 3:          # if it has multiple frames/channels, take the first
        img = img[0]
    return img.astype(np.float64)


def find_center(img):
    """Simple center estimate: image center (works well for a Siemens star)."""
    ny, nx = img.shape
    return nx / 2.0, ny / 2.0   # for a Siemens star this is almost always the image center


def radial_modulation(img, cx, cy, r_min, r_max, n_radii=200, n_angles=720):
    """
    For each radius r, samples n_angles points on the circle of radius r
    around the center (cx, cy) and computes the modulation contrast:
        C(r) = (Imax - Imin) / (Imax + Imin)
    which is used as an approximation of MTF(r) (normalized so that
    MTF(0)=1 at the radius closest to the center).
    """
    radii = np.linspace(r_min, r_max, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    contrast = np.zeros(n_radii)
    for i, r in enumerate(radii):
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        vals = map_coordinates(img, [ys, xs], order=1, mode='nearest')
        vmin, vmax = vals.min(), vals.max()
        denom = vmax + vmin
        contrast[i] = (vmax - vmin) / denom if denom != 0 else 0.0

    return radii, contrast


def radius_to_freq(radii_px, voxel_size_um, n_spokes):
    """
    Convert radius (in pixels) to spatial frequency (cycles/um) for a
    Siemens star: at radius r, the local period of one spoke pair is
        period = (2*pi*r) / n_spokes   [pixels]
    hence the frequency:
        f = n_spokes / (2*pi*r) / voxel_size_um   [cycles/um]
    """
    radii_px = np.asarray(radii_px, dtype=np.float64)
    with np.errstate(divide='ignore'):
        freq = n_spokes / (2 * np.pi * radii_px) / voxel_size_um
    return freq


def find_mtf_crossing(freq, mtf_norm, level=0.10):
    """
    Finds the frequency where the normalized MTF drops below `level`,
    scanning from high frequencies (small radius) toward low ones.
    Returns (freq_at_level, res_half_period_um).
    """
    order = np.argsort(freq)              # ascending frequency order
    f_sorted = freq[order]
    m_sorted = mtf_norm[order]

    # keep only valid points
    valid = np.isfinite(f_sorted) & np.isfinite(m_sorted)
    f_sorted, m_sorted = f_sorted[valid], m_sorted[valid]

    interp = interp1d(f_sorted, m_sorted, bounds_error=False, fill_value=np.nan)

    # dense frequency grid, find first crossing below level starting from high f
    f_dense = np.linspace(f_sorted.min(), f_sorted.max(), 5000)
    m_dense = interp(f_dense)

    below = np.where(m_dense <= level)[0]
    if len(below) == 0:
        return None, None
    f_cross = f_dense[below[0]]           # first (lowest) frequency where M<=level...
    res_half_period_um = 1.0 / (2 * f_cross) if f_cross > 0 else np.nan
    return f_cross, res_half_period_um


def analyze_star(tiff_path, voxel_size_um=0.0442, n_spokes=36,
                  r_min_px=5, r_max_px=None, plot=True, title=None):
    img = load_tiff(tiff_path)
    ny, nx = img.shape
    cx, cy = find_center(img)

    if r_max_px is None:
        r_max_px = 0.9 * min(cx, cy, nx - cx, ny - cy)

    radii_px, contrast = radial_modulation(img, cx, cy, r_min_px, r_max_px)
    freq = radius_to_freq(radii_px, voxel_size_um, n_spokes)

    # normalize with respect to the value at the lowest frequency (large radius)
    order = np.argsort(freq)
    mtf_norm = contrast / contrast[order][0] if contrast[order][0] != 0 else contrast

    f10, res10 = find_mtf_crossing(freq, mtf_norm, level=0.10)
    f50, res50 = find_mtf_crossing(freq, mtf_norm, level=0.50)

    print(f"--- {tiff_path} ---")
    print(f"MTF=10% at f = {f10:.4f} cyc/um  ->  res (half-period) = {res10:.4f} um" if f10 else "No 10% crossing found")
    print(f"MTF=50% at f = {f50:.4f} cyc/um  ->  res (half-period) = {res50:.4f} um" if f50 else "No 50% crossing found")

    if plot:
        order = np.argsort(freq)
        plt.figure(figsize=(6, 4))
        plt.plot(freq[order], mtf_norm[order], '-', lw=1.5)
        plt.axhline(0.10, color='r', ls='--', lw=1, label='10% MTF')
        plt.axhline(0.50, color='g', ls='--', lw=1, label='50% MTF')
        if f10:
            plt.axvline(f10, color='r', ls=':', lw=1)
        if f50:
            plt.axvline(f50, color='g', ls=':', lw=1)
        plt.xlabel('Spatial frequency [cyc/um]')
        plt.ylabel('Normalized MTF (modulation contrast)')
        plt.title(title or tiff_path)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'radii_px': radii_px,
        'freq': freq,
        'mtf': mtf_norm,
        'f10': f10, 'res10_um': res10,
        'f50': f50, 'res50_um': res50,
    }


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Adjust voxel_size_um and n_spokes per case if they differ
    # (e.g. 0.0442 um/px as used in the resolution summaries above).
    # n_spokes = number of spokes (dark+light pairs) of the Siemens
    # star target used — set this to the correct value for your target.
    # -----------------------------------------------------------------
    folders = {
        "3X3": "3X3",
        "4X4": "4X4",
        "10X10": "10X10",
    }
    channels = ["phase", "abs"]

    results = {}
    for name, folder in folders.items():
        for ch in channels:
            path = f"{folder}/Nano_recon_npos_{name}_{ch}.tiff"
            key = f"{name}_{ch}"
            try:
                results[key] = analyze_star(
                    path,
                    voxel_size_um=0.0442,
                    n_spokes=36,          # <-- ADJUST TO YOUR TARGET
                    title=key,
                )
            except FileNotFoundError:
                print(f"Not found: {path}")

    # Summary
    print("\n=== SUMMARY (MTF=10%) ===")
    for key, r in results.items():
        if r['f10']:
            print(f"{key}: f10 = {r['f10']:.4f} cyc/um, res = {r['res10_um']:.4f} um")