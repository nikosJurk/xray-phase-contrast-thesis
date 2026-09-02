import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cupy as cp
from matplotlib.patches import Circle, Wedge

import math
import h5py
import pyfftw
from scipy.ndimage import gaussian_filter


def check_data(arr, name="array"):
    """
    Print statistics of an array:
    - mean
    - std
    - min
    - max
    - number of zeros
    - number of NaNs
    """
    arr = np.asarray(arr)

    mean_val = np.mean(arr)
    std_val = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    n_zeros = np.sum(arr == 0)
    n_nans = np.isnan(arr).sum()

    print(f"--- {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Mean: {mean_val:.6g}")
    print(f"Std:  {std_val:.6g}")
    print(f"Min:  {min_val:.6g}")
    print(f"Max:  {max_val:.6g}")
    print(f"Zeros: {n_zeros} ({n_zeros / arr.size * 100:.3f}%)")
    print(f"NaNs:  {n_nans} ({n_nans / arr.size * 100:.3f}%)")
    print(" ")


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def view_stack_jupyter(data, cmap="gray", vmin=None, vmax=None):
    """
    Jupyter-friendly viewer for a 3D stack: (N, H, W)

    Parameters
    ----------
    data : array-like
        3D array with shape (n_frames, height, width)
    cmap : str
        Matplotlib colormap
    vmin, vmax : float or None
        Fixed contrast limits. If None, uses per-frame min/max.
    """
    data = np.asarray(data)
    assert data.ndim == 3, "data must have shape (N, H, W)"

    n_frames = data.shape[0]

    out = widgets.Output()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_frames - 1,
        step=1,
        description="Frame:",
        continuous_update=False,
        layout=widgets.Layout(width="700px")
    )

    btn_prev = widgets.Button(description="Prev")
    btn_next = widgets.Button(description="Next")

    def draw(i):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 6))

            frame = data[i]

            if vmin is None or vmax is None:
                im = ax.imshow(frame, cmap=cmap)
            else:
                im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_title(f"Frame {i+1}/{n_frames}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()

    def on_slider_change(change):
        if change["name"] == "value":
            draw(change["new"])

    def on_prev_clicked(b):
        slider.value = (slider.value - 1) % n_frames

    def on_next_clicked(b):
        slider.value = (slider.value + 1) % n_frames

    slider.observe(on_slider_change, names="value")
    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)

    controls = widgets.HBox([btn_prev, btn_next, slider])
    display(controls, out)

    draw(0)


def check_gpu_memory(threshold_gb=25):
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        print("No GPU detected.")
        return

    print(f"Number of GPUs detected: {n_devices}\n")

    for i in range(n_devices):
        with cp.cuda.Device(i):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            used_gb = total_gb - free_gb

            print(f"GPU {i}:")
            print(f"  Total memory: {total_gb:.2f} GB")
            print(f"  Used memory : {used_gb:.2f} GB")
            print(f"  Free memory : {free_gb:.2f} GB")

            if free_gb < threshold_gb:
                print("  ⚠️  GPU memory low\n")
            else:
                print("  ✅ GPU memory OK\n")



def circular_profile(img, xcenter, ycenter, radius_px, n_samples=1440):
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

    x = xcenter + radius_px * np.cos(theta)
    y = ycenter + radius_px * np.sin(theta)

    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)

    profile = img[yi, xi]
    return profile, theta


def plot_abs_phase_ring_profiles(
    q,
    xcenter,
    ycenter,
    radius_px,
    voxelsize_um=None,
    normalize=True,
    phase_vmin=None,
    phase_vmax=None,
    abs_vmin=None,
    abs_vmax=None,
):
    if isinstance(q, cp.ndarray):
        q = cp.asnumpy(q)

    phase = np.angle(q)
    absorption = np.abs(q)

    print(f"Phase: min={phase.min():.3f}, max={phase.max():.3f}, std={phase.std():.3f}")
    print(f"Abs:   min={absorption.min():.3f}, max={absorption.max():.3f}, std={absorption.std():.3f}")
    print(f"Center: x={xcenter}, y={ycenter}, radius={radius_px}px")

    phase_prof, theta = circular_profile(
        phase, xcenter, ycenter, radius_px
    )

    abs_prof, _ = circular_profile(
        absorption, xcenter, ycenter, radius_px
    )

    if voxelsize_um is None:
        x = np.degrees(theta)
        xlabel = "angle (deg)"
    else:
        x = theta * radius_px * voxelsize_um
        xlabel = "arc length (µm)"

    if normalize:
        phase_plot = (phase_prof - np.nanmean(phase_prof)) / (np.nanstd(phase_prof) + 1e-12)
        abs_plot = (abs_prof - np.nanmean(abs_prof)) / (np.nanstd(abs_prof) + 1e-12)
        ylabel = "normalized"
    else:
        phase_plot = phase_prof
        abs_plot = abs_prof
        ylabel = "raw value"

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    im0 = axs[0].imshow(
        absorption,
        cmap="gray",
        origin="upper",
        #vmin=abs_vmin,
        #vmax=abs_vmax,
    )
    axs[0].add_patch(Circle((xcenter, ycenter), radius_px, fill=False, color="red", lw=2))
    axs[0].plot(xcenter, ycenter, "r+", markersize=12)
    axs[0].set_title("Absorption with ring")
    axs[0].set_xlabel("x pixels")
    axs[0].set_ylabel("y pixels")
    plt.colorbar(im0, ax=axs[0], label="absorption / amplitude")

    im1 = axs[1].imshow(
        phase,
        cmap="gray",
        origin="upper",
        #vmin=phase_vmin,
        #vmax=phase_vmax,
    )
    axs[1].add_patch(Circle((xcenter, ycenter), radius_px, fill=False, color="red", lw=2))
    axs[1].plot(xcenter, ycenter, "r+", markersize=12)
    axs[1].set_title("Phase with ring")
    axs[1].set_xlabel("x pixels")
    axs[1].set_ylabel("y pixels")
    plt.colorbar(im1, ax=axs[1], label="phase (rad)")

    axs[2].plot(x, phase_plot, label="Phase", lw=2)
    axs[2].plot(x, abs_plot, label="Absorption", lw=2)
    axs[2].set_title(f"Ring line profiles, r = {radius_px}px")
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel(ylabel)
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    return {
        "phase_profile": phase_prof,
        "absorption_profile": abs_prof,
        "x": x,
        "theta": theta,
        "center": (xcenter, ycenter),
        "radius_px": radius_px,
    }


"""
Modregger-style resolution estimation by Fourier-spectrum analysis.

Reference: Modregger et al. 2007, "Spatial resolution in Bragg-magnified X-ray
images as determined by Fourier analysis."

The idea: at high spatial frequencies the image is dominated by noise; at low
spatial frequencies it is dominated by signal. The crossover -- where signal
power drops to ``nblfac`` times the noise power -- gives the resolution.
"""

from numpy.fft import fft, fftshift


# ---------------------------------------------------------------------------
# core 1D estimator
# ---------------------------------------------------------------------------

def _modregger_1d(data, axis, filterwidth=5, highfrq=2.0, nblfac=2.0,
                  apply_window=True):
    """
    Estimate resolution along one axis of a 2D image by Fourier analysis.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    N = data.shape[axis]
    k_full = 2 * np.pi / N * np.arange(-N // 2, N // 2)

    if apply_window:
        win = np.hanning(N)
        if axis == 1:
            data_w = data * win[np.newaxis, :]
        else:
            data_w = data * win[:, np.newaxis]
    else:
        data_w = data

    F = fftshift(fft(data_w, axis=axis), axes=axis)
    P = np.abs(F) ** 2

    # keep only positive frequencies
    pos = k_full > 0
    k = k_full[pos]
    if axis == 1:
        P = P[:, pos]
        n_lines, n_freq = P.shape
    else:
        P = P[pos, :].T
        n_lines, n_freq = P.shape

    # smooth each line
    kernel = np.ones(filterwidth) / filterwidth
    P_smooth = np.empty_like(P)
    for i in range(n_lines):
        P_smooth[i] = np.convolve(P[i], kernel, mode="same")

    per_line_res = np.full(n_lines, np.nan)
    per_line_kres = np.full(n_lines, np.nan)

    for i in range(n_lines):
        line = P_smooth[i]
        high_mask = k > highfrq
        if not np.any(high_mask):
            continue
        baseline = np.mean(line[high_mask])
        threshold = nblfac * baseline

        below = np.where(line <= threshold)[0]
        above = np.where(line >= threshold)[0]
        if below.size == 0 or above.size == 0:
            continue
        k_lo = k[below.min()]
        k_hi = k[above.max()]
        kres = 0.5 * (k_lo + k_hi)
        if kres <= 0:
            continue

        per_line_kres[i] = kres
        per_line_res[i] = 2 * np.pi / kres

    valid = np.isfinite(per_line_res)
    if not np.any(valid):
        raise RuntimeError(
            "No valid resolution estimate. Try lowering 'highfrq' or "
            "increasing the ROI size."
        )

    res_pix = np.nanmean(per_line_res)
    ures_pix = np.nanstd(per_line_res)
    kres_mean = np.nanmean(per_line_kres)
    kres_unc = np.nanstd(per_line_kres)

    power_mean = np.mean(P_smooth, axis=0)
    baseline_mean = np.mean(power_mean[k > highfrq])

    return {
        "k_grid": k,
        "power_mean": power_mean,
        "baseline": baseline_mean,
        "k_res": kres_mean,
        "uk_res": kres_unc,
        "res_pix": res_pix,
        "ures_pix": ures_pix,
        "per_line_res_pix": per_line_res[valid],
    }


# ---------------------------------------------------------------------------
# user-facing wrapper for a complex reconstruction q
# ---------------------------------------------------------------------------

def modregger_resolution(
    q,
    voxelsize_um=None,
    roi=None,
    channel="both",
    filterwidth=5,
    highfrq=2.0,
    nblfac=2.0,
    apply_window=True,
    plot=True,
):
    """
    Modregger-style resolution estimation on the phase and/or absorption
    channels of a complex reconstruction q.
    """
    if voxelsize_um is None:
        raise ValueError("voxelsize_um must be provided (in micrometers)")

    if isinstance(q, cp.ndarray):
        q = cp.asnumpy(q)

    if roi is not None:
        q = q[roi[0], roi[1]]

    channels = {}
    if channel in ("phase", "both"):
        channels["phase"] = np.angle(q)
    if channel in ("absorption", "both"):
        channels["absorption"] = np.abs(q)

    results = {}
    for name, img in channels.items():
        img = img - np.mean(img)

        out_x = _modregger_1d(img, axis=1, filterwidth=filterwidth,
                              highfrq=highfrq, nblfac=nblfac,
                              apply_window=apply_window)
        out_y = _modregger_1d(img, axis=0, filterwidth=filterwidth,
                              highfrq=highfrq, nblfac=nblfac,
                              apply_window=apply_window)

        out_x["res_um"] = out_x["res_pix"] * voxelsize_um
        out_x["ures_um"] = out_x["ures_pix"] * voxelsize_um
        out_y["res_um"] = out_y["res_pix"] * voxelsize_um
        out_y["ures_um"] = out_y["ures_pix"] * voxelsize_um

        results[name] = {"x": out_x, "y": out_y}

    print("Modregger resolution estimate")
    print("-" * 60)
    print(f"{'channel':12s} {'direction':10s} {'res (µm)':>14s} {'± unc (µm)':>14s}")
    print("-" * 60)
    for name, r in results.items():
        for direction in ("x", "y"):
            d = r[direction]
            print(f"{name:12s} {direction:10s} "
                  f"{d['res_um']:14.4f} {d['ures_um']:14.4f}")
    print("-" * 60)

    if plot:
        n_ch = len(results)
        fig, axs = plt.subplots(n_ch, 2, figsize=(13, 4.5 * n_ch),
                                squeeze=False)
        for irow, (name, r) in enumerate(results.items()):
            for icol, direction in enumerate(("x", "y")):
                d = r[direction]
                ax = axs[irow, icol]
                ax.semilogy(d["k_grid"], d["power_mean"], lw=1)
                ax.axhline(d["baseline"], color="gray", linestyle="--",
                           label=f"noise baseline ({d['baseline']:.2g})")
                ax.axhline(nblfac * d["baseline"], color="red", linestyle="--",
                           label=f"{nblfac}× baseline")
                ax.axvline(d["k_res"], color="green", linestyle=":",
                           label=f"k_res = {d['k_res']:.3f} rad/px")
                ax.axvline(highfrq, color="black", linestyle=":", alpha=0.4,
                           label=f"highfrq = {highfrq}")
                ax.set_xlabel("k (rad/sample)")
                ax.set_ylabel("|FFT|² (mean over lines)")
                ax.set_title(f"{name} — {direction} direction "
                             f"(res = {d['res_um']:.3f} ± {d['ures_um']:.3f} µm)")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()

    return results


"""
Corrected Siemens-star MTF computation for the XRESO-50HC chart.

The XRESO-50HC has 36 spokes total around the full 360 degrees, i.e. the
ring profile at radius r is periodic with 36 cycles per revolution. This
value (36) is the physical pattern frequency of the chart and does NOT
change if you mask out an angular sector -- masking only removes samples,
not periods. Always pass n_arms=36 (the chart's intrinsic spoke count).

If you prefer the line-pair convention (one black+white pair = 1 cycle),
use n_arms=18. The MTF curve shape is identical; only the x-axis scaling
differs by a factor of 2.

Public functions
----------------
compute_mtf_siemens_clean_new(...)
plot_mtf_clean_new(...)
diagnose_arm_count(...)        # FFT-based sanity check on the pattern
resolution_summary(...)        # combined MTF + Modregger report
"""


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def circular_profile_angle(img, xcenter, ycenter, radius_px, n_samples=4096):
    """Sample img along a circle of radius radius_px (in pixels)."""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    angle_deg = np.degrees(theta)

    x = xcenter + radius_px * np.cos(theta)
    y = ycenter + radius_px * np.sin(theta)

    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)

    return angle_deg, img[yi, xi]


def _fit_amplitude_lstsq(angle_rad, profile, freq):
    """
    Least-squares fit of A*cos(freq*theta) + B*sin(freq*theta) + C
    to (possibly non-uniformly sampled) data. Returns sqrt(A^2 + B^2).
    """
    angle_rad = np.asarray(angle_rad, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)

    M = np.column_stack([
        np.cos(freq * angle_rad),
        np.sin(freq * angle_rad),
        np.ones_like(angle_rad),
    ])
    coeffs, *_ = np.linalg.lstsq(M, profile, rcond=None)
    A, B, _C = coeffs
    return np.sqrt(A * A + B * B)


def modulation_lstsq(angle_deg, profile, n_arms=36, exclude_angle_ranges=None,
                     noise_freqs=None):
    """
    Estimate signal amplitude at the Siemens-star frequency, using a proper
    least-squares fit. Optionally also estimate a noise-floor amplitude at
    nearby frequencies that should contain no signal.
    """
    angle_deg = np.asarray(angle_deg)
    profile = np.asarray(profile, dtype=np.float64)

    mask = np.ones_like(angle_deg, dtype=bool)
    if exclude_angle_ranges is not None:
        for a0, a1 in exclude_angle_ranges:
            mask &= ~((angle_deg >= a0) & (angle_deg <= a1))

    angle_clean = np.deg2rad(angle_deg[mask])
    prof_clean = profile[mask]

    finite = np.isfinite(prof_clean)
    angle_clean = angle_clean[finite]
    prof_clean = prof_clean[finite]

    if len(prof_clean) < 10:
        return np.nan, np.nan

    amp_signal = _fit_amplitude_lstsq(angle_clean, prof_clean, n_arms)

    if noise_freqs is None:
        return amp_signal, np.nan

    noise_amps = []
    for f in noise_freqs:
        if f == n_arms or f <= 0:
            continue
        noise_amps.append(_fit_amplitude_lstsq(angle_clean, prof_clean, f))
    amp_noise = np.median(noise_amps) if noise_amps else np.nan

    return amp_signal, amp_noise


# ---------------------------------------------------------------------------
# main MTF computation
# ---------------------------------------------------------------------------

def compute_mtf_siemens_clean_new(
    q,
    xcenter=1380,
    ycenter=1215,
    voxelsize_um=None,
    r_min=150,
    r_max=850,
    n_radii=40,
    n_arms=36,
    exclude_radius_ranges=((450, 520), (220, 260), (100, 150)),
    exclude_angle_ranges=((127, 147),),
    n_samples=4096,
    norm_radius_range=(340, 600),
    norm_radius_range_abs=None,
    norm_mode="radius_range",
    n_low_freq_ref=5,
    estimate_noise=True,
    noise_freq_offsets=(-3, -2, +2, +3, +5, +7),
    subtract_noise_in_quadrature=False,
):
    """
    MTF from a Siemens-star image. See module docstring for details.
    """
    if voxelsize_um is None:
        raise ValueError("voxelsize_um must be provided (in micrometers)")

    if isinstance(q, cp.ndarray):
        q = cp.asnumpy(q)

    phase = np.angle(q)
    absorption = np.abs(q)

    radii_all = np.linspace(r_min, r_max, n_radii)

    def bad_radius(r):
        return any(r0 <= r <= r1 for r0, r1 in exclude_radius_ranges)

    noise_freqs = None
    if estimate_noise:
        noise_freqs = [n_arms + off for off in noise_freq_offsets if (n_arms + off) > 0]

    freqs = []
    radii_used = []
    sig_phase = []
    sig_abs = []
    noi_phase = []
    noi_abs = []

    for r in radii_all:
        if bad_radius(r):
            continue

        angle_deg, phase_prof = circular_profile_angle(
            phase, xcenter, ycenter, r, n_samples=n_samples)
        _, abs_prof = circular_profile_angle(
            absorption, xcenter, ycenter, r, n_samples=n_samples)

        amp_p, noise_p = modulation_lstsq(
            angle_deg, phase_prof, n_arms=n_arms,
            exclude_angle_ranges=exclude_angle_ranges,
            noise_freqs=noise_freqs)
        amp_a, noise_a = modulation_lstsq(
            angle_deg, abs_prof, n_arms=n_arms,
            exclude_angle_ranges=exclude_angle_ranges,
            noise_freqs=noise_freqs)

        freq = n_arms / (2 * np.pi * r * voxelsize_um)

        freqs.append(freq)
        radii_used.append(r)
        sig_phase.append(amp_p)
        sig_abs.append(amp_a)
        noi_phase.append(noise_p)
        noi_abs.append(noise_a)

    freqs = np.array(freqs)
    radii_used = np.array(radii_used)
    sig_phase = np.array(sig_phase)
    sig_abs = np.array(sig_abs)
    noi_phase = np.array(noi_phase)
    noi_abs = np.array(noi_abs)

    idx = np.argsort(freqs)
    freqs = freqs[idx]
    radii_used = radii_used[idx]
    sig_phase = sig_phase[idx]
    sig_abs = sig_abs[idx]
    noi_phase = noi_phase[idx]
    noi_abs = noi_abs[idx]

    if subtract_noise_in_quadrature and estimate_noise:
        sig_phase = np.sqrt(np.maximum(sig_phase ** 2 - noi_phase ** 2, 0.0))
        sig_abs = np.sqrt(np.maximum(sig_abs ** 2 - noi_abs ** 2, 0.0))

    if norm_mode == "radius_range":
        if norm_radius_range is None:
            raise ValueError("norm_mode='radius_range' requires norm_radius_range")
        r_lo, r_hi = norm_radius_range
        sel = (radii_used >= r_lo) & (radii_used <= r_hi)
        if not np.any(sel):
            raise ValueError(
                f"No kept radii fall inside norm_radius_range={norm_radius_range}. "
                f"Kept radii span {radii_used.min():.0f}-{radii_used.max():.0f} px.")
        norm_phase = np.nanmean(sig_phase[sel])

        if norm_radius_range_abs is not None:
            ra_lo, ra_hi = norm_radius_range_abs
            sel_abs = (radii_used >= ra_lo) & (radii_used <= ra_hi)
            if not np.any(sel_abs):
                raise ValueError(
                    f"No kept radii fall inside norm_radius_range_abs={norm_radius_range_abs}. "
                    f"Kept radii span {radii_used.min():.0f}-{radii_used.max():.0f} px.")
            norm_abs = np.nanmean(sig_abs[sel_abs])
        else:
            norm_abs = np.nanmean(sig_abs[sel])

    elif norm_mode == "low_freq":
        n_ref = min(n_low_freq_ref, len(freqs))
        norm_phase = np.nanmean(sig_phase[:n_ref])
        norm_abs = np.nanmean(sig_abs[:n_ref])

    elif norm_mode == "max":
        norm_phase = np.nanmax(sig_phase)
        norm_abs = np.nanmax(sig_abs)

    else:
        raise ValueError(f"Unknown norm_mode={norm_mode!r}")

    eps = 1e-12
    mtf_phase = sig_phase / (norm_phase + eps)
    mtf_abs = sig_abs / (norm_abs + eps)

    if estimate_noise:
        mtf_phase_noise = noi_phase / (norm_phase + eps)
        mtf_abs_noise = noi_abs / (norm_abs + eps)
    else:
        mtf_phase_noise = np.full_like(freqs, np.nan)
        mtf_abs_noise = np.full_like(freqs, np.nan)

    return {
        "freqs": freqs,
        "radii": radii_used,
        "mtf_phase": mtf_phase,
        "mtf_abs": mtf_abs,
        "mtf_phase_noise": mtf_phase_noise,
        "mtf_abs_noise": mtf_abs_noise,
        "norm_phase": norm_phase,
        "norm_abs": norm_abs,
    }


def plot_mtf_clean_new(result, voxelsize_um, show_noise=True, ylim=(0, 1.2)):
    """Plot the result of compute_mtf_siemens_clean_new."""
    freqs = result["freqs"]
    nyquist = 1 / (2 * voxelsize_um)

    plt.figure(figsize=(8, 5))

    plt.plot(freqs, result["mtf_phase"], "-o", label="Phase MTF", color="C0")
    plt.plot(freqs, result["mtf_abs"], "-o", label="Absorption MTF", color="C1")

    if show_noise and np.any(np.isfinite(result["mtf_phase_noise"])):
        plt.plot(freqs, result["mtf_phase_noise"], ":", color="C0",
                 alpha=0.7, label="Phase noise floor")
        plt.plot(freqs, result["mtf_abs_noise"], ":", color="C1",
                 alpha=0.7, label="Absorption noise floor")

    plt.axhline(0.1, linestyle="--", color="k", label="MTF 10%")
    if nyquist < freqs.max() * 1.2:
        plt.axvline(nyquist, linestyle="--", color="r", alpha=0.5, label="Nyquist")

    plt.xlabel("Spatial frequency (cycles/µm)")
    plt.ylabel("Normalized MTF")
    plt.title("MTF from Siemens star (corrected)")
    plt.grid(True)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# diagnostic: confirm n_arms from the data itself
# ---------------------------------------------------------------------------

def diagnose_arm_count(
    q,
    xcenter=1380,
    ycenter=1215,
    radii_to_probe=(300, 400, 500, 600, 700),
    exclude_angle_ranges=((127, 147),),
    n_samples=4096,
    max_freq_to_show=120,
):
    """
    Estimate the true Siemens-star pattern frequency by FFT of the ring
    profile at several radii.
    """
    if isinstance(q, cp.ndarray):
        q = cp.asnumpy(q)
    phase = np.angle(q)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    peaks = {}
    for i, r in enumerate(radii_to_probe):
        angle_deg, prof = circular_profile_angle(
            phase, xcenter, ycenter, r, n_samples=n_samples)

        mask = np.ones_like(angle_deg, dtype=bool)
        for a0, a1 in exclude_angle_ranges:
            mask &= ~((angle_deg >= a0) & (angle_deg <= a1))

        prof = prof.astype(np.float64)
        prof_for_plot = prof - np.nanmean(prof[mask])
        axs[0].plot(angle_deg, prof_for_plot, lw=1, label=f"r={r}px")

        prof_for_fft = prof.copy()
        prof_for_fft[~mask] = np.nanmean(prof_for_fft[mask])
        prof_for_fft = prof_for_fft - np.nanmean(prof_for_fft)

        F = np.abs(np.fft.rfft(prof_for_fft))
        k = np.arange(F.size)

        k_show = k[1:max_freq_to_show]
        F_show = F[1:max_freq_to_show]
        axs[1].plot(k_show, F_show, lw=1, label=f"r={r}px")

        peak_idx = np.argmax(F_show) + 1
        peaks[r] = peak_idx
        axs[1].axvline(peak_idx, color=f"C{i % 10}", linestyle=":", alpha=0.4)

    axs[0].set_xlabel("angle (degree)")
    axs[0].set_ylabel("phase (mean-removed)")
    axs[0].set_title("Ring profiles")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlabel("cycles per full revolution")
    axs[1].set_ylabel("|FFT|")
    axs[1].set_title("FFT of ring profile -- dominant peak = true n_arms")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    print("\nDominant frequency per radius (cycles/revolution):")
    for r, k in peaks.items():
        print(f"  r = {r} px  ->  peak at k = {k}")

    if len(set(peaks.values())) == 1:
        n = list(peaks.values())[0]
        print(f"\n>>> Consistent peak: n_arms = {n}")
    else:
        most_common = max(set(peaks.values()), key=list(peaks.values()).count)
        print(f"\n>>> Peaks differ between radii. Most common: n_arms = {most_common}")
        print("    If radii disagree, your center (xcenter, ycenter) may be off,")
        print("    or some radii lie outside the patterned region.")

    return peaks


# ===========================================================================
# resolution summary: combined MTF + Modregger one-call report
# ===========================================================================

# ---------------------------------------------------------------------------
# helpers (private)
# ---------------------------------------------------------------------------

def _circular_profile_angle(img, xcenter, ycenter, radius_px, n_samples=1440):
    """Sample img along a circle. Used for the diagnostic ring plots."""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    angle_deg = np.degrees(theta)
    x = xcenter + radius_px * np.cos(theta)
    y = ycenter + radius_px * np.sin(theta)
    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
    return angle_deg, img[yi, xi]


def _mtf_crossing(freqs, mtf, level):
    """First frequency where mtf crosses below level (linear interpolation)."""
    mtf = np.asarray(mtf, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    below = mtf < level
    if not np.any(below):
        return np.nan
    i = np.argmax(below)
    if i == 0:
        return float(freqs[0])
    f0, f1 = freqs[i - 1], freqs[i]
    m0, m1 = mtf[i - 1], mtf[i]
    if m0 == m1:
        return float(f1)
    return float(f0 + (level - m0) * (f1 - f0) / (m1 - m0))

def _mtf_score(freqs, mtf, fmax=1.5):
    """
    Area under the MTF curve up to fmax cycles/µm. Higher = better.

    Much more robust than the MTF=10% crossing when the curve has local
    dips or noise excursions, because integration averages them out.
    """
    freqs = np.asarray(freqs, dtype=float)
    mtf   = np.asarray(mtf,   dtype=float)
    mask = (freqs <= fmax) & np.isfinite(mtf)
    if not np.any(mask):
        return np.nan
    f = freqs[mask]
    m = np.clip(mtf[mask], 0, 1)   # cap at 1; ignore unphysical >1 bumps
    order = np.argsort(f)
    return float(np.trapz(m[order], f[order]))
# ---------------------------------------------------------------------------
# diagnostic plot: image with ring overlays + ring profiles
# ---------------------------------------------------------------------------

def _plot_image_rings_and_profiles(
    img,
    title_prefix,
    xcenter,
    ycenter,
    r_min,
    r_max,
    n_rings,
    exclude_radius_ranges,
    exclude_angle_ranges,
    n_samples,
    vmin,
    vmax,
    cbar_label,
    normalize_profiles=True,
):
    """Plot image with ring overlays and ring-profile panel side-by-side."""
    all_radii = np.linspace(r_min, r_max, 300)

    def is_bad_radius(r):
        return any(r0 <= r <= r1 for r0, r1 in exclude_radius_ranges)

    valid_radii = np.array([r for r in all_radii if not is_bad_radius(r)])
    if len(valid_radii) == 0:
        return None
    idx = np.linspace(0, len(valid_radii) - 1, n_rings).astype(int)
    radii = valid_radii[np.unique(idx)]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    im = axs[0].imshow(img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    for r in radii:
        theta = np.linspace(0, 2 * np.pi, 1000)
        axs[0].plot(xcenter + r * np.cos(theta),
                    ycenter + r * np.sin(theta), lw=1)
    axs[0].plot(xcenter, ycenter, "r+", markersize=12)

    for a0, a1 in exclude_angle_ranges:
        wedge = Wedge(center=(xcenter, ycenter), r=r_max,
                      theta1=a0, theta2=a1, width=r_max - r_min,
                      alpha=0.25, color="red")
        axs[0].add_patch(wedge)
    for r0, r1 in exclude_radius_ranges:
        annulus = Wedge(center=(xcenter, ycenter), r=r1,
                        theta1=0, theta2=360, width=(r1 - r0),
                        alpha=0.2, color="red")
        axs[0].add_patch(annulus)

    axs[0].set_title(f"{title_prefix} image with excluded regions")
    axs[0].set_xlabel("x pixels")
    axs[0].set_ylabel("y pixels")
    plt.colorbar(im, ax=axs[0], label=cbar_label)

    for a0, a1 in exclude_angle_ranges:
        axs[1].axvspan(a0, a1, alpha=0.2, color="red")
    for r in radii:
        angle_deg, prof = _circular_profile_angle(img, xcenter, ycenter, r,
                                                  n_samples=n_samples)
        prof = prof.astype(np.float64)
        if normalize_profiles:
            prof = prof - np.nanmean(prof)
            prof = prof / (np.nanstd(prof) + 1e-12)
        axs[1].plot(angle_deg, prof, lw=1.5, label=f"r={r:.0f}px")
    axs[1].set_xlabel("angle (degree)")
    axs[1].set_ylabel(f"normalized {title_prefix.lower()}")
    axs[1].set_title(f"{title_prefix} ring profiles")
    axs[1].set_xlim(0, 360)
    axs[1].grid(True)
    axs[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()
    return radii


# ---------------------------------------------------------------------------
# main reporting function
# ---------------------------------------------------------------------------
def resolution_summary(
    q,
    voxelsize_um,
    rec_params=None,

    # ---- MTF / ring-overlay parameters -----------------------------------
    xcenter=1380,
    ycenter=1215,
    r_min=50,
    r_max=850,
    n_radii=70,
    n_arms=36,
    exclude_radius_ranges=((450, 520), (220, 260), (100, 135)),
    exclude_angle_ranges=((127, 147),),
    norm_radius_range=(600, 800),

    # ---- diagnostic ring plot --------------------------------------------
    show_ring_diagnostics=True,
    n_rings_to_show=10,
    n_samples_profile=1440,
    phase_vmin=-2.0,
    phase_vmax=2.0,

    # ---- Modregger parameters --------------------------------------------
    modregger_roi=(slice(800, 1700), slice(900, 1900)),
    modregger_nblfac=2.0,
    modregger_highfrq=2.0,

    # ---- output ----------------------------------------------------------
    show_plots=True,
):
    """
    Run PHASE-ONLY MTF (Siemens star) and Modregger Fourier analysis.

    Absorption is completely excluded:
        - not calculated
        - not plotted
        - not used for normalization
        - not included in the returned results
    """

    # =======================================================================
    # Convert q to NumPy
    # =======================================================================

    if isinstance(q, cp.ndarray):
        q_np = cp.asnumpy(q)
    else:
        q_np = np.asarray(q)


    # =======================================================================
    # PHASE ONLY
    # =======================================================================

    phase_img = np.angle(q_np)


    # =======================================================================
    # PHASE RING DIAGNOSTICS
    # =======================================================================

    if show_plots and show_ring_diagnostics:

        _plot_image_rings_and_profiles(
            phase_img,
            "Phase",

            xcenter=xcenter,
            ycenter=ycenter,

            r_min=r_min,
            r_max=r_max,

            n_rings=n_rings_to_show,

            exclude_radius_ranges=exclude_radius_ranges,
            exclude_angle_ranges=exclude_angle_ranges,

            n_samples=n_samples_profile,

            vmin=phase_vmin,
            vmax=phase_vmax,

            cbar_label="phase (rad)",
        )


    # =======================================================================
    # RUN PHASE-ONLY MTF
    # =======================================================================

    mtf_result = compute_mtf_siemens_clean_new(
        q_np,

        xcenter=xcenter,
        ycenter=ycenter,

        voxelsize_um=voxelsize_um,

        r_min=r_min,
        r_max=r_max,

        n_radii=n_radii,

        n_arms=n_arms,

        exclude_radius_ranges=exclude_radius_ranges,
        exclude_angle_ranges=exclude_angle_ranges,

        norm_radius_range=norm_radius_range,

        # This function still internally supports absorption,
        # but we only use the phase outputs here.
        norm_radius_range_abs=None,

        norm_mode="radius_range",

        estimate_noise=True,
    )


    # =======================================================================
    # PHASE MTF CROSSINGS
    # =======================================================================

    f10_phase = _mtf_crossing(
        mtf_result["freqs"],
        mtf_result["mtf_phase"],
        0.10,
    )

    f50_phase = _mtf_crossing(
        mtf_result["freqs"],
        mtf_result["mtf_phase"],
        0.50,
    )


    def halfperiod_um(f):

        if (
            f
            and np.isfinite(f)
            and f > 0
        ):
            return 1.0 / (2.0 * f)

        return np.nan


    # =======================================================================
    # PHASE MTF PLOT
    # =======================================================================

    if show_plots:

        freqs = mtf_result["freqs"]
        mtf_phase = mtf_result["mtf_phase"]
        mtf_phase_noise = mtf_result["mtf_phase_noise"]

        nyquist = 1.0 / (2.0 * voxelsize_um)

        plt.figure(figsize=(8, 5))

        plt.plot(
            freqs,
            mtf_phase,
            "-o",
            label="Phase MTF",
        )

        if np.any(np.isfinite(mtf_phase_noise)):

            plt.plot(
                freqs,
                mtf_phase_noise,
                ":",
                alpha=0.7,
                label="Phase noise floor",
            )

        plt.axhline(
            0.1,
            linestyle="--",
            color="k",
            label="MTF 10%",
        )

        if nyquist < freqs.max() * 1.2:

            plt.axvline(
                nyquist,
                linestyle="--",
                color="r",
                alpha=0.5,
                label="Nyquist",
            )

        plt.xlabel("Spatial frequency (cycles/µm)")
        plt.ylabel("Normalized Phase MTF")
        plt.title("Phase MTF from Siemens star")

        plt.grid(True)
        plt.legend()

        plt.ylim(0, 1.2)

        plt.tight_layout()
        plt.show()


    # =======================================================================
    # PHASE-ONLY MODREGGER
    # =======================================================================

    mod_result = modregger_resolution(
        q=q_np,

        voxelsize_um=voxelsize_um,

        roi=modregger_roi,

        # IMPORTANT:
        # phase only
        channel="phase",

        nblfac=modregger_nblfac,

        highfrq=modregger_highfrq,

        plot=show_plots,
    )


    # =======================================================================
    # PRINT SUMMARY
    # =======================================================================

    bar = "=" * 68

    print(bar)
    print("PHASE-ONLY RESOLUTION SUMMARY")
    print(bar)


    # -----------------------------------------------------------------------
    # Reconstruction parameters
    # -----------------------------------------------------------------------

    if rec_params:

        print("\nReconstruction parameters")
        print("-" * 68)

        for k, v in rec_params.items():

            print(f"  {k:<10s} = {v}")


    print(
        f"\n  voxel size = "
        f"{voxelsize_um:.4f} µm/px"
    )


    # -----------------------------------------------------------------------
    # MTF
    # -----------------------------------------------------------------------

    print("\nPhase MTF (Siemens star)")
    print("-" * 68)

    print(
        f"  {'MTF=50% [cyc/µm]':>18s} "
        f"{'MTF=10% [cyc/µm]':>18s} "
        f"{'res 10% half-period [µm]':>26s}"
    )

    print(
        f"  {f50_phase:18.3f} "
        f"{f10_phase:18.3f} "
        f"{halfperiod_um(f10_phase):26.3f}"
    )


    # -----------------------------------------------------------------------
    # Modregger
    # -----------------------------------------------------------------------

    print(
        f"\nPhase Modregger Fourier "
        f"(SNR threshold = nblfac = {modregger_nblfac})"
    )

    print("-" * 68)

    print(
        f"  {'direction':<10s} "
        f"{'res [µm]':>12s} "
        f"{'± unc [µm]':>14s}"
    )

    for dirn in ("x", "y"):

        d = mod_result["phase"][dirn]

        print(
            f"  {dirn:<10s} "
            f"{d['res_um']:12.4f} "
            f"{d['ures_um']:14.4f}"
        )


    print("\n" + bar)


    # =======================================================================
    # RETURN PHASE-ONLY RESULTS
    # =======================================================================

    return {

        "mtf": mtf_result,

        "mtf_crossings": {

            "phase_50pct_cyc_per_um":
                f50_phase,

            "phase_10pct_cyc_per_um":
                f10_phase,

            "phase_10pct_halfperiod_um":
                halfperiod_um(f10_phase),

            "phase_score_fmax1p5":
                _mtf_score(
                    mtf_result["freqs"],
                    mtf_result["mtf_phase"],
                    fmax=1.5,
                ),
        },

        "modregger": mod_result,
    }