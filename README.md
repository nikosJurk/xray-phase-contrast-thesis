# xray-phase-contrast-thesis
Code for MSc thesis: multi-distance CTF and coded-aperture nano holography for X-ray phase-contrast imaging at DanMAX

Code for my MSc thesis on X-ray phase-contrast imaging at the DanMAX beamline (MAX IV synchrotron). The goal of the thesis is to recover the phase of an X-ray wave after it passes through a sample — not just how much the sample absorbed the beam, but how much it shifted it — using several different reconstruction techniques, and to see how well each one performs on real experimental data.

What's in X_ray_phase_contrast_retrieval.py

This script takes raw X-ray detector images and turns them into phase-contrast reconstructions, then measures how good those reconstructions are. It's meant to be read top to bottom — every section is numbered and commented, and each one builds on the previous one.

Roughly, it does this:

Loads the raw data. Five detector images taken at different sample-to-detector distances, plus a dark frame (no beam) and a flat frame (beam, no sample) used for calibration.
Cleans up the images. Removes the detector's background signal and normalizes for uneven beam illumination (standard "flat-field correction"). Also checks that the beam and detector were stable during the measurement.
Lines everything up. Because each of the five images was taken at a different distance, they also come out at slightly different magnifications. The script rescales them all to the same effective magnification, then aligns them pixel-for-pixel using cross-correlation, fixing both large and sub-pixel misalignments.
Reconstructs the phase. This is the core of the script. It applies several different reconstruction algorithms to the aligned data:
a CTF (Contrast Transfer Function) method, which is fast and doesn't need to know what the sample is made of,
and a few methods that do assume a known material (Paganin, homoCTF, sglDstCTF), which can give better results if that assumption holds.
Each method is tried using different combinations of the five distances (just one distance, two combined, all five combined, etc.), to see how much using more distances actually helps.

You'll also need: numpy, scipy, matplotlib, h5py, pandas. GPU acceleration (cupy, pyfftw) is optional — the script falls back to CPU automatically if they're not installed. It also relies on a few helper modules (utils, rec, usefull_functions, PBI_phase_retrieval_functions) that live alongside it.
Measures how good each reconstruction is. Sharper images resolve finer detail. The script quantifies this with two independent methods (MTF and a Fourier-domain resolution estimate), plus a contrast-to-noise measurement, and puts everything into result tables and comparison plots so the different techniques can be judged side by side.
