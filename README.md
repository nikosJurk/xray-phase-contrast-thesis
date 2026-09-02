# xray-phase-contrast-thesis

Code for my MSc thesis: multi-distance CTF and coded-aperture nano
holography for X-ray phase-contrast imaging at the DanMAX beamline
(MAX IV synchrotron).

The thesis compares several phase-retrieval techniques (CTF, homoCTF,
Paganin, sglDstCTF) using both real synchrotron measurements and a
matching forward-model simulation, to characterize how well each method
resolves fine structural detail in weakly absorbing, phase-shifting
samples.

## What's in this repo

| File | What it does |

|(X_ray_phase_contrast_retrieval.py) | Processes the real, measured X-ray detector images from DanMAX and 
              reconstructs the phase using several techniques. See (README_EXPERIMENTAL.md) for details. |
|(X_ray_phase_contrast_simulation.py) | Builds a known synthetic sample, simulates the detector images, and
              runs the same reconstruction/analysis pipeline — used to validate the real-data results against 
              a known ground truth. See [README_SIMULATION.md](README_SIMULATION.md) for details. |
| PBI_phase_retrieval_functions.py, usefull_functions.py, utils.py, utils_sim.py, rec.py, cuda_kernels.py, chunking.py 
            | Shared helper modules used by both scripts above. |

## Getting started

Both main scripts need their `CONFIG` section (or the matching environment
variables) pointed at your own data — see the individual READMEs linked
above for the exact variable names and what each script expects.

### Requirements
- numpy, scipy, matplotlib, h5py, pandas
- Optional GPU acceleration: cupy, pyfftw (both scripts fall back to
  CPU automatically if unavailable)
- Optional: a NIST optical-constants pipeline or xraylib, for the
  simulation script's material properties

## Thesis

This code accompanies my MSc thesis, *Development of nano tomographic
phase reconstruction at DanMAX beamline* (DTU Engineering Physics, 2026).
