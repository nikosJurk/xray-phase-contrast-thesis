# xray-phase-contrast-thesis

Code for my MSc thesis: multi-distance CTF and coded-aperture nano
holography for X-ray phase-contrast imaging at the DanMAX beamline
(MAX IV synchrotron).

The thesis compares several phase-retrieval techniques (CTF, homoCTF,
Paganin, sglDstCTF) using both real synchrotron measurements and a
matching forward-model simulation, to characterize how well each method
resolves fine structural detail in weakly absorbing, phase-shifting
samples. It also includes a separate coded-aperture nano-holography
pipeline (real data + matching simulation), evaluated the same way.

## What's in this repo

| File | What it does |
|---|---|
| `Multi_distance_phase_retrieval_Experimental_Data.py` | Processes the real, measured X-ray detector images from DanMAX and reconstructs the phase using several multi-distance techniques. See [README_EXPERIMENTAL.md](README_EXPERIMENTAL.md) for details. |
| `Multi_distance_phase_retrieval_Simulated_Data.py` | Builds a known synthetic sample, simulates the detector images, and runs the same reconstruction/analysis pipeline — used to validate the real-data results against a known ground truth. See [README_SIMULATION.md](README_SIMULATION.md) for details. |
| `Nano_Holography_with_CA_Experimental_Data.py` | Reconstructs coded-aperture (CA) nano-holography data of the same Siemens star target from real DanMAX scans. See [README_NANO_HOLOGRAPHY.md](README_NANO_HOLOGRAPHY.md) for details. |
| `Nano_Holography_with_CA_Simulated_Data.py` | Forward-simulates and reconstructs coded-aperture nano-holography data from a known synthetic sample, using the same geometry as the real CA experiment. See [README_NANO_HOLOGRAPHY.md](README_NANO_HOLOGRAPHY.md) for details. |
| `PBI_phase_retrieval_functions.py`, `usefull_functions.py`, `utils.py`, `utils_sim.py`, `rec.py`, `rec_sim.py`, `cuda_kernels.py`, `cuda_kernels_sim.py`, `chunking.py`, `chunking_sim.py` | Shared helper modules used by the scripts above. `*_sim` variants are used by the simulation scripts; the rest are shared or used by the real-data scripts. |

## Getting started

Each main script needs its `CONFIG` section (or the matching parameters
near the top) pointed at your own data — see the individual READMEs
linked above for the exact variable names and what each script expects.

### Requirements

- `numpy`, `scipy`, `matplotlib`, `h5py`, `pandas`
- Optional GPU acceleration: `cupy`, `pyfftw` — the phase-contrast scripts
  fall back to CPU automatically if unavailable; the nano-holography
  reconstruction step requires a CUDA-capable GPU via `cupy`
- Optional: a NIST optical-constants pipeline, or `xraylib` as a
  fallback, for the simulation scripts' material properties

## Thesis

This code accompanies my MSc thesis, *Development of nano tomographic
phase reconstruction at DanMAX beamline* (DTU Engineering Physics, 2026).
