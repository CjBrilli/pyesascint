# ESA-Based Detection of CIRs and CMEs from Doppler Scintillation

This repository implements a complete, reproducible pipeline for detecting solar wind structures — specifically Co-rotating Interaction Regions (CIRs) and Coronal Mass Ejections (CMEs) — using Deep Space Network (ESA) Doppler tracking data of *Venus Express (VEX)*.

The method uses phase scintillation analysis and a physically motivated normalisation pipeline to isolate heliospheric disturbances from geometric and background effects.

---

## Overview

ESA Doppler measurements contain signatures of:

- Solar wind turbulence (scintillation)  
- Large-scale structures (CIRs)  
- Transient events (CMEs)  

This project extracts those signals through a three-stage correction pipeline:

1. Remove solar elongation dependence  
2. Remove CIR-scale background structure  
3. Detect transient enhancements  

The result is a robust method for identifying solar disturbances using radio tracking data.

---

## Method Summary

### 1. Phase Scintillation

Doppler residuals are converted into phase:

$\phi(t) = 2\pi \int f(t)\ dt$

Band-limited phase RMS is computed using a power spectral density (PSD) method over:

$3 \times 10^{-4} \le f \le 3 \times 10^{-2} \ \text{Hz}$

---

### 2. Elongation Correction

Phase scintillation depends strongly on solar elongation (SEP).

A quiet baseline is constructed:

$\phi_{\text{expected}} = f(\text{elongation})$

The signal is normalised:

$\text{phase ratio} = \frac{\phi_{\text{observed}}}{\phi_{\text{expected}}}$

---

### 3. CIR Detection

CIRs are identified as long-duration enhancements using:

- 12-hour smoothing  
- hysteresis thresholds  
- minimum duration constraint (> 24 hours)  

---

### 4. CME / Transient Detection

CIR background is removed:

$\text{clean signal} = \frac{\text{phase ratio}}{\text{phase smooth}}$

Transient events are identified via:

- thresholding (clean_signal > 3)  
- duration filtering (0.25–24 hours)  

---

## Repository Structure


```
pyesascint/
├── config/
├── inputs/
├── notebooks/
│   ├── 01_doppler_processing.ipynb
│   ├── 02_cme_detection.ipynb
│   ├── 03_multi_year_summary.ipynb
│   ├── 04_candidate_validation.ipynb
│   ├── 05_pride_comparison.ipynb
│   └── 06_cactus_validation.ipynb
├── src/
│   ├── io_utils.py
│   ├── doppler_utils.py
│   ├── phase_utils.py
│   ├── detection_utils.py
│   ├── geometry_utils.py
│   ├── pride_transfer_analysis.py
│   └── plot_utils.py
├── outputs/
└── README.md
```
---


---

## Workflow
### 01_doppler_processing.ipynb

Processes raw Doppler residuals and computes:

- Daily Doppler RMS
- Reconstructed phase fluctuations
- Phase scintillation metrics

### 02_cme_detection.ipynb

Identifies enhanced scintillation intervals by:

- Removing elongation dependence
- Detecting CIR regions
- Identifying transient candidate events

### 03_multi_year_summary.ipynb

Combines yearly results and produces summary statistics and figures.

### 04_candidate_validation.ipynb

Validates transient candidates using:

- Earth–Venus line-of-sight geometry
- P-point calculations
- CACTus CME catalogue information

### 05_pride_comparison.ipynb

Performs direct comparison between ESA-derived phase scintillation measurements and PRIDE observations.

### 06_cactus_validation.ipynb

Examines relationships between validated events and external CME catalogues.

## Key Outputs

- Phase RMS time series (20-minute windows)
- Normalised phase ratio (observed / expected)
- Detected disturbance intervals (start, end, duration)
- CME match table with overlap metrics

---
## Scientific Goal

The primary goal of this project is to determine whether phase scintillation measurements derived from ESA Venus Express Doppler tracking data are consistent with independent measurements obtained through the Planetary Radio Interferometry and Doppler Experiment (PRIDE).

If successful, PRIDE observations could provide a complementary source of space-weather information for future planetary missions.

## Key Parameters

| Parameter            | Value                | Description |
|---------------------|---------------------|-------------|
| Phase band          | 3e-4 – 3e-2 Hz      | Scintillation frequency range |
| Window size         | 20 min              | Phase computation |
| Step size           | 10 min              | Window overlap |
| CIR smoothing       | 12 hr               | Background scale |
| CIR thresholds      | 1.4 / 1.2           | Hysteresis |
| Transient threshold | 3.0                 | CME detection |
| Transient step      | 20 min              | Critical for correct results |

---

## Input Data Format

### Horizons Solar Elongation File

The pipeline requires a plain text JPL Horizons ephemeris file containing solar elongation values.

The parser expects:

- a data block between `$$SOE` and `$$EOE`
- timestamps in the first two columns
- solar elongation stored in the Horizons `S-O-T /r` field

In each data row, the elongation is read as the numeric value immediately before `/L` or `/T`.

Example:

```text
$$SOE
2010-Jan-01 00:00  ...  2.6407 /L  ...
2010-Jan-02 00:00  ...  2.4124 /L  ...
$$EOE
```
## Outputs

For each year:

**Daily metrics**
- Doppler RMS vs SEP  

**Phase windows**
- Band-limited phase scintillation  

**CIR catalogue**
- Start/end times  
- Duration  
- Signal strength  

**Transient event catalogue**
- CME-like detections  
- Peak and median signals  

---

## Validation

The pipeline is validated by:

- Reproducing consistent CIR counts across years  
- Stable transient detection after parameter alignment  
- Physically consistent separation of:
  - elongation effects  
  - CIR background  
  - transient disturbances  

---

## Key Insight

This method demonstrates that ESA Doppler tracking data contains measurable and separable signatures of solar wind structures, including CMEs.

---

## Requirements

- Python 3.10+
- NumPy
- pandas
- matplotlib
- SciPy

---

## Usage

The notebooks are typically run in sequence:

1. 01_doppler_processing.ipynb
2. 02_cme_detection.ipynb
3. 03_multi_year_summary.ipynb
4. 04_candidate_validation.ipynb
5. 05_pride_comparison.ipynb
6. 06_cactus_validation.ipynb

Year-specific parameters are controlled through the files in `config/`.
## Author

Caleb

Developed as part of astrophysical data analysis using ESA tracking data of Venus Express.

Affiliation: University of Tasmania, Honours Research Project
Supervisor: Dr. Guifré Molera Calvés

---
## License

This project is licensed under the MIT License. See the LICENSE file for details.
---
