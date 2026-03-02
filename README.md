# Cosmic-Dawn-Spherical-Collapse

Python pipeline for evaluating nonlinear spherical collapse thresholds ($\delta_c$) and predicting high-redshift halo abundances using Press-Schechter and Sheth-Tormen mass functions. 

It is specifically designed to test **environment-activated gravitational coupling** models as a potential solution to the massive, early galaxies recently observed by the James Webb Space Telescope (JWST) at redshifts $z \sim 15-25$.

## What This Code Does
Standard modified gravity models often inadvertently alter the background expansion or linear perturbation growth of the universe, violating late-time constraints. This pipeline implements a "Class B" mechanism:
* **The Bump Gate:** An environment proxy based on Newtonian depth ($\tau_{vir}$) that localizes gravitational enhancement *strictly* to the intermediate stages of halo collapse.
* **Epoch Envelope:** A redshift gate that ensures the modification only operates during Cosmic Dawn, returning to standard General Relativity (GR) behavior for the modern universe ($z \le 6$).
* **Clean Baseline:** It successfully modifies the nonlinear collapse threshold ($\delta_c$) while keeping the $H(z)$ background and $D(z)$ linear growth completely GR-like.

## How to Run It
The primary script is `run_two_modes.py` (which orchestrates the `S3_Field_Sim_modes.py` engine). Running this script will generate diagnostic plots showing the bump gate shape, the collapse-threshold distortion, and the cumulative abundance ratios $n_{MG}(>M) / n_{GR}(>M)$.

## Foundational Theory & Citation
This computational pipeline is the numerical testing ground for the **$S^3$ Field Theory Framework**, a single-field ontology where gravity and matter are emergent properties of a topological field. 

If you use this code or find the environment-activated mechanism useful for your JWST research, please refer to the foundational theoretical manuscripts and cite the official Zenodo DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18091803.svg)](https://doi.org/10.5281/zenodo.18091803)

* **Main Project & Manuscripts:** [https://doi.org/10.5281/zenodo.18091803](https://doi.org/10.5281/zenodo.18091803)
