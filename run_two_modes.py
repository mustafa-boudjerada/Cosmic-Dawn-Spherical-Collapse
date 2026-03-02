#!/usr/bin/env python3
"""
Run two Paper-1 configurations back-to-back:

1) THERMAL_GLOBAL  : legacy global-thermal behavior (thermal window enters linear/background via mu_max_from_background_tau_z).
2) ENV_BUMP (GR-lin): environment-only thermal (thermal window only enters collapse via bump gate; linear extrapolation uses GR).

This script writes results into:
  outputs_THERMAL_GLOBAL/
  outputs_ENV_BUMP_GR/
"""
import os
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_PATH = HERE / "S3_Field_Sim_modes.py"

def load_sim():
    spec = importlib.util.spec_from_file_location("s3sim", str(SIM_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    # Common targets (edit freely)
    z_target = float(os.environ.get("Z_TARGET", "20.0"))
    masses = tuple(float(x) for x in os.environ.get("MASSES_MSUN", "1e9,1e10,1e11").split(","))
    k_plot_h = tuple(float(x) for x in os.environ.get("K_PLOT_H", "0.1,1.0").split(","))

    # --- Run 1: THERMAL_GLOBAL ---
    sim = load_sim()
    sim.MODEL_MODE = "THERMAL_GLOBAL"
    sim.DELTA_C_LINEAR_MU = "K"  # legacy linear extrap using mu_of_a_k
    sim.run(outdir="outputs_THERMAL_GLOBAL", z_target=z_target, masses_Msun=masses, k_plot_h=k_plot_h)

    # --- Run 2: ENV_BUMP (GR-linear) ---
    sim = load_sim()
    sim.MODEL_MODE = "ENV_BUMP"
    sim.DELTA_C_LINEAR_MU = "GR"  # truth-clean delta_c definition
    # Optional: bump-gate knobs via env vars (ENV_A, TAUN_ON, TAUN_OFF, ...)
    sim.run(outdir="outputs_ENV_BUMP_GR", z_target=z_target, masses_Msun=masses, k_plot_h=k_plot_h)

    print("\nDone.")
    print("  -> outputs_THERMAL_GLOBAL/")
    print("  -> outputs_ENV_BUMP_GR/")

if __name__ == "__main__":
    main()
