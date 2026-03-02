#!/usr/bin/env python3
"""
Option #3 (Parton / DIS) – proof-of-concept quasi-PDF from a boosted S^3 field snapshot.

Postulate-1 compliant representation:
  Phi is a real 4-vector field on the lattice with pointwise normalization ||Phi||=1.

What this script does:
  1) Load relaxed soliton Phi0 (tensor shape N,N,N,4).
  2) Construct a Lorentz-contracted snapshot along z via resampling z -> gamma*z.
  3) Measure O ~ (∂_z Phi · ∂_z Phi) (representation-invariant scalar).
  4) Compute z-power spectrum of O, map k -> x = k / Pz_est, save JSON + binned plot.

Notes:
  - Diagnostic only (no time evolution yet).
  - Designed to reveal under-resolution ("pancake") and guide N, L, gamma choices.
  - Uses vacuum-blending outside the boosted support to reduce FFT ringing.
"""

import argparse
import json
import os
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def normalize_S3(phi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return phi / (phi.norm(dim=-1, keepdim=True) + eps)


def central_diff_z(phi: torch.Tensor, dx: float) -> torch.Tensor:
    # phi shape: (N, N, N, 4)
    return (phi[:, :, 2:, :] - phi[:, :, :-2, :]) / (2.0 * dx)


def estimate_core_width_along_z(phi: torch.Tensor, Phi0: torch.Tensor, dx: float, thresh: float = 0.1) -> float:
    """
    Estimate rest-frame core width along z at the central line (x=y=mid) using indicator 1 - Phi·Phi0 > thresh.
    Returns width in physical units (same units as dx, L).
    """
    N = phi.shape[0]
    mid = N // 2
    dot0 = (phi[mid, mid, :, :] * Phi0).sum(dim=-1)  # (N,)
    indicator = (1.0 - dot0) > thresh
    idx = torch.where(indicator)[0]
    if idx.numel() < 2:
        return 0.0
    zmin = idx.min().item()
    zmax = idx.max().item()
    return (zmax - zmin) * dx


def resample_boost_contract_z(phi_rest: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Build a Lorentz-contracted snapshot at t=0 for a rest-static profile:
      Phi_boost(x,y,z) = Phi_rest(x,y,z' = gamma*z)

    Uses torch grid_sample (trilinear). Assumes phi_rest defined on normalized grid [-1,1]^3.
    In torch grid_sample for 5D:
      grid[..., 0] indexes W axis (our z index),
      grid[..., 1] indexes H axis (our y index),
      grid[..., 2] indexes D axis (our x index).

    We do NOT clamp Z_in; outside [-1,1] we sample zeros and then smoothly blend to vacuum (Phi0)
    near |Z| ~ 1/gamma to avoid sharp edges -> FFT ringing.
    """
    assert phi_rest.ndim == 4 and phi_rest.shape[-1] == 4
    device = phi_rest.device
    dtype = phi_rest.dtype
    N = phi_rest.shape[0]

    # (N,N,N,4) -> (1,4,D,H,W) = (1,4,Nx,Ny,Nz)
    vol = phi_rest.permute(3, 0, 1, 2).unsqueeze(0)  # (1,4,N,N,N)

    lin = torch.linspace(-1.0, 1.0, N, device=device, dtype=dtype)
    z_out = lin  # W
    y_out = lin  # H
    x_out = lin  # D

    # meshgrid in order (D,H,W)
    X, Y, Z = torch.meshgrid(x_out, y_out, z_out, indexing="ij")  # each (N,N,N)

    # input z' = gamma*z (in normalized coords)
    Z_in = gamma * Z

    # grid last dim is (W,H,D) = (z,y,x)
    grid = torch.stack([Z_in, Y, X], dim=-1).unsqueeze(0)  # (1,N,N,N,3)

    vol_boost = F.grid_sample(
        vol,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    phi_boost = vol_boost.squeeze(0).permute(1, 2, 3, 0).contiguous()  # (N,N,N,4)

    # Smooth vacuum blend near the boosted support edge |Z| ~ 1/gamma
    Phi0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 1, 4)
    zabs = Z.abs()

    # Blend window in OUTPUT coords:
    # inside |Z| <= z0: keep sampled field
    # outside |Z| >= z1: pure vacuum
    z1 = 1.0 / max(gamma, 1e-12)
    z0 = 0.90 * z1  # 10% cosine taper
    t = (zabs - z0) / (z1 - z0 + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    w = 0.5 * (1.0 + torch.cos(math.pi * t))  # w=1 inside, w=0 at/after edge
    w = w.unsqueeze(-1)  # (N,N,N,1)

    phi_boost = normalize_S3(w * phi_boost + (1.0 - w) * Phi0)
    return phi_boost


def energy_rest(phi: torch.Tensor, F0: float, lam: float, m: float, Phi0: torch.Tensor, dx: float) -> float:
    """
    Rest energy consistent with extractor convention (interior cells, central diffs).
    e2 = (F^2/2) Tr X
    e4 = (lam/32) ((Tr X)^2 - Tr(X^2))
    v  = m^2 F^2 (1 - Phi·Phi0)
    """
    dpx = (phi[2:, 1:-1, 1:-1, :] - phi[:-2, 1:-1, 1:-1, :]) / (2.0 * dx)
    dpy = (phi[1:-1, 2:, 1:-1, :] - phi[1:-1, :-2, 1:-1, :]) / (2.0 * dx)
    dpz = (phi[1:-1, 1:-1, 2:, :] - phi[1:-1, 1:-1, :-2, :]) / (2.0 * dx)

    Xxx = (dpx * dpx).sum(dim=-1)
    Xyy = (dpy * dpy).sum(dim=-1)
    Xzz = (dpz * dpz).sum(dim=-1)
    Xxy = (dpx * dpy).sum(dim=-1)
    Xxz = (dpx * dpz).sum(dim=-1)
    Xyz = (dpy * dpz).sum(dim=-1)

    trX = Xxx + Xyy + Xzz
    trX2 = (Xxx * Xxx + Xyy * Xyy + Xzz * Xzz + 2.0 * (Xxy * Xxy + Xxz * Xxz + Xyz * Xyz))

    Phi_int = phi[1:-1, 1:-1, 1:-1, :]
    dot0 = (Phi_int * Phi0).sum(dim=-1)

    vden = (m * m) * (F0 * F0) * (1.0 - dot0)
    e2 = 0.5 * (F0 * F0) * trX
    e4 = (lam / 32.0) * (trX * trX - trX2)

    E = (e2 + e4 + vden).sum().item() * (dx ** 3)
    return float(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi0", required=True, help="Path to phi0_N*.pt (torch tensor shape N,N,N,4).")
    ap.add_argument("--N", type=int, required=True, help="Grid size N (must match phi0).")
    ap.add_argument("--L", type=float, required=True, help="Half-box length: domain is [-L, +L].")
    ap.add_argument("--gamma", type=float, default=2.0, help="Lorentz factor (>=1).")
    ap.add_argument("--F0", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--out_dir", default=os.path.join("outputs", "dis_parton"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--x_min", type=float, default=0.06, help="High-pass cutoff in x: zero modes with x < x_min.")
    ap.add_argument("--bin", type=int, default=2, help="Bin size for saved/plot binned spectrum (>=1).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    dx = (2.0 * args.L) / (args.N - 1)

    phi = torch.load(args.phi0, map_location="cpu")
    if not (isinstance(phi, torch.Tensor) and phi.ndim == 4 and phi.shape[-1] == 4):
        raise ValueError("phi0 must be a torch Tensor of shape (N,N,N,4).")
    if phi.shape[0] != args.N:
        raise ValueError(f"phi0 N={phi.shape[0]} but you passed --N {args.N}. They must match.")
    phi = normalize_S3(phi.to(device=device, dtype=dtype))

    Phi0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

    if args.gamma < 1.0:
        raise ValueError("--gamma must be >= 1")

    w0 = estimate_core_width_along_z(phi, Phi0, dx, thresh=0.1)
    v = math.sqrt(max(0.0, 1.0 - 1.0 / (args.gamma * args.gamma)))
    wz = w0 / args.gamma if w0 > 0 else 0.0
    npts = (wz / dx) if (wz > 0 and dx > 0) else 0.0

    E0 = energy_rest(phi, args.F0, args.lam, args.m, Phi0, dx)
    Pz_est = args.gamma * v * E0  # c=1 units

    print(f"[INFO] N={args.N}  L={args.L}  dx={dx:.6f}  gamma={args.gamma:.3f}  v={v:.6f}")
    print(f"[INFO] Rest core width estimate w0≈{w0:.6f}  -> contracted wz≈{wz:.6f}  -> points across core≈{npts:.2f}")
    if w0 > 0 and npts < 8.0:
        print("[WARN] Boosted core has < 8 grid points across its width. Expect under-resolution.")
    print(f"[INFO] Rest energy E0≈{E0:.6e}  -> Pz_est≈{Pz_est:.6e}")

    phi_boost = resample_boost_contract_z(phi, args.gamma)

    dphi_dz = central_diff_z(phi_boost, dx)  # (N,N,N-2,4)
    O = (dphi_dz * dphi_dz).sum(dim=-1)      # (N,N,N-2)
    O = ((1.0 - v) ** 2 / 2.0) * O           # overall constant factor

    Nz = O.shape[2]
    O = O - O.mean(dim=2, keepdim=True)

    win = torch.hann_window(Nz, device=O.device, dtype=O.dtype)
    O = O * win.view(1, 1, Nz)


        # Collapse to 1D along z by averaging over x,y, then rFFT over z
    O1 = O.mean(dim=(0, 1))                 # (Nz,)
    O1_fft = torch.fft.rfft(O1, dim=0)      # (Nz_r,)
    Pk = (O1_fft.abs()**2).detach().cpu().numpy()

    kz = (2.0 * math.pi) * np.fft.rfftfreq(Nz, d=dx)  # (Nz_r,)
    xfrac = kz / (Pz_est + 1e-30)


    raw_Pk = Pk.copy()
    x_min = float(args.x_min)
    Pk[xfrac < x_min] = 0.0

    BIN_SAVE = int(max(1, args.bin))
    nb = (len(Pk) // BIN_SAVE) * BIN_SAVE
    if nb < BIN_SAVE:
        raise RuntimeError("Not enough FFT bins to bin-average; reduce --bin or increase N/L.")
    Pk_bin = Pk[:nb].reshape(-1, BIN_SAVE).mean(axis=1)
    x_bin = xfrac[:nb].reshape(-1, BIN_SAVE).mean(axis=1)

    tag = ("_" + args.tag) if args.tag else ""
    out_json = os.path.join(args.out_dir, f"quasi_pdf_N{args.N}_L{args.L}_g{args.gamma:.3f}{tag}.json")

    payload = {
        "phi0_path": args.phi0,
        "N": args.N,
        "L_half": args.L,
        "dx": dx,
        "gamma": args.gamma,
        "v": v,
        "F0": args.F0,
        "lam": args.lam,
        "m": args.m,
        "E0_rest": E0,
        "Pz_est": Pz_est,
        "w0_est": w0,
        "wz_est": wz,
        "core_points_est": npts,
        "x_min": x_min,
        "bin": BIN_SAVE,
        "kz": kz.tolist(),
        "x": xfrac.tolist(),
        "Pk_raw": raw_Pk.tolist(),
        "Pk": Pk.tolist(),
        "x_bin": x_bin.tolist(),
        "Pk_bin": Pk_bin.tolist(),
        "note": "Pk is the z-power spectrum of O ~ d_+Phi·d_+Phi from a contracted snapshot (no time evolution yet).",
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote: {out_json}")

    out_png = out_json.replace(".json", ".png")
    plt.figure()
    plt.plot(x_bin, Pk_bin)
    plt.xlabel("x = k_z / Pz_est")
    plt.ylabel("Pk (arb., binned)")
    plt.title(f"Quasi-PDF POC  N={args.N}  L={args.L}  gamma={args.gamma:.2f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Wrote: {out_png}")


if __name__ == "__main__":
    main()
