#!/usr/bin/env python3
"""
Traveling boosted defect: explicit Skyrme time evolution + leakage diagnostic.

Fundamental field:
  Phi(x) is a real 4-vector field with pointwise constraint ||Phi||=1.

Constraint enforcement:
  - Tangent projection: v <- v - (Phi·v) Phi
  - Geodesic drift step on S^3:
      Phi <- cos(s) Phi + sin(s) vhat,  s = dt * |v|,  vhat = v/|v|

Spatial energy density:
  e2 = (F^2/2) Tr X,    X_ij = ∂iPhi·∂jPhi
  e4 = (lam/32) [ (Tr X)^2 - Tr(X^2) ]
  v  = m^2 F^2 (1 - Phi·Phi_vac)

EOM implemented in "wave-map" style on the interior:
  Phi_tt = Δ Phi + m^2 Phi_vac + (1/F^2) ∂k J_k    (then tangent projection)
with
  J_k = (lam/8) [ A ∂kPhi - X_kn ∂nPhi ],  A = Tr X

Boundary handling (default):
  - Periodic in z.
  - Dirichlet-to-vacuum on x/y faces.

Leakage diagnostic:
  Build a 1D operator O(z) by averaging over x,y:
    op=deriv  -> O(z)=<|∂zPhi|^2>_{x,y}
    op=energy -> O(z)=<e_spatial>_{x,y}
  Window and FFT to P(k), map x=k/P_used, compute leakage outside [0,1].

Outputs:
  - JSON summary with energy/momentum diagnostics and stored spectra at t=0 and t=end
  - optional final Phi snapshot
"""
import argparse, os, math, json, time
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# Basic S3 operations
# -----------------------------
def normalize_S3(phi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return phi / (phi.norm(dim=-1, keepdim=True) + eps)

def project_tangent(phi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (phi * v).sum(dim=-1, keepdim=True) * phi


# -----------------------------
# Vacuum estimate and resampling
# -----------------------------
def estimate_vacuum_from_boundary(phi: torch.Tensor, shell: int = 3) -> torch.Tensor:
    N = phi.shape[0]
    s = int(max(1, shell))
    mask = torch.zeros((N, N, N), dtype=torch.bool, device=phi.device)
    mask[:s, :, :] = True
    mask[-s:, :, :] = True
    mask[:, :s, :] = True
    mask[:, -s:, :] = True
    mask[:, :, :s] = True
    mask[:, :, -s:] = True
    v = phi[mask].mean(dim=0)
    return v / (v.norm() + 1e-12)

def downsample_to(phi: torch.Tensor, Nsim: int) -> torch.Tensor:
    vol = phi.permute(3, 0, 1, 2).unsqueeze(0)  # (1,4,N,N,N)
    lin = torch.linspace(-1, 1, Nsim, device=phi.device, dtype=phi.dtype)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid = torch.stack([Z, Y, X], dim=-1).unsqueeze(0)  # (1,Nsim,Nsim,Nsim,3) in (z,y,x)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    phi_ds = out.squeeze(0).permute(1, 2, 3, 0).contiguous()
    return normalize_S3(phi_ds)

def enforce_dirichlet_xy(phi: torch.Tensor, Phi_vac: torch.Tensor) -> torch.Tensor:
    phi[0, :, :, :] = Phi_vac
    phi[-1, :, :, :] = Phi_vac
    phi[:, 0, :, :] = Phi_vac
    phi[:, -1, :, :] = Phi_vac
    return phi


# -----------------------------
# "Boost snapshot" contraction
# -----------------------------
def boost_contract_z(phi_rest: torch.Tensor, gamma: float, Phi_vac: torch.Tensor,
                    vacblend: bool = True) -> torch.Tensor:
    N = phi_rest.shape[0]
    vol = phi_rest.permute(3, 0, 1, 2).unsqueeze(0)  # (1,4,N,N,N)
    lin = torch.linspace(-1, 1, N, device=phi_rest.device, dtype=phi_rest.dtype)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")

    Zin = (gamma * Z).clamp(-1, 1)
    grid = torch.stack([Zin, Y, X], dim=-1).unsqueeze(0)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    phi_b = out.squeeze(0).permute(1, 2, 3, 0).contiguous()

    if vacblend:
        zabs = Z.abs()
        z1 = 1.0 / max(gamma, 1e-12)
        z0 = 0.90 * z1
        t = ((zabs - z0) / (z1 - z0 + 1e-12)).clamp(0, 1)
        w = 0.5 * (1.0 + torch.cos(math.pi * t)).unsqueeze(-1)
        phi_b = normalize_S3(w * phi_b + (1.0 - w) * Phi_vac.view(1, 1, 1, 4))

    return normalize_S3(phi_b)


# -----------------------------
# Finite differences
# -----------------------------
def roll_z(t: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(t, shifts=shift, dims=2)

def d_center_x(phi: torch.Tensor, dx: float) -> torch.Tensor:
    return (phi[2:, 1:-1, :, :] - phi[:-2, 1:-1, :, :]) / (2.0 * dx)

def d_center_y(phi: torch.Tensor, dx: float) -> torch.Tensor:
    return (phi[1:-1, 2:, :, :] - phi[1:-1, :-2, :, :]) / (2.0 * dx)

def d_center_z_periodic(phi_xy: torch.Tensor, dx: float) -> torch.Tensor:
    return (roll_z(phi_xy, -1) - roll_z(phi_xy, +1)) / (2.0 * dx)

def laplacian_periodic_z(phi: torch.Tensor, dx: float) -> torch.Tensor:
    phi_xy = phi[1:-1, 1:-1, :, :]
    lap = (
        phi[2:, 1:-1, :, :] + phi[:-2, 1:-1, :, :] +
        phi[1:-1, 2:, :, :] + phi[1:-1, :-2, :, :] +
        roll_z(phi_xy, -1) + roll_z(phi_xy, +1) -
        6.0 * phi_xy
    ) / (dx * dx)
    return lap

def div_center_x(Jx: torch.Tensor, dx: float) -> torch.Tensor:
    return (Jx[2:, :, :, :] - Jx[:-2, :, :, :]) / (2.0 * dx)

def div_center_y(Jy: torch.Tensor, dx: float) -> torch.Tensor:
    return (Jy[:, 2:, :, :] - Jy[:, :-2, :, :]) / (2.0 * dx)

def div_center_z_periodic(Jz: torch.Tensor, dx: float) -> torch.Tensor:
    return (roll_z(Jz, -1) - roll_z(Jz, +1)) / (2.0 * dx)


# -----------------------------
# Skyrme acceleration
# -----------------------------
# -----------------------------
# Full covariant Sigma+Skyrme acceleration (velocity-dependent)
# -----------------------------
@torch.no_grad()
def canonical_pi(phi: torch.Tensor, phidot: torch.Tensor, dx: float,
                 F0: float, lam: float) -> torch.Tensor:
    """
    Canonical momentum pi = dL/d(phi_t) for the covariant sigma+Skyrme model (signature -,+,+,+),
    with the action sign chosen so that the sigma-model piece is L2 = (F0^2/2)(|phi_t|^2 - |∇phi|^2).

    Identity used on the interior:
      pi = (F0^2 + (lam/8)|∇phi|^2) * phidot  -  (lam/8) * Σ_i (phidot·∂i phi) ∂i phi
    """
    alpha = lam / 8.0
    phi_xy = phi[1:-1, 1:-1, :, :]
    v_xy = phidot[1:-1, 1:-1, :, :]

    dx_phi = d_center_x(phi, dx)
    dy_phi = d_center_y(phi, dx)
    dz_phi = d_center_z_periodic(phi_xy, dx)

    grad2 = (dx_phi*dx_phi).sum(dim=-1) + (dy_phi*dy_phi).sum(dim=-1) + (dz_phi*dz_phi).sum(dim=-1)
    A = (F0*F0) + alpha * grad2

    dotx = (v_xy * dx_phi).sum(dim=-1)
    doty = (v_xy * dy_phi).sum(dim=-1)
    dotz = (v_xy * dz_phi).sum(dim=-1)

    pi_xy = (A.unsqueeze(-1) * v_xy) - alpha * (
        dotx.unsqueeze(-1) * dx_phi +
        doty.unsqueeze(-1) * dy_phi +
        dotz.unsqueeze(-1) * dz_phi
    )

    pi = torch.zeros_like(phi)
    pi[1:-1, 1:-1, :, :] = pi_xy
    return pi

def _solve_rank3_woodbury(A: torch.Tensor,
                          dx_phi: torch.Tensor, dy_phi: torch.Tensor, dz_phi: torch.Tensor,
                          b: torch.Tensor,
                          alpha: float,
                          eps: float = 1e-30) -> torch.Tensor:
    """
    Solve (A I - alpha Σ_i d_i d_i^T) x = b with d_i = (dx_phi,dy_phi,dz_phi), per-site,
    using a 3x3 Woodbury solve without constructing any (...,4,4) tensors.

    All inputs are core-shaped:
      A:      (...,)
      dx_phi: (...,4)
      dy_phi: (...,4)
      dz_phi: (...,4)
      b:      (...,4)
    """
    Ainv = 1.0 / (A + eps)
    beta = alpha * Ainv  # alpha/A

    # Gram matrix G = D^T D (3x3) components
    gxx = (dx_phi*dx_phi).sum(dim=-1)
    gyy = (dy_phi*dy_phi).sum(dim=-1)
    gzz = (dz_phi*dz_phi).sum(dim=-1)
    gxy = (dx_phi*dy_phi).sum(dim=-1)
    gxz = (dx_phi*dz_phi).sum(dim=-1)
    gyz = (dy_phi*dz_phi).sum(dim=-1)

    # S = I - (alpha/A) G  (3x3 symmetric)
    s11 = 1.0 - beta * gxx
    s22 = 1.0 - beta * gyy
    s33 = 1.0 - beta * gzz
    s12 = -beta * gxy
    s13 = -beta * gxz
    s23 = -beta * gyz

    # rhs = D^T b
    r1 = (dx_phi*b).sum(dim=-1)
    r2 = (dy_phi*b).sum(dim=-1)
    r3 = (dz_phi*b).sum(dim=-1)

    # Invert symmetric 3x3 using adjugate formula (vectorized)
    c11 = (s22*s33 - s23*s23)
    c12 = -(s12*s33 - s13*s23)
    c13 = (s12*s23 - s13*s22)
    c22 = (s11*s33 - s13*s13)
    c23 = -(s11*s23 - s12*s13)
    c33 = (s11*s22 - s12*s12)

    det = s11*c11 + s12*c12 + s13*c13
    det_inv = 1.0 / (det + eps)

    y1 = det_inv * (c11*r1 + c12*r2 + c13*r3)
    y2 = det_inv * (c12*r1 + c22*r2 + c23*r3)
    y3 = det_inv * (c13*r1 + c23*r2 + c33*r3)

    corr = (dx_phi * y1.unsqueeze(-1) +
            dy_phi * y2.unsqueeze(-1) +
            dz_phi * y3.unsqueeze(-1))

    x = b * Ainv.unsqueeze(-1) + (alpha * (Ainv*Ainv)).unsqueeze(-1) * corr
    return x

@torch.no_grad()
def accel_skyrme(phi: torch.Tensor, phidot: torch.Tensor, Phi_vac: torch.Tensor, dx: float,
                 F0: float, lam: float, m: float) -> torch.Tensor:
    """
    Full covariant EOM in explicit second-order form (velocity-dependent):

      ∂t pi(phi, phidot) + ∂i J^i(phi, phidot) + dV/dphi = Λ phi

    with:
      V = m^2 F0^2 (1 - phi·Phi_vac)
      dV/dphi = - m^2 F0^2 Phi_vac

    The constraint term Λ is handled by tangent projection at the end.
    """
    alpha = lam / 8.0

    # interior slabs
    phi_xy = phi[1:-1, 1:-1, :, :]
    v_xy = phidot[1:-1, 1:-1, :, :]

    # spatial derivatives of phi
    dx_phi = d_center_x(phi, dx)
    dy_phi = d_center_y(phi, dx)
    dz_phi = d_center_z_periodic(phi_xy, dx)

    # spatial derivatives of v (needed for Q term)
    dx_v = d_center_x(phidot, dx)
    dy_v = d_center_y(phidot, dx)
    dz_v = d_center_z_periodic(v_xy, dx)

    # invariants
    Xxx = (dx_phi*dx_phi).sum(dim=-1)
    Xyy = (dy_phi*dy_phi).sum(dim=-1)
    Xzz = (dz_phi*dz_phi).sum(dim=-1)
    Xxy = (dx_phi*dy_phi).sum(dim=-1)
    Xxz = (dx_phi*dz_phi).sum(dim=-1)
    Xyz = (dy_phi*dz_phi).sum(dim=-1)

    v2 = (v_xy*v_xy).sum(dim=-1)
    grad2 = Xxx + Xyy + Xzz
    trX = -v2 + grad2

    dotx = (v_xy*dx_phi).sum(dim=-1)
    doty = (v_xy*dy_phi).sum(dim=-1)
    dotz = (v_xy*dz_phi).sum(dim=-1)

    Sx = Xxx.unsqueeze(-1)*dx_phi + Xxy.unsqueeze(-1)*dy_phi + Xxz.unsqueeze(-1)*dz_phi
    Sy = Xxy.unsqueeze(-1)*dx_phi + Xyy.unsqueeze(-1)*dy_phi + Xyz.unsqueeze(-1)*dz_phi
    Sz = Xxz.unsqueeze(-1)*dx_phi + Xyz.unsqueeze(-1)*dy_phi + Xzz.unsqueeze(-1)*dz_phi

    # With our sign convention, J^i carries an overall minus so that lam->0 gives phi_tt = Δphi + m^2 Phi_vac.
    Jx = -((F0*F0)*dx_phi + alpha*(trX.unsqueeze(-1)*dx_phi + dotx.unsqueeze(-1)*v_xy - Sx))
    Jy = -((F0*F0)*dy_phi + alpha*(trX.unsqueeze(-1)*dy_phi + doty.unsqueeze(-1)*v_xy - Sy))
    Jz = -((F0*F0)*dz_phi + alpha*(trX.unsqueeze(-1)*dz_phi + dotz.unsqueeze(-1)*v_xy - Sz))

    # divergence on the core (needs 2-layer margin)
    dJx = div_center_x(Jx, dx)[:, 1:-1, :, :]              # (N-4,N-4,N,4)
    dJy = div_center_y(Jy, dx)[1:-1, :, :, :]              # (N-4,N-4,N,4)
    dJz = div_center_z_periodic(Jz, dx)[1:-1, 1:-1, :, :]  # (N-4,N-4,N,4)
    divJ = dJx + dJy + dJz

    # pi = A v - alpha Σ (v·d_i) d_i,  where A = F0^2 + alpha |∇phi|^2
    A = (F0*F0) + alpha * grad2

    # A_t = 2 alpha Σ_i (d_i · d_{i,t}) with d_{i,t} = ∂i v
    At = 2.0 * alpha * (
        (dx_phi*dx_v).sum(dim=-1) +
        (dy_phi*dy_v).sum(dim=-1) +
        (dz_phi*dz_v).sum(dim=-1)
    )

    # Q term from ∂t pi = (A I - alpha Σ d_i d_i^T) v_t + Q
    Q = (At.unsqueeze(-1) * v_xy) - alpha * (
        ( (v_xy*dx_v).sum(dim=-1).unsqueeze(-1) * dx_phi ) +
        ( (v_xy*dy_v).sum(dim=-1).unsqueeze(-1) * dy_phi ) +
        ( (v_xy*dz_v).sum(dim=-1).unsqueeze(-1) * dz_phi ) +
        ( dotx.unsqueeze(-1) * dx_v ) +
        ( doty.unsqueeze(-1) * dy_v ) +
        ( dotz.unsqueeze(-1) * dz_v )
    )

    # dV/dphi = - m^2 F0^2 Phi_vac
    dV = -(m*m) * (F0*F0) * Phi_vac.view(1, 1, 1, 4).expand_as(phi_xy)

    # Core slices aligned with divJ: [1:-1,1:-1] of the interior slab tensors
    Qc = Q[1:-1, 1:-1, :, :]
    dVc = dV[1:-1, 1:-1, :, :]

    RHS = -(Qc + divJ + dVc)

    # Core A and derivatives
    Ac = A[1:-1, 1:-1, :]
    dx_c = dx_phi[1:-1, 1:-1, :, :]
    dy_c = dy_phi[1:-1, 1:-1, :, :]
    dz_c = dz_phi[1:-1, 1:-1, :, :]

    vtt_core = _solve_rank3_woodbury(Ac, dx_c, dy_c, dz_c, RHS, alpha)

    a = torch.zeros_like(phi)
    phi_core = phi[2:-2, 2:-2, :, :]
    a[2:-2, 2:-2, :, :] = project_tangent(phi_core, vtt_core)
    return a


# -----------------------------
# Energy and momentum
# -----------------------------
@torch.no_grad()
def energy_total(phi: torch.Tensor, phidot: torch.Tensor, Phi_vac: torch.Tensor,
                 dx: float, F0: float, lam: float, m: float) -> tuple[float, float, float]:
    """
    Report:
      Esp_static: original static spatial energy (continuity with earlier plots)
      K_simple:   0.5 F0^2 ∫ |phidot|^2  (continuity)
      E_tot:      canonical energy (positive) from H = pi·phidot - L with
                 L = -(F0^2/2) TrX - (lam/32)[(TrX)^2 - TrX2] - V
    """
    alpha = lam / 8.0
    phi_xy = phi[1:-1, 1:-1, :, :]
    v_xy = phidot[1:-1, 1:-1, :, :]

    dx_phi = d_center_x(phi, dx)
    dy_phi = d_center_y(phi, dx)
    dz_phi = d_center_z_periodic(phi_xy, dx)

    # invariants
    Xxx = (dx_phi*dx_phi).sum(dim=-1)
    Xyy = (dy_phi*dy_phi).sum(dim=-1)
    Xzz = (dz_phi*dz_phi).sum(dim=-1)
    Xxy = (dx_phi*dy_phi).sum(dim=-1)
    Xxz = (dx_phi*dz_phi).sum(dim=-1)
    Xyz = (dy_phi*dz_phi).sum(dim=-1)

    grad2 = Xxx + Xyy + Xzz
    v2 = (v_xy*v_xy).sum(dim=-1)

    # Potential
    dot0 = (phi_xy * Phi_vac.view(1, 1, 1, 4)).sum(dim=-1)
    Vden = (m*m) * (F0*F0) * (1.0 - dot0)

    # Original static energy (spatial only)
    trX_sp = grad2
    trX2_sp = (Xxx*Xxx + Xyy*Xyy + Xzz*Xzz + 2.0*(Xxy*Xxy + Xxz*Xxz + Xyz*Xyz))
    e2_sp = 0.5 * (F0*F0) * trX_sp
    e4_sp = (lam / 32.0) * (trX_sp * trX_sp - trX2_sp)
    Esp_static = float((e2_sp + e4_sp + Vden).sum().item() * (dx**3))

    # Simple kinetic (continuity)
    K_simple = float((0.5 * (F0*F0) * v2).sum().item() * (dx**3))

    # Full covariant energy
    trX = -v2 + grad2
    X0x = (v_xy*dx_phi).sum(dim=-1)
    X0y = (v_xy*dy_phi).sum(dim=-1)
    X0z = (v_xy*dz_phi).sum(dim=-1)
    Sij2 = trX2_sp
    trX2 = (v2*v2) - 2.0*(X0x*X0x + X0y*X0y + X0z*X0z) + Sij2

    L2 = -0.5 * (F0*F0) * trX
    L4 = -(lam / 32.0) * (trX*trX - trX2)
    Lden = L2 + L4 - Vden  # minus V

    A = (F0*F0) + alpha * grad2
    pi_xy = (A.unsqueeze(-1) * v_xy) - alpha * (
        (X0x.unsqueeze(-1) * dx_phi) +
        (X0y.unsqueeze(-1) * dy_phi) +
        (X0z.unsqueeze(-1) * dz_phi)
    )
    Hden = (pi_xy * v_xy).sum(dim=-1) - Lden
    E_tot = float(Hden.sum().item() * (dx**3))

    return Esp_static, K_simple, E_tot

@torch.no_grad()
def momentum_Pz(phi: torch.Tensor, phidot: torch.Tensor, dx: float, F0: float, lam: float) -> float:
    """
    Canonical momentum Pz = ∫ pi · ∂z phi  d^3x
    """
    pi = canonical_pi(phi, phidot, dx, F0, lam)
    pi_xy = pi[1:-1, 1:-1, :, :]
    dz_phi = d_center_z_periodic(phi[1:-1, 1:-1, :, :], dx)
    dens = (pi_xy * dz_phi).sum(dim=-1)
    return float(dens.sum().item() * (dx ** 3))
# Leakage diagnostic
# -----------------------------
def trapz_uniform(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    dx = float(x[1] - x[0])
    return float(dx * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

def leakage_from_lattice(phi: torch.Tensor, P_used: float, dx: float, pad_fft: int,
                         op: str, Phi_vac: torch.Tensor,
                         F0: float, lam: float, m: float) -> dict:
    with torch.no_grad():
        phi_xy = phi[1:-1, 1:-1, :, :]
        if op == "deriv":
            dz_phi = d_center_z_periodic(phi_xy, dx)
            O1 = (dz_phi * dz_phi).sum(dim=-1).mean(dim=(0, 1)).cpu().numpy()
        elif op == "energy":
            dx_phi = d_center_x(phi, dx)
            dy_phi = d_center_y(phi, dx)
            dz_phi = d_center_z_periodic(phi_xy, dx)

            Xxx = (dx_phi * dx_phi).sum(dim=-1)
            Xyy = (dy_phi * dy_phi).sum(dim=-1)
            Xzz = (dz_phi * dz_phi).sum(dim=-1)
            Xxy = (dx_phi * dy_phi).sum(dim=-1)
            Xxz = (dx_phi * dz_phi).sum(dim=-1)
            Xyz = (dy_phi * dz_phi).sum(dim=-1)

            trX = Xxx + Xyy + Xzz
            trX2 = (Xxx*Xxx + Xyy*Xyy + Xzz*Xzz + 2.0*(Xxy*Xxy + Xxz*Xxz + Xyz*Xyz))

            dot0 = (phi_xy * Phi_vac.view(1, 1, 1, 4)).sum(dim=-1)
            e2 = 0.5 * (F0 * F0) * trX
            e4 = (lam / 32.0) * (trX * trX - trX2)
            vden = (m * m) * (F0 * F0) * (1.0 - dot0)
            esp = e2 + e4 + vden

            O1 = esp.mean(dim=(0, 1)).cpu().numpy()
        else:
            raise ValueError("op must be 'deriv' or 'energy'")

    O1c = O1 - O1.mean()
    win = np.hanning(len(O1c))
    O1w = O1c * win

    nfft = int(max(len(O1w), int(pad_fft)))
    Fk = np.fft.rfft(O1w, n=nfft)
    Pk = (Fk.conj() * Fk).real

    kz = (2.0 * np.pi) * np.fft.rfftfreq(nfft, d=dx)

    P_used = float(max(P_used, 1e-12))
    x = kz / P_used

    area = trapz_uniform(x, Pk.astype(np.float64))
    Pn = (Pk / (area + 1e-30)).astype(np.float64)

    mask = (x >= 0.0) & (x <= 1.0)
    inside = trapz_uniform(x[mask], Pk[mask].astype(np.float64)) if mask.any() else 0.0
    leak = float((area - inside) / area) if area > 0 else float("nan")

    return {
        "leak": leak,
        "x": x.astype(np.float64).tolist(),
        "Pn": Pn.tolist(),
    }


# -----------------------------
# Recentering helper
# -----------------------------
def z_center_proxy(phi: torch.Tensor, dx: float, L: float) -> float:
    with torch.no_grad():
        phi_xy = phi[1:-1, 1:-1, :, :]
        dz_phi = d_center_z_periodic(phi_xy, dx)
        W = (dz_phi * dz_phi).sum(dim=-1).mean(dim=(0, 1))
        z = torch.linspace(-L, L, phi.shape[2], device=phi.device, dtype=phi.dtype)
        return float(((W * z).sum() / (W.sum() + 1e-30)).item())

def recenter_z(phi: torch.Tensor, phidot: torch.Tensor, dx: float, zc: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    shift = int(round(zc / dx))
    if shift != 0:
        phi = roll_z(phi, -shift)
        phidot = roll_z(phidot, -shift)
    return phi, phidot, shift


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi0", required=True, help="Path to phi0_N*.pt (tensor N,N,N,4).")
    ap.add_argument("--L", type=float, required=True, help="Half-box length for SIM grid: domain [-L,+L].")
    ap.add_argument("--gamma", type=float, default=3.0, help="Lorentz factor for snapshot contraction (>=1).")
    ap.add_argument("--Nsim", type=int, default=49, help="Simulation grid size after downsample (odd recommended).")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--Nt", type=int, default=200)
    ap.add_argument("--diag_every", type=int, default=10)
    ap.add_argument("--recenter_every", type=int, default=50)
    ap.add_argument("--pad_fft", type=int, default=512)

    ap.add_argument("--F0", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=0.0)

    ap.add_argument("--v_scale", type=float, default=1.0, help="Scale factor applied to v=sqrt(1-gamma^-2).")
    ap.add_argument("--op", default="deriv", choices=["deriv", "energy"], help="Operator channel for O(z).")

    ap.add_argument("--vac_shell", type=int, default=3)
    ap.add_argument("--no_vacblend", action="store_true")
    ap.add_argument("--no_dirichlet_xy", action="store_true")

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--threads", type=int, default=0)

    ap.add_argument("--out_dir", default=os.path.join("outputs", "traveling_defect"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_final_phi", action="store_true")
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = ("_" + args.tag) if args.tag else ""

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.gamma < 1.0:
        raise ValueError("--gamma must be >= 1")

    phi0 = torch.load(args.phi0, map_location="cpu")
    if not (isinstance(phi0, torch.Tensor) and phi0.ndim == 4 and phi0.shape[-1] == 4):
        raise ValueError("phi0 must be a torch tensor of shape (N,N,N,4).")

    phi0 = normalize_S3(phi0.to(device=device, dtype=dtype))
    Phi_vac = estimate_vacuum_from_boundary(phi0, shell=args.vac_shell).to(device=device, dtype=dtype)

    phi_rest = downsample_to(phi0, args.Nsim)
    if not args.no_dirichlet_xy:
        enforce_dirichlet_xy(phi_rest, Phi_vac)

    phi = boost_contract_z(phi_rest, args.gamma, Phi_vac, vacblend=(not args.no_vacblend))
    if not args.no_dirichlet_xy:
        enforce_dirichlet_xy(phi, Phi_vac)

    dx = (2.0 * args.L) / (args.Nsim - 1)
    v = math.sqrt(max(0.0, 1.0 - 1.0 / (args.gamma * args.gamma)))
    v_eff = float(args.v_scale * v)

    dz_phi = d_center_z_periodic(phi[1:-1, 1:-1, :, :], dx)
    phidot = torch.zeros_like(phi)
    phidot[1:-1, 1:-1, :, :] = -v_eff * dz_phi
    phidot = project_tangent(phi, phidot)

    if not args.no_dirichlet_xy:
        phidot[0, :, :, :] = 0
        phidot[-1, :, :, :] = 0
        phidot[:, 0, :, :] = 0
        phidot[:, -1, :, :] = 0

    Esp0, K0, E0 = energy_total(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)
    Pz0 = momentum_Pz(phi, phidot, dx, args.F0, args.lam)

    kz_nyquist = math.pi / dx
    xmax0 = kz_nyquist / (abs(Pz0) + 1e-30)
    print(f"[INFO] dx={dx:.6f}  kz_nyquist~{kz_nyquist:.6f}  x_max(start)~{xmax0:.3f}  v_eff={v_eff:.6f}  op={args.op}")

    leak0_reason = ""
    if xmax0 < 1.0:
        leak0 = {"leak": float("nan"), "x": [], "Pn": []}
        leak0_reason = f"x_max_below_1  x_max={xmax0:.3f}"
        print(f"[WARN] leakage not testable at start: reason={leak0_reason}")
    else:
        leak0 = leakage_from_lattice(
            phi, abs(Pz0), dx,
            pad_fft=args.pad_fft, op=args.op,
            Phi_vac=Phi_vac, F0=args.F0, lam=args.lam, m=args.m
        )

    a = accel_skyrme(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)

    t_hist, E_hist, Pz_hist, zc_hist = [], [], [], []

    def diag_record(tnow: float):
        Esp, K, Et = energy_total(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)
        Pz = momentum_Pz(phi, phidot, dx, args.F0, args.lam)
        zc = z_center_proxy(phi, dx, args.L)
        t_hist.append(float(tnow))
        E_hist.append(float(Et))
        Pz_hist.append(float(Pz))
        zc_hist.append(float(zc))

    diag_record(0.0)

    t0 = time.time()
    for n in range(args.Nt):
        phidot = project_tangent(phi, phidot + 0.5 * args.dt * a)

        if not args.no_dirichlet_xy:
            phidot[0, :, :, :] = 0
            phidot[-1, :, :, :] = 0
            phidot[:, 0, :, :] = 0
            phidot[:, -1, :, :] = 0

        speed = (phidot * phidot).sum(dim=-1, keepdim=True).sqrt()
        vhat = phidot / (speed + 1e-30)
        s = args.dt * speed
        phi = torch.cos(s) * phi + torch.sin(s) * vhat
        phi = normalize_S3(phi)

        if not args.no_dirichlet_xy:
            enforce_dirichlet_xy(phi, Phi_vac)

        phidot = project_tangent(phi, phidot)
        if not args.no_dirichlet_xy:
            phidot[0, :, :, :] = 0
            phidot[-1, :, :, :] = 0
            phidot[:, 0, :, :] = 0
            phidot[:, -1, :, :] = 0

        a_new = accel_skyrme(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)

        phidot = project_tangent(phi, phidot + 0.5 * args.dt * a_new)
        if not args.no_dirichlet_xy:
            phidot[0, :, :, :] = 0
            phidot[-1, :, :, :] = 0
            phidot[:, 0, :, :] = 0
            phidot[:, -1, :, :] = 0

        a = a_new

        if args.recenter_every > 0 and (n + 1) % args.recenter_every == 0:
            zc = z_center_proxy(phi, dx, args.L)
            phi, phidot, _ = recenter_z(phi, phidot, dx, zc)
            if not args.no_dirichlet_xy:
                enforce_dirichlet_xy(phi, Phi_vac)
                phidot[0, :, :, :] = 0
                phidot[-1, :, :, :] = 0
                phidot[:, 0, :, :] = 0
                phidot[:, -1, :, :] = 0
            a = accel_skyrme(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)

        if args.diag_every > 0 and (n + 1) % args.diag_every == 0:
            diag_record((n + 1) * args.dt)

    runtime_s = float(time.time() - t0)

    Esp1, K1, E1 = energy_total(phi, phidot, Phi_vac, dx, args.F0, args.lam, args.m)
    Pz1 = momentum_Pz(phi, phidot, dx, args.F0, args.lam)
    xmax1 = kz_nyquist / (abs(Pz1) + 1e-30)
    print(f"[INFO] x_max(end)~{xmax1:.3f}")

    leak1_reason = ""
    if xmax1 < 1.0:
        leak1 = {"leak": float("nan"), "x": [], "Pn": []}
        leak1_reason = f"x_max_below_1  x_max={xmax1:.3f}"
        print(f"[WARN] leakage not testable at end: reason={leak1_reason}")
    else:
        leak1 = leakage_from_lattice(
            phi, abs(Pz1), dx,
            pad_fft=args.pad_fft, op=args.op,
            Phi_vac=Phi_vac, F0=args.F0, lam=args.lam, m=args.m
        )

    out = {
        "params": {
            "phi0": args.phi0,
            "L": args.L,
            "dx": dx,
            "gamma": args.gamma,
            "v": v,
            "v_scale": args.v_scale,
            "v_eff": v_eff,
            "Nsim": args.Nsim,
            "dt": args.dt,
            "Nt": args.Nt,
            "F0": args.F0,
            "lam": args.lam,
            "m": args.m,
            "vac_shell": args.vac_shell,
            "vacblend": (not args.no_vacblend),
            "dirichlet_xy": (not args.no_dirichlet_xy),
            "device": str(device),
            "dtype": args.dtype,
            "pad_fft": args.pad_fft,
            "op": args.op,
        },
        "runtime_s": runtime_s,
        "start": {
            "Esp": Esp0, "K": K0, "E": E0, "Pz": Pz0,
            "x_max": xmax0,
            "leak": leak0["leak"],
            "leak_reason": leak0_reason,
        },
        "end": {
            "Esp": Esp1, "K": K1, "E": E1, "Pz": Pz1,
            "x_max": xmax1,
            "leak": leak1["leak"],
            "leak_reason": leak1_reason,
        },
        "drift": {
            "E_frac": (E1 - E0) / (E0 + 1e-30),
            "Pz_frac": (Pz1 - Pz0) / (Pz0 + 1e-30),
        },
        "history": {
            "t": t_hist,
            "E": E_hist,
            "Pz": Pz_hist,
            "zc": zc_hist,
        },
        "spectrum_t0": leak0,
        "spectrum_tend": leak1,
        "Phi_vac": Phi_vac.detach().cpu().tolist(),
    }

    out_path = os.path.join(args.out_dir, f"traveling_defect_summary{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] wrote summary: {out_path}")
    print(f"[INFO] runtime_s={runtime_s:.3f}")
    print(f"[INFO] start: E={E0:.6e}, Pz={Pz0:.6e}, leak={out['start']['leak']}")
    print(f"[INFO] end:   E={E1:.6e}, Pz={Pz1:.6e}, leak={out['end']['leak']}")
    print(f"[INFO] drift: E_frac={out['drift']['E_frac']:.3e}, Pz_frac={out['drift']['Pz_frac']:.3e}")

    if args.save_final_phi:
        phi_path = os.path.join(args.out_dir, f"phi_travel_end{tag}.pt")
        torch.save(phi.detach().cpu(), phi_path)
        print(f"[OK] saved final Phi: {phi_path}")


if __name__ == "__main__":
    main()
