#!/usr/bin/env python3
"""
S3-Field Framework Cosmology Simulation
---------------------------------------
Implements the background evolution, linear growth, and spherical collapse
within the S3-Field effective theory. This engine computes the "Banded Kinetic"
coupling history and its impact on the halo mass function at high redshift.

Output:
- Background scalar field evolution tau(z)
- Effective gravitational coupling history mu(z)
- Linear growth factor ratios relative to General Relativity
- Halo abundance boost factors (Press-Schechter)
"""

import numpy as np, math, pandas as pd, os
import matplotlib.pyplot as plt

# -------------------------
# Thermal window mechanism: Z_s(T) dip near T_c = q m_rip
# This is used as a background shift of the kinetic stiffness:
#   f_eff(tau,z) = f_tau(tau) + (Zs(T(z)) - 1)
# so that in the weak-field limit (tau -> 0, wall -> 0):
#   f_eff -> Zs(T)  and  mu_max -> 1/sqrt(Zs).
# -------------------------
T0_K = 2.7255            # CMB temperature today in Kelvin
eV_to_K = 11604.51812    # 1 eV in Kelvin (k_B = 1)
m_rip_eV = 1.0e-3        # ripple mass gap (example: 1 meV)
q_factor = 5.0           # criticality factor q
Tc_K = q_factor * m_rip_eV * eV_to_K

deltaZ = 0.40            # stiffness dip depth (Zmin = 1 - deltaZ)
sigma_frac = 0.20        # window width as fraction of Tc

# =========================
# Model mode switches
# =========================
# THERMAL_GLOBAL: thermal window shifts the background/linear sector via f_eff_tau_z (legacy behavior).
# ENV_BUMP: thermal window acts only inside collapsing regions through an environmental bump gate; background/linear stay GR-like.
MODEL_MODE = os.environ.get('MODEL_MODE', 'THERMAL_GLOBAL')  # 'THERMAL_GLOBAL' or 'ENV_BUMP'

# In delta_c definition, what linear mu to use for the linear extrapolation step:
#   'GR' -> mu_lin = 1 (truth-clean baseline)
#   'K'  -> mu_lin = mu_of_a_k(a, k_eff, ...) (scale-dependent linear mu)
DELTA_C_LINEAR_MU = os.environ.get('DELTA_C_LINEAR_MU', 'GR')

# Environmental bump-gate parameters (used only when MODEL_MODE='ENV_BUMP')
ENV_A      = float(os.environ.get('ENV_A', '0.60'))
TAUN_ON    = float(os.environ.get('TAUN_ON', '2.0e-7'))
TAUN_OFF   = float(os.environ.get('TAUN_OFF', '3.3e-6'))
TAUN_P_ON  = int(os.environ.get('TAUN_P_ON', '4'))
TAUN_Q_OFF = int(os.environ.get('TAUN_Q_OFF', '4'))
ENV_COMBINE = os.environ.get('ENV_COMBINE', 'add')  # 'add' or 'mul'

# Small numerical guard
MU_GUARD_EPS = 1e-12


def T_of_z_K(z):
    return T0_K * (1.0 + z)

def Zs_of_T(TK):
    sig = max(sigma_frac * Tc_K, 1e-12)
    return 1.0 - deltaZ * math.exp(-0.5*((TK - Tc_K)/sig)**2)


def mu_th_of_z(z):
    """Pure thermal factor from stiffness dip only: mu_th(z)=1/sqrt(Zs(T(z)))."""
    return 1.0/math.sqrt(max(Zs_of_T(T_of_z_K(z)), MU_GUARD_EPS))

def bump_gate(tauN):
    """Environmental bump gate B(tauN): turns on at TAUN_ON and screens off beyond TAUN_OFF."""
    x = (max(tauN, 0.0)/max(TAUN_ON, MU_GUARD_EPS))**TAUN_P_ON
    on = x/(1.0 + x)
    y = (max(tauN, 0.0)/max(TAUN_OFF, MU_GUARD_EPS))**TAUN_Q_OFF
    off = 1.0/(1.0 + y)
    return on*off

def f_eff_tau_z(tau, z):
    # Temperature-shifted kinetic stiffness.
    return max(f_tau(tau) + (Zs_of_T(T_of_z_K(z)) - 1.0), 1e-20)

def mu_max_from_background_tau_z(tau_inf, z):
    """Maximal QS strength in the massless limit.

    - THERMAL_GLOBAL: includes thermal shift through f_eff_tau_z (legacy behavior).
    - ENV_BUMP: ignores thermal in background/linear sector (truth-clean baseline).
    """
    if MODEL_MODE.upper() == "THERMAL_GLOBAL":
        return 1.0/math.sqrt(f_eff_tau_z(tau_inf, z))
    return 1.0/math.sqrt(max(f_tau(tau_inf), MU_GUARD_EPS))
# -------------------------
# Physical Constants & Units
# -------------------------
G_SI = 6.67430e-11
c_SI = 299792458.0
Msun = 1.98847e30
Mpc  = 3.085677581e22

H0_km_s_Mpc = 70.0
h = H0_km_s_Mpc/100.0
H0_SI = (H0_km_s_Mpc*1000.0)/Mpc

# -------------------------
# Standard Cosmology Background (LCDM Reference)
# -------------------------
Omega_m0 = 0.3
Omega_L0 = 0.7

def E(a):
    """Hubble function E(a) = H(a)/H0."""
    return math.sqrt(Omega_m0/a**3 + Omega_L0)

def Omega_m(a):
    return (Omega_m0/a**3)/(E(a)**2)

def dlnH_dlna(a):
    return -1.5*Omega_m(a)

rho_crit0 = 3*H0_SI**2/(8*math.pi*G_SI)
rho_m0 = Omega_m0*rho_crit0

# -------------------------
# Banded Kinetic Function f(tau)
# -------------------------
# Parameters defining the kinetic valley (Enhancement) 
# and screening walls (Brake) used in the cosmology sector.
A = 0.40
tau_v = 1e-7
m = 6
tau_s = 3e-7
p = 8
B = 1e3
tau_w = 2e-2

def v_on(t):
    x = (t/tau_v)**m
    return x/(1+x)

def v_off(t):
    y = (t/tau_s)**p
    return 1/(1+y)

def wall(t):
    z = (t/tau_w)**4
    return z/(1+z)

def f_of_t(t):  # t >= 0
    return 1 - A*v_on(t)*v_off(t) + B*wall(t)

def f_tau(tau):
    return f_of_t(abs(tau))

def fp_tau(tau):
    """Numerical derivative of f(|tau|)."""
    t = abs(tau)
    if t == 0.0:
        return 0.0
    eps = t*1e-4 + 1e-12
    f1 = f_of_t(t+eps)
    f0 = f_of_t(max(t-eps, 0.0))
    deriv = (f1-f0)/(2*eps)
    return deriv*(1.0 if tau>=0 else -1.0)

# -------------------------
# Effective Coupling mu(tau) via Covariant Matching
# Relation: tau_N = integral sqrt(f(u)) du
# -------------------------
t_grid = np.concatenate(([0.0], np.logspace(-14, -1, 12000)))
sqrtf_grid = np.sqrt(np.array([f_of_t(t) for t in t_grid]))
F_sqrt = np.zeros_like(t_grid)
F_sqrt[1:] = np.cumsum(0.5*(sqrtf_grid[1:]+sqrtf_grid[:-1])*(t_grid[1:]-t_grid[:-1]))

def inv_F_sqrt(y):
    """Inverse mapping to find scalar field value from Newtonian potential."""
    if y <= 0:
        return 0.0
    if y >= F_sqrt[-1]:
        return float(t_grid[-1])
    return float(np.interp(y, F_sqrt, t_grid))

def mu_QS_from_tauN_with_tauinf(tauN, tau_inf):
    """Calculate QS coupling mu given boundary field tau_inf."""
    t_inf = abs(tau_inf)
    y_inf = float(np.interp(t_inf, t_grid, F_sqrt))
    t_R = inv_F_sqrt(y_inf + max(tauN, 0.0))
    return 1.0/math.sqrt(f_of_t(t_R))

def mu_max_from_background_tau(tau_inf):
    """(Deprecated) Original mu_max without thermal window."""
    return 1.0/math.sqrt(f_tau(tau_inf))


# -------------------------
# Background Field Evolution tau(z)
# Implements Two-Stage Tracking (Valley -> Vacuum)
# -------------------------
tau_tr  = 1e-7   # Target (High-z)
tau_vac = 1e-4   # Target (Low-z)
m_tr = 2.0       # Coupling strength to matter
m_v  = 0.3       # Coupling strength to vacuum

z_switch = 3.0
a_switch = 1/(1+z_switch)
rho_star = rho_m0/(a_switch**3)

def W_of_a(a):
    rho = rho_m0/a**3
    return rho/(rho+rho_star)

def Uprime_eff(a, tau):
    W = W_of_a(a)
    return W*(m_tr**2)*(tau - tau_tr) + (1-W)*(m_v**2)*(tau - tau_vac)

def integrate_tau_bg(a_ini=1/200, a_fin=1.0, n=9000, tau_ini=tau_tr, tau_x_ini=0.0):
    xs = np.linspace(math.log(a_ini), math.log(a_fin), n)
    tau = float(tau_ini)
    tau_x = float(tau_x_ini)
    taus = np.empty(n)
    for i,x in enumerate(xs):
        taus[i] = tau
        def rhs(xx, y):
            tau, tau_x = y
            a = math.exp(xx)
            H = E(a)
            dlnH = dlnH_dlna(a)
            f = f_tau(tau)
            fp = fp_tau(tau)
            tau_xx = -(dlnH+3.0)*tau_x - 0.5*(fp/max(f,1e-18))*tau_x**2 - Uprime_eff(a, tau)/(max(f,1e-18)*H**2)
            return np.array([tau_x, tau_xx], float)
        y = np.array([tau, tau_x], float)
        if i < n-1:
            h = xs[i+1]-xs[i]
            k1 = rhs(x, y)
            k2 = rhs(x+0.5*h, y+0.5*h*k1)
            k3 = rhs(x+0.5*h, y+0.5*h*k2)
            k4 = rhs(x+h, y+h*k3)
            y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            tau, tau_x = float(y[0]), float(y[1])
    return np.exp(xs), taus

# -------------------------
# Scale Dependent Coupling & Slip
# -------------------------
m_eff_phys_per_Mpc = 2.2

def s_of_a_k(a, k_Mpc, a_tau, tau_bar):
    tau_inf = float(np.interp(a, a_tau, np.abs(tau_bar)))
    z = 1.0/a - 1.0
    mu_max = mu_max_from_background_tau_z(tau_inf, z)

    # If you want the 'Brake' to produce mu<1 in growth, set ALLOW_SUPPRESSION=True.
    ALLOW_SUPPRESSION = False
    s0 = (mu_max - 1.0) if ALLOW_SUPPRESSION else max(mu_max - 1.0, 0.0)

    return s0 * (k_Mpc**2)/(k_Mpc**2 + (a*m_eff_phys_per_Mpc)**2)

def mu_of_a_k(a, k_Mpc, a_tau, tau_bar):
    return 1.0 + s_of_a_k(a, k_Mpc, a_tau, tau_bar)

def eta_of_a_k(a, k_Mpc, a_tau, tau_bar):
    s = s_of_a_k(a, k_Mpc, a_tau, tau_bar)
    return -2.0*s/(1.0+s)

def Sigma_of_a_k(a, k_Mpc, a_tau, tau_bar):
    mu = mu_of_a_k(a, k_Mpc, a_tau, tau_bar)
    eta = eta_of_a_k(a, k_Mpc, a_tau, tau_bar)
    return mu*(2.0+eta)/2.0

# -------------------------
# Linear Growth Solver
# -------------------------
def integrate_growth(mu_func, a_ini=1/200, a_fin=1.0, n=6000):
    xs = np.linspace(math.log(a_ini), math.log(a_fin), n)
    d = a_ini
    dp = d
    deltas = np.empty(n); dps = np.empty(n)
    for i,x in enumerate(xs):
        deltas[i] = d; dps[i] = dp
        def rhs(xx, y):
            d, dp = y
            a = math.exp(xx)
            A1 = 2.0 + dlnH_dlna(a)
            B1 = 1.5*Omega_m(a)*mu_func(a)
            dpp = -A1*dp + B1*d
            return np.array([dp, dpp], float)
        y = np.array([d, dp], float)
        if i < n-1:
            h = xs[i+1]-xs[i]
            k1 = rhs(x, y)
            k2 = rhs(x+0.5*h, y+0.5*h*k1)
            k3 = rhs(x+0.5*h, y+0.5*h*k2)
            k4 = rhs(x+h, y+h*k3)
            y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            d, dp = float(y[0]), float(y[1])
    a_out = np.exp(xs)
    G = deltas/deltas[0]
    f_g = dps/deltas
    return a_out, G, f_g

# -------------------------
# Spherical Collapse (Non-Linear Regime)
# -------------------------
def top_hat_r0_m(M_kg):
    return (3*M_kg/(4*math.pi*rho_m0))**(1/3)

def tauN_halo(M_kg, a, delta, r0_m):
    R = a*r0_m*(1+delta)**(-1/3)
    return (G_SI*M_kg)/(R*c_SI**2)

def delta_c_GR(z_c=10, a_ini=1/500):
    a_c = 1/(1+z_c)
    def evolve(delta_ini):
        xs = np.linspace(math.log(a_ini), math.log(a_c), 5000)
        d=delta_ini; dp=delta_ini
        for i,x in enumerate(xs[:-1]):
            h=xs[i+1]-x
            def rhs(xx,y):
                d,dp=y
                a=math.exp(xx)
                A1=2.0+dlnH_dlna(a)
                dpp=-A1*dp + 1.5*Omega_m(a)*d*(1+d) + (4/3)*(dp*dp)/(1+d)
                return np.array([dp,dpp], float)
            y=np.array([d,dp], float)
            k1=rhs(x,y); k2=rhs(x+0.5*h,y+0.5*h*k1)
            k3=rhs(x+0.5*h,y+0.5*h*k2); k4=rhs(x+h,y+h*k3)
            y=y+(h/6.0)*(k1+2*k2+2*k3+k4)
            d,dp=float(y[0]), float(y[1])
            if d>1e6:
                return True
        return False
    lo,hi=1e-6,1e-2
    for _ in range(40):
        if evolve(hi): break
        hi*=2
    for _ in range(70):
        mid=math.sqrt(lo*hi)
        if evolve(mid): hi=mid
        else: lo=mid
        if hi/lo-1<1e-3: break
    delta_ini_star=hi
    xs = np.linspace(math.log(a_ini), math.log(a_c), 7000)
    d=delta_ini_star; dp=delta_ini_star
    for i,x in enumerate(xs[:-1]):
        h=xs[i+1]-x
        def rhs(xx,y):
            d,dp=y
            a=math.exp(xx)
            A1=2.0+dlnH_dlna(a)
            dpp=-A1*dp + 1.5*Omega_m(a)*d
            return np.array([dp,dpp], float)
        y=np.array([d,dp], float)
        k1=rhs(x,y); k2=rhs(x+0.5*h,y+0.5*h*k1)
        k3=rhs(x+0.5*h,y+0.5*h*k2); k4=rhs(x+h,y+h*k3)
        y=y+(h/6.0)*(k1+2*k2+2*k3+k4)
        d,dp=float(y[0]), float(y[1])
    return d

def delta_c_env(M_Msun, z_c, a_tau, tau_bar, a_ini=1/500):
    M_kg = M_Msun*Msun
    r0_m = top_hat_r0_m(M_kg)
    r0_Mpc = r0_m/Mpc
    k_eff = 1.0/max(r0_Mpc, 1e-9)  # 1/Mpc
    a_c = 1/(1+z_c)

    def tau_inf_of_a(a):
        return float(np.interp(a, a_tau, np.abs(tau_bar)))

    def mu_state(a, d):
        tau_inf = tau_inf_of_a(a)
        tauN = tauN_halo(M_kg, a, max(d,0.0), r0_m)
        muQS = mu_QS_from_tauN_with_tauinf(tauN, tau_inf)
        s0 = max(muQS - 1.0, 0.0)
        mu_base = 1.0 + s0*(k_eff**2)/(k_eff**2 + (a*m_eff_phys_per_Mpc)**2)

        if MODEL_MODE.upper() == "ENV_BUMP":
            z = 1.0/a - 1.0
            dmu_th = ENV_A * max(mu_th_of_z(z) - 1.0, 0.0) * bump_gate(tauN)
            if ENV_COMBINE.lower() == "mul":
                mu = mu_base * (1.0 + dmu_th)
            else:
                mu = mu_base + dmu_th
            return max(mu, 1.0)

        return mu_base

    def evolve(delta_ini):
        xs = np.linspace(math.log(a_ini), math.log(a_c), 5000)
        d=delta_ini; dp=delta_ini
        for i,x in enumerate(xs[:-1]):
            h = xs[i+1]-x
            def rhs(xx,y):
                d,dp=y
                a=math.exp(xx)
                A1=2.0+dlnH_dlna(a)
                mu=mu_state(a,d)
                dpp=-A1*dp + 1.5*Omega_m(a)*mu*d*(1+d) + (4/3)*(dp*dp)/(1+d)
                return np.array([dp,dpp], float)
            y=np.array([d,dp], float)
            k1=rhs(x,y); k2=rhs(x+0.5*h,y+0.5*h*k1)
            k3=rhs(x+0.5*h,y+0.5*h*k2); k4=rhs(x+h,y+h*k3)
            y=y+(h/6.0)*(k1+2*k2+2*k3+k4)
            d,dp=float(y[0]), float(y[1])
            if d>1e6:
                return True
        return False

    lo,hi=1e-6,1e-2
    for _ in range(40):
        if evolve(hi): break
        hi*=2
    for _ in range(70):
        mid=math.sqrt(lo*hi)
        if evolve(mid): hi=mid
        else: lo=mid
        if hi/lo-1<1e-3: break
    delta_ini_star=hi

    xs = np.linspace(math.log(a_ini), math.log(a_c), 7000)
    d=delta_ini_star; dp=delta_ini_star
    for i,x in enumerate(xs[:-1]):
        h = xs[i+1]-x
        def rhs(xx,y):
            d,dp=y
            a=math.exp(xx)
            A1=2.0+dlnH_dlna(a)
            if DELTA_C_LINEAR_MU.upper() == "K":
                mu_lin = mu_of_a_k(a, k_eff, a_tau, tau_bar)
            else:
                mu_lin = 1.0
            B1=1.5*Omega_m(a)*mu_lin
            dpp=-A1*dp + B1*d
            return np.array([dp,dpp], float)
        y=np.array([d,dp], float)
        k1=rhs(x,y); k2=rhs(x+0.5*h,y+0.5*h*k1)
        k3=rhs(x+0.5*h,y+0.5*h*k2); k4=rhs(x+h,y+h*k3)
        y=y+(h/6.0)*(k1+2*k2+2*k3+k4)
        d,dp=float(y[0]), float(y[1])
    return d, k_eff

# -------------------------
# HMF Ratio (Press-Schechter)
# -------------------------
def PS_f(nu):
    return math.sqrt(2/math.pi)*nu*math.exp(-0.5*nu*nu)

# -------------------------
# Main Execution Block
# -------------------------
def run(outdir="outputs", z_target=10.0, masses_Msun=(1e11, 1e12), k_plot_h=(0.1, 1.0)):
    os.makedirs(outdir, exist_ok=True)

    # 1. Solve Background
    a_tau, tau_bar = integrate_tau_bg()
    z_tau = 1/a_tau - 1
    mu_bg_max = np.array([mu_max_from_background_tau_z(t, z) for t, z in zip(np.abs(tau_bar), (1/a_tau - 1))])

    # 2. Compute Linear Growth
    k_list = [kh*h for kh in k_plot_h]
    growth = {}
    for kh, k in zip(k_plot_h, k_list):
        mu_func = lambda a, kk=k: mu_of_a_k(a, kk, a_tau, tau_bar)
        aG, G, f_g = integrate_growth(lambda a, muf=mu_func: muf(a))
        growth[kh] = dict(a=aG, G=G, f=f_g)

    # 3. Compute GR Baseline
    aG_GR, G_GR, f_GR = integrate_growth(lambda a: 1.0)

    # 4. Compute Collapse Thresholds
    dc_GR = delta_c_GR(z_c=z_target)
    dc_MF = []
    k_eff_list = []
    for M in masses_Msun:
        dc, k_eff = delta_c_env(M, z_target, a_tau, tau_bar)
        dc_MF.append(dc); k_eff_list.append(k_eff)
    dc_MF = np.array(dc_MF)
    k_eff_list = np.array(k_eff_list)

    # 5. Compute HMF Ratios
    aZ = 1/(1+z_target)
    nus = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

    Ggr_abs = float(np.interp(aZ, aG_GR, G_GR))
    Dgr = float(np.interp(aZ, aG_GR, G_GR)/np.interp(1.0, aG_GR, G_GR))

    hmf_abs = []
    hmf_norm = []
    for dc, k_eff in zip(dc_MF, k_eff_list):
        mu_func = lambda a, kk=k_eff: mu_of_a_k(a, kk, a_tau, tau_bar)
        aGk, Gk, fk = integrate_growth(lambda a, muf=mu_func: muf(a))
        Gmf_abs = float(np.interp(aZ, aGk, Gk))
        Dmf = float(np.interp(aZ, aGk, Gk)/np.interp(1.0, aGk, Gk))

        ratio_nu_abs = (dc/dc_GR) * (Ggr_abs/Gmf_abs)
        ratio_nu_norm = (dc/dc_GR) * (Dgr/Dmf)

        hmf_abs.append([PS_f(nu*ratio_nu_abs)/PS_f(nu) for nu in nus])
        hmf_norm.append([PS_f(nu*ratio_nu_norm)/PS_f(nu) for nu in nus])

    hmf_abs = np.array(hmf_abs)
    hmf_norm = np.array(hmf_norm)

    # 6. Plotting
    plt.figure()
    plt.plot(z_tau, np.abs(tau_bar))
    plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.xlabel("Redshift z")
    plt.ylabel("|tau_bar(z)|")
    plt.title("Background Scalar Field Evolution")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "background_tau.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.plot(z_tau, mu_bg_max)
    plt.gca().invert_xaxis()
    plt.xlabel("Redshift z")
    plt.ylabel("Maximal Coupling mu_max")
    plt.title("Effective Gravitational Coupling History")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "background_mu_max.png"), dpi=180)
    plt.close()

    plt.figure()
    for kh in k_plot_h:
        aG = growth[kh]["a"]; G = growth[kh]["G"]
        ratio = (np.interp(aG, aG, G) / np.interp(aG, aG_GR, G_GR))
        plt.plot(1/aG-1, ratio, label=f"k={kh} h/Mpc")
    plt.gca().invert_xaxis()
    plt.xlabel("Redshift z")
    plt.ylabel("Growth Enhancement Factor")
    plt.title("Linear Growth Boost vs GR")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "growth_ratio_absolute.png"), dpi=180)
    plt.close()

    # 7. Save Data Tables
    df_c = pd.DataFrame({
        "Halo_Mass_Msun": list(masses_Msun),
        "delta_c_GR": [dc_GR]*len(masses_Msun),
        "delta_c_MF": dc_MF,
        "Abundance_Boost_Abs_nu5": hmf_abs[:, list(nus).index(5.0)],
        "Abundance_Boost_Norm_nu5": hmf_norm[:, list(nus).index(5.0)],
    })
    df_c.to_csv(os.path.join(outdir, "simulation_summary.csv"), index=False)

    print("=== Simulation Complete ===")
    print(f"Results saved to: {os.path.abspath(outdir)}")
    print(f"Collapse Threshold (GR): {dc_GR:.4f}")
    print(f"Collapse Threshold (MF): {dc_MF[0]:.4f}")

if __name__ == "__main__":
    run()