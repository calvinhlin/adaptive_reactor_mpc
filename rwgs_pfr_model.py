"""
1D plug-flow reactor model for RWGS over K2CO3/Al2O3 catalyst.
Follows the formulation in Lin et al. (Joule 2024) Note S11.

Inputs:
  - Steady-state spreadsheet with axial T profiles (sheet 'SS') and
    measured CO2 conversions (sheet 'Conversion').

Outputs:
  - Fitted Arrhenius parameters (A_f, Ea_f).
  - Plots of T(z), model vs experimental conversion, and parity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize


# -----------------------------------------------------------------------------
# Constants and reactor geometry
# -----------------------------------------------------------------------------
R_GAS = 8.31446261815324  # J/(mol K)
T_STP = 273.15          # K
P = 101_325.0           # Pa

# SuPEr-Lab packed bed: 1.5" x 8" cylinder, 95% packing fraction.
INCH = 0.0254
CAT_D = 1.5 * INCH               # 38.10 mm
CAT_L = 8.0 * INCH               # 203.20 mm
PACK = 0.95
V_CAT = PACK * np.pi * (CAT_D / 2.0) ** 2 * CAT_L  # ~ 2.20e-4 m^3 = 220 mL

Z_IN = 0.000             # m  reaction-zone inlet
Z_OUT = 0.200            # m  reaction-zone outlet (model length)
L_MODEL = Z_OUT - Z_IN
A_TUBE = V_CAT / L_MODEL          # effective cross-section so V = A*L = V_cat

# RWGS thermodynamics from NIST Shomate (CO + H2O - CO2 - H2)
SHOMATE = {
    "CO2": (24.99735, 55.18696, -33.69137, 7.948387,
            -0.136638, -403.6075, 228.2431, -393.5224),
    "CO":  (25.56759, 6.096130, 4.054656, -2.671301,
            0.131021, -118.0089, 227.3665, -110.5271),
    "H2":  (33.066178, -11.363417, 11.432816, -2.772874,
            -0.158558, -9.980797, 172.707974, 0.0),
    "H2O": (30.09200, 6.832514, 6.793435, -2.534480,
            0.082139, -250.8810, 223.3967, -241.8264),
}


def _gibbs_kJ(species, T):
    a, b, c, d, e, f, g, h = SHOMATE[species]
    t = T / 1000.0
    h_abs = h + (a*t + b*t**2/2 + c*t**3/3 + d*t**4/4 - e/t + f - h)
    s_J = a*np.log(t) + b*t + c*t**2/2 + d*t**3/3 - e/(2*t**2) + g
    return h_abs - T * s_J / 1000.0


def K_eq(T):
    dG_kJ = (_gibbs_kJ("CO", T) + _gibbs_kJ("H2O", T)
             - _gibbs_kJ("CO2", T) - _gibbs_kJ("H2", T))
    return float(np.exp(-dG_kJ * 1000.0 / (R_GAS * T)))


def x_eq(T, ratio_H2_CO2=3.0):
    """Equilibrium CO2 conversion for H2:CO2 = ratio:1, no inerts."""
    K = K_eq(T)
    f = lambda x: x * x - K * (1.0 - x) * (ratio_H2_CO2 - x)
    return brentq(f, 1e-9, 1.0 - 1e-9)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parent / "final_steady_state.xlsx"

T_SETPOINTS = [450, 500, 550]
FLOW_ROWS = {3.0: [2, 3, 4], 2.0: [7, 8, 9], 1.0: [12, 13, 14]}
CONV_ROWS = {3.0: [1, 2, 3], 2.0: [6, 7, 8], 1.0: [11, 12, 13]}


def load_data():
    ss = pd.read_excel(DATA_FILE, sheet_name="SS", header=None)
    z_mm = np.array(ss.iloc[1, 10:26].values, dtype=float)
    z_m = z_mm * 1e-3

    profiles = {}
    for flow, rows in FLOW_ROWS.items():
        for r, T_set in zip(rows, T_SETPOINTS):
            T_C = np.array(ss.iloc[r, 10:26].values, dtype=float)
            profiles[(flow, T_set)] = (z_m.copy(), T_C + 273.15)

    conv_df = pd.read_excel(DATA_FILE, sheet_name="Conversion", header=None)
    expt = {}
    for flow, rows in CONV_ROWS.items():
        for r, T_set in zip(rows, T_SETPOINTS):
            expt[(flow, T_set)] = float(conv_df.iloc[r, 1])  # Supermode column
    return profiles, expt


# -----------------------------------------------------------------------------
# PFR ODE
# -----------------------------------------------------------------------------
def build_rhs(T_of_z, dT_of_z, u_STP, A_f, Ea_f):
    def rhs(z, C):
        T = float(T_of_z(z))
        dTdz = float(dT_of_z(z))
        u = u_STP * T / T_STP
        kf = A_f * np.exp(-Ea_f / (R_GAS * T))
        kr = kf / K_eq(T)
        r_net = kf * C[0] * C[1] - kr * C[2] * C[3]      # mol/(m^3 s)
        nu = np.array([-1.0, -1.0, +1.0, +1.0])          # CO2, H2, CO, H2O
        return nu * r_net / u - (C / T) * dTdz
    return rhs


def simulate(profile, flow_slpm, A_f, Ea_f, ratio_H2_CO2=3.0,
             z_in=Z_IN, z_out=Z_OUT):
    z_arr, T_arr = profile
    # Keep sensors within [z_in, z_out]; pad the trailing gap with the last
    # in-bound sample.
    mask = (z_arr >= z_in - 1e-9) & (z_arr <= z_out + 1e-9)
    z_zone = z_arr[mask]
    T_zone = T_arr[mask]
    if z_zone[0] > z_in + 1e-6:
        z_zone = np.insert(z_zone, 0, z_in)
        T_zone = np.insert(T_zone, 0,
                           float(interp1d(z_arr, T_arr)(z_in)))
    if z_zone[-1] < z_out - 1e-6:
        z_zone = np.append(z_zone, z_out)
        T_zone = np.append(T_zone, T_zone[-1])   # hold last sensor value

    T_of_z = interp1d(z_zone, T_zone, kind="linear", fill_value="extrapolate")
    z_dense = np.linspace(z_zone[0], z_zone[-1], 400)
    dT_dense = np.gradient(T_of_z(z_dense), z_dense)
    dT_of_z = interp1d(z_dense, dT_dense, kind="linear",
                       fill_value="extrapolate")

    T_in = float(T_of_z(z_zone[0]))
    T_out = float(T_of_z(z_zone[-1]))

    Q_STP = flow_slpm * 1e-3 / 60.0                 # m^3/s
    u_STP = Q_STP / A_TUBE                          # m/s @ STP
    u_in = u_STP * T_in / T_STP
    u_out = u_STP * T_out / T_STP

    y_CO2 = 1.0 / (1.0 + ratio_H2_CO2)
    y_H2 = ratio_H2_CO2 / (1.0 + ratio_H2_CO2)
    C_tot_in = P / (R_GAS * T_in)
    C0 = np.array([y_CO2 * C_tot_in, y_H2 * C_tot_in, 0.0, 0.0])

    rhs = build_rhs(T_of_z, dT_of_z, u_STP, A_f, Ea_f)
    sol = solve_ivp(rhs, [z_zone[0], z_zone[-1]], C0,
                    method="LSODA", rtol=1e-8, atol=1e-12,
                    dense_output=True, max_step=(z_zone[-1]-z_zone[0])/300)
    if not sol.success:
        raise RuntimeError(sol.message)

    F_in = C0[0] * u_in
    F_out = sol.y[0, -1] * u_out
    x_CO2 = 1.0 - F_out / F_in
    return x_CO2, sol


# -----------------------------------------------------------------------------
# Fit Arrhenius parameters
# -----------------------------------------------------------------------------
def fit_parameters(profiles, expt, fit_T=(450, 500)):
    """Fit A_f, Ea_f using only the supplied setpoint temperatures."""
    conds = [(f, T) for f in [1.0, 2.0, 3.0] for T in fit_T]
    x_meas = np.array([expt[c] / 100.0 for c in conds])

    def loss(params):
        log_Af, Ea_kJ = params
        A_f = np.exp(log_Af)
        Ea_f = Ea_kJ * 1e3
        x_pred = np.empty(len(conds))
        for i, c in enumerate(conds):
            try:
                x_pred[i], _ = simulate(profiles[c], c[0], A_f, Ea_f)
            except Exception:
                x_pred[i] = -1.0
        return float(np.sum((x_pred - x_meas) ** 2))

    x0 = [np.log(1.12e7), 120.0]
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 600})
    return np.exp(res.x[0]), res.x[1] * 1e3, conds, x_meas


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    profiles, expt = load_data()

    A_f_paper, Ea_f_paper = 1.12e7, 120e3
    print("Fitting A_f, Ea_f to supermode conversions at 450 and 500 C only ...")
    A_f_fit, Ea_f_fit, fit_conds, x_meas_fit = fit_parameters(
        profiles, expt, fit_T=(450, 500))
    print(f"  Paper :  A_f = {A_f_paper:.3e} m^3/(mol s),  Ea_f = "
          f"{Ea_f_paper/1e3:.1f} kJ/mol")
    print(f"  Fitted:  A_f = {A_f_fit:.3e} m^3/(mol s),  Ea_f = "
          f"{Ea_f_fit/1e3:.1f} kJ/mol  (fit on T_set in {(450,500)})")

    # Evaluate on all conditions; 550 C is held out
    conds = [(f, T) for f in [1.0, 2.0, 3.0] for T in T_SETPOINTS]
    x_meas = np.array([expt[c] / 100.0 for c in conds])
    x_pred_paper = np.empty(len(conds))
    x_pred_fit = np.empty(len(conds))
    for i, c in enumerate(conds):
        x_pred_paper[i], _ = simulate(profiles[c], c[0], A_f_paper, Ea_f_paper)
        x_pred_fit[i],   _ = simulate(profiles[c], c[0], A_f_fit,   Ea_f_fit)

    print("\n  Flow   T_set   x_exp   x_paper   x_fit   x_eq(T_set)")
    for c, xm, xp_p, xp_f in zip(conds, x_meas, x_pred_paper, x_pred_fit):
        xe = x_eq(c[1] + 273.15)  # setpoint T = control TC reading
        print(f"  {c[0]:>4.1f}   {c[1]:>5d}  "
              f"{xm*100:>5.1f}   {xp_p*100:>6.1f}   {xp_f*100:>5.1f}   "
              f"{xe*100:>5.1f}")

    # ---------- plots ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    colors = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c"}
    ls_map = {450: ":", 500: "--", 550: "-"}
    markers = {1.0: "o", 2.0: "s", 3.0: "^"}

    ax = axes[0]
    for flow in [1.0, 2.0, 3.0]:
        for T_set in T_SETPOINTS:
            z, T = profiles[(flow, T_set)]
            ax.plot(z * 1000, T - 273.15, color=colors[flow],
                    linestyle=ls_map[T_set], lw=1.4,
                    label=f"{flow:.0f} SLPM, {T_set} C")
    ax.axvspan(Z_IN * 1000, Z_OUT * 1000, color="grey", alpha=0.12,
               label=f"reaction zone (0-{int(Z_OUT*1000)} mm, "
                     f"V_cat={V_CAT*1e6:.0f} mL)")
    ax.set_xlabel("axial position (mm)")
    ax.set_ylabel("temperature (C)")
    ax.set_title("Measured axial T profiles")
    ax.legend(fontsize=7, ncol=3, loc="lower center")

    ax = axes[1]
    T_for_eq = np.linspace(420, 600, 200)
    x_eq_curve = np.array([x_eq(T + 273.15) * 100 for T in T_for_eq])
    ax.plot(T_for_eq, x_eq_curve, "k--", lw=1.5, label="equilibrium")
    for flow in [1.0, 2.0, 3.0]:
        xm = [expt[(flow, T)] for T in T_SETPOINTS]
        xp = [x_pred_fit[conds.index((flow, T))] * 100 for T in T_SETPOINTS]
        ax.plot(T_SETPOINTS, xm, linestyle="none",
                marker=markers[flow], color=colors[flow], ms=8,
                label=f"{flow:.0f} SLPM (exp)")
        ax.plot(T_SETPOINTS, xp, "-", color=colors[flow], alpha=0.85,
                label=f"{flow:.0f} SLPM (PFR)")
    ax.set_xlabel("exit setpoint T (C)")
    ax.set_ylabel("CO$_2$ conversion (%)")
    ax.set_title("Conversion vs T (model + equilibrium)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    lim = max(x_meas.max() * 100, x_pred_fit.max() * 100) + 5
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    for i, c in enumerate(conds):
        face = colors[c[0]] if c[1] != 550 else "white"
        ax.plot(x_meas[i] * 100, x_pred_fit[i] * 100,
                marker=markers[c[0]], color=colors[c[0]],
                markerfacecolor=face, mew=1.5, ms=9,
                linestyle="none",
                label=f"{c[0]:.0f} SLPM, {c[1]} C")
    used = np.array([c[1] in (450, 500) for c in conds])
    rmse_fit = np.sqrt(np.mean((x_pred_fit[used] - x_meas[used]) ** 2)) * 100
    rmse_all = np.sqrt(np.mean((x_pred_fit - x_meas) ** 2)) * 100
    ax.set_xlabel("experimental conversion (%)")
    ax.set_ylabel("PFR model conversion (%)")
    ax.set_title(f"Parity  RMSE 450/500 = {rmse_fit:.2f}%, "
                 f"all = {rmse_all:.2f}%  (open = 550 C, held out)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    out = Path(__file__).with_name("rwgs_pfr_results.png")
    plt.savefig(out, dpi=160)
    print(f"\nSaved figure: {out}")

    summary = pd.DataFrame(
        {"flow_SLPM": [c[0] for c in conds],
         "T_set_C":   [c[1] for c in conds],
         "x_exp_%":   x_meas * 100,
         "x_model_%": x_pred_fit * 100,
         "x_eq_%":    [x_eq(c[1] + 273.15) * 100 for c in conds]})
    summary.to_csv(Path(__file__).with_name("rwgs_pfr_summary.csv"),
                   index=False, float_format="%.2f")
    print("Saved summary: rwgs_pfr_summary.csv")


if __name__ == "__main__":
    main()
