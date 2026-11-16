"""Economic-MPC digital twin of the induction-heated RWGS reactor.

Uses the LZ_South Nov-5 5-min real-time electricity price (24 h compressed
into SIM_HORIZON_S) and picks a flow rate and coil current magnitude every
minute to maximise CO revenue ($0.15/kg) minus electricity cost.

Dynamics: v3_coil29 grey-box thermal model (thermal_fit_em_v3_coil29_best.json).
Conversion: profile-PFR kinetics (Af, Ea from rwgs_pfr_fit_results.json).

Flows are 0-3 SLPM. Steady-state 450/500 C profiles come from
isothermal_opt_450C_window.json and isothermal_opt_500C_window.json.
flow=0 at T_max=450 C is the minimum-power hot-standby state.

Constraints:
    - coil current <= I_MAX_A
    - T_max in [450, 500] C at all times (no off state)
    - two-phase power ramp: 40 W/s until 90% of target, then 10 W/s

Run:
    python empc_digital_twin.py [--nmpc | --collocation] [--gif] [--no-show]
"""

from __future__ import annotations

import json
import math
import sys
import time as _wallclock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WORKSPACE = Path(__file__).resolve().parent

import thermal_greybox as tg  # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PRICE_XLSX = WORKSPACE / "LZ_South_Nov5_RealTime.xlsx"
PARAMS_FILE = WORKSPACE / "thermal_fit_em_v3_coil29_best.json"
ISO_450_FILE = WORKSPACE / "isothermal_opt_450C_window.json"
ISO_500_FILE = WORKSPACE / "isothermal_opt_500C_window.json"
RWGS_FIT = WORKSPACE / "rwgs_pfr_fit_results.json"

CO_PRICE_USD_PER_KG = 0.15    # CO sale price ($/kg)
CO_MW_KG_PER_MOL = 0.028010
H2_TO_CO2 = 3.0

T_LOW_C  = 450.0   # lower bound on T_max
T_HIGH_C = 500.0   # upper bound on T_max
I_MAX_A  = 5.2

# Measured current shapes (unit vectors) per flow. The shape is fixed by
# on_array + frequency; only the magnitude can be scaled (via Vpp).
# Mean of test5-7 runs, std < 0.002 per element.
EMPIRICAL_SHAPE_BY_FLOW = {
    0: np.array([0.5197, 0.2570, 0.2577, 0.2321, 0.2595, 0.6901]),
    1: np.array([0.6042, 0.2911, 0.2501, 0.3128, 0.2802, 0.5579]),
    2: np.array([0.6187, 0.2932, 0.2480, 0.2585, 0.2832, 0.5681]),
    3: np.array([0.6643, 0.2868, 0.2387, 0.2465, 0.2685, 0.5354]),
}

# ||I|| giving steady-state T_max = 450 / 500 C with the shapes above
# (bisection on the thermal model, 4 h sim to steady state).
EMPIRICAL_I_NORM = {
    (0, 450): 4.6832,  (1, 450): 4.4699,  (2, 450): 4.6993,  (3, 450): 4.8945,
    (0, 500): 5.0902,  (1, 500): 4.8576,  (2, 500): 5.1104,  (3, 500): 5.3216,
}

# Two-phase power ramp, same 4:1 fast/fine ratio as the hardware Vpp ramp.
RAMP_FAST_W_PER_S = 40.0
RAMP_FINE_W_PER_S = 10.0
RAMP_FAST_FRAC    = 0.90   # switch to fine ramp at 90% of target power

# Deposited power for the flow=0, T=450 mode. All model powers are scaled
# by the same factor to match this.
POWER_ANCHOR_W = 144
POWER_MAX_W    = 225            # transient power ceiling, allows ||I|| above the 500 C SS value
POWER_ANCHOR_MODE = "flow=0_T=450"

SIM_HORIZON_S = 12 * 3600
SIM_DT_S      = 60.0                # thermal sim step
CONTROL_DT_S  = 60.0                # MPC re-plan interval
PRICE_PERIOD_S = 2.5 * 60.0          # sim time per price row (24 h of prices -> 12 h)
T_INIT_C = 450.0                    # uniform initial wall T

# Animation
FRAME_INTERVAL_MS = 50              # ~20 fps
SIM_S_PER_FRAME   = 72.0            # sim seconds per frame, rounded to a multiple of SIM_DT_S

SAVE_GIF = False
GIF_PATH = WORKSPACE / "plots" / "empc_digital_twin_24h.gif"

# Receding-horizon MPC. The planner snapshots/restores the ThermalStepper
# and scores each candidate sequence with the grey-box model + PFR kinetics.
MPC_HORIZON        = 3              # lookahead steps (3 min at 60 s)
MPC_ROLLOUT_STEPS  = 10             # frozen-state terminal steps beyond the horizon
MIN_DWELL_STEPS    = 5              # min steps on a flow before switching (5 min)
N_I_GRID           = 3              # ||I|| candidates per non-zero flow


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def load_realtime_price(xlsx_path: Path) -> np.ndarray:
    """Return a length-288 vector of 5-minute real-time prices ($/MWh)."""
    df = pd.read_excel(xlsx_path)
    col = next((c for c in df.columns if "Real Time Cost" in str(c)
                or "Real-Time" in str(c)), None)
    if col is None:
        # workbook columns are time, price
        col = df.columns[1]
    arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    if arr.size < 288:
        # tile a partial file, truncate extras
        reps = int(np.ceil(288 / max(arr.size, 1)))
        arr = np.tile(arr, reps)
    return arr[:288]


def price_at(t_s: float, prices_5min: np.ndarray) -> float:
    """Zero-order-hold price ($/MWh) at t_s, wrapping after the last bin."""
    bin_id = int((t_s // PRICE_PERIOD_S)) % prices_5min.size
    return float(prices_5min[bin_id])


# --------------------------------------------------------------------------- #
# Mode bank
# --------------------------------------------------------------------------- #
@dataclass
class Mode:
    name: str
    flow_slpm: float
    target_c: float
    currents_a: np.ndarray         # (6,)
    base_T_profile_c: np.ndarray   # (N_nodes,) steady-state profile at base currents
    base_power_w: float            # deposited power at base currents [W]
    co_kg_per_s: float             # steady-state CO production [kg/s]
    is_off: bool = False


@dataclass
class FlowConfig:
    """One flow rate with its allowed ||I|| range."""
    flow_slpm: float
    shape: np.ndarray              # (6,) current shape
    I_norm_min: float              # ||I|| at SS T_max=450
    I_norm_max: float              # max of SS 500 C ||I|| and POWER_MAX_W ||I||
    I_norm_max_eff: float          # min(I_norm_max, I_MAX_A / max(shape))
    power_at_min: float            # deposited W at I_norm_min
    power_at_max: float            # deposited W at I_norm_max_eff
    co_rate_at_min: float          # kg/s CO at steady-state 450 C
    co_rate_at_max: float          # kg/s CO at steady-state 500 C


@dataclass
class PlannerAction:
    """Planner output: a (flow, ||I||) pair."""
    flow_idx: int                  # index into flow_configs (0..3)
    I_norm: float
    currents_a: np.ndarray         # shape * I_norm
    power_w: float                 # deposited power [W]
    flow_slpm: float


def total_em_power_w(currents_6: np.ndarray, params: tg.ThermalParams,
                     x_m: np.ndarray, basis: np.ndarray) -> float:
    """Total deposited power [W]: axial integral of the EM line-power density."""
    sigma = params.get("sigma_sus")
    f_em  = float(tg._em_coupling_factor(sigma))
    b_field = currents_6 @ basis      # (N,) T
    b2 = b_field * b_field
    line_w_m = tg._EM_BASE_W_PER_M_T2 * f_em * b2
    return float(np.trapezoid(line_w_m, x_m))


def co_rate_kg_per_s_from_profile(temp_c_profile: np.ndarray,
                                  x_m: np.ndarray,
                                  flow_slpm: float,
                                  ln_a_app: float,
                                  ea_j_per_mol: float,
                                  n_steps: int = 200) -> float:
    """Steady-state CO production [kg/s] from a wall T profile.

    Same PFR as rwgs_pfr_model.pfr_conversion, with T(z) taken from the
    thermal model grid (195-200 mm held at the 195 mm value).
    """
    if flow_slpm <= 1e-9:
        return 0.0

    # Catalyst geometry
    INCH = 0.0254
    cat_d = 1.5 * INCH
    cat_L = 8.0 * INCH
    cat_fill = 0.95
    Vcat = cat_fill * math.pi * (cat_d / 2) ** 2 * cat_L
    rxn_L = 0.200
    VcL = Vcat / rxn_L

    # CO2 + H2 -> CO + H2O, H2:CO2 = 3:1 feed
    co2_in = flow_slpm / (1.0 + H2_TO_CO2)
    h2_in = flow_slpm - co2_in
    mol_per_s_per_slpm = 101325.0 * 1.0e-3 / (8.31446261815324 * 273.15 * 60.0)
    co2_in_mol_s = co2_in * mol_per_s_per_slpm
    h2_in_mol_s = h2_in * mol_per_s_per_slpm
    F_total = co2_in_mol_s + h2_in_mol_s
    xi_upper = min(co2_in_mol_s, h2_in_mol_s) * (1.0 - 1e-10)

    R_GAS = 8.31446261815324

    # T(z) on the 0-200 mm catalyst bed; hold the 195 mm value beyond the
    # last sensor.
    z_sensors = x_m[x_m <= 0.195 + 1e-9]
    n_sens = len(z_sensors)
    T_sens_k = temp_c_profile[:n_sens] + 273.15
    dz = rxn_L / n_steps
    z_mid = (np.arange(n_steps) + 0.5) * dz
    T_k = np.interp(z_mid, z_sensors, T_sens_k,
                    left=float(T_sens_k[0]), right=float(T_sens_k[-1]))

    # NIST Shomate coefficients for K_eq(T)
    SHOMATE = tg.__dict__  # unused, overwritten below
    SHOMATE = {
        "CO2": (24.99735, 55.18696, -33.69137, 7.948387, -0.136638,
                -403.6075, 228.2431, -393.5224),
        "CO":  (25.56759, 6.096130, 4.054656, -2.671301, 0.131021,
                -118.0089, 227.3665, -110.5271),
        "H2":  (33.066178, -11.363417, 11.432816, -2.772874, -0.158558,
                -9.980797, 172.707974, 0.0),
        "H2O": (30.09200, 6.832514, 6.793435, -2.534480, 0.082139,
                -250.8810, 223.3967, -241.8264),
    }
    def gibbs_kj(spc: str, T: float) -> float:
        a,b,c,d,e,f,g,h = SHOMATE[spc]
        t = T/1000.0
        h_abs = h + (a*t + b*t**2/2 + c*t**3/3 + d*t**4/4 - e/t + f - h)
        s_j   = a*math.log(t) + b*t + c*t**2/2 + d*t**3/3 - e/(2*t**2) + g
        return h_abs - T*s_j/1000.0
    def keq(T: float) -> float:
        dG = (gibbs_kj("CO",T) + gibbs_kj("H2O",T)
              - gibbs_kj("CO2",T) - gibbs_kj("H2",T)) * 1000.0
        return math.exp(-dG / (R_GAS*T))
    k_eq = np.array([keq(float(T)) for T in T_k])
    # equilibrium extent xi_eq(T), basis 1 mol CO2 + 3 mol H2
    xi_eq_arr = np.zeros(n_steps)
    for i, T in enumerate(T_k):
        K = max(k_eq[i], 1e-30)
        a_ = K - 1.0
        b_ = -K * (1.0 + H2_TO_CO2)
        c_ = K * H2_TO_CO2
        if abs(a_) < 1e-12:
            x_eq = -c_/b_
        else:
            disc = max(0.0, b_*b_ - 4*a_*c_)
            sq = math.sqrt(disc)
            r1 = (-b_ + sq) / (2*a_)
            r2 = (-b_ - sq) / (2*a_)
            # physical root in [0, 1]; smaller one if both qualify
            valid = [r for r in (r1, r2) if -1e-9 <= r <= 1.0 + 1e-9]
            x_eq = min(max(r, 0.0) for r in valid) if valid else 0.0
        xi_eq_arr[i] = float(np.clip(x_eq, 0.0, 1.0)) * co2_in_mol_s

    q_m3_s = F_total * R_GAS * T_k / 101325.0
    inv_q2 = 1.0 / (q_m3_s * q_m3_s)
    inv_keq = 1.0 / np.maximum(k_eq, 1e-30)
    a_forward = math.exp(ln_a_app)
    k_forward = a_forward * np.exp(-ea_j_per_mol / (R_GAS * T_k))

    xi = 0.0
    for s in range(n_steps):
        x = float(np.clip(xi, 0.0, xi_upper))
        driving = ((co2_in_mol_s - x) * (h2_in_mol_s - x)
                   - x*x*inv_keq[s]) * inv_q2[s]
        dxi_dz = VcL * k_forward[s] * driving
        dxi = dxi_dz * dz
        if dxi > 0.0:
            dxi = min(dxi, max(0.0, xi_eq_arr[s] - x))
        elif dxi < 0.0:
            dxi = max(dxi, min(0.0, xi_eq_arr[s] - x))
        xi = float(np.clip(xi + dxi, 0.0, xi_upper))

    # xi = CO produced [mol/s]
    return float(xi * CO_MW_KG_PER_MOL)


def load_modes(params: tg.ThermalParams, x_m: np.ndarray,
               basis: np.ndarray, ln_a: float, ea_j: float) -> list[Mode]:
    raw: list[Mode] = []
    for tgt, path in [(450.0, ISO_450_FILE), (500.0, ISO_500_FILE)]:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for fl in (0.0, 1.0, 2.0, 3.0):
            r = cfg["results_by_flow_slpm"][str(fl)]
            # measured shape scaled to the ||I|| that gives T_max = tgt
            shape = EMPIRICAL_SHAPE_BY_FLOW[int(fl)]
            I_norm = EMPIRICAL_I_NORM[(int(fl), int(tgt))]
            I = shape * I_norm
            # steady-state T profile from the isothermal optimisation, used for CO rate
            T_amb_eff = float(params.get("ambient_c"))
            T_prof_raw = np.array(r["T_profile_c"], dtype=float)
            T_prof = T_prof_raw
            P_raw = total_em_power_w(I, params, x_m, basis)
            co = co_rate_kg_per_s_from_profile(T_prof, x_m, fl, ln_a, ea_j)
            raw.append(Mode(
                name=f"flow={fl:.0f}_T={int(tgt)}",
                flow_slpm=fl, target_c=tgt,
                currents_a=I, base_T_profile_c=T_prof,
                base_power_w=P_raw, co_kg_per_s=co,
            ))
    # scale all powers by one factor so POWER_ANCHOR_MODE reads POWER_ANCHOR_W
    anchor = next(m for m in raw if m.name == POWER_ANCHOR_MODE)
    scale = POWER_ANCHOR_W / max(anchor.base_power_w, 1e-9)
    for m in raw:
        m.base_power_w *= scale
    # no OFF mode: T_max must stay in [450, 500] C
    return raw


def load_flow_configs(params: tg.ThermalParams, x_m: np.ndarray,
                      basis: np.ndarray, ln_a: float, ea_j: float
                      ) -> list[FlowConfig]:
    """Build one FlowConfig per flow (0-3 SLPM).

    ||I|| runs from the SS 450 C value up to the larger of the SS 500 C value
    and the POWER_MAX_W value, capped by I_MAX_A. CO rates are at the SS
    450/500 C profiles.
    """
    # scale so flow=0/T=450 reads POWER_ANCHOR_W
    anchor_shape = EMPIRICAL_SHAPE_BY_FLOW[0]
    anchor_norm = EMPIRICAL_I_NORM[(0, 450)]
    anchor_raw_P = total_em_power_w(anchor_shape * anchor_norm, params, x_m, basis)
    power_scale = POWER_ANCHOR_W / max(anchor_raw_P, 1e-9)

    # steady-state T profiles for the CO-rate bounds
    cfg_450 = json.loads(ISO_450_FILE.read_text(encoding="utf-8"))
    cfg_500 = json.loads(ISO_500_FILE.read_text(encoding="utf-8"))

    configs: list[FlowConfig] = []
    for fl in (0.0, 1.0, 2.0, 3.0):
        shape = EMPIRICAL_SHAPE_BY_FLOW[int(fl)]
        I_min = EMPIRICAL_I_NORM[(int(fl), 450)]
        I_max_ss = EMPIRICAL_I_NORM[(int(fl), 500)]
        # allow transient power up to POWER_MAX_W; P ~ ||I||^2
        P_min_raw = total_em_power_w(shape * I_min, params, x_m, basis) * power_scale
        I_max_pwr = I_min * math.sqrt(POWER_MAX_W / max(P_min_raw, 1e-9))
        I_max = max(I_max_ss, I_max_pwr)
        I_max_eff = min(I_max, I_MAX_A / float(np.max(shape)))

        P_min = P_min_raw
        P_max = total_em_power_w(shape * I_max_eff, params, x_m, basis) * power_scale

        T_450 = np.array(
            cfg_450["results_by_flow_slpm"][str(fl)]["T_profile_c"], dtype=float)
        T_500 = np.array(
            cfg_500["results_by_flow_slpm"][str(fl)]["T_profile_c"], dtype=float)
        co_min = co_rate_kg_per_s_from_profile(T_450, x_m, fl, ln_a, ea_j)
        co_max = co_rate_kg_per_s_from_profile(T_500, x_m, fl, ln_a, ea_j)

        configs.append(FlowConfig(
            flow_slpm=fl, shape=shape,
            I_norm_min=I_min, I_norm_max=I_max,
            I_norm_max_eff=I_max_eff,
            power_at_min=P_min, power_at_max=P_max,
            co_rate_at_min=co_min, co_rate_at_max=co_max,
        ))
    return configs


# --------------------------------------------------------------------------- #
# ThermalStepper: grey-box model advanced one control step at a time
# --------------------------------------------------------------------------- #
class ThermalStepper:
    """Single-step version of thermal_greybox.simulate_experiment.

    Used as the plant in closed loop and, with snapshot/restore, for
    planner lookahead.
    """

    def __init__(self, params: tg.ThermalParams, variant: tg.ModelVariant,
                 x_m: np.ndarray, basis: np.ndarray,
                 init_profile_c: np.ndarray, max_substep_s: float = 6.0):
        n = x_m.size
        self.n = n
        self.x = x_m
        self.max_substep_s = max_substep_s

        # Geometry
        self.widths = tg.control_volume_widths(x_m)
        self.vol = tg.A_CROSS_M2 * self.widths
        self.surf = tg.PERIMETER_M * self.widths
        self.dface = np.diff(x_m)

        # Material / loss params
        self.rho_cp = params.get("rho_cp")
        self.cp_T1 = params.get("cp_T1")
        self.k_w = params.get("k_w")
        self.kT1 = params.get("kT1")
        self.h0 = params.get("h0")
        self.h_flow = params.get("h_flow")
        self.n_flow = params.get("n_flow")
        self.h_T = params.get("h_T")
        self.emis = params.get("emissivity") if variant.use_radiation else 0.0
        self.power_scale = params.get("power_scale")
        self.h_gw = params.get("h_gw")
        self.inlet_c = params.get("inlet_c")
        self.t_amb = params.get("ambient_c")
        self.t_amb_lat = params.get("ambient_lat_c")
        self.t_amb_k = self.t_amb_lat + tg.KELVIN
        self.ps_T1 = params.get("ps_T1")
        self.ps_T2 = params.get("ps_T2")
        self.gas_mdot_cp = tg.N2_RHO_STP_KG_M3 * tg.N2_CP_J_KG_K * tg.SLPM_TO_M3_S
        self.variant = variant

        # Shell
        self.shell_C = params.get("shell_C")
        self.shell_G = params.get("shell_G")
        self.shell_G_T1 = params.get("shell_G_T1")

        # End caps
        self.h_end_left = params.get("h_end_left") if variant.use_end_loss else 0.0
        self.h_end_right = params.get("h_end_right") if variant.use_end_loss else 0.0
        self.end_flow_gain = params.get("end_flow_gain")
        self.end_flow_gain_right = params.get("end_flow_gain_right")
        self.h_end_T = params.get("h_end_T")
        left_sink_c = self.inlet_c if (variant.use_end_loss
                                       and "inlet_c" in params.values) else self.t_amb
        self.left_sink_c = left_sink_c

        if variant.use_rend:
            self.rend_C = params.get("rend_C")
            self.rend_G = params.get("rend_G")
            self.rend_G_sus = params.get("rend_G_sus")
        else:
            self.rend_C = self.rend_G = self.rend_G_sus = 0.0
        if variant.use_lend:
            self.lend_C = params.get("lend_C")
            self.lend_G = params.get("lend_G")
            self.lend_G_sus = params.get("lend_G_sus")
        else:
            self.lend_C = self.lend_G = self.lend_G_sus = 0.0

        # EM power model
        if variant.use_em_power:
            self.sigma_sus = params.get("sigma_sus")
            self.sigma_T1_val = params.get("sigma_T1")
            self._f_em_ref = float(tg._em_coupling_factor(self.sigma_sus))
            self._pg = (params.get("power_gain")
                        if "power_gain" in params.values else 1.0)
        else:
            self.sigma_sus = 0.0
            self.sigma_T1_val = 0.0
            self._f_em_ref = 1.0
            self._pg = 1.0

        # Coil gain
        if "coil_mid_gain" in params.values:
            g_mid = params.get("coil_mid_gain")
            g_end = (params.get("coil_end_gain")
                     if "coil_end_gain" in params.values else 1.0)
            self.gain = np.array([g_end, g_mid, g_mid, g_mid, g_mid, g_end])
        else:
            self.gain = None

        # Coil offset
        if "coil_offset_m" in params.values:
            co = params.get("coil_offset_m")
            self._basis = (tg.coil_field_basis(x_m + co)
                           if abs(co) > 1e-7 else basis)
        else:
            self._basis = basis

        # Power axial tilt
        if "power_axial_tilt" in params.values:
            tilt = params.get("power_axial_tilt")
            if tilt != 0.0:
                x_norm = x_m / x_m[-1] - 0.5
                self._ax_fac = np.maximum(1.0 + tilt * x_norm, 0.0)
            else:
                self._ax_fac = None
        else:
            self._ax_fac = None

        # first-order power lag
        if "tau_power" in params.values:
            self._tau_power = params.get("tau_power")
        else:
            self._tau_power = 0.0
        self._qgen_prev = None

        # State
        self.T = init_profile_c.copy()
        shell_init_frac = params.get("shell_init_frac")
        self.T_sh = (self.t_amb + shell_init_frac * (float(np.mean(self.T)) - self.t_amb)
                     if variant.use_shell else self.t_amb)
        self.T_rend = float(self.T[-1]) if variant.use_rend else self.t_amb
        self.T_lend = float(self.T[0]) if variant.use_lend else self.t_amb

    def _compute_qgen(self, currents_a: np.ndarray) -> np.ndarray:
        """Heat generation per node [W]."""
        g = self.gain if self.gain is not None else np.ones(6)
        b_field = (currents_a * g) @ self._basis      # shape (N,)
        b2 = b_field * b_field
        if self.variant.use_em_power:
            gen_line = self._pg * tg._EM_BASE_W_PER_M_T2 * self._f_em_ref * b2
        else:
            gen_line = self.power_scale * (b2 / tg.B2_REF_T2)
        if self._ax_fac is not None:
            gen_line = gen_line * self._ax_fac
        return gen_line * self.widths

    # ------ snapshot / restore for the MPC tree search ------
    def snapshot(self) -> dict:
        """Copy of the thermal state."""
        snap = {
            'T': self.T.copy(),
            'T_sh': self.T_sh,
            'T_rend': self.T_rend,
            'T_lend': self.T_lend,
        }
        if self._qgen_prev is not None:
            snap['qgen_prev'] = self._qgen_prev.copy()
        else:
            snap['qgen_prev'] = None
        return snap

    def restore(self, snap: dict) -> None:
        """Restore thermal state from a snapshot."""
        self.T = snap['T'].copy()
        self.T_sh = snap['T_sh']
        self.T_rend = snap['T_rend']
        self.T_lend = snap['T_lend']
        qp = snap['qgen_prev']
        self._qgen_prev = qp.copy() if qp is not None else None

    @property
    def T_max(self) -> float:
        return float(np.max(self.T))

    def step(self, currents_a: np.ndarray, flow: float, dt: float) -> None:
        """Advance the thermal state by dt seconds."""
        qgen_k = self._compute_qgen(currents_a)

        # optional first-order power lag
        if self._tau_power > 0.5:
            if self._qgen_prev is None:
                self._qgen_prev = qgen_k.copy()
            w = np.exp(-dt / self._tau_power)
            qgen_k = w * self._qgen_prev + (1.0 - w) * qgen_k
            self._qgen_prev = qgen_k.copy()

        n = self.n
        v = self.variant
        n_sub = max(1, int(np.ceil(dt / self.max_substep_s)))
        h_sub = dt / n_sub
        T = self.T

        for _ in range(n_sub):
            cp_fac = np.clip(1.0 + self.cp_T1 * (T - tg.TREF_C), 0.2, 5.0)
            C = np.maximum(self.rho_cp * cp_fac * self.vol, 1e-3)
            k_fac = np.clip(1.0 + self.kT1 * (T - tg.TREF_C), 0.2, 20.0)
            k_node = np.maximum(self.k_w * k_fac, 1e-4)

            if v.use_conduction:
                k_face = (2.0 * k_node[:-1] * k_node[1:]
                          / (k_node[:-1] + k_node[1:]))
                Gf = k_face * tg.A_CROSS_M2 / self.dface
            else:
                Gf = np.zeros(n - 1)

            h = (self.h0 + self.h_flow * (flow ** self.n_flow)
                 + self.h_T * np.maximum(T - self.t_amb_lat, 0.0))
            h = np.maximum(h, 0.0)
            Gloss = h * self.surf

            qrad = (self.emis * tg.SIGMA_SB * self.surf
                    * ((T + tg.KELVIN) ** 4 - self.t_amb_k ** 4))

            qgas = np.zeros(n)
            if v.use_gas_advection and flow > 1e-9:
                mdot_cp = self.gas_mdot_cp * flow
                Tg_prev = self.inlet_c
                for i in range(n):
                    g = self.h_gw * self.surf[i]
                    Tg_i = (mdot_cp * Tg_prev + g * T[i]) / (mdot_cp + g)
                    qgas[i] = g * (T[i] - Tg_i)
                    Tg_prev = Tg_i

            # temperature-dependent coupling efficiency (EM or empirical)
            if v.use_em_power:
                if self.sigma_T1_val != 0.0:
                    sigma_T = self.sigma_sus * np.clip(
                        1.0 + self.sigma_T1_val * (T - tg.TREF_C), 0.01, 100.0)
                    eta = tg._em_coupling_factor(sigma_T) / self._f_em_ref
                else:
                    eta = np.ones(n)
                qgen_eff = qgen_k * np.clip(eta, 0.05, 10.0)
            else:
                dT_ref = T - tg.TREF_C
                if self.ps_T2 != 0.0:
                    eta = 1.0 + self.ps_T1 * dT_ref + self.ps_T2 * dT_ref * dT_ref
                else:
                    eta = 1.0 + self.ps_T1 * dT_ref
                qgen_eff = qgen_k * np.clip(eta, 0.05, 10.0)

            inv = 1.0 / C
            diag = np.empty(n)
            sub = np.zeros(n)
            sup = np.zeros(n)
            rhs = np.empty(n)
            diag[:] = 1.0 / h_sub + Gloss * inv
            if v.use_conduction:
                diag[:-1] += Gf * inv[:-1]
                diag[1:] += Gf * inv[1:]
                sub[1:] = -Gf * inv[1:]
                sup[:-1] = -Gf * inv[:-1]

            rhs[:] = (T / h_sub
                      + (qgen_eff - qrad - qgas + Gloss * self.T_sh) * inv)

            if v.use_end_loss:
                t_fac_l = 1.0 + self.h_end_T * max(float(T[0]) - tg.TREF_C, 0.0)
                t_fac_r = 1.0 + self.h_end_T * max(float(T[-1]) - tg.TREF_C, 0.0)
                g_left = self.h_end_left * (1.0 + self.end_flow_gain * flow) * t_fac_l
                g_right = self.h_end_right * (1.0 + self.end_flow_gain_right * flow) * t_fac_r
                if v.use_rend:
                    diag[-1] += self.rend_G_sus * inv[-1]
                    rhs[-1] += self.rend_G_sus * self.T_rend * inv[-1]
                diag[0] += g_left * inv[0]
                rhs[0] += g_left * self.left_sink_c * inv[0]
                if v.use_lend:
                    diag[0] += self.lend_G_sus * inv[0]
                    rhs[0] += self.lend_G_sus * self.T_lend * inv[0]
                diag[-1] += g_right * inv[-1]
                right_sink = (self.T_rend if v.use_rend else self.t_amb)
                rhs[-1] += g_right * right_sink * inv[-1]

            T = tg._solve_tridiag(sub, diag, sup, rhs)
            T = np.clip(T, -50.0, 4000.0)

            if v.use_shell:
                gsum = float(np.sum(Gloss))
                tsum = float(np.sum(Gloss * T))
                a_sh = self.shell_C / h_sub
                shell_G_eff = self.shell_G * float(
                    np.clip(1.0 + self.shell_G_T1 * (self.T_sh - tg.TREF_C), 0.1, 10.0))
                self.T_sh = ((a_sh * self.T_sh + tsum + shell_G_eff * self.t_amb)
                             / (a_sh + gsum + shell_G_eff))
            if v.use_rend:
                a_re = self.rend_C / h_sub
                self.T_rend = ((a_re * self.T_rend
                                + self.rend_G_sus * float(T[-1])
                                + self.rend_G * self.t_amb)
                               / (a_re + self.rend_G_sus + self.rend_G))
            if v.use_lend:
                a_le = self.lend_C / h_sub
                self.T_lend = ((a_le * self.T_lend
                                + self.lend_G_sus * float(T[0])
                                + self.lend_G * self.t_amb)
                               / (a_le + self.lend_G_sus + self.lend_G))

        self.T = T


# --------------------------------------------------------------------------- #
# MPC planner: receding horizon over (flow, ||I||)
# --------------------------------------------------------------------------- #
def plan_step_empc(stepper: ThermalStepper, flow_configs: list[FlowConfig],
                   price_horizon: list[float], prev_flow_idx: int | None,
                   x_m: np.ndarray, ln_a: float, ea_j: float,
                   control_dt_s: float = CONTROL_DT_S) -> PlannerAction:
    """Depth-first tree search over (flow, ||I||) for H control steps.

    Each branch advances the grey-box model one step, runs the PFR on the
    predicted T profile and adds step profit. Branches with T_max > T_HIGH_C
    are pruned. flow=0 has one candidate (I_norm_min); other flows have
    N_I_GRID, so 10 branches per node and 1000 leaves at H=3. Each leaf gets
    a terminal value of R more steps at its frozen T profile.

    Parameters
    ----------
    stepper : ThermalStepper
        Current state; restored on return.
    flow_configs : list[FlowConfig]
        One entry per flow (0-3 SLPM).
    price_horizon : list[float]
        Electricity price ($/MWh) at each future control step.
    prev_flow_idx : int | None
        Flow index at the previous control step.
    x_m, ln_a, ea_j : PFR kinetics inputs.
    control_dt_s : float
        Control step [s].

    Returns
    -------
    PlannerAction for the first step.
    """
    H = min(MPC_HORIZON, len(price_horizon))
    R = MPC_ROLLOUT_STEPS
    snap0 = stepper.snapshot()
    rollout_price = price_horizon[-1] if price_horizon else 0.0

    # (flow_idx, I_norm) candidates
    candidates: list[tuple[int, float]] = []
    for fidx, fc in enumerate(flow_configs):
        if fc.flow_slpm < 1e-9 or fc.I_norm_max_eff - fc.I_norm_min < 1e-6:
            candidates.append((fidx, fc.I_norm_min))
        else:
            for I_n in np.linspace(fc.I_norm_min, fc.I_norm_max_eff,
                                    N_I_GRID):
                candidates.append((fidx, float(I_n)))
    n_cand = len(candidates)

    # P ~ ||I||^2
    cand_pwr = []
    for fidx, I_norm in candidates:
        fc = flow_configs[fidx]
        cand_pwr.append(fc.power_at_min * (I_norm / fc.I_norm_min) ** 2)

    # optimistic per-step profit bounds for pruning ($/MWh -> $/J is /3.6e9)
    max_step_profit: list[float] = []
    for j in range(H):
        p = price_horizon[j]
        best_p = -math.inf
        for fc in flow_configs:
            for co_r, pwr in [(fc.co_rate_at_min, fc.power_at_min),
                              (fc.co_rate_at_max, fc.power_at_max)]:
                rev = co_r * CO_PRICE_USD_PER_KG * control_dt_s
                cost = pwr * p / 3.6e9 * control_dt_s
                best_p = max(best_p, rev - cost)
        max_step_profit.append(best_p)

    ub_rollout = -math.inf
    for fc in flow_configs:
        for co_r, pwr in [(fc.co_rate_at_min, fc.power_at_min),
                          (fc.co_rate_at_max, fc.power_at_max)]:
            rev = co_r * CO_PRICE_USD_PER_KG * control_dt_s
            cost = pwr * rollout_price / 3.6e9 * control_dt_s
            ub_rollout = max(ub_rollout, rev - cost)
    ub_rollout *= R

    ub_remaining = [0.0] * (H + 1)
    ub_remaining[H] = ub_rollout
    for j in range(H - 1, -1, -1):
        ub_remaining[j] = ub_remaining[j + 1] + max_step_profit[j]

    best: dict = {'profit': -math.inf, 'first_action': None}

    # ---- depth-first tree search ----
    def _search(step: int, prev_fidx: int, prev_pwr: float,
                cum_profit: float,
                first_action: PlannerAction | None) -> None:
        if step >= H:
            # terminal value: R steps at the current T profile
            fc_roll = flow_configs[prev_fidx]
            if fc_roll.flow_slpm > 1e-9:
                co_frozen = co_rate_kg_per_s_from_profile(
                    stepper.T, x_m, fc_roll.flow_slpm, ln_a, ea_j,
                    n_steps=30)
            else:
                co_frozen = 0.0
            term_rev = (co_frozen * CO_PRICE_USD_PER_KG * control_dt_s)
            term_cost = prev_pwr * rollout_price / 3.6e9 * control_dt_s
            terminal = (term_rev - term_cost) * R
            total = cum_profit + terminal
            if total > best['profit']:
                best['profit'] = total
                best['first_action'] = first_action
            return

        if cum_profit + ub_remaining[step] <= best['profit']:
            return

        price = price_horizon[step]
        snap = stepper.snapshot()

        for ci in range(n_cand):
            fidx, I_norm = candidates[ci]
            pwr = cand_pwr[ci]
            fc = flow_configs[fidx]

            stepper.restore(snap)

            stepper.step(fc.shape * I_norm, fc.flow_slpm, control_dt_s)

            # upper temperature constraint
            if stepper.T_max > T_HIGH_C:
                continue

            if fc.flow_slpm > 1e-9:
                co_rate = co_rate_kg_per_s_from_profile(
                    stepper.T, x_m, fc.flow_slpm, ln_a, ea_j, n_steps=30)
            else:
                co_rate = 0.0

            revenue = (co_rate * CO_PRICE_USD_PER_KG * control_dt_s)
            elec_cost = pwr * price / 3.6e9 * control_dt_s
            step_profit = revenue - elec_cost

            new_cum = cum_profit + step_profit
            fa = (PlannerAction(fidx, I_norm, fc.shape * I_norm,
                                pwr, fc.flow_slpm)
                  if step == 0 else first_action)

            if new_cum + ub_remaining[step + 1] <= best['profit']:
                continue

            _search(step + 1, fidx, pwr, new_cum, fa)

        stepper.restore(snap)

    prev = prev_flow_idx if prev_flow_idx is not None else 0
    prev_pwr = flow_configs[prev].power_at_min
    _search(0, prev, prev_pwr, 0.0, None)
    stepper.restore(snap0)

    if best['first_action'] is None:
        fc0 = flow_configs[0]
        return PlannerAction(0, fc0.I_norm_min, fc0.shape * fc0.I_norm_min,
                             fc0.power_at_min, fc0.flow_slpm)
    return best['first_action']


# --------------------------------------------------------------------------- #
# Closed-loop control trajectory: grey-box MPC + power ramp + thermal step
# --------------------------------------------------------------------------- #
def build_control_trajectory(prices_5min: np.ndarray,
                             flow_configs: list[FlowConfig],
                             sim_dt_s: float, horizon_s: float,
                             params: tg.ThermalParams,
                             variant: tg.ModelVariant,
                             x_m: np.ndarray, basis: np.ndarray,
                             ln_a: float, ea_j: float,
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, list]:
    """Closed-loop MPC: plan (flow, ||I||), ramp power, step the thermal model.

    Returns (time_s, flow_slpm, currents(n_t,6), target_P, actual_P,
             chosen: list[PlannerAction]).
    """
    n_t = int(round(horizon_s / sim_dt_s)) + 1
    time_s = np.arange(n_t) * sim_dt_s

    init_profile = np.full(x_m.size, T_INIT_C)
    stepper = ThermalStepper(params, variant, x_m, basis, init_profile)

    flow_arr = np.zeros(n_t)
    base_I = np.zeros((n_t, 6))
    base_P_arr = np.zeros(n_t)
    target_P = np.zeros(n_t)
    actual_P = np.zeros(n_t)
    chosen: list[PlannerAction] = []
    prev_flow_idx: int | None = None
    prev_actual_P = 0.0
    dwell_count = MIN_DWELL_STEPS      # allow planning at k=0

    for k in range(n_t):
        t = float(time_s[k])

        price_horizon = [price_at(t + j * CONTROL_DT_S, prices_5min)
                         for j in range(MPC_HORIZON)]

        # re-plan only once the dwell time is met
        if dwell_count >= MIN_DWELL_STEPS:
            action = plan_step_empc(stepper, flow_configs, price_horizon,
                                     prev_flow_idx, x_m, ln_a, ea_j)
            if action.flow_idx != prev_flow_idx:
                dwell_count = 0
        else:
            # hold previous action
            action = chosen[-1] if chosen else PlannerAction(
                0, flow_configs[0].I_norm_min,
                flow_configs[0].shape * flow_configs[0].I_norm_min,
                flow_configs[0].power_at_min, flow_configs[0].flow_slpm)
        dwell_count += 1

        chosen.append(action)
        target_P[k] = action.power_w
        flow_arr[k] = action.flow_slpm
        base_I[k] = action.currents_a
        base_P_arr[k] = max(action.power_w, 1e-9)

        # two-phase power ramp
        if k == 0:
            actual_P[k] = target_P[k]
        else:
            rate = (RAMP_FAST_W_PER_S
                    if prev_actual_P < RAMP_FAST_FRAC * target_P[k]
                    else RAMP_FINE_W_PER_S)
            dP_max = rate * sim_dt_s
            dP = target_P[k] - prev_actual_P
            if abs(dP) <= dP_max:
                actual_P[k] = target_P[k]
            else:
                actual_P[k] = prev_actual_P + math.copysign(dP_max, dP)
        prev_actual_P = actual_P[k]

        # I ~ sqrt(P); scale capped at 2x
        bp = base_P_arr[k]
        s = math.sqrt(max(min(actual_P[k] / bp, 4.0), 0.0))
        currents_k = action.currents_a * s
        currents_k = np.minimum(currents_k, I_MAX_A)

        # k=0 is the initial state
        if k > 0:
            dt = float(time_s[k] - time_s[k - 1])
            stepper.step(currents_k, float(flow_arr[k]), dt)

        prev_flow_idx = action.flow_idx

    scale = np.where(base_P_arr > 1e-6,
                     np.sqrt(np.clip(actual_P / base_P_arr, 0.0, 4.0)),
                     0.0)
    currents = base_I * scale[:, None]
    currents = np.minimum(currents, I_MAX_A)
    return time_s, flow_arr, currents, target_P, actual_P, chosen


# --------------------------------------------------------------------------- #
# Full-horizon trajectory optimisation (single shooting)
# --------------------------------------------------------------------------- #
COLLOC_BLOCK_S  = 300.0       # control block [s]
COLLOC_N_SUB    = 5           # thermal substeps per block (60 s each)
COLLOC_T_PENALTY = 0.005      # $/°C² penalty on T_max above T_HIGH_C (0.1x below T_LOW_C)

def solve_collocation(flow_configs: list[FlowConfig],
                      prices_5min: np.ndarray,
                      params: tg.ThermalParams,
                      variant: tg.ModelVariant,
                      x_m: np.ndarray,
                      basis: np.ndarray,
                      ln_a: float, ea_j: float,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray, object]:
    """Open-loop full-horizon optimisation by single shooting (L-BFGS-B).

    Flow (relaxed to [0, 3], shapes interpolated) and ||I|| are decision
    variables per block. Flow is rounded to the nearest SLPM afterwards and
    the trajectory rebuilt on the 60 s grid.

    Returns
    -------
    time_s, flow_arr, currents, actual_P : as in build_control_trajectory
    flow_cont, I_opt, flow_round : per-block solution
    result : scipy OptimizeResult
    """
    import scipy.optimize as sopt

    sub_dt = COLLOC_BLOCK_S / COLLOC_N_SUB
    n_blocks = int(SIM_HORIZON_S / COLLOC_BLOCK_S)

    # power scale, as in load_flow_configs
    anchor_shape = EMPIRICAL_SHAPE_BY_FLOW[0]
    anchor_norm = EMPIRICAL_I_NORM[(0, 450)]
    anchor_raw = total_em_power_w(anchor_shape * anchor_norm, params, x_m, basis)
    pscale = POWER_ANCHOR_W / max(anchor_raw, 1e-9)

    # ||I|| bounds over all flows
    I_lo = min(fc.I_norm_min for fc in flow_configs)
    I_hi = max(fc.I_norm_max_eff for fc in flow_configs)

    init_profile = np.full(x_m.size, T_INIT_C)

    def _interp_shape(fc_val: float) -> np.ndarray:
        fc_val = float(np.clip(fc_val, 0.0, 3.0))
        if abs(fc_val - round(fc_val)) < 1e-9:
            return EMPIRICAL_SHAPE_BY_FLOW[int(round(fc_val))].copy()
        lo = min(int(fc_val), 2)
        alpha = fc_val - lo
        return ((1.0 - alpha) * EMPIRICAL_SHAPE_BY_FLOW[lo]
                + alpha * EMPIRICAL_SHAPE_BY_FLOW[lo + 1])

    _eval_count = [0]

    def _objective(x: np.ndarray) -> float:
        stepper = ThermalStepper(params, variant, x_m, basis, init_profile.copy())
        profit = 0.0
        penalty = 0.0
        for b in range(n_blocks):
            fl = float(np.clip(x[2 * b], 0.0, 3.0))
            In = float(np.clip(x[2 * b + 1], I_lo, I_hi))
            shape = _interp_shape(fl)
            currents_b = shape * In
            pwr = total_em_power_w(currents_b, params, x_m, basis) * pscale
            for s in range(COLLOC_N_SUB):
                t_s = (b * COLLOC_N_SUB + s) * sub_dt
                price = price_at(t_s, prices_5min)
                stepper.step(currents_b, fl, sub_dt)
                Tm = stepper.T_max
                if fl > 0.05 and Tm > 350.0:
                    co = co_rate_kg_per_s_from_profile(
                        stepper.T, x_m, fl, ln_a, ea_j, n_steps=30)
                else:
                    co = 0.0
                profit += (co * CO_PRICE_USD_PER_KG
                           - pwr * price / 3.6e9) * sub_dt
                if Tm > T_HIGH_C:
                    penalty += COLLOC_T_PENALTY * (Tm - T_HIGH_C) ** 2
                if Tm < T_LOW_C:
                    penalty += COLLOC_T_PENALTY * 0.1 * (T_LOW_C - Tm) ** 2
        _eval_count[0] += 1
        if _eval_count[0] % 50 == 0:
            print(f"    eval {_eval_count[0]:5d}  profit=${profit:+.4f}  "
                  f"penalty=${penalty:.4f}")
        return -(profit - penalty)

    # initial guess: flow=3, mid-range ||I||
    mid_I = 0.5 * (flow_configs[3].I_norm_min + flow_configs[3].I_norm_max_eff)
    x0 = np.tile([3.0, mid_I], n_blocks)
    bounds = [(0.0, 3.0), (I_lo, I_hi)] * n_blocks

    print(f"\nDirect collocation (single shooting):")
    print(f"  {n_blocks} blocks × {COLLOC_N_SUB} sub = "
          f"{n_blocks * COLLOC_N_SUB} steps, "
          f"{2 * n_blocks} decision variables")
    print(f"  block = {COLLOC_BLOCK_S / 60:.0f} min, substep = {sub_dt:.0f} s")
    t0 = _wallclock.time()

    res = sopt.minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 200, 'disp': True,
                                 'ftol': 1e-12, 'gtol': 1e-8})

    elapsed = _wallclock.time() - t0
    print(f"  done in {elapsed:.1f} s  ({res.nit} iters, {res.nfev} f-evals, "
          f"success={res.success})")
    print(f"  message: {res.message}")

    # round flow to discrete values
    flow_cont = res.x[::2].copy()
    I_opt = res.x[1::2].copy()
    flow_round = np.clip(np.round(flow_cont).astype(int), 0, 3)

    # expand to the 60 s sim grid with the discrete shapes
    n_sim = int(SIM_HORIZON_S / SIM_DT_S) + 1
    time_s = np.arange(n_sim) * SIM_DT_S
    steps_per_block = int(COLLOC_BLOCK_S / SIM_DT_S)

    flow_arr = np.zeros(n_sim)
    currents_out = np.zeros((n_sim, 6))
    actual_P = np.zeros(n_sim)

    for b in range(n_blocks):
        fi = int(flow_round[b])
        fc = flow_configs[fi]
        In = float(np.clip(I_opt[b], fc.I_norm_min, fc.I_norm_max_eff))
        shape = EMPIRICAL_SHAPE_BY_FLOW[fi]
        cur = shape * In
        pwr = total_em_power_w(cur, params, x_m, basis) * pscale

        k0 = b * steps_per_block
        k1 = min(k0 + steps_per_block, n_sim)
        flow_arr[k0:k1] = fc.flow_slpm
        currents_out[k0:k1] = cur
        actual_P[k0:k1] = pwr

    return (time_s, flow_arr, currents_out, actual_P,
            flow_cont, I_opt, flow_round, res)


# --------------------------------------------------------------------------- #
# Closed-loop NMPC (receding-horizon single-shooting NLP)
# --------------------------------------------------------------------------- #
NMPC_HORIZON_S   = 3600.0      # lookahead [s]
NMPC_BLOCK_S     = 300.0       # control block and re-solve interval [s]
NMPC_N_SUB       = 2           # substeps per block (150 s each)
NMPC_MAX_ITER    = 15          # L-BFGS-B iterations per warm-started solve


def _build_co_lookup(x_m: np.ndarray, ln_a: float, ea_j: float,
                     n_T: int = 50, n_flow: int = 13
                     ):
    """Lookup table co_rate(T_max, flow) [kg/s], assuming a uniform T profile.

    Returns a bilinear RegularGridInterpolator (0 outside the grid).
    """
    from scipy.interpolate import RegularGridInterpolator
    T_grid = np.linspace(350.0, 520.0, n_T)
    flow_grid = np.linspace(0.0, 3.0, n_flow)
    table = np.zeros((n_T, n_flow))
    for fi, fl in enumerate(flow_grid):
        for ti, T in enumerate(T_grid):
            if fl < 0.05:
                table[ti, fi] = 0.0
            else:
                prof = np.full(x_m.size, T)
                table[ti, fi] = co_rate_kg_per_s_from_profile(
                    prof, x_m, fl, ln_a, ea_j, n_steps=30)
    return RegularGridInterpolator(
        (T_grid, flow_grid), table, method='linear',
        bounds_error=False, fill_value=0.0)

def build_control_trajectory_nmpc(
        prices_5min: np.ndarray,
        flow_configs: list[FlowConfig],
        sim_dt_s: float,
        horizon_s: float,
        params: tg.ThermalParams,
        variant: tg.ModelVariant,
        x_m: np.ndarray,
        basis: np.ndarray,
        ln_a: float, ea_j: float,
        co_lookup=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-loop NMPC: each block, solve the NLP over NMPC_HORIZON_S and
    apply the first block with flow rounded to the nearest SLPM.

    Each solve is warm-started from the previous solution shifted by one block.

    Returns (time_s, flow_arr, currents, actual_P) on the 60 s grid.
    """
    import scipy.optimize as sopt

    block_s = NMPC_BLOCK_S
    n_sub = NMPC_N_SUB
    sub_dt = block_s / n_sub
    n_total = int(horizon_s / block_s)        # blocks in the simulation
    n_hz = int(NMPC_HORIZON_S / block_s)      # blocks per NLP horizon

    anchor_shape = EMPIRICAL_SHAPE_BY_FLOW[0]
    anchor_norm = EMPIRICAL_I_NORM[(0, 450)]
    anchor_raw = total_em_power_w(anchor_shape * anchor_norm, params, x_m, basis)
    pscale = POWER_ANCHOR_W / max(anchor_raw, 1e-9)

    I_lo = min(fc.I_norm_min for fc in flow_configs)
    I_hi = max(fc.I_norm_max_eff for fc in flow_configs)

    init_profile = np.full(x_m.size, T_INIT_C)

    def _interp_shape(fc_val: float) -> np.ndarray:
        fc_val = float(np.clip(fc_val, 0.0, 3.0))
        if abs(fc_val - round(fc_val)) < 1e-9:
            return EMPIRICAL_SHAPE_BY_FLOW[int(round(fc_val))].copy()
        lo = min(int(fc_val), 2)
        alpha = fc_val - lo
        return ((1.0 - alpha) * EMPIRICAL_SHAPE_BY_FLOW[lo]
                + alpha * EMPIRICAL_SHAPE_BY_FLOW[lo + 1])

    # ---- inner NLP ----
    def _solve_block(snap, t_start, n_blk, x0):
        """Solve the horizon NLP from thermal state snap."""
        # one stepper, restored to snap on every eval
        s = ThermalStepper(params, variant, x_m, basis, init_profile.copy())

        def _obj(x):
            s.restore(snap)
            profit = 0.0
            penalty = 0.0
            for b in range(n_blk):
                fl = float(np.clip(x[2 * b], 0.0, 3.0))
                In = float(np.clip(x[2 * b + 1], I_lo, I_hi))
                shape = _interp_shape(fl)
                cur = shape * In
                pwr = total_em_power_w(cur, params, x_m, basis) * pscale
                for ss in range(n_sub):
                    t = t_start + (b * n_sub + ss) * sub_dt
                    price = price_at(t, prices_5min)
                    s.step(cur, fl, sub_dt)
                    Tm = s.T_max
                    if co_lookup is not None and fl > 0.05 and Tm > 350.0:
                        co = float(co_lookup((Tm, fl)))
                    elif fl > 0.05 and Tm > 350.0:
                        co = co_rate_kg_per_s_from_profile(
                            s.T, x_m, fl, ln_a, ea_j, n_steps=30)
                    else:
                        co = 0.0
                    profit += (co * CO_PRICE_USD_PER_KG
                               - pwr * price / 3.6e9) * sub_dt
                    if Tm > T_HIGH_C:
                        penalty += COLLOC_T_PENALTY * (Tm - T_HIGH_C) ** 2
                    if Tm < T_LOW_C:
                        penalty += COLLOC_T_PENALTY * 0.1 * (T_LOW_C - Tm) ** 2
            return -(profit - penalty)

        bounds = [(0.0, 3.0), (I_lo, I_hi)] * n_blk
        return sopt.minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': NMPC_MAX_ITER,
                                      'ftol': 1e-12, 'gtol': 1e-8})

    # ---- NMPC loop ----
    # plant state
    stepper = ThermalStepper(params, variant, x_m, basis, init_profile.copy())

    applied_flow = np.zeros(n_total, dtype=int)
    applied_I = np.zeros(n_total)

    mid_I = 0.5 * (flow_configs[3].I_norm_min + flow_configs[3].I_norm_max_eff)
    prev_x: np.ndarray | None = None

    print(f"\nClosed-loop NMPC (collocation planner):")
    print(f"  {n_total} blocks × {block_s / 60:.0f} min, "
          f"horizon = {n_hz} blocks ({NMPC_HORIZON_S / 60:.0f} min), "
          f"re-solve every block")
    t0 = _wallclock.time()

    for k in range(n_total):
        t_now = k * block_s
        n_blk = min(n_hz, n_total - k)   # horizon shrinks at the end

        # warm start: shift previous solution by one block, repeat the last
        if prev_x is not None:
            n_prev_blk = len(prev_x) // 2
            if n_prev_blk > 1:
                shifted = np.concatenate([prev_x[2:], prev_x[-2:]])
            else:
                shifted = prev_x.copy()
            x0 = shifted[:2 * n_blk]
            if len(x0) < 2 * n_blk:
                pad = np.tile(x0[-2:], (n_blk - len(x0) // 2))
                x0 = np.concatenate([x0, pad])[:2 * n_blk]
        else:
            x0 = np.tile([3.0, mid_I], n_blk)

        snap = stepper.snapshot()
        res = _solve_block(snap, t_now, n_blk, x0)
        prev_x = res.x.copy()

        # first block, flow rounded
        fi = int(np.clip(round(float(res.x[0])), 0, 3))
        fc = flow_configs[fi]
        In = float(np.clip(res.x[1], fc.I_norm_min, fc.I_norm_max_eff))
        applied_flow[k] = fi
        applied_I[k] = In

        shape = EMPIRICAL_SHAPE_BY_FLOW[fi]
        cur = shape * In
        for _ in range(n_sub):
            stepper.step(cur, float(fi), sub_dt)

        if (k + 1) % 6 == 0 or k == 0 or k == n_total - 1:
            el = _wallclock.time() - t0
            print(f"    block {k + 1:3d}/{n_total}  t={t_now / 3600:.2f}h  "
                  f"flow={fi}  ||I||={In:.3f}  P={total_em_power_w(cur, params, x_m, basis) * pscale:.1f}W  "
                  f"[{el:.0f}s, {res.nit}it]")

    elapsed = _wallclock.time() - t0
    print(f"  NMPC done in {elapsed:.1f} s")

    # expand to the sim grid
    n_sim = int(horizon_s / sim_dt_s) + 1
    time_s = np.arange(n_sim) * sim_dt_s
    steps_per_block = int(block_s / sim_dt_s)

    flow_arr = np.zeros(n_sim)
    currents_out = np.zeros((n_sim, 6))
    actual_P = np.zeros(n_sim)

    for b in range(n_total):
        fi = int(applied_flow[b])
        In = float(applied_I[b])
        shape = EMPIRICAL_SHAPE_BY_FLOW[fi]
        cur = shape * In
        pwr = total_em_power_w(cur, params, x_m, basis) * pscale
        k0 = b * steps_per_block
        k1 = min(k0 + steps_per_block, n_sim)
        flow_arr[k0:k1] = float(fi)
        currents_out[k0:k1] = cur
        actual_P[k0:k1] = pwr

    return time_s, flow_arr, currents_out, actual_P


# --------------------------------------------------------------------------- #
# Full-horizon thermal sim with the chosen controls
# --------------------------------------------------------------------------- #
def run_thermal_sim(time_s: np.ndarray, currents: np.ndarray,
                    flow_arr: np.ndarray, params: tg.ThermalParams,
                    variant: tg.ModelVariant, x_m: np.ndarray,
                    basis: np.ndarray,
                    init_profile_c: np.ndarray | None = None) -> np.ndarray:
    n_t, n_n = time_s.size, x_m.size
    b2 = tg.axial_field_squared(currents, basis)
    init = (np.full(n_n, T_INIT_C) if init_profile_c is None
            else np.asarray(init_profile_c, dtype=float))
    temps_init = np.tile(init, (n_t, 1))
    exp = tg.ExperimentData(
        name="empc_24h",
        source_path=Path("<empc>"),
        time_s=time_s.copy(),
        flow_slpm=flow_arr.copy(),
        currents_a=currents.copy(),
        currents_raw_a=currents.copy(),
        temps_c=temps_init,
        observed_mask=np.ones(n_n, dtype=bool),
        x_m=x_m,
        raw_x_m=x_m,
        dt_median_s=float(SIM_DT_S),
        field_basis=basis,
        b2=b2,
    )
    pred, _ = tg.simulate_experiment(exp, params, variant, max_substep_s=6.0)
    return pred


# --------------------------------------------------------------------------- #
# Conversion + economics post-processing (per time step)
# --------------------------------------------------------------------------- #
def post_process_economics(time_s: np.ndarray, temps_c: np.ndarray,
                           flow_arr: np.ndarray, actual_P: np.ndarray,
                           prices_5min: np.ndarray, x_m: np.ndarray,
                           ln_a: float, ea_j: float
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (CO rate [kg/s], electricity cost [$/s], price [$/MWh]) per step."""
    n = time_s.size
    co_rate = np.zeros(n)
    elec_cost = np.zeros(n)
    price_t = np.zeros(n)
    for k in range(n):
        if flow_arr[k] > 1e-9 and np.nanmax(temps_c[k]) > 350.0:
            co_rate[k] = co_rate_kg_per_s_from_profile(
                temps_c[k], x_m, float(flow_arr[k]), ln_a, ea_j, n_steps=120)
        else:
            co_rate[k] = 0.0
        price_t[k] = price_at(float(time_s[k]), prices_5min)
        elec_cost[k] = actual_P[k] * price_t[k] / 3.6e9
    return co_rate, elec_cost, price_t


# --------------------------------------------------------------------------- #
# Live animation
# --------------------------------------------------------------------------- #
def animate(time_s: np.ndarray, temps_c: np.ndarray, currents: np.ndarray,
            flow_arr: np.ndarray, actual_P: np.ndarray,
            co_rate: np.ndarray, elec_cost: np.ndarray, price_t: np.ndarray,
            x_m: np.ndarray, save_gif: bool = False, show: bool = True,
            png_path: Path | None = None) -> None:
    n_t = time_s.size
    coil_x_mm = (tg.COIL_WINDING_START_M + np.arange(tg.N_COILS) *
                 tg.COIL_START_PITCH_M + 0.5 * tg.COIL_PACK_LENGTH_M) * 1000.0
    x_mm = x_m * 1000.0
    hours = time_s / 3600.0

    # frame stride from SIM_S_PER_FRAME
    stride = max(1, int(round(SIM_S_PER_FRAME / SIM_DT_S)))
    frames = list(range(0, n_t, stride))
    if frames[-1] != n_t - 1:
        frames.append(n_t - 1)

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.0],
                          height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.32)
    ax_T = fig.add_subplot(gs[0, 0])        # T profile
    ax_I = fig.add_subplot(gs[1, 0])        # coil currents
    ax_F = fig.add_subplot(gs[2, 0])        # flow vs time
    ax_P = fig.add_subplot(gs[0, 1])        # price vs time
    ax_X = fig.add_subplot(gs[1, 1])        # cumulative profit
    ax_C = fig.add_subplot(gs[2, 1])        # power vs time

    # Temperature panel
    ln_T, = ax_T.plot(x_mm, temps_c[0], "-o", color="#c0392b", ms=4, lw=2)
    ax_T.axhspan(T_LOW_C, T_HIGH_C, color="0.85", alpha=0.5)
    ax_T.axhline(T_LOW_C, color="0.55", lw=0.8)
    ax_T.axhline(T_HIGH_C, color="0.55", lw=0.8)
    ax_T.set_xlabel("axial position [mm]")
    ax_T.set_ylabel("wall T [\N{degree sign}C]")
    ax_T.set_ylim(150, 560)
    ax_T.set_title("Wall temperature profile")
    ax_T.grid(alpha=0.3)
    txt_T = ax_T.text(0.02, 0.95, "", transform=ax_T.transAxes,
                      va="top", ha="left", fontsize=10,
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))

    # Currents panel (bars at coil positions)
    bars = ax_I.bar(coil_x_mm, currents[0], width=12.0, color="#2980b9",
                    edgecolor="black", alpha=0.85)
    ax_I.set_xlabel("coil axial position [mm]")
    ax_I.set_ylabel("coil current [A]")
    ax_I.set_ylim(0.0, I_MAX_A + 0.3)
    ax_I.axhline(I_MAX_A, color="0.5", lw=0.8, ls="--")
    ax_I.set_title("Non-normalised coil currents (6 coils)")
    ax_I.grid(alpha=0.3, axis="y")
    txt_I = ax_I.text(0.98, 0.92, "", transform=ax_I.transAxes,
                      va="top", ha="right", fontsize=9, family="monospace",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))

    # Flow vs time
    ax_F.plot(hours, flow_arr, color="#27ae60", lw=1.2, alpha=0.55)
    flow_marker, = ax_F.plot([hours[0]], [flow_arr[0]], "o",
                             color="#1e7e34", ms=8)
    ax_F.set_xlabel("time [h]")
    ax_F.set_ylabel("flow [SLPM]")
    ax_F.set_ylim(-0.3, 3.3)
    ax_F.set_xlim(0, 24)
    ax_F.set_title("Flow setpoint")
    ax_F.grid(alpha=0.3)
    cursor_F = ax_F.axvline(hours[0], color="0.5", lw=0.7)

    # Price
    ax_P.plot(hours, price_t, color="#8e44ad", lw=1.2)
    price_marker, = ax_P.plot([hours[0]], [price_t[0]], "o",
                              color="#5b2c83", ms=7)
    ax_P.set_xlabel("time [h]")
    ax_P.set_ylabel("price [$/MWh]")
    ax_P.set_xlim(0, 24)
    ax_P.set_title("Real-time electricity price (LZ South, Nov 5)")
    ax_P.grid(alpha=0.3)

    # Cumulative profit
    co_revenue = np.cumsum(co_rate) * SIM_DT_S * CO_PRICE_USD_PER_KG
    cum_cost   = np.cumsum(elec_cost) * SIM_DT_S
    profit_t   = co_revenue - cum_cost
    ax_X.plot(hours, profit_t, color="#16a085", lw=1.5,
              label="cum. profit")
    ax_X.plot(hours, co_revenue, color="#16a085", lw=0.8,
              alpha=0.45, ls="--", label="cum. CO revenue")
    ax_X.plot(hours, cum_cost, color="#c0392b", lw=0.8,
              alpha=0.45, ls=":", label="cum. elec. cost")
    profit_marker, = ax_X.plot([hours[0]], [profit_t[0]], "o",
                                color="#0f6450", ms=7)
    ax_X.set_xlabel("time [h]")
    ax_X.set_ylabel("USD")
    ax_X.set_xlim(0, 24)
    ax_X.set_title("Cumulative economics")
    ax_X.legend(loc="lower right", fontsize=8)
    ax_X.grid(alpha=0.3)

    # Power vs time
    ax_C.plot(hours, actual_P, color="#d35400", lw=1.0, alpha=0.6)
    power_marker, = ax_C.plot([hours[0]], [actual_P[0]], "o",
                              color="#a04000", ms=7)
    ax_C.set_xlabel("time [h]")
    ax_C.set_ylabel("input power [W]")
    ax_C.set_xlim(0, 24)
    ax_C.set_title(f"Total deposited power (ramp: {RAMP_FAST_W_PER_S:.0f}/{RAMP_FINE_W_PER_S:.0f} W/s)")
    ax_C.grid(alpha=0.3)

    fig.suptitle("Economic-MPC digital twin: RWGS reactor, 24 h "
                 f"(grey-box + PFR planner, H={MPC_HORIZON})", fontsize=13)

    def update(k):
        ln_T.set_ydata(temps_c[k])
        for bar, h in zip(bars, currents[k]):
            bar.set_height(max(float(h), 0.0))
        flow_marker.set_data([hours[k]], [flow_arr[k]])
        price_marker.set_data([hours[k]], [price_t[k]])
        cursor_F.set_xdata([hours[k], hours[k]])
        profit_marker.set_data([hours[k]], [profit_t[k]])
        power_marker.set_data([hours[k]], [actual_P[k]])
        t_max = float(np.nanmax(temps_c[k]))
        t_min = float(np.nanmin(temps_c[k]))
        cumulative = profit_t[k]
        txt_T.set_text(
            f"t = {hours[k]:5.2f} h\n"
            f"T_max = {t_max:5.1f} \N{degree sign}C\n"
            f"T_min = {t_min:5.1f} \N{degree sign}C\n"
            f"flow  = {flow_arr[k]:.0f} SLPM\n"
            f"price = ${price_t[k]:.2f}/MWh\n"
            f"power = {actual_P[k]:.0f} W\n"
            f"CO    = {co_rate[k]*1000.0:.3f} g/s\n"
            f"profit cum = ${cumulative:+.3f}"
        )
        txt_I.set_text("  ".join(f"I{i+1}={currents[k,i]:4.2f}" for i in range(6)))
        return (ln_T, *bars, flow_marker, price_marker, cursor_F,
                profit_marker, power_marker, txt_T, txt_I)

    anim = FuncAnimation(fig, update, frames=frames,
                         interval=FRAME_INTERVAL_MS, blit=False, repeat=False)
    if save_gif:
        GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  writing {GIF_PATH} (this can take a minute) ...")
        anim.save(str(GIF_PATH), writer="pillow", fps=20)
        print(f"  saved {GIF_PATH}")
    if png_path is not None:
        # final-frame PNG
        update(n_t - 1)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        print(f"  saved {png_path}")
    if show:
        plt.show()
    return anim


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Loading thermal model + parameters ...")
    variant_name, params = tg.load_params(PARAMS_FILE)
    variant = tg.MODEL_VARIANTS[variant_name]
    x_m = tg.model_x_grid()
    basis = tg.coil_field_basis(x_m)

    print("Loading RWGS kinetic fit ...")
    rwgs = json.loads(RWGS_FIT.read_text(encoding="utf-8"))
    fit = next(f for f in rwgs["fits"] if f["name"] == "profile_pfr")
    ln_a = float(fit["parameters"]["ln_Af_m3_per_mol_s"])
    ea_j = float(fit["parameters"]["Ea_kJ_per_mol"]) * 1000.0
    print(f"  ln(Af) = {ln_a:.3f}, Ea = {ea_j/1000:.2f} kJ/mol")

    print("Building flow configurations (4 flows x continuous power) ...")
    flow_configs = load_flow_configs(params, x_m, basis, ln_a, ea_j)
    eff_co = CO_PRICE_USD_PER_KG
    print(f"  (CO revenue: ${eff_co:.2f}/kg)")
    print(f"  {'flow':>6} {'||I|| range':>16} {'P range (W)':>18}"
          f" {'CO range (g/s)':>18} {'BE range ($/MWh)':>20}")
    print("  " + "-" * 80)
    for fc in flow_configs:
        be_min = (fc.co_rate_at_min * eff_co * 3.6e9
                  / max(fc.power_at_min, 1e-9)
                  if fc.power_at_min > 1e-9 else 0.0)
        be_max = (fc.co_rate_at_max * eff_co * 3.6e9
                  / max(fc.power_at_max, 1e-9)
                  if fc.power_at_max > 1e-9 else 0.0)
        print(f"  {fc.flow_slpm:6.0f}"
              f" [{fc.I_norm_min:.3f}, {fc.I_norm_max_eff:.3f}]"
              f"   [{fc.power_at_min:6.1f}, {fc.power_at_max:6.1f}]"
              f"   [{fc.co_rate_at_min*1000:7.3f}, {fc.co_rate_at_max*1000:7.3f}]"
              f"   [{be_min:6.1f}, {be_max:6.1f}]")

    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gif", action="store_true",
                    help="save animation to plots/empc_digital_twin_24h.gif")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open the live window (still writes PNG/GIF)")
    ap.add_argument("--collocation", action="store_true",
                    help="use full-horizon trajectory optimisation (open-loop)")
    ap.add_argument("--nmpc", action="store_true",
                    help="closed-loop NMPC with collocation planner")
    args, _ = ap.parse_known_args()

    print("\nLoading real-time price ...")
    prices = load_realtime_price(PRICE_XLSX)
    print(f"  loaded {prices.size} 5-min bins, "
          f"min ${prices.min():.2f}, mean ${prices.mean():.2f}, "
          f"max ${prices.max():.2f}")

    t0 = _wallclock.time()

    if args.nmpc:
        # ---- closed-loop NMPC ----
        print("Building CO rate lookup table ...")
        co_lookup = _build_co_lookup(x_m, ln_a, ea_j)
        print("  done")
        time_s, flow_arr, currents, actual_P = \
            build_control_trajectory_nmpc(
                prices, flow_configs, SIM_DT_S, SIM_HORIZON_S,
                params, variant, x_m, basis, ln_a, ea_j,
                co_lookup=co_lookup)
        n_t = time_s.size

        # Flow usage summary
        steps_per_block = int(NMPC_BLOCK_S / SIM_DT_S)
        n_blk_total = int(SIM_HORIZON_S / NMPC_BLOCK_S)
        from collections import Counter
        block_flows = [int(flow_arr[b * steps_per_block]) for b in range(n_blk_total)]
        block_cnts = Counter(block_flows)
        print("  flow usage (NMPC blocks):")
        for fidx in sorted(block_cnts.keys()):
            n_b = block_cnts[fidx]
            mins = n_b * NMPC_BLOCK_S / 60.0
            p_vals = []
            for b in range(n_blk_total):
                if block_flows[b] == fidx:
                    k0 = b * steps_per_block
                    p_vals.append(actual_P[min(k0, n_t - 1)])
            avg_P = float(np.mean(p_vals)) if p_vals else 0.0
            print(f"    flow={fidx} SLPM: {n_b:4d} blocks"
                  f" ({mins:5.1f} min), avg P={avg_P:.1f} W")

    elif args.collocation:
        # ---- open-loop full-horizon optimisation ----
        (time_s, flow_arr, currents, actual_P,
         flow_cont, I_opt, flow_round, opt_res) = solve_collocation(
            flow_configs, prices, params, variant, x_m, basis, ln_a, ea_j)
        n_t = time_s.size

        # Flow usage summary (rounded blocks)
        steps_per_block = int(COLLOC_BLOCK_S / SIM_DT_S)
        n_blocks = len(flow_round)
        print("  flow usage (rounded collocation blocks):")
        from collections import Counter
        block_cnts = Counter(flow_round.tolist())
        for fidx in sorted(block_cnts.keys()):
            n_blk = block_cnts[fidx]
            mins = n_blk * COLLOC_BLOCK_S / 60.0
            mask = flow_round == fidx
            avg_I = float(np.mean(I_opt[mask])) if mask.any() else 0.0
            avg_P = float(np.mean(actual_P[
                fidx * steps_per_block:(fidx + 1) * steps_per_block]))
            # avg power over all blocks at this flow
            p_vals = []
            for b in range(n_blocks):
                if flow_round[b] == fidx:
                    k0 = b * steps_per_block
                    p_vals.append(actual_P[min(k0, n_t - 1)])
            avg_P = float(np.mean(p_vals)) if p_vals else 0.0
            print(f"    flow={fidx} SLPM: {n_blk:4d} blocks"
                  f" ({mins:5.1f} min),"
                  f" avg ||I||={avg_I:.3f},"
                  f" avg P={avg_P:.1f} W")
    else:
        # ---- receding-horizon MPC ----
        print(f"Running closed-loop grey-box MPC (horizon={MPC_HORIZON} steps, "
              f"continuous power, {N_I_GRID} ||I|| grid) ...")
        time_s, flow_arr, currents, target_P, actual_P, chosen = \
            build_control_trajectory(
                prices, flow_configs, SIM_DT_S, SIM_HORIZON_S,
                params, variant, x_m, basis, ln_a, ea_j)
        n_t = time_s.size
        print(f"  sim grid: {n_t} steps @ {SIM_DT_S:.0f} s")

        # Flow usage summary
        flow_cnt: dict[int, int] = {}
        flow_I_sum: dict[int, float] = {}
        flow_P_sum: dict[int, float] = {}
        for action in chosen:
            fidx = action.flow_idx
            flow_cnt[fidx] = flow_cnt.get(fidx, 0) + 1
            flow_I_sum[fidx] = flow_I_sum.get(fidx, 0.0) + action.I_norm
            flow_P_sum[fidx] = flow_P_sum.get(fidx, 0.0) + action.power_w
        print("  flow usage (control steps):")
        for fidx, fc in enumerate(flow_configs):
            if fidx in flow_cnt:
                n = flow_cnt[fidx]
                avg_I = flow_I_sum[fidx] / n
                avg_P = flow_P_sum[fidx] / n
                mins = n * CONTROL_DT_S / 60.0
                print(f"    flow={fc.flow_slpm:.0f} SLPM: {n:4d} steps"
                      f" ({mins:5.1f} min),"
                      f" avg ||I||={avg_I:.3f},"
                      f" avg P={avg_P:.1f} W")

    print("Rolling final thermal model forward across 24 h ...")
    init_profile = np.full(x_m.size, T_INIT_C)
    temps = run_thermal_sim(time_s, currents, flow_arr, params, variant,
                            x_m, basis, init_profile_c=init_profile)
    print(f"  MPC + thermal sim done in {_wallclock.time()-t0:.1f} s")
    t_peak_per_step = np.nanmax(temps, axis=1)
    t_peak_max = float(np.nanmax(t_peak_per_step))
    t_peak_min = float(np.nanmin(t_peak_per_step))
    n_viol_low  = int(np.sum(t_peak_per_step < T_LOW_C  - 0.5))
    n_viol_high = int(np.sum(t_peak_per_step > T_HIGH_C + 0.5))
    print(f"  T_max range over the run = [{t_peak_min:.1f}, {t_peak_max:.1f}] C"
          f"  (window: [{T_LOW_C:.0f}, {T_HIGH_C:.0f}])")
    if n_viol_low or n_viol_high:
        print(f"  WARNING: T_max constraint violated -- "
              f"{n_viol_low} steps below {T_LOW_C:.0f} C, "
              f"{n_viol_high} steps above {T_HIGH_C:.0f} C")
    else:
        print(f"  T_max constraint OK at all {len(t_peak_per_step)} steps")

    print("Computing economics ...")
    co_rate, elec_cost, price_t = post_process_economics(
        time_s, temps, flow_arr, actual_P, prices, x_m, ln_a, ea_j)
    total_rev = float(np.sum(co_rate) * SIM_DT_S * CO_PRICE_USD_PER_KG)
    total_cost = float(np.sum(elec_cost) * SIM_DT_S)
    total_co_g = float(np.sum(co_rate) * SIM_DT_S * 1000.0)
    print(f"  24-h CO produced  = {total_co_g:7.2f} g")
    print(f"  24-h revenue      = ${total_rev:7.3f}  "
          f"(@${CO_PRICE_USD_PER_KG:.2f}/kg)")
    print(f"  24-h elec. cost   = ${total_cost:7.3f}")
    print(f"  24-h NET PROFIT   = ${total_rev-total_cost:+7.3f}")

    # ---- static flow vs price plot ---------------------------------------- #
    import matplotlib.pyplot as _plt_static
    hours = time_s / 3600.0
    fig_fp, ax_flow = _plt_static.subplots(figsize=(12, 4))
    color_flow = "#1f77b4"
    color_price = "#d62728"
    ax_flow.step(hours, flow_arr, where="post", color=color_flow, linewidth=1.2,
                 label="Flow (SLPM)")
    ax_flow.set_xlabel("Time (h)")
    ax_flow.set_ylabel("Flow (SLPM)", color=color_flow)
    ax_flow.set_ylim(-0.2, max(flow_arr) + 0.5)
    ax_flow.tick_params(axis="y", labelcolor=color_flow)

    ax_price = ax_flow.twinx()
    ax_price.plot(hours, price_t,                               # already $/MWh
                  color=color_price, linewidth=0.8, alpha=0.7,
                  label="Elec price ($/MWh)")
    ax_price.set_ylabel("Electricity price ($/MWh)", color=color_price)
    ax_price.tick_params(axis="y", labelcolor=color_price)
    ax_flow.set_xlim(hours[0], hours[-1])
    ax_flow.set_title("MPC Flow Decision vs Electricity Price")
    lines_f, labels_f = ax_flow.get_legend_handles_labels()
    lines_p, labels_p = ax_price.get_legend_handles_labels()
    ax_flow.legend(lines_f + lines_p, labels_f + labels_p, loc="upper left",
                   fontsize=8, framealpha=0.8)
    fig_fp.tight_layout()
    fp_png = WORKSPACE / "plots" / "empc_flow_vs_price.png"
    fp_png.parent.mkdir(parents=True, exist_ok=True)
    fig_fp.savefig(fp_png, dpi=150)
    _plt_static.close(fig_fp)
    print(f"  saved {fp_png}")

    png_out = WORKSPACE / "plots" / "empc_digital_twin_24h.png"
    print("\nLaunching live animation (close the window to exit) ...")
    animate(time_s, temps, currents, flow_arr, actual_P,
            co_rate, elec_cost, price_t, x_m,
            save_gif=args.gif, show=not args.no_show, png_path=png_out)


if __name__ == "__main__":
    main()
