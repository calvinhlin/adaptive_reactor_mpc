"""Solar-tracking MPC digital twin of the RWGS reactor (T = 500 C setpoint).

Holds T_max near a 500 C setpoint while the available power P_avail(t)
follows a synthetic 24-h cloudy-solar trace. Each control step the MPC picks
a discrete mode (flow + axial current profile) and a current scale so the
commanded power stays within P_avail(t).

Control inputs (as in empc_digital_twin.py):
    - Flow ∈ {0, 1, 2, 3} SLPM. Each flow has a fixed axial current profile
      scaled to reach T_max = 500 C (mode bank from emp.load_modes).
    - Current scale s = sqrt(P/P_base), capped by P_avail(t).

Constraints:
    - Coil currents ≤ emp.I_MAX_A (5.2 A)
    - Two-phase power ramp: 40 W/s until 90% of target, then 10 W/s
    - Commanded power ≤ P_avail(t)
    - Soft objective (T_max - 500)²; holding 500 C needs at least the
      flow=0 base power (P_FLOOR_W = 167 W).

Run:
    .\\.venv\\Scripts\\python.exe tracking_mpc_solar_digital_twin_T500.py             # live window
    .\\.venv\\Scripts\\python.exe tracking_mpc_solar_digital_twin_T500.py --gif       # also writes GIF
    .\\.venv\\Scripts\\python.exe tracking_mpc_solar_digital_twin_T500.py --no-show   # only writes PNG
"""

from __future__ import annotations

import math
import sys
import time as _wallclock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

# Mode bank, thermal sim, and helpers from the economic-MPC twin.
import empc_digital_twin as emp
import thermal_greybox as tg


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
T_SETPOINT_C    = 500.0        # tracking target
P_FLOOR_W       = 167.0       # minimum P_avail over 24 h; = flow=0/T=500 base
                               # power, enough to hold 500 C at 0 SLPM
P_PEAK_SOLAR_W  = 30.0        # clear-sky midday addition on top of the floor
                               # -> peak P_avail ≈ 167 + 30 = 197 W
CLOUD_TAU_S     = 900.0        # cloud-noise correlation time (15 min)
CLOUD_SIGMA     = 0.30         # cloud-noise std of the multiplier (clipped to [0.05, 1])
CLOUD_SEED      = 7

# Shell creep is <0.1 C over 24 h with the empirical current shapes, so no
# derate: the modes reach 500 C at scale = 1.0.
TRACKING_DERATE_T500 = 1.000

SIM_HORIZON_S   = 24 * 3600
SIM_DT_S        = 60.0
CONTROL_DT_S    = 60.0
T_INIT_C        = 490.0        # start below setpoint to avoid initial overshoot

FRAME_INTERVAL_MS = 50
SIM_S_PER_FRAME   = 72.0        # sim seconds per frame (rounded to whole SIM_DT_S steps)

GIF_PATH = WORKSPACE / "plots" / "tracking_mpc_solar_T500_24h.gif"


# --------------------------------------------------------------------------- #
# Cloudy solar power profile
# --------------------------------------------------------------------------- #
def generate_cloudy_solar(time_s: np.ndarray,
                          floor_w: float = P_FLOOR_W,
                          peak_w: float = P_PEAK_SOLAR_W,
                          cloud_tau_s: float = CLOUD_TAU_S,
                          cloud_sigma: float = CLOUD_SIGMA,
                          seed: int = CLOUD_SEED) -> np.ndarray:
    """Synthetic 24-h cloudy-solar available-power trace [W].

        P_avail(t) = floor_w + peak_w * arc(t) * cloud(t)
        arc(t)     = sin(pi (h-6.5)/11)^2 for 6.5 h <= h <= 17.5 h, else 0
        cloud(t)   = clip(1 + x(t), 0.05, 1.0), x = OU process

    The OU process gives smooth cloud dips on the cloud_tau_s timescale
    instead of independent per-step noise.
    """
    rng = np.random.default_rng(seed)
    n = time_s.size
    hours = time_s / 3600.0

    # Daylight envelope: sin² between 06:30 and 17:30, zero otherwise.
    arc = np.zeros(n)
    day_mask = (hours >= 6.5) & (hours <= 17.5)
    arc[day_mask] = np.sin(np.pi * (hours[day_mask] - 6.5) / 11.0) ** 2

    # OU cloud noise, dx = -x/tau dt + sigma sqrt(2/tau) dW, exact
    # discretisation (stationary std = sigma).
    dt = float(time_s[1] - time_s[0]) if n > 1 else 1.0
    decay = math.exp(-dt / cloud_tau_s)
    drive = cloud_sigma * math.sqrt(1.0 - decay * decay)
    x = 0.0
    cloud = np.empty(n)
    for k in range(n):
        x = decay * x + drive * float(rng.standard_normal())
        cloud[k] = x
    cloud_mult = np.clip(1.0 + cloud, 0.05, 1.0)

    p_avail = floor_w + peak_w * arc * cloud_mult
    return p_avail


# --------------------------------------------------------------------------- #
# MPC planner: each step pick (flow mode, current scale) with P ≤ P_avail and
# T_max closest to T_SETPOINT_C.
# --------------------------------------------------------------------------- #
@dataclass
class PlanStep:
    mode_idx: int     # index into the T=500 modes (flow = 0, 1, 2, 3)
    scale: float      # current scale in [0, 1]; scale² = power ratio
    p_target_w: float # commanded power for this step (≤ P_avail)


# --- Receding-horizon MPC parameters ----------------------------------------
MPC_HORIZON        = 5       # look-ahead control periods (each = CONTROL_DT_S)
TAU_THERMAL_S      = 90.0    # first-order thermal time constant for prediction [s]
SWITCH_PENALTY     = 5.0     # mode-switch cost [°C² per switch]
CURTAILMENT_WEIGHT = 0.01    # cost per W² of unused solar power per step
T_SS_BASE_C        = 500.0   # steady-state T_max at scale = 1.0 for all T=500 modes
T_AMB_EFF_DEFAULT  = 200.0   # fallback effective ambient [°C]; main() uses params

FLOW_UP_MARGIN_W   = 0.0     # extra headroom [W] for an upswitch in plan_step


def _predict_T(T_cur: float, scale: float, t_amb: float, dt_s: float) -> float:
    """First-order lag prediction of T_max.

    T_ss       = T_amb + (500 - T_amb) * scale²
    T(t + dt)  = T(t) + [T_ss - T(t)] * [1 - exp(-dt/τ)]
    """
    T_ss = t_amb + (T_SS_BASE_C - t_amb) * scale * scale
    alpha = 1.0 - math.exp(-dt_s / TAU_THERMAL_S)
    return T_cur + alpha * (T_ss - T_cur)


def plan_step_mpc(p_avail_horizon: list[float], modes_t500: list[emp.Mode],
                  prev_mode_idx: int | None, T_current: float,
                  t_amb: float = T_AMB_EFF_DEFAULT) -> PlanStep:
    """Receding-horizon MPC over H control periods.

    Branch-and-bound over mode sequences (upswitch one rank per step, only if
    P_avail ≥ base power), T_max predicted with _predict_T. Returns the first
    action of the cheapest sequence.

    Stage cost: (T - T_setpoint)² + SWITCH_PENALTY [if switched]
                + CURTAILMENT_WEIGHT * curtailed²

    Higher ranks are searched first so pruning favours high-flow paths
    (lower curtailment cost).
    """
    MAX_SCALE = 1.0

    sorted_pairs = sorted(enumerate(modes_t500),
                          key=lambda x: x[1].base_power_w)
    rank_of = {idx: rank for rank, (idx, _) in enumerate(sorted_pairs)}
    n_modes = len(sorted_pairs)

    cur_rank = rank_of[prev_mode_idx] if prev_mode_idx is not None else 0
    H = min(MPC_HORIZON, len(p_avail_horizon))

    best = {'cost': float('inf'), 'action': None}

    def _search(step: int, rank: int, T: float, cost: float,
                first_action: tuple | None) -> None:
        if step >= H:
            if cost < best['cost']:
                best['cost'] = cost
                best['action'] = first_action
            return
        if cost >= best['cost']:
            return  # prune

        p_avail = p_avail_horizon[step]

        for next_rank in reversed(range(n_modes)):
            if next_rank > rank + 1:
                continue  # single-step upswitch only
            idx, m = sorted_pairs[next_rank]
            if next_rank > rank and p_avail < m.base_power_w:
                continue  # upswitch needs at least the mode's base power

            s = math.sqrt(max(p_avail / max(m.base_power_w, 1e-9), 0.0))
            s = min(s, MAX_SCALE)

            T_new = _predict_T(T, s, t_amb, CONTROL_DT_S)

            p_target = (s * s) * m.base_power_w
            curtailed = max(0.0, p_avail - p_target)
            sw = SWITCH_PENALTY if next_rank != rank else 0.0
            new_cost = (cost + (T_new - T_SETPOINT_C) ** 2
                        + sw + CURTAILMENT_WEIGHT * curtailed * curtailed)

            if new_cost >= best['cost']:
                continue

            fa = (idx, s, p_target) if step == 0 else first_action
            _search(step + 1, next_rank, T_new, new_cost, fa)

    _search(0, cur_rank, T_current, 0.0, None)

    if best['action'] is None:
        idx, m = sorted_pairs[0]
        p0 = p_avail_horizon[0] if p_avail_horizon else 0.0
        s = math.sqrt(max(p0 / max(m.base_power_w, 1e-9), 0.0))
        s = min(s, MAX_SCALE)
        return PlanStep(idx, s, (s * s) * m.base_power_w)

    idx, s, pt = best['action']
    return PlanStep(idx, s, pt)


def plan_step(p_avail_w: float, modes_t500: list[emp.Mode],
              prev_mode_idx: int | None = None) -> PlanStep:
    """Greedy single-step planner for T = 500 C (not used by the closed loop).

    Picks the highest-flow mode whose base power fits in P_avail, with
    scale ≤ 1.0. Upswitch is limited to one flow level per control period and
    needs P_avail ≥ base power + FLOW_UP_MARGIN_W; downswitch is unrestricted.
    If P_avail is below the cheapest mode's base power, that mode runs at
    reduced scale and T_max falls below setpoint.
    """
    MAX_SCALE = 1.0

    # Sort by base power: flow=0 (cheapest) to flow=3.
    sorted_pairs = sorted(enumerate(modes_t500), key=lambda x: x[1].base_power_w)
    rank_of = {idx: rank for rank, (idx, _) in enumerate(sorted_pairs)}

    prev_rank = rank_of[prev_mode_idx] if prev_mode_idx is not None else 0

    chosen_idx, chosen_mode = sorted_pairs[0]

    for idx, m in sorted_pairs:
        rank = rank_of[idx]
        if rank > prev_rank:
            # Upswitch: one rank at a time, with headroom.
            if rank > prev_rank + 1:
                continue
            threshold = m.base_power_w + FLOW_UP_MARGIN_W
        else:
            threshold = m.base_power_w

        if threshold <= p_avail_w:
            chosen_idx, chosen_mode = idx, m

    # Scale to the available power, capped at MAX_SCALE.
    s = math.sqrt(max(p_avail_w / max(chosen_mode.base_power_w, 1e-9), 0.0))
    s = min(s, MAX_SCALE)

    return PlanStep(mode_idx=chosen_idx, scale=s,
                    p_target_w=(s * s) * chosen_mode.base_power_w)


# --------------------------------------------------------------------------- #
# ThermalStepper: grey-box model advanced one control period at a time for
# the closed-loop simulation.
# --------------------------------------------------------------------------- #
class ThermalStepper:
    """Single-step version of the thermal_greybox.simulate_experiment update."""

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

        # tau_power lag state
        if "tau_power" in params.values:
            self._tau_power = params.get("tau_power")
        else:
            self._tau_power = 0.0
        self._qgen_prev = None  # for power lag

        # Mutable state
        self.T = init_profile_c.copy()
        shell_init_frac = params.get("shell_init_frac")
        self.T_sh = (self.t_amb + shell_init_frac * (float(np.mean(self.T)) - self.t_amb)
                     if variant.use_shell else self.t_amb)
        self.T_rend = float(self.T[-1]) if variant.use_rend else self.t_amb
        self.T_lend = float(self.T[0]) if variant.use_lend else self.t_amb

    def _compute_qgen(self, currents_a: np.ndarray) -> np.ndarray:
        """Heat generation per node [W] for the given coil currents."""
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

    # ------ snapshot / restore of the mutable state ------
    def snapshot(self) -> dict:
        """Return a copy of the mutable thermal state."""
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
        """Restore mutable thermal state from a snapshot."""
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

        # Optional power lag
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

            # EM or empirical efficiency
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
# Closed-loop control trajectory with power ramp limit
# --------------------------------------------------------------------------- #
def build_control_trajectory(time_s: np.ndarray, p_avail_w: np.ndarray,
                             modes_t500: list[emp.Mode],
                             t_amb_eff: float = T_AMB_EFF_DEFAULT,
                             params=None, variant=None,
                             x_m=None, basis=None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (flow_slpm, currents (n_t, 6), target_P, actual_P).

    With the thermal-model args, runs closed loop: each step the MPC plans,
    power is ramp-limited, currents are scaled, the grey-box model steps, and
    its T_max is fed back to the next MPC call. Without them, T is propagated
    with _predict_T.
    """
    n_t = time_s.size
    target_P = np.zeros(n_t)
    flow_arr = np.zeros(n_t)
    base_I = np.zeros((n_t, 6))
    base_P_chosen = np.zeros(n_t)
    actual_P = np.zeros(n_t)

    closed_loop = all(v is not None for v in (params, variant, x_m, basis))

    stepper = None
    if closed_loop:
        init_profile = np.full(x_m.size, T_INIT_C)
        stepper = ThermalStepper(params, variant, x_m, basis, init_profile)

    prev_mode_idx: int | None = None
    T_est = T_INIT_C
    prev_actual_P = 0.0

    for k in range(n_t):
        # Feedback: T_max from the grey-box model
        if stepper is not None:
            T_est = stepper.T_max

        p_horizon = [float(p_avail_w[min(k + j, n_t - 1)])
                     for j in range(MPC_HORIZON)]
        plan = plan_step_mpc(p_horizon, modes_t500, prev_mode_idx,
                             T_est, t_amb_eff)
        m = modes_t500[plan.mode_idx]
        target_P[k] = plan.p_target_w

        # No thermal model: propagate with the first-order prediction
        if stepper is None:
            T_est = _predict_T(T_est, plan.scale, t_amb_eff, CONTROL_DT_S)

        flow_arr[k] = m.flow_slpm
        base_I[k] = m.currents_a
        base_P_chosen[k] = m.base_power_w
        prev_mode_idx = plan.mode_idx

        # Two-phase power ramp limit, clipped to [0, P_avail]
        if k == 0:
            actual_P[k] = target_P[k]
        else:
            rate = (emp.RAMP_FAST_W_PER_S
                    if prev_actual_P < emp.RAMP_FAST_FRAC * target_P[k]
                    else emp.RAMP_FINE_W_PER_S)
            dP_max = rate * SIM_DT_S
            dP = target_P[k] - prev_actual_P
            if abs(dP) <= dP_max:
                actual_P[k] = target_P[k]
            else:
                actual_P[k] = prev_actual_P + math.copysign(dP_max, dP)
            actual_P[k] = min(actual_P[k], float(p_avail_w[k]))
            actual_P[k] = max(actual_P[k], 0.0)
        prev_actual_P = actual_P[k]

        # Scale currents to the ramp-limited power, clip at I_MAX_A
        bp = max(base_P_chosen[k], 1e-9)
        s = math.sqrt(max(min(actual_P[k] / bp, 4.0), 0.0))
        currents_k = base_I[k] * s
        currents_k = np.minimum(currents_k, emp.I_MAX_A)

        if stepper is not None and k > 0:
            dt = float(time_s[k] - time_s[k - 1])
            stepper.step(currents_k, float(flow_arr[k]), dt)

    # Currents over the full trajectory from the ramp-limited power
    scale = np.where(base_P_chosen > 1e-6,
                     np.sqrt(np.clip(actual_P / np.maximum(base_P_chosen, 1e-9),
                                     0.0, 4.0)),
                     0.0)
    currents = base_I * scale[:, None]
    currents = np.minimum(currents, emp.I_MAX_A)
    return flow_arr, currents, target_P, actual_P


# --------------------------------------------------------------------------- #
# Live animation
# --------------------------------------------------------------------------- #
def animate(time_s: np.ndarray, temps_c: np.ndarray, currents: np.ndarray,
            flow_arr: np.ndarray, actual_P: np.ndarray, p_avail_w: np.ndarray,
            x_m: np.ndarray, save_gif: bool = False, show: bool = True,
            png_path: Path | None = None) -> None:
    n_t = time_s.size
    coil_x_mm = (tg.COIL_WINDING_START_M + np.arange(tg.N_COILS) *
                 tg.COIL_START_PITCH_M + 0.5 * tg.COIL_PACK_LENGTH_M) * 1000.0
    x_mm = x_m * 1000.0
    hours = time_s / 3600.0

    t_peak_hist = np.nanmax(temps_c, axis=1)

    stride = max(1, int(round(SIM_S_PER_FRAME / SIM_DT_S)))
    frames = list(range(0, n_t, stride))
    if frames[-1] != n_t - 1:
        frames.append(n_t - 1)

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.0],
                          height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.32)
    ax_T = fig.add_subplot(gs[0, 0])        # T profile
    ax_I = fig.add_subplot(gs[1, 0])        # 6 coil currents
    ax_F = fig.add_subplot(gs[2, 0])        # flow vs time
    ax_S = fig.add_subplot(gs[0, 1])        # solar P_avail vs time
    ax_H = fig.add_subplot(gs[1, 1])        # T_peak history (tracking)
    ax_P = fig.add_subplot(gs[2, 1])        # actual power vs time

    # Temperature panel
    ln_T, = ax_T.plot(x_mm, temps_c[0], "-o", color="#c0392b", ms=4, lw=2)
    ax_T.axhspan(emp.T_LOW_C, emp.T_HIGH_C, color="0.85", alpha=0.5)
    ax_T.axhline(T_SETPOINT_C, color="#16a085", lw=1.5, ls="--",
                 label=f"setpoint {T_SETPOINT_C:.0f} C")
    ax_T.set_xlabel("axial position [mm]")
    ax_T.set_ylabel("wall T [\N{degree sign}C]")
    ax_T.set_ylim(150, 610)
    ax_T.set_title("Wall temperature profile")
    ax_T.grid(alpha=0.3)
    ax_T.legend(loc="lower center", fontsize=8)
    txt_T = ax_T.text(0.02, 0.95, "", transform=ax_T.transAxes,
                      va="top", ha="left", fontsize=10,
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))

    # Currents panel
    bars = ax_I.bar(coil_x_mm, currents[0], width=12.0, color="#2980b9",
                    edgecolor="black", alpha=0.85)
    ax_I.set_xlabel("coil axial position [mm]")
    ax_I.set_ylabel("coil current [A]")
    ax_I.set_ylim(0.0, emp.I_MAX_A + 0.3)
    ax_I.axhline(emp.I_MAX_A, color="0.5", lw=0.8, ls="--")
    ax_I.set_title("Non-normalised coil currents (6 coils)")
    ax_I.grid(alpha=0.3, axis="y")
    txt_I = ax_I.text(0.98, 0.92, "", transform=ax_I.transAxes,
                      va="top", ha="right", fontsize=9, family="monospace",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))

    # Flow
    ax_F.plot(hours, flow_arr, color="#27ae60", lw=1.2, alpha=0.55)
    flow_marker, = ax_F.plot([hours[0]], [flow_arr[0]], "o", color="#1e7e34", ms=8)
    ax_F.set_xlabel("time [h]")
    ax_F.set_ylabel("flow [SLPM]")
    ax_F.set_ylim(-0.3, 3.3)
    ax_F.set_xlim(0, 24)
    ax_F.set_title("Flow setpoint")
    ax_F.grid(alpha=0.3)
    cursor_F = ax_F.axvline(hours[0], color="0.5", lw=0.7)

    # Solar P_avail
    ax_S.plot(hours, p_avail_w, color="#f39c12", lw=1.2, alpha=0.8,
              label="available")
    ax_S.axhline(P_FLOOR_W, color="0.5", lw=0.8, ls=":", label=f"floor {P_FLOOR_W:.0f} W")
    p_marker_s, = ax_S.plot([hours[0]], [p_avail_w[0]], "o",
                            color="#9a6101", ms=7)
    ax_S.set_xlabel("time [h]")
    ax_S.set_ylabel("solar P_avail [W]")
    ax_S.set_xlim(0, 24)
    ax_S.set_title("Cloudy-solar available power")
    ax_S.grid(alpha=0.3)
    ax_S.legend(loc="upper right", fontsize=8)

    # T_peak history (tracking quality)
    ax_H.plot(hours, t_peak_hist, color="#c0392b", lw=1.2, label="actual T_max")
    ax_H.axhline(T_SETPOINT_C, color="#16a085", lw=1.4, ls="--",
                 label=f"setpoint {T_SETPOINT_C:.0f} C")
    ax_H.fill_between(hours, emp.T_LOW_C, emp.T_HIGH_C, color="0.85", alpha=0.5)
    peak_marker, = ax_H.plot([hours[0]], [t_peak_hist[0]], "o",
                              color="#7b241c", ms=7)
    ax_H.set_xlabel("time [h]")
    ax_H.set_ylabel("T_max [\N{degree sign}C]")
    ax_H.set_xlim(0, 24)
    ax_H.set_ylim(450, 560)
    ax_H.set_title("Tracking quality: T_max vs setpoint")
    ax_H.legend(loc="lower right", fontsize=8)
    ax_H.grid(alpha=0.3)

    # Consumed vs available power
    ax_P.plot(hours, p_avail_w, color="#f39c12", lw=0.8, alpha=0.5,
              label="P_avail")
    ax_P.plot(hours, actual_P, color="#d35400", lw=1.0,
              label="P_consumed")
    power_marker, = ax_P.plot([hours[0]], [actual_P[0]], "o",
                              color="#a04000", ms=7)
    ax_P.set_xlabel("time [h]")
    ax_P.set_ylabel("power [W]")
    ax_P.set_xlim(0, 24)
    ax_P.set_title(f"Power: consumed vs available (ramp: {emp.RAMP_FAST_W_PER_S:.0f}/{emp.RAMP_FINE_W_PER_S:.0f} W/s)")
    ax_P.legend(loc="upper right", fontsize=8)
    ax_P.grid(alpha=0.3)

    fig.suptitle("Solar-tracking MPC digital twin: hold T = 500 \N{degree sign}C "
                 "under cloudy solar (v3_coil29 grey-box)", fontsize=13)

    def update(k):
        ln_T.set_ydata(temps_c[k])
        for bar, h in zip(bars, currents[k]):
            bar.set_height(max(float(h), 0.0))
        flow_marker.set_data([hours[k]], [flow_arr[k]])
        p_marker_s.set_data([hours[k]], [p_avail_w[k]])
        cursor_F.set_xdata([hours[k], hours[k]])
        peak_marker.set_data([hours[k]], [t_peak_hist[k]])
        power_marker.set_data([hours[k]], [actual_P[k]])
        t_peak_now = float(t_peak_hist[k])
        err = t_peak_now - T_SETPOINT_C
        txt_T.set_text(
            f"t       = {hours[k]:5.2f} h\n"
            f"T_max   = {t_peak_now:5.1f} \N{degree sign}C\n"
            f"err     = {err:+5.1f} \N{degree sign}C\n"
            f"flow    = {flow_arr[k]:.0f} SLPM\n"
            f"P_avail = {p_avail_w[k]:5.1f} W\n"
            f"P_used  = {actual_P[k]:5.1f} W\n"
            f"surplus = {p_avail_w[k]-actual_P[k]:+5.1f} W"
        )
        txt_I.set_text("  ".join(f"I{i+1}={currents[k,i]:4.2f}" for i in range(6)))
        return (ln_T, *bars, flow_marker, p_marker_s, cursor_F,
                peak_marker, power_marker, txt_T, txt_I)

    anim = FuncAnimation(fig, update, frames=frames,
                         interval=FRAME_INTERVAL_MS, blit=False, repeat=False)
    if save_gif:
        GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  writing {GIF_PATH} ...")
        anim.save(str(GIF_PATH), writer="pillow", fps=20)
        print(f"  saved {GIF_PATH}")
    if png_path is not None:
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
def _parse_start_time(s: str) -> float:
    """Parse --start-time ('HH:MM', 'HH:MM:SS' or fractional hour) to seconds
    since midnight, e.g. '14:30' -> 52200, '10.5' -> 37800.

    Row 0 of the solar profile is midnight, so result / SIM_DT_S is the start
    index.
    """
    s = s.strip()
    if ':' in s:
        parts = s.split(':')
        if len(parts) not in (2, 3):
            raise ValueError(f'expected HH:MM or HH:MM:SS, got {s!r}')
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
            raise ValueError(f'time out of range: {s!r}')
        return h * 3600 + m * 60 + sec
    hf = float(s)
    if not (0.0 <= hf < 24.0):
        raise ValueError(f'fractional hour out of [0, 24): {s!r}')
    return hf * 3600.0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Solar-tracking MPC digital twin (T=500 C) of the RWGS reactor.")
    ap.add_argument("--gif", action="store_true",
                    help="Write an animated GIF of the 24-h run.")
    ap.add_argument("--no-show", action="store_true",
                    help="Skip the live matplotlib window (still writes PNG).")
    ap.add_argument("--start-index", type=int, default=0,
                    help="First sim step (0-based) in the 24-h solar profile. "
                         "Ignored if --start-time is given.")
    ap.add_argument("--start-time", type=str, default=None,
                    help="Wall-clock start point inside the day.  Accepts "
                         "HH:MM (e.g. '10:30') or a fractional hour (e.g. "
                         "'10.5').  Overrides --start-index.")
    ap.add_argument("--duration", type=float, default=None,
                    help="Simulation duration in seconds (default: run to the "
                         "end of the 24-h profile from the start point).")
    args = ap.parse_args()

    print("Loading thermal model + parameters ...")
    variant_name, params = tg.load_params(emp.PARAMS_FILE)
    variant = tg.MODEL_VARIANTS[variant_name]
    x_m = tg.model_x_grid()
    basis = tg.coil_field_basis(x_m)

    print("Loading RWGS kinetic fit (for any CO bookkeeping) ...")
    import json
    rwgs = json.loads(emp.RWGS_FIT.read_text(encoding="utf-8"))
    fit = next(f for f in rwgs["fits"] if f["name"] == "profile_pfr")
    ln_a = float(fit["parameters"]["ln_Af_m3_per_mol_s"])
    ea_j = float(fit["parameters"]["Ea_kJ_per_mol"]) * 1000.0

    print("Building mode bank, keeping only T=500 C modes ...")
    all_modes = emp.load_modes(params, x_m, basis, ln_a, ea_j)
    modes_t500 = [m for m in all_modes if abs(m.target_c - 500.0) < 1e-6]
    # Tracking derate on currents (power and T rise scale with derate²).
    t_amb_eff = float(params.get("ambient_c"))
    for m in modes_t500:
        m.currents_a = m.currents_a * TRACKING_DERATE_T500
        m.base_T_profile_c = t_amb_eff + (TRACKING_DERATE_T500 ** 2) * (
            m.base_T_profile_c - t_amb_eff)
        m.base_power_w = m.base_power_w * (TRACKING_DERATE_T500 ** 2)
    print(f"{'mode':>20} {'P (W)':>9} {'CO g/s':>9} {'max(I) [A]':>11}")
    print("-" * 60)
    for m in modes_t500:
        print(f"{m.name:>20} {m.base_power_w:9.1f} "
              f"{m.co_kg_per_s*1000:9.3f} {float(m.currents_a.max()):11.2f}")

    # Full 24-h solar profile, sliced to the requested window below.
    print("\nGenerating cloudy-solar P_avail profile (24 h) ...")
    n_full = int(round(SIM_HORIZON_S / SIM_DT_S)) + 1
    time_full = np.arange(n_full) * SIM_DT_S
    p_avail_full = generate_cloudy_solar(time_full)

    # Resolve start index from --start-time or --start-index.
    start_idx = args.start_index
    if args.start_time is not None:
        try:
            tod_s = _parse_start_time(args.start_time)
        except ValueError as e:
            raise SystemExit(f'--start-time: {e}')
        start_idx = int(tod_s // SIM_DT_S) % n_full
        print(f"  --start-time {args.start_time} -> {tod_s/3600:.2f} h "
              f"-> start-index {start_idx}")
    if start_idx < 0 or start_idx >= n_full:
        raise SystemExit(f'--start-index {start_idx} out of range [0, {n_full-1}]')

    if args.duration is not None:
        n_run = min(int(round(args.duration / SIM_DT_S)) + 1, n_full - start_idx)
    else:
        n_run = n_full - start_idx

    time_s = time_full[start_idx : start_idx + n_run]
    p_avail = p_avail_full[start_idx : start_idx + n_run]
    n_t = time_s.size
    print(f"  Simulating from step {start_idx} ({time_s[0]/3600:.2f} h) "
          f"to step {start_idx + n_t - 1} ({time_s[-1]/3600:.2f} h)  "
          f"({n_t} steps, {(n_t-1)*SIM_DT_S/3600:.1f} h)")
    print(f"  P_avail range = [{p_avail.min():.1f}, {p_avail.max():.1f}] W, "
          f"mean = {p_avail.mean():.1f} W")

    print("Running closed-loop tracking MPC ...")
    t0 = _wallclock.time()
    flow_arr, currents, target_P, actual_P = build_control_trajectory(
        time_s, p_avail, modes_t500, t_amb_eff=t_amb_eff,
        params=params, variant=variant, x_m=x_m, basis=basis)
    from collections import Counter
    flows_used = Counter(np.round(flow_arr).astype(int))
    print(f"  flow usage (sim steps): "
          f"{dict(sorted(flows_used.items()))}")

    print("Rolling final thermal model forward ...")
    init_profile = np.full(x_m.size, T_INIT_C)
    temps = emp.run_thermal_sim(time_s, currents, flow_arr, params, variant,
                                x_m, basis, init_profile_c=init_profile)
    print(f"  done in {_wallclock.time()-t0:.1f} s")

    t_peak_hist = np.nanmax(temps, axis=1)
    tracking_rmse = float(np.sqrt(np.mean((t_peak_hist - T_SETPOINT_C) ** 2)))
    tracking_mae  = float(np.mean(np.abs(t_peak_hist - T_SETPOINT_C)))
    t_peak_max    = float(t_peak_hist.max())
    t_peak_min    = float(t_peak_hist.min())
    print(f"  T_max over the run = [{t_peak_min:.1f}, {t_peak_max:.1f}] C")
    print(f"  tracking |T_max - {T_SETPOINT_C:.0f}|: RMSE = {tracking_rmse:.2f} C, "
          f"MAE = {tracking_mae:.2f} C")
    p_surplus = (p_avail - actual_P).clip(min=0.0)
    print(f"  curtailed solar  = {p_surplus.sum() * SIM_DT_S / 3.6e6:.3f} kWh "
          f"(mean surplus {p_surplus.mean():.1f} W)")

    png_out = WORKSPACE / "plots" / "tracking_mpc_solar_T500_24h.png"
    print("\nLaunching live animation (close the window to exit) ...")
    animate(time_s, temps, currents, flow_arr, actual_P, p_avail, x_m,
            save_gif=args.gif, show=not args.no_show, png_path=png_out)


if __name__ == "__main__":
    main()
