# Solar-tracking MPC hardware driver for the induction-heated RWGS reactor.
#
# Same instruments (SDG5162, STM32F407 GPIO box, N2 alicat, Hyperion) and the
# same inner ||I|| slew loop as supermode_eMPC.py. The planner differs: it
# holds susceptor T_max near T_SETPOINT_C (450 C) using no more than the
# available solar power P_avail(t).
#
# Planner (plan_step_mpc), every CONTROL_DT_REAL_S:
#   - Short receding-horizon search over the four T = 450 C modes, using a
#     first-order thermal prediction of T_max.
#   - At most one mode step up per control period, and only if P_avail covers
#     that mode's base power; any step down is allowed.
#   - Currents are scaled by sqrt(P_avail / P_mode), capped at MAX_SCALE.
#   - Cost: (T_pred - T_setpoint)^2 + switch penalty + curtailed solar power.
# During hold, target ||I|| is trimmed by temperature feedback on T_max.
#
# Inputs:
#   --solar-xlsx <path>   optional P_avail column (W), one row per
#                         --solar-period-s. If omitted, a 24-h cloudy-solar
#                         trace is generated with the digital twin's OU model.
#   --solar-period-s <s>  seconds per row (default 60)
#   --start-index <i>     first row of the P_avail series (default 0)
#   --start-time <t>      start time of day, HH:MM or fractional hour;
#                         overrides --start-index
#   --speed-factor <s>    time compression; s=2 plays 24 h in 12 h
#   --duration <s>        simulated run length (default 24 h)
#   --cloud-seed <int>    RNG seed for the synthetic cloud noise
#
# Safety: any coil above COIL_MAX_A sets Vpp to 0 and stops the run. On exit,
# Vpp and flow go to 0 and the GPIO pins are turned off.

import argparse
import collections
import os
import sys
import time
import math
import traceback
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, r'C:\Users\Jon Fan Lab\Desktop\reactor_test_code')

from instruments import alicat  # noqa: E402

from supermode_PRBS import (  # noqa: E402
    DEFAULT_AXIAL_SPACING_MM,
    DEFAULT_CONFIG,
    DEFAULT_INTERVAL,
    DEFAULT_IP,
    DEFAULT_OUT_DIR,
    DEFAULT_SIGGEN_CHANNEL,
    DEFAULT_SIGGEN_TURN_ON,
    DEFAULT_SIGGEN_VISA_TIMEOUT_MS,
    DEFAULT_SIGGEN_VPP,
    FREQ_0_SLPM,
    FREQ_1_SLPM,
    FREQ_2_SLPM,
    FREQ_3_SLPM,
    LiveDashboard,
    NUM_CURRENT_CHANNELS,
    STM32F407Client,
    init_sdg5162,
    on_array_0_SLPM,
    on_array_1_SLPM,
    on_array_2_SLPM,
    on_array_3_SLPM,
    _meas_temperatures as read_temp,
    release_siggen_visa,
    sensors_from_config,
    setup_hyperion,
    shutdown_sdg5162,
    _temp_sort_key,
)

# Digital-twin models: mode bank, EM power model, cloudy-solar generator.
import empc_digital_twin as dtwin  # noqa: E402
import tracking_mpc_solar_digital_twin as smpc  # noqa: E402

# --- Defaults -------------------------------------------------------------------
DEFAULT_SAVE_DATA = True

# Instrument ports. A single N2 alicat stands in for the H2+CO2 feed.
FLOW_COM_PORT = 'COM14'
FLOW_BAUD = 2400
FLOW_ADDR_N2 = 'A'
STM32_COM_PORT = 'COM15'

# GPIO on_array and siggen drive frequency for each flow (SLPM).
ARRAY_BY_FLOW = {
    0: (on_array_0_SLPM, FREQ_0_SLPM),
    1: (on_array_1_SLPM, FREQ_1_SLPM),
    2: (on_array_2_SLPM, FREQ_2_SLPM),
    3: (on_array_3_SLPM, FREQ_3_SLPM),
}

# Vpp slew rates: 40 mV/s in fast_ramp, 5 mV/s in fine_ramp and hold.
VPP_MIN = 0.0
VPP_MAX = 0.8
VPP_SLEW_FAST_V_PER_S = 0.040
VPP_SLEW_FINE_V_PER_S = 0.005

CURRENT_TOL_BASE_A = 0.05
CURRENT_TOL_FRAC = 0.015
CURRENT_AVG_SAMPLES = 4

COIL_MAX_A = 5.2

# Rate limit on the target ||I|| between modes.
# Fast: 0.8 A/s until coil 1 reaches 90% of its target.
# Fine: 0.2 A/s until coil 1 is within 0.05 A of its running average, then hold.
# Phase changes use measured coil 1 current, not ||I||.
RAMP_FAST_A_PER_S = 0.8
RAMP_FINE_A_PER_S = 0.2
COIL1_THRESHOLD_FRAC = 0.90    # fast to fine at coil1 >= 90% of target
COIL1_SETTLE_TOL_A   = 0.05   # fine to hold at |coil1 - avg| <= 0.05 A
COIL1_AVG_SAMPLES    = 4      # coil 1 running-average window (samples)

# Per-coil current boost (A) for the flow = 0 hot-standby mode.
# Keeps the reactor slightly hotter than the planned profile while idling.
FLOW0_T450_CURRENT_BOOST_A = 0.05

# Temperature feedback in hold: target ||I|| is scaled by
# 1 - GAIN * (T_max - T_setpoint), with the trim limited to ±MAX_FRAC.
TEMP_FB_GAIN_PER_C  = 0.005   # 0.5% of ||I|| per °C error
TEMP_FB_MAX_FRAC    = 0.05    # trim limit, ±5% of ||I||

# MPC clock. CONTROL_DT_REAL_S is the wall-clock time between plans.
# sim_time = speed_factor * real_time and indexes the P_avail series.
CONTROL_DT_REAL_S = 60.0
DEFAULT_SOLAR_PERIOD_S = 60.0          # s per row of the P_avail series
DEFAULT_RUN_DURATION_S = 24 * 3600.0

# Tracking setpoint (450 C), from tracking_mpc_solar_digital_twin.
T_SETPOINT_C = smpc.T_SETPOINT_C


# ===========================================================================
# Solar P_avail helpers
# ===========================================================================
def synth_solar_series(period_s: float = DEFAULT_SOLAR_PERIOD_S,
                       horizon_s: float = 24 * 3600.0,
                       cloud_seed: int = smpc.CLOUD_SEED) -> np.ndarray:
    """Synthetic cloudy-solar P_avail (W), sampled every period_s.

    Uses smpc.generate_cloudy_solar (OU cloud noise). The default seed,
    smpc.CLOUD_SEED, reproduces the digital-twin trace.
    """
    n = int(round(horizon_s / period_s)) + 1
    t = np.arange(n) * period_s
    return smpc.generate_cloudy_solar(
        t, floor_w=smpc.P_FLOOR_W, peak_w=smpc.P_PEAK_SOLAR_W,
        cloud_tau_s=smpc.CLOUD_TAU_S, cloud_sigma=smpc.CLOUD_SIGMA,
        seed=cloud_seed,
    )


def load_solar_xlsx(xlsx_path) -> np.ndarray:
    """Load a P_avail series (W) from an xlsx. Uses the first column whose
    header contains 'avail', 'solar' or 'p_w' (any case), else the second column."""
    if pd is None:
        raise RuntimeError('pandas required to load --solar-xlsx')
    df = pd.read_excel(xlsx_path)
    col = next((c for c in df.columns
                if any(k in str(c).lower() for k in ('avail', 'solar', 'p_w'))),
               None)
    if col is None:
        col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    arr = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(float)
    if arr.size == 0:
        raise RuntimeError(f'No numeric W values found in column {col!r} of {xlsx_path}')
    return arr


def solar_at_sim_time(t_sim_s: float, p_avail_w: np.ndarray,
                      start_index: int, period_s: float) -> tuple[float, int]:
    """Zero-order-hold lookup, wrapping at the end; returns (P_avail in W, row index)."""
    rel = int(t_sim_s // period_s)
    idx = (start_index + rel) % p_avail_w.size
    return float(p_avail_w[idx]), idx


def parse_start_time(s: str) -> float:
    """Parse --start-time ('HH:MM', 'HH:MM:SS' or fractional hour) to seconds
    since midnight, e.g. '14:30' = 52200, '10.5' = 37800. Row 0 of the P_avail
    series is assumed to be midnight.
    """
    s = s.strip()
    if ':' in s:
        parts = s.split(':')
        if len(parts) not in (2, 3):
            raise ValueError(f'expected HH:MM or HH:MM:SS, got {s!r}')
        h = int(parts[0])
        m = int(parts[1])
        sec = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
            raise ValueError(f'time out of range: {s!r}')
        return h * 3600 + m * 60 + sec
    # Fractional hour
    hf = float(s)
    if not (0.0 <= hf < 24.0):
        raise ValueError(f'fractional hour out of [0, 24): {s!r}')
    return hf * 3600.0


def index_for_time_of_day(time_of_day_s: float, period_s: float,
                          n_bins: int) -> int:
    """Seconds since midnight to row index (row 0 = midnight), modulo n_bins."""
    return int(time_of_day_s // period_s) % max(n_bins, 1)


# ===========================================================================
# Mode bank + EM power kernel (T=450 only, with the tracking derate)
# ===========================================================================
class EMPowerKernel:
    """Anchor-scaled EM power model for computing deposited power from currents.

    deposited_W(I) = scale * total_em_power_w(I, params, x_m, basis), with
    scale = POWER_ANCHOR_W / raw anchor power, as in dtwin.load_modes().
    """
    def __init__(self, params, x_m, basis, scale):
        self.params = params
        self.x_m = x_m
        self.basis = basis
        self.scale = float(scale)

    def deposited_power_w(self, currents_6) -> float:
        I = np.asarray(currents_6, dtype=float)
        return self.scale * dtwin.total_em_power_w(I, self.params, self.x_m, self.basis)


def build_t450_mode_bank() -> tuple[list, EMPowerKernel]:
    """Return (modes_t450, kernel): the four T = 450 C modes from the digital
    twin, with currents scaled by smpc.TRACKING_DERATE_T450 (power and
    temperature rise by its square)."""
    import json

    variant_name, params = dtwin.tg.load_params(dtwin.PARAMS_FILE)
    x_m = dtwin.tg.model_x_grid()
    basis = dtwin.tg.coil_field_basis(x_m)

    rwgs = json.loads(dtwin.RWGS_FIT.read_text(encoding='utf-8'))
    fit = next(f for f in rwgs['fits'] if f['name'] == 'profile_pfr')
    ln_a = float(fit['parameters']['ln_Af_m3_per_mol_s'])
    ea_j = float(fit['parameters']['Ea_kJ_per_mol']) * 1000.0

    all_modes = dtwin.load_modes(params, x_m, basis, ln_a, ea_j)
    modes_t450 = [m for m in all_modes if abs(m.target_c - 450.0) < 1e-6]

    # Tracking derate.
    der = smpc.TRACKING_DERATE_T450
    t_amb_eff = float(params.get('ambient_c'))
    for m in modes_t450:
        m.currents_a = m.currents_a * der
        m.base_T_profile_c = t_amb_eff + (der ** 2) * (m.base_T_profile_c - t_amb_eff)
        m.base_power_w = m.base_power_w * (der ** 2)

    # Same anchor scale as dtwin.load_modes(), applied to measured currents.
    anchor = next((m for m in all_modes
                   if m.name == dtwin.POWER_ANCHOR_MODE),
                  modes_t450[0])
    raw_anchor_w = dtwin.total_em_power_w(anchor.currents_a, params, x_m, basis)
    kernel_scale = dtwin.POWER_ANCHOR_W / max(raw_anchor_w, 1e-9)
    kernel = EMPowerKernel(params, x_m, basis, kernel_scale)
    return modes_t450, kernel


MAX_SCALE = 1.0              # cap on current scale (1.0 = mode design currents)
FLOW_UP_MARGIN_W = 0.0       # used only by plan_step_with_scale

# --- Receding-horizon MPC parameters ----------------------------------------
MPC_HORIZON        = 5       # look-ahead control periods
TAU_THERMAL_S      = 90.0    # simplified first-order thermal time constant [s]
SWITCH_PENALTY     = 5.0     # mode-switch cost [°C² per switch]
CURTAILMENT_WEIGHT = 0.01    # cost per W² of unused solar power per step
T_SS_BASE_C        = 450.0   # steady-state temp at scale = 1.0
T_AMB_EFF          = 200.0   # effective ambient from fitted thermal model


def _predict_T(T_cur, scale, dt_s):
    """First-order lag of T_max toward T_ss = T_amb + (T_ss_base - T_amb) * scale^2."""
    T_ss = T_AMB_EFF + (T_SS_BASE_C - T_AMB_EFF) * scale * scale
    alpha = 1.0 - math.exp(-dt_s / TAU_THERMAL_S)
    return T_cur + alpha * (T_ss - T_cur)


_prev_plan_mode_idx = None  # previous mode, limits upswitch to one rank


def plan_step_mpc(p_avail_horizon, modes_t450, T_current):
    """Receding-horizon mode and scale choice over the T = 450 C modes.

    Branch-and-bound search over MPC_HORIZON steps with modes ranked by base
    power. Each step may move up at most one rank, and only if P_avail covers
    that mode's base power. Stage cost is (T_pred - T_SETPOINT_C)^2, plus
    SWITCH_PENALTY per switch, plus CURTAILMENT_WEIGHT * (unused W)^2. Higher
    ranks are searched first since they curtail less and prune faster.

    Returns (mode_idx, scale, p_target_w) for the first step.
    """
    global _prev_plan_mode_idx

    sorted_pairs = sorted(enumerate(modes_t450),
                          key=lambda x: x[1].base_power_w)
    rank_of = {idx: rank for rank, (idx, _) in enumerate(sorted_pairs)}
    n_modes = len(sorted_pairs)

    cur_rank = rank_of[_prev_plan_mode_idx] if _prev_plan_mode_idx is not None else 0
    H = min(MPC_HORIZON, len(p_avail_horizon))
    dt = CONTROL_DT_REAL_S

    best = {'cost': float('inf'), 'action': None}

    def _search(step, rank, T, cost, first_action):
        if step >= H:
            if cost < best['cost']:
                best['cost'] = cost
                best['action'] = first_action
            return
        if cost >= best['cost']:
            return

        p_avail = p_avail_horizon[step]

        # Highest rank first.
        for next_rank in reversed(range(n_modes)):
            if next_rank > rank + 1:
                continue
            idx, m = sorted_pairs[next_rank]
            if next_rank > rank and p_avail < m.base_power_w:
                continue

            s = math.sqrt(max(p_avail / max(m.base_power_w, 1e-9), 0.0))
            s = min(s, MAX_SCALE)

            T_new = _predict_T(T, s, dt)

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
        _prev_plan_mode_idx = idx
        return idx, float(s), float((s * s) * m.base_power_w)

    idx, s, pt = best['action']
    _prev_plan_mode_idx = idx
    return idx, float(s), float(pt)


def plan_step_with_scale(p_avail_w: float, modes_t450) -> tuple[int, float, float]:
    """Greedy single-step planner (not called by run_solar_mpc).

    Picks the highest mode whose base power fits in P_avail, moving up at most
    one rank per call and only with FLOW_UP_MARGIN_W of headroom; moving down
    is unrestricted. Currents are scaled by sqrt(P_avail / P_base), capped at
    MAX_SCALE, so below the cheapest mode's power that mode runs at reduced
    scale and T drops below setpoint.
    """
    import math
    global _prev_plan_mode_idx

    sorted_pairs = sorted(enumerate(modes_t450), key=lambda x: x[1].base_power_w)
    rank_of = {idx: rank for rank, (idx, _) in enumerate(sorted_pairs)}

    prev_rank = rank_of[_prev_plan_mode_idx] if _prev_plan_mode_idx is not None else 0

    # Default to cheapest mode (flow=0).
    chosen_idx, chosen_mode = sorted_pairs[0]

    for idx, m in sorted_pairs:
        rank = rank_of[idx]
        if rank > prev_rank:
            # Upswitch: one rank at most, with headroom.
            if rank > prev_rank + 1:
                continue
            threshold = m.base_power_w + FLOW_UP_MARGIN_W
        else:
            # Same mode or downswitch: no margin.
            threshold = m.base_power_w

        if threshold <= p_avail_w:
            chosen_idx, chosen_mode = idx, m

    # Scale currents to use available power (capped at MAX_SCALE).
    s = math.sqrt(max(p_avail_w / max(chosen_mode.base_power_w, 1e-9), 0.0))
    s = min(s, MAX_SCALE)

    _prev_plan_mode_idx = chosen_idx
    return chosen_idx, float(s), float((s * s) * chosen_mode.base_power_w)


# ===========================================================================
# Hardware helpers: alicat and STM32, opened and closed on each use
# ===========================================================================
def set_flow_setpoint(addr: str, gas: str, flow_slpm: float) -> None:
    fc = alicat.FlowController(port=FLOW_COM_PORT, address=addr, baud=FLOW_BAUD)
    try:
        try:
            fc.set_gas(gas)
        except Exception as e:
            print(f'  (flow {addr}/{gas} set_gas: {type(e).__name__}: {e})')
        fc.set_flow_rate(float(flow_slpm))
    finally:
        try:
            fc.close()
        except Exception:
            pass


def apply_total_flow_slpm(total_slpm: float) -> float:
    """Set total SLPM on the single N2 alicat (open/set/close)."""
    set_flow_setpoint(FLOW_ADDR_N2, 'N2', total_slpm)
    return total_slpm


_current_on_set = None   # type: set[int] | None  (pins currently enabled)


def apply_on_array(com_port: str, on_array) -> None:
    """Switch GPIO pins by diff: disable pins no longer needed, enable new
    ones, 100 ms apart. The first call does disable_all and a full enable."""
    global _current_on_set
    new_set = set(on_array)
    dev = STM32F407Client(com_port)
    try:
        if _current_on_set is None:
            # First call: full reset.
            dev.disable_all_gpio()
            time.sleep(0.5)
            for pin in sorted(new_set):
                dev.enable_gpio([pin])
                time.sleep(0.1)
        else:
            to_disable = sorted(_current_on_set - new_set)
            to_enable  = sorted(new_set - _current_on_set)
            for pin in to_disable:
                dev.disable_gpio([pin])
                time.sleep(0.1)
            for pin in to_enable:
                dev.enable_gpio([pin])
                time.sleep(0.1)
        # Re-send the last column's pins for reliability.
        last_col = max(p // 4 for p in on_array)
        last_col_pins = [p for p in on_array if p // 4 == last_col]
        dev.enable_gpio(last_col_pins)
        time.sleep(0.1)
        dev.enable_gpio(last_col_pins)
        time.sleep(0.5)
        dev.enable_gpio(on_array)       # confirm all pins
        _current_on_set = new_set
    finally:
        try:
            dev.close()
        except Exception:
            pass


# ===========================================================================
# Vpp control helpers
# ===========================================================================
def command_siggen_vpp_zero(siggen, channel):
    if siggen is None:
        return
    siggen.config_channel(channel, {'Vpp': 0.0})


def current_tolerance_a(target_a: float) -> float:
    return max(CURRENT_TOL_BASE_A, CURRENT_TOL_FRAC * abs(target_a))


def compute_power(dev: STM32F407Client, kernel: EMPowerKernel):
    """Read coil currents from the STM32; return (currents, deposited power in W).

    Power uses the anchor-scaled digital-twin EM model (integral of k|B(I)|^2
    along the susceptor). The Vpp loop tracks ||I||, not this power.
    """
    _, volts = dev.read_currents()
    all_currents = [dev.adc_to_current(ch + 1, v) for ch, v in enumerate(volts)]
    currents = all_currents[:NUM_CURRENT_CHANNELS]
    power_w = kernel.deposited_power_w(currents)
    return currents, power_w


def apply_vpp_slew(vpp: float, meas: float, target: float,
                   dt_s: float, tol: float,
                   slew_rate: float = VPP_SLEW_FINE_V_PER_S) -> float:
    """Step Vpp to reduce (target - meas) at slew_rate (V/s); no change inside ±tol.

    meas and target are ||I|| (A) here. On overshoot (meas > target) the step
    is scaled up with the error so large overshoots are corrected quickly.
    """
    err = target - meas
    if abs(err) <= tol:
        return vpp
    step = slew_rate * dt_s
    if err < 0:
        # Overshoot: step x |err|/tol, clipped to 1-10x.
        overshoot_mult = min(max(abs(err) / max(tol, 1e-9), 1.0), 10.0)
        step *= overshoot_mult
    direction = 1.0 if err > 0 else -1.0
    return float(np.clip(vpp + direction * step, VPP_MIN, VPP_MAX))


# ===========================================================================
# Data plumbing
# ===========================================================================
def reorder_schedule_df(df):
    meta = [c for c in ('timestamp', 'flow_SLPM', 'mVpp') if c in df.columns]
    currents = [f'Current {i}' for i in range(1, NUM_CURRENT_CHANNELS + 1)
                if f'Current {i}' in df.columns]
    fixed = meta + currents
    rest = sorted((c for c in df.columns if c not in fixed),
                  key=_temp_sort_key, reverse=True)
    return df[fixed + rest]


def save_rows(rows, out_path):
    if not rows:
        print('No data collected.')
        return
    if pd is None:
        csv_path = os.path.splitext(out_path)[0] + '.csv'
        import csv
        keys = list(rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'pandas unavailable; wrote {len(rows)} rows to {csv_path}')
        return
    try:
        df = reorder_schedule_df(pd.DataFrame(rows))
        df.to_excel(out_path, index=False)
        print(f'Wrote {len(df)} rows to {out_path}')
    except Exception as e:
        print(f'Failed to write xlsx ({type(e).__name__}: {e}); falling back to CSV.')
        csv_path = os.path.splitext(out_path)[0] + '.csv'
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'Wrote {len(rows)} rows to {csv_path}')


# ===========================================================================
# Main solar-tracking MPC loop
# ===========================================================================
def run_solar_mpc(
    ip,
    sensors,
    com_port,
    out_path,
    start_index: int,
    speed_factor: float,
    run_duration_s: float,
    p_avail_series: np.ndarray,
    solar_period_s: float,
    interval=DEFAULT_INTERVAL,
    live_plot=True,
    spacing_mm=DEFAULT_AXIAL_SPACING_MM,
    siggen=None,
    save_data=DEFAULT_SAVE_DATA,
    siggen_channel=DEFAULT_SIGGEN_CHANNEL,
    initial_vpp=DEFAULT_SIGGEN_VPP,
):
    print(f'Solar series: {p_avail_series.size} bins of {solar_period_s:.1f} s '
          f'(= {p_avail_series.size * solar_period_s / 3600:.2f} h);  '
          f'P_avail range [{p_avail_series.min():.1f}, {p_avail_series.max():.1f}] W')
    print(f'  start_index = {start_index};  speed_factor = {speed_factor:g};  '
          f'real run length = {run_duration_s/speed_factor/3600:.2f} h '
          f'(simulated {run_duration_s/3600:.2f} h)')

    print('Building T=450 mode bank + EM kernel ...')
    modes, kernel = build_t450_mode_bank()
    print(f'  EM->deposited W anchor scale = {kernel.scale:.4g} '
          f'(anchored to {dtwin.POWER_ANCHOR_W:.1f} W at {dtwin.POWER_ANCHOR_MODE})')
    for m in modes:
        print(f'  {m.name:>14}: P~{m.base_power_w:6.1f} W, '
              f'max(I)={float(m.currents_a.max()):.2f} A')

    print(f'Connecting to Hyperion at {ip} with sensors {sensors} ...')
    ctrl = setup_hyperion(ip, sensors)

    vpp = float(np.clip(initial_vpp, VPP_MIN, VPP_MAX))
    cur_freq_hz = None
    if siggen is not None:
        siggen.config_channel(
            siggen_channel,
            {'typ': 'SINE', 'freq': FREQ_0_SLPM,
             'Vpp': vpp, 'Vdc': 0.0, 'phase': 0.0},
        )
        cur_freq_hz = FREQ_0_SLPM

    rows = []
    plot = None
    cur_mode_idx = None
    cur_flow_slpm = None
    ramped_target_I_norm = 0.0
    t_max_latest = T_SETPOINT_C   # latest measured T_max, passed to the planner
    last_solar_idx = start_index
    last_sim_time_s = 0.0
    run_t0 = time.perf_counter()
    real_run_duration_s = run_duration_s / max(speed_factor, 1e-9)

    try:
        step_idx = 0
        while True:
            real_elapsed = time.perf_counter() - run_t0
            if real_elapsed >= real_run_duration_s:
                print(f'\nReached configured run duration ({real_run_duration_s:.0f} s real).')
                break

            sim_time_s = real_elapsed * speed_factor
            p_avail, solar_idx = solar_at_sim_time(
                sim_time_s, p_avail_series, start_index, solar_period_s)
            last_solar_idx = solar_idx
            last_sim_time_s = sim_time_s

            # P_avail over the MPC horizon.
            sim_dt_per_step = CONTROL_DT_REAL_S * speed_factor
            p_horizon = []
            for _h in range(MPC_HORIZON):
                future_t = sim_time_s + _h * sim_dt_per_step
                p_h, _ = solar_at_sim_time(
                    future_t, p_avail_series, start_index, solar_period_s)
                p_horizon.append(p_h)
            best_i, plan_scale, p_target_w = plan_step_mpc(
                p_horizon, modes, t_max_latest)
            mode = modes[best_i]
            mode_flow = int(round(mode.flow_slpm))
            # Mode currents (A) times the plan scale; scale < 1 when P_avail is
            # below the mode's base power.
            target_I_vec = np.asarray(mode.currents_a, dtype=float) * plan_scale
            # Hot-standby boost at flow = 0.
            if mode_flow == 0 and abs(mode.target_c - 450.0) < 1e-6:
                target_I_vec = target_I_vec + FLOW0_T450_CURRENT_BOOST_A
            target_I_norm = float(np.linalg.norm(target_I_vec))
            tol = current_tolerance_a(target_I_norm)

            print(f'\n=== MPC step {step_idx + 1}  '
                  f't_real={real_elapsed:7.1f}s  t_sim={sim_time_s:8.1f}s  '
                  f'solar_row={solar_idx}  P_avail={p_avail:.1f} W -> '
                  f'mode={mode.name}  scale={plan_scale:.3f}  '
                  f'target ||I||={target_I_norm:.3f} A  P_target~{p_target_w:.1f} W ===')

            mode_changing = (cur_flow_slpm != mode.flow_slpm or cur_mode_idx != best_i)

            # Set Vpp to 0 before changing flow, GPIO array or frequency.
            if mode_changing:
                command_siggen_vpp_zero(siggen, siggen_channel)
                vpp = 0.0
                print('  Vpp -> 0 before mode change.')
                time.sleep(2.0)
            if cur_flow_slpm != mode.flow_slpm:
                try:
                    n2_sp = apply_total_flow_slpm(float(mode.flow_slpm))
                    print(f'  flow: N2={n2_sp:.3f} SLPM (alicat closed).')
                except Exception as e:
                    print(f'  (flow change failed: {type(e).__name__}: {e})')
                cur_flow_slpm = mode.flow_slpm

            # Re-apply on_array and siggen frequency on every mode change.
            new_array, new_freq_hz = ARRAY_BY_FLOW[mode_flow]
            if mode_changing:
                try:
                    apply_on_array(com_port, new_array)
                    print(f'  on_array set for flow={mode_flow} SLPM ({len(new_array)} GPIO); STM32 closed.')
                except Exception as e:
                    print(f'  (STM32 array change failed: {type(e).__name__}: {e})')
                if siggen is not None and (cur_freq_hz is None
                                           or abs(new_freq_hz - cur_freq_hz) > 1.0):
                    siggen.config_channel(siggen_channel, {'freq': new_freq_hz})
                    print(f'  siggen freq -> {new_freq_hz:.3e} Hz')
                    cur_freq_hz = new_freq_hz
                cur_mode_idx = best_i

            # Inner ramp/hold loop for one control step. The STM32 stays open
            # for current reads and is closed at the end of the step.
            dev = STM32F407Client(com_port)
            I_norm_hist = collections.deque(maxlen=CURRENT_AVG_SAMPLES)
            coil1_hist = collections.deque(maxlen=COIL1_AVG_SAMPLES)
            target_coil1 = float(target_I_vec[0])
            last_loop_wall = time.perf_counter()
            step_wall_t0 = time.perf_counter()
            phase = 'fast_ramp'
            trip_reason = None

            try:
                while True:
                    loop_iter_t0 = time.perf_counter()
                    real_elapsed = loop_iter_t0 - run_t0
                    sim_time_s = real_elapsed * speed_factor
                    dt_loop = max(loop_iter_t0 - last_loop_wall, 1e-6)
                    last_loop_wall = loop_iter_t0

                    # Rate-limit the ||I|| setpoint (fast or fine by phase).
                    ramp_rate = (RAMP_FAST_A_PER_S
                                 if phase == 'fast_ramp'
                                 else RAMP_FINE_A_PER_S)
                    dI_max = ramp_rate * dt_loop
                    dI = target_I_norm - ramped_target_I_norm
                    if abs(dI) <= dI_max:
                        ramped_target_I_norm = target_I_norm
                    else:
                        ramped_target_I_norm += float(np.copysign(dI_max, dI))

                    currents, power_w = compute_power(dev, kernel)
                    temps = read_temp(ctrl)

                    currents_arr = np.asarray(currents, dtype=float)
                    I_norm_meas = float(np.linalg.norm(currents_arr))
                    I_norm_hist.append(I_norm_meas)
                    I_norm_avg = sum(I_norm_hist) / len(I_norm_hist)

                    # Coil 1 running average, used for phase changes.
                    coil1 = float(currents_arr[0])
                    coil1_hist.append(coil1)
                    coil1_avg = sum(coil1_hist) / len(coil1_hist)

                    if target_I_norm > 1e-9:
                        shape_scale = ramped_target_I_norm / target_I_norm
                    else:
                        shape_scale = 0.0
                    scaled_target_I = shape_scale * target_I_vec
                    per_coil_dev = currents_arr - scaled_target_I
                    max_abs_dev = float(np.max(np.abs(per_coil_dev)))

                    t_max = max((float(v) for v in temps.values()),
                                default=float('nan'))
                    if not np.isnan(t_max):
                        t_max_latest = t_max
                    t_err = (t_max - T_SETPOINT_C
                             if not np.isnan(t_max) else float('nan'))

                    # Temperature feedback: trim target ||I|| during hold.
                    if phase == 'hold' and not np.isnan(t_err):
                        trim_frac = np.clip(
                            -t_err * TEMP_FB_GAIN_PER_C,
                            -TEMP_FB_MAX_FRAC, TEMP_FB_MAX_FRAC)
                        ramped_target_I_norm = target_I_norm * (1.0 + trim_frac)

                    over_i = next((i for i, I in enumerate(currents) if I > COIL_MAX_A), None)
                    if over_i is not None:
                        command_siggen_vpp_zero(siggen, siggen_channel)
                        vpp = 0.0
                        trip_reason = (
                            f'Coil {over_i + 1} overcurrent: {currents[over_i]:.3f} A > '
                            f'{COIL_MAX_A} A (siggen 0 Vpp, program stop)'
                        )

                    if trip_reason is None:
                        # Vpp slew: fast in fast_ramp, fine otherwise.
                        vpp_slew = (VPP_SLEW_FAST_V_PER_S
                                    if phase == 'fast_ramp'
                                    else VPP_SLEW_FINE_V_PER_S)
                        vpp_new = apply_vpp_slew(
                            vpp, I_norm_avg, ramped_target_I_norm, dt_loop, tol,
                            slew_rate=vpp_slew)
                        if siggen is not None and abs(vpp_new - vpp) > 1e-12:
                            siggen.config_channel(siggen_channel, {'Vpp': float(vpp_new)})
                        vpp = vpp_new

                        # Phase changes on measured coil 1 current.
                        if (phase == 'fast_ramp'
                                and coil1 >= COIL1_THRESHOLD_FRAC * target_coil1):
                            phase = 'fine_ramp'
                            print(f'  -> coil1={coil1:.3f} A >= '
                                  f'{COIL1_THRESHOLD_FRAC:.0%} of '
                                  f'{target_coil1:.3f} A; switching to '
                                  f'fine ramp')
                        elif (phase == 'fine_ramp'
                                and len(coil1_hist) >= COIL1_AVG_SAMPLES
                                and abs(coil1 - coil1_avg)
                                    <= COIL1_SETTLE_TOL_A):
                            phase = 'hold'
                            print(f'  -> coil1 settled: {coil1:.3f} A '
                                  f'within {COIL1_SETTLE_TOL_A} A of '
                                  f'avg {coil1_avg:.3f} A; holding')

                    # --- Log row: timestamp, flow, mVpp, currents, temps ---
                    row = {
                        'timestamp': round(real_elapsed, 3),
                        'flow_SLPM': float(mode.flow_slpm),
                        'mVpp': round(vpp * 1000.0, 1),
                    }
                    row.update({f'Current {i + 1}': c for i, c in enumerate(currents)})
                    row.update(temps)
                    rows.append(row)

                    # --- Console print ---
                    sorted_temp_items = sorted(
                        temps.items(), key=lambda kv: _temp_sort_key(kv[0]), reverse=True)
                    cur_str = '  '.join(f'I{i+1}={c:.3f}' for i, c in enumerate(currents))
                    tmp_str = '  '.join(f'{k}={v:.1f}' for k, v in sorted_temp_items)
                    print(f't={real_elapsed:7.1f}s  step={step_idx + 1}  {phase:9s}  '
                          f'flow={mode_flow}  mVpp={vpp*1000:.1f}  '
                          f'{cur_str}  {tmp_str}')

                    if trip_reason is not None:
                        print(f'\n*** TRIP: {trip_reason}')
                        raise RuntimeError(trip_reason)

                    if live_plot:
                        if plot is None:
                            try:
                                plot = LiveDashboard(temps, currents, spacing_mm=spacing_mm)
                            except Exception as e:
                                print(f'  (live plot disabled: {type(e).__name__}: {e})')
                                live_plot = False
                        else:
                            plot.update(temps, currents, elapsed=real_elapsed)

                    if (time.perf_counter() - step_wall_t0) >= CONTROL_DT_REAL_S:
                        break

                    sleep_for = interval - (time.perf_counter() - loop_iter_t0)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            finally:
                try:
                    dev.close()
                except Exception:
                    pass

            step_idx += 1

    except KeyboardInterrupt:
        print('\nCtrl+C received; saving what we have...')
    except Exception as e:
        print(f'\nError in MPC loop ({type(e).__name__}: {e}); saving what we have...')
        traceback.print_exc()
    finally:
        print(
            f'\n>>> Last solar row used: {last_solar_idx} '
            f'(sim_time = {last_sim_time_s:.1f} s = '
            f'{last_sim_time_s/3600:.2f} h into the {p_avail_series.size}-bin series).'
        )
        print(f'    To resume from here:  --start-index {last_solar_idx}')

        try:
            command_siggen_vpp_zero(siggen, siggen_channel)
        except Exception:
            pass
        try:
            set_flow_setpoint(FLOW_ADDR_N2, 'N2', 0.0)
        except Exception as e:
            print(f'  (zero-flow shutdown N2: {type(e).__name__}: {e})')
        try:
            apply_on_array(com_port, [])
        except Exception:
            pass
        if plot is not None:
            plot.close()

    if save_data:
        save_rows(rows, out_path)
    else:
        print(f'save_data=False; collected {len(rows)} rows, skipping write.')


def main():
    p = argparse.ArgumentParser(
        description='Solar-tracking MPC driver: hold T_max at 450 C under a '
                    'real or synthesised cloudy-solar power budget.')
    p.add_argument('--solar-xlsx', default=None,
                   help='Optional xlsx with one column of P_avail values (W). '
                        'If omitted, a 24-h cloudy-solar trace is synthesised '
                        'with the same OU-noise model the digital twin uses.')
    p.add_argument('--solar-period-s', type=float, default=DEFAULT_SOLAR_PERIOD_S,
                   help='Seconds per row of the P_avail series (default 60).')
    p.add_argument('--cloud-seed', type=int, default=smpc.CLOUD_SEED,
                   help='RNG seed used when synthesising the P_avail series.')
    p.add_argument('--start-index', type=int, default=0,
                   help='First row of the P_avail series to use (0-based). '
                        'Ignored if --start-time is given.')
    p.add_argument('--start-time', type=str, default=None,
                   help='Wall-clock start point inside the day, used to '
                        'compute --start-index from the solar period. Accepts '
                        'either HH:MM (e.g. "10:30") or a fractional hour '
                        '(e.g. "10.5").  Row 0 of the P_avail series is '
                        'assumed to be midnight.')
    p.add_argument('--speed-factor', type=float, default=1.0,
                   help='Compress the trace in real time (2 -> 24 h of solar '
                        'plays out in 12 h of wall-clock).')
    p.add_argument('--duration', type=float, default=DEFAULT_RUN_DURATION_S,
                   help='Total *simulated* run length in seconds. '
                        'Real wall-clock = duration / speed_factor.')
    p.add_argument('--config', default=DEFAULT_CONFIG,
                   help='monitor_supermode.json (Hyperion sensor list).')
    p.add_argument('--out', default=None,
                   help='Output xlsx (default: timestamped in DEFAULT_OUT_DIR).')
    p.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                   help='Inner sample period (s).')
    p.add_argument('--no-plot', action='store_true', help='Disable live dashboard.')
    p.add_argument('--no-save', action='store_true', help='Skip writing output.')
    args = p.parse_args()

    if args.speed_factor <= 0:
        raise SystemExit('--speed-factor must be > 0')
    if args.start_index < 0:
        raise SystemExit('--start-index must be >= 0')
    if args.solar_period_s <= 0:
        raise SystemExit('--solar-period-s must be > 0')

    sensors = sensors_from_config(args.config)
    print(f'Using sensors: {sensors}')

    if args.solar_xlsx:
        p_avail = load_solar_xlsx(args.solar_xlsx)
        print(f'Loaded {p_avail.size} solar rows from {args.solar_xlsx}')
        solar_period_s = float(args.solar_period_s)
    else:
        p_avail = synth_solar_series(
            period_s=float(args.solar_period_s),
            cloud_seed=int(args.cloud_seed),
        )
        print(f'Synthesised {p_avail.size} solar rows '
              f'(seed={args.cloud_seed}, floor={smpc.P_FLOOR_W} W, '
              f'peak+={smpc.P_PEAK_SOLAR_W} W)')
        solar_period_s = float(args.solar_period_s)

    # --start-time overrides --start-index (row 0 = midnight).
    start_index = args.start_index
    if args.start_time is not None:
        try:
            tod_s = parse_start_time(args.start_time)
        except ValueError as e:
            raise SystemExit(f'--start-time: {e}')
        start_index = index_for_time_of_day(tod_s, solar_period_s, p_avail.size)
        print(f'--start-time {args.start_time} -> {tod_s/3600:.3f} h of day '
              f'-> --start-index {start_index} '
              f'(period {solar_period_s:.0f} s, {p_avail.size}-bin series).')

    if args.out:
        out_path = args.out
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out_path = os.path.join(
            DEFAULT_OUT_DIR, f'{time.strftime("%Y%m%d-%H%M%S")}_solar_mpc.xlsx')

    siggen, pyvisa_rm = None, None
    try:
        siggen, pyvisa_rm = init_sdg5162(
            freq_hz=FREQ_0_SLPM,
            vpp=DEFAULT_SIGGEN_VPP,
            channel=DEFAULT_SIGGEN_CHANNEL,
            turn_on=DEFAULT_SIGGEN_TURN_ON,
            visa_timeout_ms=DEFAULT_SIGGEN_VISA_TIMEOUT_MS,
        )
        run_solar_mpc(
            DEFAULT_IP,
            sensors,
            STM32_COM_PORT,
            out_path,
            start_index=start_index,
            speed_factor=float(args.speed_factor),
            run_duration_s=float(args.duration),
            p_avail_series=p_avail,
            solar_period_s=solar_period_s,
            interval=args.interval,
            live_plot=not args.no_plot,
            spacing_mm=DEFAULT_AXIAL_SPACING_MM,
            siggen=siggen,
            save_data=not args.no_save,
            siggen_channel=DEFAULT_SIGGEN_CHANNEL,
            initial_vpp=DEFAULT_SIGGEN_VPP,
        )
    finally:
        shutdown_sdg5162(siggen, channel=DEFAULT_SIGGEN_CHANNEL)
        release_siggen_visa(siggen, pyvisa_rm)


if __name__ == '__main__':
    main()
