# Economic-MPC hardware driver for the induction-heated RWGS reactor.
#
# Every CONTROL_DT_REAL_S the planner (empc_digital_twin.plan_step_empc)
# picks a flow (0-3 SLPM) and coil-current vector to maximise CO revenue minus
# electricity cost against the real-time LZ-South price, using the grey-box
# ThermalStepper and PFR kinetics over a short receding horizon. Flow changes
# are held for at least MIN_DWELL_STEPS. Between plans, an inner loop slews the
# SDG5162 Vpp (40 mV/s fast, 5 mV/s fine) so the measured ||I|| tracks the
# planned ||I||.
#
# Hardware:
#   - alicat: opened, set, closed on each flow setpoint change. A single N2
#     alicat stands in for the H2+CO2 feed.
#   - STM32F407 GPIO box: closed after each on_array change, reopened for
#     current reads during each control step.
#
# Inputs:
#   --schedule <xlsx>   5-min real-time price workbook
#                       (default ./LZ_South_Nov5_RealTime.xlsx)
#   --start-index <i>   first 5-min row of the price file (default 0); use to
#                       start mid-day or resume a run
#   --speed-factor <s>  time compression; s=2 plays 24 h of prices in 12 h
#   --duration <s>      simulated run length (default 24 h)
#
# Safety: any coil above COIL_MAX_A sets Vpp to 0 and stops the run. On exit,
# Vpp and flow go to 0 and the GPIO pins are turned off.

import argparse
import collections
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# --- locate sibling modules -----------------------------------------------------
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

# Digital-twin models: mode bank, flow configs, ThermalStepper, eMPC planner.
import empc_digital_twin as dtwin  # noqa: E402

# --- Defaults -------------------------------------------------------------------
DEFAULT_SCHEDULE_PATH = WORKSPACE / 'LZ_South_Nov5_RealTime.xlsx'
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

# The slew loop tracks ||I_meas|| against the planned ||I||. With one Vpp knob
# and a fixed on_array and frequency, all 6 coil currents scale together, so
# the norm is a good proxy for position along the planned current profile.
# Tolerance: max(50 mA, 1.5% of target).
CURRENT_TOL_BASE_A = 0.05
CURRENT_TOL_FRAC = 0.015

# Moving-average window (samples) on ||I_meas||.
CURRENT_AVG_SAMPLES = 4

# Safety interlocks
COIL_MAX_A = 5.2                       # per-coil limit (A); trip sets 0 Vpp and stops

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

# MPC clock. CONTROL_DT_REAL_S is the wall-clock time between plans.
# sim_time = speed_factor * real_time and indexes the 5-min price grid.
CONTROL_DT_REAL_S = 60.0
PRICE_PERIOD_S = 5 * 60.0           # one price row (sim s)
DEFAULT_RUN_DURATION_S = 24 * 3600.0  # one simulated day; real time = this / speed_factor


# ===========================================================================
# Pricing helpers
# ===========================================================================
def load_price_series(xlsx_path) -> np.ndarray:
    """Read the 5-min price column ($/MWh) from the LZ_South workbook."""
    return dtwin.load_realtime_price(Path(xlsx_path))


def price_at_sim_time(t_sim_s: float, prices_5min: np.ndarray,
                      start_index: int) -> tuple[float, int]:
    """Zero-order-hold lookup, wrapping at the end; returns (price, row index)."""
    rel = int(t_sim_s // PRICE_PERIOD_S)
    idx = (start_index + rel) % prices_5min.size
    return float(prices_5min[idx]), idx


# ===========================================================================
# Mode bank + EM power kernel
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


def build_mode_bank():
    """Return (modes, flow_configs, kernel, thermal_args).

    modes: 8 isothermal modes from empc_digital_twin (sets the power anchor).
    flow_configs: 4 per-flow configs used by the MPC planner.
    kernel: EMPowerKernel for measured currents.
    thermal_args: grey-box model inputs for the ThermalStepper and planner.
    """
    import json

    variant_name, params = dtwin.tg.load_params(dtwin.PARAMS_FILE)
    variant = dtwin.tg.MODEL_VARIANTS[variant_name]
    x_m = dtwin.tg.model_x_grid()
    basis = dtwin.tg.coil_field_basis(x_m)

    rwgs = json.loads(dtwin.RWGS_FIT.read_text(encoding='utf-8'))
    fit = next(f for f in rwgs['fits'] if f['name'] == 'profile_pfr')
    ln_a = float(fit['parameters']['ln_Af_m3_per_mol_s'])
    ea_j = float(fit['parameters']['Ea_kJ_per_mol']) * 1000.0

    modes = dtwin.load_modes(params, x_m, basis, ln_a, ea_j)
    flow_configs = dtwin.load_flow_configs(params, x_m, basis, ln_a, ea_j)

    # Same anchor scale as dtwin.load_modes(), applied to measured currents.
    anchor = next(m for m in modes if m.name == dtwin.POWER_ANCHOR_MODE)
    raw_anchor_w = dtwin.total_em_power_w(anchor.currents_a, params, x_m, basis)
    scale = dtwin.POWER_ANCHOR_W / max(raw_anchor_w, 1e-9)
    kernel = EMPowerKernel(params, x_m, basis, scale)
    thermal_args = dict(params=params, variant=variant, x_m=x_m, basis=basis,
                        ln_a=ln_a, ea_j=ea_j)
    return modes, flow_configs, kernel, thermal_args


def pick_best_mode(price_usd_per_mwh: float, modes) -> int:
    """Greedy single-step mode choice (max CO revenue - power cost). Not called by run_empc."""
    best_i, best_p = 0, -float('inf')
    for i, m in enumerate(modes):
        rev = (m.co_kg_per_s
               * dtwin.CO_PRICE_USD_PER_KG)             # $/s
        cost = m.base_power_w * price_usd_per_mwh / 3.6e9  # $/s
        p = rev - cost
        if p > best_p:
            best_p, best_i = p, i
    return best_i


# ===========================================================================
# Hardware helpers: alicat and STM32, opened and closed on each use
# ===========================================================================
def set_flow_setpoint(addr: str, gas: str, flow_slpm: float) -> None:
    """Open the alicat at addr, set gas and flow (SLPM), then close it."""
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
    """Tolerance band (A) on ||I_meas|| - ||I_target||."""
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
# Main MPC loop
# ===========================================================================
def run_empc(
    ip,
    sensors,
    com_port,
    schedule_path,
    out_path,
    start_index: int,
    speed_factor: float,
    run_duration_s: float,
    interval=DEFAULT_INTERVAL,
    live_plot=True,
    spacing_mm=DEFAULT_AXIAL_SPACING_MM,
    siggen=None,
    save_data=DEFAULT_SAVE_DATA,
    siggen_channel=DEFAULT_SIGGEN_CHANNEL,
    initial_vpp=DEFAULT_SIGGEN_VPP,
):
    print(f'Loading price file: {schedule_path}')
    prices = load_price_series(schedule_path)
    print(f'  {prices.size} 5-min bins; start row = {start_index}; '
          f'speed_factor = {speed_factor:g}; '
          f'real run length = {run_duration_s/speed_factor/3600:.2f} h '
          f'(simulated {run_duration_s/3600:.2f} h)')

    print('Building mode bank + EM power kernel ...')
    modes, flow_configs, kernel, thermal_args = build_mode_bank()
    print(f'  EM->deposited W anchor scale = {kernel.scale:.4g} '
          f'(anchored to {dtwin.POWER_ANCHOR_W:.1f} W at {dtwin.POWER_ANCHOR_MODE})')
    for fc in flow_configs:
        print(f'  flow={fc.flow_slpm:.0f} SLPM: '
              f'P=[{fc.power_at_min:.1f}, {fc.power_at_max:.1f}] W, '
              f'CO=[{fc.co_rate_at_min*1000:.3f}, {fc.co_rate_at_max*1000:.3f}] g/s, '
              f'||I||=[{fc.I_norm_min:.3f}, {fc.I_norm_max_eff:.3f}]')

    # Grey-box thermal model state for the planner, initialised at 450 C.
    ta = thermal_args
    init_profile = np.full(ta['x_m'].size, 450.0)
    stepper = dtwin.ThermalStepper(ta['params'], ta['variant'],
                                   ta['x_m'], ta['basis'], init_profile)
    print(f'  ThermalStepper initialised ({stepper.n} nodes, '
          f'MPC horizon = {dtwin.MPC_HORIZON} + {dtwin.MPC_ROLLOUT_STEPS} rollout steps)')

    print(f'Connecting to Hyperion at {ip} with sensors {sensors} ...')
    ctrl = setup_hyperion(ip, sensors)

    # Initial siggen state: sine at the flow-0 frequency, Vpp = initial_vpp.
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
    cur_flow_idx = None
    cur_flow_slpm = None
    prev_action = None
    dwell_count = dtwin.MIN_DWELL_STEPS  # dwell already met, so the first step plans freely
    # Rate-limited ||I|| setpoint (A); carried across MPC steps.
    ramped_target_I_norm = 0.0
    # Last price row used; printed on exit for --start-index.
    last_price_idx = start_index
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
            price, price_idx = price_at_sim_time(sim_time_s, prices, start_index)
            last_price_idx = price_idx
            last_sim_time_s = sim_time_s

            # Shift the model profile so its T_max matches the measured T_max.
            try:
                _sync_temps = read_temp(ctrl)
                if _sync_temps:
                    _t_max_meas = max(_sync_temps.values())
                    if not np.isnan(_t_max_meas):
                        _delta = _t_max_meas - stepper.T_max
                        stepper.T = stepper.T + _delta
            except Exception:
                pass  # on read failure, plan from the unsynced model

            # Price over the MPC horizon.
            _price_horizon = []
            for _j in range(dtwin.MPC_HORIZON):
                _future_sim = sim_time_s + _j * CONTROL_DT_REAL_S * speed_factor
                _ph, _ = price_at_sim_time(_future_sim, prices, start_index)
                _price_horizon.append(_ph)

            # Re-plan only once MIN_DWELL_STEPS have passed since the last flow
            # change; otherwise repeat the previous action.
            if dwell_count >= dtwin.MIN_DWELL_STEPS:
                action = dtwin.plan_step_empc(
                    stepper, flow_configs, _price_horizon, cur_flow_idx,
                    ta['x_m'], ta['ln_a'], ta['ea_j'],
                    control_dt_s=CONTROL_DT_REAL_S)
                if action.flow_idx != cur_flow_idx:
                    dwell_count = 0
            else:
                action = prev_action if prev_action is not None else dtwin.PlannerAction(
                    0, flow_configs[0].I_norm_min,
                    flow_configs[0].shape * flow_configs[0].I_norm_min,
                    flow_configs[0].power_at_min, flow_configs[0].flow_slpm)
            dwell_count += 1
            prev_action = action

            # Advance the model with the chosen action.
            stepper.step(action.currents_a, action.flow_slpm, CONTROL_DT_REAL_S)
            mode_flow = int(round(action.flow_slpm))
            # Planned 6-coil current vector (A); the slew loop tracks its norm.
            target_I_vec = np.asarray(action.currents_a, dtype=float)
            # Hot-standby boost at flow = 0.
            if mode_flow == 0:
                target_I_vec = target_I_vec + FLOW0_T450_CURRENT_BOOST_A
            target_I_norm = float(np.linalg.norm(target_I_vec))
            tol = current_tolerance_a(target_I_norm)
            target_w = float(action.power_w)  # diagnostic only

            print(f'\n=== MPC step {step_idx + 1}  '
                  f't_real={real_elapsed:7.1f}s  t_sim={sim_time_s:8.1f}s  '
                  f'price_row={price_idx}  ${price:.2f}/MWh -> '
                  f'flow={mode_flow} ||I||={action.I_norm:.3f} '
                  f'(target ||I||={target_I_norm:.3f} A '
                  f'[max coil {float(target_I_vec.max()):.2f}], '
                  f'P~{target_w:.1f} W) ===')

            flow_changing = (cur_flow_slpm != action.flow_slpm or cur_flow_idx is None)

            # Set Vpp to 0 before changing flow, GPIO array or frequency.
            if flow_changing:
                command_siggen_vpp_zero(siggen, siggen_channel)
                vpp = 0.0
                print('  Vpp -> 0 before mode change.')
                time.sleep(2.0)

            # Flow change (alicat open/set/close).
            if cur_flow_slpm != action.flow_slpm:
                try:
                    n2_sp = apply_total_flow_slpm(float(action.flow_slpm))
                    print(f'  flow: N2={n2_sp:.3f} SLPM (alicat closed).')
                except Exception as e:
                    print(f'  (flow change failed: {type(e).__name__}: {e})')
                cur_flow_slpm = action.flow_slpm

            # Re-apply on_array and siggen frequency on every flow change.
            new_array, new_freq_hz = ARRAY_BY_FLOW[mode_flow]
            if flow_changing:
                try:
                    apply_on_array(com_port, new_array)
                    print(f'  on_array set for flow={mode_flow} SLPM ({len(new_array)} GPIO); STM32 closed.')
                except Exception as e:
                    print(f'  (STM32 array change failed: {type(e).__name__}: {e})')
                if siggen is not None and (cur_freq_hz is None or abs(new_freq_hz - cur_freq_hz) > 1.0):
                    siggen.config_channel(siggen_channel, {'freq': new_freq_hz})
                    print(f'  siggen freq -> {new_freq_hz:.3e} Hz')
                    cur_freq_hz = new_freq_hz
                cur_flow_idx = action.flow_idx

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

                    # Per-coil deviation from the ramp-scaled target shape. All
                    # coils should scale together with Vpp, so a large deviation
                    # means the current distribution has drifted from the plan
                    # (capacitor aging, off-resonance). Diagnostic only.
                    if target_I_norm > 1e-9:
                        shape_scale = ramped_target_I_norm / target_I_norm
                    else:
                        shape_scale = 0.0
                    scaled_target_I = shape_scale * target_I_vec
                    per_coil_dev = currents_arr - scaled_target_I
                    max_abs_dev = float(np.max(np.abs(per_coil_dev)))

                    # Coil overcurrent trip
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
                        'flow_SLPM': float(action.flow_slpm),
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

                    # End the control step after CONTROL_DT_REAL_S of wall-clock time.
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
        # Print the last price row so the run can be resumed with --start-index.
        print(
            f'\n>>> Last price row used: {last_price_idx} '
            f'(sim_time = {last_sim_time_s:.1f} s = {last_sim_time_s/3600:.2f} h '
            f'into the {prices.size}-bin file).'
        )
        print(f'    To resume from here:  --start-index {last_price_idx}')

        # Shutdown: Vpp to 0, flow to 0, GPIO pins off.
        try:
            command_siggen_vpp_zero(siggen, siggen_channel)
        except Exception:
            pass
        try:
            set_flow_setpoint(FLOW_ADDR_N2, 'N2', 0.0)
        except Exception as e:
            print(f'  (zero-flow shutdown N2: {type(e).__name__}: {e})')
        try:
            apply_on_array(com_port, [])  # empty array: all pins off
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
        description='Economic-MPC driver: price-driven mode selection on the '
                    'induction-heated RWGS reactor.')
    p.add_argument('--schedule', default=str(DEFAULT_SCHEDULE_PATH),
                   help='Real-time price .xlsx (5-min bins).')
    p.add_argument('--start-index', type=int, default=0,
                   help='First row of the price file to use (0-based).')
    p.add_argument('--speed-factor', type=float, default=1.0,
                   help='Compress the price file in real time '
                        '(2 -> 24 h of price data plays in 12 h).')
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

    sensors = sensors_from_config(args.config)
    print(f'Using sensors: {sensors}')

    if args.out:
        out_path = args.out
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out_path = os.path.join(
            DEFAULT_OUT_DIR, f'{time.strftime("%Y%m%d-%H%M%S")}_empc.xlsx')

    siggen, pyvisa_rm = None, None
    try:
        siggen, pyvisa_rm = init_sdg5162(
            freq_hz=FREQ_0_SLPM,
            vpp=DEFAULT_SIGGEN_VPP,
            channel=DEFAULT_SIGGEN_CHANNEL,
            turn_on=DEFAULT_SIGGEN_TURN_ON,
            visa_timeout_ms=DEFAULT_SIGGEN_VISA_TIMEOUT_MS,
        )
        run_empc(
            DEFAULT_IP,
            sensors,
            STM32_COM_PORT,
            args.schedule,
            out_path,
            start_index=args.start_index,
            speed_factor=float(args.speed_factor),
            run_duration_s=float(args.duration),
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
