# Temperature and coil-current logging with the SDG5162 driven by an Excel
# power schedule (PRBS / step tests).
#
# Recipe (first sheet): column A = power (W), column B = mode.
#   SS: ramp to the absolute power in column A, then hold until steady state
#       (max sensor temperature within 0.02 °C over the last 5 samples) or
#       15 min.
#   W:  step the setpoint by column A relative to the previous setpoint, ramp,
#       then hold 30 min while correcting power drift.
#
# Ramp and drift correction: siggen Vpp slews at 5 mV/s on a rolling average of
# coil I^2*R power (POWER_CONTROL_AVG_SAMPLES) until within tolerance.
#
# Inputs: --schedule <xlsx> recipe, --config sensor list, --out output xlsx,
# --interval sample period (s).
#
# Safety: any coil above COIL_MAX_A zeroes Vpp and aborts. During hold, Vpp is
# zeroed and the run aborts if running-average power drops below
# UNDERPOWER_TRIP_FRAC of target.

import argparse
import collections
import os
import sys
import time
import traceback

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from openpyxl import load_workbook

sys.path.insert(0, r'C:\Users\Jon Fan Lab\Desktop\reactor_test_code')

from hyperion_temp_logger_siggen import (  # noqa: E402
    COIL_RESISTANCES_OHMS,
    DEFAULT_AXIAL_SPACING_MM,
    DEFAULT_COM_PORT,
    DEFAULT_CONFIG,
    DEFAULT_INTERVAL,
    DEFAULT_IP,
    DEFAULT_ON_ARRAY,
    DEFAULT_OUT_DIR,
    DEFAULT_SIGGEN_CHANNEL,
    DEFAULT_SIGGEN_FREQ_HZ,
    DEFAULT_SIGGEN_TURN_ON,
    DEFAULT_SIGGEN_VISA_TIMEOUT_MS,
    DEFAULT_SIGGEN_VPP,
    LiveDashboard,
    NUM_CURRENT_CHANNELS,
    STM32F407Client,
    init_sdg5162,
    read_temp,
    release_siggen_visa,
    sensors_from_config,
    setup_hyperion,
    shutdown_sdg5162,
    _temp_sort_key,
)

# --- Schedule / control defaults ------------------------------------------------
DEFAULT_SCHEDULE_PATH = r'c:\Users\linca\Downloads\prbstest.xlsx'
DEFAULT_SAVE_DATA = True

VPP_MIN = 0.0
VPP_MAX = 10.0          # safety clamp on commanded Vpp
VPP_SLEW_V_PER_S = 0.005  # 5 mV/s

POWER_TOL_BASE_W = 2.0  # tolerance floor (W)
POWER_TOL_FRAC = 0.015  # fractional tolerance; band = max(floor, frac * target)

SS_TIMEOUT_S = 15 * 60
W_HOLD_S = 30 * 60
RAMP_TIMEOUT_S = 60 * 60  # abort if the ramp never reaches the band (e.g. Vpp clamp)

STEADY_TEMP_BAND_C = 0.02
STEADY_SAMPLES = 5

# Vpp control uses mean power over the last N samples of the recipe step.
POWER_CONTROL_AVG_SAMPLES = 8

# Safety interlocks
COIL_MAX_A = 5.2  # per-coil limit (A); zero siggen and abort if exceeded
UNDERPOWER_TRIP_FRAC = 0.40  # hold: trip if running-avg power < 40% of target


def command_siggen_vpp_zero(siggen, channel):
    if siggen is None:
        return
    siggen.config_channel(channel, {'Vpp': 0.0})


def power_tolerance_w(target_w: float) -> float:
    return max(POWER_TOL_BASE_W, POWER_TOL_FRAC * abs(target_w))


def load_recipe(path: str):
    """Return list of (numeric_value, mode) with mode in {'SS','W'}."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    steps = []
    for row in ws.iter_rows(values_only=True):
        if not row:
            continue
        a = row[0]
        b = row[1] if len(row) > 1 else None
        if a is None:
            continue
        if isinstance(a, str) and str(a).strip().lower() == 'power':
            continue
        try:
            val = float(a)
        except (TypeError, ValueError):
            print(f'  (skip row: cannot parse power {a!r})')
            continue
        if b is None:
            print(f'  (skip row: missing mode for power {val})')
            continue
        mode = str(b).strip().upper()
        if mode not in ('SS', 'W'):
            print(f'  (skip row: unknown mode {b!r} for power {val})')
            continue
        steps.append((val, mode))
    wb.close()
    return steps


def compute_power(dev: STM32F407Client, resistances):
    _, volts = dev.read_currents()
    all_currents = [dev.adc_to_current(ch + 1, v) for ch, v in enumerate(volts)]
    currents = all_currents[:NUM_CURRENT_CHANNELS]
    power_w = sum(I * I * R for I, R in zip(currents, resistances))
    return currents, power_w


def global_max_temp(temps: dict) -> float:
    return max(float(v) for v in temps.values())


def steady_state_ok(max_temp_history) -> bool:
    if len(max_temp_history) < STEADY_SAMPLES:
        return False
    lo = min(max_temp_history)
    hi = max(max_temp_history)
    return (hi - lo) <= STEADY_TEMP_BAND_C


def apply_vpp_slew(vpp: float, power_meas: float, power_target: float, dt_s: float) -> float:
    """Step Vpp toward power_target at VPP_SLEW_V_PER_S; no change inside the
    tolerance band. power_meas is normally the moving-average power."""
    err = power_target - power_meas
    tol = power_tolerance_w(power_target)
    if abs(err) <= tol:
        return vpp
    step = VPP_SLEW_V_PER_S * dt_s
    direction = 1.0 if err > 0 else -1.0
    return float(np.clip(vpp + direction * step, VPP_MIN, VPP_MAX))


def reorder_schedule_df(df):
    """Column order: timestamp and recipe metadata, currents, Vpp/power, temperatures."""
    meta = [
        c
        for c in (
            'timestamp',
            'recipe_step',
            'recipe_mode',
            'phase',
            'target_power_W',
            'fault',
        )
        if c in df.columns
    ]
    currents = [
        f'Current {i}'
        for i in range(1, NUM_CURRENT_CHANNELS + 1)
        if f'Current {i}' in df.columns
    ]
    extras = [c for c in ('Vpp', 'Power (W)', 'Power_running_avg (W)') if c in df.columns]
    fixed = meta + currents + extras
    rest = sorted(
        (c for c in df.columns if c not in fixed),
        key=_temp_sort_key,
        reverse=True,
    )
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


def run_excel_schedule(
    ip,
    sensors,
    com_port,
    on_array,
    schedule_path,
    out_path,
    interval=DEFAULT_INTERVAL,
    live_plot=True,
    spacing_mm=DEFAULT_AXIAL_SPACING_MM,
    siggen=None,
    save_data=DEFAULT_SAVE_DATA,
    siggen_channel=DEFAULT_SIGGEN_CHANNEL,
    resistances=COIL_RESISTANCES_OHMS,
    initial_vpp=DEFAULT_SIGGEN_VPP,
):
    recipe = load_recipe(schedule_path)
    if not recipe:
        raise RuntimeError(f'No valid schedule rows in {schedule_path}')
    print(f'Loaded {len(recipe)} steps from {schedule_path}')

    print(f'Opening STM32 on {com_port}...')
    dev = STM32F407Client(com_port)
    dev.disable_all_gpio()
    dev.enable_gpio(on_array)
    dev.enable_gpio(on_array)

    print(f'Connecting to Hyperion at {ip} with sensors {sensors}...')
    ctrl = setup_hyperion(ip, sensors)

    rows = []
    run_t0 = time.perf_counter()
    plot = None
    vpp = float(np.clip(initial_vpp, VPP_MIN, VPP_MAX))

    if siggen is not None:
        siggen.config_channel(
            siggen_channel,
            {
                'typ': 'SINE',
                'freq': DEFAULT_SIGGEN_FREQ_HZ,
                'Vpp': vpp,
                'Vdc': 0.0,
                'phase': 0.0,
            },
        )

    setpoint_w = None  # previous step's power target (W)

    try:
        for step_idx, (pv, mode) in enumerate(recipe):
            last_loop_wall = time.perf_counter()
            if mode == 'SS':
                target_w = float(pv)
            else:
                if setpoint_w is None:
                    raise RuntimeError(
                        'Schedule has W before any SS setpoint; start with an SS row '
                        'or edit the recipe.'
                    )
                target_w = float(setpoint_w + pv)

            tol = power_tolerance_w(target_w)
            print(f'\n=== Step {step_idx + 1}/{len(recipe)}  mode={mode}  target={target_w:.3f} W '
                  f'(tol ±{tol:.3f} W) ===')

            phase = 'ramp'
            hold_t0 = None
            step_wall_t0 = time.perf_counter()
            max_temp_hist = collections.deque(maxlen=STEADY_SAMPLES)
            power_ctrl_hist = collections.deque(maxlen=POWER_CONTROL_AVG_SAMPLES)

            while True:
                loop_iter_t0 = time.perf_counter()
                elapsed_run = loop_iter_t0 - run_t0
                dt_loop = max(loop_iter_t0 - last_loop_wall, 1e-6)
                last_loop_wall = loop_iter_t0

                currents, power_w = compute_power(dev, resistances)
                temps = read_temp(ctrl)

                power_ctrl_hist.append(power_w)
                power_ctrl_avg = sum(power_ctrl_hist) / len(power_ctrl_hist)

                trip_reason = None
                over_i = next((i for i, I in enumerate(currents) if I > COIL_MAX_A), None)
                if over_i is not None:
                    command_siggen_vpp_zero(siggen, siggen_channel)
                    vpp = 0.0
                    trip_reason = (
                        f'Coil {over_i + 1} overcurrent: {currents[over_i]:.3f} A > {COIL_MAX_A} A '
                        '(siggen 0 Vpp, program stop)'
                    )
                elif (
                    phase == 'hold'
                    and target_w > 1e-6
                    and len(power_ctrl_hist) >= POWER_CONTROL_AVG_SAMPLES
                    and power_ctrl_avg < UNDERPOWER_TRIP_FRAC * target_w
                ):
                    command_siggen_vpp_zero(siggen, siggen_channel)
                    vpp = 0.0
                    trip_reason = (
                        f'Under-power interlock: running-avg {power_ctrl_avg:.3f} W < '
                        f'{UNDERPOWER_TRIP_FRAC:.0%} of target {target_w:.3f} W '
                        '(siggen set to 0 Vpp)'
                    )

                if trip_reason is None:
                    if phase == 'ramp':
                        if time.perf_counter() - step_wall_t0 > RAMP_TIMEOUT_S:
                            raise RuntimeError(
                                f'Step {step_idx + 1}: ramp exceeded {RAMP_TIMEOUT_S}s '
                                f'(P={power_w:.3f} W, target={target_w:.3f} W, Vpp={vpp:.4f}).'
                            )
                        vpp_new = apply_vpp_slew(vpp, power_ctrl_avg, target_w, dt_loop)
                        if siggen is not None and abs(vpp_new - vpp) > 1e-12:
                            siggen.config_channel(siggen_channel, {'Vpp': float(vpp_new)})
                        vpp = vpp_new

                        if abs(power_ctrl_avg - target_w) <= tol:
                            phase = 'hold'
                            hold_t0 = time.perf_counter()
                            max_temp_hist.clear()
                            print(f'  Target band reached at t_run={elapsed_run:.1f} s; '
                                  f'holding ({mode})...')
                    else:
                        # Hold: correct drift at the same slew rate on smoothed power.
                        vpp_new = apply_vpp_slew(vpp, power_ctrl_avg, target_w, dt_loop)
                        if siggen is not None and abs(vpp_new - vpp) > 1e-12:
                            siggen.config_channel(siggen_channel, {'Vpp': float(vpp_new)})
                        vpp = vpp_new

                        hold_elapsed = time.perf_counter() - hold_t0

                        if mode == 'SS':
                            mx = global_max_temp(temps)
                            max_temp_hist.append(mx)
                            if steady_state_ok(max_temp_hist):
                                print(f'  Steady state (Δmax≤{STEADY_TEMP_BAND_C} °C over '
                                      f'{STEADY_SAMPLES} samples) after {hold_elapsed:.1f} s hold.')
                                break
                            if hold_elapsed >= SS_TIMEOUT_S:
                                print(f'  SS timeout ({SS_TIMEOUT_S}s); proceeding without steady state.')
                                break
                        else:
                            if hold_elapsed >= W_HOLD_S:
                                print(f'  Completed {W_HOLD_S}s hold for W step.')
                                break

                row = {
                    'timestamp': round(elapsed_run, 3),
                    'recipe_step': step_idx + 1,
                    'recipe_mode': mode,
                    'phase': phase,
                    'target_power_W': target_w,
                    'Vpp': vpp,
                    'Power (W)': power_w,
                    'Power_running_avg (W)': round(power_ctrl_avg, 6),
                    'fault': trip_reason or '',
                }
                row.update({f'Current {i + 1}': c for i, c in enumerate(currents)})
                row.update(temps)
                rows.append(row)

                sorted_temp_items = sorted(
                    temps.items(), key=lambda kv: _temp_sort_key(kv[0]), reverse=True
                )
                cur_str = '  '.join(f'I{i + 1}={c:.3f}' for i, c in enumerate(currents))
                tmp_str = '  '.join(f'{k}={v:.3f}' for k, v in sorted_temp_items)
                mx_dbg = global_max_temp(temps)
                print(f't={elapsed_run:7.2f}s  step={step_idx + 1} {phase:4s}  '
                      f'Vpp={vpp:.4f}  P={power_w:.3f}W  Pavg={power_ctrl_avg:.3f}W  '
                      f'target={target_w:.3f}  Tmax={mx_dbg:.3f}')
                print(f'  currents: {cur_str}')
                print(f'  temps:    {tmp_str}')

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
                        plot.update(temps, currents, elapsed=elapsed_run)

                sleep_for = interval - (time.perf_counter() - loop_iter_t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            setpoint_w = target_w

    except KeyboardInterrupt:
        print('\nCtrl+C received; saving what we have...')
    except Exception as e:
        print(f'\nError in schedule loop ({type(e).__name__}: {e}); saving what we have...')
        traceback.print_exc()
    finally:
        try:
            dev.close()
        except Exception:
            pass
        if plot is not None:
            plot.close()

    if save_data:
        save_rows(rows, out_path)
    else:
        print(f'save_data=False; collected {len(rows)} rows, skipping write.')


def main():
    p = argparse.ArgumentParser(description='Run Hyperion + siggen schedule from Excel.')
    p.add_argument('--schedule', default=DEFAULT_SCHEDULE_PATH, help='Recipe .xlsx path')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='monitor_supermode.json')
    p.add_argument('--out', default=None, help='Output xlsx (default: timestamped in DEFAULT_OUT_DIR)')
    p.add_argument('--interval', type=float, default=DEFAULT_INTERVAL, help='Sample period (s)')
    p.add_argument('--no-plot', action='store_true', help='Disable live matplotlib dashboard')
    p.add_argument('--no-save', action='store_true', help='Do not write output file')
    args = p.parse_args()

    sensors = sensors_from_config(args.config)
    print(f'Using sensors: {sensors}')

    if args.out:
        out_path = args.out
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out_path = os.path.join(
            DEFAULT_OUT_DIR, f'{time.strftime("%Y%m%d-%H%M%S")}_excel_schedule.xlsx'
        )

    siggen, pyvisa_rm = None, None
    try:
        siggen, pyvisa_rm = init_sdg5162(
            freq_hz=DEFAULT_SIGGEN_FREQ_HZ,
            vpp=DEFAULT_SIGGEN_VPP,
            channel=DEFAULT_SIGGEN_CHANNEL,
            turn_on=DEFAULT_SIGGEN_TURN_ON,
            visa_timeout_ms=DEFAULT_SIGGEN_VISA_TIMEOUT_MS,
        )
        run_excel_schedule(
            DEFAULT_IP,
            sensors,
            DEFAULT_COM_PORT,
            DEFAULT_ON_ARRAY,
            args.schedule,
            out_path,
            interval=args.interval,
            live_plot=not args.no_plot,
            spacing_mm=DEFAULT_AXIAL_SPACING_MM,
            siggen=siggen,
            save_data=not args.no_save,
            siggen_channel=DEFAULT_SIGGEN_CHANNEL,
            resistances=COIL_RESISTANCES_OHMS,
            initial_vpp=DEFAULT_SIGGEN_VPP,
        )
    finally:
        shutdown_sdg5162(siggen, channel=DEFAULT_SIGGEN_CHANNEL)
        release_siggen_visa(siggen, pyvisa_rm)


if __name__ == '__main__':
    main()
