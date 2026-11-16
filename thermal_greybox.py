"""Grey-box axial thermal model of the induction-heated susceptor.

Shared by model fitting and evaluation:

- Loader for the *_excel_schedule.xlsx run logs (16 fiber-optic temperatures
  at 15 mm spacing, 6 coil currents, N2 flow setpoint).
- Kalman/RTS smoother for the coil currents.
- Biot-Savart axial field basis, B(x,t) = sum_c I_c(t) * basis_c(x), with
  heating P(x,t) = power_scale * B(x,t)^2 / B2_REF_T2 or the Bessel EM model.
- Backward-Euler rollout of a 1-D axial energy balance with conduction,
  lateral loss, radiation, end losses and an optional advected N2 gas field.
- Bounded parameters with units (PARAMETER_SPECS) and a registry of model
  variants (MODEL_VARIANTS).

Conductances, heat capacities and loss coefficients are positive and bounded.
Conduction and linear losses are implicit, so the rollout is stable for any
step size. The simulator is smooth in the parameters and is fit with scipy
least_squares (numerical Jacobian).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import jv as _bessel_j

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# --------------------------------------------------------------------------- #
# Constants and geometry
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parents[1]
TEMP_SENSOR_SPACING_M = 15e-3
SUSCEPTOR_LENGTH_M = 0.200
MIN_VALID_POINT_MAX_C = 400.0
TEMP_COL_PREFIX = "FO_"
CURRENT_COL_PREFIX = "Current"
FLOW_LEVELS_SLPM = (0.0, 1.0, 2.0, 3.0)

SIGMA_SB = 5.670374419e-8          # Stefan-Boltzmann [W/m^2/K^4]
KELVIN = 273.15
TREF_C = 300.0                     # reference temperature for Cp(T)/k(T) expansions
# Reference B^2 [T^2]. Raw Biot-Savart B^2 is ~1e-8, so normalising by this
# makes power_scale O(100) W/m (power at the reference field) and improves
# optimiser conditioning.
B2_REF_T2 = 1.0e-8

# N2 properties for the gas advection model.
N2_RHO_STP_KG_M3 = 1.2506          # density at 0 degC, 1 atm
N2_CP_J_KG_K = 1040.0              # specific heat (roughly constant 25-600 C)
SLPM_TO_M3_S = 1.0 / 60000.0       # standard litres/min -> m^3/s

# Susceptor geometry.
SUSC_R_M = 19e-3
A_CROSS_M2 = np.pi * SUSC_R_M ** 2          # conduction cross-section
PERIMETER_M = 2.0 * np.pi * SUSC_R_M        # lateral surface per unit length

# Measured RF drive frequency.
RF_FREQ_HZ = 6.15e6
RF_OMEGA_RAD_S = 2.0 * np.pi * RF_FREQ_HZ

# Coil geometry.
MU_0 = 4 * np.pi * 1e-7
N_COILS = 6
TURNS_PER_COIL = 4
INSULATION_T_M = 0.0
COIL_RADIUS_M = (2.9 / 2) * 25.4e-3          # 2.9 in coil diameter, 36.83 mm radius
COIL_GAP_M = 1.1 * 25.4e-3
TURN_PITCH_M = 3e-3
COIL_PACK_LENGTH_M = (TURNS_PER_COIL - 1) * TURN_PITCH_M
COIL_START_PITCH_M = COIL_PACK_LENGTH_M + COIL_GAP_M
WINDING_LENGTH_M = (N_COILS - 1) * COIL_START_PITCH_M + COIL_PACK_LENGTH_M
COIL_WINDING_START_M = (8 * 25.4e-3 - WINDING_LENGTH_M) / 2.0  # 8 inch susceptor
COIL_SUSCEPTOR_OFFSET_M = 0.35 * 25.4e-3


def _pandas() -> Any:
    import pandas as pd

    return pd


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x, axis=axis)


def _em_coupling_factor(sigma: float | np.ndarray) -> float | np.ndarray:
    """EM coupling factor for a solid cylindrical susceptor in an axial field.

        delta = 1 / sqrt(mu0 * pi * f * sigma)    skin depth [m]
        z     = (1+j) * Rsus / delta
        f_em  = Im(J2(z) / J0(z))                 [-]

    P_line = 2*(omega/mu0) * |B(x)|^2 * A_cross * f_em is the deposited power
    per unit length [W/m]. Limits: f_em ~ R^2/(4*delta^2) for R << delta and
    f_em ~ delta/R for R >> delta.

    Parameters
    ----------
    sigma : float or ndarray
        Susceptor conductivity [S/m], clipped to [1, 1e9].

    Returns
    -------
    float or ndarray
        Im(J2(z)/J0(z)) >= 0, same shape as sigma.
    """
    sigma_c = np.clip(sigma, 1.0, 1.0e9)
    delta = 1.0 / np.sqrt(MU_0 * np.pi * RF_FREQ_HZ * sigma_c)
    z = (1.0 + 1j) * SUSC_R_M / delta
    j0 = _bessel_j(0, z)
    j2 = _bessel_j(2, z)
    # fallback where |J0| is tiny
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.abs(j0) > 1e-30, j2 / j0, 0.5 + 0j)
    return np.maximum(np.imag(r), 0.0)


# EM prefactor 2*(omega/mu0)*A_cross [W/(m·T^2)]
_EM_BASE_W_PER_M_T2 = 2.0 * RF_OMEGA_RAD_S / MU_0 * A_CROSS_M2


# --------------------------------------------------------------------------- #
# Axial model grid
# --------------------------------------------------------------------------- #
def model_x_grid() -> np.ndarray:
    """Axial nodes at 0, 15, ..., 195, 200 mm (metres)."""
    n_full = int(round(SUSCEPTOR_LENGTH_M / TEMP_SENSOR_SPACING_M))  # 13 -> 195 mm
    nodes = [i * TEMP_SENSOR_SPACING_M for i in range(n_full + 1)]
    if abs(nodes[-1] - SUSCEPTOR_LENGTH_M) > 1e-9:
        nodes.append(SUSCEPTOR_LENGTH_M)
    return np.array(nodes, dtype=float)


def control_volume_widths(x_m: np.ndarray) -> np.ndarray:
    edges = np.empty(x_m.size + 1)
    edges[0], edges[-1] = x_m[0], x_m[-1]
    edges[1:-1] = 0.5 * (x_m[:-1] + x_m[1:])
    widths = np.diff(edges)
    half = TEMP_SENSOR_SPACING_M / 2
    widths[0] = max(widths[0], half)
    widths[-1] = max(widths[-1], half)
    return widths


# --------------------------------------------------------------------------- #
# Biot-Savart axial heating basis
# --------------------------------------------------------------------------- #
def coil_field_basis(x_m: np.ndarray) -> np.ndarray:
    """Per-coil axial field shape [T/A], shape (N_COILS, len(x))."""
    basis = np.zeros((N_COILS, x_m.size))
    winding_start = COIL_WINDING_START_M + COIL_SUSCEPTOR_OFFSET_M
    for c in range(N_COILS):
        coil_start = winding_start + c * COIL_START_PITCH_M
        for t in range(TURNS_PER_COIL):
            turn_center = coil_start + t * TURN_PITCH_M
            dx = x_m - turn_center
            basis[c] += MU_0 * COIL_RADIUS_M ** 2 / (2 * (COIL_RADIUS_M ** 2 + dx ** 2) ** 1.5)
    return basis


def axial_field_squared(currents_a: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """B(x,t)^2 [T^2] from currents (T,6) and basis (6,N) -> (T,N)."""
    b = currents_a @ basis
    return b * b


# --------------------------------------------------------------------------- #
# Kalman / RTS smoother for the noisy coil currents
# --------------------------------------------------------------------------- #
def smooth_currents_rts(
    time_s: np.ndarray,
    currents_a: np.ndarray,
    process_var: float = 5e-4,
    meas_var: Optional[float] = None,
) -> np.ndarray:
    """Random-walk Kalman filter + RTS smoother on each current channel.

    State model x_{k+1} = x_k + w, w ~ N(0, q*dt). If meas_var is None it is
    estimated from the MAD of the first differences.
    """
    arr = np.asarray(currents_a, dtype=float)
    out = np.empty_like(arr)
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.where(dt > 0, dt, np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0)
    for j in range(arr.shape[1]):
        z = arr[:, j]
        r = meas_var
        if r is None:
            d = np.diff(z)
            mad = np.median(np.abs(d - np.median(d))) if d.size else 0.0
            sigma = max(1.4826 * mad / np.sqrt(2.0), 1e-3)
            r = sigma ** 2
        n = z.size
        xf = np.empty(n)
        Pf = np.empty(n)
        xp = np.empty(n)
        Pp = np.empty(n)
        x, P = z[0], 1.0
        for k in range(n):
            q = process_var * dt[k]
            xp[k] = x
            Pp[k] = P + q
            K = Pp[k] / (Pp[k] + r)
            x = xp[k] + K * (z[k] - xp[k])
            P = (1 - K) * Pp[k]
            xf[k], Pf[k] = x, P
        xs = xf.copy()
        for k in range(n - 2, -1, -1):
            C = Pf[k] / Pp[k + 1]
            xs[k] = xf[k] + C * (xs[k + 1] - xp[k + 1])
        out[:, j] = xs
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------- #
# Experiment container + loader
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentData:
    name: str
    source_path: Path
    time_s: np.ndarray          # (T,)
    flow_slpm: np.ndarray       # (T,)
    currents_a: np.ndarray      # (T,6) smoothed
    currents_raw_a: np.ndarray  # (T,6) raw
    temps_c: np.ndarray         # (T,N) interpolated to model grid
    observed_mask: np.ndarray   # (N,) bool, node run-max >= 400 C
    x_m: np.ndarray             # (N,)
    raw_x_m: np.ndarray
    dt_median_s: float
    field_basis: np.ndarray = field(default=None, repr=False)  # (6,N)
    b2: np.ndarray = field(default=None, repr=False)           # (T,N)


def discover_data_files(data_dir: Path = DATA_DIR) -> List[Path]:
    # skip Excel lock files (~$...)
    return sorted(p for p in Path(data_dir).glob("*_excel_schedule.xlsx")
                  if not p.name.startswith("~$"))


def _cache_path(path: Path, smooth_process_var: float) -> Path:
    return path.with_suffix(f".cache_q{smooth_process_var:.0e}.npz")


def load_experiment(
    path: Path,
    x_m: Optional[np.ndarray] = None,
    smooth_process_var: float = 5e-4,
    min_point_max_c: float = MIN_VALID_POINT_MAX_C,
    use_cache: bool = True,
) -> ExperimentData:
    path = Path(path)
    x_model = model_x_grid() if x_m is None else np.asarray(x_m, dtype=float)
    cache = _cache_path(path, smooth_process_var)
    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache, allow_pickle=False)
        if d["x_m"].shape == x_model.shape and np.allclose(d["x_m"], x_model):
            basis = coil_field_basis(x_model)
            return ExperimentData(
                name=str(path.stem), source_path=path,
                time_s=d["time_s"], flow_slpm=d["flow_slpm"], currents_a=d["currents_a"],
                currents_raw_a=d["currents_raw_a"], temps_c=d["temps_c"],
                observed_mask=d["observed_mask"], x_m=d["x_m"], raw_x_m=d["raw_x_m"],
                dt_median_s=float(d["dt_median_s"]), field_basis=basis,
                b2=axial_field_squared(d["currents_a"], basis))

    pd = _pandas()
    df = pd.read_excel(path, sheet_name="Sheet1")

    temp_cols = [str(c) for c in df.columns if str(c).startswith(TEMP_COL_PREFIX)]
    cur_cols = [str(c) for c in df.columns if str(c).startswith(CURRENT_COL_PREFIX)]
    if len(temp_cols) < 14:
        raise ValueError(f"{path.name}: found {len(temp_cols)} temp columns, expected >=14")
    if len(cur_cols) != N_COILS:
        raise ValueError(f"{path.name}: found {len(cur_cols)} current columns, expected {N_COILS}")

    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    ok = np.isfinite(t)
    df = df.loc[ok].reset_index(drop=True)
    t = t[ok]
    time_s = t - t[0]
    dt = np.diff(time_s)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")

    # Fiber columns are in spatial order (0, 15, ..., 225 mm). Interpolate onto
    # the model grid, which ends at 200 mm.
    raw_temps = df[temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_x = np.arange(raw_temps.shape[1]) * TEMP_SENSOR_SPACING_M
    temps = np.vstack([np.interp(x_model, raw_x, row) for row in raw_temps])
    # A node is observed only if it sits on a sensor (within 1 µm). The 200 mm
    # node is interpolated between the 195 mm and 210 mm sensors, and 210 mm is
    # outside the susceptor, so it is excluded.
    is_sensor_node = np.any(
        np.abs(x_model[:, None] - raw_x[None, :]) < 1e-6, axis=1
    )
    observed_mask = (np.nanmax(temps, axis=0) >= min_point_max_c) & is_sensor_node

    currents_raw = df[cur_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    currents_raw = np.nan_to_num(np.abs(currents_raw), nan=0.0, posinf=0.0, neginf=0.0)
    currents = smooth_currents_rts(time_s, currents_raw, process_var=smooth_process_var)

    flow = pd.to_numeric(df["flow_setpoint"], errors="coerce").fillna(0.0).to_numpy(float)

    basis = coil_field_basis(x_model)
    b2 = axial_field_squared(currents, basis)

    if use_cache:
        try:
            np.savez(
                cache, time_s=time_s, flow_slpm=flow, currents_a=currents,
                currents_raw_a=currents_raw, temps_c=temps, observed_mask=observed_mask,
                x_m=x_model, raw_x_m=raw_x, dt_median_s=dt_med)
        except Exception:
            pass

    return ExperimentData(
        name=path.stem,
        source_path=path,
        time_s=time_s,
        flow_slpm=flow,
        currents_a=currents,
        currents_raw_a=currents_raw,
        temps_c=temps,
        observed_mask=observed_mask,
        x_m=x_model,
        raw_x_m=raw_x,
        dt_median_s=dt_med,
        field_basis=basis,
        b2=b2,
    )


def downsample_experiment(exp: ExperimentData, stride: int) -> ExperimentData:
    if stride <= 1:
        return exp
    idx = np.arange(0, exp.time_s.size, int(stride))
    if idx[-1] != exp.time_s.size - 1:
        idx = np.r_[idx, exp.time_s.size - 1]
    return replace(
        exp,
        time_s=exp.time_s[idx] - exp.time_s[idx][0],
        flow_slpm=exp.flow_slpm[idx],
        currents_a=exp.currents_a[idx],
        currents_raw_a=exp.currents_raw_a[idx],
        temps_c=exp.temps_c[idx],
        b2=exp.b2[idx],
    )


def trim_warmup(exp: ExperimentData, skip_s: float) -> ExperimentData:
    if skip_s <= 0:
        return exp
    keep = exp.time_s >= skip_s
    if keep.sum() < 5:
        return exp
    return replace(
        exp,
        time_s=exp.time_s[keep] - exp.time_s[keep][0],
        flow_slpm=exp.flow_slpm[keep],
        currents_a=exp.currents_a[keep],
        currents_raw_a=exp.currents_raw_a[keep],
        temps_c=exp.temps_c[keep],
        b2=exp.b2[keep],
    )


def estimate_sensor_noise_c(exp: ExperimentData) -> np.ndarray:
    diff = np.diff(exp.temps_c, axis=0)
    mad = np.nanmedian(np.abs(diff - np.nanmedian(diff, axis=0)), axis=0)
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return np.maximum(sigma, 0.75)


# --------------------------------------------------------------------------- #
# Parameters: specs (units / bounds / interpretation) + bounded vector handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    lower: float
    upper: float
    default: float
    log: bool
    interpretation: str


PARAMETER_SPECS: Dict[str, ParamSpec] = {
    "rho_cp": ParamSpec(
        "rho_cp", "J/(m^3.K)", 1.0e5, 1.0e7, 2.0e6, True,
        "Volumetric heat capacity rho*Cp of the susceptor (thermal mass per unit volume)."),
    "cp_T1": ParamSpec(
        "cp_T1", "1/K", -5e-4, 2.5e-3, 0.0, False,
        "Linear temperature coefficient of Cp: Cp(T)=rho_cp*(1+cp_T1*(T-Tref))."),
    "k_w": ParamSpec(
        "k_w", "W/(m.K)", 0.5, 400.0, 30.0, True,
        "Effective axial thermal conductivity of the susceptor (lumps solid conduction plus "
        "internal radiative/gas axial transport)."),
    "kT1": ParamSpec(
        "kT1", "1/K", -1.5e-3, 2e-2, 0.0, False,
        "Linear temperature coefficient of k: k(T)=k_w*(1+kT1*(T-Tref))."),
    "h0": ParamSpec(
        "h0", "W/(m^2.K)", 0.1, 200.0, 8.0, True,
        "Baseline (natural-convection + conduction-through-insulation) loss coefficient."),
    "h_flow": ParamSpec(
        "h_flow", "W/(m^2.K)/SLPM^n", 0.0, 150.0, 5.0, False,
        "Forced-convection loss gain: extra wall loss per SLPM^n_flow of nitrogen."),
    "n_flow": ParamSpec(
        "n_flow", "-", 0.3, 1.0, 0.8, False,
        "Flow exponent for forced convection (Nu ~ Re^n)."),
    "h_T": ParamSpec(
        "h_T", "W/(m^2.K)/K", 0.0, 0.6, 0.0, False,
        "Temperature dependence of the loss coefficient (h grows with wall-ambient delta)."),
    "emissivity": ParamSpec(
        "emissivity", "-", 0.0, 1.0, 0.4, False,
        "Effective surface emissivity for radiative loss."),
    "power_scale": ParamSpec(
        "power_scale", "W/m", 1.0, 5.0e4, 200.0, True,
        "Lumped electromagnetic heating efficiency: deposited axial power density (W/m) at the "
        "peak reference field (B^2 = B2_REF_T2); scales the Biot-Savart B(x)^2 shape."),
    "ps_T1": ParamSpec(
        "ps_T1", "1/K", -5.0e-4, 3.0e-3, 0.0, False,
        "Temperature coefficient of induction-heating efficiency: eta(T)=power_scale*"
        "(1+ps_T1*(T-Tref)). Positive captures rising susceptor electromagnetic coupling "
        "(conductivity) as it heats."),
    "coil_mid_gain": ParamSpec(
        "coil_mid_gain", "-", 0.3, 3.0, 1.0, False,
        "Relative electromagnetic coupling gain of the 4 interior coils vs the 2 end coils "
        "(applied to the current before the Biot-Savart B^2 power law). >1 deposits more power "
        "mid-susceptor. Only identifiable when the dataset spans multiple current distributions "
        "(e.g. peaked + uniform coil schedules)."),
    "coil_end_gain": ParamSpec(
        "coil_end_gain", "-", 0.1, 5.0, 1.0, True,
        "Electromagnetic coupling gain for the two end coils (coils 1 and 6) independently from "
        "coil_mid_gain. <1 reduces EM power near both susceptor end faces (x≈0 and x≈200mm). "
        "Corrects Biot-Savart field errors near the coil terminations, directly targeting the "
        "systematic over/under-prediction at the 15mm and 195mm sensors."),
    "coil_offset_m": ParamSpec(
        "coil_offset_m", "m", -0.010, 0.010, 0.0, False,
        "Axial offset of the entire coil assembly relative to the susceptor (metres). "
        "Positive = coils shift toward the inlet (left); negative = toward the outlet (right). "
        "Corrects mechanical assembly tolerances: the Biot-Savart field is recomputed at "
        "x_node + coil_offset_m, shifting where coil 1's leading edge and coil 6's "
        "leading edge fall relative to the sensor positions. Targets the -4 C steady-state "
        "bias at 195mm (near the coil-6 leading edge) and the -5.5 C bias at 15mm."),
    "h_gw": ParamSpec(
        "h_gw", "W/(m^2.K)", 0.5, 1e5, 40.0, True,
        "Wall-to-gas heat-transfer coefficient for the advected nitrogen field."),
    "inlet_c": ParamSpec(
        "inlet_c", "degC", 10.0, 200.0, 25.0, False,
        "Nitrogen inlet temperature at the left boundary (advection)."),
    "ambient_c": ParamSpec(
        "ambient_c", "degC", 10.0, 200.0, 25.0, False,
        "Ambient temperature for convective + radiative loss."),
    "shell_C": ParamSpec(
        "shell_C", "J/K", 1.0e2, 3.0e6, 3.0e4, True,
        "Lumped heat capacity of the insulation/support shell (the slow second thermal mass "
        "that produces the long within-run upward drift)."),
    "shell_G": ParamSpec(
        "shell_G", "W/K", 1.0e-3, 80.0, 0.5, True,
        "Conductance from the insulation shell to ambient."),
    "shell_init_frac": ParamSpec(
        "shell_init_frac", "-", 0.0, 1.0, 0.7, False,
        "Initial shell temperature as a fraction of (mean initial wall - ambient) above ambient."),
    "shell_G_T1": ParamSpec(
        "shell_G_T1", "1/K", -5.0e-3, 5.0e-3, 0.0, False,
        "Temperature coefficient of shell-to-ambient conductance: "
        "shell_G_eff = shell_G * clip(1 + shell_G_T1*(T_sh - Tref), 0.1, 10). "
        "Positive: insulation becomes more conductive as it heats up (typical for fibrous/ceramic). "
        "Negative: conductance decreases with temperature."),
    "ambient_lat_c": ParamSpec(
        "ambient_lat_c", "degC", 10.0, 100.0, 25.0, False,
        "Ambient temperature for lateral surface losses (convection + radiation). "
        "Decouples lateral cooling from end-face/shell cooling when different from ambient_c."),
    "end_flow_gain_right": ParamSpec(
        "end_flow_gain_right", "1/SLPM", 0.0, 5.0, 0.0, False,
        "Flow enhancement of the RIGHT end-face loss: h_end_right*(1+end_flow_gain_right*flow). "
        "N2 exits from the right so flow can cool it too."),
    "h_end_T": ParamSpec(
        "h_end_T", "1/K", 0.0, 1.0e-2, 0.0, False,
        "Temperature coefficient of end-face losses: h_end *= (1 + h_end_T*(T_end - Tref)). "
        "Captures increasing end conduction as the susceptor ends get hotter."),
    "rend_C": ParamSpec(
        "rend_C", "J/K", 1.0e1, 1.0e5, 500.0, True,
        "Lumped heat capacity of the right end-cap/fitting structure. "
        "Models the thermal mass of the gas outlet port and end structure that is thermally "
        "coupled to the last susceptor node. Produces the delayed response seen at the 195mm sensor."),
    "rend_G": ParamSpec(
        "rend_G", "W/K", 1.0e-3, 20.0, 0.1, True,
        "Conductance between the right end-cap and ambient. Companion to rend_C."),
    "rend_G_sus": ParamSpec(
        "rend_G_sus", "W/K", 1.0e-3, 50.0, 1.0, True,
        "Conductance from the susceptor right end face (195mm node) to the right end-cap. "
        "Controls how quickly the end-cap heats up from the susceptor."),
    "lend_C": ParamSpec(
        "lend_C", "J/K", 1.0e1, 1.0e5, 500.0, True,
        "Lumped heat capacity of the left end-cap/inlet fitting structure. "
        "Models the thermal mass of the gas inlet port coupled to the left susceptor face (0mm node). "
        "Symmetric counterpart to rend_C; explains the slow response at the 0mm and 15mm sensors."),
    "lend_G": ParamSpec(
        "lend_G", "W/K", 1.0e-3, 20.0, 0.1, True,
        "Conductance between the left end-cap and ambient. Companion to lend_C."),
    "lend_G_sus": ParamSpec(
        "lend_G_sus", "W/K", 1.0e-3, 50.0, 1.0, True,
        "Conductance from susceptor left end face (0mm node) to the left end-cap. "
        "Controls how quickly the inlet fitting heats up from the susceptor."),
    "h_end_left": ParamSpec(
        "h_end_left", "W/K", 1.0e-3, 50.0, 0.5, True,
        "Heat-loss conductance at the LEFT end face (gas inlet side) to the inlet temperature. "
        "Explains the cold upstream end despite end-peaked coil power."),
    "h_end_right": ParamSpec(
        "h_end_right", "W/K", 1.0e-3, 50.0, 0.5, True,
        "Heat-loss conductance at the RIGHT end face (outlet side) to ambient."),
    "end_flow_gain": ParamSpec(
        "end_flow_gain", "1/SLPM", 0.0, 5.0, 0.0, False,
        "Flow enhancement of the left end-face loss: h_end_left*(1+end_flow_gain*flow)."),
    "tau_sensor": ParamSpec(
        "tau_sensor", "s", 1.0, 3600.0, 25.0, True,
        "Effective first-order thermal lag between the susceptor wall and the fiber-optic reading. "
        "In practice this lumps sensor thermal inertia, the gas column between wall and fiber, "
        "and any mounting-structure lag.  Larger values (hundreds of seconds) are physically "
        "plausible when the fiber reads the gas inside the tube rather than the wall directly. "
        "IIR: T_s[k]=exp(-dt/tau)*T_s[k-1]+(1-exp(-dt/tau))*T_wall[k]."),
    "tau_sensor_right": ParamSpec(
        "tau_sensor_right", "s", 1.0, 3600.0, 25.0, True,
        "Separate sensor thermal lag for the rightmost observed node (195mm outlet sensor). "
        "The outlet fiber-optic probe may see gas that has passed through the hot end-cap fitting "
        "(thermal mass rend_C), giving a longer effective lag than the interior sensors. "
        "When this parameter is present, the 195mm column of the predicted temperature array "
        "is filtered with this time constant instead of tau_sensor. "
        "IIR: T_s[k]=exp(-dt/tau_r)*T_s[k-1]+(1-exp(-dt/tau_r))*T_wall[k] for the 195mm node."),
    "power_axial_tilt": ParamSpec(
        "power_axial_tilt", "-", -5.0, 5.0, 0.0, False,
        "Linear axial tilt of induction power deposition: P(x) *= max(1 + power_axial_tilt*(x/L - 0.5), 0). "
        ">0 tilts power towards the outlet (right end). Corrects Biot-Savart vs actual coupling mismatch."),
    "tau_power": ParamSpec(
        "tau_power", "s", 1.0, 600.0, 30.0, True,
        "First-order lag on the delivered induction power: the instantaneous Biot-Savart power is "
        "filtered through P_eff[k]=w*P_eff[k-1]+(1-w)*P_raw[k], w=exp(-dt/tau_power). "
        "Models RF-generator bandwidth limits and thermal lag in the electromagnetic coupling, "
        "explaining why the real susceptor heats more slowly than instantaneous B² predicts."),
    "ps_T2": ParamSpec(
        "ps_T2", "1/K^2", -2.0e-5, 2.0e-5, 0.0, False,
        "Quadratic temperature coefficient of induction-heating efficiency: "
        "eta(T) = power_scale * clip(1 + ps_T1*(T-Tref) + ps_T2*(T-Tref)^2, 0.05, 10). "
        "Allows the efficiency curve to bend; negative ps_T2 creates a peak efficiency at "
        "T = Tref - ps_T1/(2*ps_T2) and limits over-prediction at high temperatures."),
    # --- EM-power-model parameters (used by full_greybox_em and above) --------
    "sigma_sus": ParamSpec(
        "sigma_sus", "S/m", 1.0, 500.0, 200.0, False,
        "Effective electrical conductivity of the susceptor at Tref (300 °C). "
        "Determines the skin depth delta=1/sqrt(mu0*pi*f*sigma) and the Bessel-function "
        "EM coupling factor Im(J2(z)/J0(z)) that converts the Biot-Savart B²(x) field "
        "into induction power per unit axial length: "
        "P(x) = 2*(omega/mu0)*A_cross * sigma_coupling_factor * |B(x)|^2. "
        "Replaces the empirical 'power_scale / B2_REF' normalization with a physically "
        "grounded EM formula (matches compute_power_density in bio_savart_1d.py)."),
    "sigma_T1": ParamSpec(
        "sigma_T1", "1/K", -5.0e-3, 5.0e-3, -5.0e-4, False,
        "Temperature coefficient of susceptor conductivity: "
        "sigma(T) = sigma_sus * clip(1 + sigma_T1*(T-Tref), 0.01, 100). "
        "Negative for metals and graphitic materials (conductivity falls with temperature), "
        "which causes the Bessel coupling factor — and thus induction power — to vary with "
        "temperature.  Replaces the empirical ps_T1/ps_T2 correction with a physics-based "
        "skin-depth mechanism."),
    "power_gain": ParamSpec(
        "power_gain", "-", 0.05, 20.0, 1.0, True,
        "Global multiplicative gain on the EM induction power: P_eff = power_gain * P_biot_savart. "
        "Corrects any systematic error in the absolute Biot-Savart power level arising from "
        "coil-geometry uncertainty, coupling-gap tolerances, or power-delivery losses. "
        "=1.0 means the physics formula is trusted exactly; log-bounded for scale invariance."),
}


@dataclass
class ThermalParams:
    values: Dict[str, float]

    def get(self, name: str) -> float:
        if name in self.values:
            return float(self.values[name])
        return float(PARAMETER_SPECS[name].default)

    def asdict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.values.items()}


@dataclass(frozen=True)
class ModelVariant:
    name: str
    active_names: Tuple[str, ...]
    fixed: Dict[str, float]
    description: str
    use_conduction: bool = True
    use_radiation: bool = True
    use_gas_advection: bool = False
    use_shell: bool = False
    use_end_loss: bool = False
    use_sensor_lag: bool = False
    use_em_power: bool = False   # Bessel EM power model; else power_scale * B²
    use_gas_exit_bc: bool = False  # right end-face sink is the gas exit temp, not t_amb
    use_rend: bool = False  # right end-cap thermal mass (rend_C, rend_G, rend_G_sus)
    use_lend: bool = False  # left end-cap thermal mass (lend_C, lend_G, lend_G_sus)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([PARAMETER_SPECS[n].lower for n in self.active_names])
        hi = np.array([PARAMETER_SPECS[n].upper for n in self.active_names])
        return lo, hi

    def initial_vector(self) -> np.ndarray:
        return np.array([PARAMETER_SPECS[n].default for n in self.active_names])

    def decode(self, x: np.ndarray) -> ThermalParams:
        vals = dict(self.fixed)
        for n, v in zip(self.active_names, x):
            vals[n] = float(v)
        return ThermalParams(vals)


MODEL_VARIANTS: Dict[str, ModelVariant] = {
    "first_order": ModelVariant(
        "first_order",
        ("rho_cp", "h0", "power_scale", "ambient_c"),
        {"emissivity": 0.0},
        "Independent first-order node energy balance: thermal mass + lumped loss + EM heating. "
        "No axial conduction, no radiation, no flow dependence.",
        use_conduction=False, use_radiation=False, use_gas_advection=False),
    "multinode_rc": ModelVariant(
        "multinode_rc",
        ("rho_cp", "k_w", "h0", "emissivity", "power_scale",
         "h_end_left", "h_end_right", "ambient_c"),
        {},
        "Multi-node RC network: axial conduction + lumped lateral loss + radiation + end-face "
        "losses + EM heating. Constant (temperature/flow-independent) parameters.",
        use_conduction=True, use_radiation=True, use_end_loss=True),
    "rc_shell": ModelVariant(
        "rc_shell",
        ("rho_cp", "k_w", "h0", "emissivity", "power_scale",
         "h_end_left", "h_end_right", "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "Multi-node RC wall + a slow insulation-shell thermal mass (2C2R). The wall loses heat "
        "to the shell (h0), the shell slowly leaks to ambient (shell_G); captures the long "
        "within-run drift. Includes end-face losses. No flow term.",
        use_conduction=True, use_radiation=True, use_shell=True, use_end_loss=True),
    "flow_greybox": ModelVariant(
        "flow_greybox",
        ("rho_cp", "k_w", "h0", "h_flow", "n_flow", "emissivity", "power_scale",
         "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "RC wall + shell + end losses + flow-dependent forced-convection loss "
        "h(flow)=h0+h_flow*flow^n_flow with flow-enhanced inlet end loss.",
        use_conduction=True, use_radiation=True, use_shell=True, use_end_loss=True),
    "advect_greybox": ModelVariant(
        "advect_greybox",
        ("rho_cp", "k_w", "h0", "emissivity", "power_scale", "h_gw", "inlet_c",
         "h_end_left", "h_end_right", "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "RC wall + shell + end losses + advected nitrogen gas field (upwind, left->right). "
        "Captures flow-driven, axially asymmetric convective cooling from first principles.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True),
    "temp_greybox": ModelVariant(
        "temp_greybox",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T", "emissivity", "power_scale",
         "h_end_left", "h_end_right", "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "RC wall + shell + end losses with temperature-dependent Cp(T), k(T) and loss h(T).",
        use_conduction=True, use_radiation=True, use_shell=True, use_end_loss=True),
    "full_greybox": ModelVariant(
        "full_greybox",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_flow", "n_flow", "h_T",
         "emissivity", "power_scale", "ps_T1", "coil_mid_gain", "h_gw", "inlet_c",
         "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "Combined temperature + flow dependence: Cp(T), k(T), h(T,flow), radiation, end losses, "
        "the advected nitrogen gas field, and the insulation shell. Most flexible physics model.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True),
    "full_greybox_v2": ModelVariant(
        "full_greybox_v2",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_flow", "n_flow", "h_T",
         "emissivity", "power_scale", "ps_T1", "coil_mid_gain", "h_gw", "inlet_c",
         "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor", "power_axial_tilt"),
        {},
        "Full grey-box (21 params) plus sensor thermal-lag tau_sensor and axial power tilt: "
        "23 params total. Targets RMSE<5 C on test5/6/7 with a single shared parameter set.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_sensor_lag=True),
    "full_greybox_v3": ModelVariant(
        "full_greybox_v3",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_flow", "n_flow", "h_T",
         "emissivity", "power_scale", "ps_T1", "coil_mid_gain", "h_gw", "inlet_c",
         "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor", "power_axial_tilt", "tau_power"),
        {},
        "Full grey-box v2 (23 params) plus RF-power delivery lag tau_power: 24 params total. "
        "tau_power accounts for why the real susceptor heats more slowly than instantaneous "
        "B² power predicts; tau_sensor handles sensor thermal inertia during fast transients.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_sensor_lag=True),
    "full_greybox_v4": ModelVariant(
        "full_greybox_v4",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_flow", "n_flow", "h_T",
         "emissivity", "power_scale", "ps_T1", "ps_T2", "coil_mid_gain", "h_gw", "inlet_c",
         "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor", "power_axial_tilt", "tau_power"),
        {},
        "Full grey-box v3 (24 params) plus quadratic induction-efficiency ps_T2: 25 params total. "
        "ps_T2 allows the efficiency-temperature curve to bend, decoupling cold-start behaviour "
        "from high-temperature steady-state and directly addressing the tau mismatch in test5.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_sensor_lag=True),
    "full_greybox_em": ModelVariant(
        "full_greybox_em",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_flow", "n_flow", "h_T",
         "emissivity", "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {},
        "Full grey-box with Bessel-function EM power density (bio_savart_1d.py formula): "
        "replaces empirical ps_T1 with sigma_sus (conductivity) + sigma_T1 (conductivity "
        "temperature coefficient).  P(x) = 2*(omega/mu0)*|B(x)|^2*A_cross*Im(J2(z)/J0(z)) "
        "where z=(1+j)*Rsus/delta and delta=1/sqrt(mu0*pi*f*sigma(T)).  22 params total; "
        "fitted on all 4 runs simultaneously, target RMSE<10 C.",
        use_conduction=True, use_radiation=True, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True),
    "full_greybox_em_v2": ModelVariant(
        "full_greybox_em_v2",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v2: sigma_sus fitted in [100,300] S/m (increasing or decreasing with T "
        "via sigma_T1), no radiation (emissivity=0), no forced-convection (h_flow=0). "
        "18 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True),
    "full_greybox_em_v3": ModelVariant(
        "full_greybox_em_v3",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v3: adds sensor lag (tau_sensor) to v2. "
        "sigma_sus in [100,300] S/m; no radiation; no forced-convection. "
        "19 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v3_coil29": ModelVariant(
        "full_greybox_em_v3_coil29",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v3_coil29: v3 redone with coil diameter 2.9 inch (was 2.7 inch, "
        "COIL_RADIUS_M = 36.83 mm) and ALL coil gains fixed to 1 (no coil_mid_gain, "
        "no coil_end_gain). Tests whether the larger coil-to-susceptor radial gap fits "
        "the data without any per-coil EM coupling correction. 18 active params; "
        "no radiation; no forced-convection; fitted on test5/6/7.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v4": ModelVariant(
        "full_greybox_em_v4",
        ("h_end_right", "ambient_c"),
        {   # other params fixed at v3 best fit
            "emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0,
            "rho_cp": 381384.0, "cp_T1": -0.0005, "k_w": 2.59165, "kT1": 0.0015,
            "h0": 0.1, "h_T": 0.181055,
            "sigma_sus": 100.0, "sigma_T1": -0.000453359,
            "coil_mid_gain": 1.7426, "h_gw": 800.0,
            "inlet_c": 74.8402, "h_end_left": 0.0153799, "end_flow_gain": 1.33368,
            "shell_C": 1273.24, "shell_G": 0.574323, "shell_init_frac": 0.743675,
            "tau_sensor": 336.258,
        },
        "Fine-tune: all v3 best params fixed, only right boundary (h_end_right, ambient_c) "
        "free. 2 active params; fitted on test5/6/7.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v5": ModelVariant(
        "full_greybox_em_v5",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_G_T1", "shell_init_frac", "ambient_c",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0, "sigma_sus": 100.0},
        "EM model v5: adds temperature-dependent insulation conductance (shell_G_T1) to v3. "
        "shell_G_eff = shell_G * clip(1 + shell_G_T1*(T_sh - Tref), 0.1, 10). "
        "sigma_sus is fixed at 100 S/m; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v6": ModelVariant(
        "full_greybox_em_v6",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain", "end_flow_gain_right",
         "h_end_T", "ambient_lat_c", "ambient_c",
         "shell_C", "shell_G", "shell_G_T1", "shell_init_frac",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v6: new BC structure — separate lateral ambient (ambient_lat_c) from end-face "
        "ambient (ambient_c), flow-dependent right end (end_flow_gain_right), temperature-dependent "
        "end losses (h_end_T), and temperature-dependent shell conductance (shell_G_T1). "
        "23 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v7": ModelVariant(
        "full_greybox_em_v7",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "power_gain", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v7: adds power_gain (global EM amplitude scale) to v3. "
        "power_gain decouples the absolute power level from sigma_sus, allowing the Bessel "
        "profile shape and the total deposited power to be fitted independently. "
        "20 active params; sigma_sus in [100,300]; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True),
    "full_greybox_em_v9": ModelVariant(
        "full_greybox_em_v9",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v9: adds right end-cap thermal mass (rend_C, rend_G, rend_G_sus) to v3. "
        "The end-cap represents the gas-outlet fitting thermally coupled to the last susceptor "
        "node — its slow thermal response explains the large time-varying RMSE at the 195mm sensor. "
        "22 active params; sigma_sus in [100,300]; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True, use_rend=True),
    "full_greybox_em_v12": ModelVariant(
        "full_greybox_em_v12",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "coil_offset_m",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v12: v10 + coil_offset_m. Axial shift of the entire coil assembly "
        "as a free parameter (±10mm). Recomputes Biot-Savart field_basis at each "
        "optimisation step. Targets the -4.4C bias at 195mm and -5.5C bias at 15mm "
        "caused by the coil-6 and coil-1 leading edges being near those sensors. "
        "26 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v11": ModelVariant(
        "full_greybox_em_v11",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "coil_end_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v11: v10 + coil_end_gain. Independent gain for end coils 1 & 6, targeting "
        "the Biot-Savart field transition errors at the 15mm (coil-1 edge) and 195mm (coil-6 "
        "edge) sensors. 26 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v16": ModelVariant(
        "full_greybox_em_v16",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v16: v15 with k_fac clip ceiling raised from 5.0 to 20.0. "
        "v15 uses kT1<=0.02; at T=500 C (TREF=300) k_fac=1+0.02*200=5.0 exactly hits the old "
        "ceiling, so kT1 had no additional effect above 500 C. Raising to 20.0 allows "
        "k(T)=k_w*(1+kT1*(T-300)) to grow up to 20x baseline, enabling the optimizer to use "
        "temperature-dependent conductivity as a proxy for the unmodelled T^4 radiation term "
        "at the hot outlet end. Same kT1 bound [-1.5e-3, 2e-2] and h_gw bound [0.5, 1e5]. "
        "25 active params; warm-started from v15_best; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v15": ModelVariant(
        "full_greybox_em_v15",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v15: v13 with further widened kT1 bound [-1.5e-3, 2e-2] and h_gw bound "
        "[0.5, 1e5]. v14 hit both the v13 bounds (kT1=0.005, h_gw=4990). Wider bounds test "
        "whether even stronger temperature-dependent conductivity and fully-saturated gas-wall "
        "heat transfer reduce RMSE. 25 active params; warm-started from v14_best; "
        "fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v14": ModelVariant(
        "full_greybox_em_v14",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor", "tau_sensor_right",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v14: v10 + tau_sensor_right (separate IIR lag for the 195mm outlet sensor). "
        "The outlet fiber-optic probe sits inside the hot end-cap fitting; its effective thermal "
        "lag differs from the interior sensors. tau_sensor_right allows the optimizer to fit a "
        "longer (or shorter) lag specifically for the 195mm column, reducing the large dynamic "
        "RMSE at that node without affecting interior-sensor accuracy. "
        "26 active params; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v13": ModelVariant(
        "full_greybox_em_v13",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v13: v10 with widened kT1 bound [-1.5e-3, 5e-3] and h_gw bound [0.5, 5000]. "
        "v10 hit both bounds (kT1=0.0015, h_gw=800). Wider search tests whether stronger "
        "temperature-dependent conductivity and/or faster gas-wall equilibration reduces RMSE. "
        "25 active params; warm-started from v10_best; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v10": ModelVariant(
        "full_greybox_em_v10",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain",
         "h_gw", "inlet_c", "h_end_left", "h_end_right",
         "end_flow_gain", "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor",
         "rend_C", "rend_G", "rend_G_sus",
         "lend_C", "lend_G", "lend_G_sus"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v10: v9 + left end-cap (lend_C, lend_G, lend_G_sus). "
        "Symmetric left end-cap models the gas-inlet fitting thermal mass at the 0mm face. "
        "25 active params; targets the 0mm/15mm sensor errors alongside the 195mm error. "
        "Fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True,
        use_rend=True, use_lend=True),
    "full_greybox_em_v8": ModelVariant(
        "full_greybox_em_v8",
        ("rho_cp", "cp_T1", "k_w", "kT1", "h0", "h_T",
         "sigma_sus", "sigma_T1", "coil_mid_gain", "h_gw",
         "inlet_c", "h_end_left", "h_end_right", "end_flow_gain",
         "shell_C", "shell_G", "shell_init_frac", "ambient_c",
         "tau_sensor"),
        {"emissivity": 0.0, "h_flow": 0.0, "n_flow": 1.0},
        "EM model v8: gas-exit right BC — right end face loses heat to the N2 exit temperature "
        "(≈ wall temperature when h_gw is large) instead of ambient. "
        "Physically: the gas exits at near-wall temperature, so the right end is effectively "
        "adiabatic when flow is on. Targets the 18-19 C right-end RMSE from v3. "
        "19 active params; sigma_sus in [100,300]; fitted on test5/6/7, target RMSE<5 C.",
        use_conduction=True, use_radiation=False, use_gas_advection=True, use_shell=True,
        use_end_loss=True, use_em_power=True, use_sensor_lag=True, use_gas_exit_bc=True),
}


# --------------------------------------------------------------------------- #
# Stable backward-Euler rollout simulation
# --------------------------------------------------------------------------- #
try:
    from scipy.linalg import solve_banded as _solve_banded
except Exception:  # pragma: no cover
    _solve_banded = None


def _solve_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (a sub, b diag, c super, d rhs).

    Row i: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]; a[0] and c[-1] unused.
    Uses scipy solve_banded if available, else the Thomas algorithm.
    """
    if _solve_banded is not None:
        n = b.size
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]   # super-diagonal
        ab[1, :] = b         # diagonal
        ab[2, :-1] = a[1:]   # sub-diagonal
        return _solve_banded((1, 1), ab, d)
    n = b.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def simulate_experiment(
    exp: ExperimentData,
    params: ThermalParams,
    variant: ModelVariant,
    max_substep_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll the model forward over a run from the measured initial profile.

    Returns (predicted_temps (T,N), axial_power_w_m (T,N)). Conduction and
    linearised losses are backward Euler; radiation, gas exchange and heating
    are explicit.
    """
    x = exp.x_m
    n = x.size
    widths = control_volume_widths(x)
    vol = A_CROSS_M2 * widths            # m^3 per node
    surf = PERIMETER_M * widths          # m^2 lateral per node

    rho_cp = params.get("rho_cp")
    cp_T1 = params.get("cp_T1")
    k_w = params.get("k_w")
    kT1 = params.get("kT1")
    h0 = params.get("h0")
    h_flow = params.get("h_flow")
    n_flow = params.get("n_flow")
    h_T = params.get("h_T")
    emis = params.get("emissivity") if variant.use_radiation else 0.0
    power_scale = params.get("power_scale")
    h_gw = params.get("h_gw")
    inlet_c = params.get("inlet_c")
    t_amb = params.get("ambient_c")
    shell_C = params.get("shell_C")
    shell_G = params.get("shell_G")
    shell_G_T1 = params.get("shell_G_T1")
    shell_init_frac = params.get("shell_init_frac")
    h_end_left = params.get("h_end_left") if variant.use_end_loss else 0.0
    h_end_right = params.get("h_end_right") if variant.use_end_loss else 0.0
    end_flow_gain = params.get("end_flow_gain")
    end_flow_gain_right = params.get("end_flow_gain_right")
    h_end_T = params.get("h_end_T")
    t_amb_lat = params.get("ambient_lat_c")   # lateral surface ambient (may equal t_amb)
    left_sink_c = inlet_c if (variant.use_end_loss and "inlet_c" in params.values) else t_amb
    ps_T1 = params.get("ps_T1")
    # right end-cap thermal mass
    if variant.use_rend:
        rend_C = params.get("rend_C")
        rend_G = params.get("rend_G")         # end-cap → ambient
        rend_G_sus = params.get("rend_G_sus") # susceptor → end-cap
    else:
        rend_C = rend_G = rend_G_sus = 0.0
    # left end-cap thermal mass
    if variant.use_lend:
        lend_C = params.get("lend_C")
        lend_G = params.get("lend_G")         # left end-cap → ambient
        lend_G_sus = params.get("lend_G_sus") # susceptor left face → left end-cap
    else:
        lend_C = lend_G = lend_G_sus = 0.0
    ps_T2 = params.get("ps_T2")  # quadratic efficiency coefficient (default 0)

    # Bessel EM power model
    if variant.use_em_power:
        sigma_sus = params.get("sigma_sus")
        sigma_T1_val = params.get("sigma_T1")
        # coupling factor at the reference conductivity; normalises the T dependence
        _f_em_ref = float(_em_coupling_factor(sigma_sus))
    else:
        sigma_sus = 0.0
        sigma_T1_val = 0.0
        _f_em_ref = 1.0

    # face distances for conduction
    dface = np.diff(x)                   # length n-1
    gas_mdot_cp = N2_RHO_STP_KG_M3 * N2_CP_J_KG_K * SLPM_TO_M3_S  # per SLPM [W/K]

    # ── coil gain array ──────────────────────────────────────────────────
    if "coil_mid_gain" in params.values:
        g_mid = params.get("coil_mid_gain")
        g_end = params.get("coil_end_gain") if "coil_end_gain" in params.values else 1.0
        gain = np.array([g_end, g_mid, g_mid, g_mid, g_mid, g_end])
    else:
        gain = None   # use raw currents
    # ── optional coil assembly offset (recompute Biot-Savart basis) ──────
    if "coil_offset_m" in params.values:
        co = params.get("coil_offset_m")
        _basis = coil_field_basis(x + co) if abs(co) > 1e-7 else exp.field_basis
    else:
        _basis = exp.field_basis
    if gain is not None or "coil_offset_m" in params.values:
        _g = gain if gain is not None else np.ones(N_COILS)
        b_field = (exp.currents_a * _g) @ _basis
        b2_use = b_field * b_field
    else:
        b2_use = exp.b2
    if variant.use_em_power:
        #   p_axial = 2*(omega/mu0) * Im(J2(z)/J0(z)) * B^2 * pi*R^2   [W/m]
        # sigma_sus sets the amplitude through _f_em_ref; power_gain is an
        # optional global correction. sigma(T) is applied per node in the substep loop.
        _pg = params.get("power_gain") if "power_gain" in params.values else 1.0
        gen_line = _pg * _EM_BASE_W_PER_M_T2 * _f_em_ref * b2_use  # (T,N) W/m
    else:
        gen_line = power_scale * (b2_use / B2_REF_T2)    # (T,N) W/m
    # Optional axial tilt of the power deposition profile.
    if "power_axial_tilt" in params.values:
        tilt = params.get("power_axial_tilt")
        if tilt != 0.0:
            x_norm = x / x[-1] - 0.5              # shape (N,) in [-0.5, +0.5]
            ax_fac = np.maximum(1.0 + tilt * x_norm, 0.0)   # (N,)
            gen_line = gen_line * ax_fac           # (T,N) broadcast over nodes
    qgen = gen_line * widths                         # W per node

    temps = exp.temps_c
    time_s = exp.time_s

    # Optional first-order lag on delivered power (RF generator bandwidth,
    # coupling lag), per node.
    if "tau_power" in params.values:
        tau_p = params.get("tau_power")
        if tau_p > 0.5:
            qgen_lagged = qgen.copy()
            for k in range(1, time_s.size):
                dt_k = time_s[k] - time_s[k - 1]
                if dt_k > 0.0:
                    w = np.exp(-dt_k / tau_p)
                    qgen_lagged[k] = w * qgen_lagged[k - 1] + (1.0 - w) * qgen[k]
            qgen = qgen_lagged

    T = temps[0].copy()
    pred = np.empty_like(temps)
    pred[0] = T

    t_amb_k = t_amb_lat + KELVIN

    # Insulation shell: lateral loss goes to T_sh (T_sh = ambient if disabled).
    if variant.use_shell:
        T_sh = t_amb + shell_init_frac * (float(np.mean(T)) - t_amb)
    else:
        T_sh = t_amb

    # end-caps start at the end-node temperatures
    T_rend = float(T[-1]) if variant.use_rend else t_amb
    T_lend = float(T[0]) if variant.use_lend else t_amb

    for k in range(1, time_s.size):
        dt_total = time_s[k] - time_s[k - 1]
        if not np.isfinite(dt_total) or dt_total <= 0:
            pred[k] = T
            continue
        n_sub = max(1, int(np.ceil(dt_total / max_substep_s)))
        h_sub = dt_total / n_sub
        flow = 0.5 * (exp.flow_slpm[k] + exp.flow_slpm[k - 1])
        qgen_k = 0.5 * (qgen[k] + qgen[k - 1])

        for _ in range(n_sub):
            # T-dependent properties
            cp_fac = np.clip(1.0 + cp_T1 * (T - TREF_C), 0.2, 5.0)
            C = np.maximum(rho_cp * cp_fac * vol, 1e-3)        # J/K
            k_fac = np.clip(1.0 + kT1 * (T - TREF_C), 0.2, 20.0)
            k_node = np.maximum(k_w * k_fac, 1e-4)

            # face conductances [W/K]
            if variant.use_conduction:
                k_face = 2.0 * k_node[:-1] * k_node[1:] / (k_node[:-1] + k_node[1:])
                Gf = k_face * A_CROSS_M2 / dface               # length n-1
            else:
                Gf = np.zeros(n - 1)

            # convective/forced loss coefficient [W/m^2/K]
            h = h0 + h_flow * (flow ** n_flow) + h_T * np.maximum(T - t_amb_lat, 0.0)
            h = np.maximum(h, 0.0)
            Gloss = h * surf                                   # W/K

            # explicit radiation [W]
            qrad = emis * SIGMA_SB * surf * ((T + KELVIN) ** 4 - t_amb_k ** 4)

            # explicit gas advection [W] (upwind, left->right)
            qgas = np.zeros(n)
            Tg_exit = t_amb   # gas exit temp; ambient without advection
            if variant.use_gas_advection and flow > 1e-9:
                mdot_cp = gas_mdot_cp * flow                   # W/K
                Tg_prev = inlet_c
                for i in range(n):
                    # mdot_cp*(Tg_i - Tg_prev) = h_gw*surf_i*(T_i - Tg_i)
                    g = h_gw * surf[i]
                    Tg_i = (mdot_cp * Tg_prev + g * T[i]) / (mdot_cp + g)
                    qgas[i] = g * (T[i] - Tg_i)
                    Tg_prev = Tg_i
                Tg_exit = float(Tg_prev)   # leaving the last node

            # assemble tridiagonal backward-Euler system for T_new
            diag = np.empty(n)
            sub = np.zeros(n)
            sup = np.zeros(n)
            rhs = np.empty(n)
            inv = 1.0 / C
            diag[:] = 1.0 / h_sub + Gloss * inv
            if variant.use_conduction:
                diag[:-1] += Gf * inv[:-1]
                diag[1:] += Gf * inv[1:]
                sub[1:] = -Gf * inv[1:]
                sup[:-1] = -Gf * inv[:-1]
            # temperature-dependent induction efficiency
            if variant.use_em_power:
                # eta = f_em(sigma(T)) / f_em(sigma_sus)
                if sigma_T1_val != 0.0:
                    sigma_T = sigma_sus * np.clip(
                        1.0 + sigma_T1_val * (T - TREF_C), 0.01, 100.0)
                    f_em_T = _em_coupling_factor(sigma_T)   # (N,) array
                    eta = f_em_T / _f_em_ref
                else:
                    eta = np.ones(n)   # constant sigma
                qgen_eff = qgen_k * np.clip(eta, 0.05, 10.0)
            else:
                # empirical linear (+ optional quadratic) efficiency
                dT_ref = T - TREF_C
                if ps_T2 != 0.0:
                    eta = 1.0 + ps_T1 * dT_ref + ps_T2 * dT_ref * dT_ref
                else:
                    eta = 1.0 + ps_T1 * dT_ref
                qgen_eff = qgen_k * np.clip(eta, 0.05, 10.0)
            # lateral loss to T_sh (ambient if no shell)
            rhs[:] = (
                T / h_sub
                + (qgen_eff - qrad - qgas + Gloss * T_sh) * inv
            )
            # implicit end-face losses: left to inlet temp, right to ambient, gas exit or end-cap
            if variant.use_end_loss:
                t_fac_l = 1.0 + h_end_T * max(float(T[0]) - TREF_C, 0.0)
                t_fac_r = 1.0 + h_end_T * max(float(T[-1]) - TREF_C, 0.0)
                g_left = h_end_left * (1.0 + end_flow_gain * flow) * t_fac_l
                g_right = h_end_right * (1.0 + end_flow_gain_right * flow) * t_fac_r
                if variant.use_rend:
                    right_sink = T_rend
                elif variant.use_gas_exit_bc:
                    right_sink = Tg_exit
                else:
                    right_sink = t_amb
                # susceptor to right end-cap conduction
                if variant.use_rend:
                    diag[-1] += rend_G_sus * inv[-1]
                    rhs[-1] += rend_G_sus * T_rend * inv[-1]
                diag[0] += g_left * inv[0]
                rhs[0] += g_left * left_sink_c * inv[0]
                # susceptor to left end-cap conduction
                if variant.use_lend:
                    diag[0] += lend_G_sus * inv[0]
                    rhs[0] += lend_G_sus * T_lend * inv[0]
                diag[-1] += g_right * inv[-1]
                rhs[-1] += g_right * right_sink * inv[-1]
            T = _solve_tridiag(sub, diag, sup, rhs)
            T = np.clip(T, -50.0, 4000.0)

            if variant.use_shell:
                # backward-Euler scalar update of the shell using the new wall temps
                gsum = float(np.sum(Gloss))
                tsum = float(np.sum(Gloss * T))
                a_sh = shell_C / h_sub
                shell_G_eff = shell_G * float(np.clip(1.0 + shell_G_T1 * (T_sh - TREF_C), 0.1, 10.0))
                T_sh = (a_sh * T_sh + tsum + shell_G_eff * t_amb) / (a_sh + gsum + shell_G_eff)

            if variant.use_rend:
                # backward-Euler update of the right end-cap: heated by the last node, cools to ambient
                a_re = rend_C / h_sub
                T_rend = (a_re * T_rend + rend_G_sus * float(T[-1]) + rend_G * t_amb) / (
                    a_re + rend_G_sus + rend_G)

            if variant.use_lend:
                # backward-Euler update of the left end-cap: heated by the x=0 node, cools to ambient
                a_le = lend_C / h_sub
                T_lend = (a_le * T_lend + lend_G_sus * float(T[0]) + lend_G * t_amb) / (
                    a_le + lend_G_sus + lend_G)
        pred[k] = T

    # Fiber-optic sensor lag, applied after the rollout (wall dynamics unchanged):
    #   dT_s/dt = (T_wall - T_s) / tau_sensor
    # discretised as T_s[k] = w*T_s[k-1] + (1-w)*T_wall[k], w = exp(-dt/tau)
    if variant.use_sensor_lag:
        tau_sensor = params.get("tau_sensor")
        _use_tau_right = "tau_sensor_right" in params.values
        _tau_right = params.get("tau_sensor_right") if _use_tau_right else tau_sensor
        # 195 mm (outlet) node
        _idx_195 = int(np.argmin(np.abs(exp.x_m - 0.195))) if _use_tau_right else -1
        if tau_sensor > 0.5:
            T_sens = pred.copy()
            for k in range(1, time_s.size):
                dt_k = time_s[k] - time_s[k - 1]
                if dt_k > 0.0:
                    w = np.exp(-dt_k / tau_sensor)
                    T_sens[k] = w * T_sens[k - 1] + (1.0 - w) * pred[k]
                    if _use_tau_right:
                        # 195 mm column uses its own time constant
                        w_r = np.exp(-dt_k / _tau_right)
                        T_sens[k, _idx_195] = (
                            w_r * T_sens[k - 1, _idx_195]
                            + (1.0 - w_r) * pred[k, _idx_195]
                        )
            pred = T_sens

    return pred, gen_line


# --------------------------------------------------------------------------- #
# Residuals + metrics
# --------------------------------------------------------------------------- #
def residuals_for_variant(
    x: np.ndarray,
    variant: ModelVariant,
    experiments: Sequence[ExperimentData],
    sensor_noise: Mapping[str, np.ndarray],
    weight_by_noise: bool = True,
    reg_weight: float = 0.0,
    max_substep_s: float = 12.0,
) -> np.ndarray:
    params = variant.decode(x)
    chunks: List[np.ndarray] = []
    for exp in experiments:
        pred, _ = simulate_experiment(exp, params, variant, max_substep_s=max_substep_s)
        mask = exp.observed_mask
        resid = (pred[:, mask] - exp.temps_c[:, mask])
        if weight_by_noise:
            sig = sensor_noise.get(exp.name)
            if sig is not None:
                resid = resid / sig[mask][None, :]
        chunks.append(resid.reshape(-1) / np.sqrt(max(resid.size, 1)))
    res = np.concatenate(chunks) if chunks else np.zeros(1)
    if reg_weight > 0:
        penalties = []
        for n in ("cp_T1", "kT1", "h_T"):
            if n in variant.active_names:
                penalties.append(reg_weight * params.get(n) / max(PARAMETER_SPECS[n].upper, 1e-9))
        if penalties:
            res = np.concatenate([res, np.array(penalties)])
    return res


def rollout_metrics(
    exp: ExperimentData,
    pred: np.ndarray,
    sensor_noise: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    mask = exp.observed_mask
    meas = exp.temps_c[:, mask]
    pr = pred[:, mask]
    err = pr - meas
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rng = float(np.nanmax(meas) - np.nanmin(meas))
    nrmse = float(rmse / rng) if rng > 0 else float("nan")
    # steady-state error: last 5% of samples
    tail = max(1, int(0.05 * meas.shape[0]))
    ss_err = float(np.mean(np.abs(err[-tail:])))
    # time-constant error: time to reach 63% of the total change
    tc_meas = _time_constant(exp.time_s, np.mean(meas, axis=1))
    tc_pred = _time_constant(exp.time_s, np.mean(pr, axis=1))
    tc_err = float(abs(tc_pred - tc_meas)) if np.isfinite(tc_meas) and np.isfinite(tc_pred) else float("nan")
    out = {
        "rmse_c": rmse,
        "mae_c": mae,
        "nrmse": nrmse,
        "steady_state_mae_c": ss_err,
        "tau_meas_s": tc_meas,
        "tau_pred_s": tc_pred,
        "tau_err_s": tc_err,
        "max_abs_err_c": float(np.max(np.abs(err))),
    }
    if sensor_noise is not None:
        out["rmse_over_noise"] = float(rmse / max(np.mean(sensor_noise[mask]), 1e-6))
    return out

def _time_constant(time_s: np.ndarray, signal: np.ndarray) -> float:
    s0, s1 = signal[0], signal[-1]
    if abs(s1 - s0) < 1e-6:
        return float("nan")
    target = s0 + 0.632 * (s1 - s0)
    if s1 > s0:
        idx = np.argmax(signal >= target)
    else:
        idx = np.argmax(signal <= target)
    return float(time_s[idx]) if idx > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Persistence + I/O helpers
# --------------------------------------------------------------------------- #
def save_params_json(path: Path, variant_name: str, params: ThermalParams, extra: dict) -> None:
    rows = []
    for name, val in params.values.items():
        spec = PARAMETER_SPECS.get(name)
        rows.append({
            "name": name,
            "value": float(val),
            "unit": spec.unit if spec else "",
            "lower": spec.lower if spec else None,
            "upper": spec.upper if spec else None,
            "interpretation": spec.interpretation if spec else "",
        })
    payload = {"variant": variant_name, "parameters": rows, **extra}
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_params(path: Path) -> Tuple[str, ThermalParams]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vals = {row["name"]: float(row["value"]) for row in data["parameters"]}
    return data["variant"], ThermalParams(vals)
