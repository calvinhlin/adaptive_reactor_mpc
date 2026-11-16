# MPC for a six-coil induction-heated RWGS reactor

Code for the model predictive controllers run on a six-coil induction-heated
reverse water-gas shift (RWGS) reactor:

- an economic MPC that schedules flow and coil power against a real-time
  electricity price, and
- a solar-tracking MPC that holds the reactor at a temperature setpoint using
  a variable solar power budget.

Each controller has a hardware driver and a digital twin that runs the same
controller on the grey-box thermal model.

## Files

### Hardware drivers
| File | Description |
| --- | --- |
| `supermode_eMPC.py` | Economic MPC on the reactor. |
| `supermode_solar_MPC.py` | Solar-tracking MPC on the reactor (450 °C setpoint). |
| `supermode_PRBS.py` | Schedule-driven logging loop used for system identification. Also provides the instrument settings and helpers used by both MPC drivers. |

The drivers need lab instruments and in-house instrument modules
(`instruments.alicat`, `hyperion_temp_logger_siggen`) that are not in this
repository, so they will not run without that hardware. They are included to
document the control loops used in the experiments.

### Models and digital twins
| File | Description |
| --- | --- |
| `thermal_greybox.py` | 1-D axial grey-box thermal model of the susceptor, with the Biot–Savart coil heating basis. |
| `rwgs_pfr_model.py` | 1-D plug-flow reactor model for RWGS; fits the Arrhenius parameters in `rwgs_pfr_fit_results.json`. |
| `empc_digital_twin.py` | Economic-MPC digital twin: operating modes, EM power model, PFR kinetics, MPC planner and closed-loop simulation over the price trace. Used by both hardware drivers. |
| `tracking_mpc_solar_digital_twin.py` | Solar-tracking MPC digital twin, 450 °C setpoint. Also generates the solar trace used by the solar hardware driver. |
| `tracking_mpc_solar_digital_twin_T500.py` | Solar-tracking MPC digital twin, 500 °C setpoint. |

### Input data
| File | Description |
| --- | --- |
| `thermal_fit_em_v3_coil29_best.json` | Fitted thermal model parameters. |
| `isothermal_opt_450C_window.json`, `isothermal_opt_500C_window.json` | Steady-state axial temperature profiles at 450 °C and 500 °C peak for each flow rate, used to compute the CO production rate of each operating mode. |
| `rwgs_pfr_fit_results.json` | Fitted RWGS kinetic parameters. |
| `LZ_South_Nov5_RealTime.xlsx` | ERCOT LZ_SOUTH real-time electricity prices used by the economic MPC. |

The economic MPC uses a CO price of $0.15/kg (`CO_PRICE_USD_PER_KG` in
`empc_digital_twin.py`).

## Installation

Tested with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The digital twins run from the data in this repository. From the repository
root:

```bash
# Economic MPC
python empc_digital_twin.py --no-show

# Solar-tracking MPC, 450 °C and 500 °C setpoints
python tracking_mpc_solar_digital_twin.py --no-show
python tracking_mpc_solar_digital_twin_T500.py --no-show
```

`--no-show` skips the live animation and only saves the final figure to
`plots/`. See the `argparse` block at the bottom of each script for other
options.

`rwgs_pfr_model.py` needs the steady-state conversion data
(`final_steady_state.xlsx`), which is not included; its fitted output is
provided in `rwgs_pfr_fit_results.json`. Re-fitting the thermal model likewise
needs the raw experiment logs (`*_excel_schedule.xlsx`); the fitted parameters
are provided in `thermal_fit_em_v3_coil29_best.json`.

## License

MIT. See [LICENSE](LICENSE).
