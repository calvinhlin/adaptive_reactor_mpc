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
| `LZ_South_Nov5_RealTime.xlsx` | Electricity prices used by the economic MPC (see Data sources). |

The economic MPC uses a CO price of $0.15/kg (`CO_PRICE_USD_PER_KG` in
`empc_digital_twin.py`).

## System requirements

- Python 3.12 (tested with 3.12.9). Package versions are pinned in
  `requirements.txt`.
- Tested on Windows 11. The digital twins use only standard scientific Python
  packages and should run on Linux or macOS as well.
- A standard desktop or laptop computer. No GPU or other special hardware is
  needed for the digital twins. The hardware drivers need the reactor
  instruments described above.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installation takes about a minute on a standard desktop computer.

## Demo

From the repository root:

```bash
# Economic MPC
python empc_digital_twin.py --no-show

# Solar-tracking MPC, 450 °C and 500 °C setpoints
python tracking_mpc_solar_digital_twin.py --no-show
python tracking_mpc_solar_digital_twin_T500.py --no-show
```

`--no-show` skips the live animation and saves the final figures to `plots/`.
See the `argparse` block at the bottom of each script for other options.

Expected output and run time on a standard desktop computer:

| Script | Run time | Output in `plots/` | Expected printed results |
| --- | --- | --- | --- |
| `empc_digital_twin.py` | ~3.5 min | `empc_flow_vs_price.png`, `empc_digital_twin_24h.png` | CO produced 94.52 g, revenue $0.014, electricity cost $0.060, net profit −$0.046 |
| `tracking_mpc_solar_digital_twin.py` | ~10 s | `tracking_mpc_solar_24h.png` | T_max 440.0–453.7 °C, tracking RMSE 2.89 °C, MAE 2.52 °C, curtailed solar 0.033 kWh |
| `tracking_mpc_solar_digital_twin_T500.py` | ~10 s | `tracking_mpc_solar_T500_24h.png` | T_max 490.0–503.5 °C, tracking RMSE 4.52 °C, MAE 3.90 °C, curtailed solar 0.042 kWh |

The economic-MPC twin simulates 12 h, with the 24 h price trace compressed
into it (2.5 min per 5-min price interval). Its printed totals are labelled
"24-h" and cover that full price day. It also reports how many control steps
T_max spent outside the 450–500 °C window.

The solar trace is generated with a fixed random seed (`CLOUD_SEED`), so the
solar-tracking results are reproducible.

`rwgs_pfr_model.py` needs the steady-state conversion data
(`final_steady_state.xlsx`), which is not included; its fitted output is
provided in `rwgs_pfr_fit_results.json`. Re-fitting the thermal model likewise
needs the raw experiment logs (`*_excel_schedule.xlsx`); the fitted parameters
are provided in `thermal_fit_em_v3_coil29_best.json`.

## Data sources

`LZ_South_Nov5_RealTime.xlsx` contains Electric Reliability Council of Texas
(ERCOT) real-time settlement point prices for the LZ_SOUTH load zone on
5 November, at 5-min resolution (first sheet), with the day-ahead prices for
the same day (second sheet). ERCOT market data are publicly available from
ERCOT (https://www.ercot.com).

## License

MIT. See [LICENSE](LICENSE).
