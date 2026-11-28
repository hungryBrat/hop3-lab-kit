# HOP-3: neural-network forward kinematics for a 3-DOF ST3215 arm

BITS F327 (AI for Robotics), Hands-On Project 03, Option A, Team 11.

A MATLAB `fitnet [40 20]` network learns the DH forward kinematics of the course arm inside an elbow-up region. The network is then inverted with damped least squares on its own Jacobian. The result is a joint trajectory that makes the tip trace a capital **B** in the air, in the vertical plane x = 260 mm. That trajectory is streamed to the arm, the servo encoders are logged at 50 Hz, and the traced B is compared with the planned one.

**To reproduce it on a new Ubuntu laptop, follow [TUTORIAL.md](TUTORIAL.md).**

## Results on the reference arm (2026-11-27)

| Quantity | Value |
|---|---|
| NN test RMSE vs DH | 0.0009 mm per axis, R² = 1.000000 |
| Traced-B contour error (3 runs, `--decim 10`) | 1.29 ± 0.04 mm RMS |
| Lag | about 33 ms |
| Max deviation between repeats | 1.2 mm |

The remaining ~1.3 mm comes from about 0.4° of gravity sag at J2, which is a hardware limit. Reference plots are in `figures/reference/`.

## Layout

| Path | Contents |
|---|---|
| `config.json` | DH parameters (d1 = 0), joint limits, servo↔DH map, training region, trajectory spec |
| `code/python/` | Safety layer, servo bus, trajectory executor, mock bus, live dashboard, probe and convention tools, tests |
| `code/matlab/` | Dataset, training, evaluation, export, Jacobian, letter-B generator, DLS inversion, run analysis; `run_all.m` drives it all |
| `data/` | Training and test sets, `cmd_letterB.csv` (the joint trajectory that traces the B) |
| `models/` | Trained net (`Forward_Kinematics_Model.mat`) and toolbox-free exports (`fkNet.m`, `fk_weights.json/.mat`) |
| `logs/` | The three reported robot runs, the jitter experiments, probe logs |
| `results/` | Metrics JSON for the reported runs |
| `session/session_data.json` | Values measured at the robot (port, servo offsets, sign convention, run list) |
| `kit_capture.py`, `CHECKLIST.txt` | Minimal fallback tool and the original lab-session checklist |

Hardware: ST3215 servos IDs 1–3 on a 1 Mbaud bus over a CH343 USB adapter. The gripper servo (ID 4) is dead, so no tool ever commands it.
