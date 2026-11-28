# Tutorial: trace the letter B on a fresh Ubuntu laptop

This guide takes a clean Ubuntu laptop to the point where the arm traces the B and the trace is checked against our numbers. Steps 1–5 need only Python. Steps 6–7 need MATLAB.

Tested on Ubuntu 24.04 with Python 3.12 and MATLAB R2026b.

## 1. Install (5 min)

```bash
sudo apt install git python3-venv python3-tk
git clone https://github.com/hungryBrat/hop3-lab-kit.git
cd hop3-lab-kit
bash setup.sh            # must print SETUP OK
P=.venv/bin/python ; C=code/python
```

Run every command below from the repo root, with `P` and `C` set as above (set them again in each new terminal).

If `setup.sh` says you are not in `dialout`, run `sudo usermod -aG dialout $USER`, then log out and back in.

## 2. Check the install with no robot (3 min)

```bash
$P $C/test_exec.py                                                    # must end 70/70 passed
$P $C/safety.py data/cmd_letterB.csv                                  # must print PASS
$P $C/run_traj.py data/cmd_letterB.csv --mock --decim 10 --dashboard  # simulated run
```

The last command opens the live figure and draws the B on a simulated arm. Close the window when the run ends. Mock logs go to `logs/mock/` and never touch the real session files.

## 3. Connect the arm

1. Power the servo board (check that the PSU voltage matches the servo label), then plug in the USB cable.
2. `ls /dev/ttyACM* /dev/ttyUSB*` should show a port, usually `/dev/ttyACM0`.
   - If `lsusb` shows the adapter (`1a86:55d3` or `1a86:7523`) but no port appears, brltty has taken it. Run `sudo systemctl mask --now brltty-udev.service brltty.service` and replug.
   - If you get `Permission denied`, run `sudo chmod a+rw /dev/ttyACM0`.
3. Probe:
   ```bash
   $P $C/probe.py                    # add --port /dev/ttyUSB0 if the port differs
   ```
   IDs 1–3 must say `ok` and the probe must end with `PROBE OK`. ID 4 shows `dead/ignored`, which is expected. The probe also records the port and the servo offsets in `session/session_data.json`.

## 4. Confirm the sign convention (2 min, always on a new arm or laptop)

Our arm read reg31 offsets of **522 / −455 / −162** (IDs 1/2/3). If yours differ, it is another arm or it was recalibrated. Run this test in either case: it is cheap, and a wrong sign would send the arm off the B.

```bash
$P $C/convention_test.py           # answer y/N before each small move
$P $C/retarget_cmds.py data/cmd_letterB.csv
```

The test makes two small moves and asks what you saw:
- s1 → 110°: did the base turn counter-clockwise seen from above? Yes means σ1 = +1. We measured σ1 = −1.
- s3 → 200°: did the forearm tilt toward the front (+X0)? Yes means σ3 = −1. We measured σ3 = −1.

`retarget_cmds.py` rewrites the command file only if the signs changed. With our arm it reports 0 rows changed, and it must end `OK`.

## 5. Trace the B

Clear the space in front of the arm and keep a hand on the power switch. The tip moves in the plane x = 260 mm (in front of the base), with the B centred 84 mm above the J2 axis. A run lasts about 36 s, plus a slow approach and a slow return to neutral.

1. Dry run (reads the bus, sends no moves):
   ```bash
   $P $C/run_traj.py data/cmd_letterB.csv --mode stream --dry
   ```
2. Real run with the live figure. Start a phone video first if you want one.
   ```bash
   $P $C/run_traj.py data/cmd_letterB.csv --mode stream --decim 10 --dashboard
   ```
   - Add `--torque-limit 400` to the first run if you want it gentler.
   - `--decim 10` is required for a smooth trace. It sends a goal every 0.2 s with each joint's speed matched to the motion. Plain 50 Hz streaming makes the servos stick-slip and the B jitters.
   - In the front view, black is the planned B, orange is what the NN predicts for the commanded joints, and blue is the B the arm actually traces (DH of the measured encoder angles).
   - ESC or Ctrl-C holds the arm where it is.
3. The log lands in `logs/run_letterB_<timestamp>.csv`. Repeat step 2 for more runs.

If the arm hesitates or stutters, try `--mode wait`. If the approach is refused (`approach path unsafe`), turn the torque off, move the arm by hand near neutral (90, 180, 180), and retry.

## 6. Analyse the trace (MATLAB, any toolbox set)

1. In `session/session_data.json`, set `run_logs` to the new CSVs:
   ```json
   "run_logs": ["logs/run_letterB_20261205_101500.csv", "logs/run_letterB_20261205_101620.csv"]
   ```
2. In MATLAB, from the repo root:
   ```matlab
   addpath code/matlab
   run_all('post')
   ```
   This writes `results/run_letterB*.json` and `results/repeatability_letterB.json`, plus `figures/run_letterB_front.png`, `_joints`, `_z` and `_errors`. The front plot shows the desired B, the NN prediction, DH of the commanded angles and DH of the measured angles.
3. Compare against our runs: contour RMS should be about 1.3 mm, and the shape should match `figures/reference/run_letterB_front.png`. A different arm may differ a little because of its gearbox play and J2 sag.

To check this step before touching the robot, leave `run_logs` as shipped. `run_all('post')` should then print `repeatability letterB: contour RMS 1.290 +- 0.045 mm over 3 runs`.

## 7. Rebuild the network and the B from scratch (optional, needs Deep Learning Toolbox)

The shipped `models/` and `data/cmd_letterB.csv` are what we ran. To regenerate them:

```matlab
addpath code/matlab
run_all('pre')   % DH test, dataset, network sweep, fitnet [40 20] trainlm (5 seeds), eval, B trajectory, tests
```

The sweep trains many networks and takes a while. To retrain only the final network and regenerate the B:

```matlab
make_dataset(); train_fk([40 20], 'trainlm', 5); eval_fk(load_weights()); gen_trajectories();
```

Then rerun steps 2 and 5. Quick check of the trained model against the DH ground truth:

```matlab
addpath models
S = load('models/Forward_Kinematics_Model.mat');
q = [30; 45; -60];
[S.net(q)'; fkNet(q)'; dh_fk(q)']    % NN, NN without toolbox, DH: all three rows match
```

## Frame and conventions

- d1 = 0: the origin is on the J2 axis and z is height above it. The table is at z = −65.7 mm, and the safety floor is z = −45.7 mm.
- DH links: L2 = 136.5 mm, L3 = 97.6 mm, f = 98.8 mm (to the fixed gripper tip).
- Servo to DH map: θ1 = −(s1 − 90), θ2 = 270 − s2, θ3 = 180 − s3 (degrees).
- Training region: θ1 ±60°, θ2 10…125°, θ3 −125…−10°.
- The safety layer refuses any command outside the joint limits (5° margin), any step over 2° per row, anything below the z floor, and any write to ID 4.

## Troubleshooting

| Symptom | Fix |
|---|---|
| no `/dev/ttyACM*` | replug the cable; see the brltty fix in step 3 |
| IDs 1–3 do not answer | PSU on, jumper on A, daisy-chain cable seated, 1 Mbaud |
| `SAFETY FAIL ... [map]` | the signs changed: rerun step 4 |
| `OFFSETS (reg 31) CHANGED` | a servo was recalibrated during the run: stop and redo steps 3–4 |
| visible jitter | use `--decim 10` |
| no dashboard window | `sudo apt install python3-tk`, then `bash setup.sh` again |
| `refusing to write servo id 4` | by design (dead gripper) |

If the main scripts break, `kit_capture.py` is a minimal fallback that needs only the servo SDK: `$P kit_capture.py probe`, `$P kit_capture.py play data/cmd_letterB.csv`.
