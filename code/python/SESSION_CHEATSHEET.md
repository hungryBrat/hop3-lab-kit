# HOP-3 robot session cheatsheet (runbook order) — letter B in air, gripper dead

Run everything from `HOP_3/` (or the kit root). `P=.venv/bin/python; C=code/python`. Add `--mock` to any command to rehearse with no hardware: mock outputs go to `logs/mock/` and never touch the real `config.json` or `session_data.json`.
Before leaving home, run `$P $C/test_exec.py`. It takes about 3 min and must end `70/70 passed`.

**Task:** the tip traces a capital **B** in the air in the vertical front plane x = 260 mm (spine y = −12.5, z 100→200, then top bump, then bottom bump, bumps toward +y = viewer's right seen from +X0). ~36 s per run. No pen, no paper, no touch-off, no fiducial sheet, no platform.

**Gripper (ID 4) is DEAD.** Leave it physically as it is: do not unplug it, do not power-cycle it, never command it. All tools only ping/command IDs 1–3 (`config.json`: `gripper_enabled=false`, `servo_ids_active=[1,2,3]`). Probe shows ID 4 as `dead/ignored`; that is expected and not a failure. The "tip" is the gripper fingertip as it sits now (f = 98.8 mm).

## 0. Plug in (min 0)
```
lsusb ; dmesg | tail ; ls /dev/ttyACM* /dev/ttyUSB*
$P $C/probe.py                      # or --port /dev/ttyUSB0
```
Expected: the port table, ids 1-3 `ok`, counts/deg/θ, reg31 offsets, `4  gripper: dead/ignored`, `PROBE OK`. Writes `logs/probe_<ts>.csv` and session `port` + `servo_offsets_reg31` (ids 1-3).

| Symptom | Fix |
|---|---|
| no ttyACM/ttyUSB, but `lsusb` shows 1a86:7523 | brltty: `sudo systemctl mask brltty-udev.service brltty.service; sudo systemctl stop brltty-udev brltty`, then replug |
| nothing in `lsusb` | cable/USB-C, adapter power, jumper in position **A** |
| `Permission denied` | `groups` must list dialout (re-login). Quick fix: `sudo chmod 666 /dev/ttyACM0` |
| port opens, ids 1-3 ping `NO` | PSU on? Voltage label matches the servos (12 V vs 7.4 V)? Check the daisy-chain cable into the first silent servo. Baud must be 1M. An ESP32 driver board needs serial-forwarding at 115200 (then the port/baud differs: ask the TA) |
| ID 4 `NO` / `dead/ignored` | expected (dead gripper). Do not touch it |
| some of 1-3 answer, sync read incomplete | cable/power fault at the first non-answering ID; reseat it (not the gripper) |
| config port differs | pass `--port <dev>` to every tool (session `port` is used automatically after probe) |

Check the PSU label → session notes.

## 1. Convention test (min 10)
```
$P $C/convention_test.py            # slow 5 deg/s, torque limit 300
```
Every move asks `y/N`; answer N to stop. Prompts in order: neutral → measured tip height (DH 398.6) → s2→200 (tape the +X0 arrow on the table) → neutral → s1→110 "CCW from above?" → neutral → s3→200 "toward +X0?" → neutral.
Writes session `sigma1`, `sigma3`, `neutral_tip_height_mm`. If the signs differ from the config, it asks before editing `config.json`. Then **always**:
```
$P $C/retarget_cmds.py data/cmd_letterB.csv data/cmd_letterB_outside.csv
```
`cmd_letterB.csv` is valid under **both** J3 signs: it must print OK; there is no backup trajectory. `cmd_letterB_outside.csv` is only valid for sigma3 = −1 (its θ3 reaches −125°): under sigma3 = +1 it prints `FAIL [limits]` and is left untouched → skip the outside run and note it. Arm moves the wrong way or anything looks off → stop, don't push through.

## 2. Letter-B runs, in air (min 15)
```
$P $C/safety.py data/cmd_letterB.csv                                   # PASS required
$P $C/run_traj.py data/cmd_letterB.csv --mode stream --dry             # read-only: pings, offsets, approach check
$P $C/run_traj.py data/cmd_letterB.csv --mode stream --torque-limit 400 --dashboard   # run 1
$P $C/run_traj.py data/cmd_letterB.csv --mode stream --dashboard       # runs 2 and 3 (repeatability)
$P $C/run_traj.py data/cmd_letterB_outside.csv --mode stream --dashboard   # part (f), outside region, in air
```
The run prints `segment 1/3 spine ... 2/3 top ... 3/3 bottom` as it goes. The dashboard shows the **front view** (y right, z up, as seen from +X0): desired B (black), NN-predicted (orange dashed), live DH(q_meas) (blue).
Expected end line: `log: logs/run_letterB_<ts>.csv rows=<n_rows> achieved rate=~50 Hz dropped replies=<small>`.
ESC or Ctrl-C holds the arm in place with torque on and closes the log (exit 2, no neutral return).
Stream stutters or rate < 45 Hz → `--mode wait` (lab-script pattern, ~90 points); fallback of the fallback: `$P kit_capture.py play data/cmd_letterB.csv`.

| Symptom | Fix |
|---|---|
| `SAFETY FAIL ... [map]` | config signs changed → run `retarget_cmds.py` (section 1) |
| `approach path ... unsafe` | the arm rests outside the safe box: torque off, hand-guide it near neutral, retry |
| achieved rate < 45 Hz, many dropped replies, or the arm stutters | rerun with `--mode wait` |
| `consecutive failed sync reads` | bus died: check power/cable, rerun `probe.py` |
| `OFFSETS (reg 31) CHANGED` (exit 3) | something recalibrated a servo: stop, tell the team, the calibration is invalid |
| `refusing to write servo id 4` | something tried to command the gripper: it is blocked by design; report it, carry on |
| dashboard lags or crashes | ignore it; the CSV log is the record. Drop `--dashboard` if needed |

## 3. Static accuracy, hand-guided poses, link lengths (min 30)
```
$P kit_capture.py goto S1 S2 S3 ; $P kit_capture.py pose ACC1      # x5 poses, commanded vs measured
$P kit_capture.py torque off       # SUPPORT THE ARM
$P kit_capture.py pose P01         # x~10 diverse poses; tape-measure tip x,y,z from base centre each time
$P kit_capture.py torque on
```
Tape: d1, L2, L3, gripper tip length f (fingertip as it sits now). Write everything on the paper checklist too.

## 4. Optional video ground truth
Bright orange/green/pink tape or sticker on the fingertip. Print `data/video_board.pdf` at 100 % (check the 100 mm bar), cut out the window, mount it vertically facing +X0 with its centre ticks at y = 0, z = 150, just **behind** the B plane (toward the robot) so the tip moves inside the window and the forearm passes through it without touching. Measure the gap d from the board face to x = 260 and note `board_offset_x_mm = -d` in session notes. Phone on a stand at +X0, 0.6–1 m away, facing the robot, whole board in frame, 60 fps if possible. Record **every** run; start recording before the run and tap the board edge once in view (sync check), keep the laptop screen in frame if possible. Afterwards: `$P $C/track_tip_video.py video/<file>.mp4 --run-log logs/run_<name>_<ts>.csv --colour <c> --board-offset -<d>`.

## 5. Close-out
Neutral (runs end there), torque off, power off. Copy `logs/` (not `logs/mock/`), `session/`, `photos/`, `video/` to Drive + a pen drive, and open one file from each copy to check it.
