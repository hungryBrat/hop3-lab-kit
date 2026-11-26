# HOP-3 robot session cheatsheet (runbook order)

Run everything from `HOP_3/`. `P=.venv/bin/python; C=code/python`. Add `--mock` to any command to rehearse with no hardware: mock outputs go to `logs/mock/` and never touch the real `config.json` or `session_data.json`.
Before leaving home, run `$P $C/test_exec.py`. It takes about 2 min and must end `48/48 passed`.

## 0. Plug in (min 0)
```
lsusb ; dmesg | tail ; ls /dev/ttyACM* /dev/ttyUSB*
$P $C/probe.py                      # or --port /dev/ttyUSB0
```
Expected: the port table, then ids 1-4 `ok`, counts/deg/θ, reg31 offsets, `PROBE OK`. Writes `logs/probe_<ts>.csv` and session `port` + `servo_offsets_reg31`.

| Symptom | Fix |
|---|---|
| no ttyACM/ttyUSB, but `lsusb` shows 1a86:7523 | brltty: `sudo systemctl mask brltty-udev.service brltty.service; sudo systemctl stop brltty-udev brltty`, then replug |
| nothing in `lsusb` | cable/USB-C, adapter power, jumper in position **A** |
| `Permission denied` | `groups` must list dialout (re-login). Quick fix: `sudo chmod 666 /dev/ttyACM0` |
| port opens, pings `NO` | PSU on? Voltage label matches the servos (12 V vs 7.4 V)? Check the daisy-chain cable into the first silent servo. Baud must be 1M. An ESP32 driver board needs serial-forwarding at 115200 (then the port/baud differs: ask the TA) |
| some IDs answer, sync read incomplete | cable/power fault at the first non-answering ID; reseat it |
| config port differs | pass `--port <dev>` to every tool (session `port` is used automatically after probe) |

Check the PSU label → session notes. Open MATLAB with the net loaded **now**.

## 1. Convention test (min 10)
```
$P $C/convention_test.py            # slow 5 deg/s, torque limit 300
```
Every move asks `y/N`; answer N to stop. The prompts, in order: neutral → enter the measured tip height (DH 398.6) → s2→200 (tape the +X0 arrow) → neutral → s1→110 "CCW from above?" → neutral → s3→200 "toward +X0?" → neutral.
Writes session `sigma1`, `sigma3`, `neutral_tip_height_mm`. If the signs differ from the config, it asks before editing `config.json`. Then:
```
$P $C/retarget_cmds.py data/cmd_*.csv      # FAIL rows = file invalid under the new map, left untouched
```
- `DECISION: sigma3 = -1` → **primary** family.
- `sigma3 = +1` → **backup** `data/cmd_backup.csv` (primary files will FAIL retarget; expected). Note the reason for the report.
- Arm moves the wrong way or anything looks off → kill criterion: stop, don't push through.

## 2. Touch-off (min 18)
```
$P $C/touchoff.py
```
Torque goes OFF on 1-3, so **hold the arm**. Enter 3× per F1..F4, approaching each point from the same side. At the end, guide the arm back near upright and press Enter: it sets goal = present, then torque on (no jump). Writes `logs/touchoff_<ts>.csv` and session `touchoff_log`. MATLAB `touchoff_fit` → plane z → pick or generate the z_c file → coordinator updates `z_floor_mm`.

## 3. Runs (min 30)
```
$P $C/safety.py data/cmd_<name>.csv                      # PASS required
$P $C/run_traj.py data/cmd_<name>.csv --mode stream --dry   # read-only: pings, offsets, approach-path check
$P $C/run_traj.py data/cmd_inair_zNNN.csv --mode stream --torque-limit 400 --dashboard
$P $C/run_traj.py data/cmd_primary_zNNN.csv --mode stream --dashboard     # pen down, x2
```
Expected end line: `log: logs/run_<name>_<ts>.csv rows=1001 achieved rate=~50 Hz dropped replies=<small>`.
ESC or Ctrl-C holds the arm in place with torque on and closes the log (exit 2, no neutral return).

| Symptom | Fix |
|---|---|
| `SAFETY FAIL ... [map]` | config signs changed → run `retarget_cmds.py` |
| `SAFETY FAIL ... [zfloor]` | the circle is below `z_floor_mm`: pick a higher z_c, or the coordinator lowers the floor after touch-off |
| `approach path ... unsafe` | the arm rests outside the safe box: torque off, hand-guide it near neutral, retry |
| achieved rate < 45 Hz, many dropped replies, or the arm stutters | rerun with `--mode wait` (lab-script pattern, ~90 points) |
| `consecutive failed sync reads` | bus died: check power/cable, rerun `probe.py` |
| `OFFSETS (reg 31) CHANGED` (exit 3) | something recalibrated a servo: stop, tell the team, the calibration is invalid |
| dashboard lags or crashes | ignore it; the CSV log is the record. Drop `--dashboard` if needed |

## 4. Close-out
Neutral (runs end there), power off. Copy `logs/` (not `logs/mock/`), photos and video to Drive + a pen drive, and open one file from each copy to check it.
