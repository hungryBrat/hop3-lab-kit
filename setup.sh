#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv || { echo "need: sudo apt install python3-venv python3-tk"; exit 1; }
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q ftservo-python-sdk==2.0.1 pyserial "numpy<2" "matplotlib<3.10" scipy opencv-python-headless
.venv/bin/python -c "from scservo_sdk import sms_sts; import numpy, matplotlib, scipy, cv2; print('SETUP OK')"
.venv/bin/python -c "import tkinter" 2>/dev/null || echo "no tkinter (live dashboard window needs it): sudo apt install python3-tk, then rerun setup.sh"
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "no serial device yet - plug the arm in"
groups | grep -q dialout || echo "NOT in dialout: sudo usermod -aG dialout $USER (relogin) OR sudo chmod a+rw /dev/ttyACM0"
systemctl is-active brltty.service >/dev/null 2>&1 && echo "brltty active: sudo systemctl mask --now brltty-udev.service brltty.service"
true
