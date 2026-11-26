#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m venv --system-site-packages .venv || { echo "need: sudo apt install python3-venv"; exit 1; }
.venv/bin/pip install -q ftservo-python-sdk pyserial "numpy<2" matplotlib || .venv/bin/pip install -q ftservo-python-sdk pyserial
.venv/bin/python -c "from scservo_sdk import sms_sts; import numpy, matplotlib; print('SETUP OK')"
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "no serial device yet - plug the arm in"
groups | grep -q dialout || echo "NOT in dialout: sudo usermod -aG dialout $USER (relogin) OR sudo chmod a+rw /dev/ttyACM0"
systemctl is-active brltty.service >/dev/null 2>&1 && echo "brltty active: sudo systemctl mask --now brltty-udev.service brltty.service"
true
