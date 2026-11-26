"""Thin scservo_sdk wrapper (lab-script compatible) with the safety layer on every write.

Only SRAM registers 40 (torque 0/1), 41.. (acc/goal/speed) and 48 (torque limit) are ever written.
"""
import math
import time

import numpy as np

import hop3cfg as hc
from safety import SafetyError, bounds, check_goal

REG_OFS = 31
REG_TORQUE_ENABLE = 40
REG_TORQUE_LIMIT = 48
REG_PRESENT_POS = 56
REG_PRESENT_SPEED = 58
REG_PRESENT_LOAD = 60
REG_MOVING = 66
SYNC_READ_LEN = 11
WRITABLE = {REG_TORQUE_ENABLE, REG_TORQUE_LIMIT}


def assert_safe_register(addr, value):
    assert addr in WRITABLE, f"refusing to write register {addr}: only 40 and 48 (SRAM) are allowed"
    if addr == REG_TORQUE_ENABLE:
        assert value in (0, 1), f"register 40 accepts only 0/1 (128 re-centres the servo), got {value}"
    if addr == REG_TORQUE_LIMIT:
        assert 0 < value <= 1000, f"torque limit must be 1..1000, got {value}"


class BusBase:
    """Safety layer shared by Bus and MockBus. Subclasses implement the _low-level methods."""

    mock = False

    def __init__(self, cfg, z_floor=None):
        self.cfg = cfg
        self.z_floor = z_floor
        self.ids = hc.joint_ids(cfg)
        self.gid = hc.gripper_id(cfg)
        self.last_goal_s = None
        self.port = None

    def ping_all(self, ids=None):
        return {i: self._ping(i) for i in (ids or self.ids + [self.gid])}

    def read_offsets(self, ids=None):
        return {i: self._read_offset(i) for i in (ids or self.ids + [self.gid])}

    def set_torque(self, sid, on):
        assert isinstance(on, bool), "set_torque takes a bool"
        v = 1 if on else 0
        assert_safe_register(REG_TORQUE_ENABLE, v)
        return self._write1(sid, REG_TORQUE_ENABLE, v)

    def set_torque_limit(self, sid, val):
        val = int(val)
        assert_safe_register(REG_TORQUE_LIMIT, val)
        return self._write2(sid, REG_TORQUE_LIMIT, val)

    @staticmethod
    def _check_speed_acc(speed, acc):
        assert int(speed) > 0, "speed must be > 0 (0 means MAX speed on ST3215)"
        assert 0 < int(acc) <= 254, "acc must be 1..254 (0 means MAX acceleration)"

    def sync_write(self, goals_counts, speed, acc, max_step=None):
        """goals_counts: {id: counts} for exactly the three arm joints. Gripper only via hold_gripper()."""
        self._check_speed_acc(speed, acc)
        if set(goals_counts) != set(self.ids):
            raise SafetyError(f"sync_write needs exactly joint ids {self.ids}, got {sorted(goals_counts)}")
        c = np.array([goals_counts[i] for i in self.ids], float)
        s = hc.counts_to_deg(c, self.cfg)
        check_goal(s, self.cfg, self.last_goal_s, max_step, self.z_floor)
        self._write_goals({i: int(goals_counts[i]) for i in self.ids}, int(speed), int(acc))
        self.last_goal_s = s
        return s

    def write_goal_deg(self, s_deg, speed, acc, max_step=None):
        cnt = hc.deg_to_counts(s_deg, self.cfg)
        return self.sync_write(dict(zip(self.ids, cnt.tolist())), speed, acc, max_step)

    def hold_gripper(self, speed=200, acc=20):
        self._check_speed_acc(speed, acc)
        s = float(self.cfg["gripper_cmd_deg"])
        lo, hi = self.cfg["servo_limits_deg"]["j4"]
        if not (lo <= s <= hi):
            raise SafetyError(f"gripper_cmd_deg {s} outside j4 limits [{lo}, {hi}]")
        self._write_goals({self.gid: int(hc.deg_to_counts(s, self.cfg))}, int(speed), int(acc))

    def hold_present(self, ids=None, speed=100, acc=10, tries=10):
        """Goal := present position (zero motion), used before torque-on and to hold after an abort.
        Checks hard limits only: the arm is already there, so the margin/z-floor rules do not apply."""
        ids = ids or self.ids
        for _ in range(tries):
            rd, ok = self.sync_read(ids)
            if ok:
                break
        else:
            raise IOError("hold_present: could not read present positions")
        lo, hi = bounds(self.cfg, margin=False)
        goals = {}
        for i in ids:
            c = rd[i]["pos"]
            if i in self.ids:
                j = self.ids.index(i)
                s = c * 360.0 / hc.counts_per_rev(self.cfg)
                if not (lo[j] <= s <= hi[j]):
                    raise SafetyError(f"id {i} present {s:.1f} deg outside hard limits; not writing a goal")
            goals[i] = int(c)
        self._check_speed_acc(speed, acc)
        self._write_goals(goals, speed, acc)
        if set(ids) == set(self.ids):
            self.last_goal_s = hc.counts_to_deg([goals[i] for i in self.ids], self.cfg)
        return goals

    def torque_on_safely(self, ids=None):
        ids = ids or self.ids
        self.hold_present(ids)
        for i in ids:
            self.set_torque(i, True)

    def sync_read(self, ids=None):
        """-> ({id: {pos, speed, load, moving} | None}, comm_ok)."""
        ids = ids or self.ids
        rd = self._read(ids)
        return rd, all(rd.get(i) is not None for i in ids)

    def present_deg(self, ids=None, tries=10):
        ids = ids or self.ids
        for _ in range(tries):
            rd, ok = self.sync_read(ids)
            if ok:
                return hc.counts_to_deg([rd[i]["pos"] for i in ids], self.cfg)
        raise IOError("could not read present positions")


class Bus(BusBase):
    def open(self, port, baud=1000000):
        from scservo_sdk import PortHandler, sms_sts, GroupSyncRead
        import serial
        self._GroupSyncRead = GroupSyncRead
        self.ph = PortHandler(port)
        try:
            if not self.ph.openPort():
                raise IOError(f"openPort failed on {port}")
        except serial.SerialException as e:
            hint = ""
            if "Permission" in str(e):
                hint = " -> permission: `groups` must list dialout (re-login) or `sudo chmod 666 " + port + "`"
            elif "No such file" in str(e) or "could not open" in str(e):
                hint = " -> port missing: check `ls /dev/ttyACM* /dev/ttyUSB*`, brltty (see cheatsheet)"
            raise IOError(f"cannot open {port}: {e}{hint}") from e
        if not self.ph.setBaudRate(baud):
            raise IOError(f"setBaudRate({baud}) failed")
        self.pk = sms_sts(self.ph)
        self._readers = {}
        self.port = port
        return self

    def close(self):
        if getattr(self, "ph", None) and self.ph.is_open:
            self.ph.closePort()

    def _ping(self, sid):
        from scservo_sdk import COMM_SUCCESS
        model, res, err = self.pk.ping(sid)
        return model if res == COMM_SUCCESS else None

    def _read_offset(self, sid):
        from scservo_sdk import COMM_SUCCESS
        raw, res, err = self.pk.read2ByteTxRx(sid, REG_OFS)
        return self.pk.scs_tohost(raw, 11) if res == COMM_SUCCESS else None

    def _write1(self, sid, addr, val):
        assert_safe_register(addr, val)
        return self.pk.write1ByteTxRx(sid, addr, val)

    def _write2(self, sid, addr, val):
        assert_safe_register(addr, val)
        return self.pk.write2ByteTxRx(sid, addr, val)

    def _write_goals(self, goals, speed, acc):
        assert speed > 0
        gsw = self.pk.groupSyncWrite
        gsw.clearParam()
        for sid, c in goals.items():
            self.pk.SyncWritePosEx(sid, int(c), speed, acc)
        gsw.txPacket()
        gsw.clearParam()

    def _read(self, ids):
        key = tuple(ids)
        g = self._readers.get(key)
        if g is None:
            g = self._GroupSyncRead(self.pk, REG_PRESENT_POS, SYNC_READ_LEN)
            for i in ids:
                g.addParam(i)
            self._readers[key] = g
        for i in ids:
            g.data_dict[i] = []  # the SDK keeps stale data when a reply is short; clear so isAvailable is honest
        g.txRxPacket()
        out = {}
        for i in ids:
            ok, _ = g.isAvailable(i, REG_PRESENT_POS, SYNC_READ_LEN)
            if not ok:
                out[i] = None
                continue
            out[i] = {"pos": self.pk.scs_tohost(g.getData(i, REG_PRESENT_POS, 2), 15),
                      "speed": self.pk.scs_tohost(g.getData(i, REG_PRESENT_SPEED, 2), 15),
                      "load": self.pk.scs_tohost(g.getData(i, REG_PRESENT_LOAD, 2), 10),
                      "moving": g.getData(i, REG_MOVING, 1)}
        return out


def make_bus(cfg, mock=False, z_floor=None, **mock_kw):
    if mock:
        from mock_bus import MockBus
        return MockBus(cfg, z_floor, **mock_kw)
    return Bus(cfg, z_floor)


def wait_settled(bus, timeout=5.0, tol_counts=3, log=None):
    """Lab-script pattern: poll sync read until MOVING==0 on all joints (and near goal)."""
    t_end = time.perf_counter() + timeout
    goal_c = hc.deg_to_counts(bus.last_goal_s, bus.cfg)
    time.sleep(0.01)
    while time.perf_counter() < t_end:
        rd, ok = bus.sync_read()
        if log:
            log(rd, ok)
        if ok and all(rd[i]["moving"] == 0 and abs(rd[i]["pos"] - goal_c[j]) <= tol_counts
                      for j, i in enumerate(bus.ids)):
            return True
        time.sleep(0.005)
    return False


def plan_slow_path(start_s, target_s, deg_per_s, rate_hz):
    start_s, target_s = np.asarray(start_s, float), np.asarray(target_s, float)
    n = max(1, math.ceil(np.max(np.abs(target_s - start_s)) / (deg_per_s / rate_hz)))
    a = np.linspace(0.0, 1.0, n + 1)[:, None]
    return start_s + a * (target_s - start_s)


def check_path(bus, path):
    prev = None
    for p in path:
        check_goal(p, bus.cfg, prev, bus.cfg["max_step_deg"], bus.z_floor)
        prev = p


def move_slow(bus, target_s, deg_per_s=10.0, speed=None, acc=None, rate_hz=50, abort=None):
    """Interpolated joint-space approach from the present pose (every point pre-checked), then wait to settle."""
    st = bus.cfg["stream"]
    speed, acc = speed or st["speed"], acc or st["acc"]
    path = plan_slow_path(bus.present_deg(), target_s, deg_per_s, rate_hz)
    check_path(bus, path)
    bus.last_goal_s = None
    period = 1.0 / rate_hz
    t0 = time.perf_counter()
    for k, p in enumerate(path):
        dt = t0 + k * period - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        if abort and abort():
            raise KeyboardInterrupt
        bus.write_goal_deg(p, speed, acc, bus.cfg["max_step_deg"])
        bus.sync_read()
    return wait_settled(bus)
