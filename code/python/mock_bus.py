"""Fake ST3215 bus with the same interface as bus.Bus (safety layer shared via BusBase).

Present position follows the goal through a first-order lag (tau ~80 ms) capped at the commanded
speed, +-1 count read noise, 1% dropped replies that stall the read ~50 ms (like the SDK timeout).
With torque off the joints random-walk slightly, standing in for hand guiding.
"""
import math
import time

import numpy as np

import hop3cfg as hc
from bus import BusBase, assert_safe_register, REG_TORQUE_ENABLE, REG_TORQUE_LIMIT

MODEL_ST3215 = 777
MOCK_OFFSETS = {1: 12, 2: -7, 3: 0, 4: 3}


class MockBus(BusBase):
    mock = True

    def __init__(self, cfg, z_floor=None, tau=0.08, drop_p=0.01, stall_s=0.05, wire_s=0.0015, seed=None,
                 start_s=None):
        super().__init__(cfg, z_floor)
        self.tau, self.drop_p, self.stall_s, self.wire_s = tau, drop_p, stall_s, wire_s
        self.rng = np.random.default_rng(seed)
        start = list(start_s or hc.NEUTRAL_S) + [cfg["gripper_cmd_deg"]]
        self.all_ids = self.ids + [self.gid]
        c0 = hc.deg_to_counts(start, cfg)
        self.pos = {i: float(c) for i, c in zip(self.all_ids, c0)}
        self.goal = dict(self.pos)
        self.vel = {i: 0.0 for i in self.all_ids}
        self.speed = {i: 1000.0 for i in self.all_ids}
        self.torque = {i: True for i in self.all_ids}
        self.torque_limit = {i: 1000 for i in self.all_ids}
        self.t_last = time.perf_counter()
        self.dropped = 0
        self.writes = []

    def open(self, port="mock", baud=1000000):
        self.port = "mock"
        return self

    def close(self):
        pass

    def _advance(self):
        now = time.perf_counter()
        dt = now - self.t_last
        self.t_last = now
        if dt <= 0:
            return
        a = 1.0 - math.exp(-dt / self.tau)
        for i in self.all_ids:
            if self.torque[i]:
                step = (self.goal[i] - self.pos[i]) * a
                cap = self.speed[i] * dt
                step = max(-cap, min(cap, step))
            else:
                step = self.rng.normal(0.0, 1.5) * min(1.0, dt / 0.02)
            self.pos[i] += step
            self.vel[i] = step / dt

    def _ping(self, sid):
        time.sleep(self.wire_s)
        return MODEL_ST3215 if sid in self.all_ids else None

    def _read_offset(self, sid):
        time.sleep(self.wire_s)
        return MOCK_OFFSETS.get(sid)

    def _write1(self, sid, addr, val):
        assert_safe_register(addr, val)
        self.writes.append((sid, addr, val))
        if addr == REG_TORQUE_ENABLE:
            self._advance()
            self.torque[sid] = bool(val)
        return 0, 0

    def _write2(self, sid, addr, val):
        assert_safe_register(addr, val)
        self.writes.append((sid, addr, val))
        if addr == REG_TORQUE_LIMIT:
            self.torque_limit[sid] = val
        return 0, 0

    def _write_goals(self, goals, speed, acc):
        assert speed > 0
        self._advance()
        time.sleep(0.0004)
        for sid, c in goals.items():
            self.goal[sid] = float(c)
            self.speed[sid] = float(speed)
            self.torque[sid] = True  # assumed (unverified on hardware): a goal write re-enables torque

    def _read(self, ids):
        time.sleep(self.wire_s)
        drop = None
        if self.rng.random() < self.drop_p:
            drop = ids[int(self.rng.integers(len(ids)))]
            self.dropped += 1
            time.sleep(self.stall_s)
        self._advance()
        out = {}
        for i in ids:
            if i == drop:
                out[i] = None
                continue
            err = self.goal[i] - self.pos[i]
            out[i] = {"pos": int(round(self.pos[i])) + int(self.rng.integers(-1, 2)),
                      "speed": int(round(self.vel[i])),
                      "load": int(max(-1000, min(1000, 4 * err))) if self.torque[i] else 0,
                      "moving": int(self.torque[i] and abs(err) > 1.5)}
        return out
