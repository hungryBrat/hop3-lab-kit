"""Shared config/session loading, servo<->DH mapping, DH FK and elbow-up IK for HOP-3."""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HOP3_ROOT", Path(__file__).resolve().parents[2]))

CMD_HEADER = ["k", "t_s", "s1_deg", "s2_deg", "s3_deg", "c1", "c2", "c3", "th1", "th2", "th3",
              "x_des", "y_des", "z_des", "x_nn", "y_nn", "z_nn"]
RUN_HEADER = ["t_s", "k", "goal_c1", "goal_c2", "goal_c3", "pos_c1", "pos_c2", "pos_c3",
              "spd1", "spd2", "spd3", "load1", "load2", "load3", "comm_ok"]
TOUCHOFF_HEADER = ["point_id", "rep", "pos_c1", "pos_c2", "pos_c3"]
NEUTRAL_S = (90.0, 180.0, 180.0)
JOINTS = ("j1", "j2", "j3")
THETAS = ("theta1", "theta2", "theta3")


def config_path():
    return ROOT / "config.json"


def mock_dir():
    return ROOT / "logs" / "mock"


def session_path(mock=False):
    # Mock rehearsals write to logs/mock/ so they can never pollute the real session record.
    return mock_dir() / "session_data.json" if mock else ROOT / "session" / "session_data.json"


def logs_dir(mock=False):
    d = mock_dir() if mock else ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def load_config(path=None):
    with open(path or config_path()) as f:
        return json.load(f)


def joint_ids(cfg):
    return [cfg["servo_ids"][j] for j in JOINTS]


def gripper_id(cfg):
    return cfg["servo_ids"]["gripper"]


def gripper_enabled(cfg):
    # Missing key = disabled: the gripper servo (ID 4) is dead and must never be written.
    return bool(cfg.get("gripper_enabled", False))


def active_ids(cfg):
    """Servo IDs that are commanded and must reply. Always exactly the three arm joints."""
    ids = [int(i) for i in cfg.get("servo_ids_active", joint_ids(cfg))]
    if sorted(ids) != sorted(joint_ids(cfg)):
        raise ValueError(f"servo_ids_active {ids} must be exactly the joint ids {joint_ids(cfg)}")
    return joint_ids(cfg)


def writable_ids(cfg):
    return set(active_ids(cfg)) | ({gripper_id(cfg)} if gripper_enabled(cfg) else set())


def load_cmd_meta(cmd_path):
    p = Path(cmd_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


def is_letter_b(cmd_path, meta=None):
    meta = load_cmd_meta(cmd_path) if meta is None else meta
    return "letterb" in str(meta.get("name", Path(cmd_path).stem)).lower() or meta.get("plane") == "front"


def segments(meta, n_rows=None):
    """meta['segments'] -> [(name, k_start, k_end)] sorted by k_start (k_end inclusive), [] if absent."""
    out = []
    for s in meta.get("segments") or []:
        out.append((str(s["name"]), int(s["k_start"]), int(s["k_end"])))
    out.sort(key=lambda x: x[1])
    if n_rows is not None:
        out = [x for x in out if 0 <= x[1] < n_rows]
    return out


def signs_offsets(cfg):
    m = cfg["map"]
    return (np.array([m[t]["sign"] for t in THETAS], float),
            np.array([m[t]["offset"] for t in THETAS], float))


def s_to_theta(s, cfg):
    sign, off = signs_offsets(cfg)
    return sign * (np.asarray(s, float) - off)


def theta_to_s(th, cfg):
    sign, off = signs_offsets(cfg)
    return off + sign * np.asarray(th, float)


def counts_per_rev(cfg):
    return cfg["encoder"]["counts_per_rev"]


def deg_to_counts(s, cfg):
    return np.round(np.asarray(s, float) * counts_per_rev(cfg) / 360.0).astype(int)


def counts_to_deg(c, cfg):
    return np.asarray(c, float) * 360.0 / counts_per_rev(cfg)


def fk_dh(th, cfg):
    """th (...,3) DH deg -> (...,3) tip mm, closed form of 0T4."""
    d = cfg["dh"]
    lt = d["L3"] + d["f"]
    th = np.radians(np.asarray(th, float))
    t1, t2, t3 = th[..., 0], th[..., 1], th[..., 2]
    r = d["L2"] * np.cos(t2) + lt * np.cos(t2 + t3)
    z = d["d1"] + d["L2"] * np.sin(t2) + lt * np.sin(t2 + t3)
    return np.stack([np.cos(t1) * r, np.sin(t1) * r, z], axis=-1)


def fk_servo(s, cfg):
    return fk_dh(s_to_theta(s, cfg), cfg)


def ik_elbow_up(p, cfg):
    """p (...,3) mm -> (...,3) DH deg, elbow-up (theta3 < 0). Test use only."""
    d = cfg["dh"]
    l2, lt = d["L2"], d["L3"] + d["f"]
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    t1 = np.arctan2(y, x)
    r = np.hypot(x, y)
    h = z - d["d1"]
    c3 = (r ** 2 + h ** 2 - l2 ** 2 - lt ** 2) / (2 * l2 * lt)
    t3 = -np.arccos(np.clip(c3, -1, 1))
    t2 = np.arctan2(h, r) - np.arctan2(lt * np.sin(t3), l2 + lt * np.cos(t3))
    return np.degrees(np.stack([t1, t2, t3], axis=-1))


SESSION_FIELDS = ["date", "port", "usb_chip_lsusb", "psu_voltage_V", "servo_offsets_reg31", "sigma1", "sigma3",
                  "neutral_tip_height_mm", "touchoff_log", "fiducials_mm", "paper_plane_z_mm", "pen_offset_mm",
                  "printed_scale_check_mm", "chosen_trajectory", "z_c_mm", "run_logs", "sheet_photos", "video",
                  "notes"]


def _default_zc(cfg):
    tr = cfg["trajectories"]
    if cfg.get("primary_trajectory") == "letterB" and "letterB" in tr:
        return float(tr["letterB"]["centre_yz"][1])
    return float(tr["primary"]["z_family"][0])


class Session:
    """values: session fields with defaults filled; placeholder[field] True where the default was used."""

    def __init__(self, raw, cfg):
        sign, _ = signs_offsets(cfg)
        nominal = fk_servo(NEUTRAL_S, cfg)[2]
        defaults = {"port": cfg.get("port"), "sigma1": int(sign[0]), "sigma3": int(sign[2]),
                    "neutral_tip_height_mm": float(nominal), "paper_plane_z_mm": 0.0, "pen_offset_mm": 0.0,
                    "z_c_mm": _default_zc(cfg)}
        self.raw = raw
        self.values, self.placeholder = {}, {}
        for k in SESSION_FIELDS:
            v = raw.get(k)
            if isinstance(v, dict) and v and all(x is None for x in v.values()):
                v = None
            self.placeholder[k] = v is None
            self.values[k] = defaults.get(k) if v is None else v

    @property
    def any_placeholder(self):
        return any(self.placeholder[k] for k in ("sigma1", "sigma3", "servo_offsets_reg31"))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_session_raw(mock=False):
    p = session_path(mock)
    if mock and not p.exists():
        return _read_json(session_path(False))
    return _read_json(p)


def load_session(cfg=None, mock=False):
    return Session(load_session_raw(mock), cfg or load_config())


def update_session(fields, mock=False):
    """Read-modify-write only the given keys; every other field is preserved."""
    data = load_session_raw(mock)
    data.update(fields)
    _write_json_atomic(session_path(mock), data)
    return session_path(mock)


def set_config_signs(sigma1, sigma3, path=None):
    """Targeted edit of map.theta1/theta3 sign that keeps the coordinator's file layout."""
    path = Path(path or config_path())
    text = path.read_text()
    for name, val in (("theta1", sigma1), ("theta3", sigma3)):
        pat = r'("%s"\s*:\s*\{\s*"sign"\s*:\s*)-?\d+' % name
        text, n = re.subn(pat, lambda m: m.group(1) + str(int(val)), text)
        if n != 1:
            raise RuntimeError(f"could not locate map.{name}.sign in {path}")
    cfg = json.loads(text)
    assert cfg["map"]["theta1"]["sign"] == sigma1 and cfg["map"]["theta3"]["sign"] == sigma3
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def read_cmd(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float, ndmin=2)
    return header, arr


def col(header, arr, name):
    return arr[:, header.index(name)]


def write_cmd(path, arr):
    fmt = ["%d", "%.4f"] + ["%.6f"] * 3 + ["%d"] * 3 + ["%.6f"] * 3 + ["%.4f"] * 6
    np.savetxt(path, arr, delimiter=",", fmt=fmt, header=",".join(CMD_HEADER), comments="")


def cmd_name(path):
    stem = Path(path).stem
    return stem[4:] if stem.startswith("cmd_") else stem


class Prompter:
    """Interactive y/N + value prompts, or scripted answers (--answers a,b,c) / --yes for tests."""

    def __init__(self, answers=None, yes=False):
        self.answers = list(answers) if answers is not None else None
        self.yes = yes

    def _get(self, prompt):
        if self.answers is not None:
            if not self.answers:
                raise SystemExit(f"scripted answers exhausted at: {prompt}")
            a = self.answers.pop(0)
            print(f"{prompt}{a}  [scripted]")
            return a
        if self.yes:
            print(f"{prompt}y  [--yes]")
            return "y"
        return input(prompt)

    def confirm(self, msg):
        return self._get(f"{msg} [y/N] ").strip().lower() in ("y", "yes")

    def yes_no(self, msg):
        while True:
            a = self._get(f"{msg} [y/n] ").strip().lower()
            if a in ("y", "yes", "n", "no"):
                return a.startswith("y")
            if self.answers is not None or self.yes:
                raise SystemExit(f"invalid scripted answer {a!r} for: {msg}")
            print("  answer y or n")

    def number(self, msg):
        while True:
            a = self._get(f"{msg} (blank = skip): ").strip()
            if a in ("", "y") and (self.yes or a == ""):
                return None
            try:
                return float(a)
            except ValueError:
                if self.answers is not None:
                    raise SystemExit(f"invalid scripted number {a!r} for: {msg}")
                print("  enter a number or leave blank")

    def enter(self, msg):
        self._get(f"{msg} [Enter] ")


def prompter_from_args(args):
    return Prompter(args.answers.split(",") if args.answers else None, args.yes)


def add_prompt_args(ap):
    ap.add_argument("--yes", action="store_true", help="answer y to every prompt (tests only)")
    ap.add_argument("--answers", help="comma-separated scripted answers, consumed in order (tests only)")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)
