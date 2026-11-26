"""Recompute s/c columns from th columns with the CURRENT config.json map (after a sign change).

python retarget_cmds.py data/cmd_*.csv
Each file is re-validated; it is rewritten in place (old copy -> .csv.bak) only if it passes.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import hop3cfg as hc
import safety


def retarget(path, cfg):
    path = Path(path)
    header, arr = hc.read_cmd(path)
    if header != hc.CMD_HEADER:
        return False, ["header != CONTRACTS header"]
    th = arr[:, 8:11]
    s = hc.theta_to_s(th, cfg)
    new = arr.copy()
    new[:, 2:5] = s
    new[:, 5:8] = hc.deg_to_counts(s, cfg)
    errors, checks = safety.validate_arrays(header, new, cfg)
    if errors:
        return False, errors
    shutil.copy2(path, path.with_name(path.name + ".bak"))
    hc.write_cmd(path, new)
    mp = path.with_suffix(".json")
    if mp.exists():
        meta = json.loads(mp.read_text())
        shutil.copy2(mp, mp.with_name(mp.name + ".bak"))
        meta["map_used"] = cfg["map"]
        meta["checks"] = {k: checks[k] for k in ("limits", "step", "zfloor", "finite")}
        mp.write_text(json.dumps(meta, indent=2) + "\n")
    changed = int(np.sum(np.any(new[:, 5:8] != arr[:, 5:8], axis=1)))
    return True, [f"{changed} row(s) changed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    cfg = hc.load_config()
    m = cfg["map"]
    print("map: " + "  ".join(f"{t}: sign {m[t]['sign']:+d} offset {m[t]['offset']}" for t in hc.THETAS))
    all_ok = True
    for f in args.files:
        if f.endswith(".bak"):
            continue
        ok, msgs = retarget(f, cfg)
        print(f"{'OK  ' if ok else 'FAIL'}  {f}" + ("" if ok else "  (left untouched)"))
        for e in msgs:
            print("   ", e)
        all_ok &= ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
