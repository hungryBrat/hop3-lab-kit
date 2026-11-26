"""STANDIN fitnet clone: mapminmax -> tansig hidden -> purelin, trainlm-style LM, 15% validation stop."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def mapminmax_fit(X):
    xmin, xmax = X.min(axis=1), X.max(axis=1)
    return {"xoffset": xmin, "gain": 2.0 / (xmax - xmin), "ymin": -1.0}


def apply_map(m, X):
    return (X - m["xoffset"][:, None]) * m["gain"][:, None] + m["ymin"]


def init_layers(sizes, rng):
    layers = []
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        W = rng.uniform(-1, 1, (n_out, n_in))
        if k < len(sizes) - 2:
            beta = 0.7 * n_out ** (1.0 / n_in)
            W *= beta / np.linalg.norm(W, axis=1, keepdims=True)
            b = beta * np.linspace(-1, 1, n_out) * np.sign(rng.uniform(-1, 1, n_out))
        else:
            b = rng.uniform(-1, 1, n_out)
        layers.append([W, b])
    return layers


def pack(layers):
    return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in layers])


def unpack(theta, sizes):
    layers, i = [], 0
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        W = theta[i:i + n_in * n_out].reshape(n_out, n_in); i += n_in * n_out
        b = theta[i:i + n_out]; i += n_out
        layers.append([W, b])
    return layers


def forward(layers, X):
    acts = [X]
    for k, (W, b) in enumerate(layers):
        z = W @ acts[-1] + b[:, None]
        acts.append(np.tanh(z) if k < len(layers) - 1 else z)
    return acts


def sse(layers, X, T, inv_g):
    r = (forward(layers, X)[-1] - T) * inv_g[:, None]
    return float(np.sum(r * r))


def normal_eqs(layers, X, T, inv_g, chunk=4000):
    P = sum(W.size + b.size for W, b in layers)
    offs, o = [], 0
    for W, b in layers:
        offs.append(o); o += W.size + b.size
    JtJ, Jtr, s = np.zeros((P, P)), np.zeros(P), 0.0
    L = len(layers)
    buf = np.zeros((3 * chunk, P))
    for c0 in range(0, X.shape[1], chunk):
        acts = forward(layers, X[:, c0:c0 + chunk])
        n = acts[0].shape[1]
        r = (acts[-1] - T[:, c0:c0 + chunk]) * inv_g[:, None]
        s += float(np.sum(r * r))
        J = buf[:3 * n]
        for k in range(3):
            Jk = J[k * n:(k + 1) * n]
            delta = np.zeros((n, 3)); delta[:, k] = inv_g[k]
            for l in range(L - 1, -1, -1):
                W = layers[l][0]
                n_out, n_in = W.shape
                a_prev = acts[l].T
                o = offs[l]
                view = Jk[:, o:o + n_out * n_in].reshape(n, n_out, n_in)
                np.multiply(delta[:, :, None], a_prev[:, None, :], out=view)
                Jk[:, o + n_out * n_in:o + n_out * n_in + n_out] = delta
                if l > 0:
                    delta = (delta @ W) * (1 - a_prev ** 2)
        JtJ += J.T @ J
        Jtr += J.T @ r.reshape(-1)
    return JtJ, Jtr, s


def train_lm(sizes, Xtr, Ttr, Xva, Tva, inv_g, rng, epochs, max_fail,
             mu=1e-3, mu_dec=0.1, mu_inc=10.0, mu_max=1e10, min_grad=1e-7):
    layers = init_layers(sizes, rng)
    theta = pack(layers)
    best = (sse(layers, Xva, Tva, inv_g), theta.copy(), 0)
    fails, stop, ep = 0, "epochs", 0
    for ep in range(1, epochs + 1):
        JtJ, Jtr, s = normal_eqs(unpack(theta, sizes), Xtr, Ttr, inv_g)
        if np.linalg.norm(2 * Jtr) / Xtr.shape[1] < min_grad:
            stop = "min_grad"; break
        I = np.eye(len(theta))
        while mu <= mu_max:
            step = np.linalg.solve(JtJ + mu * I, -Jtr)
            s_new = sse(unpack(theta + step, sizes), Xtr, Ttr, inv_g)
            if s_new < s:
                theta = theta + step; mu *= mu_dec; break
            mu *= mu_inc
        if mu > mu_max:
            stop = "mu_max"; break
        v = sse(unpack(theta, sizes), Xva, Tva, inv_g)
        if v < best[0]:
            best, fails = (v, theta.copy(), ep), 0
        else:
            fails += 1
            if fails >= max_fail:
                stop = "max_fail"; break
    return unpack(best[1], sizes), best, ep, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=int, nargs="+", default=[20, 10])
    ap.add_argument("--n", type=int, default=0, help="use the first N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data/arm_workspace_points.csv")
    ap.add_argument("--out", default="models/fk_weights_standin.json")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--max_fail", type=int, default=10)
    a = ap.parse_args()

    D = np.loadtxt(ROOT / a.data if not Path(a.data).is_absolute() else a.data,
                   delimiter=",", skiprows=1)
    if a.n:
        D = D[:a.n]
    Q, P = D[:, :3].T, D[:, 3:].T
    # mapminmax settings from all data handed to train(), as fitnet does
    im, om = mapminmax_fit(Q), mapminmax_fit(P)
    X, T = apply_map(im, Q), apply_map(om, P)
    inv_g = 1.0 / om["gain"]

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(X.shape[1])
    n_tr = int(round(0.85 * len(idx)))
    tr, va = idx[:n_tr], idx[n_tr:]
    sizes = [3] + a.arch + [3]

    t0 = time.time()
    layers, best, ep, stop = train_lm(sizes, X[:, tr], T[:, tr], X[:, va], T[:, va],
                                      inv_g, rng, a.epochs, a.max_fail)
    dt = time.time() - t0
    val_rmse = np.sqrt(best[0] / (3 * len(va)))

    acts = ["tansig"] * len(a.arch) + ["purelin"]
    out = {
        "meta": {"source": "STANDIN", "arch": a.arch, "created": datetime.now().isoformat(timespec="seconds"),
                 "train_fcn": "numpy LM (trainlm clone)", "n_data": int(X.shape[1]), "n_train": int(len(tr)),
                 "n_val": int(len(va)), "seed": a.seed, "epochs": ep, "best_epoch": best[2],
                 "stop": stop, "val_rmse_mm": float(val_rmse), "train_time_s": round(dt, 2)},
        "in_map": {"xoffset": im["xoffset"].tolist(), "gain": im["gain"].tolist(), "ymin": -1},
        "out_map": {"xoffset": om["xoffset"].tolist(), "gain": om["gain"].tolist(), "ymin": -1},
        "layers": [{"W": W.tolist(), "b": b.tolist(), "act": act} for (W, b), act in zip(layers, acts)],
    }
    outp = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out))
    print(f"arch={a.arch} n={X.shape[1]} seed={a.seed} epochs={ep} (best {best[2]}, stop={stop}) "
          f"val_rmse={val_rmse:.4f} mm time={dt:.1f}s -> {outp}")


if __name__ == "__main__":
    main()
