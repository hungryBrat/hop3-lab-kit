"""Live run dashboard in a separate process (matplotlib multiprocess pattern, Pipe, blitting).

The control loop only ever calls DashLink.send(), which never blocks: samples go to a bounded queue drained by
a sender thread; a full queue drops the frame. Use via run_traj.py --dashboard [--offscreen].
"""
import multiprocessing as mp
import queue
import threading
import time

import numpy as np

import hop3cfg as hc

REFRESH_MS = 60


class DashLink:
    def __init__(self, cmd_path, cfg, offscreen=False, out_png=None, t_max=None):
        ctx = mp.get_context("spawn")
        recv_conn, self.conn = ctx.Pipe(duplex=False)
        self.offscreen = offscreen
        self.proc = ctx.Process(target=dashboard_main, name="hop3-dashboard",
                                args=(recv_conn, str(cmd_path), cfg, offscreen, str(out_png) if out_png else None,
                                      t_max))
        self.proc.start()
        self.q = queue.Queue(maxsize=200)
        self.dropped = 0
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while True:
            item = self.q.get()
            try:
                self.conn.send(item)
            except (BrokenPipeError, OSError):
                return
            if item is None:
                return

    def send(self, t, k, goal_s, meas_s):
        try:
            self.q.put_nowait((t, k, tuple(goal_s), None if meas_s is None else tuple(meas_s)))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout=30.0):
        try:
            self.q.put(None, timeout=1.0)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)
        if self.offscreen:
            self.proc.join(timeout)
        elif self.proc.is_alive():
            print("dashboard: close its window to exit")


def dashboard_main(conn, cmd_path, cfg, offscreen, out_png, t_max):
    import matplotlib
    matplotlib.use("Agg" if offscreen else "TkAgg")
    import matplotlib.pyplot as plt

    header, arr = hc.read_cmd(cmd_path)
    c = lambda n: hc.col(header, arr, n)
    xd, yd, zd = c("x_des"), c("y_des"), c("z_des")
    xn, yn = c("x_nn"), c("y_nn")
    s_cmd = np.column_stack([c("s1_deg"), c("s2_deg"), c("s3_deg")])
    t_max = t_max or float(c("t_s")[-1]) * 1.1

    fig = plt.figure(figsize=(13, 7.5))
    fig.suptitle(f"HOP-3 run: {hc.cmd_name(cmd_path)}")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1])
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_z = fig.add_subplot(gs[0, 1])
    ax_j = [fig.add_subplot(gs[i + 1, 1], sharex=ax_z) for i in range(3)]

    ax_xy.plot(xd, yd, "k-", lw=1.2, label="desired circle")
    ax_xy.plot(xn, yn, "--", color="tab:orange", lw=1.0, label="NN-predicted NN(q_cmd)")
    trail, = ax_xy.plot([], [], "-", color="tab:blue", lw=1.4, label="live DH(q_meas)", animated=True)
    head, = ax_xy.plot([], [], "o", color="tab:blue", ms=6, animated=True)
    pad = 25.0
    ax_xy.set_xlim(min(xd.min(), xn.min()) - pad, max(xd.max(), xn.max()) + pad)
    ax_xy.set_ylim(min(yd.min(), yn.min()) - pad, max(yd.max(), yn.max()) + pad)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x0 [mm]")
    ax_xy.set_ylabel("y0 [mm]")
    ax_xy.set_title("top view")
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc="upper right", fontsize=8)
    status = ax_xy.text(0.02, 0.02, "", transform=ax_xy.transAxes, fontsize=9, family="monospace", animated=True)

    zg_line, = ax_z.plot([], [], "k-", lw=1.0, label="desired z", animated=True)
    zm_line, = ax_z.plot([], [], "-", color="tab:blue", lw=1.2, label="DH(q_meas) z", animated=True)
    ax_z.set_ylim(zd.min() - 15, zd.max() + 15)
    ax_z.set_ylabel("z [mm]")
    ax_z.legend(loc="upper right", fontsize=8)
    ax_z.grid(alpha=0.3)
    ax_z.set_xlim(0, t_max)

    jg, jm = [], []
    for j, ax in enumerate(ax_j):
        g, = ax.plot([], [], "k-", lw=1.0, label="goal", animated=True)
        m, = ax.plot([], [], "-", color="tab:red", lw=1.2, label="present", animated=True)
        jg.append(g)
        jm.append(m)
        lo, hi = s_cmd[:, j].min(), s_cmd[:, j].max()
        span = max(hi - lo, 4.0)
        ax.set_ylim(lo - 0.3 * span, hi + 0.3 * span)
        ax.set_ylabel(f"s{j + 1} [deg]")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    ax_j[-1].set_xlabel("t [s]")
    fig.tight_layout()

    animated = [trail, head, status, zg_line, zm_line] + jg + jm
    st = {"bg": None, "done": False, "n": 0, "tmax": t_max}
    T, G, M, Pm, Tm = [], [], [], [], []
    k_last = [0]

    def on_draw(event):
        st["bg"] = fig.canvas.copy_from_bbox(fig.bbox)
        for a in animated:
            fig.draw_artist(a)

    fig.canvas.mpl_connect("draw_event", on_draw)

    def drain():
        try:
            while conn.poll():
                msg = conn.recv()
                if msg is None:
                    st["done"] = True
                    return
                t, k, g, m = msg
                T.append(t)
                G.append(g)
                k_last[0] = k
                if m is not None:
                    Tm.append(t)
                    M.append(m)
                    Pm.append(hc.fk_servo(m, cfg))
        except (EOFError, OSError):
            st["done"] = True

    def refresh():
        if T and T[-1] > st["tmax"]:
            st["tmax"] = T[-1] * 1.5
            ax_z.set_xlim(0, st["tmax"])
            fig.canvas.draw()
        if T:
            g = np.asarray(G)
            zg = hc.fk_servo(g, cfg)[:, 2]
            zg_line.set_data(T, zg)
            for j in range(3):
                jg[j].set_data(T, g[:, j])
        if M:
            p, m = np.asarray(Pm), np.asarray(M)
            trail.set_data(p[:, 0], p[:, 1])
            head.set_data([p[-1, 0]], [p[-1, 1]])
            zm_line.set_data(Tm, p[:, 2])
            for j in range(3):
                jm[j].set_data(Tm, m[:, j])
        status.set_text(f"k={k_last[0]:5d}  t={T[-1] if T else 0:6.2f}s  samples={len(T)}")
        if st["bg"] is None:
            return
        fig.canvas.restore_region(st["bg"])
        for a in animated:
            fig.draw_artist(a)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

    if offscreen:
        fig.canvas.draw()
        while not st["done"]:
            conn.poll(REFRESH_MS / 1000.0)
            drain()
            refresh()
        for a in animated:
            a.set_animated(False)
        fig.savefig(out_png or (hc.ROOT / "figures" / "dashboard_snapshot.png"), dpi=150)
        return

    def tick():
        drain()
        refresh()
        if st["done"]:
            timer.stop()
            for a in animated:
                a.set_animated(False)
            fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=REFRESH_MS)
    timer.add_callback(tick)
    timer.start()
    plt.show()
