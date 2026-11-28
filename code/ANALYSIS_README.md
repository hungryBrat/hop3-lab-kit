Ground-truth analysis, part (e): desired vs NN vs actually traced  (supersedes ANALYSIS_README.md for the letter B)
Python: HOP_3/.venv/bin/python.  MATLAB: ~/MATLAB/R2026b/bin/matlab -batch "cd('<HOP_3>/code/matlab'); <cmd>"
Primary trajectory: letter 'B' in air, front plane x=260 mm. Ground truth = servo encoders (core) + optional phone video.
Pen-on-paper is legacy and OFF by default ("use_pen": true in the session json enables touch-off + digitise).

BEFORE THE SESSION (video, optional)
1. python code/python/make_video_board.py -> data/video_board.{pdf,png,json}. Print A4 at 100%, cut out the dashed window.
2. Board face in the plane x=260 (else measure the offset -> placement.board_offset_x_mm, negative = behind the tip plane),
   centre ticks at (y 0, z 150), +u to the right as seen from +X0.
3. Orange (or green/pink) marker on the tip. Phone static on a stand at +X0 (~0.45 m), whole board in frame, 30 fps.

AFTER THE SESSION
1. Fill session/session_data.json: run_logs (repeats of one command are fine), video (path, or list aligned with run_logs),
   video_run (run log the single video belongs to; default first), optional video_board, video_colour.
2. run_all('post')  [or run_all('post','synthetic')]: tracks the video (track_tip_video.py -> results/video_<stem>.json +
   figures/video_<stem>_overlay.png), analyze_run on every run log (repeats -> <name>, <name>_rep2, ...),
   results/repeatability_<name>.json, then report/build.sh [--synthetic].
3. Manual: python code/python/track_tip_video.py video.mp4 --run-log logs/run_letterB_<ts>.csv [--colour orange] [--debug]
   A = analyze_run('logs/run_letterB_<ts>.csv','',struct('video','results/video_<stem>.json'))
Per run: results/run_<name>.json (error terms, lag, backlash, shape{desired,nn,dh_cmd,dh_meas,video})
         figures/run_<name>_{front,z,joints,errors}.png

CONVENTIONS
- Tracking = DH(q_meas) vs DH(goal counts); NN error uses th columns. Contour = point-to-polyline. comm_ok=0 rows dropped.
- mechanics = video - DH(q_meas) at equal run time (y-z); null without video. total = video - p_des (y-z contour) with
  video, else DH(q_meas) - p_des (3-D contour). See total_basis / mechanics_basis in the json.
- Shape: segments from cmd meta `segments` (k 0-based, inclusive). Spine straightness = RMS distance to fitted 3-D line
  (2-D for video); spine length = extent along that line; R = geometric circle fit in y-z; contour/Hausdorff vs ideal B.
- Video sync: t_run = t_video + sync_offset_s (xcorr of tracked z(t) vs DH(q_meas) z(t), then bounded refinement).
  Frame times from ffprobe when installed, else fps (no ffmpeg on this box). Non-zero board offset -> solvePnP with an
  assumed focal length (--fpx, default 0.77 x long image side): pass the real phone focal length in px.
- Lag = delay + backlash model fit. On the slow B the joints move < 1 encoder count per 80 ms, so the 1-count
  quantisation biases lag by about +-10 ms (synthetic: 72 ms for true 80 ms); backlash within one grid step (0.05 deg).
- Anything under synthetic/ writes to results/synthetic, figures/synthetic.

SYNTHETIC DATA (treated as ground truth until the real session)
synth_session.py letterB : uses data/cmd_letterB.csv if present (--own-cmd = analytic IK + per-segment min-jerk into
  data/synthetic), 3 repeat logs in logs/synthetic (80 ms lag, +-0.5 deg backlash, 1-count quantisation, 1% drops,
  seeds 101-103), data/synthetic/{session_data_synth.json, phone_letterB.mp4 (board + orange dot at
  DH(q_true) + (0, 0.6, -1.375) mm), video_truth_letterB.csv, truth_letterB.json}.
synth_session.py circle  : legacy pen/paper set (truth_circle.json, session_data_synth_circle.json).
test_analysis            : 'all' (default) = letter-B recovery test + legacy circle/pen test; test_analysis('letterB').
