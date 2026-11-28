function test_analysis(which)
% End-to-end checks of the analysis pipeline on SYNTHETIC data with known truth.
% which: 'all' (default) | 'letterB' | 'circle' (legacy pen-on-paper path).
if nargin < 1, which = 'all'; end
if any(strcmp(which, {'all', 'letterB'})), test_letterB(); end
if any(strcmp(which, {'all', 'circle'})), test_circle_pen(); end
end

function test_letterB()
root = hop3_root();
py = ['"' fullfile(root, '.venv', 'bin', 'python') '"'];
pyd = fullfile(root, 'code', 'python');
run_py = @(args) run_cmd(sprintf('cd "%s" && env -u LD_LIBRARY_PATH %s %s', root, py, args));
run_py(['"' fullfile(pyd, 'make_video_board.py') '"']);
run_py(['"' fullfile(pyd, 'synth_session.py') '" letterB']);
truth = jsondecode(fileread(fullfile(root, 'data', 'synthetic', 'truth_letterB.json')));
S = load_session(fullfile(root, truth.files.session));
logs = truth.files.run_logs;
run_py(sprintf('"%s" "%s" --run-log "%s"', fullfile(pyd, 'track_tip_video.py'), truth.files.video, logs{1}));
[~, vstem] = fileparts(truth.files.video);
vf = fullfile('results', 'synthetic', ['video_' vstem '.json']);
V = jsondecode(fileread(fullfile(root, vf)));
vt = readtable(fullfile(root, truth.files.video_truth));
fr = V.frame + 1;
vid_err = rms(hypot(V.y_mm - vt.y_mm(fr), V.z_mm - vt.z_mm(fr)));

A = analyze_run(logs{1}, '', struct('session', S, 'video', vf));
lags = A.lag_ms; bl = A.backlash_halfwidth_deg_per_joint;
for i = 2:numel(logs)
    Ai = analyze_run(logs{i}, '', struct('session', S, 'name', sprintf('letterB_rep%d', i)));
    lags(end+1) = Ai.lag_ms; bl(end+1, :) = Ai.backlash_halfwidth_deg_per_joint; %#ok<AGROW>
end
D = A.shape.desired;
chk = {
  'lag err, worst repeat [ms]',         max(abs(lags - truth.lag_ms)), 15
  'backlash err J2/J3, worst [deg]',    max(max(abs(bl(:, 2:3) - truth.backlash_halfwidth_deg))), 0.1
  'backlash err J1, worst [deg]',       max(abs(bl(:, 1) - truth.backlash_halfwidth_deg)), 0.2
  'video path vs truth RMS [mm]',       vid_err, 1.0
  'video sync offset err [frames]',     abs(V.sync_offset_s - truth.sync_offset_s)*truth.fps, 1.0
  'mechanics rms - injected [mm]',      abs(A.mechanics.rms_mm - truth.mechanics_mm), 0.3
  'desired top R err [mm]',             abs(D.top_R_mm - truth.B.R_mm), 1e-3
  'desired bottom R err [mm]',          abs(D.bottom_R_mm - truth.B.R_mm), 1e-3
  'desired spine length err [mm]',      abs(D.spine_length_mm - truth.B.spine_length_mm), 1e-3
  'desired spine straightness [mm]',    D.spine_straightness_rms_mm, 1e-3
  'desired contour max vs ideal B [mm]', D.contour_max_mm, 1e-3
};
fprintf('\n%-38s %10s %8s  %s\n', 'check (SYNTHETIC letter B)', 'value', 'tol', 'result');
pass = true;
for i = 1:size(chk, 1)
    ok = chk{i,2} <= chk{i,3}; pass = pass && ok;
    fprintf('%-38s %10.4f %8.4f  %s\n', chk{i,1}, chk{i,2}, chk{i,3}, string(ok));
end
assert(A.synthetic && V.synthetic && isstruct(A.shape.video), 'synthetic/video flags not propagated');
fprintf('\nrecovered lag per repeat %s ms (truth %.0f); backlash per joint (rows=repeats) %s deg (truth %.2f)\n', ...
    mat2str(lags, 4), truth.lag_ms, mat2str(bl, 3), truth.backlash_halfwidth_deg);
fprintf('video: RMS vs truth %.3f mm, sync %.4f s (truth %.4f), board reproj %.3f mm; mechanics %.3f mm (truth %.3f)\n', ...
    vid_err, V.sync_offset_s, truth.sync_offset_s, V.reproj_rms_mm, A.mechanics.rms_mm, truth.mechanics_mm);
for c = {'desired', 'nn', 'dh_cmd', 'dh_meas', 'video'}
    m = A.shape.(c{1});
    fprintf('shape %-8s contour rms %.3f max %.3f | spine str %.3f len %.3f | R top %.3f bot %.3f | haus %.3f\n', c{1}, ...
        m.contour_rms_mm, m.contour_max_mm, m.spine_straightness_rms_mm, m.spine_length_mm, m.top_R_mm, m.bottom_R_mm, m.hausdorff_mm);
end
assert(pass, 'test_analysis letterB: recovery checks FAILED');
fprintf('test_analysis letterB: ALL PASS\n');
end

function test_circle_pen()
root = hop3_root();
py = ['"' fullfile(root, '.venv', 'bin', 'python') '"'];
pyd = fullfile(root, 'code', 'python');
run_py = @(args) run_cmd(sprintf('env -u LD_LIBRARY_PATH %s %s', py, args));
run_py(['"' fullfile(pyd, 'make_fiducial_sheet.py') '"']);
run_py(['"' fullfile(pyd, 'synth_session.py') '" circle']);
truth = jsondecode(fileread(fullfile(root, 'data', 'synthetic', 'truth_circle.json')));
sf = fullfile(root, truth.files.session);
S = load_session(sf);

T = touchoff_fit(truth.files.touchoff, fullfile(root, 'data', 'fiducials.json'), S);
run_py(sprintf('"%s" "%s" --session "%s"', fullfile(pyd, 'digitize_trace.py'), ...
    fullfile(root, truth.files.photo), sf));
[~, pstem] = fileparts(truth.files.photo);
trf = fullfile(root, 'results', 'synthetic', ['trace_' pstem '.json']);
tr = jsondecode(fileread(trf));
% no-pen pass first so the saved run_* outputs are the pen version
A0 = analyze_run(truth.files.run_log, truth.files.cmd, struct('session', S));
A = analyze_run(truth.files.run_log, truth.files.cmd, struct('trace', trf, ...
    'touchoff', fullfile(root, 'results', 'synthetic', 'touchoff.json'), 'session', S));

chk = {
  'touch-off translation err [mm]', norm(T.t_mm(:) - truth.sheet_to_robot.t_mm(:)), 0.5
  'touch-off yaw err [deg]',        abs(T.yaw_deg - truth.sheet_to_robot.yaw_deg), 0.2
  'plane z err [mm]',               abs(T.plane.z_at_sheet_origin_mm - truth.plane_z_mm), 0.5
  'lag err [ms]',                   abs(A.lag_ms - truth.lag_ms), 20
  'mechanics rms - injected [mm]',  abs(A.mechanics.rms_mm - truth.mechanics_radial_mm), 0.5
  'digitised R err [mm]',           abs(tr.fit.R_mm - truth.trace_sheet_fit.R_mm), 0.5
  'digitised centre err [mm]',      norm(tr.fit.centre_mm(:) - truth.trace_sheet_fit.centre_mm(:)), 0.5
  'pen circle R err (robot) [mm]',  abs(A.circle_fit_pen.R_mm - truth.pen_robot_fit.R_mm), 0.5
  'DH(q_meas) circle R err [mm]',   abs(A.circle_fit_dh_meas.R_mm - truth.dh_meas_fit.R_mm), 0.5
  'no-pen run: lag err [ms]',       abs(A0.lag_ms - truth.lag_ms), 20
};
fprintf('\n%-34s %10s %8s  %s\n', 'check (SYNTHETIC)', 'value', 'tol', 'result');
pass = true;
for i = 1:size(chk, 1)
    ok = chk{i,2} <= chk{i,3}; pass = pass && ok;
    fprintf('%-34s %10.4f %8.3f  %s\n', chk{i,1}, chk{i,2}, chk{i,3}, string(ok));
end
assert(A.synthetic && T.synthetic && tr.synthetic, 'synthetic flag not propagated');
assert(~isstruct(A0.mechanics) && isnan(A0.mechanics) && contains(A0.total_basis, 'no video / pen'), 'no-pen labelling wrong');
fprintf('\nrecovered: t=(%.3f %.3f %.3f) yaw=%.3f lag=%.1f ms (per joint %s) mech rms=%.3f p95=%.3f\n', ...
    T.t_mm, T.yaw_deg, A.lag_ms, mat2str(A.lag_ms_per_joint, 3), A.mechanics.rms_mm, A.mechanics.p95_mm);
fprintf('touch-off residual rms %.3f mm; inv_residual rms %.3f; nn_model_err rms %.3f; tracking temporal %.3f contour %.3f; total %.3f\n', ...
    T.residual_rms_mm, A.inv_residual.rms_mm, A.nn_model_err.rms_mm, A.tracking_temporal.rms_mm, ...
    A.tracking_contour.rms_mm, A.total.rms_mm);
assert(pass, 'test_analysis: recovery checks FAILED');
fprintf('test_analysis circle/pen (legacy): ALL PASS\n');
end

function run_cmd(cmd)
[st, out] = system(cmd);
fprintf('%s', out);
assert(st == 0, 'command failed: %s', cmd);
end
