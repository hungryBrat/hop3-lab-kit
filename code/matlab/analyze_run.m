function [A, CV] = analyze_run(runlog, cmdfile, opts)
% Error decomposition + shape metrics of one run (any trajectory; letter-B aware via cmd meta `segments`).
% runlog: logs/run_<name>_<ts>.csv; cmdfile: data/cmd_<name>.csv ('' = log meta cmd_file).
% opts.video: results/video_<stem>.json from track_tip_video.py ('' = none) -> mechanics = video - DH(q_meas).
% opts.trace / opts.touchoff: legacy pen-on-paper path (off unless opts.trace is given).
% opts.session: load_session struct or file; opts.name: output name override (repeats).
% CV (second output): curves used, for repeatability summaries.
if nargin < 3, opts = struct(); end
root = hop3_root(); cfg = load_config();
runlog = absr(runlog);
o = struct('trace', '', 'touchoff', '', 'session', '', 'video', '', 'name', '');
for f = fieldnames(opts)', o.(f{1}) = opts.(f{1}); end
if isstruct(o.session), S = o.session; else, S = load_session(o.session); end

[~, stem] = fileparts(runlog);
tok = regexp(stem, '^run_(.+)_\d{8}_\d{6}$', 'tokens', 'once');
if isempty(tok), name = regexprep(stem, '^run_', ''); else, name = tok{1}; end
if ~isempty(o.name), name = o.name; end
metaf = fullfile(fileparts(runlog), [stem '.json']);
if isfile(metaf), meta = jsondecode(fileread(metaf)); else, meta = struct(); end
if nargin < 2 || isempty(cmdfile), cmdfile = meta.cmd_file; end
cmdfile = absr(cmdfile);
[cd_, cstem] = fileparts(cmdfile);
cmeta = struct('placeholder', false);
if isfile(fullfile(cd_, [cstem '.json'])), cmeta = jsondecode(fileread(fullfile(cd_, [cstem '.json']))); end
synthetic = getf(meta, 'synthetic', false) || getf(cmeta, 'synthetic', false) || S.synthetic;
mock = getf(meta, 'mock', false);

L = readtable(runlog); C = readtable(cmdfile);
n_rows = height(L); ok = L.comm_ok == 1;
L = L(ok, :);
kk = L.k + 1; t = L.t_s;
q_goal = counts_to_theta([L.goal_c1 L.goal_c2 L.goal_c3]', cfg);
q_meas = counts_to_theta([L.pos_c1 L.pos_c2 L.pos_c3]', cfg);
q_cmd = [C.th1 C.th2 C.th3]';
p_des = [C.x_des C.y_des C.z_des]';
p_nn = [C.x_nn C.y_nn C.z_nn]';
p_dhc = dh_fk(q_cmd, cfg);
p_goal = dh_fk(q_goal, cfg);
p_meas = dh_fk(q_meas, cfg);
nc = size(p_des, 2);
s_des = [0 cumsum(vecnorm(diff(p_des, 1, 2)))];

isB = isfield(cmeta, 'segments') || contains(lower(name), 'letterb') || strcmp(getf(cmeta, 'plane', ''), 'front');
isCircle = ~isB && isfield(cmeta, 'R') && isfield(cmeta, 'centre');
if isB
    Bg = b_geom(cmeta, cfg, name);
    lab_c = seg_labels(cmeta, p_des, Bg);
    ideal = b_ideal(Bg, 0.05);
    seg_names = {'spine', 'top', 'bottom'};
else
    lab_c = ones(1, nc); seg_names = {};
end

E.inv_residual = vecnorm(p_nn - p_des);
E.nn_model_err = vecnorm(p_dhc - p_nn);
E.tracking_temporal = vecnorm(p_meas - p_goal);
E.tracking_contour = dist_to_polyline(p_meas, p_dhc, false);

dt = median(diff(t));
[lag_j, lag_all, w_j, lag_xc] = estimate_lag(t, q_goal, q_meas, dt);
q_goal_lag = interp1(t, q_goal', t - lag_all, 'linear', 'extrap')';
E_lagcomp = vecnorm(p_meas - dh_fk(q_goal_lag, cfg));
dq = q_meas - q_goal; dq_lag = q_meas - q_goal_lag;

has_video = ~isempty(o.video);
has_pen = ~isempty(o.trace);
pv = []; tv = []; V = struct();
if has_video
    V = jsondecode(fileread(absr(o.video)));
    xpl = getf(V, 'plane_x_mm', mean(p_des(1,:)));
    tv = V.t_s(:)'; pv = [xpl*ones(1, numel(tv)); V.y_mm(:)'; V.z_mm(:)'];
    in = tv >= t(1) & tv <= t(end);
    pm_at_v = interp1(t, p_meas', tv(in), 'linear')';
    E.mechanics = nan(1, numel(tv));
    E.mechanics(in) = vecnorm(pv(2:3, in) - pm_at_v(2:3, :));
    E.total = dist_to_polyline(pv(2:3,:), p_des(2:3,:), false);
    total_basis = 'video - p_des (y-z contour in the letter plane; video synced to run time)';
    mech_basis = 'video - DH(q_meas) at equal run time (y-z)';
elseif has_pen
    cpen = cfg; cpen.dh.f = cfg.dh.f + S.pen_offset_mm;
    p_meas_pen = dh_fk(q_meas, cpen);
    trace = jsondecode(fileread(absr(o.trace)));
    if isempty(o.touchoff)
        sub = ''; if synthetic, sub = 'synthetic'; end
        o.touchoff = fullfile(root, 'results', sub, 'touchoff.json');
    end
    if isstruct(o.touchoff), TO = o.touchoff; else, TO = jsondecode(fileread(absr(o.touchoff))); end
    sp = trace.points_mm';
    pen = TO.R * [sp; zeros(1, size(sp,2))] + TO.t_mm(:);
    E.mechanics = dist_to_polyline(p_meas_pen(1:2,:), pen(1:2,:), true);
    E.total = dist_to_polyline(pen(1:2,:), p_des(1:2,:), false);
    total_basis = 'pen - p_des (xy contour; pen mapped by touch-off transform)';
    mech_basis = 'DH_pen(q_meas) - pen trace (xy contour)';
else
    E.mechanics = NaN;
    E.total = dist_to_polyline(p_meas, p_des, false);
    total_basis = 'DH(q_meas) - p_des (3-D contour; no video / pen ground truth)';
    mech_basis = 'null: needs video (or pen) ground truth';
end

kv = [];
if has_video, kv = round(interp1(C.t_s', 1:nc, min(max(tv, C.t_s(1)), C.t_s(end)), 'linear')); end
shape = NaN;
if isB
    shape = struct();
    shape.desired = shape_metrics(p_des, lab_c, Bg, ideal, false);
    shape.nn = shape_metrics(p_nn, lab_c, Bg, ideal, false);
    shape.dh_cmd = shape_metrics(p_dhc, lab_c, Bg, ideal, false);
    shape.dh_meas = shape_metrics(p_meas, lab_c(kk), Bg, ideal, false);
    if has_video, shape.video = shape_metrics(pv, lab_c(kv), Bg, ideal, true); else, shape.video = NaN; end
    if has_video, gt = shape.video; gt_src = 'video'; else, gt = shape.dh_meas; gt_src = 'DH(q_meas)'; end
    haus = gt.hausdorff_mm; haus_basis = [gt_src ' vs ideal B (letter plane)'];
end

cf = NaN; cf_meas = NaN; cf_pen = [];
if isCircle
    ctr = cmeta.centre(:); Rdes = cmeta.R;
    cf_meas = circle_metrics(p_meas(1:2,:), ctr, Rdes);
    if has_pen, cf_pen = circle_metrics(pen(1:2,:), ctr, Rdes); cf = cf_pen; cf_src = 'pen';
    else, cf = cf_meas; cf_src = 'DH(q_meas)'; end
    cf = rmfield(cf, 'centre_mm'); cf.source = cf_src;
end
if ~isB
    if has_pen, hc = pen(1:2,:); pd = p_des(1:2,:); hb = 'pen vs p_des (xy)';
    elseif has_video, hc = pv(2:3,:); pd = p_des(2:3,:); hb = 'video vs p_des (yz)';
    else, hc = p_meas; pd = p_des; hb = 'DH(q_meas) vs p_des (3-D)'; end
    pdd = densify(pd, 0.1);
    haus = max(max(min_dist(hc, pdd)), max(dist_to_polyline(pdd(:, 1:5:end), hc, false)));
    haus_basis = hb;
end

placeholder = S.placeholder || mock || getf(cmeta, 'placeholder', false) || getf(V, 'placeholder', false) || ...
    (has_pen && (getf(trace, 'placeholder', false) || getf(TO, 'placeholder', false)));

A = struct();
A.name = name; A.runlog = rel(runlog); A.cmdfile = rel(cmdfile);
A.placeholder = placeholder; A.synthetic = synthetic; A.mock = mock;
A.trajectory_kind = pick(isB, 'letterB', pick(isCircle, 'circle', 'generic'));
for f = {'inv_residual', 'nn_model_err', 'tracking_temporal', 'tracking_contour', 'mechanics', 'total'}
    A.(f{1}) = stats3(E.(f{1}));
end
if ~has_video && ~has_pen, A.mechanics = NaN; end
A.tracking_temporal_lagcomp = stats3(E_lagcomp);
A.total_basis = total_basis;
A.mechanics_basis = mech_basis;
A.lag_ms = 1000*lag_all;
A.lag_ms_per_joint = 1000*lag_j(:)';
A.lag_method = 'delay + backlash model fit (grid); xcorr lag kept as lag_xcorr_ms_per_joint';
A.lag_xcorr_ms_per_joint = 1000*lag_xc(:)';
A.backlash_halfwidth_deg_per_joint = w_j(:)';
A.per_joint_rms_deg = rms(dq, 2)';
A.per_joint_rms_lagcomp_deg = rms(dq_lag, 2)';
A.shape = shape;
A.circle_fit = cf;
if isCircle, A.circle_fit_dh_meas = cf_meas; end
if has_pen, A.circle_fit_pen = cf_pen; end
A.hausdorff_mm = haus;
A.hausdorff_basis = haus_basis;
A.n_rows = n_rows; A.n_dropped = n_rows - height(L); A.dt_median_s = dt;
A.video = pick(has_video, rel(o.video), NaN);
if has_video
    A.video_sync_offset_s = getf(V, 'sync_offset_s', NaN);
    A.video_reproj_rms_mm = getf(V, 'reproj_rms_mm', NaN);
end
A.pen_trace = pick(has_pen, o.trace, NaN);
A.pen_offset_mm = pick(has_pen, S.pen_offset_mm, NaN);
if isB
    A.segments = struct('name', seg_names, 's_start_mm', num2cell(arrayfun(@(i) s_des(find(lab_c == i, 1)), 1:3)), ...
        's_end_mm', num2cell(arrayfun(@(i) s_des(find(lab_c == i, 1, 'last')), 1:3)));
end

CV = struct('name', name, 't', t', 'p_meas', p_meas, 'p_des', p_des, 'kk', kk', 'pv', pv, 'tv', tv);

sub = ''; if synthetic, sub = 'synthetic'; end
rd = fullfile(root, 'results', sub); fd = fullfile(root, 'figures', sub);
if ~isfolder(rd), mkdir(rd); end
if ~isfolder(fd), mkdir(fd); end
fw = fopen(fullfile(rd, ['run_' name '.json']), 'w'); fwrite(fw, jsonencode(A, 'PrettyPrint', true)); fclose(fw);

tag = '';
if synthetic, tag = 'SYNTHETIC - '; end
if placeholder, tag = [tag 'PLACEHOLDER - ']; end
base = fullfile(fd, ['run_' name]);
col = struct('des', [0 0 0], 'nn', [0 0.7 0.9], 'dhc', [0 0.3 1], 'meas', [0.85 0.1 0.1], 'vid', [0 0.6 0]);
lg0 = {'p_{des}', 'NN(q_{cmd})', 'DH(q_{cmd})', 'DH(q_{meas})'};

f = figure('Visible', 'off', 'Position', [0 0 900 1000]); theme(f, 'light'); hold on; axis equal; grid on;
plot(p_des(2,:), p_des(3,:), '--', 'Color', col.des, 'LineWidth', 1.2);
plot(p_nn(2,:), p_nn(3,:), '-', 'Color', col.nn);
plot(p_dhc(2,:), p_dhc(3,:), '-', 'Color', col.dhc);
plot(p_meas(2,:), p_meas(3,:), '-', 'Color', col.meas);
lg = lg0;
if has_video, plot(pv(2,:), pv(3,:), '-', 'Color', col.vid, 'LineWidth', 1.1); lg{end+1} = 'video (tracked)'; end
xlabel('Y0 [mm]  (+y to the right, viewed from +X0)'); ylabel('Z0 [mm]');
legend(lg, 'Location', 'eastoutside');
ttl = [tag 'front view: ' name];
if isB
    g = gt;
    ttl = {ttl, sprintf('%s: contour RMS %.2f / max %.2f mm, spine straightness %.2f mm, R_top %.2f, R_bottom %.2f mm', ...
        gt_src, g.contour_rms_mm, g.contour_max_mm, g.spine_straightness_rms_mm, g.top_R_mm, g.bottom_R_mm)};
end
title(ttl, 'Interpreter', 'none');
exportgraphics(f, [base '_front.png'], 'Resolution', 300); close(f);

if isCircle
    f = figure('Visible', 'off', 'Position', [0 0 800 700]); theme(f, 'light'); hold on; axis equal; grid on;
    plot(p_des(1,:), p_des(2,:), '--', 'Color', col.des); plot(p_nn(1,:), p_nn(2,:), '-', 'Color', col.nn);
    plot(p_dhc(1,:), p_dhc(2,:), '-', 'Color', col.dhc); plot(p_meas(1,:), p_meas(2,:), '-', 'Color', col.meas);
    lg = lg0;
    if has_pen, plot(pen(1,:), pen(2,:), '-', 'Color', col.vid, 'LineWidth', 1.2); lg{end+1} = 'pen trace'; end
    xlabel('X0 [mm]'); ylabel('Y0 [mm]'); legend(lg, 'Location', 'best');
    title([tag 'top view: ' name], 'Interpreter', 'none');
    exportgraphics(f, [base '_topview.png'], 'Resolution', 300); close(f);
end

f = figure('Visible', 'off', 'Position', [0 0 1000 450]); theme(f, 'light'); hold on; grid on;
tc = C.t_s';
plot(tc, p_des(3,:), '--', 'Color', col.des); plot(tc, p_nn(3,:), '-', 'Color', col.nn);
plot(tc, p_dhc(3,:), '-', 'Color', col.dhc); plot(t, p_meas(3,:), '-', 'Color', col.meas);
lg = lg0;
if has_video, plot(tv, pv(3,:), '-', 'Color', col.vid); lg{end+1} = 'video (synced)'; end
if has_pen, yline(TO.plane.z_at_sheet_origin_mm, 'Color', col.vid, 'LineWidth', 1.2); lg{end+1} = 'paper plane (touch-off)'; end
xlabel('t [s]'); ylabel('Z0 [mm]'); legend(lg, 'Location', 'best');
title([tag 'tip height: ' name], 'Interpreter', 'none');
exportgraphics(f, [base '_z.png'], 'Resolution', 300); close(f);

f = figure('Visible', 'off', 'Position', [0 0 1000 900]); theme(f, 'light');
for j = 1:3
    subplot(3,1,j); hold on; grid on;
    plot(t, q_goal(j,:), '-', 'Color', col.dhc); plot(t, q_meas(j,:), '-', 'Color', col.meas);
    ylabel(sprintf('\\theta_%d [deg]', j));
    title(sprintf('%sjoint %d: RMS %.3f deg (lag-comp %.3f), lag %.0f ms, backlash \\pm%.2f deg', tag, j, ...
        A.per_joint_rms_deg(j), A.per_joint_rms_lagcomp_deg(j), A.lag_ms_per_joint(j), w_j(j)));
    if j == 1, legend({'goal', 'measured'}, 'Location', 'best'); end
end
xlabel('t [s]');
exportgraphics(f, [base '_joints.png'], 'Resolution', 300); close(f);

s_log = s_des(kk);
f = figure('Visible', 'off', 'Position', [0 0 1200 550]); theme(f, 'light'); hold on; grid on;
plot(s_des, E.inv_residual, 'Color', col.nn); plot(s_des, E.nn_model_err, 'Color', col.dhc);
plot(s_log, E.tracking_temporal, 'Color', [0.95 0.6 0.1]); plot(s_log, E.tracking_contour, 'Color', col.meas);
lg = {'inversion NN-p_{des}', 'NN model DH-NN', 'tracking (temporal)', 'tracking (contour)'};
if has_video
    s_v = s_des(kv);
    plot(s_v, E.mechanics, 'Color', col.vid); plot(s_v, E.total, 'k-', 'LineWidth', 1.2);
    lg = [lg {'mechanics video-DH(q_{meas})', 'total video-p_{des}'}];
elseif has_pen
    plot(s_log, E.mechanics, 'Color', col.vid); lg{end+1} = 'mechanics pen-DH(q_{meas})';
else
    plot(s_log, E.total, 'k-', 'LineWidth', 1.2); lg{end+1} = 'total DH(q_{meas})-p_{des} (no video)';
end
if isB
    yl = ylim;
    for i = 1:3
        xline(A.segments(i).s_start_mm, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
        text(A.segments(i).s_start_mm, yl(2), [' ' seg_names{i}], 'VerticalAlignment', 'top', 'Color', [0.3 0.3 0.3]);
    end
end
xlabel('path arc length along p_{des} [mm]'); ylabel('error [mm]'); xlim([0 s_des(end)]);
legend(lg, 'Location', 'eastoutside');
title(sprintf('%serror terms vs arc length: %s', tag, name), 'Interpreter', 'none');
exportgraphics(f, [base '_errors.png'], 'Resolution', 300); close(f);
end

function Bg = b_geom(cmeta, cfg, name)
key = 'letterB'; if contains(lower(name), 'outside'), key = 'letterB_outside'; end
lb = cfg.trajectories.(key);
Bg.x = lb.x_plane_mm; Bg.yz = lb.centre_yz(:)'; Bg.H = lb.H;
if isfield(cmeta, 'x_plane_mm'), Bg.x = cmeta.x_plane_mm; elseif isfield(cmeta, 'centre'), Bg.x = cmeta.centre(1); end
if isfield(cmeta, 'centre_yz'), Bg.yz = cmeta.centre_yz(:)'; elseif isfield(cmeta, 'centre'), Bg.yz = cmeta.centre(2:3)'; end
if isfield(cmeta, 'H'), Bg.H = cmeta.H; end
Bg.R = Bg.H/4; Bg.ys = Bg.yz(1) - Bg.H/8;
Bg.ctop = [Bg.ys; Bg.yz(2) + Bg.H/4]; Bg.cbot = [Bg.ys; Bg.yz(2) - Bg.H/4];
end

function P = b_ideal(Bg, step)
sp = Bg.yz(2) - Bg.H/2 : step : Bg.yz(2) + Bg.H/2;
ph = linspace(pi/2, -pi/2, ceil(pi*Bg.R/step) + 1);
yz = [[Bg.ys*ones(1, numel(sp)); sp], Bg.ctop + Bg.R*[cos(ph); sin(ph)], Bg.cbot + Bg.R*[cos(ph); sin(ph)]];
P = [Bg.x*ones(1, size(yz, 2)); yz];
end

function lab = seg_labels(cmeta, p_des, Bg)
n = size(p_des, 2); lab = zeros(1, n);
names = {'spine', 'top', 'bottom'};
if isfield(cmeta, 'segments')
    sg = cmeta.segments;
    if iscell(sg), sg = [sg{:}]; end
    for i = 1:numel(sg)
        j = find(cellfun(@(nm) startsWith(lower(sg(i).name), nm), names), 1);
        if isempty(j), continue; end
        lab(sg(i).k_start+1 : min(sg(i).k_end+1, n)) = j;
    end
    return;
end
yz = p_des(2:3,:);
d = [abs(yz(1,:) - Bg.ys) + 1e3*(yz(1,:) < Bg.ys - 1e-6); ...
     abs(vecnorm(yz - Bg.ctop) - Bg.R) + 1e3*(yz(1,:) < Bg.ys - 1e-6); ...
     abs(vecnorm(yz - Bg.cbot) - Bg.R) + 1e3*(yz(1,:) < Bg.ys - 1e-6)];
[~, lab] = min(d, [], 1);
end

function m = shape_metrics(P, lab, Bg, ideal, planar)
m = struct();
Ps = P(:, lab == 1);
if planar, Ps = Ps(2:3,:); end
c = mean(Ps, 2); [U, ~, ~] = svd(Ps - c, 'econ'); u = U(:,1);
pr = u'*(Ps - c); perp = Ps - c - u*pr;
m.spine_straightness_rms_mm = rms(vecnorm(perp));
m.spine_length_mm = max(pr) - min(pr);
m.spine_length_err_mm = m.spine_length_mm - Bg.H;
segs = {'top', Bg.ctop; 'bottom', Bg.cbot};
for i = 1:2
    yz = P(2:3, lab == i + 1);
    [cc, R] = circle_fit(yz);
    m.([segs{i,1} '_R_mm']) = R;
    m.([segs{i,1} '_R_err_mm']) = R - Bg.R;
    m.([segs{i,1} '_centre_err_mm']) = norm(cc - segs{i,2});
end
if planar, Q = P(2:3,:); I = ideal(2:3,:); else, Q = P; I = ideal; end
d = dist_to_polyline(Q, I, false);
m.contour_rms_mm = rms(d); m.contour_max_mm = max(d);
m.hausdorff_mm = max(m.contour_max_mm, max(dist_to_polyline(I(:, 1:10:end), Q, false)));
end

function [c, R] = circle_fit(xy)
[c, R] = kasa(xy);
for it = 1:20
    d = xy - c; r = vecnorm(d);
    J = [-(d ./ r)' -ones(numel(r), 1)];
    dx = -J \ (r - R)';
    c = c + dx(1:2); R = R + dx(3);
    if norm(dx) < 1e-10, break; end
end
end

function TH = counts_to_theta(Cn, cfg)
m = [cfg.map.theta1 cfg.map.theta2 cfg.map.theta3];
TH = [m.sign]' .* (Cn * 360 / cfg.encoder.counts_per_rev - [m.offset]');
end

function s = stats3(e)
e = e(~isnan(e));
if isempty(e), s = struct('rms_mm', NaN, 'max_mm', NaN, 'p95_mm', NaN); return; end
s = struct('rms_mm', rms(e), 'max_mm', max(e), 'p95_mm', prctile(e, 95));
end

function [lag_j, lag_all, w_j, lag_xc] = estimate_lag(t, g, m, dt)
% Lag from a delay + backlash (play operator) model fit on a (tau, w) grid. Plain cross-correlation is
% kept as lag_xc for reference only: backlash makes the output trail by a fixed angle, which xcorr reads as lag.
taus = 0:0.004:0.5; ws = 0:0.05:1.5;
tot = zeros(numel(taus), numel(ws));
lag_j = zeros(3,1); w_j = zeros(3,1); lag_xc = zeros(3,1);
for j = 1:3
    G = g(j,:)'; M = m(j,:)';
    C = zeros(numel(taus), numel(ws));
    for a = 1:numel(taus)
        u = interp1(t, G, t - taus(a), 'linear', G(1));
        for b = 1:numel(ws), C(a,b) = mean((M - play(u, ws(b))).^2); end
    end
    [cm, ib] = min(C, [], 2);
    lag_j(j) = min_lag(cm', taus);
    [~, ia] = min(cm); w_j(j) = ws(ib(ia));
    tot = tot + C/min(C(:));
    vg = movmean(gradient(G, t), round(0.2/dt)); vm = movmean(gradient(M, t), round(0.2/dt));
    [c, lags] = xcorr(vm - mean(vm), vg - mean(vg), round(0.5/dt));
    lag_xc(j) = peak_lag(c', lags*dt);
end
lag_all = min_lag(min(tot, [], 2)', taus);
end

function y = play(u, w)
y = u;
for k = 2:numel(u), y(k) = min(max(y(k-1), u(k) - w), u(k) + w); end
end

function L = min_lag(c, taus)
c(isnan(c)) = inf;
L = peak_lag(-c, taus);
end

function L = peak_lag(c, lags)
[~, i] = max(c);
L = lags(i);
if i > 1 && i < numel(c) && all(isfinite(c(i-1:i+1)))
    d = c(i-1) - 2*c(i) + c(i+1);
    if d ~= 0, L = L + 0.5*(c(i-1) - c(i+1))/d*(lags(2) - lags(1)); end
end
end

function d = dist_to_polyline(P, V, closed)
if closed, V = [V V(:,1)]; end
A = V(:,1:end-1); B = V(:,2:end); AB = B - A;
ab2 = sum(AB.^2, 1); ab2(ab2 == 0) = eps;
d = zeros(1, size(P,2));
for i = 1:size(P,2)
    AP = P(:,i) - A;
    u = min(max(sum(AP.*AB, 1)./ab2, 0), 1);
    d(i) = sqrt(min(sum((AP - AB.*u).^2, 1)));
end
end

function d = min_dist(P, Q)
d = zeros(1, size(P,2));
for i = 1:size(P,2), d(i) = sqrt(min(sum((Q - P(:,i)).^2, 1))); end
end

function V = densify(V, step)
s = [0 cumsum(vecnorm(diff(V, 1, 2)))];
[s, iu] = unique(s);
V = interp1(s, V(:, iu)', 0:step:s(end))';
end

function [c, R] = kasa(xy)
x = xy(1,:)'; y = xy(2,:)';
p = [x y ones(size(x))] \ (-(x.^2 + y.^2));
c = -p(1:2)/2; R = sqrt(sum(c.^2) - p(3));
end

function cf = circle_metrics(xy, ctr, Rdes)
[c, R] = kasa(xy);
cf = struct('R_mm', R, 'centre_err_mm', norm(c - ctr(1:2)), 'R_err_mm', R - Rdes, 'centre_mm', c');
end

function v = getf(s, name, def)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = def; end
end

function v = pick(c, a, b)
if c, v = a; else, v = b; end
end

function p = absr(p)
p = char(p);
if ~startsWith(p, filesep), p = fullfile(hop3_root(), p); end
end

function p = rel(p)
p = char(p); r = [hop3_root() filesep];
if startsWith(p, r), p = p(numel(r)+1:end); end
end
