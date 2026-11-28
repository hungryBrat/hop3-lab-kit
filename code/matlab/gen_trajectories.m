function out = gen_trajectories(zc, which, opts)
% gen_trajectories()            letterB (primary) + letterB_outside (box released), then outside_region() for outside.json.
% gen_trajectories(zc)          only session_zc (primary circle at centre z = zc), fast, no figures.
% which: weights selector passed to load_weights (default 'auto').
% opts.circles (false): also the legacy circle family primary_zNNN, inair_zNNN, backup, outside.
if nargin < 2 || isempty(which), which = 'auto'; end
if nargin < 3, opts = struct(); end
cfg = load_config(); W = load_weights(which); tr = cfg.trajectories;
pr = tr.primary; cxy = pr.centre_xy(:);
if nargin >= 1 && ~isempty(zc)
    out = run_one('session_zc', [cxy; zc], pr, false, false, W, cfg, struct('zc', zc));
    return
end
out = {};
out{end+1} = run_B('letterB', tr.letterB, false, true, W, cfg);
out{end+1} = run_B('letterB_outside', tr.letterB_outside, true, true, W, cfg);
for i = 1:2
    fprintf('%-16s N %d  T %.2f s  res max %.4f  dh rms %.4f max %.4f  step %.3f  dq_ik rms [%s]  robust %d  export %s\n', ...
        out{i}.name, out{i}.n_points, out{i}.T_s, out{i}.inv_residual_max_mm, out{i}.dh_err_rms_mm, out{i}.dh_err_max_mm, ...
        out{i}.max_joint_step_deg, num2str(out{i}.q_nn_minus_ik_rms_deg, '%.4f '), out{i}.robust_to_signs, out{i}.export);
end
outside_region(which);
if ~(isfield(opts, 'circles') && opts.circles), out = [out{:}]; return, end
specs = {};
for z = pr.z_family(:)'
    specs(end+1,:) = {sprintf('primary_z%03d', z), [cxy; z], pr, false, struct('zc', z)}; %#ok<AGROW>
end
for z = pr.z_family(:)'
    zi = z + tr.inair_offset_mm;
    specs(end+1,:) = {sprintf('inair_z%03d', z), [cxy; zi], pr, false, struct('zc', zi)}; %#ok<AGROW>
end
specs(end+1,:) = {'backup', tr.backup.centre(:), tr.backup, false, struct()};
specs(end+1,:) = {'outside', tr.outside.centre(:), tr.outside, true, struct()};
figs = {'primary_z040', 'backup', 'outside'};
for i = 1:size(specs, 1)
    r = run_one(specs{i,1}, specs{i,2}, specs{i,3}, specs{i,4}, ...
        any(strcmp(specs{i,1}, figs)), W, cfg, specs{i,5});
    out{end+1} = r; %#ok<AGROW>
    fprintf('%-14s res max %.4f  dh rms %.3f max %.3f  step %.3f  dq_ik rms [%s]  export %s\n', r.name, ...
        r.inv_residual_max_mm, r.dh_err_rms_mm, r.dh_err_max_mm, r.max_joint_step_deg, ...
        num2str(r.q_nn_minus_ik_rms_deg, '%.3f '), r.export);
end
end

function r = run_B(name, spec, leave, fig, W, cfg)
B = make_letterB(spec, cfg.stream.rate_hz);
P = B.P;
[Q, info] = invert_fk(W, P, struct('cfg', cfg, 'leave_box', leave));
Qik = ik_analytic(P, cfg);
dh = vecnorm(dh_fk(Q, cfg) - P);
d = Q - Qik;
r.name = name;
r.n_points = size(P, 2);
r.centre = B.centre; r.H = B.H; r.R = B.R; r.T_s = B.T_s; r.rate_hz = B.rate_hz;
r.seg_T_s = B.seg_T_s; r.v_max_mm_s = spec.v_max_mm_s; r.corner_dwell_s = spec.corner_dwell_s;
r.inv_residual_rms_mm = rms(info.residual_mm);
r.inv_residual_max_mm = max(info.residual_mm);
r.dh_err_rms_mm = rms(dh);
r.dh_err_max_mm = max(dh);
r.max_joint_step_deg = max(max(abs(diff(Q, 1, 2))));
r.q_nn_minus_ik_rms_deg = rms(d, 2)';
r.q_nn_minus_ik_max_deg = max(abs(d), [], 2)';
r.theta_min_deg = min(Q, [], 2)'; r.theta_max_deg = max(Q, [], 2)';
r.n_converged = nnz(info.converged); r.n_hit_bound = nnz(info.hit_bound);
r.max_iterations = max(info.iterations); r.leave_box = leave;
r.model_source = W.meta.source;
r.segments = struct('name', {B.segments.name}, 'k_start', num2cell([B.segments.idx_start] - 1), ...
    'k_end', num2cell([B.segments.idx_end] - 1));
r.corners_k = B.corners - 1;
[r.robust_to_signs, r.sign_checks] = sign_robustness(Q, cfg);
meta = struct('centre', B.centre, 'R', B.R, 'H', B.H, 'T_s', B.T_s, 'timed', true, 'plane', 'front', ...
    'segments', B.segments, 'corners_k', B.corners - 1, 'leave_box', leave);
try
    [~, chk] = export_cmds(name, Q, P, W, cfg, meta);
    r.within_limits = chk.ok; r.max_row_step_deg = chk.max_step_deg; r.min_z_mm = chk.min_z_mm;
    r.servo_min_deg = chk.servo_min_deg; r.servo_max_deg = chk.servo_max_deg;
    r.export = 'written';
catch err
    chk = check_cmds(Q, cfg);
    r.within_limits = false; r.max_row_step_deg = NaN; r.min_z_mm = chk.min_z_mm;
    r.servo_min_deg = chk.servo_min_deg; r.servo_max_deg = chk.servo_max_deg;
    r.export = err.message;
end
write_json(['traj_' name '.json'], r);
if fig
    joint_fig(name, Q, Qik, B.t, B.t(B.corners), 'time (s)');
    shape_fig(name, B, Q, W, cfg, r);
end
end

function [ok, c] = sign_robustness(Q, cfg)
c = struct('sigma1', {}, 'sigma3', {}, 'ok', {}, 'servo_min_deg', {}, 'servo_max_deg', {});
for s1 = [1 -1]
    for s3 = [-1 1]
        cf = cfg; cf.map.theta1.sign = s1; cf.map.theta3.sign = s3;
        k = check_cmds(Q, cf);
        c(end+1) = struct('sigma1', s1, 'sigma3', s3, 'ok', k.limits && k.finite, ...
            'servo_min_deg', k.servo_min_deg, 'servo_max_deg', k.servo_max_deg); %#ok<AGROW>
    end
end
ok = all([c.ok]);
end

function r = run_one(name, c, spec, leave, fig, W, cfg, extra)
P = make_circle(c, spec.R, spec.n_points);
[Q, info] = invert_fk(W, P, struct('cfg', cfg, 'leave_box', leave));
Qik = ik_analytic(P, cfg);
dh = vecnorm(dh_fk(Q, cfg) - P);
d = Q - Qik;
r.name = name;
r.n_points = spec.n_points;
r.centre = c(:)'; r.R = spec.R;
r.inv_residual_rms_mm = rms(info.residual_mm);
r.inv_residual_max_mm = max(info.residual_mm);
r.dh_err_rms_mm = rms(dh);
r.dh_err_max_mm = max(dh);
r.max_joint_step_deg = max(max(abs(diff([Q Q(:,1)], 1, 2))));
r.q_nn_minus_ik_rms_deg = rms(d, 2)';
r.q_nn_minus_ik_max_deg = max(abs(d), [], 2)';
r.theta_min_deg = min(Q, [], 2)'; r.theta_max_deg = max(Q, [], 2)';
r.n_converged = nnz(info.converged); r.n_hit_bound = nnz(info.hit_bound);
r.max_iterations = max(info.iterations); r.leave_box = leave;
r.model_source = W.meta.source;
meta = extra; meta.centre = c(:)'; meta.R = spec.R; meta.T_s = spec.T_s; meta.leave_box = leave;
try
    [~, chk] = export_cmds(name, Q, P, W, cfg, meta);
    r.within_limits = chk.ok; r.max_row_step_deg = chk.max_step_deg; r.min_z_mm = chk.min_z_mm;
    r.export = 'written';
catch err
    chk = check_cmds(Q, cfg);
    r.within_limits = false; r.max_row_step_deg = NaN; r.min_z_mm = chk.min_z_mm;
    r.export = err.message;
end
write_json(['traj_' name '.json'], r);
if fig, joint_fig(name, Q, Qik, 1:size(Q, 2), [], 'circle point index'); end
end

function write_json(fn, s)
rd = fullfile(hop3_root(), 'results'); if ~isfolder(rd), mkdir(rd); end
fid = fopen(fullfile(rd, fn), 'w');
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true)); fclose(fid);
end

function joint_fig(name, Q, Qik, x, xl, xlab)
f = figure('Visible', 'off', 'Position', [100 100 900 700]); theme(f, 'light');
for j = 1:3
    subplot(3, 1, j);
    plot(x, Q(j,:), 'b-', 'LineWidth', 1.4); hold on;
    plot(x, Qik(j,:), 'r--', 'LineWidth', 1.0);
    for c = xl, xline(c, ':', 'Color', [0.4 0.4 0.4]); end
    ylabel(sprintf('\\theta_%d (deg)', j)); grid on; xlim([x(1) x(end)]);
    if j == 1, title(strrep(['Joint trajectory ' name], '_', '\_')); legend('NN inversion', 'analytic IK', 'Location', 'best'); end
end
xlabel(xlab);
save_fig(f, ['traj_' name '_joints.png']);
end

function shape_fig(name, B, Q, W, cfg, r)
P = B.P; Pn = fk_net_forward(W, Q); Pd = dh_fk(Q, cfg);
f = figure('Visible', 'off', 'Position', [100 100 1300 600]); theme(f, 'light');
subplot(1, 2, 1);
plot3(P(1,:), P(2,:), P(3,:), 'k-', 'LineWidth', 2); hold on;
seg = B.segments;
ks = [seg(1).idx_start, seg(1).idx_end, round(mean([seg(2).idx_start seg(2).idx_end]))];
cols = [0 0.45 0.74; 0.85 0.33 0.1; 0.47 0.67 0.19];
for i = 1:3
    J = arm_points(Q(:,ks(i)), cfg);
    plot3(J(1,:), J(2,:), J(3,:), '-o', 'Color', cols(i,:), 'LineWidth', 2, 'MarkerSize', 4, ...
        'MarkerFaceColor', cols(i,:));
end
plot3([0 0], [0 0], [0 cfg.dh.d1], 'k-', 'LineWidth', 4);
axis equal; grid on; view(-35, 20);
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
legend('desired B', 'arm at start (bottom)', 'arm at top corner', 'arm at top bump', 'Location', 'northwest');
title(strrep(sprintf('%s in plane x = %g mm (%s net)', name, B.centre(1), W.meta.source), '_', '\_'));
subplot(1, 2, 2);
plot(P(2,:), P(3,:), 'k-', 'LineWidth', 3); hold on;
plot(Pn(2,:), Pn(3,:), '--', 'Color', [0 0.45 0.74], 'LineWidth', 1.4);
plot(Pd(2,:), Pd(3,:), ':', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.8);
plot(P(2,B.corners), P(3,B.corners), 'ks', 'MarkerFaceColor', 'k');
axis equal; grid on; xlabel('y (mm)  (viewer at +X: +y to the right)'); ylabel('z (mm)');
xlim([min(P(2,:)) - 15, max(P(2,:)) + 15]); ylim([min(P(3,:)) - 10, max(P(3,:)) + 10]);
legend('desired', 'NN(q_{NN})', 'DH(q_{NN})', 'corners', 'Location', 'eastoutside');
title(sprintf('Front view: |DH - des| rms %.4f max %.4f mm', r.dh_err_rms_mm, r.dh_err_max_mm));
save_fig(f, ['traj_' name '_3d.png']);
end

function J = arm_points(q, cfg)
d = cfg.dh; c1 = cosd(q(1)); s1 = sind(q(1));
u2 = [c1*cosd(q(2)); s1*cosd(q(2)); sind(q(2))];
u3 = [c1*cosd(q(2)+q(3)); s1*cosd(q(2)+q(3)); sind(q(2)+q(3))];
sh = [0; 0; d.d1]; el = sh + d.L2*u2; wr = el + d.L3*u3; tip = wr + d.f*u3;
J = [sh el wr tip];
end

function save_fig(f, fn)
fd = fullfile(hop3_root(), 'figures'); if ~isfolder(fd), mkdir(fd); end
exportgraphics(f, fullfile(fd, fn), 'Resolution', 300);
close(f);
end
