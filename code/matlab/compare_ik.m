function s = compare_ik(which, shape)
% NN inversion vs analytic IK; figure figures/traj_compare_ik.png, summary results/traj_compare_ik.json.
% shape: 'letterB' (default; reference letterB, per_trajectory letterB + letterB_outside) or 'circle'
% (reference primary_z040 over the primary circle family).
if nargin < 1 || isempty(which), which = 'auto'; end
if nargin < 2, shape = 'letterB'; end
cfg = load_config(); W = load_weights(which); tr = cfg.trajectories;
if strcmp(shape, 'letterB')
    names = {'letterB', 'letterB_outside'}; leave = [false true]; ref = 'letterB';
    for i = 1:2
        B = make_letterB(tr.(names{i}), cfg.stream.rate_hz); paths{i} = B.P; xs{i} = B.t; %#ok<AGROW>
    end
    xlab = 'time (s)';
else
    pr = tr.primary; zs = pr.z_family(:)'; ref = 'primary_z040';
    names = arrayfun(@(z) sprintf('primary_z%03d', z), zs, 'UniformOutput', false);
    leave = false(size(zs));
    for i = 1:numel(zs)
        paths{i} = make_circle([pr.centre_xy(:); zs(i)], pr.R, pr.n_points); xs{i} = 1:pr.n_points; %#ok<AGROW>
    end
    xlab = 'circle point index';
end
closed = ~strcmp(shape, 'letterB');
for i = 1:numel(names)
    P = paths{i};
    [Q, info] = invert_fk(W, P, struct('cfg', cfg, 'leave_box', leave(i)));
    D = Q - ik_analytic(P, cfg); E = dh_fk(Q, cfg) - P; e = vecnorm(E);
    per(i) = struct('name', names{i}, 'q_nn_minus_ik_rms_deg', rms(D, 2)', ...
        'q_nn_minus_ik_max_deg', max(abs(D), [], 2)', 'dh_err_rms_mm', rms(e), 'dh_err_max_mm', max(e), ...
        'inv_residual_max_mm', max(info.residual_mm)); %#ok<AGROW>
    if strcmp(names{i}, ref), Dr = D; Er = E; er = e; Qr = Q; resr = info.residual_mm; xr = xs{i}; end
end
f = figure('Visible', 'off', 'Position', [100 100 900 700]); theme(f, 'light');
subplot(2, 1, 1);
plot(xr, Dr', 'LineWidth', 1.2); grid on; ylabel('q_{NN} - q_{IK} (deg)');
legend('\theta_1', '\theta_2', '\theta_3', 'Location', 'best');
title(strrep(sprintf('NN inversion vs analytic IK, %s (%s net)', ref, W.meta.source), '_', '\_'));
subplot(2, 1, 2);
plot(xr, Er', 'LineWidth', 1.0); hold on; plot(xr, er, 'k-', 'LineWidth', 1.4); grid on;
ylabel('dh\_fk(q_{NN}) - p_{des} (mm)'); xlabel(xlab);
legend('x', 'y', 'z', '|e|', 'Location', 'best');
exportgraphics(f, fullfile(hop3_root(), 'figures', 'traj_compare_ik.png'), 'Resolution', 300);
close(f);
s.name = 'compare_ik';
s.shape = shape;
s.reference = ref;
s.n_points = size(Qr, 2);
s.inv_residual_rms_mm = rms(resr); s.inv_residual_max_mm = max(resr);
s.dh_err_rms_mm = rms(er); s.dh_err_max_mm = max(er);
if closed, Qc = [Qr Qr(:,1)]; else, Qc = Qr; end
s.max_joint_step_deg = max(max(abs(diff(Qc, 1, 2))));
s.q_nn_minus_ik_rms_deg = rms(Dr, 2)';
s.within_limits = all(check_cmds(Qr, cfg).limits);
s.model_source = W.meta.source;
s.per_trajectory = per;
fid = fopen(fullfile(hop3_root(), 'results', 'traj_compare_ik.json'), 'w');
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true)); fclose(fid);
end
