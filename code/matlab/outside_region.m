function s = outside_region(which, shape)
% B7 outside-region experiment: invert the outside path with the box released, measure true (DH) error
% vs overshoot past the training box, the net's own error along rays leaving the box, and in_map detection.
% shape: 'letterB' (default, config trajectories.letterB_outside) or 'circle' (trajectories.outside).
if nargin < 1 || isempty(which), which = 'auto'; end
if nargin < 2, shape = 'letterB'; end
cfg = load_config(); W = load_weights(which);
box = [cfg.region.theta1(:)'; cfg.region.theta2(:)'; cfg.region.theta3(:)'];
mlo = W.in_map.xoffset; mhi = W.in_map.xoffset + 2./W.in_map.gain;
isB = strcmp(shape, 'letterB');
if isB
    Bo = make_letterB(cfg.trajectories.letterB_outside, cfg.stream.rate_hz); P = Bo.P;
else
    oc = cfg.trajectories.outside; P = make_circle(oc.centre, oc.R, oc.n_points);
end
[Q, info] = invert_fk(W, P, struct('cfg', cfg, 'leave_box', true));
e = vecnorm(dh_fk(Q, cfg) - P);
ov = max(max(box(:,1) - Q, Q - box(:,2)), 0);
ov1 = ov(1,:); ovm = max(ov, [], 1);
out = ovm > 0;
flag = any(Q < mlo - 1e-9 | Q > mhi + 1e-9, 1);
Qik = ik_analytic(P, cfg);

faces = {}; R = []; rng(3); nr = 20; d = -30:0.5:20; sn = {'lo', 'hi'};
for j = 1:3
    for side = [1 2]
        B = box(:,1) + rand(3, nr).*diff(box, 1, 2);
        Er = zeros(nr, numel(d));
        for i = 1:nr
            Qr = repmat(B(:,i), 1, numel(d));
            if side == 1, Qr(j,:) = box(j,1) - d; else, Qr(j,:) = box(j,2) + d; end
            Er(i,:) = vecnorm(fk_net_forward(W, Qr) - dh_fk(Qr, cfg));
        end
        nm = sprintf('theta%d_%s', j, sn{side});
        faces{end+1} = nm; %#ok<AGROW>
        R = [R; median(Er, 1)]; %#ok<AGROW>
        ray.(nm) = struct('median_err_mm_at_0', interp1(d, median(Er,1), 0), ...
            'median_err_mm_at_10', interp1(d, median(Er,1), 10), 'median_err_mm_at_20', interp1(d, median(Er,1), 20), ...
            'median_err_mm_inside_rms', rms(median(Er(:, d <= 0), 1)));
    end
end

s.frac_outside = mean(out);
s.err_inside_rms_mm = rms(e(~out));
s.err_outside_rms_mm = rms(e(out));
s.err_vs_overshoot = struct('overshoot_deg', ovm, 'err_mm', e);
s.theta1_overshoot_deg = ov1;
s.theta1_range_deg = [min(Q(1,:)) max(Q(1,:))];
s.theta1_ik_range_deg = [min(Qik(1,:)) max(Qik(1,:))];
s.nn_residual_max_mm = max(info.residual_mm);
s.err_outside_max_mm = max(e);
s.detect_in_map_frac_flagged = mean(flag);
s.detect_agrees_with_box = mean(flag == out);
s.in_map_min_deg = mlo'; s.in_map_max_deg = mhi';
s.rays = ray; s.ray_offsets_deg = d;
s.theta_overshoot_max_deg = max(ov, [], 2)';
s.shape = shape;
s.model_source = W.meta.source;
fid = fopen(fullfile(hop3_root(), 'results', 'outside.json'), 'w');
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true)); fclose(fid);

fd = fullfile(hop3_root(), 'figures');
f = figure('Visible', 'off', 'Position', [100 100 800 500]); theme(f, 'light');
scatter(ovm(~out), e(~out), 12, [0 0.45 0.74], 'filled'); hold on;
scatter(ovm(out), e(out), 12, [0.85 0.33 0.1], 'filled');
plot(ovm, info.residual_mm, 'k.', 'MarkerSize', 4);
grid on; xlabel('overshoot beyond training box (deg)'); ylabel('error (mm)');
legend('true error |dh(q)-p|, inside box', 'true error, outside box', 'net residual |NN(q)-p|', 'Location', 'northwest');
title(sprintf('Outside %s: confident-but-wrong (%s net)', shape, W.meta.source));
exportgraphics(f, fullfile(fd, 'outside_err_vs_overshoot.png'), 'Resolution', 300); close(f);

f = figure('Visible', 'off', 'Position', [100 100 800 500]); theme(f, 'light');
semilogy(d, max(R, 1e-4)', 'LineWidth', 1.3); hold on; xline(0, 'k--');
grid on; xlabel('degrees beyond box face (negative = inside)'); ylabel('median |NN(q) - dh(q)| (mm)');
legend(strrep(faces, '_', '\_'), 'Location', 'northwest');
title('Net error along rays leaving the training box');
exportgraphics(f, fullfile(fd, 'outside_rays.png'), 'Resolution', 300); close(f);

Pd = dh_fk(Q, cfg); Pn = fk_net_forward(W, Q);
f = figure('Visible', 'off', 'Position', [100 100 700 650]); theme(f, 'light');
if isB
    h = 1; v = 2; e1 = P(1,1)*tand(box(1,2));
else
    h = 0; v = 1;
end
plot(P(h+1,:), P(v+1,:), 'k-', 'LineWidth', 1.5); hold on;
plot(Pd(h+1,:), Pd(v+1,:), '-', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.2);
plot(Pn(h+1,:), Pn(v+1,:), ':', 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
plot(Pd(h+1,out), Pd(v+1,out), '.', 'Color', [0.85 0.33 0.1], 'MarkerSize', 8);
axis equal; grid on;
if isB
    xline(e1, 'k--');
    xlabel('y (mm)  (viewer at +X: +y to the right)'); ylabel('z (mm)');
    title(sprintf('Outside-region B, front view (plane x = %g mm)', P(1,1)));
    xlim([min([P(2,:) e1]) - 15, max([P(2,:) e1]) + 15]); ylim([min(P(3,:)) - 10, max(P(3,:)) + 10]);
    fn = 'outside_letterB_front.png';
else
    rr = [0 330]; plot(rr*cosd(box(1,2)), rr*sind(box(1,2)), 'k--');
    xlabel('x (mm)'); ylabel('y (mm)');
    title('Outside-region circle, top view');
    xlim([min(P(1,:)) - 40, max(P(1,:)) + 40]); ylim([min(P(2,:)) - 40, max(P(2,:)) + 40]);
    fn = 'outside_circle_topview.png';
end
legend('desired', 'actual dh\_fk(q_{inv})', 'net NN(q_{inv})', 'points with q outside box', ...
    sprintf('\\theta_1 = %g^\\circ box edge', box(1,2)), 'Location', 'southwest');
exportgraphics(f, fullfile(fd, fn), 'Resolution', 300); close(f);
end
