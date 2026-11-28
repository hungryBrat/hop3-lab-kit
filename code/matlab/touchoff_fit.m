function T = touchoff_fit(logfile, fiducials, S)
% Fit paper plane + rigid sheet->robot transform from a touch-off log.
% logfile: logs/touchoff_<ts>.csv (abs or root-relative); fiducials: data/fiducials.json path or struct;
% S: load_session() struct (default: session/session_data.json). Robot p = R*[s;0] + t, s in sheet mm.
root = hop3_root(); cfg = load_config();
if nargin < 3 || isempty(S), S = load_session(); end
if nargin < 2 || isempty(fiducials), fiducials = fullfile(root, S.fiducials_mm); end
if nargin < 1 || isempty(logfile), logfile = S.touchoff_log; end
logfile = absr(logfile);
if ischar(fiducials) || isstring(fiducials), fid = jsondecode(fileread(absr(fiducials))); else, fid = fiducials; end

tb = readtable(logfile, 'TextType', 'string');
ids = unique(tb.point_id, 'stable');
n = numel(ids);
P = zeros(3, n); spread = zeros(1, n); Ssheet = zeros(3, n);
c2 = cfg; c2.dh.f = cfg.dh.f + S.pen_offset_mm;
for i = 1:n
    rows = tb(tb.point_id == ids(i), :);
    C = [rows.pos_c1 rows.pos_c2 rows.pos_c3]';
    Pi = dh_fk(counts_to_theta(C, cfg), c2);
    P(:, i) = mean(Pi, 2);
    spread(i) = max(vecnorm(Pi - P(:, i)));
    Ssheet(:, i) = [S.scale * fid.touch_points.(char(ids(i))); 0];
end

A = [P(1,:)' P(2,:)' ones(n,1)];
abc = A \ P(3,:)';
nrm = [-abc(1); -abc(2); 1]; nrm = nrm / norm(nrm);
plane_res = P(3,:)' - A*abc;

[R, t] = kabsch(Ssheet, P);
res = R*Ssheet + t - P;
res_mm = vecnorm(res);
res_xy = vecnorm(res(1:2,:));

synthetic = S.synthetic || contains(logfile, 'synthetic');
T = struct();
T.logfile = logfile; T.synthetic = synthetic;
T.placeholder = S.placeholder;
T.point_ids = cellstr(ids)';
T.sheet_mm = Ssheet; T.robot_mm = P; T.rep_spread_mm = spread;
T.R = R; T.t_mm = t(:)'; T.yaw_deg = atan2d(R(2,1), R(1,1));
T.plane = struct('z_at_sheet_origin_mm', [t(1) t(2) 1]*abc, 'coef_z_eq_ax_by_c', abc', ...
    'normal', nrm', 'tilt_deg', acosd(nrm(3)), 'rms_mm', rms(plane_res));
T.residual_mm = res_mm; T.residual_xy_mm = res_xy;
T.residual_rms_mm = rms(res_mm); T.residual_max_mm = max(res_mm);
T.scale_applied = S.scale; T.pen_offset_mm = S.pen_offset_mm;
T.z_floor_suggested_mm = T.plane.z_at_sheet_origin_mm - 5;

sub = ''; if synthetic, sub = 'synthetic'; end
rd = fullfile(root, 'results', sub); fd = fullfile(root, 'figures', sub);
if ~isfolder(rd), mkdir(rd); end
if ~isfolder(fd), mkdir(fd); end
fidw = fopen(fullfile(rd, 'touchoff.json'), 'w');
fwrite(fidw, jsonencode(T, 'PrettyPrint', true)); fclose(fidw);

f = figure('Visible', 'off', 'Position', [0 0 900 700]); theme(f, 'light');
hold on; axis equal; grid on;
hw = S.scale*fid.sheet_mm(1)/2; hh = S.scale*fid.sheet_mm(2)/2;
corn = R*[hw -hw -hw hw hw; hh hh -hh -hh hh; zeros(1,5)] + t;
plot(corn(1,:), corn(2,:), '-', 'Color', [.6 .6 .6]);
tt = linspace(0, 2*pi, 361);
cc = R*[S.scale*(fid.circle.centre(1) + fid.circle.R*cos(tt)); S.scale*(fid.circle.centre(2) + fid.circle.R*sin(tt)); 0*tt] + t;
plot(cc(1,:), cc(2,:), 'r-');
Q = R*Ssheet + t;
plot(Q(1,:), Q(2,:), 'ks', 'MarkerSize', 10);
plot(P(1,:), P(2,:), 'b+', 'MarkerSize', 10, 'LineWidth', 1.5);
k = 20;
quiver(Q(1,:), Q(2,:), k*(P(1,:)-Q(1,:)), k*(P(2,:)-Q(2,:)), 0, 'b');
for i = 1:n
    text(P(1,i)+4, P(2,i)+4, sprintf('%s  %.2f mm', ids(i), res_mm(i)));
end
plot([0 40], [0 0], 'k-', [0 0], [0 40], 'k-'); text(42, 0, 'X0'); text(0, 44, 'Y0');
xlabel('X0 [mm]'); ylabel('Y0 [mm]');
legend({'sheet edge', 'printed circle', 'fiducial (fitted)', 'FK touch (mean)', sprintf('residual x%d', k)}, 'Location', 'bestoutside');
ttl = sprintf('Touch-off fit: yaw %.2f deg, t=(%.1f, %.1f, %.1f) mm, plane tilt %.2f deg, resid RMS %.2f mm', ...
    T.yaw_deg, t, T.plane.tilt_deg, T.residual_rms_mm);
if synthetic, ttl = ['SYNTHETIC - ' ttl]; end
if T.placeholder, ttl = [ttl ' [PLACEHOLDER]']; end
title(ttl, 'FontSize', 9);
exportgraphics(f, fullfile(fd, 'touchoff_fit.png'), 'Resolution', 300);
close(f);
end

function TH = counts_to_theta(C, cfg)
m = [cfg.map.theta1 cfg.map.theta2 cfg.map.theta3];
s = C * 360 / cfg.encoder.counts_per_rev;
TH = [m.sign]' .* (s - [m.offset]');
end

function [R, t] = kabsch(A, B)
ca = mean(A, 2); cb = mean(B, 2);
[U, ~, V] = svd((B - cb) * (A - ca)');
D = diag([1 1 sign(det(U*V'))]);
R = U * D * V';
t = cb - R*ca;
end

function p = absr(p)
p = char(p);
if ~startsWith(p, filesep), p = fullfile(hop3_root(), p); end
end
