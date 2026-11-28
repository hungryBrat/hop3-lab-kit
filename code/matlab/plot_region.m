function plot_region()
cfg = load_config(); root = hop3_root(); fd = fullfile(root, 'figures');
r = cfg.region; box = [r.theta1(:)'; r.theta2(:)'; r.theta3(:)'];
D = readmatrix(fullfile(root, 'data', 'arm_workspace_points.csv'));
Q = D(:,1:3)'; P = D(:,4:6)';

lim = cfg.servo_limits_deg; m = cfg.map;
rng(7); n = 40000;
S = [lim.j1(1) + diff(lim.j1)*rand(1,n); lim.j2(1) + diff(lim.j2)*rand(1,n); lim.j3(1) + diff(lim.j3)*rand(1,n)];
Qf = [m.theta1.sign*(S(1,:) - m.theta1.offset); m.theta2.sign*(S(2,:) - m.theta2.offset); ...
      m.theta3.sign*(S(3,:) - m.theta3.offset)];
Pf = dh_fk(Qf, cfg);
grey = [0.82 0.82 0.82];

f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 900 700]);
scatter3(P(1,:), P(2,:), P(3,:), 3, P(3,:), 'filled'); hold on;
draw_base(cfg); axis equal; grid on; colormap(parula); cb = colorbar; cb.Label.String = 'z [mm]';
xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]'); view(-35, 25);
title(sprintf('Selected region: %d DH samples, z >= %g mm', size(P,2), cfg.z_floor_mm));
exportgraphics(f, fullfile(fd, 'region_3d.png'), 'Resolution', 300); close(f);

rs = reach(Q, cfg); rf = reach(Qf, cfg);
f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1300 560]);
subplot(1,2,1);
scatter(Pf(1,:), Pf(2,:), 2, grey, 'filled'); hold on;
scatter(P(1,:), P(2,:), 2, P(3,:), 'filled'); plot(0, 0, 'k^', 'MarkerFaceColor', 'k');
axis equal; grid on; xlabel('x [mm]'); ylabel('y [mm]'); title('Top view (x-y)');
subplot(1,2,2);
scatter(rf, Pf(3,:), 2, grey, 'filled'); hold on;
scatter(rs, P(3,:), 2, P(3,:), 'filled');
yline(cfg.z_floor_mm, 'r--', 'z floor'); plot([0 0], [0 cfg.dh.d1], 'k-', 'LineWidth', 3);
axis equal; grid on; xlabel('in-plane reach r = L_2c_2 + L_tc_{23} [mm]'); ylabel('z [mm]');
title('Side view (r-z)'); colormap(parula); cb = colorbar; cb.Label.String = 'z [mm]';
sgtitle('Selected region (colour) over full joint-limit workspace (grey)');
exportgraphics(f, fullfile(fd, 'region_top_side.png'), 'Resolution', 300); close(f);

tr = cfg.trajectories;
f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1300 560]);
names = {}; C = {};
for zc = tr.primary.z_family(:)'
    C{end+1} = ik_up(circle([tr.primary.centre_xy(:); zc], tr.primary.R, tr.primary.n_points), cfg); %#ok<AGROW>
    names{end+1} = sprintf('primary z=%d', zc); %#ok<AGROW>
end
C{end+1} = ik_up(circle(tr.backup.centre(:), tr.backup.R, tr.backup.n_points), cfg); names{end+1} = 'backup';
C{end+1} = ik_up(circle(tr.outside.centre(:), tr.outside.R, tr.outside.n_points), cfg); names{end+1} = 'outside';
cols = [parula(numel(tr.primary.z_family)); 0.85 0.33 0.1; 0.8 0 0];
subplot(1,2,1); hold on;
draw_box(box);
for k = 1:numel(C), plot3(C{k}(1,:), C{k}(2,:), C{k}(3,:), 'LineWidth', 1.5, 'Color', cols(k,:)); end
grid on; view(-40, 20); xlabel('\theta_1 [deg]'); ylabel('\theta_2 [deg]'); zlabel('\theta_3 [deg]');
title('Joint-space training box with planned circles');
subplot(1,2,2); hold on;
rectangle('Position', [box(2,1) box(3,1) diff(box(2,:)) diff(box(3,:))], 'EdgeColor', 'k', 'LineWidth', 1.5);
h = gobjects(numel(C),1);
for k = 1:numel(C), h(k) = plot(C{k}(2,:), C{k}(3,:), 'LineWidth', 1.5, 'Color', cols(k,:)); end
grid on; axis equal; xlabel('\theta_2 [deg]'); ylabel('\theta_3 [deg]'); title('\theta_2-\theta_3 projection');
legend(h, names, 'Location', 'eastoutside');
exportgraphics(f, fullfile(fd, 'region_jointbox.png'), 'Resolution', 300); close(f);
for k = 1:numel(C)
    fprintf('%-14s th1 [%6.1f %6.1f] th2 [%6.1f %6.1f] th3 [%7.1f %7.1f]\n', names{k}, ...
        min(C{k}(1,:)), max(C{k}(1,:)), min(C{k}(2,:)), max(C{k}(2,:)), min(C{k}(3,:)), max(C{k}(3,:)));
end
end

function r = reach(Q, cfg)
Lt = cfg.dh.L3 + cfg.dh.f;
r = cfg.dh.L2*cosd(Q(2,:)) + Lt*cosd(Q(2,:) + Q(3,:));
end

function p = circle(c, R, n)
t = linspace(0, 2*pi, n + 1); t(end) = [];
p = [c(1) + R*cos(t); c(2) + R*sin(t); c(3)*ones(1, n)];
end

function draw_base(cfg)
[X, Y, Z] = cylinder(40, 40);
surf(X, Y, Z*cfg.dh.d1, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
end

function draw_box(b)
c = [b(1,[1 2 2 1 1 2 2 1]); b(2,[1 1 2 2 1 1 2 2]); b(3,[1 1 1 1 2 2 2 2])];
e = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
for k = 1:size(e,1), plot3(c(1,e(k,:)), c(2,e(k,:)), c(3,e(k,:)), 'k-'); end
end

function q = ik_up(p, cfg)
d = cfg.dh; Lt = d.L3 + d.f;
r = hypot(p(1,:), p(2,:)); h = p(3,:) - d.d1;
c3 = (r.^2 + h.^2 - d.L2^2 - Lt^2)/(2*d.L2*Lt);
t3 = -acosd(c3);
t2 = atan2d(h, r) - atan2d(Lt*sind(t3), d.L2 + Lt*cosd(t3));
q = [atan2d(p(2,:), p(1,:)); t2; t3];
end
