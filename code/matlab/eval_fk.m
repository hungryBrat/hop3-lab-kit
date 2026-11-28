function R = eval_fk(W)
if nargin < 1, W = load_weights(); end
if ischar(W) || isstring(W), W = load_weights(W); end
cfg = load_config(); root = hop3_root(); fd = fullfile(root, 'figures');
r = cfg.region; box = [r.theta1(:)'; r.theta2(:)'; r.theta3(:)'];
standin = strcmp(W.meta.source, 'STANDIN');
tag = ''; if standin, tag = ' [STANDIN]'; end
arch = W.meta.arch(:)';

D = readmatrix(fullfile(root, 'data', 'arm_workspace_points.csv'));
T = readmatrix(fullfile(root, 'data', 'test_points.csv'));
J = readmatrix(fullfile(root, 'data', 'traj_test_points.csv'));
Qt = T(:,1:3)'; Pt = T(:,4:6)';
Mt = metrics(Pt, fk_net_forward(W, Qt));
Mj = metrics(J(:,4:6)', fk_net_forward(W, J(:,1:3)'));

A = [D(:,1:3), ones(size(D,1),1)];
C = A \ D(:,4:6);
lin = sqrt(mean(([Qt', ones(size(Qt,2),1)]*C - Pt').^2, 1));

et = fk_net_forward(W, Qt) - Pt; en = sqrt(sum(et.^2, 1));
[dface, iface] = min([Qt - box(:,1); box(:,2) - Qt], [], 1);

edges = [0 2.5 5 10 15 20 30 60];
bc = edges(1:end-1) + diff(edges)/2; bm = nan(size(bc)); bp = bm; bn = bm;
for k = 1:numel(bc)
    s = dface >= edges(k) & dface < edges(k+1);
    bn(k) = nnz(s);
    if bn(k), bm(k) = mean(en(s)); bp(k) = prctile_(en(s), 95); end
end
f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 800 500]);
plot(bc, bm, 'o-', 'LineWidth', 1.5); hold on; plot(bc, bp, 's--', 'LineWidth', 1.5);
grid on; xlabel('distance to nearest joint-box face [deg]'); ylabel('Euclidean error [mm]');
legend('mean', 'p95'); title(['Test error vs distance to region boundary' tag]);
exportgraphics(f, fullfile(fd, 'fk_err_vs_boundary.png'), 'Resolution', 300); close(f);

t2 = linspace(box(2,1), box(2,2), 116); t3 = linspace(box(3,1), box(3,2), 116);
[G2, G3] = meshgrid(t2, t3); sl = [-50 0 50];
f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1500 450]);
H = cell(1,3); hmax = 0;
for k = 1:3
    Q = [sl(k)*ones(1, numel(G2)); G2(:)'; G3(:)'];
    Ptrue = dh_fk(Q, cfg);
    e = sqrt(sum((fk_net_forward(W, Q) - Ptrue).^2, 1));
    e(Ptrue(3,:) < cfg.z_floor_mm) = NaN;
    H{k} = reshape(e, size(G2)); hmax = max(hmax, max(H{k}(:)));
end
for k = 1:3
    subplot(1,3,k);
    imagesc(t2, t3, H{k}, 'AlphaData', ~isnan(H{k})); set(gca, 'YDir', 'normal');
    clim([0 hmax]); colorbar; axis square;
    xlabel('\theta_2 [deg]'); ylabel('\theta_3 [deg]'); title(sprintf('|e| [mm], \\theta_1 = %d', sl(k)));
end
sgtitle(['Network error over (\theta_2,\theta_3); blank = below z floor' tag]);
exportgraphics(f, fullfile(fd, 'fk_err_heatmaps.png'), 'Resolution', 300); close(f);

[~, ord] = sort(en, 'descend'); w = ord(1:ceil(0.01*numel(en)));
f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 900 700]);
scatter3(Pt(1,:), Pt(2,:), Pt(3,:), 2, [0.8 0.8 0.8], 'filled'); hold on;
scatter3(Pt(1,w), Pt(2,w), Pt(3,w), 25, en(w), 'filled');
cb = colorbar; cb.Label.String = '|e| [mm]'; axis equal; grid on; view(-35, 25);
xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]');
title(['Worst 1% test errors (colour) in task space' tag]);
exportgraphics(f, fullfile(fd, 'fk_worst1pct_3d.png'), 'Resolution', 300); close(f);

f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1000 800]);
names = {'X', 'Y', 'Z'}; ns = min(300, size(Pt,2)); Pn = fk_net_forward(W, Qt);
for i = 1:3
    subplot(3,1,i);
    plot(Pt(i,1:ns), 'b', 'LineWidth', 1.5); hold on; plot(Pn(i,1:ns), 'r--', 'LineWidth', 1.5);
    ylabel([names{i} ' [mm]']); legend('Actual (DH)', 'Predicted (NN)'); grid on;
end
xlabel(sprintf('Independent test samples (first %d of %d)', ns, size(Pt,2)));
sgtitle(['Neural Network Forward Kinematics' tag]);
exportgraphics(f, fullfile(fd, 'fk_actual_vs_pred.png'), 'Resolution', 300); close(f);

jn = {'theta1', 'theta2', 'theta3'}; sd = {'lower', 'upper'};
wf = mode(iface(w)); fw = mean(iface(w) == wf & dface(w) < 10);
near = dface < 5; far = dface >= 20;
rr = reach(Qt, cfg);
sentence = sprintf(['Error is largest at the region boundary: mean |e| %.3f mm within 5 deg of a box face vs ' ...
    '%.3f mm deeper than 20 deg; %.0f%% of the worst-1%% test points lie within 10 deg of the %s %s face ' ...
    '(%s = %g deg), mean in-plane reach of worst 1%% %.0f mm vs %.0f mm overall.'], ...
    mean(en(near)), mean(en(far)), 100*fw, jn{1 + mod(wf-1, 3)}, sd{1 + (wf > 3)}, ...
    jn{1 + mod(wf-1, 3)}, box(1 + mod(wf-1, 3), 1 + (wf > 3)), mean(rr(w)), mean(rr));

nw = sum(cellfun(@(L) numel(L.W) + numel(L.b), W.layers));
sf = fullfile(root, 'results', 'fk_seeds.json');
smean = NaN; sstd = NaN;
if isfile(sf)
    S = jsondecode(fileread(sf));
    if isequal(S.arch(:)', arch) && strcmp(S.train_fcn, W.meta.train_fcn)
        smean = S.rmse_mean_mm(:)'; sstd = S.rmse_std_mm(:)';
    end
end
R = struct('model_source', W.meta.source, 'weights_file', W.file, 'arch', arch, ...
    'train_fcn', W.meta.train_fcn, 'n_weights', nw, ...
    'rmse_mm', Mt.rmse, 'mae_mm', Mt.mae, 'r2', Mt.r2, ...
    'eucl_mean_mm', Mt.mean, 'eucl_median_mm', Mt.median, 'eucl_p95_mm', Mt.p95, 'eucl_max_mm', Mt.max, ...
    'seeds_rmse_mean_mm', smean, 'seeds_rmse_std_mm', sstd, ...
    'linear_baseline_rmse_mm', lin, 'largest_error_region', sentence, ...
    'n_test', size(Qt,2), 'traj_test', Mj, ...
    'err_vs_boundary', struct('bin_edges_deg', edges, 'n', bn, 'mean_mm', bm, 'p95_mm', bp));
fid = fopen(fullfile(root, 'results', 'fk_eval.json'), 'w');
fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

fprintf('%s %s on %d test pts: RMSE [%.4f %.4f %.4f] mm, R2 [%.6f %.6f %.6f]\n', W.meta.source, ...
    mat2str(arch), size(Qt,2), Mt.rmse, Mt.r2);
fprintf('  Euclid mean %.4f median %.4f p95 %.4f max %.4f mm\n', Mt.mean, Mt.median, Mt.p95, Mt.max);
fprintf('  traj-test: RMSE [%.4f %.4f %.4f] p95 %.4f max %.4f mm\n', Mj.rmse, Mj.p95, Mj.max);
fprintf('  linear baseline RMSE [%.2f %.2f %.2f] mm\n', lin);
fprintf('  %s\n', sentence);
end

function M = metrics(P, Pn)
e = P - Pn; en = sqrt(sum(e.^2, 1));
M = struct('rmse', sqrt(mean(e.^2, 2))', 'mae', mean(abs(e), 2)', ...
    'r2', (1 - sum(e.^2, 2)./sum((P - mean(P, 2)).^2, 2))', ...
    'mean', mean(en), 'median', median(en), 'p95', prctile_(en, 95), 'max', max(en));
end

function p = prctile_(x, q)
x = sort(x(:)); n = numel(x);
p = interp1((0.5:n)/n, x, q/100, 'linear', 'extrap');
p = min(max(p, x(1)), x(end));
end

function r = reach(Q, cfg)
Lt = cfg.dh.L3 + cfg.dh.f;
r = cfg.dh.L2*cosd(Q(2,:)) + Lt*cosd(Q(2,:) + Q(3,:));
end
