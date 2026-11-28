function S = sweep_fk(mode, nSeeds, standinEpochs, nPar)
% mode 'fitnet' (Deep Learning Toolbox) or 'standin' (numpy LM clone, outputs labelled STANDIN).
if nargin < 1 || isempty(mode), mode = 'fitnet'; end
if nargin < 2 || isempty(nSeeds), nSeeds = 5; end
if nargin < 3 || isempty(standinEpochs), standinEpochs = 1000; end
if nargin < 4 || isempty(nPar), nPar = 6; end
standin = strcmp(mode, 'standin');
if ~standin && exist('fitnet') ~= 2 %#ok<EXIST>
    error('sweep_fk:noToolbox', 'fitnet not found. Install Deep Learning Toolbox; meanwhile run sweep_fk(''standin'')');
end
cfg = load_config(); root = hop3_root(); fd = fullfile(root, 'figures');
D = readmatrix(fullfile(root, 'data', 'arm_workspace_points.csv'));
T = readmatrix(fullfile(root, 'data', 'test_points.csv'));
Qt = T(:,1:3)'; Pt = T(:,4:6)';

sizesReq = [500 2000 8000 25000];
sizes = min(sizesReq, size(D,1));
archs = {5, 10, [20 10], [40 20]};
anames = cellfun(@mat2str, archs, 'UniformOutput', false);

% Trajectory error is invert-free: the net is evaluated on the analytic-IK joints of the letter B.
LB = make_letterB(cfg.trajectories.letterB, cfg.stream.rate_hz);
pdes = LB.P(:, 1:5:end);
qik = ik_up(pdes, cfg);

ni = numel(sizes); nj = numel(archs);
rmse = nan(ni, nj, nSeeds); trms = rmse; tmax = rmse;
tag = ''; if standin, tag = ' [STANDIN]'; end

if standin
    wd = fullfile(root, 'models', 'sweep_standin'); if ~isfolder(wd), mkdir(wd); end
    jobs = {}; files = cell(ni, nj, nSeeds);
    for i = 1:ni, for j = 1:nj, for s = 1:nSeeds
        rel = sprintf('models/sweep_standin/a%s_n%d_s%d.json', strjoin(string(archs{j}), '-'), sizes(i), s);
        files{i,j,s} = fullfile(root, rel);
        jobs{end+1} = sprintf('.venv/bin/python code/python/standin_train.py --arch %s --n %d --seed %d --epochs %d --out %s', ...
            num2str(archs{j}), sizes(i), s, standinEpochs, rel); %#ok<AGROW>
    end, end, end
    jf = fullfile(wd, 'jobs.txt'); fid = fopen(jf, 'w'); fprintf(fid, '%s\n', jobs{:}); fclose(fid);
    t0 = tic;
    st = system(sprintf('cd "%s" && xargs -P %d -d ''\\n'' -I CMD sh -c CMD < "%s" > "%s" 2>&1', ...
        root, nPar, jf, fullfile(wd, 'jobs.log')));
    assert(st == 0, 'sweep_fk: a stand-in training job failed, see %s', fullfile(wd, 'jobs.log'));
    fprintf('stand-in sweep: %d jobs in %.1f min\n', numel(jobs), toc(t0)/60);
end

for i = 1:ni, for j = 1:nj, for s = 1:nSeeds
    if standin
        W = load_weights(files{i,j,s});
        f = @(Q) fk_net_forward(W, Q);
    else
        rng(s);
        net = fitnet(archs{j}, 'trainlm');
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.85; net.divideParam.valRatio = 0.15; net.divideParam.testRatio = 0;
        net.performFcn = 'mse'; net.trainParam.epochs = standinEpochs; net.trainParam.max_fail = 10;
        net.trainParam.showWindow = false; net.trainParam.showCommandLine = false;
        net = train(net, D(1:sizes(i),1:3)', D(1:sizes(i),4:6)');
        f = @(Q) net(Q);
    end
    e = f(Qt) - Pt; rmse(i,j,s) = sqrt(mean(e(:).^2));
    et = sqrt(sum((f(qik) - pdes).^2, 1)); trms(i,j,s) = sqrt(mean(et.^2)); tmax(i,j,s) = max(et);
    fprintf('N=%5d %-8s seed %d: test RMSE %.4f mm, traj RMS %.4f max %.4f mm\n', ...
        sizes(i), anames{j}, s, rmse(i,j,s), trms(i,j,s), tmax(i,j,s));
end, end, end

src = 'fitnet'; if standin, src = 'STANDIN'; end
S = struct('model_source', src, 'sizes', sizes, 'sizes_requested', sizesReq, 'archs', {anames}, ...
    'n_seeds', nSeeds, 'rmse_mean_mm', mean(rmse, 3), 'rmse_std_mm', std(rmse, 0, 3), ...
    'traj_err_rms_mm', mean(trms, 3), 'traj_err_rms_std_mm', std(trms, 0, 3), 'traj_err_max_mm', mean(tmax, 3), ...
    'rmse_def', 'sqrt(mean over test points and x,y,z of squared error), independent test set', ...
    'traj_def', 'invert-free: ||net(q_IK) - p_des|| over the letter B (every 5th 50 Hz sample), analytic elbow-up IK', ...
    'epochs_cap', standinEpochs, 'rows_i_sizes_cols_j_archs', true);
fid = fopen(fullfile(root, 'results', 'sweep.json'), 'w');
fprintf(fid, '%s', jsonencode(S, 'PrettyPrint', true)); fclose(fid);

f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1300 520]);
M = {S.rmse_mean_mm, S.traj_err_rms_mm}; tt = {'Test RMSE [mm]', 'Trajectory error RMS [mm]'};
for p = 1:2
    subplot(1,2,p); imagesc(log10(M{p})); cb = colorbar; cb.Label.String = 'log_{10} mm';
    set(gca, 'XTick', 1:nj, 'XTickLabel', anames, 'YTick', 1:ni, 'YTickLabel', string(sizes));
    xlabel('hidden layers'); ylabel('dataset size N');
    for i = 1:ni, for j = 1:nj
        text(j, i, sprintf('%.3g', M{p}(i,j)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
    end, end
    title([tt{p} tag]);
end
exportgraphics(f, fullfile(fd, 'sweep_heatmap.png'), 'Resolution', 300); close(f);

f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1300 520]);
SD = {S.rmse_std_mm, S.traj_err_rms_std_mm};
for p = 1:2
    subplot(1,2,p); hold on;
    for j = 1:nj, errorbar(sizes, M{p}(:,j), SD{p}(:,j), 'o-', 'LineWidth', 1.5); end
    set(gca, 'XScale', 'log', 'YScale', 'log'); grid on;
    xlabel('dataset size N'); ylabel(tt{p}); legend(anames, 'Location', 'northeast');
    title(sprintf('%s, mean \\pm std over %d seeds%s', tt{p}, nSeeds, tag));
end
exportgraphics(f, fullfile(fd, 'sweep_lines.png'), 'Resolution', 300); close(f);
end

function q = ik_up(p, cfg)
d = cfg.dh; Lt = d.L3 + d.f;
r = hypot(p(1,:), p(2,:)); h = p(3,:) - d.d1;
c3 = (r.^2 + h.^2 - d.L2^2 - Lt^2)/(2*d.L2*Lt);
t3 = -acosd(c3);
t2 = atan2d(h, r) - atan2d(Lt*sind(t3), d.L2 + Lt*cosd(t3));
q = [atan2d(p(2,:), p(1,:)); t2; t3];
end
