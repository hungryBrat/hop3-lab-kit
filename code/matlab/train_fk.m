function [net, tr] = train_fk(arch, trainFcn, nSeeds)
if exist('fitnet') ~= 2 %#ok<EXIST>
    error('train_fk:noToolbox', 'fitnet not found. Install Deep Learning Toolbox; meanwhile run standin_train.py');
end
if nargin < 1 || isempty(arch), arch = [20 10]; end
if nargin < 2 || isempty(trainFcn), trainFcn = 'trainlm'; end
if nargin < 3 || isempty(nSeeds), nSeeds = 5; end
root = hop3_root(); md = fullfile(root, 'models'); fd = fullfile(root, 'figures');

data = readtable(fullfile(root, 'data', 'arm_workspace_points.csv'));
inputs = [data.theta1_deg, data.theta2_deg, data.theta3_deg]';
targets = [data.x, data.y, data.z]';
test = readtable(fullfile(root, 'data', 'test_points.csv'));
testIn = [test.theta1_deg, test.theta2_deg, test.theta3_deg]';
actualTest = [test.x, test.y, test.z]';
fprintf('pool %s, independent test %s\n', mat2str(size(inputs)), mat2str(size(testIn)));

rmse = zeros(nSeeds, 3); p95 = zeros(nSeeds, 1); vperf = zeros(nSeeds, 1);
epochs = zeros(nSeeds, 1); secs = zeros(nSeeds, 1); nets = cell(nSeeds, 1); trs = cell(nSeeds, 1);
for s = 1:nSeeds
    rng(s);
    n = fitnet(arch, trainFcn);
    if strcmp(trainFcn, 'trainbr')
        n.divideFcn = 'dividetrain';
    else
        n.divideFcn = 'dividerand';
        n.divideParam.trainRatio = 0.85;
        n.divideParam.valRatio = 0.15;
        n.divideParam.testRatio = 0;
    end
    n.performFcn = 'mse';
    n.trainParam.epochs = 1000;
    n.trainParam.max_fail = 10;
    n.trainParam.showWindow = false;
    n.trainParam.showCommandLine = false;
    t0 = tic; [n, t] = train(n, inputs, targets); secs(s) = toc(t0);
    e = actualTest - n(testIn);
    rmse(s,:) = sqrt(mean(e.^2, 2))';
    en = sort(sqrt(sum(e.^2, 1))); p95(s) = en(ceil(0.95*numel(en)));
    if strcmp(trainFcn, 'trainbr'), vperf(s) = t.best_perf; else, vperf(s) = t.best_vperf; end
    epochs(s) = t.best_epoch; nets{s} = n; trs{s} = t;
    fprintf('seed %d: test RMSE [%.4f %.4f %.4f] mm, p95 %.4f mm, best epoch %d (%s), %.0f s\n', ...
        s, rmse(s,:), p95(s), t.best_epoch, t.stop, secs(s));
end
% pick by validation performance so the independent test set stays untouched by selection
[~, b] = min(vperf);
net = nets{b}; tr = trs{b};

S = struct('arch', arch, 'train_fcn', trainFcn, 'seeds', 1:nSeeds, 'rmse_mm', rmse, ...
    'eucl_p95_mm', p95', 'best_epoch', epochs', 'train_time_s', secs', ...
    'rmse_mean_mm', mean(rmse, 1), 'rmse_std_mm', std(rmse, 0, 1), 'chosen_seed', b, ...
    'selection', 'lowest validation (trainbr: training) performance', ...
    'tr_stop', tr.stop, 'mu_final', tr.mu(end), 'epochs_run', tr.num_epochs);
fid = fopen(fullfile(root, 'results', 'fk_seeds.json'), 'w');
fprintf(fid, '%s', jsonencode(S, 'PrettyPrint', true)); fclose(fid);
fprintf('%d seeds: RMSE mean [%.4f %.4f %.4f] +- [%.4f %.4f %.4f] mm; chosen seed %d\n', ...
    nSeeds, S.rmse_mean_mm, S.rmse_std_mm, b);

predictedTest = net(testIn);
err = actualTest - predictedTest;
RMSE = sqrt(mean(err.^2, 2));
MAE = mean(abs(err), 2);
R2 = 1 - sum(err.^2, 2)./sum((actualTest - mean(actualTest, 2)).^2, 2);
fprintf('X RMSE = %.4f mm\nY RMSE = %.4f mm\nZ RMSE = %.4f mm\n', RMSE);
fprintf('X MAE = %.4f mm\nY MAE = %.4f mm\nZ MAE = %.4f mm\n', MAE);
fprintf('X R2 = %.6f\nY R2 = %.6f\nZ R2 = %.6f\n', R2);

f = figure('Visible', 'off', 'Theme', 'light', 'Position', [0 0 1000 800]);
names = {'X', 'Y', 'Z'};
for i = 1:3
    subplot(3,1,i);
    plot(actualTest(i,:), 'b', 'LineWidth', 1.5); hold on;
    plot(predictedTest(i,:), 'r--', 'LineWidth', 1.5);
    ylabel(names{i}); legend('Actual', 'Predicted'); grid on;
end
xlabel('Test samples (independent test set)');
sgtitle('Neural Network Forward Kinematics');
exportgraphics(f, fullfile(fd, 'fk_guide_actual_vs_pred.png'), 'Resolution', 300); close(f);

newAngles = [0; 45; -60];
newPosition = net(newAngles);
fprintf('\nPredicted Position for [%g; %g; %g] deg:\n', newAngles);
fprintf('X = %.4f mm\nY = %.4f mm\nZ = %.4f mm\n', newPosition);
fprintf('DH reference: (%.4f, %.4f, %.4f) mm\n', dh_fk(newAngles));

save(fullfile(md, 'Forward_Kinematics_Model.mat'), 'net', 'tr');
export_weights(net, fullfile(md, 'fk_weights.json'));
end
