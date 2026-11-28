function run_all(stage, which)
% stage: 'pre' (default) = everything before the session; 'post' = session analysis + report.
% run_all('post','synthetic') runs the post pipeline on data/synthetic/session_data_synth.json.
if nargin < 1, stage = 'pre'; end
if nargin < 2, which = ''; end
synth = strcmp(which, 'synthetic');
root = hop3_root();
py = sprintf('env -u LD_LIBRARY_PATH "%s"', fullfile(root, '.venv', 'bin', 'python'));
switch stage
    case 'pre'
        test_dh();
        make_dataset();
        plot_region();
        if exist('fitnet', 'file') == 2
            sweep_fk('fitnet');
            train_fk([40 20], 'trainlm', 5);
        elseif ~isfile(fullfile(root, 'models', 'fk_weights_standin.json'))
            sh(sprintf('cd "%s" && %s code/python/standin_train.py', root, py));
        end
        test_jacobian();
        eval_fk(load_weights());
        gen_trajectories();
        compare_ik();
        outside_region();
        test_invert();
        sh(sprintf('cd "%s" && %s code/python/make_video_board.py', root, py));
        sh(sprintf('cd "%s" && %s code/python/test_exec.py --quick', root, py));
    case 'post'
        sf = ''; if synth, sf = fullfile(root, 'data', 'synthetic', 'session_data_synth.json'); end
        S = load_session(sf);
        post(S, root, py);
end
rflag = ''; if synth, rflag = ' --synthetic'; end
if ~isfile(fullfile(root, 'report', 'main.tex')), return; end
sh(sprintf('cd "%s" && bash report/build.sh%s', root, rflag));
end

function sh(cmd)
[st, out] = system(cmd);
disp(out);
if st ~= 0, error('run_all:cmd', 'Command failed: %s', cmd); end
end

function post(S, root, py)
if S.use_pen
    touchoff_fit(S.touchoff_log, fullfile(root, S.fiducials_mm), S);
    for i = 1:numel(S.sheet_photos)
        sh(sprintf('cd "%s" && %s code/python/digitize_trace.py "%s" --session "%s"', root, py, S.sheet_photos{i}, S.file));
    end
end
logs = S.run_logs; n = numel(logs);
base = cell(1, n);
for i = 1:n
    [~, st] = fileparts(logs{i});
    tok = regexp(st, '^run_(.+)_\d{8}_\d{6}$', 'tokens', 'once');
    if isempty(tok), base{i} = regexprep(st, '^run_', ''); else, base{i} = tok{1}; end
end
vid = cell(1, n);
if numel(S.video) == n && n > 1
    vid = S.video;
elseif ~isempty(S.video)
    j = 1;
    if ~isempty(S.video_run), j = find(strcmp(logs, S.video_run), 1); end
    vid{j} = S.video{1};
end
res = {}; cvs = {};
for i = 1:n
    opts = struct('session', S);
    rep = find(strcmp(base, base{i}));
    if numel(rep) > 1 && find(rep == i) > 1, opts.name = sprintf('%s_rep%d', base{i}, find(rep == i)); end
    if S.use_pen && i <= numel(S.sheet_photos)
        [~, stem] = fileparts(S.sheet_photos{i});
        sub = ''; if S.synthetic, sub = 'synthetic'; end
        opts.trace = fullfile(root, 'results', sub, ['trace_' stem '.json']);
    end
    if ~isempty(vid{i})
        [~, vstem] = fileparts(vid{i});
        sh(sprintf('cd "%s" && %s code/python/track_tip_video.py "%s" --board-json "%s" --colour %s --run-log "%s"', ...
            root, py, vid{i}, S.video_board, S.video_colour, logs{i}));
        sub = ''; if contains(vid{i}, 'synthetic'), sub = 'synthetic'; end
        opts.video = fullfile('results', sub, ['video_' vstem '.json']);
    end
    [res{i}, cvs{i}] = analyze_run(fullfile(root, logs{i}), '', opts);
end
for b = unique(base)
    rep = find(strcmp(base, b{1}));
    if numel(rep) > 1, repeatability(b{1}, res(rep), cvs(rep), root); end
end
end

function repeatability(name, R, CV, root)
cr = cellfun(@(a) a.shape.dh_meas.contour_rms_mm, R(cellfun(@(a) isstruct(a.shape), R)));
tot = cellfun(@(a) a.total.rms_mm, R); lag = cellfun(@(a) a.lag_ms, R);
n = numel(CV); dmax = 0; drms = [];
for i = 1:n
    for j = i+1:n
        d = pt2poly(CV{i}.p_meas, CV{j}.p_meas);
        dmax = max(dmax, max(d)); drms(end+1) = rms(d); %#ok<AGROW>
        k = intersect(CV{i}.kk, CV{j}.kk);
        [~, a] = ismember(k, CV{i}.kk); [~, c] = ismember(k, CV{j}.kk);
        dmax_t(i, j) = max(vecnorm(CV{i}.p_meas(:, a) - CV{j}.p_meas(:, c))); %#ok<AGROW>
    end
end
Q = struct('name', name, 'n_repeats', n, 'runs', {cellfun(@(a) a.name, R, 'UniformOutput', false)}, ...
    'synthetic', any(cellfun(@(a) a.synthetic, R)), 'placeholder', any(cellfun(@(a) a.placeholder, R)), ...
    'contour_rms_mm', struct('mean', mean(cr), 'std', std(cr), 'values', cr), ...
    'total_rms_mm', struct('mean', mean(tot), 'std', std(tot), 'values', tot), ...
    'total_basis', {cellfun(@(a) a.total_basis, R, 'UniformOutput', false)}, ...
    'lag_ms', struct('mean', mean(lag), 'std', std(lag), 'values', lag), ...
    'max_deviation_between_repeats_mm', dmax, 'rms_deviation_between_repeats_mm', mean(drms), ...
    'max_deviation_same_k_mm', max(dmax_t(:)), ...
    'basis', 'DH(q_meas) path of each repeat; contour = nearest point on the other repeat''s path; same_k = equal command index');
sub = ''; if Q.synthetic, sub = 'synthetic'; end
fw = fopen(fullfile(root, 'results', sub, ['repeatability_' name '.json']), 'w');
fwrite(fw, jsonencode(Q, 'PrettyPrint', true)); fclose(fw);
fprintf('repeatability %s: contour RMS %.3f +- %.3f mm over %d runs; max deviation between repeats %.3f mm\n', ...
    name, Q.contour_rms_mm.mean, Q.contour_rms_mm.std, n, dmax);
end

function d = pt2poly(P, V)
A = V(:,1:end-1); B = V(:,2:end); AB = B - A;
ab2 = sum(AB.^2, 1); ab2(ab2 == 0) = eps;
d = zeros(1, size(P,2));
for i = 1:size(P,2)
    AP = P(:,i) - A;
    u = min(max(sum(AP.*AB, 1)./ab2, 0), 1);
    d(i) = sqrt(min(sum((AP - AB.*u).^2, 1)));
end
end
