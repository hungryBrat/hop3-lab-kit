function S = load_session(file)
% Session values with defaults for nulls. S.missing lists defaulted required fields; S.placeholder = any missing.
% Pen-on-paper fields are optional (S.use_pen=false by default) and never count as missing unless use_pen.
root = hop3_root();
if nargin < 1 || isempty(file), file = fullfile(root, 'session', 'session_data.json'); end
cfg = load_config();
raw = jsondecode(fileread(file));
S = raw; S.file = file; S.missing = {};

S.use_pen = isfield(raw, 'use_pen') && isequal(raw.use_pen, true);
prim = 'letterB'; if isfield(cfg, 'primary_trajectory'), prim = cfg.primary_trajectory; end
defaults = struct( ...
    'sigma1', cfg.map.theta1.sign, ...
    'sigma3', cfg.map.theta3.sign, ...
    'chosen_trajectory', prim);
for f = fieldnames(defaults)'
    S = fill(S, f{1}, defaults.(f{1}));
end
S = fill(S, 'neutral_tip_height_mm', cfg.dh.d1 + cfg.dh.L2 + cfg.dh.L3 + cfg.dh.f, false);
pen = struct('pen_offset_mm', 0, 'printed_scale_check_mm', 100, 'fiducials_mm', 'data/fiducials.json');
for f = fieldnames(pen)'
    S = fill(S, f{1}, pen.(f{1}), S.use_pen);
end
vid = struct('video', '', 'video_run', '', 'video_board', 'data/video_board.json', 'video_colour', 'orange');
for f = fieldnames(vid)'
    S = fill(S, f{1}, vid.(f{1}), false);
end

if ~isfield(S, 'z_c_mm') || isempty(S.z_c_mm)
    meta = fullfile(root, 'data', ['cmd_' S.chosen_trajectory '.json']);
    if isfile(meta)
        m = jsondecode(fileread(meta)); zc = m.centre(3);
    elseif isfield(cfg.trajectories, S.chosen_trajectory) && isfield(cfg.trajectories.(S.chosen_trajectory), 'centre_yz')
        zc = cfg.trajectories.(S.chosen_trajectory).centre_yz(2);
    else
        zc = 40;
    end
    S = fill(S, 'z_c_mm', zc);
end
S = fill(S, 'paper_plane_z_mm', S.z_c_mm, S.use_pen);

if ~isfield(S, 'touchoff_log') || isempty(S.touchoff_log)
    d = dir(fullfile(root, 'logs', 'touchoff_*.csv'));
    if isempty(d), S = fill(S, 'touchoff_log', '', S.use_pen);
    else, [~, i] = max([d.datenum]); S = fill(S, 'touchoff_log', fullfile('logs', d(i).name), S.use_pen); end
end
if ~isfield(S, 'run_logs') || isempty(S.run_logs), S.run_logs = {}; S.missing{end+1} = 'run_logs'; end
if ~isfield(S, 'sheet_photos') || isempty(S.sheet_photos)
    S.sheet_photos = {}; if S.use_pen, S.missing{end+1} = 'sheet_photos'; end
end
if isempty(S.video), S.video = {}; end
S.video = cellstr(S.video);
S.run_logs = cellstr(S.run_logs); S.sheet_photos = cellstr(S.sheet_photos);

S.scale = S.printed_scale_check_mm / 100;
S.synthetic = isfield(raw, 'synthetic') && isequal(raw.synthetic, true);
S.placeholder = ~isempty(S.missing);
if S.sigma1 ~= cfg.map.theta1.sign || S.sigma3 ~= cfg.map.theta3.sign
    warning('hop3:sigma', 'session sigma1/sigma3 differ from config.json map; config.json is what the analysis uses.');
end
end

function S = fill(S, name, val, required)
if nargin < 4, required = true; end
if ~isfield(S, name) || isempty(S.(name))
    S.(name) = val;
    if required, S.missing{end+1} = name; end
end
end
