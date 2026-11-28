function [cmd, chk] = export_cmds(name, Q, Pdes, W, cfg, meta)
% Write data/cmd_<name>.csv + .json. Default: resample a closed circle (Q 3xn deg at Pdes 3xn mm) to meta.T_s
% at the stream rate. meta.timed = true: Q/Pdes are already sampled at the stream rate, written as is;
% meta.segments (1-based idx_start/idx_end) is written to the json as {name, k_start, k_end} in k-column values.
% Errors without writing if check_cmds fails on the commanded counts.
if nargin < 5 || isempty(cfg), cfg = load_config(); end
rate = cfg.stream.rate_hz; n = size(Q, 2);
timed = isfield(meta, 'timed') && meta.timed;
if timed
    nr = n; T = (nr - 1)/rate; t = (0:nr-1)/rate; TH = Q; PD = Pdes;
else
    T = meta.T_s; nr = round(T*rate) + 1;
    t = (0:nr-1)/rate;
    u = t/T*n;
    TH = interp1(0:n, [Q Q(:,1)]', u, 'linear')';
    PD = interp1(0:n, [Pdes Pdes(:,1)]', u, 'linear')';
end
sg = zeros(3,1); of = zeros(3,1);
for j = 1:3
    mp = cfg.map.(sprintf('theta%d', j)); sg(j) = mp.sign; of(j) = mp.offset;
end
S = of + sg.*TH;
cpr = cfg.encoder.counts_per_rev;
C = round(S*cpr/360);
chk = check_cmds(C*360/cpr, cfg, 'servo');
if ~chk.ok
    error('export_cmds:checks', ['cmd_%s refused: limits=%d step=%d zfloor=%d finite=%d ' ...
        '(max step %.3f deg, min z %.2f mm, servo min [%s] max [%s])'], name, chk.limits, chk.step, ...
        chk.zfloor, chk.finite, chk.max_step_deg, chk.min_z_mm, num2str(chk.servo_min_deg, '%.1f '), ...
        num2str(chk.servo_max_deg, '%.1f '));
end
PN = fk_net_forward(W, TH);
cmd = [(0:nr-1); t; S; C; TH; PD; PN];
dd = fullfile(hop3_root(), 'data');
if ~isfolder(dd), mkdir(dd); end
fid = fopen(fullfile(dd, ['cmd_' name '.csv']), 'w');
fprintf(fid, 'k,t_s,s1_deg,s2_deg,s3_deg,c1,c2,c3,th1,th2,th3,x_des,y_des,z_des,x_nn,y_nn,z_nn\n');
fprintf(fid, ['%d,%.4f,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,' ...
    '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'], cmd);
fclose(fid);
src = 'unknown';
if isfield(W, 'meta') && isfield(W.meta, 'source'), src = W.meta.source; end
m = struct('name', name, 'centre', meta.centre(:)', 'R', meta.R, 'T_s', T, 'rate_hz', rate, ...
    'n_rows', nr, 'map_used', cfg.map, 'model_source', src, 'model_file', W.file, ...
    'checks', struct('limits', chk.limits, 'step', chk.step, 'zfloor', chk.zfloor, 'finite', chk.finite), ...
    'max_step_deg', chk.max_step_deg, 'min_z_mm', chk.min_z_mm, 'z_floor_mm', cfg.z_floor_mm, ...
    'placeholder', false);
if isfield(meta, 'segments')
    sgm = meta.segments;
    meta.segments = struct('name', {sgm.name}, 'k_start', num2cell([sgm.idx_start] - 1), 'k_end', num2cell([sgm.idx_end] - 1));
end
ex = setdiff(fieldnames(meta), fieldnames(m));
for i = 1:numel(ex), m.(ex{i}) = meta.(ex{i}); end
fid = fopen(fullfile(dd, ['cmd_' name '.json']), 'w');
fprintf(fid, '%s\n', jsonencode(m, 'PrettyPrint', true));
fclose(fid);
end
