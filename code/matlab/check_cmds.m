function chk = check_cmds(X, cfg, kind)
% MATLAB-side command safety check (CONTRACTS). X 3xN: DH theta deg (kind 'theta', default) or servo deg ('servo').
% Rows are columns of X in stream order; bad_* list offending 1-based rows (step: the row stepped into).
if nargin < 2 || isempty(cfg), cfg = load_config(); end
if nargin < 3, kind = 'theta'; end
sg = zeros(3,1); of = zeros(3,1); L = zeros(3,2);
for j = 1:3
    mp = cfg.map.(sprintf('theta%d', j)); sg(j) = mp.sign; of(j) = mp.offset;
    L(j,:) = cfg.servo_limits_deg.(sprintf('j%d', j))(:)';
end
if strcmp(kind, 'servo'), S = X; TH = sg.*(S - of); else, TH = X; S = of + sg.*TH; end
m = cfg.safety_margin_deg;
fin = all(isfinite(S), 1);
lim = all(S >= L(:,1) + m & S <= L(:,2) - m, 1);
dS = abs(diff(S, 1, 2));
stp = [true, all(dS <= cfg.max_step_deg, 1)];
P = dh_fk(TH, cfg);
zf = P(3,:) >= cfg.z_floor_mm;
chk.limits = all(lim); chk.step = all(stp); chk.zfloor = all(zf); chk.finite = all(fin);
chk.ok = chk.limits && chk.step && chk.zfloor && chk.finite;
chk.bad_limits = find(~lim); chk.bad_step = find(~stp);
chk.bad_zfloor = find(~zf); chk.bad_finite = find(~fin);
if isempty(dS), chk.max_step_deg = 0; else, chk.max_step_deg = max(dS(:)); end
chk.min_z_mm = min(P(3,:));
chk.servo_min_deg = min(S, [], 2)'; chk.servo_max_deg = max(S, [], 2)';
end
