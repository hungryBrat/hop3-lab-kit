function test_letterB(which)
% Letter B: residual < 0.05 mm, 50 Hz step < 2 deg, zero velocity at corners, closed, DH error within net
% test error + inversion tolerance, check_cmds passes under the current map and under both sigma1 and sigma3.
if nargin < 1, which = 'auto'; end
cfg = load_config(); W = load_weights(which); rate = cfg.stream.rate_hz;
sp = cfg.trajectories.letterB;
B = make_letterB(sp, rate); P = B.P; N = size(P, 2);
assert(max(abs(diff(B.t) - 1/rate)) < 1e-12, 'not sampled at %d Hz', rate);
assert(norm(P(:,1) - P(:,end)) < 1e-9, 'path not closed');
assert(all(abs(P(1,:) - sp.x_plane_mm) < 1e-12), 'left the front plane');
H = sp.H; y0 = sp.centre_yz(1) - H/8; zc = sp.centre_yz(2);
sg = B.segments; sp_k = sg(1).idx_start:sg(1).idx_end;
assert(max(abs(P(2,sp_k) - y0)) < 1e-9, 'spine not straight');
tk = sg(2).idx_start:sg(2).idx_end; bk = sg(3).idx_start:sg(3).idx_end;
assert(max(abs(vecnorm(P(2:3,tk) - [y0; zc + H/4]) - H/4)) < 1e-9, 'top bump radius');
assert(max(abs(vecnorm(P(2:3,bk) - [y0; zc - H/4]) - H/4)) < 1e-9, 'bottom bump radius');
assert(all(P(2,[tk bk]) >= y0 - 1e-9), 'bumps must bulge toward +y');
V = vecnorm(diff(P, 1, 2))*rate;
assert(max(V) <= sp.v_max_mm_s + 1e-9, 'peak speed %.3f mm/s', max(V));
cv = max(V([B.corners - 1, B.corners]));
assert(cv < 0.01, 'corner speed %.3g mm/s', cv);
[Q, info] = invert_fk(W, P, struct('cfg', cfg));
assert(all(info.residual_mm < 0.05), 'residual max %.4f', max(info.residual_mm));
assert(all(Q(3,:) < 0), 'left elbow-up branch');
st = max(max(abs(diff(Q, 1, 2))));
assert(st < 2, '50 Hz step %.3f deg', st);
qv = max(max(abs(Q(:, B.corners + 1) - Q(:, B.corners - 1))))*rate/2;
assert(qv < 0.05, 'corner joint rate %.3g deg/s', qv);
fe = jsondecode(fileread(fullfile(hop3_root(), 'results', 'fk_eval.json')));
e = vecnorm(dh_fk(Q, cfg) - P);
lim = fe.eucl_max_mm + max(info.residual_mm);
assert(max(e) <= lim, 'DH err max %.4f > net max %.4f + residual', max(e), fe.eucl_max_mm);
chk = check_cmds(Q, cfg);
assert(chk.ok, 'check_cmds failed under current map');
for s1 = [1 -1]
    for s3 = [-1 1]
        cf = cfg; cf.map.theta1.sign = s1; cf.map.theta3.sign = s3;
        k = check_cmds(Q, cf);
        assert(k.limits && k.finite, 'limits fail at sigma1=%d sigma3=%d', s1, s3);
        fprintf('  sigma1=%+d sigma3=%+d servo [%s] .. [%s]\n', s1, s3, num2str(k.servo_min_deg, '%.1f '), num2str(k.servo_max_deg, '%.1f '));
    end
end
f = fullfile(hop3_root(), 'data', 'cmd_letterB.csv');
if isfile(f)
    M = readmatrix(f);
    assert(size(M, 1) == N, 'cmd_letterB.csv rows %d vs %d', size(M, 1), N);
    kc = check_cmds(M(:, 6:8)'*360/cfg.encoder.counts_per_rev, cfg, 'servo');
    assert(kc.ok, 'cmd_letterB.csv fails check_cmds');
    m = jsondecode(fileread(fullfile(hop3_root(), 'data', 'cmd_letterB.json')));
    assert(isequal([m.segments.k_start], [sg.idx_start] - 1) && isequal([m.segments.k_end], [sg.idx_end] - 1), 'meta segments');
end
fprintf(['test_letterB PASS: N %d (%.2f s), residual max %.4f mm, step max %.4f deg, corner speed %.1e mm/s, ' ...
    'DH err rms %.4f max %.4f mm (limit %.4f), %s weights\n'], N, B.T_s, max(info.residual_mm), st, cv, rms(e), max(e), lim, W.meta.source);
end
