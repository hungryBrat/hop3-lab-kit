function test_invert(which)
% Unit test: net inversion residual < 0.05 mm on every primary point, 50 Hz joint steps < 2 deg, IK round-trip < 1e-9 mm.
if nargin < 1, which = 'auto'; end
cfg = load_config(); W = load_weights(which); pr = cfg.trajectories.primary;
box = [cfg.region.theta1(:)'; cfg.region.theta2(:)'; cfg.region.theta3(:)'];
rng(11); Qr = box(:,1) + rand(3, 2000).*diff(box, 1, 2);
rt = max(vecnorm(dh_fk(ik_analytic(dh_fk(Qr, cfg), cfg), cfg) - dh_fk(Qr, cfg)));
fw = cfg.dh.L2*cosd(Qr(2,:)) + (cfg.dh.L3 + cfg.dh.f)*cosd(Qr(2,:) + Qr(3,:)) > 1;
dq = max(max(abs(ik_analytic(dh_fk(Qr(:,fw), cfg), cfg) - Qr(:,fw))));
assert(rt < 1e-9, 'IK round-trip %.3g mm', rt);
assert(dq < 1e-6, 'IK joint round-trip %.3g deg (forward-reach poses)', dq);
for z = pr.z_family(:)'
    P = make_circle([pr.centre_xy(:); z], pr.R, pr.n_points);
    [Q, info] = invert_fk(W, P, struct('cfg', cfg));
    assert(all(info.residual_mm < 0.05), 'z=%d residual max %.4f', z, max(info.residual_mm));
    assert(all(Q(3,:) < 0), 'z=%d left elbow-up branch', z);
    T = pr.T_s; n = pr.n_points; u = (0:round(T*cfg.stream.rate_hz))/cfg.stream.rate_hz/T*n;
    TH = interp1(0:n, [Q Q(:,1)]', u, 'linear')';
    st = max(max(abs(diff(TH, 1, 2))));
    assert(st < 2, 'z=%d 50 Hz step %.3f deg', z, st);
    fprintf('z=%3d  residual max %.4f mm  row step max %.3f deg  iters max %d\n', z, max(info.residual_mm), st, max(info.iterations));
end
c = check_cmds([0 0; 90 91; 0 0], cfg);
assert(c.ok && c.max_step_deg == 1, 'check_cmds nominal');
c = check_cmds([0 3; 90 90; -30 -30], cfg);
assert(~c.step && isequal(c.bad_step, 2), 'check_cmds step detection');
c = check_cmds([0; 140; -30], cfg);
assert(~c.limits, 'check_cmds limit detection');
c = check_cmds([0; 20; -120], cfg);
assert(~c.zfloor, 'check_cmds z floor detection');
fprintf('test_invert PASS (IK round-trip %.2e mm, %s weights)\n', rt, W.meta.source);
end
