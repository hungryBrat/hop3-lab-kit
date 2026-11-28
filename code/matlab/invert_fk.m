function [Q, info] = invert_fk(W, Pdes, opts)
% Damped-least-squares inversion of the FK net (never analytic IK) along a path Pdes (3xN mm).
% opts: cfg, lambda0 (0.5), max_iter (50), tol_mm (0.01), step_max_deg (5), grid_n (25),
%       q0 (seed for point 1; default nearest net prediction on a grid of the region box),
%       leave_box (false; true clamps to servo limits minus safety margin instead of the box).
if nargin < 3, opts = struct(); end
cfg = getopt(opts, 'cfg', []); if isempty(cfg), cfg = load_config(); end
lam0 = getopt(opts, 'lambda0', 0.5);
maxit = getopt(opts, 'max_iter', 50);
tol = getopt(opts, 'tol_mm', 0.01);
smax = getopt(opts, 'step_max_deg', 5);
ng = getopt(opts, 'grid_n', 25);
leave = getopt(opts, 'leave_box', false);
box = [cfg.region.theta1(:)'; cfg.region.theta2(:)'; cfg.region.theta3(:)'];
if leave, lim = servo_theta_limits(cfg); else, lim = box; end
lo = lim(:,1); hi = lim(:,2);
N = size(Pdes, 2);
Q = nan(3, N); res = nan(1, N); iters = zeros(1, N);
q = getopt(opts, 'q0', []);
if isempty(q), q = grid_seed(W, Pdes(:,1), box, ng); end
q = min(max(q(:), lo), hi);
I = eye(3);
for k = 1:N
    p = Pdes(:,k); lam = lam0;
    e = p - fk_net_forward(W, q); ne = norm(e); it = 0;
    while ne >= tol && it < maxit
        it = it + 1;
        J = net_jacobian(W, q);
        dq = J'*((J*J' + lam^2*I)\e);
        m = max(abs(dq)); if m > smax, dq = dq*(smax/m); end
        qn = min(max(q + dq, lo), hi);
        en = p - fk_net_forward(W, qn); nn = norm(en);
        if nn < ne
            q = qn; e = en; ne = nn; lam = max(lam/10, 1e-7);
        else
            lam = min(lam*10, 1e7);
        end
    end
    Q(:,k) = q; res(k) = ne; iters(k) = it;
end
info.residual_mm = res;
info.iterations = iters;
info.converged = res < tol;
info.hit_bound = any(abs(Q - lo) < 1e-9 | abs(Q - hi) < 1e-9, 1);
info.lo = lo; info.hi = hi; info.leave_box = leave;
end

function q = grid_seed(W, p, box, ng)
[a, b, c] = ndgrid(linspace(box(1,1), box(1,2), ng), linspace(box(2,1), box(2,2), ng), ...
    linspace(box(3,1), box(3,2), ng));
G = [a(:)'; b(:)'; c(:)'];
[~, i] = min(vecnorm(fk_net_forward(W, G) - p));
q = G(:,i);
end

function lim = servo_theta_limits(cfg)
m = cfg.safety_margin_deg; lim = zeros(3, 2);
for j = 1:3
    s = cfg.servo_limits_deg.(sprintf('j%d', j))(:)' + [m, -m];
    mp = cfg.map.(sprintf('theta%d', j));
    lim(j,:) = sort(mp.sign*(s - mp.offset));
end
end

function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
