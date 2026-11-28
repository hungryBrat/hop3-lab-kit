function [Q, ok] = ik_analytic(P, cfg)
% Closed-form elbow-up (theta3 < 0) IK. P 3xN mm -> Q 3xN deg; unreachable columns are NaN, ok false.
if nargin < 2, cfg = load_config(); end
d = cfg.dh; Lt = d.L3 + d.f;
x = P(1,:); y = P(2,:); z = P(3,:);
t1 = atan2d(y, x);
r = hypot(x, y); h = z - d.d1;
c3 = (r.^2 + h.^2 - d.L2^2 - Lt^2) / (2*d.L2*Lt);
ok = abs(c3) <= 1 & all(isfinite(P), 1);
t3 = -acosd(min(max(c3, -1), 1));
t2 = atan2d(h, r) - atan2d(Lt*sind(t3), d.L2 + Lt*cosd(t3));
Q = [t1; t2; t3];
Q(:, ~ok) = NaN;
end
