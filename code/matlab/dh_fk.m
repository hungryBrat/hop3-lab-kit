function P = dh_fk(Q, cfg)
% Q: 3xN DH joint angles [deg]; P: 3xN tip position [mm] (closed form of 0T4).
if nargin < 2, cfg = load_config(); end
d = cfg.dh; Lt = d.L3 + d.f;
c1 = cosd(Q(1,:)); s1 = sind(Q(1,:));
r = d.L2*cosd(Q(2,:)) + Lt*cosd(Q(2,:)+Q(3,:));
P = [c1.*r; s1.*r; d.d1 + d.L2*sind(Q(2,:)) + Lt*sind(Q(2,:)+Q(3,:))];
end
