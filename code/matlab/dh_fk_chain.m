function [P, T] = dh_fk_chain(Q, cfg)
% Q: 3xN DH joint angles [deg]; P: 3xN tip position [mm] from 0T1*1T2*2T3*3T4; T: 4x4xN.
if nargin < 2, cfg = load_config(); end
d = cfg.dh; al = d.alpha_deg;
n = size(Q, 2);
P = zeros(3, n);
if nargout > 1, T = zeros(4, 4, n); end
for k = 1:n
    A = dh(Q(1,k), d.d1, 0, al(1)) * dh(Q(2,k), 0, d.L2, al(2)) * ...
        dh(Q(3,k), 0, d.L3, al(3)) * dh(0, 0, d.f, al(4));
    P(:,k) = A(1:3,4);
    if nargout > 1, T(:,:,k) = A; end
end
end

function A = dh(th, dd, a, al)
ct = cosd(th); st = sind(th); ca = cosd(al); sa = sind(al);
A = [ct, -ca*st,  sa*st, a*ct;
     st,  ca*ct, -sa*ct, a*st;
     0,   sa,     ca,    dd;
     0,   0,      0,     1];
end
