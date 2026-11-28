function J = net_jacobian(W, q)
% dP/dq of the exported net at one point q (3x1 deg); J is 3x3 in mm/deg.
a = (q(:) - W.in_map.xoffset).*W.in_map.gain + W.in_map.ymin;
J = diag(W.in_map.gain);
for k = 1:numel(W.layers)
    L = W.layers{k};
    z = L.W*a + L.b;
    if strcmp(L.act, 'tansig')
        a = tanh(z); J = diag(1 - a.^2)*L.W*J;
    else
        a = z; J = L.W*J;
    end
end
J = diag(1./W.out_map.gain)*J;
end
