function P = fk_net_forward(W, Q)
% Forward pass of the exported fitnet: Q 3xN deg -> P 3xN mm.
a = (Q - W.in_map.xoffset).*W.in_map.gain + W.in_map.ymin;
for k = 1:numel(W.layers)
    L = W.layers{k};
    a = L.W*a + L.b;
    if strcmp(L.act, 'tansig'), a = tanh(a); end
end
P = (a - W.out_map.ymin)./W.out_map.gain + W.out_map.xoffset;
end
