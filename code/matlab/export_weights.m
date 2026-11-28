function W = export_weights(net, outfile)
root = hop3_root(); md = fullfile(root, 'models');
if nargin < 2, outfile = fullfile(md, 'fk_weights.json'); end
[od, ~] = fileparts(outfile);

ip = net.inputs{1}; op = net.outputs{net.numLayers};
mi = mapminmax_settings(ip); mo = mapminmax_settings(op);

L = net.numLayers; layers = cell(1, L);
for k = 1:L
    if k == 1, Wk = net.IW{1,1}; else, Wk = net.LW{k,k-1}; end
    tf = net.layers{k}.transferFcn;
    assert(any(strcmp(tf, {'tansig', 'purelin'})), 'export_weights: unsupported transferFcn %s', tf);
    layers{k} = struct('W', Wk, 'b', net.b{k}, 'act', tf);
end
arch = cellfun(@(l) l.size, net.layers(1:L-1))';
W = struct('meta', struct('source', 'fitnet', 'arch', arch, 'created', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')), ...
                         'train_fcn', net.trainFcn, 'matlab', version), ...
    'in_map', struct('xoffset', mi.xoffset, 'gain', mi.gain, 'ymin', mi.ymin), ...
    'out_map', struct('xoffset', mo.xoffset, 'gain', mo.gain, 'ymin', mo.ymin), ...
    'layers', {layers});
fid = fopen(outfile, 'w'); fprintf(fid, '%s', jsonencode(W)); fclose(fid);
save(fullfile(od, 'fk_weights.mat'), 'W');
genFunction(net, fullfile(od, 'fkNet'), 'MatrixOnly', 'yes');

cfg = load_config(); r = cfg.region; box = [r.theta1(:)'; r.theta2(:)'; r.theta3(:)'];
rng(123); X = box(:,1) + (box(:,2) - box(:,1)).*rand(3, 2000);
Y = net(X);
Wj = load_weights(outfile);
addpath(od); rehash path;
e1 = max(abs(fk_net_forward(Wj, X) - Y), [], 'all');
e2 = max(abs(feval('fkNet', X) - Y), [], 'all');
fprintf('export check on 2000 pts: |fk_net_forward(json) - net| = %.3e, |fkNet - net| = %.3e\n', e1, e2);
assert(e1 < 1e-9 && e2 < 1e-9, 'export_weights: exported model does not reproduce net(X)');
end

function s = mapminmax_settings(p)
% R2026b fitnet uses {'mapminmax'} only; older releases prepend removeconstantrows.
for i = 1:numel(p.processFcns)
    switch p.processFcns{i}
        case 'mapminmax', s = p.processSettings{i};
        case 'removeconstantrows'
            assert(isempty(p.processSettings{i}.remove), 'export_weights: removeconstantrows removed rows');
        otherwise, error('export_weights: unsupported processFcn %s', p.processFcns{i});
    end
end
end
