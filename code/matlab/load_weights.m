function W = load_weights(which)
% which: 'auto' (default: real if present else STANDIN), 'real', 'standin', or a file path.
if nargin < 1, which = 'auto'; end
md = fullfile(hop3_root(), 'models');
real = fullfile(md, 'fk_weights.json'); stand = fullfile(md, 'fk_weights_standin.json');
switch which
    case 'auto'
        if isfile(real), f = real; else, f = stand;
            warning('hop3:standin', 'Using STANDIN weights (no fitnet model yet).'); end
    case 'real', f = real;
    case 'standin', f = stand;
    otherwise, f = which;
end
W = jsondecode(fileread(f));
W.file = f;
if ~iscell(W.layers), W.layers = num2cell(W.layers); end
for k = 1:numel(W.layers)
    W.layers{k}.W = reshape_mat(W.layers{k}.W);
    W.layers{k}.b = W.layers{k}.b(:);
end
W.in_map.xoffset = W.in_map.xoffset(:); W.in_map.gain = W.in_map.gain(:);
W.out_map.xoffset = W.out_map.xoffset(:); W.out_map.gain = W.out_map.gain(:);
end
function M = reshape_mat(M)
if iscell(M), M = cell2mat(cellfun(@(r) r(:)', M, 'UniformOutput', false)); end
end
