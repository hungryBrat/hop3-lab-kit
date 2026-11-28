function B = make_letterB(spec, rate_hz)
% Timed closed letter 'B' in the front plane x = spec.x_plane_mm, sampled at rate_hz.
% spec: x_plane_mm, centre_yz, H, v_max_mm_s, corner_dwell_s, hold_s (default 1).
% B: P 3xN mm, t 1xN s, segments(i) {name, idx_start, idx_end} (1-based), corners (1-based), seg_T_s, T_s.
if nargin < 2, rate_hz = 50; end
H = spec.H; yc = spec.centre_yz(1); zc = spec.centre_yz(2); x0 = spec.x_plane_mm;
v = spec.v_max_mm_s; dw = spec.corner_dwell_s;
hold_s = 1; if isfield(spec, 'hold_s'), hold_s = spec.hold_s; end
y0 = yc - H/8; R = H/4;
bot = [y0; zc - H/2]; top = [y0; zc + H/2];
segs = {'spine', @(s) bot + (top - bot)*s, H; ...
        'top', @(s) [y0 + R*cos(pi/2 - pi*s); zc + H/4 + R*sin(pi/2 - pi*s)], pi*R; ...
        'bottom', @(s) [y0 + R*cos(pi/2 - pi*s); zc - H/4 + R*sin(pi/2 - pi*s)], pi*R};
YZ = bot; seg = struct('name', {}, 'idx_start', {}, 'idx_end', {});
corners = []; segT = zeros(1, 3);
YZ = [YZ, repmat(bot, 1, round((hold_s + dw)*rate_hz))];
corners(end+1) = size(YZ, 2);
for i = 1:3
    n = ceil(1.875*segs{i,3}/v*rate_hz); segT(i) = n/rate_hz;
    tau = (1:n)/n;
    s = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;
    k0 = size(YZ, 2);
    YZ = [YZ, segs{i,2}(s)]; %#ok<AGROW>
    seg(i) = struct('name', segs{i,1}, 'idx_start', k0, 'idx_end', size(YZ, 2));
    corners(end+1) = size(YZ, 2); %#ok<AGROW>
    nd = round(dw*rate_hz); if i == 3, nd = nd + round(hold_s*rate_hz); end
    YZ = [YZ, repmat(YZ(:,end), 1, nd)]; %#ok<AGROW>
end
N = size(YZ, 2);
B.P = [repmat(x0, 1, N); YZ];
B.t = (0:N-1)/rate_hz;
B.segments = seg;
B.corners = corners;
B.seg_T_s = segT;
B.T_s = B.t(end);
B.rate_hz = rate_hz;
B.centre = [x0, yc, zc];
B.H = H; B.R = R;
end
