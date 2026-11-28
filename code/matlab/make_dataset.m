function make_dataset()
cfg = load_config(); ds = cfg.dataset; root = hop3_root();
r = cfg.region; box = [r.theta1(:)'; r.theta2(:)'; r.theta3(:)'];
zf = cfg.z_floor_mm;

rng(ds.seed_pool);
[Qp, Pp] = sample_box(box, ds.n_pool, cfg, zf);
rng(ds.seed_test);
[Qt, Pt] = sample_box(box, ds.n_test, cfg, zf);

tp = cfg.trajectories.primary;
th = linspace(0, 2*pi, tp.n_points + 1); th(end) = [];
pc = [tp.centre_xy(1) + tp.R*cos(th); tp.centre_xy(2) + tp.R*sin(th); 40*ones(size(th))];
qc = ik_up(pc, cfg);
rng(ds.seed_test + 1);
Qj = qc(:, randi(size(qc,2), 1, 3*ds.n_traj_test)) + 10*rand(3, 3*ds.n_traj_test) - 5;
Qj = min(max(Qj, box(:,1)), box(:,2));
Pj = dh_fk_chain(Qj, cfg);
keep = find(Pj(3,:) >= zf, ds.n_traj_test);
Qj = Qj(:,keep); Pj = Pj(:,keep);

dd = fullfile(root, 'data');
write_csv(fullfile(dd, 'arm_workspace_points.csv'), Qp, Pp);
write_csv(fullfile(dd, 'test_points.csv'), Qt, Pt);
write_csv(fullfile(dd, 'traj_test_points.csv'), Qj, Pj);

S = struct('n_pool_raw', ds.n_pool, 'n_pool_kept', size(Qp,2), ...
    'n_test_raw', ds.n_test, 'n_test', size(Qt,2), 'n_traj_test', size(Qj,2), ...
    'traj_test_circle', struct('centre', [tp.centre_xy(:); 40]', 'R', tp.R, 'jitter_deg', 5), ...
    'box', struct('theta1', box(1,:), 'theta2', box(2,:), 'theta3', box(3,:)), ...
    'z_floor_mm', zf, 'seed_pool', ds.seed_pool, 'seed_test', ds.seed_test, ...
    'seed_traj_test', ds.seed_test + 1);
fid = fopen(fullfile(root, 'results', 'dataset.json'), 'w');
fprintf(fid, '%s', jsonencode(S, 'PrettyPrint', true)); fclose(fid);
fprintf('pool %d/%d kept, test %d/%d kept, traj-test %d\n', size(Qp,2), ds.n_pool, ...
    size(Qt,2), ds.n_test, size(Qj,2));
end

function [Q, P] = sample_box(box, n, cfg, zf)
Q = box(:,1) + (box(:,2) - box(:,1)).*rand(3, n);
P = dh_fk_chain(Q, cfg);
k = P(3,:) >= zf;
Q = Q(:,k); P = P(:,k);
end

function write_csv(f, Q, P)
fid = fopen(f, 'w');
fprintf(fid, 'theta1_deg,theta2_deg,theta3_deg,x,y,z\n');
fprintf(fid, '%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', [Q; P]);
fclose(fid);
end

function q = ik_up(p, cfg)
d = cfg.dh; Lt = d.L3 + d.f;
r = hypot(p(1,:), p(2,:)); h = p(3,:) - d.d1;
c3 = (r.^2 + h.^2 - d.L2^2 - Lt^2)/(2*d.L2*Lt);
t3 = -acosd(c3);
t2 = atan2d(h, r) - atan2d(Lt*sind(t3), d.L2 + Lt*cosd(t3));
q = [atan2d(p(2,:), p(1,:)); t2; t3];
end
