function test_jacobian()
W = load_weights('standin');
cfg = load_config(); b = box_of(cfg);
rng(0);
Q = b(:,1) + (b(:,2) - b(:,1)).*rand(3, 100);
h = 1e-4; worst = 0;
for k = 1:100
    q = Q(:,k); J = net_jacobian(W, q); Jn = zeros(3);
    for j = 1:3
        e = zeros(3,1); e(j) = h;
        Jn(:,j) = (fk_net_forward(W, q+e) - fk_net_forward(W, q-e))/(2*h);
    end
    worst = max(worst, norm(J - Jn)/norm(Jn));
end
fprintf('net_jacobian vs central differences, 100 pts: max rel err = %.3e\n', worst);
assert(worst < 1e-6, 'Jacobian mismatch');
fprintf('test_jacobian PASS\n');
end

function b = box_of(cfg)
r = cfg.region; b = [r.theta1(:)'; r.theta2(:)'; r.theta3(:)'];
end
