function test_dh()
cfg = load_config();
rng(0);
Q = [360*rand(1,1e4)-180; 360*rand(1,1e4)-180; 360*rand(1,1e4)-180];
err = max(abs(dh_fk_chain(Q, cfg) - dh_fk(Q, cfg)), [], 'all');
fprintf('chain vs closed form, 1e4 poses: max |diff| = %.3e mm\n', err);
assert(err < 1e-9, 'chain/closed-form mismatch');
pn = dh_fk_chain([0; 90; 0], cfg);
fprintf('neutral (0,90,0) -> (%.4f, %.4f, %.4f)\n', pn);
assert(max(abs(pn - [0; 0; 332.9])) < 1e-9, 'neutral pose mismatch');
fprintf('test_dh PASS\n');
end
