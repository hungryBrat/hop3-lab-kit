# Builder A: data + FK network

## Run now (no toolbox)
```
matlab -batch "cd('code/matlab'); make_dataset; plot_region; test_dh"
.venv/bin/python code/python/standin_train.py            # -> models/fk_weights_standin.json
matlab -batch "cd('code/matlab'); test_jacobian; eval_fk(load_weights('standin')); sweep_fk('standin',5,300)"
```

## After installing Deep Learning Toolbox (in this order)
```
matlab -batch "cd('code/matlab'); exist('fitnet')"          % must print 2
matlab -batch "cd('code/matlab'); train_fk"                 % [20 10] trainlm, 5 seeds -> models/Forward_Kinematics_Model.mat,
                                                            %   fk_weights.json/.mat, fkNet.m, results/fk_seeds.json
matlab -batch "cd('code/matlab'); eval_fk(load_weights('real'))"   % overwrites results/fk_eval.json + figures/fk_* (no STANDIN label)
matlab -batch "cd('code/matlab'); sweep_fk"                 % fitnet sweep, overwrites results/sweep.json + sweep_* figures
```
Optional: `train_fk([40 20])`, `train_fk([20 10],'trainbr')` (trainbr uses dividetrain). Each call overwrites the model files and `fk_seeds.json`; rerun the one you choose last.
Fresh-session check: `load('models/Forward_Kinematics_Model.mat','net','tr'); net([0;45;-60])` and `addpath models; fkNet([0;45;-60])`.

## Verified
- `dh_fk_chain` vs closed form: 1.4e-13 mm max on 1e4 poses; neutral (0,90,0) -> (0,0,398.6).
- `net_jacobian` vs central differences on stand-in weights: 1.4e-8 relative.
- `make_dataset`, `plot_region`, `eval_fk`, `sweep_fk('standin')`, `standin_train.py` run clean.

## Not verified (toolbox missing)
- `train_fk.m`, `export_weights.m`, `sweep_fk('fitnet')` have never run. They were written against the documented API:
  `net.IW{1,1}`, `net.LW{k,k-1}`, `net.b{k}`, `net.inputs{1}.processSettings{2}` and `net.outputs{end}.processSettings{2}`
  (mapminmax `xoffset/gain/ymin`), and `removeconstantrows` `.remove` empty. `export_weights` asserts on its own that
  JSON->`fk_net_forward` and `fkNet` both match `net(X)` to < 1e-9, so a schema mistake fails loudly.
- The stand-in is a numpy trainlm clone (same mu schedule, 85/15 split, max_fail 10, mm-space MSE). Its numbers only stand in for fitnet's.
