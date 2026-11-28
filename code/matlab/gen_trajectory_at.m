function r = gen_trajectory_at(zc, which)
% In-session: primary circle at centre z = zc -> data/cmd_session_zc.* and results/traj_session_zc.json (~1 s, no figures).
if nargin < 2, which = 'auto'; end
r = gen_trajectories(zc, which);
end
