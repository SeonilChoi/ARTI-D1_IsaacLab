#!/bin/bash
# usage: scripts/porting/run_tests.sh <test names...>   (logs in logs/porting/test_<name>.log, summary in logs/porting/tests.txt)
[ -n "$VIRTUAL_ENV" ] || source /opt/isaac/IsaacLab-main/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
cd "$(dirname "$0")/../.."
S=${OUT:-logs/porting/out}; mkdir -p $S
L=logs/porting; T=Template-Arti-D1-Amp-Direct-v0
ck() { grep -h "Saved final agent checkpoint to" $L/test_train_amp_$1.log 2>/dev/null | tail -1 | sed 's/.*to: //'; }
pt() { ls $(ck $1)/*.pt 2>/dev/null | sort -V | tail -1; }
run() { local name=$1; shift; echo "== $name: $*" > $L/test_$name.log; timeout ${TO:-1200} "$@" >> $L/test_$name.log 2>&1; local rc=$?
  local tb=$(grep -c "Traceback" $L/test_$name.log); local jo=$(grep -o "JOINT-ORDER\] \(OK\|FAILED\)" $L/test_$name.log | sort -u | tr '\n' ' ')
  echo "$(date +%T) $name exit=$rc tracebacks=$tb $jo" | tee -a $L/tests.txt; }
for t in "$@"; do case $t in
  train_amp_physx)  run $t python scripts/skrl/train.py --task $T physics=isaacsim_physx --num_envs 64 --max_iterations 30 agent.agent.experiment.write_interval=48 ;;
  train_amp_newton) run $t python scripts/skrl/train.py --task $T physics=newton_mjwarp --num_envs 64 --max_iterations 30 agent.agent.experiment.write_interval=48 ;;
  train_ppo_physx)  run $t isaaclab train --rl_library skrl --task Template-Arti-D1-Direct-v0 physics=isaacsim_physx --num_envs 32 --max_iterations 40 ;;
  train_ppo_newton) run $t isaaclab train --rl_library skrl --task Template-Arti-D1-Direct-v0 physics=newton_mjwarp --num_envs 32 --max_iterations 40 ;;
  fall_physx)  run $t python scripts/skrl/train.py --task $T physics=isaacsim_physx --num_envs 64 --max_iterations 15 agent.agent.experiment.write_interval=48 env.balance_support_kp=0.0 env.balance_support_kd=0.0 ;;
  fall_newton) run $t python scripts/skrl/train.py --task $T physics=newton_mjwarp --num_envs 64 --max_iterations 15 agent.agent.experiment.write_interval=48 env.balance_support_kp=0.0 env.balance_support_kd=0.0 ;;
  train_kamino)     run $t isaaclab train --rl_library skrl --task $T physics=newton_kamino --num_envs 16 --max_iterations 3 ;;
  train_ovphysx)    run $t isaaclab train --rl_library skrl --task $T physics=ovphysx --num_envs 16 --max_iterations 3 ;;
  amp_newton)       run $t python scripts/porting/check_amp_consistency.py physics=newton_mjwarp ;;
  play_physx)       run $t python scripts/skrl/play.py --task $T --checkpoint $(pt physx) --num_envs 4 physics=isaacsim_physx --video --video_length 900 ;;
  play_newton)      run $t python scripts/skrl/play.py --task $T --checkpoint $(pt newton) --num_envs 4 physics=newton_mjwarp --viz newton_gl --video viz --video_length 900 ;;
  onnx)             run $t bash -c "python scripts/pt_to_onnx.py --checkpoint $(pt physx) --output $S/agent.onnx && python -u scripts/tuning/verify_onnx.py $(pt physx) $S/agent.onnx" ;;
  check_onnx)       run $t python -u scripts/tuning/check_onnx_in_env.py --onnx $S/agent.onnx --seconds 6 --save_log $S/onnx_log.csv ;;
  eval_ref)         run $t python -u scripts/tuning/eval_gait.py --mode reference --num_envs 16 --seconds 4 --warmup 1 ;;
  eval_seq)         run $t python -u scripts/tuning/eval_gait.py --mode sequence --checkpoint $(pt physx) --num_envs 16 --uniform_action_scale 0.0 ;;
  ctrl_obs)         run $t python -u scripts/tuning/check_controller_obs.py --steps 60 ;;
  ilc)              run $t python -u scripts/tuning/ilc_nominal.py --iters 1 --num_envs 16 --out $S/ilc_table.npy ;;
  replay)           run $t python scripts/replay_demo.py --index 0 --no_realtime --output $S/replay0.yaml ;;
esac; done
